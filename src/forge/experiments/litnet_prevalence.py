"""e12, RQ1 and E2 on LITNET-2020, the false omission rate and the dismissed share of each method in the last part of
every capture, its attack family thinned to a range of prevalences.

LITNET-2020 as published is eleven captures of about six minutes on eleven days, each with the attacks of one family,
two in the icmp flood capture. each capture is cut in time order at the ranks of its attacks into training, selection,
calibration and test parts, so every part holds its share of the capture's attacks whatever their timing, and a capture
whose families follow each other without overlap is first cut at the first attack of each later family. every family
is then known to the scorer, and the test part of a capture, or of a piece of one, is a window whose attacks come from
its own family, a vertex of the simplex of family mixes, which makes this E2 as well. the scanning and udp flood
captures hold benign flows the authors filtered, so their benign flows are left out everywhere and their attacks used.

a LightGBM scorer is trained on the training parts and stopped early on the selection parts. the grid holds benign
quantiles of the selection parts and a few deeper levels. the calibration parts keep all their benign flows and have
their attacks thinned to the calibration prevalence, every family capped at one common count so that only the largest
families lose attacks, and each draw is a calibration set of that size drawn with replacement from them, as in e11. the
fixed threshold and the beta of conformal FNR read the selection parts at the same mix.

every window comes in three arms.
    reference   the benign count, prevalence and mix of the window with the score distributions of the calibration parts
                for both classes, so label shift holds by construction. RQ1 is read here
    s2          the window's own benign flows with the calibration distributions of the attack families, so only the
                attacks keep their calibration distributions, as Theorem 1b assumes
    real        the window's own benign flows and attacks
the captures with filtered benign flows have reference windows only, with the mean benign count of the other windows. a
window keeps all its benign flows and has its attacks thinned to each prevalence. above its capture's own prevalence a
real window keeps all its test attacks and has its benign flows subsampled instead. the true rate of a threshold in a
window follows from the score distributions of its population, and each draw also takes one window of that size, on
which the online certificate runs and the realized share of attacks among the dismissed flows is counted. each real
window is also evaluated at its own prevalence, apart.

what each method claims.
    forge_simplex   Theorem 1, every prevalence up to pi_max and every mix of the families, valid while both classes keep
                    their calibration score distributions
    forge_pooled    the same for the calibration mix of the families only
    forge_online    Theorem 1b, the same for one window from its unlabeled scores, valid while the attack families keep
                    their calibration score distributions
    ltt_iid         the FOR at the prevalence and the family mix of the calibration set
    conformal_fnr   a miss rate of at most beta on average over the calibration draw, with beta the miss rate of the
                    fixed threshold in the selection parts
    fixed_sel       nothing, the loosest threshold whose FOR at the calibration prevalence and mix, estimated on the
                    selection parts, stays at or below epsilon

a family whose expected calibration attacks could not certify, with no miss, a threshold dismissing family_floor of the
benign flows is pooled with the other such families, and the pool joins the largest family while it is still too
small. the simplex claim then covers a mix only where the families of each pool keep their calibration proportions, so
a window of one pooled family lies outside it. such windows are marked, and the pooled tables keep them apart.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from forge import runs
from forge.certify.certificate import (certify, conformal_fnr, family_groups, fixed_sequence, for_from_rates,
                                       grid_from, group_rows, ltt_iid, min_family_size, online_certifier)
from forge.data import litnet2020
from forge.data.synthetic import draw_counts
from forge.eval import windows as judge
from forge.eval.metrics import auc
from forge.models import boosting
from forge.models.encoding import Encoder
from forge.viz import layout, prevalence, style
from forge.viz.export import save_figure
from forge.viz.prevalence import BEST_LINES

METHODS = ["forge_simplex", "forge_pooled", "forge_online", "ltt_iid", "conformal_fnr", "fixed_sel"]
LABELS = {
    "forge_simplex": "FORGE offline, Theorem 1",
    "forge_pooled": "FORGE offline, calibration mix",
    "forge_online": "FORGE online, Theorem 1b",
    "ltt_iid": "LTT, i.i.d.",
    "conformal_fnr": "conformal FNR, Şahin and Mert",
    "fixed_sel": "fixed threshold, selection parts",
}
# the markers of e10 and e11 for the same methods
MARKERS = {"forge_simplex": "o", "forge_pooled": "s", "forge_online": "v", "ltt_iid": "^", "conformal_fnr": "p",
           "fixed_sel": "X"}
PARTS = ["train", "selection", "calibration", "test"]
# the mix of attack families in a window, the families of its capture or the mix of the calibration set
MIXES = ["window", "calibration"]
MIX_LABELS = {"window": "family of the capture", "calibration": "calibration mix of families"}
# a reference window takes the score distributions of the calibration parts for both classes, an s2 window its own
# benign flows and the calibration attacks, a real window its own flows
ARMS = ["reference", "s2", "real"]
ARM_LABELS = {"reference": "reference windows", "s2": "S2 windows", "real": "real windows"}
WINDOW_WORDS = {"reference": "reference window", "s2": "S2 window", "real": "real window"}
COLUMNS = ["te", "row_id", "label", "family", "capture"]
# derived features and the column of the processed table each comes from
DERIVED = {"sp_range": ("sp", litnet2020.port_range)}
# rows encoded and scored at a time
CHUNK = 2_000_000
# a third yardstick, the loosest threshold safe at pi_max for every mix the claim covers, in a dash the other lines
# leave free
EVERY_LINE = ("best_every_mix", (0, (4, 2)), r"best threshold for every mix up to $\pi_{\max}$")


def raw_columns(features: list[str]) -> list[str]:
    return list(dict.fromkeys(DERIVED[f][0] if f in DERIVED else f for f in features))


def load_flows(cfg: dict, features: list[str]) -> pd.DataFrame:
    flows = litnet2020.load(cfg, columns=COLUMNS + [c for c in raw_columns(features) if c not in COLUMNS])
    for name, (source, derive) in DERIVED.items():
        if name in features:
            flows[name] = derive(flows[source].to_numpy())
    return flows


def balanced_shares(n: np.ndarray, budget: float) -> np.ndarray:
    # the mix that keeps min(n_k, cap) attacks of every family, the common cap chosen so that they add up to the budget,
    # so the smaller families keep all their attacks and only the largest are thinned
    n = np.asarray(n, dtype=float)
    order = np.sort(n)
    kept = 0.0
    for k, size in enumerate(order):
        cap = (budget - kept) / (len(order) - k)
        if cap <= size:
            break
        kept += size
    return np.minimum(n, cap) / budget


def calibration_shares(n: np.ndarray, budget: float, mix: str) -> np.ndarray:
    # share of each family among the attacks of a calibration set at the calibration prevalence
    n = np.asarray(n, dtype=float)
    if n.sum() < budget:
        raise ValueError(f"the calibration parts hold {n.sum():.0f} attacks, fewer than the {budget:.0f} that the "
                         "calibration prevalence takes")
    if mix == "natural":
        return n / n.sum()
    if mix == "balanced":
        return balanced_shares(n, budget)
    raise ValueError(f"calibration mix is {mix}, expected natural or balanced")


def in_claim(shares: np.ndarray, calibration: np.ndarray, groups: np.ndarray) -> bool:
    # the simplex claim over the groups covers a mix whose families keep, inside every merged group, the proportions
    # of the calibration set, with any weights across the groups
    for g in np.unique(groups):
        s, c = shares[groups == g], calibration[groups == g]
        if s.sum() > 0 and (c.sum() == 0 or not np.allclose(s / s.sum(), c / c.sum())):
            return False
    return True


def score_rows(booster, enc: Encoder, flows: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    # chunks keep one encoded block of rows in memory at a time
    out = np.empty(len(rows), dtype=np.float32)
    for lo in range(0, len(rows), CHUNK):
        out[lo:lo + CHUNK] = boosting.score(booster, enc.transform(flows.iloc[rows[lo:lo + CHUNK]]))
    return out


def split_flows(flows: pd.DataFrame, shares_split, filtered: set) -> dict:
    """the segment and the part of every flow of the processed table, which flows are kept, the attack families present
    and the class code of every flow, 0 for a benign flow and 1 and up for the families in their order.

    the benign flows of the filtered captures are left out everywhere and their attacks stay.
    """
    label = flows["label"].to_numpy()
    segment, part = litnet2020.capture_parts(np.asarray(flows["capture"]), label, np.asarray(flows["family"]),
                                             flows["te"].to_numpy(), shares_split)
    capture_codes = flows["capture"].cat.codes.to_numpy()
    left_out = [i for i, name in enumerate(flows["capture"].cat.categories) if name in filtered]
    kept = (label == 1) | ~np.isin(capture_codes, left_out)
    present = set(flows.loc[label == 1, "family"].astype(str))
    families = [f for f in litnet2020.ATTACK_FAMILIES if f in present]
    remap = np.zeros(len(litnet2020.FAMILY_ORDER), dtype=np.int8)
    for k, f in enumerate(families):
        remap[litnet2020.FAMILY_ORDER.index(f)] = k + 1
    return {"label": label, "segment": segment, "part": part, "kept": kept, "families": families,
            "code": remap[flows["family"].cat.codes.to_numpy()]}


def fit_scorer(flows: pd.DataFrame, label: np.ndarray, train: np.ndarray, sel: np.ndarray, sc: dict, seed: int):
    # the scorer sees the rows of train, and the rows of sel only to stop. returns the booster, the encoder fitted on
    # the training rows and the loss on the selection rows at the best round
    features = sc["features"]
    categorical = [f for f in features if f in sc["categorical"]]
    enc = Encoder([f for f in features if f not in categorical], categorical).fit(flows.iloc[train])
    booster = boosting.fit(enc.transform(flows.iloc[train]), label[train], enc.transform(flows.iloc[sel]), label[sel],
                           sc["params"], sc["rounds"], sc["early_stopping"], seed, enc.columns, enc.categorical_index)
    return booster, enc, float(next(iter(booster.best_score["selection"].values())))


def segment_table(flows: pd.DataFrame, segment: np.ndarray, part: np.ndarray, filtered: set) -> pd.DataFrame:
    # one row per segment and part, with the first and last flow of the part, the cut points of the split
    label = flows["label"].to_numpy()
    family = flows["family"].astype(str).to_numpy()
    capture = flows["capture"].astype(str).to_numpy()
    te, row_id = flows["te"].to_numpy(), flows["row_id"].to_numpy()
    out = []
    starts = np.flatnonzero(np.r_[True, segment[1:] != segment[:-1]])
    for s, e in zip(starts, np.r_[starts[1:], len(segment)]):
        names = [f for f in litnet2020.ATTACK_FAMILIES if f in set(family[s:e][label[s:e] == 1])]
        for p, name in enumerate(PARTS):
            rows = s + np.flatnonzero(part[s:e] == p)
            attack = label[rows] == 1
            out.append({
                "segment": int(segment[s]), "label": " and ".join(litnet2020.FAMILY_LABELS[f] for f in names),
                "capture": capture[s], "families": "+".join(names), "filtered": capture[s] in filtered, "part": name,
                "first_te": te[rows[0]], "first_row_id": int(row_id[rows[0]]), "last_te": te[rows[-1]],
                "last_row_id": int(row_id[rows[-1]]), "flows": len(rows), "benign": int((~attack).sum()),
                "attacks": int(attack.sum()),
            })
    return pd.DataFrame(out)


def run(cfg: dict, run: runs.Run) -> None:
    d, sc, c, cc, dep, sim = (cfg["dataset"], cfg["scorer"], cfg["certify"], cfg["calibration"], cfg["deployment"],
                              cfg["simulation"])
    eps, delta, pi_max, part_b = c["epsilon"], c["delta"], c["pi_max"], c["delta_benign"]
    start = {"start_misses": c["start_misses"], "start_slack": c["start_slack"]}
    pi_cal, prevalences, mixes = cc["prevalence"], dep["prevalences"], dep["mixes"]
    methods, draws, shares_split = sim["methods"], sim["draws"], cfg["split"]["shares"]
    min_attacks = dep["min_family_attacks"]
    if len(shares_split) != len(PARTS):
        raise ValueError(f"split shares are {shares_split}, one for each of {', '.join(PARTS)}")
    if dep["main_prevalence"] not in prevalences:
        raise ValueError(f"main prevalence {dep['main_prevalence']} is not among the prevalences {prevalences}")
    if set(mixes) - set(MIXES) or len(set(mixes)) != len(mixes):
        raise ValueError(f"mixes are {mixes}, expected some of {', '.join(MIXES)}, each once")
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown methods {', '.join(sorted(unknown))}")
    filtered = set(d.get("filtered_captures", []))
    manifest = litnet2020.read_manifest(cfg)
    flows = load_flows(cfg, sc["features"])
    label = flows["label"].to_numpy()
    run.log(f"{d['name']} {d['processed_version']}, fingerprint {manifest['fingerprint']}, {len(flows)} flows, "
            f"{int(label.sum())} attacks, {flows['capture'].nunique()} captures")

    # the parts inside each capture, then the benign flows of the filtered captures leave every part
    split = split_flows(flows, shares_split, filtered)
    segment, part, kept, families, code = (split[key] for key in ("segment", "part", "kept", "families", "code"))
    cuts = segment_table(flows, segment, part, filtered)
    run.save_table(cuts, "cuts")
    segments = cuts.drop_duplicates("segment").reset_index(drop=True)
    for g in segments.itertuples():
        own = cuts[cuts["segment"] == g.segment]
        test = own[own["part"] == "test"].iloc[0]
        run.log(f"{g.label}, capture {g.capture}, flows " + ", ".join(str(v) for v in own["flows"])
                + ", attacks " + ", ".join(str(v) for v in own["attacks"]) + " in " + ", ".join(PARTS)
                + f", test part from te {test['first_te']} row {test['first_row_id']}, test prevalence "
                + f"{test['attacks'] / test['flows']:.5f}" + (", benign flows left out" if g.filtered else ""))

    # the scorer sees the training parts, and the selection parts only to stop
    features = sc["features"]
    train = np.flatnonzero((part == 0) & kept)
    sel = np.flatnonzero((part == 1) & kept)
    booster, enc, sel_loss = fit_scorer(flows, label, train, sel, sc, cfg["seed"])
    (run.results / "models").mkdir()
    booster.save_model(str(run.results / "models" / "scorer.txt"), num_iteration=booster.best_iteration)
    used = np.flatnonzero((part >= 1) & kept)
    score = score_rows(booster, enc, flows, used)
    u_part, u_segment, u_code, u_label = part[used], segment[used], code[used], label[used]
    sel_auc = auc(score[u_part == 1], u_label[u_part == 1])
    cal_auc = auc(score[u_part == 2], u_label[u_part == 2])
    run.log(f"trained LightGBM on {len(train)} flows with {int(label[train].sum())} attacks, best round "
            f"{booster.best_iteration}, selection loss {sel_loss:.5f}, selection AUC {sel_auc:.5f}, calibration AUC "
            f"{cal_auc:.5f}")
    run.save_arrays("scores", score=score, row=used.astype(np.int32), segment=u_segment.astype(np.int16),
                    part=u_part, family=u_code, families=np.array(["benign", *families]))

    grid = grid_from(score[(u_part == 1) & (u_code == 0)], c["grid_size"], c.get("grid_tail", []))
    run.log(f"grid of {len(grid)} distinct thresholds from {c['grid_size']} benign quantiles of the selection parts "
            f"and {len(c.get('grid_tail', []))} deeper levels")

    def counts_of(mask):
        return judge.class_counts(score[mask], u_code[mask], grid, len(families))

    sel_counts = counts_of(u_part == 1)
    cal_counts = counts_of(u_part == 2)
    windows = [counts_of((u_part == 3) & (u_segment == g)) for g in segments["segment"]]
    empty = [g.label for g, w in zip(segments.itertuples(), windows) if w["sizes"][0] == 0 and not g.filtered]
    if empty:
        raise ValueError(f"the test parts of {', '.join(empty)} hold no benign flow, so their windows have no benign "
                         "population")
    test_auc = [auc(score[(u_part == 3) & (u_segment == g)], u_label[(u_part == 3) & (u_segment == g)])
                for g in segments["segment"]]
    b_sel, m_sel_family, _, _ = judge.rates(sel_counts)
    b_cal, m_cal_family, _, _ = judge.rates(cal_counts)

    # calibration sets at the calibration prevalence, with all benign flows of the calibration parts and the attacks of
    # each family thinned to the mix of the config
    n_benign_cal, n_attack_cal = int(cal_counts["sizes"][0]), cal_counts["sizes"][1:]
    shares_cal = calibration_shares(n_attack_cal, pi_cal / (1 - pi_cal) * n_benign_cal, cc["mix"])
    n_cal = int(round(n_benign_cal / (1 - pi_cal)))
    expected = n_cal * pi_cal * shares_cal
    minimum = min_family_size(eps, pi_max, delta - part_b, c["family_floor"])
    groups = family_groups(expected, minimum)
    merged = [[f for f, g in zip(families, groups) if g == k] for k in range(groups.max() + 1)]
    run.log(f"calibration sets of {n_cal} flows at prevalence {pi_cal}, {cc['mix']} mix, expected attacks per family "
            + ", ".join(f"{f} {e:.0f}" for f, e in zip(families, expected))
            + f", a family needs {minimum} to stand alone, families " + ", ".join("+".join(f) for f in merged))
    if (sel_counts["sizes"][1:][shares_cal > 0] == 0).any():
        raise ValueError("a family of the calibration mix has no attack in the selection parts")

    # the two baselines tuned on the selection parts at the calibration prevalence and mix
    m_sel = shares_cal @ m_sel_family
    fixed = int(fixed_sequence(for_from_rates(pi_cal, m_sel, b_sel) <= eps, 0))
    beta = float(m_sel[fixed]) if fixed >= 0 else 0.0
    run.log(f"fixed threshold dismisses benign share {b_sel[fixed] if fixed >= 0 else 0.0:.4f} of the selection parts "
            f"with miss rate {beta:.5f} at the calibration mix, which conformal FNR takes as beta")

    # one stream for the calibration sets, one for each window, prevalence and mix of each arm in the order of ARMS,
    # whether the window has that arm or not, then one for each window at its own prevalence
    cells = len(segments) * len(prevalences) * len(mixes)
    seeds = np.random.SeedSequence(cfg["seed"]).spawn(1 + len(ARMS) * cells + len(segments))
    cal = draw_counts(np.random.default_rng(seeds[0]), n_cal, pi_cal, shares_cal, b_cal, m_cal_family, draws)
    grouped_below, grouped_n = group_rows(cal.family_below, cal.n_family, groups)
    pooled_below, pooled_n = cal.family_below.sum(1), cal.n_family.sum(1)
    args = (eps, delta, pi_max, part_b)
    no_floor = np.full(draws, np.nan)
    # LTT starts by the rule of FORGE where the config asks for it, runs from before the switch start it at the first
    # threshold that could pass with no attack
    ltt_start = {"n": cal.n_benign + cal.n_family.sum(1), **start} if c.get("ltt_guarded_start", False) else {}
    offline = {
        "forge_simplex": certify(cal.benign_below, cal.n_benign, grouped_below, grouped_n, *args, **start),
        "forge_pooled": certify(cal.benign_below, cal.n_benign, pooled_below[:, None], pooled_n[:, None], *args,
                                **start),
        "ltt_iid": {"index": ltt_iid(cal.benign_below, pooled_below, eps, delta, **ltt_start)["index"],
                    "floor": no_floor},
        "conformal_fnr": {"index": conformal_fnr(pooled_below, pooled_n, beta), "floor": no_floor},
        "fixed_sel": {"index": np.full(draws, fixed), "floor": no_floor},
    }
    for method, out in offline.items():
        hit = out["index"] >= 0
        run.log(f"{method}, refused in {np.mean(~hit):.3f} of calibration sets, mean benign share dismissed in the "
                f"calibration parts {np.where(hit, b_cal[np.maximum(out['index'], 0)], 0.0).mean():.4f}")
    # the bounds of the calibration sets serve every window
    certify_window = online_certifier(grouped_below, grouped_n, *args, **start)

    # rates at the offline FORGE thresholds, averaged over the calibration draws that certify one
    forge_index = offline["forge_simplex"]["index"]
    forge_thresholds = forge_index[forge_index >= 0]

    def at_forge(rate: np.ndarray) -> np.ndarray:
        if len(forge_thresholds) == 0:
            return np.full(rate.shape[:-1], np.nan)
        return rate[..., forge_thresholds].mean(-1)

    forge_cal = {"benign_dismissed": float(at_forge(b_cal)),
                 "miss": {f: float(v) for f, v in zip(families, at_forge(m_cal_family))}}
    run.log(f"calibration parts at the offline FORGE thresholds, benign share dismissed "
            f"{forge_cal['benign_dismissed']:.4f}, miss rate "
            + ", ".join(f"{f} {v:.4f}" for f, v in forge_cal["miss"].items()))

    filtered_window = segments["filtered"].to_numpy()
    benign_test = np.array([w["sizes"][0] for w in windows])
    # a window of a filtered capture takes the mean benign count of the other windows
    reference_benign = int(round(benign_test[~filtered_window].mean()))
    rows, natural_rows, values, natural_values, skipped = [], [], {}, {}, []
    shape = (len(segments), len(prevalences), len(mixes), draws)
    online_index = {arm: np.full(shape, -1, dtype=np.int16) for arm in ARMS}
    online_floor = {arm: np.zeros(shape, dtype=np.float32) for arm in ARMS}
    natural_index = np.full((len(segments), draws), -1, dtype=np.int16)
    natural_floor = np.zeros((len(segments), draws), dtype=np.float32)
    offset = {arm: 1 + a * cells for a, arm in enumerate(ARMS)}
    window_rows = []
    for i, (win, g) in enumerate(zip(windows, segments.itertuples())):
        # a window of a filtered capture holds no benign flow, and only its attacks are used
        with np.errstate(divide="ignore", invalid="ignore"):
            b_w, m_w_family, _, own = judge.rates(win)
        test_attacks = win["sizes"][1:]
        populations = {"reference": (b_cal, m_cal_family, n_attack_cal)}
        if not g.filtered:
            populations["s2"] = (b_w, m_cal_family, n_attack_cal)
            populations["real"] = (b_w, m_w_family, test_attacks)
        benign_w = reference_benign if g.filtered else int(win["sizes"][0])
        cell_base = {"window": i, "segment": g.label, "capture": g.capture}
        # the threshold safe for every covered mix needs the attack distributions of every family, which a real window
        # lacks, it holds those of its own capture only
        every = {arm: judge.every_mix_threshold(pi_max, m_family, b, groups, shares_cal, eps)
                 for arm, (b, m_family, _) in populations.items() if arm != "real"}
        for j, pi in enumerate(prevalences):
            full = int(round(benign_w / (1 - pi)))
            for k, mix in enumerate(mixes):
                shares = judge.mix_shares(mix, own, shares_cal, families)
                claim = in_claim(shares, shares_cal, groups)
                for arm, (b, m_family, attacks) in populations.items():
                    n_w, subsampled = full, False
                    # a real window stands for its test part thinned to its prevalence and mix, so the test part must
                    # hold at least the attacks of each family that the window takes on average. one that does not
                    # keeps all of them and has its benign flows subsampled, or is left out
                    if arm == "real" and np.any(attacks < full * pi * shares):
                        need = shares > 0
                        if not dep["subsample_benign"] or (attacks[need] == 0).any():
                            skipped.append([g.label, pi, mix])
                            continue
                        n_w, subsampled = int(np.floor((attacks[need] / (pi * shares[need])).min())), True
                    rng = np.random.default_rng(seeds[offset[arm] + (i * len(prevalences) + j) * len(mixes) + k])
                    drawn = draw_counts(rng, n_w, pi, shares, b, m_family, draws)
                    attack_below = drawn.family_below.sum(1)
                    window_below = drawn.benign_below + attack_below
                    online = certify_window(window_below, np.full(draws, n_w))
                    online_index[arm][i, j, k], online_floor[arm][i, j, k] = online["index"], online["floor"]
                    m = shares @ m_family
                    j_every = every.get(arm, -1)
                    every_share = (np.nan if arm not in every else
                                   float(pi * m[j_every] + (1 - pi) * b[j_every]) if j_every >= 0 else 0.0)
                    rows += judge.evaluate(methods, offline, online, pi, m, b, attack_below, window_below, eps, values,
                                           {"arm": arm, **cell_base, "prevalence": pi, "mix": mix,
                                            "window_sessions": n_w, "benign_subsampled": subsampled,
                                            "in_claim": claim,
                                            "few_attacks": bool((attacks[shares > 0] < min_attacks).any()),
                                            "draws": draws, **judge.best_thresholds(pi, pi_max, m, b, eps),
                                            "best_every_mix_index": j_every,
                                            "best_every_mix_dismissed": every_share})
        own_claim = in_claim(own, shares_cal, groups)
        natural = np.nan if g.filtered else float(test_attacks.sum() / win["sizes"].sum())
        if not g.filtered:
            # the real window at its own prevalence, all flows of its test part, no thinning
            n_w = int(win["sizes"].sum())
            rng = np.random.default_rng(seeds[1 + len(ARMS) * cells + i])
            drawn = draw_counts(rng, n_w, natural, own, b_w, m_w_family, draws)
            attack_below = drawn.family_below.sum(1)
            window_below = drawn.benign_below + attack_below
            online = certify_window(window_below, np.full(draws, n_w))
            natural_index[i], natural_floor[i] = online["index"], online["floor"]
            m = own @ m_w_family
            natural_rows += judge.evaluate(methods, offline, online, natural, m, b_w, attack_below, window_below, eps,
                                           natural_values,
                                           {"arm": "real", **cell_base, "prevalence": natural, "mix": "window",
                                            "window_sessions": n_w, "in_claim": own_claim,
                                            "up_to_pi_max": natural <= pi_max,
                                            "few_attacks": bool((test_attacks[own > 0] < min_attacks).any()),
                                            "draws": draws, **judge.best_thresholds(natural, pi_max, m, b_w, eps)})
        # a family absent from a window has no miss rate there
        miss = np.where(test_attacks > 0, at_forge(m_w_family), np.nan)
        window_rows.append({
            **cell_base, "families": g.families, "filtered": bool(g.filtered), "in_claim": own_claim,
            "test_flows": int(win["sizes"].sum()), "benign": int(win["sizes"][0]),
            **{f: int(n) for f, n in zip(families, test_attacks)}, "natural_prevalence": natural,
            "auc": float(test_auc[i]), "forge_benign_dismissed": float(at_forge(b_w)) if not g.filtered else np.nan,
            **{f"forge_miss_{f}": float(v) for f, v in zip(families, miss)},
        })
    if skipped:
        run.log(f"{len(skipped)} real windows, prevalences and mixes left out, their test parts lack the attacks of a "
                f"family of the mix")
    claims = pd.DataFrame(rows)
    run.save_table(claims, "claims")
    natural_table = pd.DataFrame(natural_rows)
    run.save_table(natural_table, "natural")
    window_table = pd.DataFrame(window_rows)
    run.save_table(window_table, "windows")
    for w in window_table.itertuples():
        rates_of = ", ".join(f"{f} {getattr(w, 'forge_miss_' + f):.4f} over {getattr(w, f)} test attacks against "
                             f"{forge_cal['miss'][f]:.4f}" for f in families if getattr(w, f) > 0)
        benign_of = ("benign flows left out, at the offline FORGE thresholds" if w.filtered else
                     f"AUC {w.auc:.4f}, test prevalence {w.natural_prevalence:.5f}, at the offline FORGE thresholds "
                     f"benign share dismissed {w.forge_benign_dismissed:.4f} against "
                     f"{forge_cal['benign_dismissed']:.4f} in calibration,")
        run.log(f"{w.segment}, {benign_of} miss rate {rates_of}" + ("" if w.in_claim else ", outside the FORGE claim"))

    # pooled over all windows of each arm and over those inside the FORGE claim, which a window of one pooled family
    # lies outside
    summary_rows = []
    for (arm, method, pi, mix), cell in claims.groupby(["arm", "method", "prevalence", "mix"], sort=False):
        for subset, g in [("all", cell), ("in_claim", cell[cell["in_claim"]])]:
            if len(g):
                certified = np.concatenate([values[(arm, method, pi, mix, w)] for w in g["window"]])
                summary_rows.append({"arm": arm, "method": method, "prevalence": pi, "mix": mix, "subset": subset,
                                     **judge.pooled(g, certified, draws),
                                     "best_every_mix_dismissed": g["best_every_mix_dismissed"].mean()})
    by_prevalence = pd.DataFrame(summary_rows)
    run.save_table(by_prevalence, "by_prevalence")

    # the claims of FORGE cover prevalences up to pi_max and the windows inside the claim, and the largest window
    # violation share there, within each arm and mix, is the number to set against delta
    covered = claims[(claims["prevalence"] <= pi_max) & claims["in_claim"]]
    largest = {}
    for (arm, method, mix), g in covered.groupby(["arm", "method", "mix"], sort=False):
        top = g.loc[g["rate"].idxmax()]
        largest.setdefault(arm, {}).setdefault(method, {})[mix] = {
            "segment": top["segment"], "prevalence": float(top["prevalence"]), "rate": float(top["rate"]),
            "low": float(top["low"]), "high": float(top["high"])}
    main = dep["main_prevalence"]
    for arm in ARMS:
        for mix in mixes:
            same = by_prevalence[(by_prevalence["arm"] == arm) & (by_prevalence["subset"] == "in_claim")
                                 & (by_prevalence["mix"] == mix)]
            for method in methods:
                g = same[same["method"] == method].sort_values("prevalence")
                if g.empty:
                    continue
                top = largest.get(arm, {}).get(method, {}).get(mix)
                if top is None:
                    worst = "no window up to pi_max"
                elif top["rate"] == 0:
                    worst = "none in any window up to pi_max"
                else:
                    worst = (f"at most {top['rate']:.4f} in one window up to pi_max, {top['segment']} at prevalence "
                             f"{top['prevalence']}")
                dismissed = g.loc[g["prevalence"] == main, "dismissed_mean"]
                run.log(f"{method}, {ARM_LABELS[arm]} in the claim, {mix} mix, FOR above epsilon in "
                        + ", ".join(f"{r:.4f} at {pi}" for pi, r in zip(g["prevalence"], g["violation_rate"]))
                        + f" of draw and window pairs, {worst}, refused {g['refused'].mean():.3f}, mean share of window "
                        f"flows dismissed {dismissed.iloc[0] if len(dismissed) else np.nan:.4f} at prevalence {main}")
            # the yardsticks follow from the windows alone, so the rows of any method hold them
            g = same[same["method"] == methods[0]].sort_values("prevalence")
            if g.empty:
                continue
            best = g.loc[g["prevalence"] == main, "best_up_to_pi_max_dismissed"]
            every = g.loc[g["prevalence"] == main, "best_every_mix_dismissed"]
            every_part = (f", by the loosest safe for every mix the claim covers {every.iloc[0]:.4f} at prevalence {main}"
                          if len(every) and np.isfinite(every.iloc[0]) else "")
            run.log(f"best thresholds, {ARM_LABELS[arm]} in the claim, {mix} mix, mean share of window flows dismissed "
                    f"by the loosest threshold with FOR at or below epsilon up to pi_max "
                    f"{best.iloc[0] if len(best) else np.nan:.4f} at prevalence {main}{every_part}, by the loosest at "
                    "the window's prevalence "
                    + ", ".join(f"{s:.4f} at {pi}" for pi, s in zip(g["prevalence"], g["best_at_prevalence_dismissed"])))
    # the table of E2, one line per window and arm with the family of its capture at the main prevalence, and one per
    # real window at its own prevalence
    at_main = claims[(claims["prevalence"] == main) & (claims["mix"] == "window")]
    for (i, arm), g in at_main.groupby(["window", "arm"], sort=True):
        run.log(f"{g['segment'].iloc[0]}, {WINDOW_WORDS[arm]} at prevalence {main}, FOR above epsilon in "
                + ", ".join(f"{r.method} {r.rate:.4f}" for r in g.itertuples())
                + ("" if g["in_claim"].iloc[0] else ", outside the FORGE claim"))
    for i, g in natural_table.groupby("window", sort=True):
        run.log(f"{g['segment'].iloc[0]}, real window at its own prevalence {g['prevalence'].iloc[0]:.5f}, FOR above "
                f"epsilon in " + ", ".join(f"{r.method} {r.rate:.4f}" for r in g.itertuples())
                + ", mean share dismissed " + ", ".join(f"{r.method} {r.dismissed_mean:.4f}" for r in g.itertuples())
                + ("" if g["up_to_pi_max"].iloc[0] else ", above pi_max"))

    # the attack columns at the calibration mix, the one the baselines read
    grid_table = pd.DataFrame({"index": np.arange(len(grid)), "threshold": grid, "selection_benign": b_sel,
                               "selection_attack": m_sel, "calibration_benign": b_cal,
                               "calibration_attack": shares_cal @ m_cal_family})
    for f, m in zip(families, m_cal_family):
        grid_table[f"calibration_{f}"] = m
    run.save_table(grid_table, "grid")

    def stacked(key):
        return np.stack([w[key] for w in windows])

    run.save_arrays("histograms", grid=grid, families=np.array(families), groups=groups, shares=shares_cal,
                    selection_benign_below=sel_counts["benign_below"],
                    selection_family_below=sel_counts["family_below"], selection_sizes=sel_counts["sizes"],
                    calibration_benign_below=cal_counts["benign_below"],
                    calibration_family_below=cal_counts["family_below"], calibration_sizes=cal_counts["sizes"],
                    window_benign_below=stacked("benign_below"), window_family_below=stacked("family_below"),
                    window_sizes=stacked("sizes"))
    run.save_arrays("draws", prevalences=np.array(prevalences), mixes=np.array(mixes), arms=np.array(ARMS),
                    forge_floor=offline["forge_simplex"]["floor"], forge_pooled_floor=offline["forge_pooled"]["floor"],
                    **{f"{m}_index": out["index"] for m, out in offline.items()},
                    online_index=np.stack([online_index[arm] for arm in ARMS]),
                    online_floor=np.stack([online_floor[arm] for arm in ARMS]),
                    natural_online_index=natural_index, natural_online_floor=natural_floor,
                    counts=np.column_stack([cal.n_benign, cal.n_family]))
    run.save_json({"encoder": enc.to_dict(), "features": features, "params": sc["params"], "rounds": sc["rounds"],
                   "early_stopping": sc["early_stopping"], "best_iteration": booster.best_iteration,
                   "selection_loss": sel_loss, "train_flows": int(len(train)),
                   "train_attacks": int(label[train].sum()), "selection_auc": sel_auc, "calibration_auc": cal_auc},
                  "scorer")
    run.save_json({
        "dataset": d["name"], "processed_version": d["processed_version"], "fingerprint": manifest["fingerprint"],
        "split_shares": shares_split, "filtered_captures": sorted(filtered), "segments": segments["label"].tolist(),
        "epsilon": eps, "delta": delta, "delta_benign": part_b, "pi_max": pi_max, **start,
        "grid_size": c["grid_size"], "grid_tail": c.get("grid_tail", []), "grid_thresholds": len(grid),
        "family_floor": c["family_floor"], "family_minimum": minimum, "families": merged,
        "calibration_prevalence": pi_cal, "calibration_mix": cc["mix"], "calibration_sessions": n_cal,
        "calibration_attacks": dict(zip(families, n_attack_cal.tolist())),
        "calibration_expected_attacks": dict(zip(families, expected.tolist())), "reference_benign": reference_benign,
        "fixed_index": fixed, "beta": beta, "prevalences": prevalences, "main_prevalence": main, "arms": ARMS,
        "mixes": mixes, "subsample_benign": dep["subsample_benign"], "skipped": skipped, "methods": methods,
        "draws": draws, "windows": len(segments),
        "offline_refused": {m: float(np.mean(out["index"] < 0)) for m, out in offline.items()},
        "selection_auc": sel_auc, "calibration_auc": cal_auc, "forge_calibration": forge_cal,
        "min_family_attacks": min_attacks, "largest_violation_up_to_pi_max": largest,
    }, "summary")

    for name, draw in FIGURES.items():
        save_figure(draw(run.dir), run.figures, name)
        run.log(f"saved figure {name}")


def _line(method: str) -> dict:
    return {"color": style.color_for(method), "marker": MARKERS[method], "label": LABELS[method]}


def _for_by_prevalence(run_dir, arm: str) -> plt.Figure:
    # the windows inside the FORGE claim, every window at the calibration mix and the windows of families that stand
    # alone at their own mix
    style.use()
    table = runs.load_table(run_dir, "by_prevalence")
    summary = runs.load_json(run_dir, "summary")
    methods = [m for m in METHODS if m in summary["methods"]]
    data = table[(table["arm"] == arm) & (table["subset"] == "in_claim")]
    return prevalence.for_by_prevalence(data, summary, methods, _line, summary["mixes"], MIX_LABELS, WINDOW_WORDS[arm],
                                        "flows", [*BEST_LINES, EVERY_LINE])


def fig_for_by_prevalence(run_dir) -> plt.Figure:
    # the reference windows, where label shift holds by construction, the figure of RQ1
    return _for_by_prevalence(run_dir, "reference")


def fig_for_by_prevalence_s2(run_dir) -> plt.Figure:
    # the windows with their own benign flows and the calibration attacks, the setting of Theorem 1b
    return _for_by_prevalence(run_dir, "s2")


def fig_for_by_prevalence_real(run_dir) -> plt.Figure:
    return _for_by_prevalence(run_dir, "real")


def _window_axis(axes, windows: pd.DataFrame):
    # windows in time order on the shared x axis, named by their families, and a light band behind the windows of one
    # pooled family, which lie outside the FORGE claim. returns the legend handle of the band
    for ax, letter in zip(axes, "abcdefgh"):
        for w in windows.loc[~windows["in_claim"], "window"]:
            ax.axvspan(w - 0.5, w + 0.5, color=style.GRAY, alpha=0.15, linewidth=0)
        layout.panel_letter(ax, letter)
    axes[-1].set_xticks(windows["window"], windows["segment"], rotation=30, ha="right")
    axes[-1].set_xlim(windows["window"].min() - 0.6, windows["window"].max() + 0.6)
    return Patch(color=style.GRAY, alpha=0.15, linewidth=0)


def fig_violation_by_window(run_dir) -> plt.Figure:
    style.use()
    claims = runs.load_table(run_dir, "claims")
    windows = runs.load_table(run_dir, "windows").sort_values("window")
    summary = runs.load_json(run_dir, "summary")
    methods = [m for m in METHODS if m in summary["methods"]]
    arms = summary["arms"]
    data = claims[(claims["prevalence"] == summary["main_prevalence"]) & (claims["mix"] == "window")]
    fig, axes = plt.subplots(len(arms) + 1, 1, figsize=(style.WIDTH_DOUBLE, 1.2 + 1.5 * (len(arms) + 1)),
                             sharex=True)
    # one row per arm with the share of draws with FOR above epsilon in each window and its 95% interval, the windows
    # of the filtered captures in the reference row only, and a last row with the mean share of the reference window
    # flows dismissed and the best threshold up to pi_max
    for ax, arm in zip(axes, arms):
        for k, method in enumerate(methods):
            g = data[(data["arm"] == arm) & (data["method"] == method)].sort_values("window")
            layout.points(ax, layout.dodge(g["window"], k, len(methods), 0.1), g["rate"], g["low"], g["high"],
                          linestyle="none", **_line(method))
        ax.axhline(summary["delta"], color=style.GRAY, linestyle="--", linewidth=1, label=r"$\delta$")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(f"share with FOR > $\\varepsilon$\n{ARM_LABELS[arm]}")
    reference = data[data["arm"] == "reference"]
    for k, method in enumerate(methods):
        g = reference[reference["method"] == method].sort_values("window")
        axes[-1].plot(layout.dodge(g["window"], k, len(methods), 0.1), g["dismissed_mean"], linestyle="none",
                      markersize=3.5, **_line(method))
    g = reference[reference["method"] == methods[0]].sort_values("window")
    name, _, label = BEST_LINES[0]
    best = axes[-1].plot(g["window"], g[f"{name}_dismissed"], color=style.INK, linestyle="none", marker="_",
                         markersize=14, markeredgewidth=1.2, label=label)[0]
    axes[-1].set_ylim(0, 1)
    axes[-1].yaxis.set_major_formatter(style.percent)
    axes[-1].set_ylabel(f"window flows dismissed\n{ARM_LABELS['reference']}")
    band = _window_axis(axes, windows)
    layout.legend_below(fig, axes[0], [LABELS[m] for m in methods],
                        [(best, label), (band, "outside the FORGE claim")])
    return fig


def fig_drift_by_window(run_dir) -> plt.Figure:
    style.use()
    windows = runs.load_table(run_dir, "windows").sort_values("window")
    summary = runs.load_json(run_dir, "summary")
    cal = summary["forge_calibration"]
    families = [f for f in cal["miss"] if f in windows]
    fig, axes = plt.subplots(3, 1, figsize=(style.WIDTH_DOUBLE, 1.2 + 1.5 * 3), sharex=True)
    # top, the AUC of the scorer in each test part. middle, the miss rate of each family of a window at the offline
    # FORGE thresholds over the same in its calibration part, from windows with enough test attacks of the family.
    # bottom, the benign share dismissed at those thresholds over the same in the calibration parts. a ratio of 1 is no
    # drift, the offline claim needs both ratios near 1, the online claim the middle one
    point = {"color": style.INK, "markersize": 3.5, "linestyle": "none"}
    axes[0].plot(windows["window"], windows["auc"], marker="o", **point)
    calibration = axes[0].axhline(summary["calibration_auc"], color=style.GRAY, linestyle="--", linewidth=1)
    axes[0].set_ylabel("AUC of the scorer")
    x, ratio = [], []
    for f in families:
        kept = windows[windows[f] >= summary["min_family_attacks"]]
        x += kept["window"].tolist()
        ratio += (kept[f"forge_miss_{f}"] / cal["miss"][f]).tolist()
    ratios = {"attack": np.array(ratio, dtype=float),
              "benign": (windows["forge_benign_dismissed"] / cal["benign_dismissed"]).to_numpy()}
    # a window with no miss at the thresholds has a ratio of 0, which sits on the lower edge of the log axis
    shown = np.concatenate(list(ratios.values()))
    shown = shown[np.isfinite(shown) & (shown > 0)]
    low = 10.0 ** np.floor(np.log10(min(shown.min(), 1.0))) if len(shown) else 0.1
    high = 10.0 ** np.ceil(np.log10(max(shown.max(), 1.0))) if len(shown) else 10.0
    axes[1].plot(x, np.where(ratios["attack"] == 0, low, ratios["attack"]), marker="s", **point)
    axes[2].plot(windows["window"], np.where(ratios["benign"] == 0, low, ratios["benign"]), marker="o", **point)
    for ax in axes[1:]:
        ax.axhline(1, color=style.GRAY, linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_ylim(low, high)
    axes[1].set_ylabel("attacks dismissed,\nwindow over calibration")
    axes[2].set_ylabel("benign flows dismissed,\nwindow over calibration")
    band = _window_axis(axes, windows)
    layout.legend_below(fig, axes[0], [], [(calibration, "calibration parts"), (band, "outside the FORGE claim")])
    return fig


FIGURES = {
    "for_by_prevalence": fig_for_by_prevalence,
    "for_by_prevalence_s2": fig_for_by_prevalence_s2,
    "for_by_prevalence_real": fig_for_by_prevalence_real,
    "violation_by_window": fig_violation_by_window,
    "drift_by_window": fig_drift_by_window,
}
