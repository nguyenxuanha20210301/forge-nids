"""e13, RQ1 on UGR'16 and the source of E3 there, the false omission rate and the dismissed share of each method in
the deployment windows of six hours of August 2016, their attacks thinned to a range of prevalences.

UGR'16 is the netflow of a Spanish ISP with a clock, so the roles are stretches of time cut at window boundaries and
the archive week of a flow is provenance only. the scorer trains on the first days of the capture, the selection and
the calibration parts are the two halves of the week after, and the deployment is every window from the start of the
week after that to the last flow, 86 windows of six hours in the published weeks, through which the attacks the
authors launched stop after three bursts, the spam campaign surges and fades and the benign traffic runs its daily
cycle.

the blacklist is left out of every class, role and window, since its label marks an address on a public list and not
the traffic, which a scorer without addresses cannot learn, and as an attack family it would decide every
certificate. a labelled family without calibration attacks leaves the same way, since the claim on a family rests on
its calibration sample, so no window is marked outside the claim on a handful of flows of a family the certificate
never saw. its flows in the scored span are not wasted, the share of them dismissed at the threshold of every method
in every draw is the price of a real anomaly outside every claim, saved as a table per such family.

a LightGBM scorer is trained on the training part and stopped early on the selection part, the grid holds benign
quantiles of the selection part and a few deeper levels, the calibration part keeps all its benign flows and has its
attacks thinned to the calibration prevalence with one common cap on every family, and each draw is a calibration set
of that size drawn with replacement from it, as in e12. the fixed threshold and the beta of conformal FNR read the
selection part at the same mix.

every window comes in three arms, as in e12.
    reference   the benign count, prevalence and mix of the window with the score distributions of the calibration
                part for both classes, so label shift holds by construction. RQ1 is read here
    s2          the window's own benign flows with the calibration distributions of the attack families, Theorem 1b
                against the daily cycle and three weeks of benign drift
    real        the window's own benign flows and attacks
a window keeps all its benign flows and has its attacks thinned to each prevalence, with its own mix and the
calibration mix. a real window short of attacks keeps them all and has its benign flows subsampled, a window without
attacks has no mix of its own and reference and s2 windows at the calibration mix only, and each real window is also
evaluated at its own prevalence, apart, which on this dataset runs from nothing to above pi_max. the methods and their
claims are those of e12.

the deployment window of every scored flow is saved with its score, so the streams of e24 can replay the real order
of every window, and the class counts of every window at the grid are saved as e11 saves its block histograms.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from forge import runs
from forge.certify.certificate import (certify, conformal_fnr, counts_below, family_groups, fixed_sequence,
                                       for_from_rates, grid_from, group_rows, ltt_iid, min_family_size,
                                       online_certifier)
from forge.data import splits, ugr16
from forge.data.synthetic import draw_counts
from forge.eval import windows as judge
from forge.eval.metrics import auc
from forge.experiments.for_vs_prevalence import (ESTIMATED, PER_WINDOW, estimate_settings, estimate_summary,
                                                 estimated_methods, log_estimates)
from forge.experiments.litnet_prevalence import LABELS as E12_LABELS
from forge.experiments.litnet_prevalence import (ARM_LABELS, ARMS, DERIVED, EVERY_LINE, MARKERS, METHODS,
                                                 WINDOW_WORDS, calibration_shares, fit_scorer, in_claim,
                                                 raw_columns, score_rows)
from forge.viz import layout, prevalence, style
from forge.viz.export import save_figure
from forge.viz.prevalence import BEST_LINES

# the labels of e12, with the one selection part of the roles by clock
LABELS = {**E12_LABELS, "fixed_sel": "fixed threshold, selection part"}
ROLES = ["train", "selection", "calibration", "deployment"]
# the config key that starts each role after the first one, which starts at the first flow
ROLE_STARTS = {"selection": "selection_start", "calibration": "calibration_start", "deployment": "deployment_start"}
# the mix of attack families in a window, the window's own or the mix of the calibration set
MIXES = ["window", "calibration"]
MIX_LABELS = {"window": "own mix of the window", "calibration": "calibration mix of families"}
COLUMNS = ["te", "week", "label", "family"]
BENIGN = ugr16.FAMILY_ORDER[0]
# the table of a family left out for lack of calibration attacks is named without the prefix of the authors' labels
UNSEEN_PREFIX = "anomaly_"
# markers and dashes of the families in the drift figure, which also differ by color
FAMILY_MARKERS = ["o", "s", "^", "D", "v", "p", "<", ">"]


def load_flows(cfg: dict, features: list[str]) -> pd.DataFrame:
    flows = ugr16.load(cfg, columns=COLUMNS + [c for c in raw_columns(features) if c not in COLUMNS])
    for name, (source, derive) in DERIVED.items():
        if name in features:
            flows[name] = derive(flows[source].to_numpy())
    return flows


def role_cuts(split: dict, length: str) -> np.ndarray:
    """the start of selection, calibration and deployment as datetime64, each the start of a window of the fixed
    length and later than the one before, so every role is a run of whole windows."""
    cuts = []
    for role in ROLES[1:]:
        key = ROLE_STARTS[role]
        t = pd.Timestamp(split[key])
        if t != t.floor(length):
            raise ValueError(f"{key} is {t}, which is not the start of a window of {length}")
        cuts.append(t)
    for before, after, role in zip(cuts, cuts[1:], ROLES[2:]):
        if after <= before:
            raise ValueError(f"{ROLE_STARTS[role]} {after} must come after "
                             f"{ROLE_STARTS[ROLES[ROLES.index(role) - 1]]} {before}")
    return np.array([t.to_datetime64() for t in cuts]).astype("datetime64[ms]")


def assign_roles(te, cuts: np.ndarray) -> np.ndarray:
    # 0 for a flow before the first cut, then 1, 2 and 3 from each cut on, in the order of ROLES
    return np.searchsorted(cuts, np.asarray(te).astype("datetime64[ms]"), side="right").astype(np.int8)


def deployment_windows(ids: np.ndarray, table: pd.DataFrame, deployed: np.ndarray,
                       kept: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, int]:
    """the deployment window of every flow, -1 outside the deployment, and the table of the deployment windows that
    hold a kept flow, numbered in time order from 0 with their start and end, and the count of windows of the span
    without one, which have no population and are left out. ids and table come from the fixed windows of the clock."""
    held_rows = deployed & kept
    if not held_rows.any():
        raise ValueError("no kept flow falls in the deployment, which starts after the last flow")
    first = int(ids[held_rows].min())
    span = table[table["window"] >= first].reset_index(drop=True)
    held = np.bincount(ids[held_rows] - first, minlength=len(span)) > 0
    remap = np.full(len(span), -1, dtype=np.int32)
    remap[held] = np.arange(int(held.sum()))
    window = np.full(len(ids), -1, dtype=np.int32)
    in_span = deployed & (ids >= first)
    window[in_span] = remap[ids[in_span] - first]
    windows = span[held].reset_index(drop=True)
    windows["window"] = np.arange(len(windows))
    return window, windows, int((~held).sum())


def family_codes(family: pd.Series, role: np.ndarray, left_out) -> dict:
    """the families of the experiment, every labelled family outside left_out with at least one calibration attack,
    in the category order of the table, the class code of every flow, 0 for a benign flow, 1 and up for the families
    in their order and -1 for a flow left out, and the families left out with the reason."""
    names = list(family.cat.categories)
    codes = family.cat.codes.to_numpy()
    total = np.bincount(codes, minlength=len(names))
    in_cal = np.bincount(codes[role == 2], minlength=len(names))
    families, left = [], {}
    remap = np.full(len(names), -1, dtype=np.int8)
    for k, name in enumerate(names):
        if name == BENIGN:
            remap[k] = 0
        elif name in left_out:
            if total[k]:
                left[name] = "left out"
        elif in_cal[k] > 0:
            remap[k] = len(families) + 1
            families.append(name)
        elif total[k]:
            left[name] = "no calibration attack"
    if not families:
        raise ValueError("no labelled family has a calibration attack")
    return {"names": names, "codes": codes, "families": families, "code": remap[codes], "left": left}


def unseen_name(family: str) -> str:
    return f"unseen_{family.removeprefix(UNSEEN_PREFIX)}"


def roles_table(te, ids: np.ndarray, table: pd.DataFrame, role: np.ndarray, fam: dict, cuts: np.ndarray,
                left_count) -> pd.DataFrame:
    # one row per role with its clock span, its windows, its first and last flow, its kept flows by class and the
    # flows of every family left out
    names, codes, code = fam["names"], fam["codes"], fam["code"]
    starts, ends = table["start"].to_numpy(), table["end"].to_numpy()
    out = []
    for r, name in enumerate(ROLES):
        rows = role == r
        if not rows.any():
            raise ValueError(f"the {name} part holds no flow")
        mine = ids[rows]
        kept = rows & (code >= 0)
        by_family = np.bincount(codes[rows], minlength=len(names))
        out.append({
            "role": name,
            "start": pd.Timestamp(starts[mine.min()] if r == 0 else cuts[r - 1]),
            "end": pd.Timestamp(cuts[r] if r < len(cuts) else ends[mine.max()]),
            "windows": int(mine.max() - mine.min() + 1),
            "first_te": pd.Timestamp(te[rows].min()), "last_te": pd.Timestamp(te[rows].max()),
            "flows": int(kept.sum()), "benign": int((rows & (code == 0)).sum()), "attacks": int((rows & (code > 0)).sum()),
            **{f: int(by_family[names.index(f)]) for f in fam["families"]},
            **{f: int(by_family[names.index(f)]) for f in left_count},
        })
    return pd.DataFrame(out)


def unseen_table(family: str, below: np.ndarray, n: int, offline: dict, natural: dict, methods) -> pd.DataFrame:
    """the share of the flows of a family outside every claim at or below the threshold of every method, over the
    calibration draws for a method with one threshold per draw and over the real windows at their own prevalence and
    the draws for a method with a threshold per window, a refusal dismissing nothing."""
    rows = []
    for method in methods:
        if method in natural:
            index, windows = np.asarray(natural[method]).ravel(), np.asarray(natural[method]).shape[0]
        else:
            index, windows = np.asarray(offline[method]["index"]), 0
        hit = index >= 0
        share = np.where(hit, below[np.maximum(index, 0)] / n, 0.0)
        rows.append({
            "family": family, "method": method, "flows": int(n), "thresholds": int(len(index)),
            "natural_windows": int(windows), "refused": float(np.mean(~hit)), "dismissed_mean": float(share.mean()),
            "dismissed_median": float(np.median(share[hit])) if hit.any() else np.nan,
            "dismissed_low": float(np.quantile(share[hit], judge.LOW)) if hit.any() else np.nan,
            "dismissed_high": float(np.quantile(share[hit], judge.HIGH)) if hit.any() else np.nan,
        })
    return pd.DataFrame(rows)


def _counts(family: str, names: list[str]) -> str:
    return ", ".join(f"{f} {n}" for f, n in names) if names else "none"


def run(cfg: dict, run: runs.Run) -> None:
    d, sc, c, cc, dep, sim = (cfg["dataset"], cfg["scorer"], cfg["certify"], cfg["calibration"], cfg["deployment"],
                              cfg["simulation"])
    eps, delta, pi_max, part_b = c["epsilon"], c["delta"], c["pi_max"], c["delta_benign"]
    start = {"start_misses": c["start_misses"], "start_slack": c["start_slack"]}
    pi_cal, prevalences, mixes = cc["prevalence"], dep["prevalences"], dep["mixes"]
    methods, draws = sim["methods"], sim["draws"]
    min_attacks = dep["min_family_attacks"]
    length, t_col = d["window_freq"], d["time_column"]
    if t_col != "te":
        raise ValueError(f"time_column is {t_col}, the processed table is in te order and the windows follow it")
    if dep["main_prevalence"] not in prevalences:
        raise ValueError(f"main prevalence {dep['main_prevalence']} is not among the prevalences {prevalences}")
    if set(mixes) - set(MIXES) or len(set(mixes)) != len(mixes):
        raise ValueError(f"mixes are {mixes}, expected some of {', '.join(MIXES)}, each once")
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown methods {', '.join(sorted(unknown))}")
    left_out = list(cfg["families"]["left_out"])
    cuts = role_cuts(cfg["split"], length)
    manifest = ugr16.read_manifest(cfg)
    flows = load_flows(cfg, sc["features"])
    label = flows["label"].to_numpy()
    te = flows["te"].to_numpy()
    run.log(f"{d['name']} {d['processed_version']}, fingerprint {manifest['fingerprint']}, {len(flows)} flows, every "
            f"{manifest['sample_every']}th of {manifest['rows_in_weeks']}, {int(label.sum())} labelled, from "
            f"{pd.Timestamp(te[0])} to {pd.Timestamp(te[-1])}")

    # the roles by clock, the families with calibration attacks, then the deployment windows of the kept flows
    role = assign_roles(te, cuts)
    fam = family_codes(flows["family"], role, set(left_out))
    families, code, names, cat_codes = fam["families"], fam["code"], fam["names"], fam["codes"]
    kept = code >= 0
    ids, clock = splits.fixed_windows(flows["te"], length), splits.fixed_window_table(flows["te"], length)
    window, win_table, empty = deployment_windows(ids, clock, role == 3, kept)
    roles = roles_table(te, ids, clock, role, fam, cuts, fam["left"])
    run.save_table(roles, "roles")
    run.save_table(win_table, "deployment_windows")
    for r in roles.itertuples():
        run.log(f"{r.role}, {r.start:%Y-%m-%d %H:%M} to {r.end:%Y-%m-%d %H:%M}, {r.windows} windows of {length}, "
                f"flows from {r.first_te} to {r.last_te}, kept {r.flows}, benign {r.benign}, attacks "
                + _counts("", [(f, getattr(r, f)) for f in families]) + ", left out "
                + _counts("", [(f, getattr(r, f)) for f in fam["left"]]))
    run.log(f"families with calibration attacks {', '.join(families)}, left out "
            + ", ".join(f"{f} {why}" for f, why in fam["left"].items())
            + f", {len(win_table)} deployment windows hold kept flows"
            + (f", {empty} windows of the deployment span hold none and are left out" if empty else ""))
    if len(fam["left"]) != len(left_out) + sum(why != "left out" for why in fam["left"].values()):
        missing = [f for f in left_out if f not in fam["left"]]
        run.log(f"left out by the config but absent from the table, {', '.join(missing)}")

    # the scorer sees the training part, and the selection part only to stop
    features = sc["features"]
    train = np.flatnonzero((role == 0) & kept)
    sel = np.flatnonzero((role == 1) & kept)
    booster, enc, sel_loss = fit_scorer(flows, label, train, sel, sc, cfg["seed"])
    (run.results / "models").mkdir()
    booster.save_model(str(run.results / "models" / "scorer.txt"), num_iteration=booster.best_iteration)
    used = np.flatnonzero((role >= 1) & kept)
    score = score_rows(booster, enc, flows, used)
    u_role, u_window, u_code, u_label = role[used], window[used], code[used], label[used]
    sel_auc = auc(score[u_role == 1], u_label[u_role == 1])
    cal_auc = auc(score[u_role == 2], u_label[u_role == 2])
    run.log(f"trained LightGBM on {len(train)} flows with {int(label[train].sum())} attacks, best round "
            f"{booster.best_iteration}, selection loss {sel_loss:.5f}, selection AUC {sel_auc:.5f}, calibration AUC "
            f"{cal_auc:.5f}")
    run.save_arrays("scores", score=score, row=used.astype(np.int32), role=u_role, window=u_window.astype(np.int16),
                    family=u_code, families=np.array([BENIGN, *families]), roles=np.array(ROLES))
    # the flows of a family left out for lack of calibration attacks that lie in the scored span, scored for the
    # share of them each method would dismiss
    unseen = {}
    for f, why in fam["left"].items():
        rows = np.flatnonzero((role >= 1) & (cat_codes == names.index(f)))
        if why != "left out" and len(rows):
            unseen[f] = {"score": score_rows(booster, enc, flows, rows), "rows": rows}

    grid = grid_from(score[(u_role == 1) & (u_code == 0)], c["grid_size"], c.get("grid_tail", []))
    run.log(f"grid of {len(grid)} distinct thresholds from {c['grid_size']} benign quantiles of the selection part "
            f"and {len(c.get('grid_tail', []))} deeper levels")

    def counts_of(mask):
        return judge.class_counts(score[mask], u_code[mask], grid, len(families))

    sel_counts = counts_of(u_role == 1)
    cal_counts = counts_of(u_role == 2)
    # the deployment rows in window order, so each window is one slice
    order = np.argsort(u_window, kind="stable")
    edges = np.searchsorted(u_window[order], np.arange(len(win_table) + 1))
    windows, test_auc = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        rows = order[lo:hi]
        windows.append(judge.class_counts(score[rows], u_code[rows], grid, len(families)))
        test_auc.append(auc(score[rows], u_label[rows]))
    empty_benign = [f"{s:%Y-%m-%d %H:%M}" for s, w in zip(win_table["start"], windows) if w["sizes"][0] == 0]
    if empty_benign:
        raise ValueError(f"the windows {', '.join(empty_benign)} hold no benign flow, so they have no benign "
                         "population")
    b_sel, m_sel_family, _, _ = judge.rates(sel_counts)
    b_cal, m_cal_family, _, _ = judge.rates(cal_counts)

    # calibration sets at the calibration prevalence, with all benign flows of the calibration part and the attacks of
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
        raise ValueError("a family of the calibration mix has no attack in the selection part")

    # the two baselines tuned on the selection part at the calibration prevalence and mix
    m_sel = shares_cal @ m_sel_family
    fixed = int(fixed_sequence(for_from_rates(pi_cal, m_sel, b_sel) <= eps, 0))
    beta = float(m_sel[fixed]) if fixed >= 0 else 0.0
    run.log(f"fixed threshold dismisses benign share {b_sel[fixed] if fixed >= 0 else 0.0:.4f} of the selection part "
            f"with miss rate {beta:.5f} at the calibration mix, which conformal FNR takes as beta")

    # one stream for the calibration sets, one for each window, prevalence and mix of each arm in the order of ARMS,
    # whether the window has that arm or not, then one for each window at its own prevalence
    cells = len(windows) * len(prevalences) * len(mixes)
    seeds = np.random.SeedSequence(cfg["seed"]).spawn(1 + len(ARMS) * cells + len(windows))
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
                f"calibration part {np.where(hit, b_cal[np.maximum(out['index'], 0)], 0.0).mean():.4f}")
    # the bounds of the calibration sets serve every window
    certify_window = online_certifier(grouped_below, grouped_n, *args, **start)
    estimate = estimate_settings(c, delta, eps, [m for m in ESTIMATED if m in methods], b_sel, m_sel,
                                 cal.benign_below, cal.n_benign, pooled_below, pooled_n, start, run.log)

    # rates at the offline FORGE thresholds, averaged over the calibration draws that certify one
    forge_index = offline["forge_simplex"]["index"]
    forge_thresholds = forge_index[forge_index >= 0]

    def at_forge(rate: np.ndarray) -> np.ndarray:
        if len(forge_thresholds) == 0:
            return np.full(rate.shape[:-1], np.nan)
        return rate[..., forge_thresholds].mean(-1)

    forge_cal = {"benign_dismissed": float(at_forge(b_cal)),
                 "miss": {f: float(v) for f, v in zip(families, at_forge(m_cal_family))}}
    run.log(f"calibration part at the offline FORGE thresholds, benign share dismissed "
            f"{forge_cal['benign_dismissed']:.4f}, miss rate "
            + ", ".join(f"{f} {v:.4f}" for f, v in forge_cal["miss"].items()))

    rows, natural_rows, values, natural_values, skipped, estimate_rows = [], [], {}, {}, [], []
    shape = (len(windows), len(prevalences), len(mixes), draws)
    online_index = {arm: np.full(shape, -1, dtype=np.int16) for arm in ARMS}
    online_floor = {arm: np.zeros(shape, dtype=np.float32) for arm in ARMS}
    estimated_index = {arm: {m: np.full(shape, -1, dtype=np.int16) for m in estimate["methods"]} for arm in ARMS}
    estimated_pi = {arm: {m: np.zeros(shape, dtype=np.float32) for m in estimate["methods"]} for arm in ARMS}
    natural_index = np.full((len(windows), draws), -1, dtype=np.int16)
    natural_floor = np.zeros((len(windows), draws), dtype=np.float32)
    natural_estimated = {m: np.full((len(windows), draws), -1, dtype=np.int16) for m in estimate["methods"]}
    natural_pi = {m: np.zeros((len(windows), draws), dtype=np.float32) for m in estimate["methods"]}
    offset = {arm: 1 + a * cells for a, arm in enumerate(ARMS)}
    # the flows of the families left out in each window, which the window's prevalence leaves out
    left_in_window = {f: np.bincount(window[(window >= 0) & (cat_codes == names.index(f))], minlength=len(windows))
                      for f in fam["left"]}
    window_rows, without_attacks = [], []
    for i, (win, g) in enumerate(zip(windows, win_table.itertuples())):
        # a window without attacks has no mix of its own, its reference and s2 windows come at the calibration mix
        with np.errstate(divide="ignore", invalid="ignore"):
            b_w, m_w_family, _, own = judge.rates(win)
        test_attacks = win["sizes"][1:]
        has_attacks = bool(test_attacks.sum() > 0)
        populations = {"reference": (b_cal, m_cal_family, n_attack_cal), "s2": (b_w, m_cal_family, n_attack_cal)}
        if has_attacks:
            populations["real"] = (b_w, m_w_family, test_attacks)
        else:
            own = np.zeros(len(families))
            without_attacks.append(i)
        benign_w = int(win["sizes"][0])
        cell_base = {"window": i, "start": g.start}
        # the threshold safe for every covered mix needs the attack distributions of every family, which a real window
        # lacks, it holds those of its own attacks only
        every = {arm: judge.every_mix_threshold(pi_max, m_family, b, groups, shares_cal, eps)
                 for arm, (b, m_family, _) in populations.items() if arm != "real"}
        for j, pi in enumerate(prevalences):
            full = int(round(benign_w / (1 - pi)))
            for k, mix in enumerate(mixes):
                if mix == "window" and not has_attacks:
                    continue
                shares = judge.mix_shares(mix, own, shares_cal, families)
                claim = in_claim(shares, shares_cal, groups)
                for arm, (b, m_family, attacks) in populations.items():
                    n_w, subsampled = full, False
                    # a real window stands for the window thinned to its prevalence and mix, so it must hold at least
                    # the attacks of each family that the thinned window takes on average. one that does not keeps all
                    # of them and has its benign flows subsampled, or is left out
                    if arm == "real" and np.any(attacks < full * pi * shares):
                        need = shares > 0
                        if not dep["subsample_benign"] or (attacks[need] == 0).any():
                            skipped.append([i, pi, mix])
                            continue
                        n_w, subsampled = int(np.floor((attacks[need] / (pi * shares[need])).min())), True
                    rng = np.random.default_rng(seeds[offset[arm] + (i * len(prevalences) + j) * len(mixes) + k])
                    drawn = draw_counts(rng, n_w, pi, shares, b, m_family, draws)
                    attack_below = drawn.family_below.sum(1)
                    window_below = drawn.benign_below + attack_below
                    online = certify_window(window_below, np.full(draws, n_w))
                    online_index[arm][i, j, k], online_floor[arm][i, j, k] = online["index"], online["floor"]
                    per_window, estimated = estimated_methods(estimate, window_below, np.full(draws, n_w), pi)
                    for name, out in estimated.items():
                        estimated_index[arm][name][i, j, k] = out["index"]
                        estimated_pi[arm][name][i, j, k] = out["prevalence"]
                    if estimated:
                        estimate_rows.append({"arm": arm, **cell_base, "prevalence": pi, "mix": mix,
                                              "natural": False, "window_sessions": n_w, "draws": draws,
                                              **estimate_summary(estimated, pi)})
                    m = shares @ m_family
                    j_every = every.get(arm, -1)
                    every_share = (np.nan if arm not in every else
                                   float(pi * m[j_every] + (1 - pi) * b[j_every]) if j_every >= 0 else 0.0)
                    rows += judge.evaluate(methods, offline, {"forge_online": online, **per_window}, pi, m, b,
                                           attack_below, window_below, eps, values,
                                           {"arm": arm, **cell_base, "prevalence": pi, "mix": mix,
                                            "window_sessions": n_w, "benign_subsampled": subsampled,
                                            "in_claim": claim,
                                            "few_attacks": bool((attacks[shares > 0] < min_attacks).any()),
                                            "draws": draws, **judge.best_thresholds(pi, pi_max, m, b, eps),
                                            "best_every_mix_index": j_every,
                                            "best_every_mix_dismissed": every_share})
        own_claim = in_claim(own, shares_cal, groups) if has_attacks else True
        natural = float(test_attacks.sum() / win["sizes"].sum())
        if has_attacks:
            # the real window at its own prevalence, all its flows, no thinning
            n_w = int(win["sizes"].sum())
            rng = np.random.default_rng(seeds[1 + len(ARMS) * cells + i])
            drawn = draw_counts(rng, n_w, natural, own, b_w, m_w_family, draws)
            attack_below = drawn.family_below.sum(1)
            window_below = drawn.benign_below + attack_below
            online = certify_window(window_below, np.full(draws, n_w))
            natural_index[i], natural_floor[i] = online["index"], online["floor"]
            per_window, estimated = estimated_methods(estimate, window_below, np.full(draws, n_w), natural)
            for name, out in estimated.items():
                natural_estimated[name][i], natural_pi[name][i] = out["index"], out["prevalence"]
            if estimated:
                estimate_rows.append({"arm": "real", **cell_base, "prevalence": natural, "mix": "window",
                                      "natural": True, "window_sessions": n_w, "draws": draws,
                                      **estimate_summary(estimated, natural)})
            m = own @ m_w_family
            natural_rows += judge.evaluate(methods, offline, {"forge_online": online, **per_window}, natural, m, b_w,
                                           attack_below, window_below, eps, natural_values,
                                           {"arm": "real", **cell_base, "prevalence": natural, "mix": "window",
                                            "window_sessions": n_w, "in_claim": own_claim,
                                            "up_to_pi_max": natural <= pi_max,
                                            "few_attacks": bool((test_attacks[own > 0] < min_attacks).any()),
                                            "draws": draws, **judge.best_thresholds(natural, pi_max, m, b_w, eps)})
        # a family absent from a window has no miss rate there
        miss = np.where(test_attacks > 0, at_forge(m_w_family), np.nan)
        left_here = {f: int(counts[i]) for f, counts in left_in_window.items()}
        window_rows.append({
            **cell_base, "end": g.end, "in_claim": own_claim, "has_attacks": has_attacks,
            "flows": int(win["sizes"].sum()), "benign": benign_w,
            **{f: int(n) for f, n in zip(families, test_attacks)}, **left_here,
            "natural_prevalence": natural,
            "prevalence_with_left_out": float((test_attacks.sum() + sum(left_here.values()))
                                              / (win["sizes"].sum() + sum(left_here.values()))),
            "auc": float(test_auc[i]), "forge_benign_dismissed": float(at_forge(b_w)),
            **{f"forge_miss_{f}": float(v) for f, v in zip(families, miss)},
        })
    if skipped:
        run.log(f"{len(skipped)} real windows, prevalences and mixes left out, their windows lack the attacks of a "
                f"family of the mix")
    if without_attacks:
        run.log(f"{len(without_attacks)} windows hold no attack and have reference and S2 windows at the calibration "
                "mix only")
    claims = pd.DataFrame(rows)
    run.save_table(claims, "claims")
    natural_table = pd.DataFrame(natural_rows)
    run.save_table(natural_table, "natural")
    if estimate_rows:
        estimates = pd.DataFrame(estimate_rows)
        run.save_table(estimates, "estimates")
        log_estimates(estimates[~estimates["natural"]], estimate, dep["main_prevalence"], run.log)
    window_table = pd.DataFrame(window_rows)
    run.save_table(window_table, "windows")
    for w in window_table.itertuples():
        present = [f for f in families if getattr(w, f) > 0]
        rates_of = (", ".join(f"{f} {getattr(w, 'forge_miss_' + f):.4f} over {getattr(w, f)} attacks against "
                              f"{forge_cal['miss'][f]:.4f}" for f in present) if present else "no attack")
        run.log(f"window {w.start:%Y-%m-%d %H:%M}, flows {w.flows}, benign {w.benign}, attacks "
                + _counts("", [(f, getattr(w, f)) for f in present]) + ", left out "
                + _counts("", [(f, getattr(w, f)) for f in fam["left"]])
                + f", prevalence {w.natural_prevalence:.5f}, AUC {w.auc:.4f}, at the offline FORGE thresholds benign "
                f"share dismissed {w.forge_benign_dismissed:.4f} against {forge_cal['benign_dismissed']:.4f} in "
                f"calibration, miss rate {rates_of}" + ("" if w.in_claim else ", outside the FORGE claim"))

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
            "window": int(top["window"]), "start": f"{top['start']:%Y-%m-%d %H:%M}",
            "prevalence": float(top["prevalence"]), "rate": float(top["rate"]), "low": float(top["low"]),
            "high": float(top["high"])}
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
                    worst = (f"at most {top['rate']:.4f} in one window up to pi_max, {top['start']} at prevalence "
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
    # one line per window and arm with the window's own mix at the main prevalence, and one per real window at its
    # own prevalence
    at_main = claims[(claims["prevalence"] == main) & (claims["mix"] == "window")]
    for (i, arm), g in at_main.groupby(["window", "arm"], sort=True):
        run.log(f"window {g['start'].iloc[0]:%Y-%m-%d %H:%M}, {WINDOW_WORDS[arm]} at prevalence {main}, FOR above "
                "epsilon in " + ", ".join(f"{r.method} {r.rate:.4f}" for r in g.itertuples())
                + ("" if g["in_claim"].iloc[0] else ", outside the FORGE claim"))
    for i, g in natural_table.groupby("window", sort=True):
        run.log(f"window {g['start'].iloc[0]:%Y-%m-%d %H:%M}, real window at its own prevalence "
                f"{g['prevalence'].iloc[0]:.5f}, FOR above epsilon in "
                + ", ".join(f"{r.method} {r.rate:.4f}" for r in g.itertuples())
                + ", mean share dismissed " + ", ".join(f"{r.method} {r.dismissed_mean:.4f}" for r in g.itertuples())
                + ("" if g["up_to_pi_max"].iloc[0] else ", above pi_max"))

    # the families outside every claim, the share of their scored flows at or below the thresholds of every method,
    # the per window methods read at the real windows at their own prevalence
    natural_rows_of = [i for i in range(len(windows)) if i not in without_attacks]
    natural_per_method = {"forge_online": natural_index[natural_rows_of],
                          **{m: natural_estimated[m][natural_rows_of] for m in estimate["methods"]}}
    unseen_summary = {}
    for f, u in unseen.items():
        below = counts_below(u["score"], grid)
        table = unseen_table(f, below, len(u["score"]), offline, natural_per_method, methods)
        run.save_table(table, unseen_name(f))
        by_role = np.bincount(role[u["rows"]], minlength=len(ROLES))
        unseen_summary[f] = {"table": unseen_name(f), "flows": int(len(u["score"])),
                             "flows_by_role": {r: int(n) for r, n in zip(ROLES, by_role) if n},
                             "dismissed_mean": {r.method: float(r.dismissed_mean) for r in table.itertuples()}}
        run.log(f"{f}, {len(u['score'])} flows outside every claim, "
                + ", ".join(f"{r} {n}" for r, n in unseen_summary[f]["flows_by_role"].items())
                + ", share dismissed at the thresholds of "
                + ", ".join(f"{r.method} {r.dismissed_mean:.4f}" for r in table.itertuples())
                + f", the per window methods over {len(natural_rows_of)} real windows at their own prevalence")

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
                    window_sizes=stacked("sizes"),
                    **{f"{unseen_name(f)}_below": counts_below(u["score"], grid) for f, u in unseen.items()})
    run.save_arrays("draws", prevalences=np.array(prevalences), mixes=np.array(mixes), arms=np.array(ARMS),
                    forge_floor=offline["forge_simplex"]["floor"], forge_pooled_floor=offline["forge_pooled"]["floor"],
                    **{f"{m}_index": out["index"] for m, out in offline.items()},
                    online_index=np.stack([online_index[arm] for arm in ARMS]),
                    online_floor=np.stack([online_floor[arm] for arm in ARMS]),
                    natural_online_index=natural_index, natural_online_floor=natural_floor,
                    **{f"{PER_WINDOW[m]}_index": np.stack([estimated_index[arm][m] for arm in ARMS])
                       for m in estimate["methods"]},
                    **{f"{PER_WINDOW[m]}_prevalence": np.stack([estimated_pi[arm][m] for arm in ARMS])
                       for m in estimate["methods"]},
                    **{f"natural_{PER_WINDOW[m]}_index": natural_estimated[m] for m in estimate["methods"]},
                    **{f"natural_{PER_WINDOW[m]}_prevalence": natural_pi[m] for m in estimate["methods"]},
                    counts=np.column_stack([cal.n_benign, cal.n_family]))
    run.save_json({"encoder": enc.to_dict(), "features": features, "params": sc["params"], "rounds": sc["rounds"],
                   "early_stopping": sc["early_stopping"], "best_iteration": booster.best_iteration,
                   "selection_loss": sel_loss, "train_flows": int(len(train)),
                   "train_attacks": int(label[train].sum()), "selection_auc": sel_auc, "calibration_auc": cal_auc},
                  "scorer")
    run.save_json({
        "dataset": d["name"], "processed_version": d["processed_version"], "fingerprint": manifest["fingerprint"],
        "sample_every": manifest["sample_every"], "window_length": length,
        "roles": {r.role: {"start": f"{r.start:%Y-%m-%d %H:%M}", "end": f"{r.end:%Y-%m-%d %H:%M}",
                           "windows": int(r.windows), "flows": int(r.flows), "benign": int(r.benign),
                           "attacks": int(r.attacks)} for r in roles.itertuples()},
        "left_out": left_out, "families_left_out": fam["left"], "unseen": unseen_summary,
        "epsilon": eps, "delta": delta, "delta_benign": part_b, "pi_max": pi_max, **start,
        "grid_size": c["grid_size"], "grid_tail": c.get("grid_tail", []), "grid_thresholds": len(grid),
        "family_floor": c["family_floor"], "family_minimum": minimum, "families": merged,
        "calibration_prevalence": pi_cal, "calibration_mix": cc["mix"], "calibration_sessions": n_cal,
        "calibration_attacks": dict(zip(families, n_attack_cal.tolist())),
        "calibration_expected_attacks": dict(zip(families, expected.tolist())),
        "fixed_index": fixed, "beta": beta, "prevalences": prevalences, "main_prevalence": main, "arms": ARMS,
        "mixes": mixes, "subsample_benign": dep["subsample_benign"], "skipped": skipped, "methods": methods,
        "draws": draws, "windows": len(windows), "empty_windows": empty, "windows_without_attacks": without_attacks,
        "offline_refused": {m: float(np.mean(out["index"] < 0)) for m, out in offline.items()},
        "selection_auc": sel_auc, "calibration_auc": cal_auc, "forge_calibration": forge_cal,
        "min_family_attacks": min_attacks, "largest_violation_up_to_pi_max": largest,
        "estimate": estimate["summary"],
    }, "summary")

    for name, draw in FIGURES.items():
        save_figure(draw(run.dir), run.figures, name)
        run.log(f"saved figure {name}")


def _line(method: str) -> dict:
    return {"color": style.color_for(method), "marker": MARKERS[method], "label": LABELS[method]}


def _for_by_prevalence(run_dir, arm: str) -> plt.Figure:
    # the windows inside the FORGE claim, every window at the calibration mix and the windows whose own mix the claim
    # covers
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


def _dates(values) -> np.ndarray:
    return mdates.date2num(pd.to_datetime(values).to_numpy())


def _window_axis(axes, windows: pd.DataFrame, summary: dict):
    # time on the shared x axis, a labelled tick on every monday and a bare one on every other day, and a light band
    # behind the windows outside the FORGE claim. returns the legend handle of the band
    starts, ends = _dates(windows["start"]), _dates(windows["end"])
    outside = ~windows["in_claim"].to_numpy()
    for ax, letter in zip(axes, "abcdefgh"):
        for s, e in zip(starts[outside], ends[outside]):
            ax.axvspan(s, e, color=style.GRAY, alpha=0.15, linewidth=0)
        layout.panel_letter(ax, letter)
    pad = ends[0] - starts[0]
    axes[-1].set_xlim(starts.min() - pad, ends.max() + pad)
    # a span of a week or more is labelled on its mondays, the first day of a test week, a shorter one on every day
    weekly = ends.max() - starts.min() >= 7
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO) if weekly else mdates.DayLocator())
    axes[-1].xaxis.set_minor_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].set_xlabel(f"start of the window of {summary['window_length']}, "
                        f"{pd.Timestamp(windows['start'].iloc[0]):%Y}")
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
    # without attacks in the reference and S2 rows only, and a last row with the mean share of the reference window
    # flows dismissed and the best threshold up to pi_max. the methods are dodged within a tenth of the window
    step = 0.1 * float(_dates(windows["end"])[0] - _dates(windows["start"])[0])
    for ax, arm in zip(axes, arms):
        for k, method in enumerate(methods):
            g = data[(data["arm"] == arm) & (data["method"] == method)].sort_values("window")
            layout.points(ax, layout.dodge(_dates(g["start"]), k, len(methods), step), g["rate"], g["low"], g["high"],
                          linestyle="none", **_line(method))
        ax.axhline(summary["delta"], color=style.GRAY, linestyle="--", linewidth=1, label=r"$\delta$")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(f"share with FOR > $\\varepsilon$\n{ARM_LABELS[arm]}")
    reference = data[data["arm"] == "reference"]
    for k, method in enumerate(methods):
        g = reference[reference["method"] == method].sort_values("window")
        axes[-1].plot(layout.dodge(_dates(g["start"]), k, len(methods), step), g["dismissed_mean"], linestyle="none",
                      markersize=3.5, **_line(method))
    g = reference[reference["method"] == methods[0]].sort_values("window")
    name, _, label = BEST_LINES[0]
    best = axes[-1].plot(_dates(g["start"]), g[f"{name}_dismissed"], color=style.INK, linestyle="none", marker="_",
                         markersize=6, markeredgewidth=1.2, label=label)[0]
    axes[-1].set_ylim(0, 1)
    axes[-1].yaxis.set_major_formatter(style.percent)
    axes[-1].set_ylabel(f"window flows dismissed\n{ARM_LABELS['reference']}")
    band = _window_axis(axes, windows, summary)
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
    # top, the AUC of the scorer in each window. middle, the miss rate of each family in a window at the offline FORGE
    # thresholds over the same in the calibration part, from the windows with enough attacks of the family. bottom,
    # the benign share dismissed at those thresholds over the same in the calibration part, through the daily cycle.
    # a ratio of 1 is no drift, the offline claim needs both ratios near 1, the online claim the middle one
    x = _dates(windows["start"])
    line = {"markersize": 2.5, "linewidth": 1}
    axes[0].plot(x, windows["auc"], marker="o", color=style.INK, **line)
    calibration = axes[0].axhline(summary["calibration_auc"], color=style.GRAY, linestyle="--", linewidth=1)
    axes[0].set_ylabel("AUC of the scorer")
    ratios = {}
    for f in families:
        enough = windows[f].to_numpy() >= summary["min_family_attacks"]
        # a family without a miss in the calibration part has no ratio, so its rate there is read as one miss over
        # its calibration attacks, the smallest rate the part tells from zero, and the ratio shown is a floor
        base = max(cal["miss"][f], 1 / summary["calibration_attacks"][f])
        ratios[f] = np.where(enough, windows[f"forge_miss_{f}"].to_numpy() / base, np.nan)
    ratios["benign"] = (windows["forge_benign_dismissed"] / cal["benign_dismissed"]).to_numpy()
    # a window with no miss at the thresholds has a ratio of 0, which sits on the lower edge of the log axis
    shown = np.concatenate(list(ratios.values()))
    shown = shown[np.isfinite(shown) & (shown > 0)]
    low = 10.0 ** np.floor(np.log10(min(shown.min(), 1.0))) if len(shown) else 0.1
    high = 10.0 ** np.ceil(np.log10(max(shown.max(), 1.0))) if len(shown) else 10.0
    for k, f in enumerate(families):
        axes[1].plot(x, np.where(ratios[f] == 0, low, ratios[f]), marker=FAMILY_MARKERS[k % len(FAMILY_MARKERS)],
                     linestyle=style.DASHES[k % len(style.DASHES)], color=style.COLORS[k % len(style.COLORS)],
                     label=ugr16.FAMILY_LABELS.get(f, f), **line)
    axes[2].plot(x, np.where(ratios["benign"] == 0, low, ratios["benign"]), marker="o", color=style.INK, **line)
    for ax in axes[1:]:
        ax.axhline(1, color=style.GRAY, linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_ylim(low, high)
    axes[1].set_ylabel("attacks dismissed,\nwindow over calibration")
    axes[2].set_ylabel("benign flows dismissed,\nwindow over calibration")
    band = _window_axis(axes, windows, summary)
    layout.legend_below(fig, axes[1], [ugr16.FAMILY_LABELS.get(f, f) for f in families],
                        [(calibration, "calibration part"), (band, "outside the FORGE claim")])
    return fig


FIGURES = {
    "for_by_prevalence": fig_for_by_prevalence,
    "for_by_prevalence_s2": fig_for_by_prevalence_s2,
    "for_by_prevalence_real": fig_for_by_prevalence_real,
    "violation_by_window": fig_violation_by_window,
    "drift_by_window": fig_drift_by_window,
}
