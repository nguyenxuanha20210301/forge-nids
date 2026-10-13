"""e11, RQ1 on AnoShift, the false omission rate and the dismissed share of each method in deployment windows whose
attacks are thinned to a range of prevalences.

a LightGBM scorer is trained on the train years, stopped early on the selection year, and scores every session of the
selection, calibration and deployment years. the grid holds G benign quantiles of the selection year, and every method
works on counts at or below its thresholds.

the calibration year keeps all its benign sessions and has its attacks thinned to the calibration prevalence, as the
calibration set of an operator would look at a realistic attack rate. each draw is a calibration set of that size drawn
with replacement from the calibration year, so the share of draws in which a method fails estimates the probability
over the calibration draw that its guarantee speaks of, with the calibration year as the population. a deployment
window is one block of a deployment year, the year split into blocks of equal session count. it keeps all its benign
sessions and has its attacks thinned to each target prevalence, and the true rate of a threshold in it follows from
the score distributions of each class and attack type in the whole block. the blocks of AnoShift's deployment years
hold 15% to 99.4% attacks, above pi_max in every block, so only thinned windows are evaluated. each draw also takes one
window of the thinned size from the block, with replacement, on which the online certificate runs and the realized
share of attacks among the dismissed sessions is counted.

each block of the main period also has a reference window, with the benign count, prevalence and mix of the block's
window but the score distributions of the calibration year for each class and attack type, so label shift holds by
construction. its true rate follows from those distributions, and its drawn window comes from them too. a reference
window needs no attacks of the block, so it has every prevalence and mix.

two thresholds of every window serve as yardsticks for the dismissed share. the best one up to pi_max is the loosest
grid threshold whose true FOR in the window stays at or below epsilon at every prevalence up to pi_max, the best a
method that knows the window's distributions but not its prevalence could dismiss. the best one at the prevalence is
the loosest whose true FOR stays at or below epsilon at the window's own prevalence, which knowing the prevalence too
would allow.

what each method claims.
    forge_simplex   Theorem 1, every prevalence up to pi_max and every mix of the attack families, valid while both
                    classes keep their calibration score distributions
    forge_online    Theorem 1b, the same for one window from its unlabeled scores, valid while the attack families keep
                    their calibration score distributions
    ltt_iid         the FOR at the prevalence and the family mix of the calibration set
    conformal_fnr   a miss rate of at most beta on average over the calibration draw, with beta the miss rate of the
                    fixed threshold in the selection year
    fixed_sel       nothing, the loosest threshold whose FOR at the calibration prevalence, estimated on the selection
                    year, stays at or below epsilon

the attack types are the families. a type whose expected number of calibration attacks could not certify, with no miss
at all, a threshold dismissing family_floor of the benign sessions is merged with the others, a rule that sees the
family sizes only. once merged, the FORGE claim covers the calibration mix of attack types only. the attacks of a
window come in more than one mix, the block's own, which an operator would meet, and one type alone, which keeps a
change of mix out of the comparison when types are merged and is a corner of the covered mixes when they are not. the
calibration mix itself would weight the few unknown sessions of a late block up to their calibration share, so their
sampling noise would decide the rate. a block that holds fewer attacks of a type than its thinned window would take
on average has no window at that prevalence and mix.

the offline claim holds while both classes keep their calibration score distributions, the online claim while the
attack types do. AnoShift drifts in the scores of both classes from block to block, in the IID year after calibration
as well, so a real window mixes a change of prevalence with drift. RQ1 is read on the reference windows, and the real
windows show what the drift does to every method. the AUC of each block, its benign share dismissed and its miss rate
per attack type at the offline FORGE thresholds, set against the same in the calibration year, show which assumption
breaks where.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forge import runs
from forge.certify.certificate import (certify, conformal_fnr, family_groups, grid_from, group_rows, ltt_iid,
                                       min_family_size, online_certifier, plug_in)
from forge.data import anoshift, splits
from forge.data.synthetic import draw_counts
from forge.eval import windows as judge
from forge.eval.metrics import auc
from forge.eval.windows import add_counts, best_thresholds, evaluate, rates
from forge.models import boosting
from forge.models.encoding import Encoder
from forge.viz import layout, prevalence, style
from forge.viz.export import save_figure
from forge.viz.prevalence import BEST_LINES

METHODS = ["forge_simplex", "forge_online", "ltt_iid", "conformal_fnr", "fixed_sel"]
LABELS = {
    "forge_simplex": "FORGE offline, Theorem 1",
    "forge_online": "FORGE online, Theorem 1b",
    "ltt_iid": "LTT, i.i.d.",
    "conformal_fnr": "conformal FNR, Şahin and Mert",
    "fixed_sel": "fixed threshold, selection year",
}
MARKERS = {"forge_simplex": "o", "forge_online": "v", "ltt_iid": "^", "conformal_fnr": "p", "fixed_sel": "X"}
ROLES = ["train", "selection", "calibration", "deployment"]
# the mix of attack types in a window, the block's own mix, the mix of the calibration set, or one attack type alone
MIXES = ["window", "calibration", *anoshift.ATTACK_TYPES[1:]]
MIX_LABELS = {"window": "own mix of attack types", "calibration": "calibration mix of attack types",
              **{t: f"{t} attack type only" for t in anoshift.ATTACK_TYPES[1:]}}
FAMILIES = anoshift.ATTACK_TYPES[1:]
FEATURES = [name for name, _, _, _, _ in anoshift.FEATURES]
COLUMNS = ["year", "order", "label", "attack_type", *FEATURES]
# a block's miss rate of an attack type enters the drift figure only from this many sessions of the type on, below it
# a handful of sessions would decide the rate
MIN_TYPE_SESSIONS = 1000
# a real window takes the score distributions of its block, a reference window those of the calibration year
ARMS = ["reference", "real"]


def encoder() -> Encoder:
    # a bin of -1 marks a value outside the bins of AnoShift, which carries no order, so it counts as missing
    kinds = {name: kind for name, _, kind, _, _ in anoshift.FEATURES}
    return Encoder(numeric=[c for c in FEATURES if kinds[c] != "category"],
                   categorical=[c for c in FEATURES if kinds[c] == "category"],
                   missing={c: -1 for c in FEATURES if kinds[c] == "bins"})


def role_years(split: dict) -> dict:
    # every role covers its first to its last year, and the roles follow each other in time without overlap
    years = {role: list(range(split[role][0], split[role][1] + 1)) for role in ROLES}
    for before, after in zip(ROLES, ROLES[1:]):
        if not years[before] or not years[after] or years[before][-1] >= years[after][0]:
            raise ValueError(f"the {before} years {split[before]} must come before the {after} years {split[after]}")
    return years


def load_years(cfg: dict, years: list[int]) -> pd.DataFrame:
    flows = anoshift.load(cfg, columns=COLUMNS, years=years)
    if flows.empty:
        raise ValueError(f"no session of {cfg['dataset']['name']} in the years {years}")
    return flows


def class_counts(score: np.ndarray, code: np.ndarray, grid: np.ndarray) -> dict:
    # counts at or below each threshold of the benign sessions, code 0, and of each attack type, codes 1 and up
    return judge.class_counts(score, code, grid, len(FAMILIES))


def mix_shares(mix: str, own: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    # share of each attack type among the attacks of a window
    return judge.mix_shares(mix, own, calibration, FAMILIES)


def score_years(cfg: dict, booster, enc: Encoder, years: list[int]):
    # one year at a time, so only the scores and labels of the sessions stay in memory
    per_year = cfg["dataset"]["blocks_per_year"]
    for year in years:
        flows = load_years(cfg, [year])
        yield year, {
            "score": boosting.score(booster, enc.transform(flows)),
            "label": flows["label"].to_numpy(np.int8),
            "attack_type": flows["attack_type"].cat.codes.to_numpy(np.int8),
            "order": flows["order"].to_numpy(np.int32),
            "block": splits.order_blocks(np.zeros(len(flows), dtype=np.int8), per_year),
        }


def _type_range(windows: pd.DataFrame, family: str) -> str:
    # the miss rates of one attack type over the blocks that hold enough of its sessions to measure one
    kept = windows.loc[windows[family] >= MIN_TYPE_SESSIONS, f"forge_miss_{family}"]
    if kept.empty:
        return f"{family} in no block with {MIN_TYPE_SESSIONS} sessions"
    return f"{family} {kept.min():.4f} to {kept.max():.4f} over {len(kept)} blocks"


def _type_rate(window: pd.Series, family: str) -> str:
    # the miss rate of one attack type in one block, with the sessions it rests on
    if window[family] == 0:
        return f"{family} absent"
    return f"{family} {window[f'forge_miss_{family}']:.4f} over {window[family]} sessions"


def run(cfg: dict, run: runs.Run) -> None:
    d, sc, c, dep, sim = cfg["dataset"], cfg["scorer"], cfg["certify"], cfg["deployment"], cfg["simulation"]
    eps, delta, pi_max, part = c["epsilon"], c["delta"], c["pi_max"], c["delta_benign"]
    start = {"start_misses": c["start_misses"], "start_slack": c["start_slack"]}
    pi_cal, prevalences, mixes = cfg["calibration"]["prevalence"], dep["prevalences"], dep["mixes"]
    methods, draws = sim["methods"], sim["draws"]
    if dep["main_prevalence"] not in prevalences:
        raise ValueError(f"main prevalence {dep['main_prevalence']} is not among the prevalences {prevalences}")
    if set(mixes) - set(MIXES) or len(set(mixes)) != len(mixes):
        raise ValueError(f"mixes are {mixes}, expected some of {', '.join(MIXES)}, each once")
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown methods {', '.join(sorted(unknown))}")
    years = role_years(cfg["split"])
    periods = list(dict.fromkeys(anoshift.period(y) for y in years["deployment"]))
    if dep["main_period"] not in periods:
        raise ValueError(f"main period {dep['main_period']} is not among the periods of the deployment years, "
                         f"{', '.join(periods)}")
    manifest = anoshift.read_manifest(cfg)
    run.log(f"{d['name']} {d['processed_version']}, fingerprint {manifest['fingerprint']}, years "
            + ", ".join(f"{role} {years[role][0]} to {years[role][-1]}" for role in ROLES))

    # the scorer sees the train years, and the selection year only to stop
    train = load_years(cfg, years["train"])
    selection = load_years(cfg, years["selection"])
    enc = encoder().fit(train)
    x_selection = enc.transform(selection)
    booster = boosting.fit(enc.transform(train), train["label"].to_numpy(), x_selection,
                           selection["label"].to_numpy(), sc["params"], sc["rounds"], sc["early_stopping"],
                           cfg["seed"], enc.columns, enc.categorical_index)
    n_train, train_attacks = len(train), int(train["label"].sum())
    del train
    sel_score = boosting.score(booster, x_selection)
    del x_selection
    sel_code = selection["attack_type"].cat.codes.to_numpy(np.int8)
    sel_auc = auc(sel_score, selection["label"].to_numpy())
    sel_loss = float(next(iter(booster.best_score["selection"].values())))
    run.log(f"trained LightGBM on {n_train} sessions with {train_attacks} attacks, best round "
            f"{booster.best_iteration}, selection loss {sel_loss:.5f}, selection AUC {sel_auc:.5f}")
    (run.results / "models").mkdir()
    booster.save_model(str(run.results / "models" / "scorer.txt"), num_iteration=booster.best_iteration)

    grid = grid_from(sel_score[sel_code == 0], c["grid_size"])
    sel_counts = class_counts(sel_score, sel_code, grid)
    run.log(f"grid of {len(grid)} distinct thresholds from {c['grid_size']} benign quantiles of the selection year")
    kept = {"score": [sel_score], "year": [selection["year"].to_numpy(np.int16)],
            "order": [selection["order"].to_numpy(np.int32)], "label": [selection["label"].to_numpy(np.int8)],
            "attack_type": [sel_code]}
    del selection

    # the calibration years as one population, the deployment years one block at a time
    cal_counts, cal_labels, cal_scores, windows = None, [], [], []
    for role in ["calibration", "deployment"]:
        for year, s in score_years(cfg, booster, enc, years[role]):
            for key in kept:
                kept[key].append(np.full(len(s["score"]), year, dtype=np.int16) if key == "year" else s[key])
            if role == "calibration":
                counts = class_counts(s["score"], s["attack_type"], grid)
                cal_counts = counts if cal_counts is None else add_counts(cal_counts, counts)
                cal_scores.append(s["score"])
                cal_labels.append(s["label"])
                continue
            edges = np.searchsorted(s["block"], np.arange(d["blocks_per_year"] + 1))
            for block, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                if hi == lo:
                    continue
                counts = class_counts(s["score"][lo:hi], s["attack_type"][lo:hi], grid)
                if counts["sizes"][0] == 0 or counts["sizes"][1:].sum() == 0:
                    raise ValueError(f"block {block} of {year} lacks benign sessions or attacks, it cannot be thinned")
                windows.append({"year": year, "block": block, "auc": auc(s["score"][lo:hi], s["label"][lo:hi]),
                                **counts})
    cal_auc = auc(np.concatenate(cal_scores), np.concatenate(cal_labels))
    del cal_scores, cal_labels
    # attack_type holds codes into attack_types
    run.save_arrays("scores", **{key: np.concatenate(parts) for key, parts in kept.items()},
                    attack_types=np.array(anoshift.ATTACK_TYPES))
    del kept
    run.log(f"scored the selection, calibration and deployment years, calibration AUC {cal_auc:.5f}, "
            f"{len(windows)} deployment windows")

    # the two baselines tuned on the selection year at the calibration prevalence
    b_sel, _, m_sel, _ = rates(sel_counts)
    fixed = int(plug_in(sel_counts["benign_below"], sel_counts["sizes"][0], sel_counts["family_below"].sum(0),
                        sel_counts["sizes"][1:].sum(), eps, pi_cal))
    beta = float(m_sel[fixed]) if fixed >= 0 else 0.0
    run.log(f"fixed threshold dismisses benign share {b_sel[fixed] if fixed >= 0 else 0.0:.4f} of the selection year "
            f"with miss rate {beta:.5f}, which conformal FNR takes as beta")

    # calibration sets at the calibration prevalence, with all benign sessions of the calibration years
    b_cal, m_cal_family, m_cal, shares_cal = rates(cal_counts)
    n_cal = int(round(cal_counts["sizes"][0] / (1 - pi_cal)))
    expected = n_cal * pi_cal * shares_cal
    minimum = min_family_size(eps, pi_max, delta - part, c["family_floor"])
    groups = family_groups(expected, minimum)
    families = [[f for f, g in zip(FAMILIES, groups) if g == k] for k in range(groups.max() + 1)]
    run.log(f"calibration sets of {n_cal} sessions at prevalence {pi_cal}, expected attacks per type "
            + ", ".join(f"{f} {e:.0f}" for f, e in zip(FAMILIES, expected))
            + f", a type needs {minimum} to stand alone, families " + ", ".join("+".join(f) for f in families))

    # one stream for the calibration sets, one for each window, prevalence and mix, then one for each reference window,
    # prevalence and mix in the same layout, whether its block has a reference window or not
    cells = len(windows) * len(prevalences) * len(mixes)
    seeds = np.random.SeedSequence(cfg["seed"]).spawn(1 + 2 * cells)
    cal = draw_counts(np.random.default_rng(seeds[0]), n_cal, pi_cal, shares_cal, b_cal, m_cal_family, draws)
    grouped_below, grouped_n = group_rows(cal.family_below, cal.n_family, groups)
    pooled_below, pooled_n = cal.family_below.sum(1), cal.n_family.sum(1)
    no_floor = np.full(draws, np.nan)
    offline = {
        "forge_simplex": certify(cal.benign_below, cal.n_benign, grouped_below, grouped_n, eps, delta, pi_max, part,
                                 **start),
        "ltt_iid": {"index": ltt_iid(cal.benign_below, pooled_below, eps, delta)["index"], "floor": no_floor},
        "conformal_fnr": {"index": conformal_fnr(pooled_below, pooled_n, beta), "floor": no_floor},
        "fixed_sel": {"index": np.full(draws, fixed), "floor": no_floor},
    }
    for method, out in offline.items():
        hit = out["index"] >= 0
        run.log(f"{method}, refused in {np.mean(~hit):.3f} of calibration sets, mean benign share dismissed in the "
                f"calibration year {np.where(hit, b_cal[np.maximum(out['index'], 0)], 0.0).mean():.4f}")
    # the bounds of the calibration sets serve every window
    certify_window = online_certifier(grouped_below, grouped_n, eps, delta, pi_max, part, **start)

    # rates at the offline FORGE thresholds, averaged over the calibration draws that certify one
    forge_index = offline["forge_simplex"]["index"]
    forge_thresholds = forge_index[forge_index >= 0]

    def at_forge(rate: np.ndarray) -> np.ndarray:
        if len(forge_thresholds) == 0:
            return np.full(rate.shape[:-1], np.nan)
        return rate[..., forge_thresholds].mean(-1)

    forge_cal = {"benign_dismissed": float(at_forge(b_cal)),
                 "miss": {f: float(v) for f, v in zip(FAMILIES, at_forge(m_cal_family))}}
    run.log(f"calibration year at the offline FORGE thresholds, benign share dismissed "
            f"{forge_cal['benign_dismissed']:.4f}, miss rate "
            + ", ".join(f"{f} {v:.4f}" for f, v in forge_cal["miss"].items()))

    rows, values, skipped = [], {}, []
    shape = (len(windows), len(prevalences), len(mixes), draws)
    online_index = {arm: np.full(shape, -1, dtype=np.int16) for arm in ARMS}
    online_floor = {arm: np.zeros(shape, dtype=np.float32) for arm in ARMS}
    offset = {"real": 1, "reference": 1 + cells}
    for i, win in enumerate(windows):
        b_w, m_w_family, _, shares_w = rates(win)
        period = anoshift.period(win["year"])
        position = win["year"] + (win["block"] + 0.5) / d["blocks_per_year"]
        # the score distributions each window is drawn from and evaluated on
        population = {"real": (b_w, m_w_family)}
        if period == dep["main_period"]:
            population["reference"] = (b_cal, m_cal_family)
        for j, pi in enumerate(prevalences):
            n_w = int(round(win["sizes"][0] / (1 - pi)))
            for k, mix in enumerate(mixes):
                shares = mix_shares(mix, shares_w, shares_cal)
                for arm, (b, m_family) in population.items():
                    # a real window stands for the block thinned to its prevalence and mix, so the block must hold at
                    # least the attacks of each type that the window takes on average. a rare type of the calibration
                    # mix may lack them in a block, whose few sessions of that type would otherwise decide the rate
                    if arm == "real" and np.any(win["sizes"][1:] < n_w * pi * shares):
                        skipped.append([win["year"], win["block"], pi, mix])
                        continue
                    rng = np.random.default_rng(seeds[offset[arm] + (i * len(prevalences) + j) * len(mixes) + k])
                    drawn = draw_counts(rng, n_w, pi, shares, b, m_family, draws)
                    attack_below = drawn.family_below.sum(1)
                    window_below = drawn.benign_below + attack_below
                    online = certify_window(window_below, np.full(draws, n_w))
                    online_index[arm][i, j, k], online_floor[arm][i, j, k] = online["index"], online["floor"]
                    m = shares @ m_family
                    rows += evaluate(methods, offline, online, pi, m, b, attack_below, window_below, eps, values,
                                     {"arm": arm, "window": i, "year": win["year"], "block": win["block"],
                                      "period": period, "position": position, "prevalence": pi, "mix": mix,
                                      "window_sessions": n_w, "draws": draws,
                                      **best_thresholds(pi, pi_max, m, b, eps)})
    for mix in mixes:
        left = [s for s in skipped if s[-1] == mix]
        if left:
            run.log(f"{mix} mix, {len(left)} of {len(windows) * len(prevalences)} windows and prevalences left out, "
                    f"their blocks lack the attacks of a type that the thinned window would take")
    claims = pd.DataFrame(rows)
    run.save_table(claims, "claims")

    window_rows = []
    for i, w in enumerate(windows):
        b_w, m_w_family, _, _ = rates(w)
        # a type absent from the block has no miss rate
        miss = np.where(w["sizes"][1:] > 0, at_forge(m_w_family), np.nan)
        window_rows.append({
            "window": i, "year": w["year"], "block": w["block"], "period": anoshift.period(w["year"]),
            "position": w["year"] + (w["block"] + 0.5) / d["blocks_per_year"], "sessions": int(w["sizes"].sum()),
            "benign": int(w["sizes"][0]), **{f: int(n) for f, n in zip(FAMILIES, w["sizes"][1:])},
            "attack_share": float(w["sizes"][1:].sum() / w["sizes"].sum()), "auc": float(w["auc"]),
            "forge_benign_dismissed": float(at_forge(b_w)),
            **{f"forge_miss_{f}": float(v) for f, v in zip(FAMILIES, miss)},
        })
    window_table = pd.DataFrame(window_rows)
    run.save_table(window_table, "windows")
    for period, g in window_table.groupby("period", sort=False):
        run.log(f"{period} windows {len(g)}, AUC {g['auc'].min():.4f} to {g['auc'].max():.4f}, at the offline FORGE "
                f"thresholds benign share dismissed {g['forge_benign_dismissed'].min():.4f} to "
                f"{g['forge_benign_dismissed'].max():.4f}, miss rate " + ", ".join(_type_range(g, f) for f in FAMILIES))
    for _, w in window_table[window_table["period"] == dep["main_period"]].iterrows():
        run.log(f"{w['year']} block {w['block']}, AUC {w['auc']:.4f}, attack share {w['attack_share']:.4f}, at the "
                f"offline FORGE thresholds benign share dismissed {w['forge_benign_dismissed']:.4f}, miss rate "
                + ", ".join(_type_rate(w, f) for f in FAMILIES))

    # pooled over the windows of each arm within each period and over all deployment years, the FOR quantiles over the
    # certified draw and window pairs, the violation share over all pairs with its range over the windows
    summary_rows = []
    for (arm, method, pi, mix), cell in claims.groupby(["arm", "method", "prevalence", "mix"], sort=False):
        for name in ["all", *dict.fromkeys(cell["period"])]:
            g = cell if name == "all" else cell[cell["period"] == name]
            certified = np.concatenate([values[(arm, method, pi, mix, w)] for w in g["window"]])
            summary_rows.append({"arm": arm, "method": method, "prevalence": pi, "mix": mix, "period": name,
                                 **judge.pooled(g, certified, draws)})
    by_prevalence = pd.DataFrame(summary_rows)
    run.save_table(by_prevalence, "by_prevalence")

    # the claim of each FORGE method covers prevalences up to pi_max, at the calibration mix once rare types are merged,
    # and the largest window violation share there, within each arm and period, is the number to set against delta
    covered = claims[claims["prevalence"] <= pi_max]
    largest = {}
    for (arm, method, period, mix), g in covered.groupby(["arm", "method", "period", "mix"], sort=False):
        top = g.loc[g["rate"].idxmax()]
        largest.setdefault(arm, {}).setdefault(method, {}).setdefault(period, {})[mix] = {
            "year": int(top["year"]), "block": int(top["block"]), "prevalence": float(top["prevalence"]),
            "rate": float(top["rate"]), "low": float(top["low"]), "high": float(top["high"])}
    main = dep["main_prevalence"]
    # the reference windows first, RQ1 is read on them, and only the main period has them
    for arm in ARMS:
        for period in periods:
            name = f"{period} windows" if arm == "real" else f"{period} reference windows"
            for mix in mixes:
                same = by_prevalence[(by_prevalence["arm"] == arm) & (by_prevalence["period"] == period)
                                     & (by_prevalence["mix"] == mix)]
                for method in methods:
                    g = same[same["method"] == method].sort_values("prevalence")
                    if g.empty:
                        continue
                    top = largest.get(arm, {}).get(method, {}).get(period, {}).get(mix)
                    if top is None:
                        worst = "no window up to pi_max"
                    elif top["rate"] == 0:
                        worst = "none in any window up to pi_max"
                    else:
                        worst = (f"at most {top['rate']:.4f} in one window up to pi_max, {top['year']} block "
                                 f"{top['block']} at prevalence {top['prevalence']}")
                    dismissed = g.loc[g["prevalence"] == main, "dismissed_mean"]
                    run.log(f"{method}, {name}, {mix} mix, FOR above epsilon in "
                            + ", ".join(f"{r:.4f} at {pi}" for pi, r in zip(g["prevalence"], g["violation_rate"]))
                            + f" of draw and window pairs, {worst}, refused {g['refused'].mean():.3f}, mean share of "
                            f"window sessions dismissed {dismissed.iloc[0] if len(dismissed) else np.nan:.4f} at "
                            f"prevalence {main}")
                # the yardsticks follow from the windows alone, so the rows of any method hold them
                g = same[same["method"] == methods[0]].sort_values("prevalence")
                if g.empty:
                    continue
                best = g.loc[g["prevalence"] == main, "best_up_to_pi_max_dismissed"]
                run.log(f"best thresholds, {name}, {mix} mix, mean share of window sessions dismissed by the loosest "
                        f"threshold with FOR at or below epsilon up to pi_max "
                        f"{best.iloc[0] if len(best) else np.nan:.4f} at prevalence {main}, by the loosest at the "
                        f"window's prevalence "
                        + ", ".join(f"{s:.4f} at {pi}" for pi, s in zip(g["prevalence"],
                                                                       g["best_at_prevalence_dismissed"])))

    grid_table = pd.DataFrame({"index": np.arange(len(grid)), "threshold": grid, "selection_benign": b_sel,
                               "selection_attack": m_sel, "calibration_benign": b_cal, "calibration_attack": m_cal})
    for f, m in zip(FAMILIES, m_cal_family):
        grid_table[f"calibration_{f}"] = m
    run.save_table(grid_table, "grid")

    def stacked(sets, key):
        return np.stack([s[key] for s in sets])

    run.save_arrays("histograms", grid=grid, families=np.array(FAMILIES), groups=groups,
                    selection_benign_below=sel_counts["benign_below"],
                    selection_family_below=sel_counts["family_below"], selection_sizes=sel_counts["sizes"],
                    calibration_benign_below=cal_counts["benign_below"],
                    calibration_family_below=cal_counts["family_below"], calibration_sizes=cal_counts["sizes"],
                    window_benign_below=stacked(windows, "benign_below"),
                    window_family_below=stacked(windows, "family_below"), window_sizes=stacked(windows, "sizes"))
    run.save_arrays("draws", prevalences=np.array(prevalences), mixes=np.array(mixes),
                    forge_floor=offline["forge_simplex"]["floor"],
                    **{f"{m}_index": out["index"] for m, out in offline.items()},
                    online_index=online_index["real"], online_floor=online_floor["real"],
                    reference_online_index=online_index["reference"],
                    reference_online_floor=online_floor["reference"],
                    counts=np.column_stack([cal.n_benign, cal.n_family]))
    run.save_json({"encoder": enc.to_dict(), "params": sc["params"], "rounds": sc["rounds"],
                   "early_stopping": sc["early_stopping"], "best_iteration": booster.best_iteration,
                   "selection_loss": sel_loss,
                   "train_sessions": n_train, "train_attacks": train_attacks, "selection_auc": sel_auc,
                   "calibration_auc": cal_auc}, "scorer")
    run.save_json({
        "dataset": d["name"], "processed_version": d["processed_version"], "fingerprint": manifest["fingerprint"],
        "years": years, "periods": {p: list(s) for p, s in anoshift.PERIODS.items()},
        "blocks_per_year": d["blocks_per_year"], "epsilon": eps, "delta": delta, "delta_benign": part,
        "pi_max": pi_max, **start, "grid_size": c["grid_size"], "grid_thresholds": len(grid),
        "family_floor": c["family_floor"], "family_minimum": minimum, "families": families,
        "calibration_prevalence": pi_cal, "calibration_sessions": n_cal,
        "calibration_expected_attacks": dict(zip(FAMILIES, expected.tolist())),
        "fixed_index": fixed, "beta": beta, "prevalences": prevalences, "main_prevalence": main,
        "deployment_periods": periods, "main_period": dep["main_period"], "arms": ARMS, "mixes": mixes,
        "skipped": skipped, "methods": methods, "draws": draws, "windows": len(windows),
        "offline_refused": {m: float(np.mean(out["index"] < 0)) for m, out in offline.items()},
        "selection_auc": sel_auc, "calibration_auc": cal_auc, "forge_calibration": forge_cal,
        "min_type_sessions": MIN_TYPE_SESSIONS, "largest_violation_up_to_pi_max": largest,
    }, "summary")

    for name, draw in FIGURES.items():
        save_figure(draw(run.dir), run.figures, name)
        run.log(f"saved figure {name}")


def _line(method: str) -> dict:
    return {"color": style.color_for(method), "marker": MARKERS[method], "label": LABELS[method]}


def _arm(table: pd.DataFrame, arm: str) -> pd.DataFrame:
    # runs from before the reference windows hold real windows only
    return table[table["arm"] == arm] if "arm" in table else table


def fig_for_by_prevalence(run_dir) -> plt.Figure:
    # the reference windows of the main period, where label shift holds by construction, the figure of RQ1
    return _for_by_prevalence(run_dir, "reference")


def fig_for_by_prevalence_real(run_dir) -> plt.Figure:
    # the real windows of the main period, with the drift of their blocks
    return _for_by_prevalence(run_dir, "real")


def _for_by_prevalence(run_dir, arm: str) -> plt.Figure:
    style.use()
    table = runs.load_table(run_dir, "by_prevalence")
    summary = runs.load_json(run_dir, "summary")
    methods = [m for m in METHODS if m in summary["methods"]]
    mixes = summary["mixes"]
    # the windows of the main period, all deployment windows in runs from before the periods were set apart
    data = _arm(table, arm)
    data = data[data["period"] == summary.get("main_period", "all")]
    window = "reference window" if arm == "reference" and "arm" in table else "window"
    return prevalence.for_by_prevalence(data, summary, methods, _line, mixes, MIX_LABELS, window)


def fig_violation_by_window(run_dir) -> plt.Figure:
    style.use()
    claims = runs.load_table(run_dir, "claims")
    summary = runs.load_json(run_dir, "summary")
    methods = [m for m in METHODS if m in summary["methods"]]
    mixes = summary["mixes"]
    data = _arm(claims, "real")
    data = data[data["prevalence"] == summary["main_prevalence"]]
    fig, axes = plt.subplots(len(mixes) + 1, 1, figsize=(style.WIDTH_DOUBLE, 0.8 + 1.7 * (len(mixes) + 1)),
                             sharex=True)
    # one row per mix with the share of draws with FOR above epsilon in each real window and its 95% interval, the last
    # row with the mean share of the window sessions dismissed at the first mix, and the best threshold up to pi_max. a
    # block that cannot be thinned to a mix has no point in its row
    step = 0.15 / summary["blocks_per_year"]
    for k, method in enumerate(methods):
        for ax, mix in zip(axes, mixes):
            g = data[(data["method"] == method) & (data["mix"] == mix)].sort_values("position")
            layout.points(ax, layout.dodge(g["position"], k, len(methods), step), g["rate"], g["low"], g["high"],
                          **_line(method))
        g = data[(data["method"] == method) & (data["mix"] == mixes[0])].sort_values("position")
        axes[-1].plot(g["position"], g["dismissed_mean"], markersize=2.5, linewidth=1, **_line(method))
    # the best threshold up to pi_max of each window, the yardstick of FORGE, absent from runs from before it
    g = data[(data["method"] == methods[0]) & (data["mix"] == mixes[0])].sort_values("position")
    yardstick = []
    if "best_up_to_pi_max_dismissed" in g:
        name, dash, label = BEST_LINES[0]
        yardstick = [(axes[-1].plot(g["position"], g[f"{name}_dismissed"], color=style.INK, linestyle=dash,
                                    linewidth=1, label=label)[0], label)]
    for ax, mix in zip(axes, mixes):
        ax.axhline(summary["delta"], color=style.GRAY, linestyle="--", linewidth=1, label=r"$\delta$")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(f"share with FOR > $\\varepsilon$\n{MIX_LABELS[mix]}")
    _time_axes(axes, summary)
    axes[-1].set_ylim(0, 1)
    axes[-1].yaxis.set_major_formatter(style.percent)
    axes[-1].set_ylabel(f"window sessions dismissed\n{MIX_LABELS[mixes[0]]}")
    axes[-1].set_xlabel(f"deployment year, each split into {summary['blocks_per_year']} blocks of equal session "
                        f"count, attack prevalence {style.percent(summary['main_prevalence'])}")
    layout.legend_below(fig, axes[0], [LABELS[m] for m in methods], yardstick)
    return fig


def fig_drift_by_window(run_dir) -> plt.Figure:
    style.use()
    windows = runs.load_table(run_dir, "windows").sort_values("position")
    summary = runs.load_json(run_dir, "summary")
    cal = summary["forge_calibration"]
    fig, axes = plt.subplots(3, 1, figsize=(style.WIDTH_DOUBLE, 0.8 + 1.7 * 3), sharex=True)
    # top, the AUC of the scorer in each block. middle, the miss rate of each attack type at the offline FORGE
    # thresholds over the same in the calibration year, from blocks with enough sessions of the type. bottom, the
    # benign share dismissed at those thresholds over the same in the calibration year. a ratio of 1 is no drift, and
    # the offline claim needs both ratios near 1, the online claim the middle one
    x = windows["position"].to_numpy()
    line = {"color": style.INK, "markersize": 2.5, "linewidth": 1}
    axes[0].plot(x, windows["auc"], marker="o", **line)
    calibration = axes[0].axhline(summary["calibration_auc"], color=style.GRAY, linestyle="--", linewidth=1)
    axes[0].set_ylabel("AUC of the scorer")
    ratios = {}
    for family in FAMILIES:
        kept = windows[family].to_numpy() >= MIN_TYPE_SESSIONS
        ratios[family] = np.where(kept, windows[f"forge_miss_{family}"].to_numpy() / cal["miss"][family], np.nan)
    ratios["benign"] = windows["forge_benign_dismissed"].to_numpy() / cal["benign_dismissed"]
    # a block with no miss at the thresholds has a ratio of 0, which sits on the lower edge of the log axis
    shown = np.concatenate(list(ratios.values()))
    shown = shown[np.isfinite(shown) & (shown > 0)]
    low = 10.0 ** np.floor(np.log10(min(shown.min(), 1.0))) if len(shown) else 0.1
    high = 10.0 ** np.ceil(np.log10(max(shown.max(), 1.0))) if len(shown) else 10.0
    # dashed and dotted gray lines already mark the calibration year and the period boundaries
    for family, marker, dash in zip(FAMILIES, ["o", "s", "^", "D"], ["-", "-.", (0, (5, 1, 1, 1)), (0, (1, 1))]):
        axes[1].plot(x, np.where(ratios[family] == 0, low, ratios[family]), marker=marker, linestyle=dash,
                     label=f"{family} attack type", **line)
    axes[2].plot(x, np.where(ratios["benign"] == 0, low, ratios["benign"]), marker="o", **line)
    for ax in axes[1:]:
        ax.axhline(1, color=style.GRAY, linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_ylim(low, high)
    axes[1].set_ylabel("attacks dismissed,\nwindow over calibration")
    axes[2].set_ylabel("benign sessions dismissed,\nwindow over calibration")
    _time_axes(axes, summary)
    axes[-1].set_xlabel(f"deployment year, each split into {summary['blocks_per_year']} blocks of equal session "
                        f"count")
    layout.legend_below(fig, axes[1], [f"{f} attack type" for f in FAMILIES], [(calibration, "calibration year")])
    return fig


def _time_axes(axes, summary: dict) -> None:
    # the boundaries between AnoShift periods inside the deployment years, the panel letters, and one tick at the
    # start of every deployment year on the shared x axis
    deployment = summary["years"]["deployment"]
    for ax, letter in zip(axes, "abcdef"):
        for first, _ in summary["periods"].values():
            if deployment[0] < first <= deployment[-1]:
                ax.axvline(first, color=style.GRAY, linestyle=":", linewidth=1, label="AnoShift period boundary")
        layout.panel_letter(ax, letter)
    axes[-1].set_xlim(deployment[0], deployment[-1] + 1)
    axes[-1].set_xticks(np.arange(deployment[0], deployment[-1] + 2))


FIGURES = {
    "for_by_prevalence": fig_for_by_prevalence,
    "for_by_prevalence_real": fig_for_by_prevalence_real,
    "violation_by_window": fig_violation_by_window,
    "drift_by_window": fig_drift_by_window,
}
