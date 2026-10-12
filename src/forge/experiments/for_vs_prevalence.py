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
the score distributions of each class and attack type in the whole block. the blocks of the deployment years hold
63% to 99.4% attacks, far outside any claim, so only thinned windows are evaluated. each draw also takes one window of
the thinned size from the block, with replacement, on which the online certificate runs and the realized share of
attacks among the dismissed sessions is counted.

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
family sizes only. once merged, the FORGE claim covers the calibration mix of attack types only, and the unknown type
makes up 0.8% of the attacks of 2010, 1.8% of those of 2011 and 0.002% of those of 2015. the attacks of a window
therefore come in more than one mix, the block's own, which an operator would meet, and the known type alone, which
keeps the change of mix out of the comparison and differs from the calibration mix by the 0.8% of unknown attacks.
the calibration mix itself would weight the few unknown sessions of a late block up to 0.8% of the attacks, so their
sampling noise would decide the rate. a block that holds fewer attacks of a type than its thinned window would take
on average has no window at that prevalence and mix. the deployment years also drift away from the calibration year in
the scores of both classes, so a FORGE failure with the known type alone can still come from drift, which the miss and
benign dismiss rates saved per window show.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter

from forge import runs
from forge.certify.certificate import (certify, certify_online, conformal_fnr, counts_below, family_groups,
                                       for_from_rates, group_rows, ltt_iid, min_family_size, plug_in)
from forge.data import anoshift, splits
from forge.data.synthetic import draw_counts
from forge.eval.metrics import auc, share_with_interval
from forge.models import boosting
from forge.models.encoding import Encoder
from forge.viz import layout, style
from forge.viz.export import save_figure

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
# quantiles of the FOR over the certified draw and window pairs in the figure over prevalences
LOW, HIGH = 0.05, 0.95


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
    # counts at or below each threshold of the benign sessions, code 0, and of each attack type, codes 1 and up, and
    # the size of each class
    return {"benign_below": counts_below(score[code == 0], grid),
            "family_below": np.stack([counts_below(score[code == k], grid) for k in range(1, len(FAMILIES) + 1)]),
            "sizes": np.bincount(code, minlength=len(FAMILIES) + 1)}


def add_counts(a: dict, b: dict) -> dict:
    return {key: a[key] + b[key] for key in a}


def rates(counts: dict) -> tuple:
    # empirical cdfs at the grid of the benign class, of each attack type and of all attacks, and the mix of the types
    sizes = counts["sizes"]
    n_family = sizes[1:]
    benign = counts["benign_below"] / sizes[0]
    family = np.divide(counts["family_below"], n_family[:, None], out=np.zeros(counts["family_below"].shape),
                       where=n_family[:, None] > 0)
    attack = counts["family_below"].sum(0) / n_family.sum()
    return benign, family, attack, n_family / n_family.sum()


def mix_shares(mix: str, own: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    # share of each attack type among the attacks of a window
    if mix == "window":
        return own
    if mix == "calibration":
        return calibration
    return np.array([f == mix for f in FAMILIES], dtype=float)


def grid_from(benign_scores: np.ndarray, size: int) -> np.ndarray:
    # benign scores at the levels j / (size + 1), each the smallest score whose empirical cdf reaches the level, so
    # every threshold is a score that occurs. ties among the scores merge some levels into one threshold
    ordered = np.sort(benign_scores)
    levels = np.arange(1, size + 1) / (size + 1)
    positions = np.clip(np.ceil(levels * len(ordered)).astype(int) - 1, 0, len(ordered) - 1)
    return np.unique(ordered[positions])


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


def evaluate(methods, offline, online, pi, m_w, b_w, attack_below, window_below, eps, values, cell) -> list[dict]:
    # one row per method for one window, prevalence and mix. the true rate and dismissed share follow from the cdfs of
    # the block at the grid, the realized share of attacks among the dismissed sessions from the drawn window
    true_rate = for_from_rates(pi, m_w, b_w)
    true_share = pi * m_w + (1 - pi) * b_w
    rows = []
    for method in methods:
        out = online if method == "forge_online" else offline[method]
        hit = out["index"] >= 0
        safe = np.maximum(out["index"], 0)
        rate = np.where(hit, true_rate[safe], 0.0)
        share = np.where(hit, true_share[safe], 0.0)
        over = rate > eps
        # a FORGE draw that dismisses less of the window than its floor breaks its claim as well
        under = hit & (share < out["floor"])
        caught = np.take_along_axis(attack_below, safe[:, None], 1)[:, 0]
        dismissed = np.take_along_axis(window_below, safe[:, None], 1)[:, 0]
        realized = hit & (caught > eps * dismissed)
        values[(method, pi, cell["mix"], cell["window"])] = rate[hit]
        draws = len(hit)
        rows.append({
            "method": method, **cell, "refused": float(np.mean(~hit)), "dismissed_mean": float(share.mean()),
            "benign_dismissed_mean": float(np.where(hit, b_w[safe], 0.0).mean()),
            "miss_mean": float(np.where(hit, m_w[safe], 0.0).mean()),
            "for_median": float(np.median(rate[hit])) if hit.any() else np.nan,
            "violations": int(over.sum()), **share_with_interval(int(over.sum()), draws),
            "floor_violations": int(under.sum()),
            "realized_violations": int(realized.sum()), "realized_rate": float(realized.mean()),
        })
    return rows


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
                windows.append({"year": year, "block": block, **counts})
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

    # one stream for the calibration sets and one for each window, prevalence and mix
    seeds = np.random.SeedSequence(cfg["seed"]).spawn(1 + len(windows) * len(prevalences) * len(mixes))
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

    rows, values, skipped = [], {}, []
    shape = (len(windows), len(prevalences), len(mixes), draws)
    online_index = np.full(shape, -1, dtype=np.int16)
    online_floor = np.zeros(shape, dtype=np.float32)
    for i, win in enumerate(windows):
        b_w, m_w_family, _, shares_w = rates(win)
        period = anoshift.period(win["year"])
        position = win["year"] + (win["block"] + 0.5) / d["blocks_per_year"]
        for j, pi in enumerate(prevalences):
            n_w = int(round(win["sizes"][0] / (1 - pi)))
            for k, mix in enumerate(mixes):
                shares = mix_shares(mix, shares_w, shares_cal)
                # a window stands for the block thinned to its prevalence and mix, so the block must hold at least the
                # attacks of each type that the window takes on average. a rare type of the calibration mix may lack
                # them in a block, whose few sessions of that type would otherwise decide the rate alone
                if np.any(win["sizes"][1:] < n_w * pi * shares):
                    skipped.append([win["year"], win["block"], pi, mix])
                    continue
                rng = np.random.default_rng(seeds[1 + (i * len(prevalences) + j) * len(mixes) + k])
                drawn = draw_counts(rng, n_w, pi, shares, b_w, m_w_family, draws)
                attack_below = drawn.family_below.sum(1)
                window_below = drawn.benign_below + attack_below
                online = certify_online(window_below, np.full(draws, n_w), grouped_below, grouped_n, eps, delta,
                                        pi_max, part, **start)
                online_index[i, j, k], online_floor[i, j, k] = online["index"], online["floor"]
                m_w = shares @ m_w_family
                rows += evaluate(methods, offline, online, pi, m_w, b_w, attack_below, window_below, eps, values,
                                 {"window": i, "year": win["year"], "block": win["block"], "period": period,
                                  "position": position, "prevalence": pi, "mix": mix, "window_sessions": n_w,
                                  "draws": draws})
    for mix in mixes:
        left = [s for s in skipped if s[-1] == mix]
        if left:
            run.log(f"{mix} mix, {len(left)} of {len(windows) * len(prevalences)} windows and prevalences left out, "
                    f"their blocks lack the attacks of a type that the thinned window would take")
    claims = pd.DataFrame(rows)
    run.save_table(claims, "claims")

    # pooled over the windows of each period and of all deployment years, the FOR quantiles over the certified draw
    # and window pairs, the violation share over all pairs with its range over the windows
    summary_rows = []
    for (method, pi, mix), cell in claims.groupby(["method", "prevalence", "mix"], sort=False):
        for name in ["all", *dict.fromkeys(cell["period"])]:
            g = cell if name == "all" else cell[cell["period"] == name]
            certified = np.concatenate([values[(method, pi, mix, w)] for w in g["window"]])
            pairs = len(g) * draws
            summary_rows.append({
                "method": method, "prevalence": pi, "mix": mix, "period": name, "windows": len(g), "pairs": pairs,
                "violation_rate": g["violations"].sum() / pairs, "violation_window_min": g["rate"].min(),
                "violation_window_max": g["rate"].max(), "refused": g["refused"].mean(),
                "dismissed_mean": g["dismissed_mean"].mean(),
                "for_median": float(np.median(certified)) if len(certified) else np.nan,
                "for_low": float(np.quantile(certified, LOW)) if len(certified) else np.nan,
                "for_high": float(np.quantile(certified, HIGH)) if len(certified) else np.nan,
                "realized_rate": g["realized_violations"].sum() / pairs,
            })
    by_prevalence = pd.DataFrame(summary_rows)
    run.save_table(by_prevalence, "by_prevalence")
    for r in by_prevalence[by_prevalence["period"] == "all"].itertuples():
        run.log(f"{r.method}, {r.mix} mix, prevalence {r.prevalence}, FOR above epsilon in {r.violation_rate:.4f} of "
                f"draw and window pairs, from {r.violation_window_min:.4f} to {r.violation_window_max:.4f} by window, "
                f"median FOR of certified pairs {r.for_median:.2e}, mean share of window sessions dismissed "
                f"{r.dismissed_mean:.4f}, refused {r.refused:.3f}")

    run.save_table(pd.DataFrame([{
        "window": i, "year": w["year"], "block": w["block"], "period": anoshift.period(w["year"]),
        "position": w["year"] + (w["block"] + 0.5) / d["blocks_per_year"], "sessions": int(w["sizes"].sum()),
        "benign": int(w["sizes"][0]), **{f: int(n) for f, n in zip(FAMILIES, w["sizes"][1:])},
        "attack_share": float(w["sizes"][1:].sum() / w["sizes"].sum()),
    } for i, w in enumerate(windows)]), "windows")
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
                    **{f"{m}_index": out["index"] for m, out in offline.items()}, online_index=online_index,
                    online_floor=online_floor, counts=np.column_stack([cal.n_benign, cal.n_family]))
    run.save_json({"encoder": enc.to_dict(), "params": sc["params"], "rounds": sc["rounds"],
                   "early_stopping": sc["early_stopping"], "best_iteration": booster.best_iteration,
                   "selection_loss": sel_loss,
                   "train_sessions": n_train, "train_attacks": train_attacks, "selection_auc": sel_auc,
                   "calibration_auc": cal_auc}, "scorer")

    # the claim of each FORGE method covers prevalences up to pi_max at the calibration mix once rare types are merged,
    # the largest window violation share there is the number to set against delta
    covered = claims[claims["prevalence"] <= pi_max]
    largest = {}
    for (method, mix), g in covered.groupby(["method", "mix"], sort=False):
        top = g.loc[g["rate"].idxmax()]
        largest.setdefault(method, {})[mix] = {
            "year": int(top["year"]), "block": int(top["block"]), "prevalence": float(top["prevalence"]),
            "rate": float(top["rate"]), "low": float(top["low"]), "high": float(top["high"])}
        run.log(f"{method}, {mix} mix, largest share of draws with FOR above epsilon at prevalences up to pi_max "
                f"{top['rate']:.4f}, window {int(top['year'])} block {int(top['block'])}, prevalence "
                f"{top['prevalence']}, 95% interval {top['low']:.4f} to {top['high']:.4f}")
    run.save_json({
        "dataset": d["name"], "processed_version": d["processed_version"], "fingerprint": manifest["fingerprint"],
        "years": years, "periods": {p: list(s) for p, s in anoshift.PERIODS.items()},
        "blocks_per_year": d["blocks_per_year"], "epsilon": eps, "delta": delta, "delta_benign": part,
        "pi_max": pi_max, **start, "grid_size": c["grid_size"], "grid_thresholds": len(grid),
        "family_floor": c["family_floor"], "family_minimum": minimum, "families": families,
        "calibration_prevalence": pi_cal, "calibration_sessions": n_cal,
        "calibration_expected_attacks": dict(zip(FAMILIES, expected.tolist())),
        "fixed_index": fixed, "beta": beta, "prevalences": prevalences, "main_prevalence": dep["main_prevalence"],
        "mixes": mixes, "skipped": skipped, "methods": methods, "draws": draws, "windows": len(windows),
        "offline_refused": {m: float(np.mean(out["index"] < 0)) for m, out in offline.items()},
        "selection_auc": sel_auc, "calibration_auc": cal_auc, "largest_violation_up_to_pi_max": largest,
    }, "summary")

    for name, draw in FIGURES.items():
        save_figure(draw(run.dir), run.figures, name)
        run.log(f"saved figure {name}")


def _line(method: str) -> dict:
    return {"color": style.color_for(method), "marker": MARKERS[method], "label": LABELS[method]}


def _prevalence_axis(ax, summary: dict, bottom: bool) -> None:
    ax.axvline(summary["calibration_prevalence"], color=style.GRAY, linestyle=":", linewidth=1,
               label=r"$\pi$ of the calibration set")
    ax.axvline(summary["pi_max"], color=style.GRAY, linestyle="-.", linewidth=1, label=r"$\pi_{\max}$")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(style.percent)
    ax.xaxis.set_minor_formatter(NullFormatter())
    if bottom:
        ax.set_xlabel("attack prevalence in the window")


def fig_for_by_prevalence(run_dir) -> plt.Figure:
    style.use()
    table = runs.load_table(run_dir, "by_prevalence")
    summary = runs.load_json(run_dir, "summary")
    methods = [m for m in METHODS if m in summary["methods"]]
    mixes = summary["mixes"]
    data = table[table["period"] == "all"]
    fig, axes = plt.subplots(len(mixes), 3, figsize=(style.WIDTH_DOUBLE, 0.5 + 2.2 * len(mixes)), sharex=True,
                             squeeze=False)
    # one row per mix of attack types. left, the FOR of the certified draw and window pairs, median and the 5% to 95%
    # range. middle, the share of all pairs with FOR above epsilon, a refusal counting as no violation, with the range
    # of the per window shares. right, the mean share of the window sessions dismissed, a refusal counting as none
    for row, mix in zip(axes, mixes):
        for k, method in enumerate(methods):
            g = data[(data["method"] == method) & (data["mix"] == mix)].sort_values("prevalence")
            x = layout.dodge(g["prevalence"], k, len(methods), 0.035, log=True)
            layout.points(row[0], x, g["for_median"], g["for_low"], g["for_high"], **_line(method))
            layout.points(row[1], x, g["violation_rate"], g["violation_window_min"], g["violation_window_max"],
                          **_line(method))
            row[2].plot(x, g["dismissed_mean"], markersize=3.5, linewidth=1.2, **_line(method))
        row[0].axhline(summary["epsilon"], color=style.GRAY, linestyle="--", linewidth=1, label=r"$\varepsilon$")
        delta_line = row[1].axhline(summary["delta"], color=style.GRAY, linestyle="--", linewidth=1)
        row[0].set_ylabel(f"FOR in the window\n{MIX_LABELS[mix]}")
        row[1].set_ylim(-0.02, 1.02)
        row[1].set_ylabel(r"share with FOR > $\varepsilon$")
        row[2].set_ylim(0, 1)
        row[2].yaxis.set_major_formatter(style.percent)
        row[2].set_ylabel("window sessions dismissed")
    # a FOR of 0, where no attack of the block falls at or below the threshold, sits on the lower edge
    shown = data[["for_median", "for_low", "for_high"]].to_numpy().ravel()
    shown = shown[shown > 0]
    for r, row in enumerate(axes):
        for ax in row:
            _prevalence_axis(ax, summary, r == len(axes) - 1)
        if len(shown):
            row[0].set_yscale("log")
            row[0].set_ylim(10.0 ** np.floor(np.log10(shown.min())),
                            10.0 ** np.ceil(np.log10(max(shown.max(), summary["epsilon"]))))
    for ax, letter in zip(axes.flat, "abcdefghi"):
        layout.panel_letter(ax, letter)
    layout.legend_below(fig, axes[0, 0], [LABELS[m] for m in methods], [(delta_line, r"$\delta$")])
    return fig


def fig_violation_by_window(run_dir) -> plt.Figure:
    style.use()
    claims = runs.load_table(run_dir, "claims")
    summary = runs.load_json(run_dir, "summary")
    methods = [m for m in METHODS if m in summary["methods"]]
    mixes = summary["mixes"]
    data = claims[claims["prevalence"] == summary["main_prevalence"]]
    fig, axes = plt.subplots(len(mixes) + 1, 1, figsize=(style.WIDTH_DOUBLE, 0.8 + 1.7 * (len(mixes) + 1)),
                             sharex=True)
    # one row per mix with the share of draws with FOR above epsilon in each window and its 95% interval, the last row
    # with the mean share of the window sessions dismissed at the first mix. a block that cannot be thinned to a mix
    # has no point in its row
    step = 0.15 / summary["blocks_per_year"]
    for k, method in enumerate(methods):
        for ax, mix in zip(axes, mixes):
            g = data[(data["method"] == method) & (data["mix"] == mix)].sort_values("position")
            layout.points(ax, layout.dodge(g["position"], k, len(methods), step), g["rate"], g["low"], g["high"],
                          **_line(method))
        g = data[(data["method"] == method) & (data["mix"] == mixes[0])].sort_values("position")
        axes[-1].plot(g["position"], g["dismissed_mean"], markersize=2.5, linewidth=1, **_line(method))
    for ax, mix in zip(axes, mixes):
        ax.axhline(summary["delta"], color=style.GRAY, linestyle="--", linewidth=1, label=r"$\delta$")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(f"share with FOR > $\\varepsilon$\n{MIX_LABELS[mix]}")
    deployment = summary["years"]["deployment"]
    for ax, letter in zip(axes, "abcdef"):
        for first, _ in summary["periods"].values():
            if deployment[0] < first <= deployment[-1]:
                ax.axvline(first, color=style.GRAY, linestyle=":", linewidth=1, label="AnoShift period boundary")
        layout.panel_letter(ax, letter)
    axes[-1].set_ylim(0, 1)
    axes[-1].yaxis.set_major_formatter(style.percent)
    axes[-1].set_ylabel(f"window sessions dismissed\n{MIX_LABELS[mixes[0]]}")
    axes[-1].set_xlim(deployment[0], deployment[-1] + 1)
    axes[-1].set_xticks(np.arange(deployment[0], deployment[-1] + 2))
    axes[-1].set_xlabel(f"deployment year, each split into {summary['blocks_per_year']} blocks of equal session "
                        f"count, attack prevalence {style.percent(summary['main_prevalence'])}")
    layout.legend_below(fig, axes[0], [LABELS[m] for m in methods])
    return fig


FIGURES = {
    "for_by_prevalence": fig_for_by_prevalence,
    "violation_by_window": fig_violation_by_window,
}
