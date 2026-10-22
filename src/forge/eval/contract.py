"""the contract table of E7, the share of the flows FORGE dismisses and the workload it leaves against the two terms an
operator sets, the FOR target epsilon and the largest prevalence pi_max the certificate covers.

a threshold has FOR at most epsilon at pi_max exactly when its miss rate over its benign share is at most kappa,
epsilon (1 - pi_max) over (1 - epsilon) pi_max, so the certificate, its feasible start and the yardsticks read the two
terms through kappa alone, and pairs with the same kappa certify the same threshold.

each run of RQ1 certifies at one epsilon and one pi_max, and the certificate depends on the calibration set through
its counts at the grid alone, so every other pair of terms is certified again here from the class counts of the
calibration part that the run saved, on calibration sets redrawn from the stream of the run's own sets, as
forge.eval.starts does. the bounds on the family miss rates and the benign share depend on the calibration set and
delta alone, so they are computed once and serve every pair, while the start and the fixed sequence follow each
pair. the family groups stay those of the run, so every pair covers the same mixes as the claim of the paper. the
rule that merges a family too small to stand alone would merge more of them at a stricter pair, and the claim would
then cover fewer mixes as the terms tighten. the start keeps the run's rule, the feasible start on AnoShift and
LITNET and the share start on UGR16.

a contract is read on the calibration distributions, where the claim of Theorem 1 holds by construction, the share of
the flows dismissed at the calibration prevalence and mix with the workload it leaves, the escalated flows and the
audits of a monitor riding on the dismissed ones, the share of the draws whose threshold breaks the claim at pi_max
for the worst covered mix, and the two yardsticks at the same terms, the loosest threshold safe up to pi_max at the
calibration mix and the loosest safe for every covered mix. the redrawn sets are the run's own sets where numpy draws
the streams of Colab and sets of the same law elsewhere.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from forge import runs
from forge.certify.certificate import _miss_upper, _share_lower, _start, fixed_sequence, for_from_rates, group_rows
from forge.data.synthetic import draw_counts
from forge.eval import seeds
from forge.eval.paper import uniform
from forge.eval.windows import best_thresholds, every_mix_threshold, rates

# the FOR targets of E5 and the prevalences of E1 up to its largest
EPSILONS = (0.0001, 0.0005, 0.001, 0.005, 0.01)
PI_MAX = (0.01, 0.02, 0.05, 0.1, 0.2)
MILLION = 1e6
# the columns of the pooled table, in their median, low, high and pooled forms
COLUMNS = ["refused", "violation_rate", "benign_dismissed_mean", "dismissed_mean", "workload_per_million",
           "best_up_to_pi_max_dismissed", "best_every_mix_dismissed", "price_every_mix"]


def calibration_population(hist: dict) -> dict:
    # the cdfs of the calibration part at the grid and the mix of the calibration sets, which a run of e11 keeps as
    # the mix of its calibration year
    b, m_family, _, natural = rates({"benign_below": hist["calibration_benign_below"],
                                     "family_below": hist["calibration_family_below"],
                                     "sizes": hist["calibration_sizes"]})
    shares = np.asarray(hist["shares"]) if "shares" in hist else natural
    return {"b": b, "m_family": m_family, "shares": shares, "m": shares @ m_family}


def run_rows(run_dir, epsilons=EPSILONS, pi_maxes=PI_MAX, audit_rate: float = 0.005) -> pd.DataFrame:
    """one row per pair of epsilon and pi_max for one run of e11, e12 or e13, on its calibration sets redrawn."""
    run_dir = Path(run_dir)
    hist = runs.load_arrays(run_dir, "histograms")
    summary = runs.load_json(run_dir, "summary")
    seed = int(runs.load_run_config(run_dir).get("seed", 0))
    delta, part_b = summary["delta"], summary["delta_benign"]
    part_m = delta - part_b
    pi_cal, n_cal, draws = summary["calibration_prevalence"], int(summary["calibration_sessions"]), summary["draws"]
    start = {"start": summary.get("start") or "feasible", "start_slack": summary["start_slack"],
             "start_share": summary.get("start_share")}
    misses = int(summary["start_misses"])
    pop = calibration_population(hist)
    b, m_family, shares, m = pop["b"], pop["m_family"], pop["shares"], pop["m"]
    groups = np.asarray(hist["groups"])
    # the share of the flows of the calibration mix dismissed at each threshold, attacks included, and the miss rate
    # of every group at its calibration proportions, the worst of which sets the true rate of the claim at pi_max
    dismissed = pi_cal * m + (1 - pi_cal) * b
    weights = np.stack([np.where(groups == g, shares, 0.0) for g in np.unique(groups)])
    worst_miss = ((weights / weights.sum(1, keepdims=True)) @ m_family).max(0)

    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    cal = draw_counts(rng, n_cal, pi_cal, shares, b, m_family, draws)
    b_lower = _share_lower(cal.benign_below, cal.n_benign, part_b)
    m_upper, m_with = _miss_upper(*group_rows(cal.family_below, cal.n_family, groups), part_m, misses)
    rows = []
    for eps in epsilons:
        for pi_max in pi_maxes:
            first = _start([for_from_rates(pi_max, w[..., None], b_lower) for w in m_with], b_lower, eps,
                           start["start"], start["start_slack"], start["start_share"])
            index = fixed_sequence(for_from_rates(pi_max, m_upper, b_lower) <= eps, first)
            hit = index >= 0
            at = np.maximum(index, 0)
            worst = for_from_rates(pi_max, worst_miss, b)
            share = np.where(hit, dismissed[at], 0.0)
            every = every_mix_threshold(pi_max, m_family, b, groups, shares, eps)
            best = best_thresholds(pi_cal, pi_max, m, b, eps)
            every_share = float(dismissed[every]) if every >= 0 else 0.0
            rows.append({
                "seed": seed, "run_id": run_dir.name, "dataset": summary["dataset"], "epsilon": eps,
                "pi_max": pi_max, "kappa": eps * (1 - pi_max) / ((1 - eps) * pi_max),
                "operating": bool(np.isclose(eps, summary["epsilon"]) and np.isclose(pi_max, summary["pi_max"])),
                "calibration_prevalence": pi_cal, "start": start["start"], "groups": int(len(np.unique(groups))),
                "draws": int(draws), "audit_rate": audit_rate, "refused": float(np.mean(~hit)),
                "violation_rate": float(np.mean(hit & (worst[at] > eps))),
                "benign_dismissed_mean": float(np.where(hit, b[at], 0.0).mean()),
                "dismissed_mean": float(share.mean()),
                "workload_per_million": float(((1 - share) + audit_rate * share).mean() * MILLION),
                "best_up_to_pi_max_dismissed": best["best_up_to_pi_max_dismissed"],
                "best_every_mix_dismissed": every_share,
                "price_every_mix": every_share - float(share.mean()),
            })
    return pd.DataFrame(rows)


def contract_table(folders: dict, epsilons=EPSILONS, pi_maxes=PI_MAX, audit_rate: float = 0.005) -> pd.DataFrame:
    """the contracts of every dataset over seeds, folders mapping a dataset to its run folders by seed, one row per
    dataset, epsilon and pi_max with the median and the 5% to 95% range over seeds and the shares of draws pooled over
    the draws of every seed."""
    parts = []
    for dataset, by_seed in folders.items():
        if not by_seed:
            continue
        frames = {seed: run_rows(folder, epsilons, pi_maxes, audit_rate) for seed, folder in sorted(by_seed.items())}
        keys = ["dataset", "epsilon", "pi_max"]
        pooled = uniform(seeds.pool({s: f.drop(columns=["seed", "run_id"]) for s, f in frames.items()}, keys,
                                    "draws"), COLUMNS)
        # a share no seed moved has no pooled form, and its one value is the share over every seed
        for col in COLUMNS:
            if seeds._is_rate(col) and f"{col}_pooled" not in pooled:
                pooled[f"{col}_pooled"] = pooled[f"{col}_median"]
        first = frames[min(frames)]
        for col in ("kappa", "operating", "calibration_prevalence", "start", "audit_rate"):
            pooled[col] = first[col].to_numpy()
        parts.append(pooled)
    if not parts:
        raise ValueError("no run folder to read the contracts from")
    table = pd.concat(parts, ignore_index=True)
    out = ["dataset", "epsilon", "pi_max", "kappa", "operating", "calibration_prevalence", "start", "audit_rate",
           "seeds"]
    out += [c for c in ["groups", "groups_median", "groups_low", "groups_high", "draws"] if c in table]
    return table[out + seeds._stats(table, COLUMNS)].reset_index(drop=True)
