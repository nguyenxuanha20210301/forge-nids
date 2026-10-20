"""statistics over scorer seeds for the tables of the paper, the price of every method against the yardsticks, every
method against FORGE and every monitoring arm against the operating rule, each paired within seeds, the violation
share of FORGE per seed against delta, and Clopper-Pearson intervals on the shares pooled over the draws of every
seed.

a seed is a scorer of its own, and inside its run every method and every arm is judged on the same calibration draws
and the same windows, so a difference between two of them is paired within the seed while the seeds are independent.
a difference gets its median, its 5% and 95% quantiles and its least and largest value over seeds, the seeds where it
lies above, at and below zero, and the exact sign test, the binomial at one half on the seeds off zero, two sided.
the p values of one table are adjusted together by Holm, the table being one family.

a share pooled over seeds counts the draws of every seed as its units, a draw being one calibration set with its
windows. the windows of one draw share its calibration set, so a share over draw and window pairs takes the draws and
not the pairs as its units, which widens its interval. a share of sequences counts the sequences of every draw and
the share of alarms above the bound counts the alarms. every count comes from the run folders behind the pooled
tables.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, binomtest

from forge import runs
from forge.eval import seeds
from forge.eval.metrics import share_with_interval

# a difference this close to zero is none, a mean over draws of the share at one threshold differing from that share
# by rounding only
TIE = 1e-12
# the method every other one is set against, offline FORGE, and the forms of FORGE whose claim is read per seed
REFERENCE = "forge_simplex"
FORGE = ("forge_simplex", "forge_online")
# the arms whose windows lie inside the claim, label shift by construction on the reference windows and the benign
# flows of the window with the attacks of calibration on the S2 windows, which the online claim covers
CLAIM_ARMS = ("reference", "s2")
# the operating rule of the paper per dataset, a sequence per year where the stream has years, per clock week where it
# has weeks, and one sequence from the start of the deployment where it has neither
OPERATING_ARMS = {"anoshift": "monitor_year", "litnet2020": "monitor", "ugr16": "monitor_week"}
# the yardsticks of the price, the loosest threshold safe at every prevalence up to pi_max and the loosest safe for
# every mix the claim covers, which AnoShift does not have
YARDSTICKS = {"price": "best_up_to_pi_max_dismissed", "price_every_mix": "best_every_mix_dismissed"}
# the shares of E1, over the draw and window pairs of every seed, and those of E3 with the units each is taken of
E1_SHARES = ("violation_rate", "refused", "realized_rate")
E3_SHARES = {"alarm_rate": "draws", "false_alarm_rate": "draws", "onset_true_rate": "draws",
             "onset_realized_rate": "draws", "miscover_rate": "draws", "miscover_sequence_rate": "sequences",
             "over_bound_rate": "alarms"}
# the columns that name a row of the E1 and of the E3 table
E1_KEYS = ["dataset", "epsilon", "method", "mix", "prevalence"]
E3_KEYS = ["dataset", "stream", "policy", "composition", "arm"]
# the columns that name a window in the claims tables, a block of a year on AnoShift, a segment on LITNET and a start
# on UGR16
WINDOW_COLUMNS = ["window", "year", "block", "segment", "start"]


def sign_test(d) -> dict:
    """the seeds whose difference lies above, at and below zero, and the two sided exact sign test on the seeds off
    zero, with a p value of one when every seed is at zero and none when no seed has a difference."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    tied = np.abs(d) <= TIE
    above, below = int(np.sum(~tied & (d > 0))), int(np.sum(~tied & (d < 0)))
    if not len(d):
        p = np.nan
    elif above + below == 0:
        p = 1.0
    else:
        p = float(binomtest(above, above + below, 0.5).pvalue)
    return {"above": above, "tied": int(tied.sum()), "below": below, "p": p}


def holm(p) -> np.ndarray:
    """the p values of one family adjusted by the step down rule of Holm, a missing one staying missing and left out of
    the family."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    have = np.flatnonzero(np.isfinite(p))
    order = have[np.argsort(p[have], kind="stable")]
    factors = len(order) - np.arange(len(order))
    out[order] = np.minimum(1.0, np.maximum.accumulate(factors * p[order]))
    return out


def spread(values, prefix: str) -> dict:
    # the median, the 5% and 95% quantiles and the least and largest value over seeds
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    stats = {"median": np.median, "low": lambda x: np.quantile(x, seeds.LOW),
             "high": lambda x: np.quantile(x, seeds.HIGH), "min": np.min, "max": np.max}
    return {f"{prefix}_{k}": float(f(v)) if len(v) else np.nan for k, f in stats.items()}


def signs(d, prefix: str) -> dict:
    # the seeds above, at and below zero and the p value of the sign test, under one prefix
    test = sign_test(d)
    return {f"{prefix}_seeds_above": test["above"], f"{prefix}_seeds_tied": test["tied"],
            f"{prefix}_seeds_below": test["below"], f"{prefix}_p": test["p"]}


def adjust(table: pd.DataFrame, prefixes) -> pd.DataFrame:
    """the table with the p values of every prefix adjusted together by Holm, the table being one family, each
    adjusted value as <prefix>_p_holm beside its own."""
    t = table.copy()
    columns = [f"{prefix}_p" for prefix in prefixes if f"{prefix}_p" in t]
    if t.empty or not columns:
        return t
    adjusted = holm(t[columns].to_numpy(dtype=float).ravel()).reshape(len(t), len(columns))
    for k, col in enumerate(columns):
        t.insert(t.columns.get_loc(col) + 1, f"{col}_holm", adjusted[:, k])
    return t


def interval(count, units) -> dict:
    # the share and its two sided 95% Clopper-Pearson interval, none where a count or its units are missing
    count, units = np.asarray(count, dtype=float), np.asarray(units, dtype=float)
    out = {k: np.full(count.shape, np.nan) for k in ("rate", "low", "high")}
    ok = np.isfinite(count) & np.isfinite(units) & (units > 0)
    if ok.any():
        got = share_with_interval(count[ok], units[ok])
        for k in out:
            out[k][ok] = got[k]
    return out


def _settings(summary: dict) -> dict:
    return {"dataset": summary["dataset"], "epsilon": summary["epsilon"], "delta": summary["delta"],
            "pi_max": summary["pi_max"], "draws": summary["draws"]}


def _total(values: pd.Series) -> float:
    # a sum over seeds that a missing value makes missing
    return float(values.to_numpy(dtype=float).sum())


def _whole(t: pd.DataFrame) -> pd.DataFrame:
    # the year and the block of a window as whole numbers, missing where the dataset names its windows otherwise
    for col in ("year", "block"):
        if col in t:
            t[col] = t[col].astype("Int64")
    return t


def e1_seed_rows(folders: dict, experiment: str) -> pd.DataFrame:
    """the rows of the E1 table of every seed of one experiment of RQ1, folders mapping a seed to its run folder, on
    the windows RQ1 is read on as forge.eval.seeds cuts the pooled table, each with its seed and run and the dataset,
    epsilon, delta, pi_max and draws of the run."""
    parts = []
    for seed, folder in sorted(folders.items()):
        summary = runs.load_json(folder, "summary")
        t = seeds.e1_table(runs.load_table(folder, "by_prevalence"), summary, experiment)
        parts.append(t.assign(**_settings(summary), seed=int(seed), run_id=Path(folder).name))
    if not parts:
        raise ValueError(f"no {experiment} run folder to read the E1 rows from")
    return pd.concat(parts, ignore_index=True)


def e1_paired_price(rows: pd.DataFrame) -> pd.DataFrame:
    """the price of every method at pi_max per dataset, epsilon and mix, from the E1 rows of every seed, the share of
    the window flows a yardstick dismisses less the share the method dismisses, per seed, against the loosest threshold
    safe at every prevalence up to pi_max and against the loosest safe for every mix the claim covers, in flows per
    million as well. a threshold safe at pi_max dismisses no more than the loosest safe one, so the price of a method
    that holds its claim lies at or above zero in every seed by construction, and its size is the reading."""
    keys = ["dataset", "epsilon", "method", "mix", "prevalence"]
    at = rows[np.isclose(rows["prevalence"], rows["pi_max"])]
    for (dataset, eps), g in rows.groupby(["dataset", "epsilon"], sort=False):
        if not np.isclose(g["prevalence"], g["pi_max"]).any():
            raise ValueError(f"{dataset} at epsilon {eps} has no E1 row at pi_max {g['pi_max'].iloc[0]}")
    out = []
    for key, g in at.groupby(keys, sort=False):
        row = {**dict(zip(keys, key)), "seeds": int(g["seed"].nunique())}
        for prefix, yardstick in YARDSTICKS.items():
            price = g[yardstick] - g["dismissed_mean"] if yardstick in g else pd.Series(np.nan, index=g.index)
            row.update(spread(price, prefix))
            row.update({f"{prefix}_per_million_{k}": row[f"{prefix}_{k}"] * 1e6 for k in ("median", "low", "high")})
            row.update(signs(price, prefix))
        out.append(row)
    return adjust(pd.DataFrame(out), YARDSTICKS)


def e1_paired_baselines(rows: pd.DataFrame, reference: str = REFERENCE) -> pd.DataFrame:
    """every other method against offline FORGE per dataset, epsilon, mix and prevalence, from the E1 rows of every
    seed, its violation share and its dismissed share less those of FORGE in the same seed, both judged on the same
    draws and windows. the other forms of FORGE stand among the baselines."""
    keys = ["dataset", "epsilon", "mix", "prevalence"]
    base = rows[rows["method"] == reference].set_index(keys + ["seed"])
    out = []
    for key, g in rows[rows["method"] != reference].groupby(["dataset", "epsilon", "mix", "method", "prevalence"],
                                                            sort=False):
        own = g.set_index(keys + ["seed"])
        same = base.reindex(own.index)
        row = {"dataset": key[0], "epsilon": key[1], "method": key[3], "mix": key[2], "prevalence": key[4],
               "main": bool(g["main"].iloc[0]), "seeds": int(g["seed"].nunique())}
        for measure, prefix in (("violation_rate", "violation_diff"), ("dismissed_mean", "dismissed_diff")):
            d = own[measure] - same[measure]
            row.update({**spread(d, prefix), **signs(d, prefix)})
        out.append(row)
    return adjust(pd.DataFrame(out), ["violation_diff", "dismissed_diff"])


def e1_claim_cells(folders: dict) -> pd.DataFrame:
    """the windows of FORGE's claim in every seed of one experiment of RQ1, the rows of the claims table of offline and
    online FORGE on the reference and S2 windows inside the claim at the prevalences up to pi_max, the cells the run's
    summary reads its largest violation share from, each with its seed and run and the dataset, epsilon, delta and
    pi_max of the run."""
    parts = []
    for seed, folder in sorted(folders.items()):
        summary = runs.load_json(folder, "summary")
        claims = runs.load_table(folder, "claims")
        keep = (claims["method"].isin(FORGE) & claims["arm"].isin(CLAIM_ARMS)
                & (claims["prevalence"] <= summary["pi_max"]))
        if "in_claim" in claims:
            keep &= claims["in_claim"].astype(bool)
        columns = (["arm", "method", "mix"] + [c for c in WINDOW_COLUMNS if c in claims]
                   + ["prevalence", "draws", "violations", "rate", "low", "high"])
        settings = {k: v for k, v in _settings(summary).items() if k != "draws"}
        parts.append(claims.loc[keep, columns].assign(**settings, seed=int(seed), run_id=Path(folder).name))
    if not parts:
        raise ValueError("no run folder to read the claim of FORGE from")
    return pd.concat(parts, ignore_index=True)


def e1_violation_by_seed(cells: pd.DataFrame) -> pd.DataFrame:
    """the largest violation share of a window of FORGE's claim per dataset, epsilon, arm, method, mix and seed, with
    its window, its prevalence and its two sided 95% Clopper-Pearson interval over the draws of the seed, and whether
    the lower end of that interval lies above delta, which would put the window's share above delta in that seed."""
    keys = ["dataset", "epsilon", "arm", "method", "mix", "seed"]
    worst = cells.loc[cells.groupby(keys, sort=False)["rate"].idxmax()]
    labels = [c for c in WINDOW_COLUMNS if c in worst]
    t = worst[keys + ["run_id"] + labels + ["prevalence", "draws", "violations", "rate", "low", "high", "delta"]]
    t = t.reset_index(drop=True)
    t["above_delta"] = t["low"] > t["delta"]
    return _whole(t)


def e1_violation_over_seeds(cells: pd.DataFrame) -> pd.DataFrame:
    """the violation share of FORGE's claim over seeds per dataset, epsilon, arm, method and mix. the seeds whose
    largest window share lies above delta, by the lower end of its interval and by its value, the median, range and
    extremes of that largest share over seeds, and the window and prevalence whose share pooled over the draws of every
    seed is the largest, with its two sided 95% Clopper-Pearson interval and the one sided binomial p value of its
    violations against delta, small only where the share lies above delta in some seed. the p values of every window
    and prevalence of the table are adjusted together by Holm, so the largest share is read with the search over the
    windows it came from."""
    group = ["dataset", "epsilon", "arm", "method", "mix"]
    labels = [c for c in WINDOW_COLUMNS if c in cells and c != "window"]
    pooled = cells.groupby(group + ["window", "prevalence"], sort=False).agg(
        **{c: (c, "first") for c in labels}, delta=("delta", "first"), seeds=("seed", "nunique"),
        violations=("violations", "sum"), draws=("draws", "sum")).reset_index()
    pooled = pooled.assign(**interval(pooled["violations"], pooled["draws"]))
    # the chance of at least this many violations in the draws of every seed were each seed's share delta
    pooled["p"] = binom.sf(pooled["violations"] - 1, pooled["draws"], pooled["delta"])
    pooled["p_holm"] = holm(pooled["p"])
    by_seed = dict(list(e1_violation_by_seed(cells).groupby(group, sort=False)))
    out = []
    for key, g in pooled.groupby(group, sort=False):
        top = g.loc[g["rate"].idxmax()]
        largest = by_seed[key]
        out.append({**dict(zip(group, key)), "delta": top["delta"], "seeds": int(largest["seed"].nunique()),
                    "cells": len(g), "seeds_above_delta": int(largest["above_delta"].sum()),
                    "seeds_rate_above_delta": int((largest["rate"] > largest["delta"]).sum()),
                    **spread(largest["rate"], "largest"),
                    **{c: top[c] for c in ["window", *labels, "prevalence", "violations", "draws", "rate", "low",
                                           "high", "p", "p_holm"]}})
    return _whole(pd.DataFrame(out))


def e3_seed_rows(folders: dict, dataset: str) -> pd.DataFrame:
    """the claims table of every seed of one experiment of RQ2, folders mapping a seed to its run folder, each row with
    its seed and run and the dataset the thresholds come from."""
    parts = [runs.load_table(folder, "claims").assign(dataset=dataset, seed=int(seed), run_id=Path(folder).name)
             for seed, folder in sorted(folders.items())]
    if not parts:
        raise ValueError(f"no run folder of {dataset} to read the claims of E3 from")
    return pd.concat(parts, ignore_index=True)


def e3_paired_arms(rows: pd.DataFrame, operating: dict | None = None) -> pd.DataFrame:
    """the operating rule of each dataset against every other arm per stream, policy and composition, from the claims
    of every seed, the mean attacks let through and the mean share of the stream dismissed under the other arm less
    those under the operating rule in the same seed, both on the same draws of the stream."""
    operating = OPERATING_ARMS if operating is None else operating
    keys = [k for k in ("dataset", "stream", "policy", "composition") if k in rows]
    out = []
    for key, g in rows.groupby(keys, sort=False):
        row_key = dict(zip(keys, key))
        rule = operating[row_key["dataset"]]
        own = g[g["arm"] == rule].set_index("seed")
        if own.empty:
            continue
        for arm, other in g[g["arm"] != rule].groupby("arm", sort=False):
            other = other.set_index("seed")
            row = {**row_key, "prevalence": g["prevalence"].iloc[0], "operating": rule, "arm": arm,
                   "seeds": len(other)}
            for measure, prefix in (("leak_mean", "leak_diff"), ("dismissed_share_mean", "dismissed_diff")):
                d = other[measure] - own[measure].reindex(other.index)
                row.update({**spread(d, prefix), **signs(d, prefix)})
            out.append(row)
    return adjust(pd.DataFrame(out), ["leak_diff", "dismissed_diff"])


def e1_intervals(rows: pd.DataFrame) -> pd.DataFrame:
    """the two sided 95% Clopper-Pearson interval of every share of the E1 table over seeds, from the E1 rows of every
    seed with its draws as the units, the count of a seed being its share times its draws, so the violations of a
    draw are the share of its windows that break."""
    t = rows.copy()
    shares = [s for s in E1_SHARES if s in t]
    for s in shares:
        t[f"{s}_count"] = t[s] * t["draws"]
    totals = t.groupby(E1_KEYS, sort=False)[["draws"] + [f"{s}_count" for s in shares]].agg(_total)
    out = totals.reset_index()[E1_KEYS]
    for s in shares:
        got = interval(totals[f"{s}_count"], totals["draws"])
        out[f"{s}_ci_low"], out[f"{s}_ci_high"] = got["low"], got["high"]
    return out


def e3_intervals(rows: pd.DataFrame) -> pd.DataFrame:
    """every share of the E3 table pooled over seeds with its two sided 95% Clopper-Pearson interval, per dataset,
    stream, policy, composition and arm, from the claims of every seed with the units each share is taken of, the
    draws, the sequences of every draw, or the alarms for the share of alarms above the bound. the pooled share comes
    from the same counts as its interval, so the share of alarms above the bound is pooled over the alarms of every
    seed, where pooling by draws counts a seed without alarms as one with no alarm above its bound, and a share no seed
    has, the miscoverage of an arm without a sequence, stays missing."""
    t = rows.copy()
    draws = t["draws"].astype(float)
    units = {"draws": draws,
             "sequences": draws * t["sequences"] if "sequences" in t else pd.Series(np.nan, index=t.index),
             "alarms": np.rint(t["alarm_rate"] * draws)}
    shares = [s for s in E3_SHARES if s in t]
    for s in shares:
        n = units[E3_SHARES[s]]
        # a run takes the share of alarms over one alarm at least, so a seed without alarms adds no count
        t[f"{s}_count"] = np.rint(t[s] * (np.maximum(n, 1) if E3_SHARES[s] == "alarms" else n))
        t[f"{s}_units"] = n
    keys = [k for k in E3_KEYS if k in t]
    totals = t.groupby(keys, sort=False)[[f"{s}_{x}" for s in shares for x in ("count", "units")]].agg(_total)
    out = totals.reset_index()[keys]
    for s in shares:
        got = interval(totals[f"{s}_count"], totals[f"{s}_units"])
        out[f"{s}_pooled"], out[f"{s}_ci_low"], out[f"{s}_ci_high"] = got["rate"], got["low"], got["high"]
    return out


def with_intervals(table: pd.DataFrame, intervals: pd.DataFrame, keys) -> pd.DataFrame:
    """the table with the interval of every pooled share beside it, <share>_ci_low and <share>_ci_high after
    <share>_pooled, matched on the keys both tables hold. a pooled share that intervals holds as well is taken from
    there."""
    keys = [k for k in keys if k in table and k in intervals]
    extra = [c for c in intervals if c not in keys]
    merged = table.drop(columns=[c for c in extra if c in table]).merge(intervals[keys + extra], on=keys, how="left",
                                                                         validate="many_to_one")
    order = []
    for col in table.columns:
        order.append(col)
        if col.endswith("_pooled"):
            share = col[:-len("_pooled")]
            order += [c for c in (f"{share}_ci_low", f"{share}_ci_high") if c in merged]
    return merged[order + [c for c in merged if c not in order]]
