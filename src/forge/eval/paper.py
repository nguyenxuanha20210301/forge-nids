"""the tables of the paper, cut from the pooled tables of the experiments over seeds and from the run folders.

every table a paper figure draws is written first, so a figure is redrawn from the written tables alone. the pooled
tables come from forge.eval.seeds, one folder per experiment, and E7 reads the run folders of every seed, since its
workload is computed per run.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from forge import paths, runs
from forge.eval import seeds, workload

# the pooled folder of each experiment the paper reads, under pooled/ of the storage root
EXPERIMENTS = {"e11": "e11_anoshift_prevalence", "e12": "e12_litnet_prevalence", "e21": "e21_monitor_streams",
               "e22": "e22_monitor_sweep", "e23": "e23_unseen_family", "e31": "e31_anoshift_scorers",
               "e32": "e32_litnet_scorers"}
DATASETS = {"e11": "anoshift", "e12": "litnet2020", "e21": "anoshift", "e22": "anoshift", "e23": "litnet2020",
            "e31": "anoshift", "e32": "litnet2020"}
# the columns of the pooled tables that reach the paper, in their median, low, high and pooled forms
E1_COLUMNS = ["violation_rate", "violation_window_max", "refused", "dismissed_mean", "for_median",
              "best_up_to_pi_max_dismissed", "best_at_prevalence_dismissed", "best_every_mix_dismissed",
              "realized_rate"]
E2_COLUMNS = ["rate", "refused", "dismissed_mean", "for_median", "best_up_to_pi_max_dismissed",
              "best_every_mix_dismissed", "realized_rate"]
E5_COLUMNS = ["dismissed_mean", "refused", "best_every_mix_dismissed", "violation_rate", "violation_window_max",
              "for_median"]
E6_COLUMNS = ["alarm_in_unseen_rate", "alarm_before_unseen_rate", "alarm_rate", "false_alarm_rate",
              "unseen_violated_rate", "detect_sessions_median", "unseen_leak_median", "unseen_attacks_median",
              "leak_median", "unseen_for_true_mean", "unseen_miss_mean", "miscover_rate", "over_bound_rate"]
E7_COLUMNS = ["dismissed", "violation_rate", "refused", "escalations_per_million", "audits_per_million",
              "workload_per_million", "leaked_per_million", "leak_median", "realized_for_median", "alarm_rate",
              "excess_median"]
# the stream the paper reads where a table holds both kinds, the replay on the real order
PAPER_STREAM = "replay"


def paper_rows(table: pd.DataFrame) -> pd.DataFrame:
    """the rows of the stream the paper reads, per dataset where the table has one. a dataset whose rows carry the
    replay keeps those rows only, one without it keeps its rows as they are, and a table without streams is returned
    whole."""
    if "stream" not in table:
        return table
    groups = table.groupby("dataset", sort=False) if "dataset" in table else [(None, table)]
    parts = []
    for _, g in groups:
        has = g["stream"] == PAPER_STREAM
        parts.append(g[has] if has.any() else g)
    return pd.concat(parts) if parts else table


def read_pooled(folder) -> dict:
    """the pooled tables of one experiment by name, with the settings every seed shares read from the summaries by
    seed, since the pooled folder holds no summary of its own."""
    folder = Path(folder)
    tables = {p.stem: pd.read_csv(p) for p in sorted(folder.glob("*.csv"))}
    if not tables:
        raise FileNotFoundError(f"no pooled table in {folder}")
    return tables


def settings(tables: dict, keys) -> dict:
    # the scalar settings shared by every seed, from the summaries by seed, a key that moves with the seed stops it
    by_seed = tables["summary_by_seed"]
    out = {}
    for key in keys:
        if key not in by_seed:
            continue
        values = by_seed[key].dropna().unique()
        if len(values) > 1:
            raise ValueError(f"{key} differs between the seeds, {values.tolist()}")
        out[key] = values[0].item() if hasattr(values[0], "item") else values[0]
    return out


def _keep(table: pd.DataFrame, first: list[str], columns) -> pd.DataFrame:
    return table[[c for c in first + seeds._stats(table, columns) if c in table]].reset_index(drop=True)


def e1_table(pooled: dict) -> pd.DataFrame:
    """E1, the FOR, the violation share and the dismissed share of every method against the prevalence on the windows
    RQ1 is read on, both datasets in one table, every mix."""
    parts = []
    for name in ("e11", "e12"):
        if name not in pooled or "e1_for_by_prevalence" not in pooled[name]:
            continue
        t = pooled[name]["e1_for_by_prevalence"].copy()
        t.insert(0, "dataset", DATASETS[name])
        parts.append(_keep(t, ["dataset", "method", "mix", "prevalence", "main", "windows", "seeds"], E1_COLUMNS))
    if not parts:
        raise ValueError("no E1 table in the pooled folders of e11 and e12")
    return pd.concat(parts, ignore_index=True)


def e1_rows(table: pd.DataFrame, mix: str = "window") -> pd.DataFrame:
    """the E1 table at one mix with the columns the figure over prevalences reads, the median over seeds as the
    value, the 5% and 95% quantiles over seeds as its range, and the share of pairs with FOR above epsilon pooled
    over the draws of every seed with the range of the per seed shares."""
    t = table[table["mix"] == mix].copy()
    names = {"for_median_median": "for_median", "for_median_low": "for_low", "for_median_high": "for_high",
             "violation_rate_pooled": "violation_rate", "violation_rate_low": "violation_window_min",
             "violation_rate_high": "violation_window_max", "dismissed_mean_median": "dismissed_mean",
             "dismissed_mean_low": "dismissed_low", "dismissed_mean_high": "dismissed_high"}
    for stick in ("best_up_to_pi_max", "best_at_prevalence", "best_every_mix"):
        names[f"{stick}_dismissed_median"] = f"{stick}_dismissed"
    # a column the seed never moved kept its plain name in the pooled table and already carries it
    return t.rename(columns={k: v for k, v in names.items() if k in t})


def e2_tables(pooled: dict) -> dict:
    """E2, the violation share of every method on the window of each family of LITNET at the main prevalence, per
    arm, and the real windows at their own prevalence."""
    out = {}
    if "e12" in pooled and "e2_by_family" in pooled["e12"]:
        out["e2_by_family"] = _keep(pooled["e12"]["e2_by_family"], ["arm", "window", "segment", "method", "in_claim",
                                                                  "few_attacks", "window_sessions", "seeds"],
                                    E2_COLUMNS)
    if "e12" in pooled and "e2_natural" in pooled["e12"]:
        out["e2_natural"] = _keep(pooled["e12"]["e2_natural"], ["window", "segment", "method", "prevalence",
                                                              "up_to_pi_max", "in_claim", "window_sessions", "seeds"],
                                  E2_COLUMNS)
    return out


def e5_tables(pooled: dict) -> dict:
    """E5, the certified share of every scorer at the main epsilon with its diagnostics, and the certified share by
    epsilon on the reference and the real windows, both datasets in one table each."""
    scorer_parts, epsilon_parts = [], []
    for name in ("e31", "e32"):
        if name not in pooled:
            continue
        if "e5_by_scorer" in pooled[name]:
            t = pooled[name]["e5_by_scorer"].copy()
            t.insert(0, "dataset", DATASETS[name])
            scorer_parts.append(t)
        if "by_epsilon" in pooled[name]:
            t = pooled[name]["by_epsilon"].copy()
            t.insert(0, "dataset", DATASETS[name])
            epsilon_parts.append(_keep(t, ["dataset", "scorer", "kind", "epsilon", "arm", "subset", "windows",
                                           "seeds"], E5_COLUMNS))
    out = {}
    if scorer_parts:
        out["e5_by_scorer"] = pd.concat(scorer_parts, ignore_index=True)
    if epsilon_parts:
        out["e5_by_epsilon"] = pd.concat(epsilon_parts, ignore_index=True)
    return out


def e6_tables(pooled: dict) -> dict:
    """E6, per held out family the alarm, the leakage and the miss rate under every arm and policy, and the AUC, the
    benign share and the miss rate of its window with the family unseen against known."""
    out = {}
    if "e23" in pooled and "e6_by_family" in pooled["e23"]:
        out["e6_by_family"] = _keep(pooled["e23"]["e6_by_family"], ["stream", "unit", "case", "composition", "policy",
                                                                  "arm", "unseen_segment", "unseen_sessions",
                                                                  "seeds"], E6_COLUMNS)
    if "e23" in pooled and "e6_windows" in pooled["e23"]:
        out["e6_windows"] = pooled["e23"]["e6_windows"].copy()
    return out


def e3_tables(pooled: dict) -> dict:
    """E3 and E4, the alarm, the leakage, the bound and the coverage of every arm on the real streams of AnoShift
    per policy and composition over seeds, and the same per cell of the sweep."""
    out = {}
    for name, table in [("e21", "e3_by_policy"), ("e22", "e4_by_cell")]:
        if name in pooled and table in pooled[name]:
            t = pooled[name][table].copy()
            t.insert(0, "dataset", DATASETS[name])
            out[table] = t
    return out


def run_folders(tables: dict, experiment: str, root=None) -> dict:
    """the run folder of every seed behind a pooled folder, from its list of runs, under the runs folder of the
    storage root unless root names another one. a run that is not there is left out with a line."""
    base = (Path(root) if root else paths.runs_dir()) / experiment
    found = {}
    for r in tables["runs"].itertuples():
        folder = base / r.run_id
        if (folder / "results").is_dir():
            found[int(r.seed)] = folder
        else:
            print(f"{experiment} run {r.run_id} of seed {r.seed} is not in the runs folder, left out")
    return found


def e7_certification(folders: dict, audit_rate: float) -> pd.DataFrame:
    """E7 at the certification stage over seeds, the workload of every method on the RQ1 windows against its
    violation share, folders mapping a seed to an e11 or e12 run folder. a method is safe when its violation share
    over the draws of every seed stays within delta, which every seed shares."""
    frames = {seed: workload.certification(folder, audit_rate) for seed, folder in folders.items()}
    delta = runs.load_json(next(iter(folders.values())), "summary")["delta"]
    keys = ["dataset", "stage", "policy", "mix", "prevalence", "main"]
    pooled = seeds.pool(frames, keys)
    t = uniform(_keep(pooled, keys + ["seeds"], E7_COLUMNS), E7_COLUMNS)
    # every seed holds the same draw and window pairs, so the share over all of them is the mean over seeds
    both = pd.concat([f.assign(seed=s) for s, f in frames.items()], ignore_index=True)
    mean = both.groupby(keys, sort=False)["violation_rate"].mean().rename("violation_rate_pooled").reset_index()
    t = t.merge(mean, on=keys, how="left")
    t["delta"] = delta
    t["safe"] = t["violation_rate_pooled"] <= delta
    return t


def e7_monitoring(folders: dict) -> pd.DataFrame:
    """E7 at the monitoring stage over seeds, the workload, the audits and the attacks let through of every policy
    under each arm on the real streams, folders mapping a seed to an e21 or e23 run folder."""
    frames = {seed: workload.monitoring(folder) for seed, folder in folders.items()}
    keys = ["dataset", "stage", "stream", "composition", "policy", "arm"]
    pooled = seeds.pool(frames, keys)
    return uniform(_keep(pooled, keys + ["sessions", "seeds"], E7_COLUMNS), E7_COLUMNS)


def uniform(table: pd.DataFrame, columns) -> pd.DataFrame:
    """gives every column its median, low and high form, so tables pooled over one seed, where no column moves and
    every one keeps its plain name, line up with tables pooled over many."""
    t = table.copy()
    for col in columns:
        if col in t and f"{col}_median" not in t:
            for suffix in ("_median", "_low", "_high"):
                t[f"{col}{suffix}"] = t[col]
            t = t.drop(columns=[col])
    return t


def _value(table: pd.DataFrame, column: str) -> pd.Series:
    # the pooled share of a rate, else its median, else the column itself when no seed moved it
    for name in (f"{column}_pooled", f"{column}_median", column):
        if name in table:
            return table[name]
    raise KeyError(column)


def e7_frontier(certification: pd.DataFrame) -> pd.DataFrame:
    """the methods no other method beats on both the workload and the violation share at the main prevalence with the
    window's own mix, per dataset, the violation share over the draws of every seed as the risk and the median
    workload as the cost."""
    shown = certification[(certification["mix"] == "window") & (certification["main"] == 1)]
    shown = shown[~shown["policy"].isin(dict(workload.YARDSTICKS))].copy()
    shown["risk"] = _value(shown, "violation_rate")
    shown["cost"] = _value(shown, "workload_per_million")
    parts = [workload.frontier(g, "cost", "risk") for _, g in shown.groupby("dataset")]
    return pd.concat(parts, ignore_index=True) if parts else shown.iloc[:0]


def scalar(table: pd.DataFrame, column: str, default=np.nan):
    # the one value of a column every row shares, or the default when the column is missing
    if column not in table or table[column].dropna().empty:
        return default
    return table[column].dropna().iloc[0]
