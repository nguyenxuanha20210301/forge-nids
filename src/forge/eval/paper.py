"""the tables of the paper, cut from the pooled tables of the experiments over seeds and from the run folders.

every table a paper figure draws is written first, so a figure is redrawn from the written tables alone. the pooled
tables come from forge.eval.seeds, one folder per experiment, and E7 reads the run folders of every seed, since its
workload is computed per run.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from forge import paths, runs
from forge.certify import certificate
from forge.certify.bounds import clopper_pearson_upper, dkw_margin
from forge.certify.certificate import for_from_rates
from forge.data import ugr16
from forge.eval import seeds, workload

# the pooled folder of each experiment the paper reads, under pooled/ of the storage root. e13_floor is e13 at the
# paper's epsilon, where the certificate refuses, beside the operating target of UGR16
EXPERIMENTS = {"e11": "e11_anoshift_prevalence", "e12": "e12_litnet_prevalence", "e13": "e13_ugr16_prevalence",
               "e13_floor": "e13_ugr16_floor", "e21": "e21_monitor_streams", "e22": "e22_monitor_sweep",
               "e23": "e23_unseen_family", "e24": "e24_ugr16_streams", "e31": "e31_anoshift_scorers",
               "e32": "e32_litnet_scorers"}
DATASETS = {"e11": "anoshift", "e12": "litnet2020", "e13": "ugr16", "e13_floor": "ugr16", "e21": "anoshift",
            "e22": "anoshift", "e23": "litnet2020", "e24": "ugr16", "e31": "anoshift", "e32": "litnet2020"}
# the experiments whose rows make the E1 table, each row carrying the epsilon of its experiment
E1_EXPERIMENTS = ("e11", "e12", "e13", "e13_floor")
# the columns of the pooled tables that reach the paper, in their median, low, high and pooled forms
E1_COLUMNS = ["violation_rate", "violation_window_max", "refused", "dismissed_mean", "for_median",
              "best_up_to_pi_max_dismissed", "best_at_prevalence_dismissed", "best_every_mix_dismissed",
              "realized_rate", "realized_for_mean", "miss_mean"]
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
    RQ1 is read on, the three datasets in one table, every mix, each row with the epsilon of its experiment, so UGR16
    at its operating target and at the paper's epsilon stand apart."""
    parts = []
    for name in E1_EXPERIMENTS:
        if name not in pooled or "e1_for_by_prevalence" not in pooled[name]:
            continue
        t = uniform(pooled[name]["e1_for_by_prevalence"], E1_COLUMNS)
        t.insert(0, "dataset", DATASETS[name])
        t.insert(1, "epsilon", settings(pooled[name], ["epsilon"])["epsilon"])
        # a share the seed never moved has no pooled form, and its one value is the share over every seed
        for col in E1_COLUMNS:
            if seeds._is_rate(col) and f"{col}_median" in t and f"{col}_pooled" not in t:
                t[f"{col}_pooled"] = t[f"{col}_median"]
        parts.append(_keep(t, ["dataset", "epsilon", "method", "mix", "prevalence", "main", "windows", "seeds"],
                           E1_COLUMNS))
    if not parts:
        raise ValueError("no E1 table in the pooled folders of " + ", ".join(EXPERIMENTS[n] for n in E1_EXPERIMENTS))
    return pd.concat(parts, ignore_index=True)


def e1_rows(table: pd.DataFrame, mix: str = "window", epsilon: float | None = None) -> pd.DataFrame:
    """the E1 table at one mix, and at one epsilon when given, with the columns the figure over prevalences reads,
    the median over seeds as the value, the 5% and 95% quantiles over seeds as its range, and the share of pairs with
    FOR above epsilon pooled over the draws of every seed with the range of the per seed shares."""
    t = table[table["mix"] == mix]
    if epsilon is not None:
        t = t[np.isclose(t["epsilon"], epsilon)]
    t = t.copy()
    names = {"for_median_median": "for_median", "for_median_low": "for_low", "for_median_high": "for_high",
             "violation_rate_pooled": "violation_rate", "violation_rate_low": "violation_window_min",
             "violation_rate_high": "violation_window_max", "dismissed_mean_median": "dismissed_mean",
             "dismissed_mean_low": "dismissed_low", "dismissed_mean_high": "dismissed_high"}
    for stick in ("best_up_to_pi_max", "best_at_prevalence", "best_every_mix"):
        names[f"{stick}_dismissed_median"] = f"{stick}_dismissed"
    # a column the seed never moved kept its plain name in the pooled table and already carries it
    return t.rename(columns={k: v for k, v in names.items() if k in t})


def e2_tables(pooled: dict) -> dict:
    """E2, the violation share of every method on the window of each family of LITNET and on each window of six
    hours of UGR16 at the main prevalence, per arm, and the real windows of both at their own prevalence, one table
    each. a window of LITNET is named by its segment and one of UGR16 by its start."""
    out = {}
    for table, first in [("e2_by_family", ["dataset", "arm", "window", "segment", "start", "method", "in_claim",
                                           "few_attacks", "window_sessions", "seeds"]),
                         ("e2_natural", ["dataset", "window", "segment", "start", "method", "prevalence",
                                         "up_to_pi_max", "in_claim", "window_sessions", "seeds"])]:
        parts = []
        for name in ("e12", "e13"):
            if name in pooled and table in pooled[name]:
                t = uniform(pooled[name][table], E2_COLUMNS)
                t.insert(0, "dataset", DATASETS[name])
                parts.append(_keep(t, first, E2_COLUMNS))
        if parts:
            out[table] = pd.concat(parts, ignore_index=True)
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
    and UGR16 per policy and composition over seeds, both datasets in one table, and the same per cell of the
    sweep."""
    out = {}
    for name, table in [("e21", "e3_by_policy"), ("e24", "e3_by_policy"), ("e22", "e4_by_cell")]:
        if name in pooled and table in pooled[name]:
            t = uniform(pooled[name][table], seeds.E3_COLUMNS if table == "e3_by_policy" else seeds.E4_COLUMNS)
            t.insert(0, "dataset", DATASETS[name])
            out[table] = pd.concat([out[table], t], ignore_index=True) if table in out else t
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
    under each arm on the real streams, folders mapping a seed to an e21, e23 or e24 run folder."""
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


def e3_bounds_over_time(folders: dict) -> pd.DataFrame:
    """the trace of one run of e21 or e24 per dataset, folders mapping a dataset to its run folder, on the stream the
    paper reads, for the policy and the composition the config names for the trace, at its first draw, under the one
    sequence over the whole stream. two kinds of rows, the audited attacks, where the bounds move, with the true count
    and both bounds there, and the window ends with the true count and the period the restart rule of the paper
    follows, the year on AnoShift and the week on UGR16. epsilon, the end of the stream and the alarm come as
    columns, the alarm -1 where the sequence never fired."""
    parts = []
    for dataset, folder in folders.items():
        folder = Path(folder)
        arrays = runs.load_arrays(folder, "trace")
        summary = runs.load_json(folder, "summary")
        found = sorted({key.split("|", 1)[0] for key in arrays if "|" in key})
        stream = PAPER_STREAM if PAPER_STREAM in found else (found[0] if found else None)
        t = {k.split("|", 1)[1]: v for k, v in arrays.items() if k.startswith(f"{stream}|")} if stream else arrays
        period = t["years"] if "years" in t else t.get("segments", np.zeros(len(t["ends"]), dtype=np.int64))
        audits = pd.DataFrame({"point": "audit", "position": t["ev_pos"], "attacks": t["ev_count"],
                               "lower": t["lcb"], "upper": t["ucb"]})
        ends = pd.DataFrame({"point": "window", "position": t["ends"], "attacks": t["cum_attacks"],
                             "window": np.arange(len(t["ends"])), "period": period})
        table = pd.concat([audits, ends], ignore_index=True)
        trace = summary.get("trace", {})
        lead = {"dataset": dataset, "run_id": folder.name, "seed": runs.load_run_config(folder).get("seed", 0),
                "stream": stream or "", "policy": trace.get("policy"), "composition": trace.get("composition")}
        for k, (name, value) in enumerate(lead.items()):
            table.insert(k, name, value)
        table["epsilon"] = summary["epsilon"]
        table["n_end"] = int(t["n_end"])
        table["alarm"] = int(t["alarm_pos"])
        parts.append(table)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# the case study of the paper, the certificate of seed 0 on the calibration part of UGR16 and its stream through the
# deployment by the clock, the policy and the composition the case study reads
CASE_POLICY = "forge_simplex"
CASE_COMPOSITION = "natural"
CASE_ARMS = ("monitor", "monitor_week", "monitor_window", "any_start", "none", "window_audit", "naive")


def case_study_certificate(folder, streams_folder=None) -> pd.DataFrame:
    """the certificate of one e13 run as an operator reads it, recomputed on the expected calibration set from the
    grid and the summary of the run with the certificate's own functions, one row per item. the terms of the
    contract, the calibration part, the start, the grid, the certified threshold with its score, its benign quantile,
    its lower benign bound and its floor, then per family the expected calibration attacks, the misses at the
    threshold, the miss rate, its upper bound and the worst case FOR at pi_max it gives, and the bound of the
    certificate against epsilon. streams_folder names the e24 run that read the certificate, whose threshold index is
    checked against the recomputed one."""
    folder = Path(folder)
    summary = runs.load_json(folder, "summary")
    grid = runs.load_table(folder, "grid")
    roles = runs.load_table(folder, "roles").set_index("role")
    eps, delta, pi_max = float(summary["epsilon"]), float(summary["delta"]), float(summary["pi_max"])
    part_b = float(summary["delta_benign"])
    part_m = delta - part_b
    names = ["+".join(g) for g in summary["families"]]
    expected = np.array([sum(float(summary["calibration_expected_attacks"][f]) for f in g)
                         for g in summary["families"]])
    n_benign = float(roles.loc["calibration", "benign"])
    b_cal = grid["calibration_benign"].to_numpy(dtype=float)
    m_cal = np.stack([sum(grid[f"calibration_{f}"].to_numpy(dtype=float) * summary["calibration_expected_attacks"][f]
                          for f in g) / e for g, e in zip(summary["families"], expected)])
    start = certificate.start_settings(summary)
    out = certificate.certify((b_cal * n_benign)[None], np.array([n_benign]), (m_cal * expected[:, None])[None],
                              expected[None], eps, delta, pi_max, delta_benign=part_b, **start)
    index = int(out["index"][0])
    if index < 0:
        raise ValueError(f"the certificate of {folder.name} refuses on its expected calibration set")
    if streams_folder is not None:
        by_window = runs.load_table(streams_folder, "by_window")
        read = by_window.loc[(by_window["policy"] == CASE_POLICY) & by_window["threshold_mean"].notna(),
                             "threshold_mean"]
        if len(read) and not np.allclose(read, index):
            raise ValueError(f"{Path(streams_folder).name} read the threshold {read.iloc[0]} of {CASE_POLICY}, "
                             f"the certificate recomputed from {folder.name} gives {index}")
    margin = float(dkw_margin(n_benign, part_b))
    b_lower = max(b_cal[index] - margin, 0.0)
    # the threshold after the certified one, where the walk stopped, with the family whose bound stopped it
    uppers = np.stack([clopper_pearson_upper(m_cal[k] * expected[k], expected[k], part_m) for k in range(len(names))])
    bounds = for_from_rates(pi_max, uppers.max(0), np.clip(b_cal - margin, 0, None))
    after = index + 1 if index + 1 < len(grid) else index
    rows = [("epsilon", eps, ""), ("delta", delta, ""), ("delta_benign", part_b, ""), ("delta_family", part_m, ""),
            ("pi_max", pi_max, ""),
            ("calibration_start", roles.loc["calibration", "start"], "clock"),
            ("calibration_end", roles.loc["calibration", "end"], "clock"),
            ("calibration_flows", float(roles.loc["calibration", "flows"]), "flows"),
            ("calibration_benign", n_benign, "flows"),
            ("calibration_prevalence", float(summary["calibration_prevalence"]), "share"),
            ("calibration_set_flows", float(summary["calibration_sessions"]), "flows"),
            ("families", float(len(names)), "count"),
            ("family_minimum", float(summary["family_minimum"]), "attacks"),
            ("grid_thresholds", float(len(grid)), "count"),
            ("start_rule", summary.get("start", "feasible"), ""),
            ("start_share", float(summary.get("start_share", np.nan)), "share"),
            ("certified_index", float(index), "index"),
            ("certified_score", float(grid["threshold"].iloc[index]), "score"),
            ("certified_benign_quantile", float(b_cal[index]), "share"),
            ("benign_margin", margin, "share"),
            ("benign_lower_bound", b_lower, "share"),
            ("floor", (1 - pi_max) * b_lower, "share"),
            ("bound", float(out["bound"][0]), "share"),
            ("next_index", float(after), "index"),
            ("next_benign_quantile", float(b_cal[after]), "share"),
            ("next_bound", float(bounds[after]), "share"),
            ("next_binding_family", names[int(np.argmax(uppers[:, after]))], ""),
            ("next_binding_miss_rate", float(m_cal[int(np.argmax(uppers[:, after])), after]), "share"),
            ("refused_share", float(summary["offline_refused"][CASE_POLICY]), "share"),
            ("selection_auc", float(summary["selection_auc"]), ""),
            ("calibration_auc", float(summary["calibration_auc"]), "")]
    for k, name in enumerate(names):
        misses = float(m_cal[k, index] * expected[k])
        upper = float(clopper_pearson_upper(misses, expected[k], part_m))
        rows += [(f"{name}_attacks", expected[k], "attacks"), (f"{name}_misses", misses, "attacks"),
                 (f"{name}_miss_rate", float(m_cal[k, index]), "share"), (f"{name}_miss_upper", upper, "share"),
                 (f"{name}_for_at_pi_max", float(for_from_rates(pi_max, upper, b_lower)), "share")]
    table = pd.DataFrame(rows, columns=["item", "value", "unit"])
    table.insert(0, "run_id", folder.name)
    table.insert(1, "seed", int(runs.load_run_config(folder).get("seed", 0)))
    return table


def case_study_windows(folder, streams_folder) -> pd.DataFrame:
    """the deployment of the case study window by window, the attacks per family, the natural prevalence and the
    AUC of every window with the benign share dismissed and the miss rate per family at the certified threshold
    from the e13 run, and from the e24 run on the replay of the real order under the policy and
    at the composition of the case study the share dismissed, the FOR at the certified threshold, the attacks among the
    dismissed flows, the realized FOR and the monitor's state at the window's end under one sequence over the
    deployment, with the week of the operating rule."""
    windows = runs.load_table(folder, "windows")
    by_window = runs.load_table(streams_folder, "by_window")
    stream = PAPER_STREAM if PAPER_STREAM in set(by_window["stream"]) else sorted(set(by_window["stream"]))[0]
    rows = by_window[(by_window["stream"] == stream) & (by_window["policy"] == CASE_POLICY)
                     & (by_window["composition"] == CASE_COMPOSITION)].sort_values("window")
    keep = ["window", "segment", "window_sessions", "threshold_mean", "dismissed_share_mean", "for_true_mean",
            "violated_rate", "attacks_mean", "realized_for_mean", "state_certified", "state_running",
            "state_violation", "window_audit_alarm_rate"]
    families = [c for c in windows.columns if c in ugr16.ATTACK_FAMILIES]
    misses = [c for c in windows.columns if c.startswith("forge_miss_")]
    left = windows[["window", "start", "end", "flows", "benign", *families, "natural_prevalence", "auc",
                    "forge_benign_dismissed", *misses]]
    table = left.merge(rows[keep], on="window", how="left")
    table.insert(0, "stream", stream)
    table.insert(1, "policy", CASE_POLICY)
    table.insert(2, "composition", CASE_COMPOSITION)
    table.insert(3, "certification_run", Path(folder).name)
    table.insert(4, "streams_run", Path(streams_folder).name)
    return table


def case_study_stream(streams_folder) -> pd.DataFrame:
    """the stream of the case study under every monitoring arm, the claims rows of the e24 run on the replay under
    the policy and at the composition of the case study, with the epsilon and the audit rate of the run."""
    claims = runs.load_table(streams_folder, "claims")
    summary = runs.load_json(streams_folder, "summary")
    stream = PAPER_STREAM if PAPER_STREAM in set(claims["stream"]) else sorted(set(claims["stream"]))[0]
    rows = claims[(claims["stream"] == stream) & (claims["policy"] == CASE_POLICY)
                  & (claims["composition"] == CASE_COMPOSITION) & claims["arm"].isin(CASE_ARMS)].copy()
    rows["arm"] = pd.Categorical(rows["arm"], [a for a in CASE_ARMS if a in set(rows["arm"])], ordered=True)
    rows = rows.sort_values("arm").reset_index(drop=True)
    rows["arm"] = rows["arm"].astype(str)
    rows.insert(0, "run_id", Path(streams_folder).name)
    rows["epsilon"] = float(summary["epsilon"])
    rows["audit_rate"] = float(summary["audit_rate"])
    return rows


def case_study_figure_rows(windows: pd.DataFrame, certificate: pd.DataFrame) -> pd.DataFrame:
    # the windows of the case study with the terms of the contract the figure draws as lines, read from the certificate
    terms = certificate.set_index("item")["value"]
    out = windows.copy()
    for item in ("epsilon", "pi_max", "calibration_prevalence"):
        out[item] = float(terms[item])
    return out


# the operating point of the example of proposal 2.2, a miss rate of 1% at 90% of the benign flows dismissed, beside a
# tighter and a looser miss rate, at the paper's epsilon
INTRO_MISS_RATES = (0.001, 0.01, 0.05)
INTRO_BENIGN = 0.9
INTRO_EPSILON = 0.001


def intro_for_by_prevalence(miss_rates=INTRO_MISS_RATES, benign: float = INTRO_BENIGN, epsilon: float = INTRO_EPSILON,
                            prevalences=None) -> pd.DataFrame:
    """the FOR of one threshold against the prevalence, pi m / (pi m + (1 - pi) b), for a few miss rates m at one
    benign share dismissed b, with the largest prevalence at which each keeps FOR at or below epsilon,
    epsilon b / (m (1 - epsilon) + epsilon b), the same at every row of a miss rate."""
    pis = np.geomspace(1e-4, 0.3, 161) if prevalences is None else np.asarray(prevalences, dtype=float)
    rows = []
    for m in miss_rates:
        crossing = epsilon * benign / (m * (1 - epsilon) + epsilon * benign)
        rows += [{"miss_rate": m, "benign_dismissed": benign, "prevalence": float(pi),
                  "for": float(pi * m / (pi * m + (1 - pi) * benign)), "epsilon": epsilon, "crossing": crossing}
                 for pi in pis]
    return pd.DataFrame(rows)
