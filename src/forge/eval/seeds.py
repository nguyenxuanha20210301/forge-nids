"""the runs of one experiment over scorer seeds, pooled into one table per saved table with the median and the 5% to
95% range over seeds of every number, and the tables of E1 to E6 cut from them.

every real data experiment draws its calibration sets and its windows a thousand times inside one run, with one
scorer seed, and proposal 8.7 asks for 20 seeds. a seed is a run of its own, so the runs of an experiment are grouped
by the seed in their saved config, one finished run per seed, the latest, and their configs must agree in everything
but the seed. a column that every seed agrees on row by row, such as the sessions of a window or the family of a
capture, is kept as it is, a number that moves with the seed gets its median and its 5% and 95% quantiles over
seeds, and a share of draws is also pooled over the draws of every seed. an experiment that reads its thresholds
from the run of another experiment, e21 and e22 from e11, names the source run of the same seed in its config, so
the pool ignores that run id and checks the seed of the source run instead.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from forge import paths, runs

LOW, HIGH = 0.05, 0.95
# the tables of each experiment, with the columns that identify a row within one run and the column that weights
# a share of draws when it is pooled over seeds
TABLES = {
    "e11_anoshift_prevalence": {
        "by_prevalence": (["arm", "period", "method", "prevalence", "mix"], "pairs"),
        "claims": (["arm", "window", "year", "block", "period", "method", "prevalence", "mix"], "draws"),
        "windows": (["window", "year", "block", "period"], None),
    },
    "e12_litnet_prevalence": {
        "by_prevalence": (["arm", "subset", "method", "prevalence", "mix"], "pairs"),
        "claims": (["arm", "window", "segment", "method", "prevalence", "mix"], "draws"),
        "natural": (["arm", "window", "segment", "method"], "draws"),
        "windows": (["window", "segment"], None),
    },
    "e23_unseen_family": {
        "claims": (["unit", "case", "composition", "policy", "arm"], "draws"),
        "blocks": (["unit", "case", "composition", "policy", "window", "segment"], None),
        "windows": (["unit", "window", "segment"], None),
        "units": (["unit"], None),
    },
    "e31_anoshift_scorers": {
        "scorers": (["scorer"], None),
        "by_epsilon": (["scorer", "epsilon", "arm", "subset"], "pairs"),
        "claims": (["scorer", "epsilon", "arm", "window", "segment"], "draws"),
        "windows": (["scorer", "window", "segment"], None),
    },
    "e32_litnet_scorers": {
        "scorers": (["scorer"], None),
        "by_epsilon": (["scorer", "epsilon", "arm", "subset"], "pairs"),
        "claims": (["scorer", "epsilon", "arm", "window", "segment"], "draws"),
        "windows": (["scorer", "window", "segment"], None),
    },
    "e21_monitor_streams": {
        "claims": (["stream", "policy", "composition", "arm"], "draws"),
        "blocks": (["stream", "policy", "composition", "window"], None),
        "windows": (["window"], None),
    },
    "e22_monitor_sweep": {
        "claims": (["policy", "composition", "name"], "draws"),
        "cells": (["name"], None),
        "policies": (["rate", "allocation", "floor", "bin"], None),
    },
}
# the experiments whose thresholds come from the source run of the same seed, so their configs differ in the run id
# of the source by design and the pool checks the seed of that run instead
SOURCE_PER_SEED = ("e21_monitor_streams", "e22_monitor_sweep")
SOURCE_KEYS = ("source.run",)
# the columns of the scorers table that stand beside the certified share in the E5 table
E5_SCORER_COLUMNS = ["selection_yardstick", "selection_auc", "calibration_auc", "benign_at_miss_pooled",
                     "benign_at_miss_hardest", "ece"]
# the columns of the E6 table, per unit, case, composition, policy and arm
E6_COLUMNS = ["alarm_rate", "false_alarm_rate", "alarm_in_unseen_rate", "alarm_in_unseen_violated_rate",
              "alarm_before_unseen_rate", "unseen_violated_rate", "detect_sessions_median", "unseen_leak_median",
              "unseen_attacks_median", "leak_median", "unseen_for_true_mean", "unseen_miss_mean", "miscover_rate",
              "miscover_sequence_rate", "over_bound_rate", "bound_median", "windows_certified_mean"]
# the columns of the E3 table, per stream, policy, composition and arm of e21
E3_COLUMNS = ["alarm_rate", "false_alarm_rate", "alarm_block_median", "alarm_session_median", "onset_true_rate",
              "onset_true_block_median", "onset_realized_rate", "leak_median", "leak_mean", "excess_median",
              "sessions_median", "dismissed_share_mean", "realized_for_median", "audits_mean", "bound_median",
              "over_bound_rate", "miscover_rate", "miscover_sequence_rate", "sequences", "blocks_certified_mean",
              "blocks_violation_mean"]
# the columns of the E4 table, per cell, policy and composition of e22
E4_COLUMNS = ["audits_per_million", "alarm_rate", "false_alarm_rate", "alarm_block_median", "alarm_session_median",
              "leak_median", "leak_mean", "leak_in_delay_median", "excess_median", "dismissed_share_mean",
              "bound_median", "over_bound_rate", "miscover_sequence_rate", "sequences", "blocks_certified_mean"]


def flatten(cfg: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def config_differences(configs, ignore=("seed",)) -> dict:
    # the flattened keys on which the configs differ, each with the values seen, apart from the keys in ignore
    flat = [flatten(c) for c in configs]
    out = {}
    for key in sorted(set().union(*(f.keys() for f in flat))):
        if key in ignore:
            continue
        seen = list(dict.fromkeys(repr(f.get(key, "missing")) for f in flat))
        if len(seen) > 1:
            out[key] = seen
    return out


def seed_runs(experiment: str, root=None) -> pd.DataFrame:
    # every run folder of the experiment with its seed, status and commit
    rows = []
    for run_dir in runs.run_dirs(experiment, root):
        meta, cfg = runs.load_meta(run_dir), runs.load_run_config(run_dir)
        rows.append({"run_id": run_dir.name, "seed": int(cfg.get("seed", 0)), "status": meta.get("status"),
                     "commit": str(meta.get("git_commit", ""))[:7], "seconds": meta.get("seconds", np.nan),
                     "variant": cfg.get("variant"), "dir": run_dir, "config": cfg})
    return pd.DataFrame(rows, columns=["run_id", "seed", "status", "commit", "seconds", "variant", "dir", "config"])


def select_runs(table: pd.DataFrame, ignore=(), first: str | None = None, log=print) -> pd.DataFrame:
    """the finished runs at or after the run id first, one per seed, the latest, whose configs agree in everything
    but the seed and the keys in ignore. stops when two seeds were run with other settings, since their numbers
    would not measure the seed."""
    if first is not None:
        for r in table[table["run_id"] < first].itertuples():
            log(f"left out {r.run_id}, before {first}")
        table = table[table["run_id"] >= first]
    for r in table[table["status"] != "done"].itertuples():
        log(f"left out {r.run_id}, status {r.status}")
    done = table[table["status"] == "done"].sort_values("run_id")
    kept = done.drop_duplicates("seed", keep="last")
    for r in done[~done["run_id"].isin(kept["run_id"])].itertuples():
        log(f"left out {r.run_id}, seed {r.seed} has a later run")
    if kept.empty:
        raise ValueError("no finished run")
    diff = config_differences(kept["config"].tolist(), ("seed", *ignore))
    if diff:
        raise ValueError("the runs differ in their settings and not only in the seed, "
                         + ", ".join(f"{k} {' against '.join(v)}" for k, v in diff.items()))
    return kept.sort_values("seed").reset_index(drop=True)


def _is_rate(column: str) -> bool:
    return column in ("rate", "refused") or column.endswith("_rate")


def pool(frames: dict, keys: list[str], weight: str | None = None) -> pd.DataFrame:
    """frames maps a seed to the same table of its run. returns one row per key with the columns every seed agrees on
    kept as they are, the median and the 5% and 95% quantiles over seeds of every other number as <column>_median,
    _low and _high, a share of draws also pooled over the draws of every seed as <column>_pooled when weight names
    the draws, and seeds, the seeds behind the row. a text column that moves with the seed is left out."""
    keys = [k for k in keys if all(k in f for f in frames.values())]
    if not keys:
        raise ValueError("no key column is shared by the tables")
    parts = []
    for seed, frame in frames.items():
        if frame.duplicated(keys).any():
            raise ValueError(f"the keys {', '.join(keys)} do not identify the rows of seed {seed}")
        parts.append(frame.assign(seed=seed))
    both = pd.concat(parts, ignore_index=True)
    for col in both.columns:
        if pd.api.types.is_bool_dtype(both[col]):
            both[col] = both[col].astype(float)
    grouped = both.groupby(keys, sort=False, dropna=False)
    out = grouped.size().rename("seeds").reset_index()
    stats = {}
    for col in both.columns:
        if col in keys or col == "seed":
            continue
        if grouped[col].nunique(dropna=False).le(1).all():
            stats[col] = grouped[col].first().to_numpy()
        elif pd.api.types.is_numeric_dtype(both[col]):
            stats[f"{col}_median"] = grouped[col].median().to_numpy()
            stats[f"{col}_low"] = grouped[col].quantile(LOW).to_numpy()
            stats[f"{col}_high"] = grouped[col].quantile(HIGH).to_numpy()
            if weight in both and _is_rate(col):
                w = both[weight].astype(float)
                num = (both[col].astype(float) * w).groupby([both[k] for k in keys], sort=False, dropna=False).sum()
                den = w.groupby([both[k] for k in keys], sort=False, dropna=False).sum()
                stats[f"{col}_pooled"] = (num / den).to_numpy()
    return pd.concat([out, pd.DataFrame(stats, index=out.index)], axis=1)


def load_frames(selected: pd.DataFrame, name: str) -> dict:
    return {int(r.seed): runs.load_table(r.dir, name) for r in selected.itertuples()
            if (Path(r.dir) / "results" / f"{name}.parquet").exists()}


def summaries(selected: pd.DataFrame) -> pd.DataFrame:
    # the scalar fields of summary.json of every run, one row per seed, lists and nested fields flattened one level
    rows = []
    for r in selected.itertuples():
        summary = runs.load_json(r.dir, "summary")
        row = {"seed": int(r.seed), "run_id": r.run_id, "commit": r.commit, "seconds": r.seconds}
        for key, value in flatten(summary).items():
            if isinstance(value, (bool, int, float, str)):
                row[key] = value
            elif isinstance(value, list) and all(isinstance(v, (str, int, float)) for v in value):
                row[key] = ", ".join(str(v) for v in value)
            elif isinstance(value, list) and all(isinstance(v, list) for v in value):
                row[key] = ", ".join("+".join(str(x) for x in v) for v in value)
        rows.append(row)
    return pd.DataFrame(rows)


def _stats(table: pd.DataFrame, columns) -> list[str]:
    # the pooled forms of the columns that exist, the constant form or the median, low, high and pooled ones
    out = []
    for col in columns:
        out += [c for c in [col, f"{col}_median", f"{col}_low", f"{col}_high", f"{col}_pooled"] if c in table]
    return out


def e1_table(pooled: pd.DataFrame, summary: dict, experiment: str) -> pd.DataFrame:
    # E1, the FOR and the dismissed share against the prevalence on the windows RQ1 is read on, the reference
    # windows of the main period on AnoShift and the reference windows inside the claim on LITNET
    t = pooled[pooled["arm"] == "reference"] if "arm" in pooled else pooled
    if experiment == "e11_anoshift_prevalence":
        t = t[t["period"] == summary.get("main_period", "all")]
    else:
        t = t[t["subset"] == "in_claim"]
    t = t.assign(main=t["prevalence"] == summary["main_prevalence"])
    columns = ["method", "mix", "prevalence", "main", "windows", "seeds"] + _stats(t, [
        "violation_rate", "violation_window_max", "refused", "dismissed_mean", "for_median",
        "best_up_to_pi_max_dismissed", "best_at_prevalence_dismissed", "best_every_mix_dismissed", "realized_rate"])
    return t.sort_values(["mix", "method", "prevalence"])[[c for c in columns if c in t]].reset_index(drop=True)


def e2_table(pooled: pd.DataFrame, summary: dict) -> pd.DataFrame:
    # E2, the violation share of every method on each window of one family at the main prevalence, per arm
    t = pooled[(pooled["prevalence"] == summary["main_prevalence"]) & (pooled["mix"] == "window")]
    columns = ["arm", "window", "segment", "method", "in_claim", "few_attacks", "window_sessions", "seeds"] + _stats(
        t, ["rate", "refused", "dismissed_mean", "for_median", "best_up_to_pi_max_dismissed",
            "best_every_mix_dismissed", "realized_rate"])
    return t.sort_values(["arm", "window", "method"])[[c for c in columns if c in t]].reset_index(drop=True)


def e2_natural_table(pooled: pd.DataFrame) -> pd.DataFrame:
    columns = ["window", "segment", "method", "prevalence", "up_to_pi_max", "in_claim", "window_sessions",
               "seeds"] + _stats(pooled, ["rate", "refused", "dismissed_mean", "for_median",
                                          "best_up_to_pi_max_dismissed", "realized_rate"])
    return pooled.sort_values(["window", "method"])[[c for c in columns if c in pooled]].reset_index(drop=True)


def e5_table(pooled: pd.DataFrame, scorers: pd.DataFrame | None, summary: dict) -> pd.DataFrame:
    # E5, the certified share of every scorer at the main epsilon on the reference windows with the window's mix,
    # over every window for the share and over the windows in the claim for the violation check, with the scorer's
    # yardstick, AUC, low tail and calibration error beside it, the scorers in the order of the certified share
    main = pooled[(pooled["epsilon"] == summary["main_epsilon"]) & (pooled["arm"] == "reference")]
    every = main[main["subset"] == "all"].set_index("scorer")
    claim = main[main["subset"] == "in_claim"].set_index("scorer")
    columns = ["kind", "windows", "seeds"] + _stats(every, ["dismissed_mean", "refused", "best_every_mix_dismissed"])
    t = every[[c for c in columns if c in every]].copy()
    t["claim_windows"] = claim["windows"].reindex(t.index)
    for col in _stats(claim, ["violation_rate", "violation_window_max"]):
        t[f"claim_{col}"] = claim[col].reindex(t.index)
    if scorers is not None:
        s = scorers.set_index("scorer")
        for col in _stats(s, E5_SCORER_COLUMNS):
            t[col] = s[col].reindex(t.index)
    share = "dismissed_mean_median" if "dismissed_mean_median" in t else "dismissed_mean"
    return t.sort_values(share, ascending=False).reset_index()


def e6_table(pooled: pd.DataFrame) -> pd.DataFrame:
    columns = ["unit", "case", "composition", "policy", "arm", "unseen_segment", "unseen_sessions", "seeds"] + \
        _stats(pooled, E6_COLUMNS)
    return pooled.sort_values(["unit", "case", "composition", "policy", "arm"])[[c for c in columns if c in pooled]] \
        .reset_index(drop=True)


def e6_windows_table(pooled: pd.DataFrame, summary: dict) -> pd.DataFrame:
    # the AUC, the benign share dismissed and the miss rate of each family's window at the offline FORGE thresholds
    # with the family unseen against the control with every family known, one row per unit and family
    control = pooled[pooled["unit"] == summary["control"]].set_index("window")
    rows = []
    for u in summary["units"]:
        own = pooled[(pooled["unit"] == u["name"]) & (pooled["unseen"] == 1)]
        if own.empty:
            continue
        own = own.iloc[0]
        known = control.loc[own["window"]]
        for family in u["families"]:
            row = {"unit": u["name"], "segment": own["segment"], "family": family, "seeds": own["seeds"]}
            for col, name in [("auc", "auc"), ("forge_benign_dismissed", "benign_dismissed"),
                              (f"forge_miss_{family}", "miss")]:
                for case, source in [("unseen", own), ("known", known)]:
                    # a column the seed never moves has one form, which stands for its median and its range
                    for suffix in ("_median", "_low", "_high"):
                        row[f"{name}_{case}{suffix}"] = source[col] if col in pooled else source[f"{col}{suffix}"]
            rows.append(row)
    return pd.DataFrame(rows)


def e3_table(pooled: pd.DataFrame) -> pd.DataFrame:
    # E3, the alarm, the leakage, the bound and the coverage of every arm on each policy's stream per composition
    columns = ["stream", "policy", "composition", "prevalence", "arm", "seeds"] + _stats(pooled, E3_COLUMNS)
    return pooled[[c for c in dict.fromkeys(columns) if c in pooled]].reset_index(drop=True)


def e4_table(pooled: pd.DataFrame) -> pd.DataFrame:
    # E4, the same per cell of the sweep, the main cell among them
    columns = ["name", "axis", "rate", "allocation", "floor", "kind", "restart", "delay", "target", "policy",
               "composition", "prevalence", "seeds"] + _stats(pooled, E4_COLUMNS)
    return pooled[[c for c in dict.fromkeys(columns) if c in pooled]].reset_index(drop=True)


def source_of(cfg: dict) -> dict | None:
    # the experiment and run id of the source run a config names, none when it has no source
    src = cfg.get("source")
    if not isinstance(src, dict) or not src.get("experiment") or not src.get("run"):
        return None
    return {"experiment": str(src["experiment"]), "run": str(src["run"])}


def check_sources(selected: pd.DataFrame, root=None, log=print) -> pd.DataFrame:
    """the seed of the source run of every selected run against the run's own seed, for an experiment whose
    thresholds come from the source run of the same seed. returns one row per run with its source run and that
    run's seed, which is missing when the source run is not in the runs folder, and stops on a seed that differs,
    since the pool would then mix scorer seeds."""
    base = Path(root) if root else paths.runs_dir()
    rows = []
    wrong = []
    for r in selected.itertuples():
        src = source_of(r.config)
        row = {"seed": int(r.seed), "run_id": r.run_id, "source_run": None if src is None else src["run"],
               "source_seed": np.nan}
        if src is not None:
            folder = base / src["experiment"] / src["run"]
            if (folder / "config.yaml").is_file():
                row["source_seed"] = int(runs.load_run_config(folder).get("seed", 0))
                if row["source_seed"] != row["seed"]:
                    wrong.append(f"seed {r.seed} reads {src['run']} of seed {row['source_seed']}")
            else:
                log(f"seed {r.seed}, source run {src['run']} is not in the runs folder, its seed is not checked")
        rows.append(row)
    if wrong:
        raise ValueError("the source run of a seed has another seed, " + ", ".join(wrong))
    checked = sum(np.isfinite(row["source_seed"]) for row in rows)
    log(f"source runs of {checked} of {len(rows)} seeds checked, every one with the seed of its run")
    return pd.DataFrame(rows)


def pool_experiment(experiment: str, root=None, out=None, ignore=(), first: str | None = None, log=print) -> dict:
    """pools the finished runs of the experiment in the runs folder, at or after the run id first, and writes the
    pooled tables, the summaries by seed, the list of runs and the E tables of the experiment into out. returns the
    tables by name."""
    if experiment not in TABLES:
        raise ValueError(f"no pooling rule for {experiment}, known are {', '.join(TABLES)}")
    if experiment in SOURCE_PER_SEED:
        ignore = (*ignore, *SOURCE_KEYS)
    selected = select_runs(seed_runs(experiment, root), ignore, first, log)
    log(f"{experiment}, {len(selected)} seeds, " + ", ".join(f"seed {r.seed} run {r.run_id}"
                                                               for r in selected.itertuples()))
    summary = runs.load_json(selected["dir"].iloc[0], "summary")
    listed = selected.drop(columns=["dir", "config"])
    sources = [source_of(c) for c in selected["config"]]
    if any(src is not None for src in sources):
        listed["source_run"] = [None if src is None else src["run"] for src in sources]
    if experiment in SOURCE_PER_SEED:
        check_sources(selected, root, log)
    tables = {"runs": listed, "summary_by_seed": summaries(selected)}
    for name, (keys, weight) in TABLES[experiment].items():
        frames = load_frames(selected, name)
        if len(frames) < len(selected):
            log(f"table {name} is missing in {len(selected) - len(frames)} runs and is not pooled")
            continue
        tables[name] = pool(frames, keys, weight)
    if experiment in ("e11_anoshift_prevalence", "e12_litnet_prevalence") and "by_prevalence" in tables:
        tables["e1_for_by_prevalence"] = e1_table(tables["by_prevalence"], summary, experiment)
    if experiment == "e12_litnet_prevalence":
        if "claims" in tables:
            tables["e2_by_family"] = e2_table(tables["claims"], summary)
        if "natural" in tables:
            tables["e2_natural"] = e2_natural_table(tables["natural"])
    if experiment == "e23_unseen_family":
        if "claims" in tables:
            tables["e6_by_family"] = e6_table(tables["claims"])
        if "windows" in tables:
            tables["e6_windows"] = e6_windows_table(tables["windows"], summary)
    if experiment in ("e31_anoshift_scorers", "e32_litnet_scorers") and "by_epsilon" in tables:
        tables["e5_by_scorer"] = e5_table(tables["by_epsilon"], tables.get("scorers"), summary)
    if experiment == "e21_monitor_streams" and "claims" in tables:
        tables["e3_by_policy"] = e3_table(tables["claims"])
    if experiment == "e22_monitor_sweep" and "claims" in tables:
        tables["e4_by_cell"] = e4_table(tables["claims"])
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(out / f"{name}.csv", index=False)
        with open(out / "summary.json", "w", encoding="utf-8") as f:
            json.dump({"experiment": experiment, "seeds": selected["seed"].tolist(),
                       "runs": selected["run_id"].tolist(), "commits": selected["commit"].tolist()}, f, indent=2)
        log(f"wrote {', '.join(tables)} to {out}")
    return tables


def check_e23_controls(selected: pd.DataFrame, root=None, log=print) -> dict:
    """the offline thresholds of the control of every e23 run against those of the e12 run with the same seed, when
    the runs folder holds one, since the control at a seed is e12 at that seed. returns the result per seed."""
    out = {}
    for r in selected.itertuples():
        source = r.config["source"]["experiment"]
        twins = seed_runs(source, root)
        twins = twins[(twins["seed"] == r.seed) & (twins["status"] == "done")]
        if twins.empty:
            log(f"seed {r.seed}, no finished {source} run with this seed to check the control against")
            continue
        twin = twins.iloc[-1]
        hist = runs.load_arrays(r.dir, "histograms")
        saved = runs.load_arrays(twin["dir"], "draws")
        control = runs.load_json(r.dir, "summary")["control"]
        same = {p: int(np.sum(hist[f"{control}|{p}_index"] != saved[f"{p}_index"]))
                for p in ("forge_simplex", "ltt_iid", "conformal_fnr", "fixed_sel")
                if f"{control}|{p}_index" in hist and f"{p}_index" in saved}
        out[int(r.seed)] = {"run": twin["run_id"], **same}
        log(f"seed {r.seed}, control against {source} run {twin['run_id']}, calibration sets with another "
            "threshold " + ", ".join(f"{p} {n}" for p, n in same.items()))
    return out
