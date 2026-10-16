import json

import numpy as np
import pandas as pd

from forge import runs
from forge.config import load_config
from forge.eval import workload
from forge.paths import configs_dir

from test_litnet_prevalence import OVERRIDES, run_e12


def test_certification_workload_from_an_e12_run(litnet_signal, tmp_path):
    cfg = load_config(configs_dir() / "experiments" / "e12_litnet_prevalence.yaml", OVERRIDES)
    run, _ = run_e12(cfg, tmp_path / "runs")
    table = workload.certification(run.dir, 0.005)
    summary = runs.load_json(run.dir, "summary")
    by_prevalence = runs.load_table(run.dir, "by_prevalence")
    rq1 = by_prevalence[(by_prevalence["arm"] == "reference") & (by_prevalence["subset"] == "in_claim")]
    methods = table[table["policy"].isin(summary["methods"])]
    assert len(methods) == len(rq1) and set(table["dataset"]) == {"litnet2020"}
    row = methods[(methods["policy"] == "forge_simplex") & (methods["mix"] == "window")
                  & (methods["prevalence"] == summary["main_prevalence"])].iloc[0]
    source = rq1[(rq1["method"] == "forge_simplex") & (rq1["mix"] == "window")
                 & (rq1["prevalence"] == summary["main_prevalence"])].iloc[0]
    assert np.isclose(row["escalations_per_million"], (1 - source["dismissed_mean"]) * 1e6)
    assert np.isclose(row["audits_per_million"], 0.005 * source["dismissed_mean"] * 1e6)
    assert np.isclose(row["workload_per_million"], row["escalations_per_million"] + row["audits_per_million"])
    assert row["safe"] == (source["violation_rate"] <= summary["delta"])
    # the yardsticks come as safe policies of their own, one row per prevalence and mix where the run holds them
    sticks = table[table["policy"] == "best_up_to_pi_max"]
    assert len(sticks) == rq1.groupby(["prevalence", "mix"]).ngroups and sticks["safe"].all()
    every = table[table["policy"] == "best_every_mix"]
    assert len(every) == len(sticks) and (every["violation_rate"] == 0).all()
    assert np.isclose(every[(every["mix"] == "window") & (every["prevalence"] == summary["main_prevalence"])]
                      ["dismissed"].iloc[0],
                      source["best_every_mix_dismissed"])


def test_monitoring_workload_and_frontier(tmp_path):
    root = tmp_path / "runs" / "e21_monitor_streams" / "r1"
    (root / "results").mkdir(parents=True)
    claims = pd.DataFrame({
        "policy": ["forge_simplex", "forge_simplex", "ltt_iid", "ltt_iid"], "composition": ["typical"] * 4,
        "arm": ["monitor", "none", "monitor", "none"], "dismissed_share_mean": [0.5, 0.5, 0.3, 0.7],
        "audits_mean": [500.0, 500.0, 300.0, 700.0], "leak_median": [100.0, 100.0, 300.0, 2000.0],
        "realized_for_median": [0.001, 0.001, 0.005, 0.014], "alarm_rate": [0.0, 0.0, 1.0, 0.0],
        "excess_median": [-50.0, -50.0, 200.0, 1900.0],
    })
    blocks = pd.DataFrame({"policy": ["forge_simplex"] * 2 + ["ltt_iid"] * 2, "composition": ["typical"] * 4,
                           "window": [0, 1, 0, 1], "window_sessions": [60000, 140000] * 2})
    claims.to_parquet(root / "results" / "claims.parquet", index=False)
    blocks.to_parquet(root / "results" / "blocks.parquet", index=False)
    (root / "results" / "summary.json").write_text(json.dumps({"epsilon": 0.001}), encoding="utf-8")
    table = workload.monitoring(root)
    assert set(table["dataset"]) == {"anoshift"} and (table["sessions"] == 200000).all()
    ltt = table[(table["policy"] == "ltt_iid") & (table["arm"] == "monitor")].iloc[0]
    assert np.isclose(ltt["escalations_per_million"], 0.7e6) and np.isclose(ltt["audits_per_million"], 1500)
    assert np.isclose(ltt["leaked_per_million"], 1500) and np.isclose(ltt["workload_per_million"], 701500)
    # the frontier keeps the rows no other row beats on both cost and risk
    front = workload.frontier(table, "workload_per_million", "leaked_per_million")
    assert front["policy"].tolist() == ["forge_simplex", "ltt_iid"] and front["arm"].tolist() == ["monitor", "none"]
    front = workload.frontier(pd.DataFrame({"c": [3.0, 2.0, 1.0], "r": [0.0, 0.0, 0.5]}), "c", "r")
    assert front["c"].tolist() == [2.0, 1.0]


def test_monitoring_keeps_the_stream_of_a_run_with_both_kinds(tmp_path):
    root = tmp_path / "runs" / "e21_monitor_streams" / "r2"
    (root / "results").mkdir(parents=True)
    claims = pd.DataFrame({
        "stream": ["simulated", "replay"] * 2, "policy": ["forge_simplex"] * 2 + ["ltt_iid"] * 2,
        "composition": ["typical"] * 4, "arm": ["monitor"] * 4, "dismissed_share_mean": [0.5, 0.5, 0.3, 0.3],
        "audits_mean": [500.0] * 4, "leak_median": [100.0, 120.0, 300.0, 350.0], "realized_for_median": [0.001] * 4,
        "alarm_rate": [0.0, 0.0, 1.0, 1.0], "excess_median": [-50.0] * 4,
    })
    blocks = pd.DataFrame({"stream": ["simulated", "replay"] * 2, "policy": ["forge_simplex"] * 2 + ["ltt_iid"] * 2,
                           "composition": ["typical"] * 4, "window": [0, 0, 1, 1], "window_sessions": [60000] * 4})
    claims.to_parquet(root / "results" / "claims.parquet", index=False)
    blocks.to_parquet(root / "results" / "blocks.parquet", index=False)
    (root / "results" / "summary.json").write_text(json.dumps({"epsilon": 0.001}), encoding="utf-8")
    table = workload.monitoring(root)
    assert table["stream"].tolist() == ["simulated", "replay"] * 2 and (table["sessions"] == 120000).all()
    assert np.isclose(table.loc[1, "leaked_per_million"], 1000.0)
