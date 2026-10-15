import json

import numpy as np
import pandas as pd
import pytest
import yaml

from forge import runs
from forge.config import load_config
from forge.eval import seeds
from forge.paths import configs_dir

from test_litnet_prevalence import OVERRIDES, run_e12


def fake_run(root, experiment, run_id, cfg, status="done", tables=(), arrays=(), summary=None):
    # a run folder with its meta.json, config.yaml and the tables and arrays given
    folder = root / experiment / run_id
    (folder / "results").mkdir(parents=True)
    (folder / "meta.json").write_text(json.dumps({"run_id": run_id, "experiment": experiment, "status": status,
                                                  "git_commit": "abcdef0123", "seconds": 1.0}), encoding="utf-8")
    (folder / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    for name, table in tables:
        table.to_parquet(folder / "results" / f"{name}.parquet", index=False)
    for name, data in arrays:
        np.savez_compressed(folder / "results" / f"{name}.npz", **data)
    if summary is not None:
        (folder / "results" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return folder


def test_config_differences_and_pooling():
    a = {"seed": 0, "certify": {"epsilon": 0.001, "grid_size": 1000}, "methods": ["x", "y"]}
    b = {"seed": 1, "certify": {"epsilon": 0.001, "grid_size": 200}, "methods": ["x", "y"]}
    c = {"seed": 2, "certify": {"epsilon": 0.001, "grid_size": 1000}}
    assert seeds.flatten(a) == {"seed": 0, "certify.epsilon": 0.001, "certify.grid_size": 1000, "methods": ["x", "y"]}
    assert seeds.config_differences([a, {**a, "seed": 5}]) == {}
    assert seeds.config_differences([a, b]) == {"certify.grid_size": ["1000", "200"]}
    assert seeds.config_differences([a, c]) == {"methods": ["['x', 'y']", "'missing'"]}
    assert seeds.config_differences([a, b], ignore=("seed", "certify.grid_size")) == {}

    def frame(rate, dismissed, draws=100):
        return pd.DataFrame({"method": ["m1", "m1", "m2", "m2"], "prevalence": [0.01, 0.1, 0.01, 0.1],
                             "window_sessions": [500, 400, 500, 400], "in_claim": [True, False, True, False],
                             "segment": ["a", "b", "a", "b"], "rate": rate, "dismissed_mean": dismissed,
                             "draws": [draws] * 4, "label": ["x", "y", "x", "y"]})

    frames = {0: frame([0.0, 0.2, 0.0, 0.5], [0.6, 0.5, 0.7, 0.6]),
              1: frame([0.0, 0.4, 0.0, 0.7], [0.7, 0.6, 0.8, 0.7]),
              2: frame([0.0, 0.3, 0.0, 0.6], [0.8, 0.7, 0.9, 0.8], draws=200)}
    pooled = seeds.pool(frames, ["method", "prevalence"], "draws")
    assert len(pooled) == 4 and (pooled["seeds"] == 3).all()
    # the columns every seed agrees on stay, the others get their quantiles over seeds, and the shares of draws their
    # pooled value weighted by the draws
    assert pooled["window_sessions"].tolist() == [500, 400, 500, 400]
    assert pooled["in_claim"].tolist() == [1.0, 0.0, 1.0, 0.0] and pooled["segment"].tolist() == ["a", "b"] * 2
    assert "label" in pooled and "draws_median" in pooled and "draws" not in pooled
    row = pooled.set_index(["method", "prevalence"]).loc[("m1", 0.1)]
    assert np.isclose(row["rate_median"], 0.3) and np.isclose(row["rate_low"], 0.21) and np.isclose(row["rate_high"],
                                                                                                    0.39)
    assert np.isclose(row["rate_pooled"], (0.2 * 100 + 0.4 * 100 + 0.3 * 200) / 400)
    assert np.isclose(row["dismissed_mean_median"], 0.6) and "dismissed_mean_pooled" not in pooled
    assert np.isclose(pooled.set_index(["method", "prevalence"]).loc[("m1", 0.01), "rate_pooled"], 0.0)
    # a text column that moves with the seed is left out, and keys must identify the rows
    frames[2]["label"] = ["x", "z", "x", "y"]
    assert "label" not in seeds.pool(frames, ["method", "prevalence"], "draws")
    with pytest.raises(ValueError, match="do not identify"):
        seeds.pool(frames, ["method"], "draws")
    with pytest.raises(ValueError, match="no key column"):
        seeds.pool(frames, ["year"], "draws")


def test_select_runs_keeps_one_finished_run_per_seed_with_one_config(tmp_path):
    root = tmp_path / "runs"
    base = {"seed": 0, "experiment": {"name": "e99"}, "certify": {"grid_size": 10}}
    fake_run(root, "e99", "20261010-000000_a", base)
    fake_run(root, "e99", "20261010-000001_a", {**base, "seed": 1})
    fake_run(root, "e99", "20261010-000002_a", {**base, "seed": 1}, status="failed")
    fake_run(root, "e99", "20261010-000003_a", {**base, "seed": 1})
    fake_run(root, "e99", "20261010-000004_a", {**base, "seed": 2}, status="running")
    table = seeds.seed_runs("e99", root)
    assert table["seed"].tolist() == [0, 1, 1, 1, 2] and table["status"].tolist()[2] == "failed"
    lines = []
    kept = seeds.select_runs(table, log=lines.append)
    assert kept["run_id"].tolist() == ["20261010-000000_a", "20261010-000003_a"] and kept["seed"].tolist() == [0, 1]
    assert any("20261010-000001_a, seed 1 has a later run" in line for line in lines)
    assert any("20261010-000004_a, status running" in line for line in lines)
    fake_run(root, "e99", "20261010-000005_a", {**base, "seed": 3, "certify": {"grid_size": 20}})
    with pytest.raises(ValueError, match="certify.grid_size 10 against 20"):
        seeds.select_runs(seeds.seed_runs("e99", root), log=lines.append)
    kept = seeds.select_runs(seeds.seed_runs("e99", root), ignore=["certify.grid_size"], log=lines.append)
    assert kept["seed"].tolist() == [0, 1, 3]
    # the runs before a run id stay out, which keeps the runs of an earlier design apart
    kept = seeds.select_runs(seeds.seed_runs("e99", root), first="20261010-000005_a", log=lines.append)
    assert kept["seed"].tolist() == [3] and any("20261010-000000_a, before" in line for line in lines)
    assert seeds.seed_runs("missing", root).empty
    with pytest.raises(ValueError, match="no finished run"):
        seeds.select_runs(seeds.seed_runs("missing", root))


def test_pool_e12_over_two_seeds(litnet_signal, tmp_path):
    root = tmp_path / "runs"
    for seed in (0, 1):
        cfg = load_config(configs_dir() / "experiments" / "e12_litnet_prevalence.yaml", OVERRIDES + [f"seed={seed}"])
        run_e12(cfg, root)
    lines = []
    tables = seeds.pool_experiment("e12_litnet_prevalence", root, out=tmp_path / "pooled", log=lines.append)
    assert set(tables) >= {"runs", "summary_by_seed", "by_prevalence", "claims", "natural", "windows",
                           "e1_for_by_prevalence", "e2_by_family", "e2_natural"}
    assert tables["runs"]["seed"].tolist() == [0, 1]
    assert (tmp_path / "pooled" / "e1_for_by_prevalence.csv").exists()
    written = json.loads((tmp_path / "pooled" / "summary.json").read_text())
    assert written["seeds"] == [0, 1] and len(written["runs"]) == 2
    first, second = [runs.load_table(d, "by_prevalence") for d in runs.run_dirs("e12_litnet_prevalence", root)]
    keys = ["arm", "subset", "method", "prevalence", "mix"]
    pooled = tables["by_prevalence"].set_index(keys)
    assert len(pooled) == len(first) and (pooled["seeds"] == 2).all()
    a, b = first.set_index(keys), second.set_index(keys).loc[first.set_index(keys).index]
    # a share of draws pools over both seeds, its quantiles sit between the two values, and a number the seed does
    # not move, the windows behind a row, stays one column
    assert np.allclose(pooled["violation_rate_pooled"], (a["violation_rate"] + b["violation_rate"]) / 2)
    assert np.allclose(pooled["violation_rate_median"], (a["violation_rate"] + b["violation_rate"]) / 2)
    assert np.allclose(pooled["violation_rate_low"], np.minimum(a["violation_rate"], b["violation_rate"])
                       + 0.05 * (a["violation_rate"] - b["violation_rate"]).abs())
    assert pooled["windows"].tolist() == a["windows"].tolist() and "windows_median" not in pooled
    assert not np.allclose(a["dismissed_mean"], b["dismissed_mean"])
    e1 = tables["e1_for_by_prevalence"]
    summary = runs.load_json(runs.run_dirs("e12_litnet_prevalence", root)[0], "summary")
    assert set(e1["mix"]) == {"window", "calibration"} and "violation_rate_pooled" in e1
    assert (e1["main"] == (e1["prevalence"] == summary["main_prevalence"])).all() and e1["main"].any()
    assert len(e1) == len(first[(first["arm"] == "reference") & (first["subset"] == "in_claim")])
    e2 = tables["e2_by_family"]
    claims = runs.load_table(runs.run_dirs("e12_litnet_prevalence", root)[0], "claims")
    at_main = claims[(claims["prevalence"] == summary["main_prevalence"]) & (claims["mix"] == "window")]
    assert len(e2) == len(at_main) and {"segment", "rate_pooled", "in_claim"} <= set(e2)
    assert tables["summary_by_seed"]["seed"].tolist() == [0, 1]
    assert "selection_auc" in tables["summary_by_seed"] and "families" in tables["summary_by_seed"]
    assert any(line.startswith("e12_litnet_prevalence, 2 seeds, seed 0 run ") for line in lines)


def test_pool_e23_tables_and_the_control_check(tmp_path):
    root = tmp_path / "runs"
    units = [{"name": "land", "families": ["land"], "window": 1, "label": "LAND"},
             {"name": "smurf+icmp_flood", "families": ["smurf", "icmp_flood"], "window": 2, "label": "Smurf"}]
    summary = {"control": "all_known", "units": units, "windows": ["SYN", "LAND", "Smurf"]}
    thresholds = {0: np.array([5, 6, 6]), 1: np.array([4, 6, 6])}
    for seed in (0, 1):
        claims = pd.DataFrame({
            "unit": ["land"] * 4 + ["smurf+icmp_flood"] * 4, "case": ["unseen", "unseen", "known", "known"] * 2,
            "composition": ["typical"] * 8, "policy": ["forge_simplex"] * 8,
            "arm": ["monitor", "none"] * 4, "draws": [1000] * 8, "unseen_segment": ["LAND"] * 4 + ["Smurf"] * 4,
            "unseen_sessions": [100] * 8, "alarm_rate": np.linspace(0.1, 0.8, 8) + 0.1 * seed,
            "alarm_in_unseen_rate": np.linspace(0.1, 0.8, 8) + 0.1 * seed,
            "alarm_in_unseen_violated_rate": [np.nan] * 8, "unseen_leak_median": [50.0 + seed] * 8,
            "miscover_rate": [0.01 + 0.01 * seed] * 8, "unseen_miss_mean": [0.9, 0.9, 0.0, 0.0] * 2,
        })
        windows = pd.DataFrame({
            "unit": ["all_known"] * 3 + ["land"] * 3 + ["smurf+icmp_flood"] * 3, "window": [0, 1, 2] * 3,
            "segment": ["SYN", "LAND", "Smurf"] * 3, "unseen": [False, False, False, False, True, False, False,
                                                                 False, True],
            "auc": [0.9, 0.95, 0.99, 0.9, 0.3 + 0.1 * seed, 0.99, 0.9, 0.95, 0.2],
            "forge_benign_dismissed": [0.5] * 9, "forge_miss_land": [np.nan, 0.0, np.nan] * 3,
            "forge_miss_smurf": [np.nan, np.nan, 0.0, np.nan, np.nan, 0.0, np.nan, np.nan, 0.8 + 0.1 * seed],
            "forge_miss_icmp_flood": [np.nan, np.nan, 0.0] * 2 + [np.nan, np.nan, 1.0],
        })
        units_table = pd.DataFrame({"unit": ["all_known", "land", "smurf+icmp_flood"], "best_iteration": [10, 12, 9]})
        cfg = {"seed": seed, "experiment": {"name": "e23_unseen_family"},
               "source": {"experiment": "e12_litnet_prevalence", "run": "r0"}}
        fake_run(root, "e23_unseen_family", f"20261010-00000{seed}_a", cfg,
                 tables=[("claims", claims), ("windows", windows), ("units", units_table)],
                 arrays=[("histograms", {"all_known|forge_simplex_index": thresholds[seed],
                                         "all_known|ltt_iid_index": thresholds[seed]})], summary=summary)
    fake_run(root, "e12_litnet_prevalence", "20261010-000000_b", {"seed": 0, "experiment": {"name": "e12"}},
             arrays=[("draws", {"forge_simplex_index": thresholds[0], "ltt_iid_index": np.array([5, 6, 7])})])
    tables = seeds.pool_experiment("e23_unseen_family", root, out=tmp_path / "pooled", log=lambda s: None)
    e6 = tables["e6_by_family"].set_index(["unit", "case", "composition", "policy", "arm"])
    assert len(e6) == 8 and np.isclose(e6.loc[("land", "unseen", "typical", "forge_simplex", "monitor"),
                                              "alarm_in_unseen_rate_pooled"], 0.15)
    assert np.isclose(e6.loc[("land", "unseen", "typical", "forge_simplex", "monitor"), "unseen_leak_median_median"],
                      50.5)
    # a column that is nan in every seed stays one column, and the miss rates of the family's window come with the
    # control's, one row per family of the unit
    assert e6["alarm_in_unseen_violated_rate"].isna().all() and "unseen_miss_mean" in e6
    w = tables["e6_windows"].set_index(["unit", "family"])
    assert w.index.tolist() == [("land", "land"), ("smurf+icmp_flood", "smurf"), ("smurf+icmp_flood", "icmp_flood")]
    assert np.isclose(w.loc[("smurf+icmp_flood", "smurf"), "miss_unseen_median"], 0.85)
    assert np.isclose(w.loc[("smurf+icmp_flood", "smurf"), "miss_unseen_low"], 0.805)
    assert w.loc[("smurf+icmp_flood", "smurf"), "miss_known_median"] == 0.0
    assert w.loc[("land", "land"), "miss_known_median"] == 0.0 and w.loc[("land", "land"), "miss_known_high"] == 0.0
    assert np.isclose(w.loc[("land", "land"), "auc_unseen_median"], 0.35)
    assert w.loc[("land", "land"), "auc_known_median"] == 0.95
    lines = []
    out = seeds.check_e23_controls(tables["runs"].assign(
        dir=[root / "e23_unseen_family" / r for r in tables["runs"]["run_id"]],
        config=[runs.load_run_config(root / "e23_unseen_family" / r) for r in tables["runs"]["run_id"]]),
        root, log=lines.append)
    assert out == {0: {"run": "20261010-000000_b", "forge_simplex": 0, "ltt_iid": 1}}
    assert any("seed 1, no finished e12_litnet_prevalence run" in line for line in lines)


def test_pool_e11_e1_table_reads_the_main_period(tmp_path):
    root = tmp_path / "runs"
    for seed in (0, 1, 2):
        by_prevalence = pd.DataFrame({
            "arm": ["reference"] * 4 + ["real"] * 4, "period": ["iid", "iid", "near", "near"] * 2,
            "method": ["forge_simplex", "ltt_iid"] * 4, "prevalence": [0.1] * 8, "mix": ["window"] * 8,
            "windows": [12, 12, 36, 36] * 2, "pairs": [12000, 12000, 36000, 36000] * 2,
            "violation_rate": [0.001 * seed, 0.9 + 0.01 * seed, 0.2, 0.5, 0.1, 0.6, 0.3, 0.7],
            "dismissed_mean": [0.44 + 0.01 * seed] * 8, "refused": [0.0] * 8,
        })
        fake_run(root, "e11_anoshift_prevalence", f"20261010-00000{seed}_a",
                 {"seed": seed, "experiment": {"name": "e11_anoshift_prevalence"}},
                 tables=[("by_prevalence", by_prevalence)], summary={"main_period": "iid", "main_prevalence": 0.1})
    tables = seeds.pool_experiment("e11_anoshift_prevalence", root, log=lambda s: None)
    e1 = tables["e1_for_by_prevalence"].set_index("method")
    assert len(e1) == 2 and e1["windows"].tolist() == [12, 12] and e1["seeds"].tolist() == [3, 3]
    assert e1["main"].all()
    assert np.isclose(e1.loc["forge_simplex", "violation_rate_pooled"], 0.001)
    assert np.isclose(e1.loc["ltt_iid", "violation_rate_median"], 0.91)
    assert np.isclose(e1.loc["ltt_iid", "violation_rate_high"], 0.919)
    assert "claims" not in tables and "windows" not in tables
