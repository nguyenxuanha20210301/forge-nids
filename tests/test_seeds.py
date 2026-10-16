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



def test_pool_e32_e5_table_ranks_the_scorers(tmp_path):
    root = tmp_path / "runs"
    for seed in (0, 1, 2):
        rows = []
        for scorer, share in [("lightgbm", 0.61), ("xgboost", 0.66)]:
            for eps in (0.0005, 0.001):
                for arm in ("reference", "real"):
                    for subset, windows in [("all", 11), ("in_claim", 7)]:
                        rows.append({"scorer": scorer, "kind": "base", "epsilon": eps, "arm": arm, "subset": subset,
                                     "windows": windows, "pairs": windows * 1000,
                                     "violation_rate": 0.001 * seed if subset == "in_claim" else 0.05,
                                     "violation_window_max": 0.01 * seed, "refused": 0.0,
                                     "dismissed_mean": share + 0.01 * seed, "best_every_mix_dismissed": share + 0.004,
                                     "family_minimum": 1364, "groups": 9})
        scorers = pd.DataFrame({"scorer": ["lightgbm", "xgboost"], "kind": ["base", "base"],
                                "selection_auc": [0.980, 0.981], "calibration_auc": [0.973 + 0.001 * seed] * 2,
                                "benign_at_miss_hardest": [0.69, 0.75], "ece": [0.0056, 0.0053],
                                "selection_yardstick": [0.87, 0.89], "benign_at_miss_pooled": [0.81, 0.79],
                                "hardest_family": ["land", "land" if seed else "smurf"]})
        fake_run(root, "e32_litnet_scorers", f"20261015-00000{seed}_a",
                 {"seed": seed, "experiment": {"name": "e32_litnet_scorers"}},
                 tables=[("by_epsilon", pd.DataFrame(rows)), ("scorers", scorers)], summary={"main_epsilon": 0.001})
    tables = seeds.pool_experiment("e32_litnet_scorers", root, log=lambda s: None)
    e5 = tables["e5_by_scorer"]
    assert e5["scorer"].tolist() == ["xgboost", "lightgbm"] and e5["seeds"].tolist() == [3, 3]
    assert e5["windows"].tolist() == [11, 11] and e5["claim_windows"].tolist() == [7, 7]
    assert np.isclose(e5["dismissed_mean_median"].iloc[0], 0.67)
    assert np.isclose(e5["dismissed_mean_high"].iloc[1], 0.629)
    # the violation check reads the windows in the claim, pooled over the pairs of every seed
    assert np.isclose(e5["claim_violation_rate_pooled"].iloc[0], 0.001)
    assert np.isclose(e5["benign_at_miss_hardest"].iloc[0], 0.75) and "hardest_family" not in tables["scorers"]
    assert np.isclose(e5["calibration_auc_median"].iloc[1], 0.974)
    assert "claims" not in tables and "windows" not in tables


def e21_tables(seed):
    # the claims, blocks and windows of an e21 run, the alarm share and the leakage moving with the seed
    rows = []
    for policy in ("forge_simplex", "ltt_iid"):
        for composition, pi in (("typical", 0.02), ("worst", 0.1)):
            for arm in ("monitor", "monitor_year", "none"):
                rows.append({"policy": policy, "composition": composition, "prevalence": pi, "arm": arm,
                             "draws": 1000, "alarm_rate": 0.1 + 0.1 * seed if arm != "none" else 0.0,
                             "false_alarm_rate": 0.0, "alarm_block_median": 3.0 + seed, "alarm_session_median": 5e4,
                             "onset_true_rate": 1.0, "onset_true_block_median": 1.0, "onset_realized_rate": 1.0,
                             "leak_median": 600.0 + 10 * seed, "leak_mean": 550.0, "excess_median": -3000.0,
                             "sessions_median": 4e6, "dismissed_share_mean": 0.5, "realized_for_median": 1e-4,
                             "audits_mean": 2e4, "bound_median": 2000.0 if arm != "none" else np.nan,
                             "over_bound_rate": 0.0, "miscover_rate": 0.01 * seed,
                             "miscover_sequence_rate": 0.002 * seed, "sequences": 6 if arm == "monitor_year" else 1,
                             "blocks_certified_mean": 50.0, "blocks_violation_mean": 2.0})
    claims = pd.DataFrame(rows)
    blocks = pd.DataFrame([{"policy": p, "composition": c, "prevalence": pi, "window": w, "year": 2010 + w // 2,
                            "block": w % 2, "window_sessions": 1000 * (w + 1), "dismissed_share_mean": 0.5,
                            "for_true_mean": 1e-4 * (seed + 1), "attacks_mean": 10.0}
                           for p in ("forge_simplex", "ltt_iid") for c, pi in (("typical", 0.02), ("worst", 0.1))
                           for w in range(4)])
    windows = pd.DataFrame({"window": range(4), "year": [2010, 2010, 2011, 2011], "sessions": [2000] * 4,
                            "auc": [0.9 + 0.01 * seed] * 4})
    return [("claims", claims), ("blocks", blocks), ("windows", windows)]


def test_pool_e21_and_e22_tables_and_check_the_seed_of_the_source_run(tmp_path):
    root = tmp_path / "runs"
    for seed in (0, 1):
        fake_run(root, "e11_anoshift_prevalence", f"20261015-00000{seed}_a",
                 {"seed": seed, "experiment": {"name": "e11_anoshift_prevalence"}})
        fake_run(root, "e21_monitor_streams", f"20261016-00000{seed}_b",
                 {"seed": seed, "experiment": {"name": "e21_monitor_streams"},
                  "source": {"experiment": "e11_anoshift_prevalence", "run": f"20261015-00000{seed}_a"}},
                 tables=e21_tables(seed), summary={"epsilon": 0.001, "source": {"dataset": "anoshift"}})
    lines = []
    tables = seeds.pool_experiment("e21_monitor_streams", root, out=tmp_path / "pooled", log=lines.append)
    assert set(tables) >= {"runs", "summary_by_seed", "claims", "blocks", "windows", "e3_by_policy"}
    # the run id of the source differs between the seeds by design, its seed is checked, and the runs list it
    assert tables["runs"]["source_run"].tolist() == ["20261015-000000_a", "20261015-000001_a"]
    assert any("source runs of 2 of 2 seeds checked" in line for line in lines)
    e3 = tables["e3_by_policy"].set_index(["policy", "composition", "arm"])
    assert len(e3) == 12 and "stream" not in e3 and (e3["seeds"] == 2).all()
    row = e3.loc[("forge_simplex", "typical", "monitor")]
    assert np.isclose(row["alarm_rate_pooled"], 0.15) and np.isclose(row["alarm_rate_median"], 0.15)
    assert np.isclose(row["leak_median_median"], 605.0) and np.isclose(row["leak_median_high"], 609.5)
    assert row["sequences"] == 1 and e3.loc[("forge_simplex", "typical", "monitor_year"), "sequences"] == 6
    blocks = tables["blocks"].set_index(["policy", "composition", "window"])
    assert len(blocks) == 16 and blocks.loc[("ltt_iid", "worst", 3), "window_sessions"] == 4000
    assert np.isclose(blocks.loc[("ltt_iid", "worst", 3), "for_true_mean_median"], 1.5e-4)
    assert np.isclose(tables["windows"].set_index("window").loc[2, "auc_median"], 0.905)
    assert (tmp_path / "pooled" / "e3_by_policy.csv").exists()
    # a source run with another seed stops the pool, and one missing from the runs folder is not checked
    fake_run(root, "e21_monitor_streams", "20261016-000002_b",
             {"seed": 2, "experiment": {"name": "e21_monitor_streams"},
              "source": {"experiment": "e11_anoshift_prevalence", "run": "20261015-000000_a"}},
             tables=e21_tables(2), summary={"epsilon": 0.001})
    with pytest.raises(ValueError, match="seed 2 reads 20261015-000000_a of seed 0"):
        seeds.pool_experiment("e21_monitor_streams", root, log=lines.append)
    fake_run(root, "e21_monitor_streams", "20261016-000003_b",
             {"seed": 2, "experiment": {"name": "e21_monitor_streams"},
              "source": {"experiment": "e11_anoshift_prevalence", "run": "20261015-000009_a"}},
             tables=e21_tables(2), summary={"epsilon": 0.001})
    tables = seeds.pool_experiment("e21_monitor_streams", root, log=lines.append)
    assert tables["runs"]["seed"].tolist() == [0, 1, 2]
    assert any("seed 2, source run 20261015-000009_a is not in the runs folder" in line for line in lines)
    # e22 pools its claims per cell and keeps the cells and the audit policies as they are
    for seed in (0, 1):
        claims = pd.DataFrame([{"policy": "conformal_fnr", "composition": "typical", "prevalence": 0.02,
                                "name": name, "axis": axis, "rate": rate, "allocation": "uniform", "floor": 0.0005,
                                "kind": "thinning", "restart": "year", "delay": 0, "target": "count", "draws": 1000,
                                "audits_per_million": rate * 1e6, "alarm_rate": 0.9 + 0.05 * seed,
                                "false_alarm_rate": 0.0, "alarm_block_median": 28.0, "alarm_session_median": 1e6,
                                "leak_median": 3000.0 - 100 * seed, "leak_mean": 3000.0, "leak_in_delay_median": 0.0,
                                "excess_median": 100.0, "dismissed_share_mean": 0.5, "bound_median": 16000.0,
                                "over_bound_rate": 0.0, "miscover_sequence_rate": 0.01, "sequences": 6,
                                "blocks_certified_mean": 0.0}
                               for name, axis, rate in [("main", "main", 0.005), ("rate_0.01", "rate", 0.01)]])
        cells = pd.DataFrame({"name": ["main", "rate_0.01"], "axis": ["main", "rate"], "rate": [0.005, 0.01]})
        policies = pd.DataFrame({"rate": [0.005, 0.01], "allocation": ["uniform"] * 2, "floor": [0.0005] * 2,
                                 "bin": [0, 0], "q": [0.005, 0.01]})
        fake_run(root, "e22_monitor_sweep", f"20261016-00000{seed}_c",
                 {"seed": seed, "experiment": {"name": "e22_monitor_sweep"},
                  "source": {"experiment": "e11_anoshift_prevalence", "run": f"20261015-00000{seed}_a"}},
                 tables=[("claims", claims), ("cells", cells), ("policies", policies)], summary={"epsilon": 0.001})
    tables = seeds.pool_experiment("e22_monitor_sweep", root, log=lines.append)
    e4 = tables["e4_by_cell"].set_index("name")
    assert e4.index.tolist() == ["main", "rate_0.01"] and e4.loc["rate_0.01", "axis"] == "rate"
    assert np.isclose(e4.loc["main", "alarm_rate_pooled"], 0.925)
    assert np.isclose(e4.loc["main", "leak_median_median"], 2950.0) and e4.loc["main", "sequences"] == 6
    assert tables["cells"]["rate"].tolist() == [0.005, 0.01] and len(tables["policies"]) == 2
