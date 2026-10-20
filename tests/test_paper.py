import json

import numpy as np
import pandas as pd
import pytest

from forge.eval import paper
from forge.viz import paper as figures
from forge.viz.export import save_figure

from test_seeds import fake_run


def pooled_e1(dataset, methods, seeds=3):
    # a pooled E1 table with the forms pool_seeds writes, one row per method, mix and prevalence
    rows = []
    for method in methods:
        for mix in ("window", "calibration"):
            for k, pi in enumerate([0.01, 0.1]):
                share = 0.6 - 0.1 * methods.index(method) - 0.05 * k
                rows.append({"method": method, "mix": mix, "prevalence": pi, "main": pi == 0.1, "windows": 4,
                             "seeds": seeds, "violation_rate_median": 0.01 * k, "violation_rate_low": 0.0,
                             "violation_rate_high": 0.02 * k, "violation_rate_pooled": 0.012 * k,
                             "violation_window_max_median": 0.05, "violation_window_max_low": 0.0,
                             "violation_window_max_high": 0.1, "refused": 0.0, "dismissed_mean_median": share,
                             "dismissed_mean_low": share - 0.02, "dismissed_mean_high": share + 0.02,
                             "for_median_median": 1e-4 * (k + 1), "for_median_low": 0.0 if k == 0 else 5e-5,
                             "for_median_high": 2e-4 * (k + 1), "best_up_to_pi_max_dismissed_median": 0.7,
                             "best_up_to_pi_max_dismissed_low": 0.68, "best_up_to_pi_max_dismissed_high": 0.72,
                             "best_at_prevalence_dismissed_median": 0.8, "best_at_prevalence_dismissed_low": 0.78,
                             "best_at_prevalence_dismissed_high": 0.82})
    return pd.DataFrame(rows)


def pooled_folder(e1: pd.DataFrame, epsilon=1e-3) -> dict:
    # the pooled folder of an E1 experiment as read_pooled gives it, its E1 table and the summaries by seed
    n = int(e1["seeds"].max())
    return {"e1_for_by_prevalence": e1, "summary_by_seed": pd.DataFrame({"seed": range(n), "epsilon": [epsilon] * n})}


def test_e1_table_and_rows_rename_the_pooled_forms(tmp_path):
    pooled = {"e11": pooled_folder(pooled_e1("anoshift", ["forge_simplex", "ltt_iid"])),
              "e12": pooled_folder(pooled_e1("litnet2020", ["forge_simplex", "forge_pooled", "ltt_iid"]))}
    table = paper.e1_table(pooled)
    assert table["dataset"].tolist()[:8] == ["anoshift"] * 8 and len(table) == 8 + 12
    assert "for_median_median" in table and "best_at_prevalence_dismissed_low" in table
    rows = paper.e1_rows(table[table["dataset"] == "litnet2020"], "window")
    assert set(rows["mix"]) == {"window"} and len(rows) == 6
    # the medians become the values the figure reads, the quantiles its ranges, the pooled share the violation
    for name in ("for_median", "for_low", "for_high", "violation_rate", "violation_window_min",
                 "violation_window_max", "dismissed_mean", "dismissed_low", "dismissed_high",
                 "best_up_to_pi_max_dismissed", "best_at_prevalence_dismissed"):
        assert name in rows, name
    r = rows[(rows["method"] == "forge_simplex") & (rows["prevalence"] == 0.1)].iloc[0]
    assert r["violation_rate"] == pytest.approx(0.012) and r["violation_window_max"] == pytest.approx(0.02)
    assert r["dismissed_mean"] == pytest.approx(0.55) and r["dismissed_low"] == pytest.approx(0.53)
    settings = {"epsilon": 1e-3, "delta": 0.05, "calibration_prevalence": 0.02, "pi_max": 0.1}
    fig = figures.e1_for_by_prevalence([
        {"data": paper.e1_rows(table[table["dataset"] == "anoshift"]), "dataset": "anoshift", "summary": settings,
         "label": "AnoShift"},
        {"data": rows, "dataset": "litnet2020", "summary": settings, "label": "LITNET"}])
    assert len(fig.axes) == 6
    # the FOR axis starts at the lowest median, not at the interpolated low quantile
    assert fig.axes[0].get_ylim()[0] == pytest.approx(1e-4)
    labels = fig.legends[0].get_texts()
    assert any("calibration mix" in t.get_text() for t in labels)
    assert all(p.stat().st_size > 0 for p in save_figure(fig, tmp_path, "e1"))


def pooled_e5():
    rows, scorers = [], []
    for dataset in ("anoshift", "litnet2020"):
        for scorer, kind in [("lightgbm", "base"), ("mlp", "base"), ("s_stack", "combiner"), ("s_best", "combiner")]:
            for eps in (1e-4, 1e-3, 1e-2):
                for arm in ("reference", "real"):
                    for subset in ("all", "in_claim"):
                        share = 0.5 + 0.1 * np.log10(eps / 1e-3) + (0.05 if scorer == "s_stack" else 0)
                        rows.append({"dataset": dataset, "scorer": scorer, "kind": kind, "epsilon": eps, "arm": arm,
                                     "subset": subset, "windows": 5, "seeds": 3, "dismissed_mean_median": share,
                                     "dismissed_mean_low": share - 0.03, "dismissed_mean_high": share + 0.03,
                                     "refused": 0.0, "violation_rate_median": 0.001, "violation_rate_low": 0.0,
                                     "violation_rate_high": 0.002, "violation_rate_pooled": 0.001})
            scorers.append({"dataset": dataset, "scorer": scorer, "kind": kind, "windows": 5, "seeds": 3,
                            "dismissed_mean_median": 0.5, "dismissed_mean_low": 0.47, "dismissed_mean_high": 0.53,
                            "benign_at_miss_hardest_median": 0.6, "benign_at_miss_hardest_low": 0.55,
                            "benign_at_miss_hardest_high": 0.65, "ece_median": np.nan if scorer == "mlp" else 0.01,
                            "ece_low": np.nan if scorer == "mlp" else 0.005,
                            "ece_high": np.nan if scorer == "mlp" else 0.02})
    return pd.DataFrame(rows), pd.DataFrame(scorers)


def test_e5_figures_leave_s_best_out(tmp_path):
    by_epsilon, by_scorer = pooled_e5()
    fig = figures.e5_dismissed_by_epsilon(by_epsilon, {"anoshift": 1e-3, "litnet2020": 1e-3})
    assert len(fig.axes) == 2
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "S-stack" in labels and "S-best" not in labels and "LightGBM" in labels
    save_figure(fig, tmp_path, "e5")
    fig = figures.e5_dismissed_vs_tail(by_scorer)
    assert len(fig.axes) == 2
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "S-best" not in labels and "AnoShift" in labels and "LITNET-2020" in labels
    save_figure(fig, tmp_path, "e5_tail")


def test_e6_figure_orders_the_families_as_the_windows_table(tmp_path):
    rows = []
    for unit, segment in [("land", "LAND"), ("syn_flood", "SYN flood")]:
        for composition in ("typical", "worst"):
            for arm in ("monitor", "monitor_any", "none", "window_audit", "naive"):
                for case in ("unseen", "known"):
                    rows.append({"unit": unit, "case": case, "composition": composition, "policy": "forge_simplex",
                                 "arm": arm, "unseen_segment": segment, "unseen_sessions": 1000, "seeds": 5,
                                 "alarm_in_unseen_rate_median": 0.8, "alarm_in_unseen_rate_low": 0.6,
                                 "alarm_in_unseen_rate_high": 0.9, "alarm_in_unseen_rate_pooled": 0.78,
                                 "unseen_leak_median_median": 300.0, "unseen_leak_median_low": 100.0,
                                 "unseen_leak_median_high": 800.0})
    table = pd.DataFrame(rows)
    windows = pd.DataFrame({"unit": ["syn_flood", "land"], "segment": ["SYN flood", "LAND"],
                            "family": ["syn_flood", "land"], "seeds": [5, 5]})
    fig = figures.e6_leakage_by_family(table, windows)
    assert len(fig.axes) == 4
    assert [t.get_text() for t in fig.axes[2].get_xticklabels()] == ["SYN flood", "LAND"]
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "no monitoring" in labels and "running estimate, no bound" not in labels
    save_figure(fig, tmp_path, "e6")


def fake_e12(root, run_id, seed, violation):
    # the by_prevalence rows and the summary the workload of the certification stage reads
    rows = []
    for method in ("forge_simplex", "ltt_iid"):
        for pi in (0.02, 0.1):
            rows.append({"arm": "reference", "subset": "in_claim", "method": method, "prevalence": pi,
                         "mix": "window", "dismissed_mean": 0.6 if method == "forge_simplex" else 0.7,
                         "violation_rate": violation if method == "ltt_iid" else 0.0, "refused": 0.0,
                         "best_up_to_pi_max_dismissed": 0.75, "best_every_mix_dismissed": 0.6, "pairs": 100})
    summary = {"dataset": "litnet2020", "main_prevalence": 0.1, "delta": 0.05}
    return fake_run(root, "e12_litnet_prevalence", run_id, {"seed": seed}, tables=[("by_prevalence",
                                                                                     pd.DataFrame(rows))],
                    summary=summary)


def test_e7_certification_pools_the_workload_over_seeds(tmp_path):
    folders = {0: fake_e12(tmp_path, "20261015-000000_aaaaaaa", 0, 0.04),
               1: fake_e12(tmp_path, "20261015-000001_aaaaaaa", 1, 0.08)}
    table = paper.e7_certification(folders, 0.005)
    main = table[(table["mix"] == "window") & (table["main"] == 1)].set_index("policy")
    assert main.loc["forge_simplex", "seeds"] == 2 and "workload_per_million_median" in table
    # the violation share over the draws of both seeds is their mean, and safety follows it against delta
    assert main.loc["ltt_iid", "violation_rate_pooled"] == pytest.approx(0.06)
    assert not main.loc["ltt_iid", "safe"] and main.loc["forge_simplex", "safe"]
    assert main.loc["forge_simplex", "workload_per_million_median"] == pytest.approx(0.4e6 + 0.005 * 0.6e6)
    assert main.loc["best_up_to_pi_max", "safe"] and main.loc["best_every_mix", "violation_rate_pooled"] == 0
    frontier = paper.e7_frontier(table)
    assert frontier["policy"].tolist() == ["forge_simplex", "ltt_iid"]
    fig = figures.e7_workload_vs_risk(table, pd.DataFrame())
    assert len(fig.axes) == 2 and fig.axes[1].axison is False
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert any("LTT" in t for t in labels) and any("every mix" in t for t in labels)
    save_figure(fig, tmp_path / "figures", "e7")


def test_e7_monitoring_over_one_seed_has_the_pooled_forms(tmp_path):
    claims = pd.DataFrame([{"composition": "typical", "policy": p, "arm": arm, "dismissed_share_mean": 0.5,
                            "audits_mean": 10.0, "leak_median": 3.0, "realized_for_median": 1e-4,
                            "alarm_rate": 0.2 if arm == "monitor" else 0.0}
                           for p in ("forge_simplex", "recalibration") for arm in ("monitor", "none")])
    blocks = pd.DataFrame({"composition": ["typical"] * 2, "window": [0, 1], "window_sessions": [500, 500]})
    folder = fake_run(tmp_path, "e21_monitor_streams", "20261014-000000_bbbbbbb", {"seed": 0},
                      tables=[("claims", claims), ("blocks", blocks)], summary={})
    table = paper.e7_monitoring({0: folder})
    assert (table["seeds"] == 1).all() and "workload_per_million_median" in table
    assert "workload_per_million" not in table
    row = table[(table["policy"] == "forge_simplex") & (table["arm"] == "monitor")].iloc[0]
    assert row["workload_per_million_low"] == row["workload_per_million_high"] == row["workload_per_million_median"]
    assert row["leaked_per_million_median"] == pytest.approx(3.0 / 1000 * 1e6)
    fig = figures.e7_workload_vs_risk(pd.DataFrame(), table)
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert any("recalibration" in t for t in labels) and "under FORGE-Monitor" in labels


def test_read_pooled_and_settings(tmp_path):
    folder = tmp_path / "e12_litnet_prevalence"
    folder.mkdir()
    pd.DataFrame({"seed": [0, 1], "epsilon": [1e-3, 1e-3], "delta": [0.05, 0.05], "dataset": ["litnet2020"] * 2,
                  "draws": [1000, 500]}).to_csv(folder / "summary_by_seed.csv", index=False)
    pd.DataFrame({"run_id": ["a", "b"], "seed": [0, 1]}).to_csv(folder / "runs.csv", index=False)
    tables = paper.read_pooled(folder)
    assert set(tables) == {"summary_by_seed", "runs"}
    assert paper.settings(tables, ["epsilon", "dataset", "missing"]) == {"epsilon": 1e-3, "dataset": "litnet2020"}
    with pytest.raises(ValueError):
        paper.settings(tables, ["draws"])
    with pytest.raises(FileNotFoundError):
        paper.read_pooled(tmp_path / "nothing")
    # the run folders behind a pooled folder, those in the runs folder only
    fake_run(tmp_path / "runs", "e12_litnet_prevalence", "a", {"seed": 0}, summary={})
    found = paper.run_folders(tables, "e12_litnet_prevalence", tmp_path / "runs")
    assert list(found) == [0] and found[0].name == "a"
    assert json.loads((found[0] / "results" / "summary.json").read_text()) == {}


def test_e7_monitoring_over_two_seeds_and_the_e3_tables(tmp_path):
    folders = {}
    for seed in (0, 1):
        claims = pd.DataFrame([{"composition": "typical", "policy": p, "arm": arm, "dismissed_share_mean": 0.5,
                                "audits_mean": 10.0, "leak_median": 3.0 + seed, "realized_for_median": 1e-4,
                                "alarm_rate": 0.2 + 0.2 * seed if arm == "monitor_year" else 0.0}
                               for p in ("forge_simplex", "ltt_iid") for arm in ("monitor_year", "none")])
        blocks = pd.DataFrame({"composition": ["typical"] * 2, "window": [0, 1], "window_sessions": [500, 500]})
        folders[seed] = fake_run(tmp_path, "e21_monitor_streams", f"20261016-00000{seed}_b", {"seed": seed},
                                 tables=[("claims", claims), ("blocks", blocks)], summary={})
    table = paper.e7_monitoring(folders)
    assert (table["seeds"] == 2).all() and (table["sessions"] == 1000).all()
    row = table[(table["policy"] == "ltt_iid") & (table["arm"] == "monitor_year")].iloc[0]
    # the leakage per million is the median over the two seeds with their range, the alarm share its median
    assert row["leaked_per_million_median"] == pytest.approx(3.5 / 1000 * 1e6)
    assert row["leaked_per_million_low"] == pytest.approx(3.05 / 1000 * 1e6)
    assert row["alarm_rate_median"] == pytest.approx(0.3)
    fig = figures.e7_workload_vs_risk(pd.DataFrame(), table)
    assert len(fig.axes) == 2
    pooled = {"e21": {"e3_by_policy": pd.DataFrame({"policy": ["forge_simplex"], "composition": ["typical"],
                                                    "arm": ["monitor"], "seeds": [2], "alarm_rate_pooled": [0.1]})},
              "e22": {"e4_by_cell": pd.DataFrame({"name": ["main"], "policy": ["ltt_iid"], "seeds": [2],
                                                  "leak_median_median": [3000.0]})}}
    out = paper.e3_tables(pooled)
    assert set(out) == {"e3_by_policy", "e4_by_cell"}
    assert out["e3_by_policy"]["dataset"].tolist() == ["anoshift"] and out["e4_by_cell"].columns[0] == "dataset"
    assert paper.e3_tables({"e11": {}}) == {}


def test_the_tables_take_the_third_dataset():
    pooled = {"e11": pooled_folder(pooled_e1("anoshift", ["forge_simplex"])),
              "e13": pooled_folder(pooled_e1("ugr16", ["forge_simplex", "ltt_iid"]), 0.005)}
    table = paper.e1_table(pooled)
    assert table["dataset"].tolist() == ["anoshift"] * 4 + ["ugr16"] * 8
    # the E2 tables name a window of LITNET by its segment and one of UGR16 by its start
    natural = pd.DataFrame({"window": [0], "method": ["forge_simplex"], "prevalence": [0.03], "up_to_pi_max": [True],
                            "in_claim": [True], "window_sessions": [100], "seeds": [2], "rate_pooled": [0.0],
                            "rate_median": [0.0]})
    out = paper.e2_tables({"e12": {"e2_natural": natural.assign(segment="LAND")},
                           "e13": {"e2_natural": natural.assign(start="2016-08-08 00:00")}})
    t = out["e2_natural"]
    assert t["dataset"].tolist() == ["litnet2020", "ugr16"] and "rate_pooled" in t
    assert t["segment"].iloc[0] == "LAND" and t["start"].iloc[1] == "2016-08-08 00:00"
    assert pd.isna(t["segment"].iloc[1]) and pd.isna(t["start"].iloc[0])
    assert paper.e2_tables({"e11": {}}) == {}
    # the E3 table holds both stream datasets
    claims = pd.DataFrame({"policy": ["forge_simplex"], "composition": ["typical"], "arm": ["monitor_week"],
                           "seeds": [2], "alarm_rate_pooled": [0.0]})
    out = paper.e3_tables({"e21": {"e3_by_policy": claims.assign(arm="monitor_year")},
                           "e24": {"e3_by_policy": claims}})
    assert out["e3_by_policy"]["dataset"].tolist() == ["anoshift", "ugr16"]
    assert out["e3_by_policy"]["arm"].tolist() == ["monitor_year", "monitor_week"]


def test_e1_table_lines_up_one_seed_with_many():
    # a dataset pooled over one seed keeps the plain names of its run, and gets the forms of the others
    one = pooled_e1("ugr16", ["forge_simplex", "ltt_iid"], seeds=1)
    plain = {c: c[:-len("_median")] for c in one.columns if c.endswith("_median")}
    one = one.rename(columns=plain).drop(columns=[c for c in one.columns if c.endswith(("_low", "_high", "_pooled"))])
    pooled = {"e11": pooled_folder(pooled_e1("anoshift", ["forge_simplex"])),
              "e13": {"e1_for_by_prevalence": one,
                      "summary_by_seed": pd.DataFrame({"seed": [0], "epsilon": [0.005]})}}
    table = paper.e1_table(pooled)
    assert table.columns.is_unique and "for_median" not in table and "violation_rate_pooled" in table
    rows = paper.e1_rows(table[table["dataset"] == "ugr16"], "window")
    assert rows.columns.is_unique and len(rows) == 4
    r = rows[(rows["method"] == "ltt_iid") & (rows["prevalence"] == 0.1)].iloc[0]
    assert r["violation_rate"] == pytest.approx(0.01) and r["for_low"] == r["for_median"] == r["for_high"]
    assert r["dismissed_low"] == r["dismissed_mean"] == pytest.approx(0.45)


def test_e1_table_carries_the_epsilon_of_every_experiment():
    # UGR16 at its operating target and at the paper's epsilon, where the certificate refuses, stand apart in one
    # table, and the rows of the figure keep the operating target
    pooled = {"e11": pooled_folder(pooled_e1("anoshift", ["forge_simplex"])),
              "e13": pooled_folder(pooled_e1("ugr16", ["forge_simplex", "ltt_iid"]), 0.005),
              "e13_floor": pooled_folder(pooled_e1("ugr16", ["forge_simplex", "ltt_iid"]), 0.001)}
    table = paper.e1_table(pooled)
    assert table.columns[:3].tolist() == ["dataset", "epsilon", "method"]
    assert table.groupby(["dataset", "epsilon"]).size().to_dict() == {("anoshift", 0.001): 4, ("ugr16", 0.001): 8,
                                                                      ("ugr16", 0.005): 8}
    ugr16 = table[table["dataset"] == "ugr16"]
    rows = paper.e1_rows(ugr16, "window", 0.005)
    assert len(rows) == 4 and (rows["epsilon"] == 0.005).all() and len(paper.e1_rows(ugr16, "window")) == 8
    assert paper.EXPERIMENTS["e13_floor"] == "e13_ugr16_floor" and paper.DATASETS["e13_floor"] == "ugr16"
    with pytest.raises(ValueError, match="no E1 table in the pooled folders of e11_anoshift_prevalence"):
        paper.e1_table({"e21": {}})


def test_e7_monitoring_reads_the_windows_of_ugr16(tmp_path):
    claims = pd.DataFrame([{"stream": stream, "composition": "typical", "policy": p, "arm": arm,
                            "dismissed_share_mean": 0.9, "audits_mean": 20.0, "leak_median": 5.0,
                            "realized_for_median": 1e-4, "alarm_rate": 0.1 if arm == "monitor_week" else 0.0}
                           for stream in ("simulated", "replay") for p in ("forge_simplex", "ltt_iid")
                           for arm in ("monitor_week", "monitor", "none")])
    by_window = pd.DataFrame({"stream": ["simulated"] * 2 + ["replay"] * 2, "composition": ["typical"] * 4,
                              "window": [0, 1, 0, 1], "start": ["2016-08-08 00:00", "2016-08-08 06:00"] * 2,
                              "window_sessions": [400, 600] * 2})
    folder = fake_run(tmp_path, "e24_ugr16_streams", "20261018-000000_c", {"seed": 0},
                      tables=[("claims", claims), ("by_window", by_window)],
                      summary={"source": {"dataset": "ugr16"}})
    table = paper.e7_monitoring({0: folder})
    assert set(table["dataset"]) == {"ugr16"} and (table["sessions"] == 1000).all()
    row = table[(table["stream"] == "replay") & (table["policy"] == "ltt_iid")
                & (table["arm"] == "monitor_week")].iloc[0]
    assert row["leaked_per_million_median"] == pytest.approx(5.0 / 1000 * 1e6)
    assert row["workload_per_million_median"] == pytest.approx(0.1e6 + 20.0 / 1000 * 1e6)
    # the figure draws the sequence per week as the monitor of UGR16, filled, and no monitoring hollow
    fig = figures.e7_workload_vs_risk(pd.DataFrame(), table)
    assert len(fig.axes) == 2 and "UGR'16" in fig.axes[1].get_xlabel()
    points = [line for line in fig.axes[1].get_lines() if len(line.get_xdata()) == 1]
    assert sum(line.get_markerfacecolor() == "white" for line in points) == 2 and len(points) == 4


def test_paper_rows_keep_the_replay_where_a_dataset_has_it(tmp_path):
    t = pd.DataFrame({"dataset": ["anoshift"] * 2 + ["litnet2020"], "stream": ["simulated", "replay", "simulated"],
                      "policy": ["a", "a", "b"]})
    kept = paper.paper_rows(t)
    assert kept["stream"].tolist() == ["replay", "simulated"] and kept["policy"].tolist() == ["a", "b"]
    assert paper.paper_rows(t.drop(columns=["stream"])).equals(t.drop(columns=["stream"]))
    one = pd.DataFrame({"stream": ["simulated", "replay"], "x": [1, 2]})
    assert paper.paper_rows(one)["x"].tolist() == [2]
    # the monitoring stage pools with the stream among its keys, and the E7 figure draws the replay
    folders = {}
    for seed in (0, 1):
        claims = pd.DataFrame([{"stream": stream, "composition": "typical", "policy": p, "arm": arm,
                                "dismissed_share_mean": 0.5, "audits_mean": 10.0,
                                "leak_median": 3.0 + seed + (10.0 if stream == "replay" else 0.0),
                                "realized_for_median": 1e-4, "alarm_rate": 0.2 if arm == "monitor" else 0.0}
                               for stream in ("simulated", "replay") for p in ("forge_simplex", "ltt_iid")
                               for arm in ("monitor", "none")])
        blocks = pd.DataFrame({"stream": ["simulated"] * 2 + ["replay"] * 2, "composition": ["typical"] * 4,
                               "window": [0, 1, 0, 1], "window_sessions": [500] * 4})
        folders[seed] = fake_run(tmp_path, "e21_monitor_streams", f"20261016-00001{seed}_b", {"seed": seed},
                                 tables=[("claims", claims), ("blocks", blocks)], summary={})
    table = paper.e7_monitoring(folders)
    assert set(table["stream"]) == {"simulated", "replay"} and (table["seeds"] == 2).all()
    row = table[(table["stream"] == "replay") & (table["policy"] == "ltt_iid") & (table["arm"] == "monitor")].iloc[0]
    assert row["leaked_per_million_median"] == pytest.approx(13.5 / 1000 * 1e6)
    fig = figures.e7_workload_vs_risk(pd.DataFrame(), table)
    xs = [line.get_xdata() for line in fig.axes[1].get_lines() if len(line.get_xdata()) == 1]
    assert all(x[0] >= 10000 for x in xs) and len(xs) == 4
