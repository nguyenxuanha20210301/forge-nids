import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from forge import experiments, runs
from forge.config import load_config
from forge.eval import checks, paper
from forge.paths import configs_dir
from forge.runs import Run
from forge.viz import paper as figures
from forge.viz.export import save_figure

from test_seeds import fake_run


def run_small(name, overrides, root, edit=None):
    # a small run of a check experiment, at the sizes its own test runs
    cfg = load_config(configs_dir() / "experiments" / f"{name}.yaml", overrides)
    if edit is not None:
        edit(cfg)
    run = Run(cfg, root=root)
    experiments.get(cfg["experiment"]["module"]).run(cfg, run)
    run.finish()
    return run, cfg


def few_streams(cfg):
    cfg["stream"]["scenarios"] = [
        {"name": "safe", "segments": [{"flows": 300000, "rate": 0.5}]},
        {"name": "tenfold", "segments": [{"flows": 300000, "rate": 10}]},
        {"name": "abrupt", "segments": [{"flows": 100000, "rate": 0.5}, {"flows": 200000, "rate": 10}]},
    ]
    cfg["stream"]["fixed"] = ["tenfold"]


def write_and_read(tables: dict, folder):
    # the tables written and read back as paper_results draws from them
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(folder / f"{name}.csv", index=False)
    return {p.stem: pd.read_csv(p) for p in sorted(folder.glob("*.csv"))}


def legend_styles_differ(fig) -> bool:
    # no two entries of the legend share a color, a line style and a marker, since a linewidth alone does not tell two
    # lines apart. a point stands in the legend by its error bars and a band by its color
    styles = []
    for h in fig.legends[0].legend_handles:
        if isinstance(h, Line2D):
            styles.append((to_rgba(h.get_color()), h.get_linestyle(), h.get_marker()))
        elif isinstance(h, LineCollection):
            styles.append(("errorbar", to_rgba(h.get_color()[0])))
        else:
            styles.append((type(h).__name__, to_rgba(h.get_facecolor())))
    return len(set(styles)) == len(styles)


def test_e0_tables_keep_the_rows_of_the_runs_and_draw(tmp_path):
    e10, cfg10 = run_small("e10_synthetic_certify", ["simulation.draws=200", "simulation.sizes=[300000, 30000000]",
                                                     "simulation.main_size=30000000"], tmp_path)
    e14, cfg14 = run_small("e14_synthetic_online", ["simulation.draws=200", "deployment.windows=12",
                                                    "deployment.window_sizes=[100000, 1000000]"], tmp_path)
    e20, cfg20 = run_small("e20_synthetic_monitor", ["simulation.draws=120", "simulation.draws_per_chunk=60",
                                                     "adaptive.draws=30", "adaptive.flows=4000",
                                                     "audits.rates=[0.005, 0.02]", "audits.delays=[0, 5000]"],
                           tmp_path, few_streams)

    # Theorem 1 by size is the claims table of the run, with the main size marked and the settings of the run
    by_size = checks.certify_by_size(e10.dir)
    claims = runs.load_table(e10.dir, "claims")
    assert by_size[claims.columns].equals(claims)
    assert by_size.loc[by_size["main"], "size"].unique().tolist() == [30000000]
    c = cfg10["certify"]
    assert (by_size[["epsilon", "delta", "pi_max"]] == [c["epsilon"], c["delta"], c["pi_max"]]).all().all()
    assert (by_size["calibration_prevalence"] == cfg10["synthetic"]["prevalence"]).all()
    # by prevalence, the main size alone, every method, prevalence and mix, each share the violations over the draws
    by_prev = checks.certify_by_prevalence(e10.dir)
    sim = cfg10["simulation"]
    assert set(by_prev["size"]) == {30000000}
    assert len(by_prev) == len(sim["methods"]) * len(sim["prevalences"]) * (2 + len(cfg10["synthetic"]["families"]))
    assert np.allclose(by_prev["rate"], by_prev["violations"] / by_prev["draws"])

    # Theorem 1b keeps every window of the run, the main window size marked and the jump of the abrupt scenario
    on = checks.online(e14.dir)
    claims14 = runs.load_table(e14.dir, "claims")
    assert on[claims14.columns].equals(claims14)
    jumps = {sc["name"]: sc.get("shift_at") for sc in cfg14["deployment"]["scenarios"]}
    assert set(on.loc[on["scenario"] == "abrupt", "shift_at"]) == {jumps["abrupt"]}
    assert on.loc[on["scenario"] == "stable", "shift_at"].isna().all()
    assert set(on.loc[on["main"], "window_size"]) == {cfg14["deployment"]["main_window_size"]}

    # Theorems 2 and 3, the claims of e20 with the main policy marked, one row per scenario and delay there
    mon = checks.monitor(e20.dir)
    claims20 = runs.load_table(e20.dir, "claims")
    assert mon[claims20.columns].equals(claims20)
    main = mon[mon["main"]]
    assert len(main) == len(cfg20["stream"]["scenarios"]) * len(cfg20["audits"]["delays"])
    assert set(main["rate"]) == {cfg20["audits"]["rates"][0]} and set(main["kind"]) == {"thinning"}
    adaptive = checks.monitor_adaptive(e20.dir)
    assert adaptive["cell"].tolist() == [cell["name"] for cell in cfg20["adaptive"]["cells"]]
    # the trace, the bounds at every audited attack and the true count on a grid ending at the last flow, the count
    # at an audited attack being the attacks at or before it
    trace = checks.monitor_trace(e20.dir)
    t = runs.load_arrays(e20.dir, "trace")
    audits, counts = trace[trace["point"] == "audit"], trace[trace["point"] == "count"]
    assert len(audits) == int(t["valid"].sum()) and (audits["lower"] <= audits["upper"]).all()
    assert (np.searchsorted(np.sort(t["attacks"]), audits["position"], side="right") == audits["attacks"]).all()
    assert counts["position"].iloc[-1] == int(t["n_end"]) and counts["attacks"].iloc[-1] == len(t["attacks"])
    assert counts["attacks"].is_monotonic_increasing and set(trace["scenario"]) == {"abrupt"}
    assert set(trace["violation_start"]) == {100001}

    read = write_and_read({"e0_certify_by_size": by_size, "e0_certify_by_prevalence": by_prev, "e0_online": on,
                           "e0_monitor_trace": trace}, tmp_path / "tables")
    fig = figures.e0_certify(read["e0_certify_by_prevalence"], read["e0_certify_by_size"])
    assert len(fig.axes) == 3 and fig.axes[2].get_xscale() == "log" and legend_styles_differ(fig)
    assert all(p.stat().st_size > 0 for p in save_figure(fig, tmp_path / "figures", "e0_certify"))
    fig = figures.e0_online_and_monitor(read["e0_online"], read["e0_monitor_trace"])
    assert [ax.get_xlabel() for ax in fig.axes[:2]] == ["window, benign scores drifting",
                                                        "window, benign scores jumping"]
    assert legend_styles_differ(fig)
    assert all(p.stat().st_size > 0 for p in save_figure(fig, tmp_path / "figures", "e0_online_and_monitor"))
    # c ends as many flows after the alarm as the alarm came after the start of the violation at flow 100001, at the
    # height of what it shows, and at the end of the stream when that lies beyond it or no alarm came
    early = read["e0_monitor_trace"].assign(alarm=120000)
    panel = figures.e0_online_and_monitor(read["e0_online"], early).axes[2]
    shown = early[early["position"] <= 139999]
    assert panel.get_xlim()[1] == pytest.approx(0.139999)
    assert panel.get_ylim()[1] == pytest.approx(1.1 * max(shown.loc[shown["point"] == "audit", "upper"].max(),
                                                          shown.loc[shown["point"] == "count", "attacks"].max(), 1))
    for alarm in (250000, -1):
        panel = figures.e0_online_and_monitor(read["e0_online"], early.assign(alarm=alarm)).axes[2]
        assert panel.get_xlim()[1] == pytest.approx(0.3)
    plt.close("all")


def trace_arrays(stream: str, ends, cum, period_key: str, period, alarm=-1):
    # a saved trace of one stream as e21 and e24 write it, the events of the audited attacks and the window ends
    return {f"{stream}|ev_pos": np.array([0, 50, 120]), f"{stream}|ev_count": np.array([0, 3, 9]),
            f"{stream}|lcb": np.array([0, 1, 2]), f"{stream}|ucb": np.array([40, 44, 50]),
            f"{stream}|ends": np.asarray(ends), f"{stream}|cum_attacks": np.asarray(cum),
            f"{stream}|alarm_pos": np.array(alarm), f"{stream}|n_end": np.array(ends[-1]),
            f"{stream}|{period_key}": np.asarray(period)}


def test_e3_bounds_over_time_reads_the_replay_of_each_dataset(tmp_path):
    summary = {"epsilon": 0.001, "trace": {"policy": "forge_simplex", "composition": "typical"}}
    e21 = fake_run(tmp_path, "e21_monitor_streams", "20261016-173018_38ae553", {"seed": 0}, arrays=[(
        "trace", {**trace_arrays("simulated", [60, 150, 200], [4, 10, 12], "years", [2010, 2010, 2011]),
                  **trace_arrays("replay", [70, 140, 200], [5, 11, 13], "years", [2010, 2011, 2011], alarm=130)})],
        summary=summary)
    e24 = fake_run(tmp_path, "e24_ugr16_streams", "20261021-104508_825dea4", {"seed": 0}, arrays=[(
        "trace", {**trace_arrays("replay", [80, 160, 240, 300], [6, 12, 15, 18], "segments", [0, 0, 1, 1])})],
        summary=summary)
    table = paper.e3_bounds_over_time({"anoshift": e21, "ugr16": e24})
    assert set(table["stream"]) == {"replay"} and set(table["policy"]) == {"forge_simplex"}
    a = table[table["dataset"] == "anoshift"]
    assert a.loc[a["point"] == "audit", "attacks"].tolist() == [0, 3, 9]
    assert a.loc[a["point"] == "window", "period"].tolist() == [2010, 2011, 2011]
    assert set(a["alarm"]) == {130} and set(a["n_end"]) == {200} and set(a["run_id"]) == {e21.name}
    u = table[table["dataset"] == "ugr16"]
    assert u.loc[u["point"] == "window", "period"].tolist() == [0, 0, 1, 1] and set(u["alarm"]) == {-1}
    read = write_and_read({"e3_bounds_over_time": table}, tmp_path / "tables")
    fig = figures.e3_bounds_over_time(read["e3_bounds_over_time"])
    assert [ax.get_xlabel() for ax in fig.axes] == ["millions of dismissed sessions, AnoShift",
                                                   "millions of dismissed flows, UGR'16"]
    assert legend_styles_differ(fig)
    assert all(p.stat().st_size > 0 for p in save_figure(fig, tmp_path / "figures", "e3_bounds_over_time"))
    plt.close("all")
    # a trace saved before the two streams has no stream in its keys and is read whole
    old = fake_run(tmp_path, "e21_monitor_streams", "20261015-041654_6cb2cb3", {"seed": 0}, arrays=[(
        "trace", {k.split("|", 1)[1]: v for k, v in trace_arrays("x", [60, 120], [4, 9], "years",
                                                                 [2010, 2010]).items()})], summary=summary)
    assert set(paper.e3_bounds_over_time({"anoshift": old})["stream"]) == {""}


def test_intro_for_by_prevalence_follows_the_example_of_the_proposal(tmp_path):
    t = paper.intro_for_by_prevalence(prevalences=[0.0018, 0.05, 0.158])
    one = t[t["miss_rate"] == 0.01].set_index("prevalence")["for"]
    # proposal 2.2, a miss rate of 1% at 90% of the benign flows dismissed
    assert one[0.0018] == pytest.approx(2.0e-5, rel=0.01)
    assert one[0.05] == pytest.approx(5.8e-4, rel=0.01)
    assert one[0.158] == pytest.approx(2.1e-3, rel=0.02)
    # at the crossing prevalence the FOR is epsilon
    for m, g in t.groupby("miss_rate"):
        pi = g["crossing"].iloc[0]
        assert pi * m / (pi * m + (1 - pi) * 0.9) == pytest.approx(0.001)
    curves = paper.intro_for_by_prevalence()
    natural = pd.DataFrame({"dataset": ["litnet2020"] * 3 + ["ugr16"] * 3, "window": [0, 1, 1, 0, 1, 2],
                            "method": ["a"] * 6, "prevalence": [0.0, 0.02, 0.02, 0.0025, 0.29, 0.0]})
    read = write_and_read({"intro_for_by_prevalence": curves, "e2_natural": natural}, tmp_path / "tables")
    fig = figures.intro_for_by_prevalence(read["intro_for_by_prevalence"], read["e2_natural"])
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "real windows, LITNET-2020" in labels and "real windows, UGR'16" in labels
    assert {"miss rate 0.1%", "miss rate 1%", "miss rate 5%", r"$\varepsilon$"} <= set(labels)
    assert legend_styles_differ(fig)
    assert all(p.stat().st_size > 0 for p in save_figure(fig, tmp_path / "figures", "intro_for_by_prevalence"))
    plt.close("all")


def case_study_runs(root, threshold=6.0):
    # a small e13 run whose certificate on its expected calibration set lands at index 6 of ten thresholds, the
    # families passing with no miss up to there and failing from index 7 on, and an e24 run that read that threshold
    g = 10
    b_cal = (np.arange(g) + 1) / g
    miss = np.where(np.arange(g) >= 7, 0.5, 0.0)
    grid = pd.DataFrame({"index": np.arange(g), "threshold": np.linspace(-9, 0, g), "selection_benign": b_cal,
                         "selection_attack": miss, "calibration_benign": b_cal, "calibration_attack": miss,
                         "calibration_dos": miss, "calibration_anomaly_spam": miss})
    roles = pd.DataFrame({"role": ["train", "selection", "calibration", "deployment"],
                          "start": ["2016-07-27 12:00:00", "2016-08-01 00:00:00", "2016-08-04 12:00:00",
                                    "2016-08-08 00:00:00"],
                          "end": ["2016-08-01 00:00:00", "2016-08-04 12:00:00", "2016-08-08 00:00:00",
                                  "2016-08-09 00:00:00"],
                          "windows": [18, 14, 14, 4], "flows": [100000, 80000, 50000, 40000],
                          "benign": [99000, 79000, 49000, 39000], "attacks": [1000, 1000, 1000, 1000]})
    starts = pd.date_range("2016-08-08 00:00", periods=4, freq="6h")
    windows = pd.DataFrame({"window": np.arange(4), "start": starts.astype(str), "end": (starts + pd.Timedelta(hours=6)).astype(str),
                            "in_claim": True, "has_attacks": [True, True, False, True],
                            "flows": [10000, 10000, 10000, 10000], "benign": [9500, 9900, 10000, 9990],
                            "dos": [400, 0, 0, 0], "scan11": [0, 0, 0, 0], "scan44": [0, 0, 0, 0],
                            "nerisbotnet": [0, 0, 0, 0], "anomaly_spam": [100, 100, 0, 10],
                            "natural_prevalence": [0.05, 0.01, 0.0, 0.001], "auc": [0.99, 0.98, np.nan, 0.97],
                            "forge_benign_dismissed": [0.7, 0.71, 0.72, 0.7],
                            "forge_miss_dos": [0.0, np.nan, np.nan, np.nan], "forge_miss_anomaly_spam": [0.01, 0.0, np.nan, 0.0]})
    summary = {"epsilon": 0.05, "delta": 0.05, "delta_benign": 0.025, "pi_max": 0.1, "start": "share",
               "start_share": 0.5, "start_misses": 2, "start_slack": 0.05, "families": [["dos"], ["anomaly_spam"]],
               "calibration_expected_attacks": {"dos": 400.0, "anomaly_spam": 600.0}, "calibration_prevalence": 0.02,
               "calibration_sessions": 50000, "family_minimum": 10, "offline_refused": {"forge_simplex": 0.0},
               "selection_auc": 0.99, "calibration_auc": 0.97}
    e13 = fake_run(root, "e13_ugr16_prevalence", "20261022-144455_7157a7b", {"seed": 0},
                   tables=[("grid", grid), ("roles", roles), ("windows", windows)], summary=summary)
    rows = []
    for stream in ("simulated", "replay"):
        for policy in ("forge_simplex", "ltt_iid"):
            for composition in ("typical", "natural"):
                for w in range(4):
                    rows.append({"stream": stream, "policy": policy, "composition": composition, "window": w,
                                 "start": str(starts[w]), "segment": w // 2, "prevalence": 0.02, "subsampled": False,
                                 "window_sessions": 10000, "refused_rate": 0.0,
                                 "threshold_mean": threshold if policy == "forge_simplex" else 8.0,
                                 "dismissed_share_mean": 0.7 + 0.01 * w, "for_true_mean": 0.001 * (w + 1),
                                 "violated_rate": 0.0, "attacks_mean": 7.0 + w, "realized_for_mean": 0.001,
                                 "state_certified": 0.9, "state_running": 0.1, "state_violation": 0.0,
                                 "window_audit_alarm_rate": 0.0})
    by_window = pd.DataFrame(rows)
    claims = pd.DataFrame([{"stream": stream, "policy": policy, "composition": composition, "arm": arm,
                            "draws": 10, "alarm_rate": 0.0 if arm != "naive" else 0.3, "leak_median": 30.0,
                            "sessions_median": 28000.0, "audits_mean": 140.0, "blocks_certified_mean": 3.6}
                           for stream in ("simulated", "replay") for policy in ("forge_simplex", "ltt_iid")
                           for composition in ("typical", "natural")
                           for arm in ("naive", "none", "monitor_week", "monitor")])
    e24 = fake_run(root, "e24_ugr16_streams", "20261021-104508_825dea4", {"seed": 0},
                   tables=[("by_window", by_window), ("claims", claims)],
                   summary={"epsilon": 0.05, "audit_rate": 0.005, "segments": {"first_window": [0, 2]}})
    return e13, e24


def test_case_study_tables_read_seed_0_and_draw(tmp_path):
    e13, e24 = case_study_runs(tmp_path)
    certificate = paper.case_study_certificate(e13, e24)
    items = certificate.set_index("item")["value"]
    assert float(items["certified_index"]) == 6 and float(items["certified_benign_quantile"]) == pytest.approx(0.7)
    assert float(items["benign_lower_bound"]) == pytest.approx(0.7 - np.sqrt(np.log(40) / (2 * 49000)))
    assert float(items["floor"]) == pytest.approx(0.9 * float(items["benign_lower_bound"]))
    assert float(items["dos_attacks"]) == 400 and float(items["dos_misses"]) == 0
    assert 0 < float(items["anomaly_spam_miss_upper"]) < 0.01 and float(items["bound"]) <= 0.05
    assert float(items["next_index"]) == 7 and float(items["next_bound"]) > 0.05
    assert items["next_binding_family"] in ("dos", "anomaly_spam") and float(items["next_binding_miss_rate"]) == 0.5
    assert set(certificate["run_id"]) == {e13.name} and set(certificate["seed"]) == {0}
    # the e24 run read another threshold than the one recomputed from the certification run
    with pytest.raises(ValueError):
        paper.case_study_certificate(*case_study_runs(tmp_path / "other", threshold=5.0))
    windows = paper.case_study_windows(e13, e24)
    assert len(windows) == 4 and set(windows["stream"]) == {"replay"} and set(windows["composition"]) == {"natural"}
    assert windows["dos"].tolist() == [400, 0, 0, 0] and windows["segment"].tolist() == [0, 0, 1, 1]
    assert windows["state_certified"].tolist() == [0.9] * 4 and windows["threshold_mean"].tolist() == [6.0] * 4
    assert windows["forge_miss_dos"].iloc[0] == 0.0 and windows["forge_miss_anomaly_spam"].iloc[0] == 0.01
    stream = paper.case_study_stream(e24)
    assert stream["arm"].tolist() == ["monitor", "monitor_week", "none", "naive"]
    assert set(stream["epsilon"]) == {0.05} and set(stream["audit_rate"]) == {0.005}
    read = write_and_read({"case_study_certificate": certificate, "case_study_windows": windows,
                           "case_study_stream": stream}, tmp_path / "tables")
    rows = paper.case_study_figure_rows(read["case_study_windows"], read["case_study_certificate"])
    assert set(rows["epsilon"]) == {0.05} and set(rows["pi_max"]) == {0.1}
    fig = figures.case_study_by_clock(rows)
    assert len(fig.axes) == 3 and fig.axes[2].get_xlabel() == "window start, August 2016"
    assert legend_styles_differ(fig)
    assert all(p.stat().st_size > 0 for p in save_figure(fig, tmp_path / "figures", "case_study_by_clock"))
    plt.close("all")
