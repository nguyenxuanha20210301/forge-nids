import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

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
    assert len(fig.axes) == 3 and fig.axes[2].get_xscale() == "log"
    assert all(p.stat().st_size > 0 for p in save_figure(fig, tmp_path / "figures", "e0_certify"))
    fig = figures.e0_online_and_monitor(read["e0_online"], read["e0_monitor_trace"])
    assert [ax.get_xlabel() for ax in fig.axes[:2]] == ["window, benign scores drifting",
                                                        "window, benign scores jumping"]
    assert all(p.stat().st_size > 0 for p in save_figure(fig, tmp_path / "figures", "e0_online_and_monitor"))
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
    assert all(p.stat().st_size > 0 for p in save_figure(fig, tmp_path / "figures", "intro_for_by_prevalence"))
    plt.close("all")
