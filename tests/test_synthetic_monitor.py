import numpy as np

from forge import experiments, runs
from forge.config import load_config
from forge.experiments.synthetic_monitor import (FLAGS, Tables, audited_events, judge, segments_of, simulate_adaptive,
                                                 violation_start)
from forge.monitor.audit import bin_profile
from forge.monitor.sequence import Sequence
from forge.paths import configs_dir
from forge.runs import Run

RATIOS = ([1.5, 2, 3, 5, 10, 30, 100], [0, 0.25, 0.5, 0.75])


def test_segments_and_violation_start():
    eps = 1e-3
    ramp = segments_of({"ramp": {"flows": 1001, "start": 0.5, "end": 2.5, "steps": 5}}, eps)
    assert [f for f, _ in ramp] == [200, 200, 200, 200, 201]
    assert np.allclose([p / eps for _, p in ramp], [0.5, 1.0, 1.5, 2.0, 2.5])
    assert violation_start(ramp, eps) == 401
    assert violation_start([(100, 0.5 * eps), (10, eps)], eps) == 0


def test_judge_recomputes_what_the_events_say():
    # a stream of one row at a known prevalence, the outcomes of judge checked against a direct reading of the events
    rng = np.random.default_rng(5)
    eps, q = 1e-3, 0.01
    positions = np.cumsum(rng.geometric(10 * eps, size=(50, 3000)), axis=1)
    n_end = 200000
    positions = np.where(positions <= n_end, positions, n_end + 1)
    counts = (positions <= n_end).sum(1)
    bins = np.zeros(positions.shape, dtype=np.int8)
    q_by_bin = np.array([q])
    events = audited_events(np.random.default_rng(6).random(positions.shape), positions, bins, q_by_bin, n_end)
    ev_pos, ev_g, ev_q, valid = events
    assert (ev_pos[:, 0] == 0).all() and valid[:, 0].all() and (ev_q[:, 1:][valid[:, 1:]] == q).all()
    # the count at an event is the number of attacks at or before its position
    for i in range(50):
        for r in np.flatnonzero(valid[i])[1:]:
            assert ev_g[i, r] == (positions[i] <= ev_pos[i, r]).sum()
    seq = Sequence("thinning", *RATIOS, q, q, 0.025, 0.025)
    tables = Tables(seq, q, q)
    out = judge(seq, tables, events, positions, counts, n_end, eps, [0, 1000], q_by_bin, bin_profile(1, 2.0),
                np.ones(1))
    lcb, ucb = tables.up_to(0, n_end, eps)
    alarmed = (out["flags"] & FLAGS["alarm"]) > 0
    assert alarmed.mean() > 0.9 and (out["alarm_time"][alarmed] >= 1000).all()
    # an alarm after the floor fires at the first audited attack whose bound exceeds eps times its position
    event = alarmed & (out["alarm_time"] > 1000)
    k, t = out["alarm_audited"][event], out["alarm_time"][event]
    assert event.sum() > 0 and (lcb[k] > eps * t).all() and (lcb[k - 1] <= eps * t).all()
    # the leakage at the alarm is the count at its position, and with a delay the attacks of the delay join it
    assert (out["leak"][0][alarmed] == out["leak_at_alarm"][alarmed]).all()
    assert (out["leak"][1000] >= out["leak"][0]).all()
    assert (out["leak"][0][~alarmed] == counts[~alarmed]).all()
    assert (out["bound"][alarmed] >= 1).all() and (out["bound"][~alarmed] == -1).all()
    assert (out["audits"] >= out["alarm_audited"].clip(0)).all()


def test_adaptive_adversary_cannot_break_the_bounds():
    eps, q, flows = 1e-3, 0.02, 3000
    seq = Sequence("thinning", *RATIOS, q, q, 0.025, 0.025)
    lcb, ucb = seq.tables(q, flows + 1)
    cell = {"rate": 50, "evade_below": 0.5, "skip_after_audit": True, "benign_delay": 5, "attack_delay": 40}
    out = simulate_adaptive(np.random.default_rng(7), cell, eps, q, lcb, ucb, 400, flows)
    assert (out["miss_lower"] | out["miss_upper"]).mean() <= 0.05 + 0.03
    assert (out["attacked"] > 0).all() and (out["lag_at_end"] >= 5).all()
    alarmed = out["alarmed"]
    assert alarmed.mean() > 0.5 and (out["alarm_time"][alarmed] >= 1000).all()
    assert (out["leak_at_alarm"][alarmed] >= out["leak_at_check"][alarmed]).all()
    # the bound of Theorem 3 holds at the last check that passed whenever the bounds cover
    covered = ~(out["miss_lower"] | out["miss_upper"])
    assert (out["leak_at_check"][alarmed & covered] <= out["bound"][alarmed & covered]).all()


def test_e20_tables_and_figures(tmp_path):
    cfg = load_config(configs_dir() / "experiments" / "e20_synthetic_monitor.yaml",
                      ["simulation.draws=120", "simulation.draws_per_chunk=50", "adaptive.draws=30",
                       "adaptive.flows=4000", "audits.rates=[0.005, 0.02]", "audits.delays=[0, 5000]"])
    cfg["stream"]["scenarios"] = [
        {"name": "safe", "segments": [{"flows": 300000, "rate": 0.5}]},
        {"name": "tenfold", "segments": [{"flows": 300000, "rate": 10}]},
        {"name": "hundredfold", "segments": [{"flows": 50000, "rate": 100}]},
        {"name": "abrupt", "segments": [{"flows": 100000, "rate": 0.5}, {"flows": 200000, "rate": 10}]},
    ]
    cfg["stream"]["fixed"] = ["tenfold"]
    run = Run(cfg, root=tmp_path)
    module = experiments.get("synthetic_monitor")
    module.run(cfg, run)
    run.finish()
    m, au, st = cfg["monitor"], cfg["audits"], cfg["stream"]
    claims = runs.load_table(run.dir, "claims")
    cells = len(st["scenarios"]) * len(au["rates"]) * len(au["allocations"]) * len(m["kinds"])
    assert len(claims) == cells * len(au["delays"])
    assert (claims["miscover"] >= claims[["miscover_lower", "miscover_upper"]].max(axis=1)).all()
    main = claims[(claims["rate"] == 0.005) & (claims["allocation"] == "uniform") & (claims["kind"] == "thinning")
                  & (claims["delay"] == 0)].set_index("scenario")
    assert main.loc["hundredfold", "alarm_rate"] == 1.0 and main.loc["hundredfold", "delay_median"] < 20000
    assert main.loc["safe", "alarm_rate"] <= 0.1 and np.isnan(main.loc["safe", "delay_median"])
    assert main.loc["abrupt", "violation_start"] == 100001
    # more audits find a violation sooner
    tenfold = claims[(claims["scenario"] == "tenfold") & (claims["allocation"] == "uniform")
                     & (claims["kind"] == "thinning") & (claims["delay"] == 0)].set_index("rate")
    assert tenfold.loc[0.02, "delay_median"] < tenfold.loc[0.005, "delay_median"]
    # a delay adds leakage and never removes any
    late = claims[claims["delay"] == 5000].set_index(["scenario", "rate", "allocation", "kind"])
    early = claims[claims["delay"] == 0].set_index(["scenario", "rate", "allocation", "kind"])
    assert (late["leak_median"] >= early.loc[late.index, "leak_median"]).all()
    draws = runs.load_arrays(run.dir, "draws")
    key = "tenfold|0.005|uniform|thinning|"
    assert draws[key + "flags"].shape == (120,) and draws[key + "leak_5000"].shape == (120,)
    # a fixed stream has one count of attacks for every draw without an alarm
    assert len(np.unique(draws["tenfold|0.005|uniform|thinning|leak_0"][draws[key + "alarm_time"] < 0])) <= 1
    adaptive = runs.load_table(run.dir, "adaptive")
    assert len(adaptive) == len(cfg["adaptive"]["cells"])
    summary = runs.load_json(run.dir, "summary")
    assert set(summary["largest_miscoverage"]) == set(m["kinds"])
    trace = runs.load_arrays(run.dir, "trace")
    assert int(trace["n_end"]) == 300000 and trace["valid"][0]
    names = {p.name for p in run.figures.iterdir()}
    for name in module.FIGURES:
        assert {f"{name}.png", f"{name}.pdf", f"{name}_a4.pdf"} <= names
