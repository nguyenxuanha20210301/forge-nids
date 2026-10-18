import numpy as np
import pandas as pd
import pytest

from forge import experiments, runs
from forge.config import load_config
from forge.eval import seeds
from forge.experiments import ugr16_streams as e24
from forge.experiments.monitor_streams import POLICIES, STATE_CODES, STREAMS
from forge.paths import configs_dir
from forge.runs import Run

from test_ugr16_prevalence import OVERRIDES, run_e13

# the three compositions at two prevalences of the e13 fixture run and the natural one, the eight deployment windows
# of six hours cut into stretches of 18 h from the first one, three windows, three windows and two, and the last
# stretch joined to the one before, so two segments of three and five windows
E24 = ["monitor.epsilon=0.05", "audits.rate=0.2",
       "compositions=[{name: typical, prevalence: 0.05}, {name: worst, prevalence: 0.2}, {name: natural}]",
       'segments.length="18h"', "segments.minimum_windows=3"]
WINDOWS = 8
COMPOSITIONS = ["typical", "worst", "natural"]


def frame_of(starts):
    return pd.DataFrame({"window": np.arange(len(starts)), "start": pd.to_datetime(starts)})


def test_segments_follow_the_clock_and_join_a_short_tail():
    starts = pd.date_range("2016-08-08", periods=10, freq="6h")
    segments, joined = e24.segments_of(frame_of(starts), "1 day", 3)
    assert segments.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1] and joined == 2
    segments, joined = e24.segments_of(frame_of(starts), "1 day", 2)
    assert segments.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2] and joined == 0
    # a span shorter than the length is one segment, and the stretches are numbered without gaps when windows of the
    # span are missing
    segments, joined = e24.segments_of(frame_of(starts[:3]), "7 days", 1)
    assert segments.tolist() == [0, 0, 0] and joined == 0
    segments, joined = e24.segments_of(frame_of(starts.delete([4, 5, 6, 7])), "1 day", 1)
    assert segments.tolist() == [0, 0, 0, 0, 1, 1] and joined == 0
    with pytest.raises(ValueError, match="minimum_windows"):
        e24.segments_of(frame_of(starts), "1 day", 0)
    with pytest.raises(ValueError, match="time order"):
        e24.segments_of(frame_of(starts[::-1]), "1 day", 1)


def test_window_plan_follows_the_real_windows_of_e13():
    counts = {"benign_below": np.array([100, 400, 800, 1000]),
              "family_below": np.array([[0, 10, 40, 50], [0, 0, 0, 0]]), "sizes": np.array([1000, 50, 0])}
    typical = e24.window_plan(counts, {"name": "typical", "prevalence": 0.02})
    assert typical["sessions"] == round(1000 / 0.98) and not typical["subsampled"] and typical["prevalence"] == 0.02
    assert typical["shares"].tolist() == [1.0, 0.0] and typical["has_attacks"]
    # 1111 flows at 10% would take 111 attacks, so the window keeps its 50 attacks and 450 of its benign flows
    worst = e24.window_plan(counts, {"name": "worst", "prevalence": 0.1})
    assert worst["sessions"] == 500 and worst["subsampled"] and worst["prevalence"] == 0.1
    natural = e24.window_plan(counts, {"name": "natural"})
    assert natural["sessions"] == 1050 and natural["prevalence"] == 50 / 1050 and not natural["subsampled"]
    assert np.allclose(natural["benign"], [0.1, 0.4, 0.8, 1.0]) and np.allclose(natural["miss"][0], [0, 0.2, 0.8, 1])
    empty = {"benign_below": counts["benign_below"], "family_below": np.zeros((2, 4), dtype=int),
             "sizes": np.array([1000, 0, 0])}
    for co in ({"name": "typical", "prevalence": 0.02}, {"name": "natural"}):
        plan = e24.window_plan(empty, co)
        assert plan["prevalence"] == 0 and plan["sessions"] == 1000 and not plan["has_attacks"]
        assert plan["shares"].tolist() == [0.0, 0.0] and not plan["subsampled"]


def test_e24_on_an_e13_run(ugr16_signal, tmp_path):
    cfg13 = load_config(configs_dir() / "experiments" / "e13_ugr16_prevalence.yaml", OVERRIDES)
    run13, _ = run_e13(cfg13, None)
    cfg = load_config(configs_dir() / "experiments" / "e24_ugr16_streams.yaml", [f"source.run={run13.run_id}", *E24])
    run = Run(cfg)
    module = experiments.get("ugr16_streams")
    module.run(cfg, run)
    run.finish()
    claims = runs.load_table(run.dir, "claims")
    by_window = runs.load_table(run.dir, "by_window")
    windows = runs.load_table(run.dir, "windows")
    summary = runs.load_json(run.dir, "summary")
    draws = runs.load_arrays(run.dir, "draws")
    n_draws = cfg13["simulation"]["draws"]
    arms = module.ARMS
    assert summary["source"]["run"] == run13.run_id and summary["source"]["windows"] == WINDOWS
    assert summary["streams"] == STREAMS and summary["main_stream"] == "replay" and summary["chunk"] == 100
    assert summary["segments"]["count"] == 2 and summary["segments"]["joined"] == 2
    assert summary["segments"]["first_window"] == [0, 3] and summary["segments"]["windows"] == [3, 5]
    assert windows["segment"].tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
    assert summary["restart_sequences"] == {"monitor_week": 2, "monitor_window": WINDOWS, "any_start": WINDOWS}
    assert len(claims) == 2 * len(POLICIES) * 3 * len(arms) and len(by_window) == 2 * len(POLICIES) * 3 * WINDOWS
    assert set(claims["policy"]) == set(POLICIES) and set(claims["arm"]) == set(arms)
    assert set(claims["composition"]) == set(COMPOSITIONS) and set(claims["stream"]) == set(STREAMS)
    assert claims.loc[claims["composition"] == "natural", "prevalence"].isna().all()
    assert (claims.loc[claims["composition"] == "worst", "prevalence"] == 0.2).all()
    log = (run.dir / "log.txt").read_text()
    assert log.count("online thresholds of the source run reproduced in every window with attacks") == 3
    assert "the window histograms of the source run reproduced from the saved scores in every window" in log
    assert log.count("every attack row of a window kept with probability") == 3
    assert "2 segments of 18h for the sequence per week, the last 2 windows joined to the segment before" in log
    for stream in STREAMS:
        assert f"{stream}, FORGE-Monitor, largest share of draws leaving the bounds" in log
        assert f"{stream}, FORGE-Monitor per week, largest share of sequences leaving the bounds" in log
        assert f"{stream}, FORGE-Monitor from every window, largest share of draws leaving the bounds" in log
    floor = summary["alarm_floor"]
    assert floor == 20
    # the windows of a composition follow the real windows of e13, a window short of attacks keeps the flows e13 gave
    # it with its benign flows subsampled, a window without attacks goes through at prevalence 0 with all its benign
    # flows and no attack, and the natural composition keeps every flow at the window's own prevalence
    claims13 = runs.load_table(run13.dir, "claims")
    real13 = claims13[(claims13["arm"] == "real") & (claims13["mix"] == "window")
                      & (claims13["method"] == "forge_simplex")]
    w13 = runs.load_table(run13.dir, "windows").sort_values("window").set_index("window")
    with_attacks = w13.index[w13["has_attacks"]].to_numpy()
    without = w13.index[~w13["has_attacks"]].to_numpy()
    assert len(without) == 2 and summary["windows_at_prevalence_zero"] == without.tolist()
    for stream in STREAMS:
        g = by_window[(by_window["stream"] == stream) & (by_window["policy"] == "forge_simplex")]
        for co, pi in (("typical", 0.05), ("worst", 0.2)):
            h = g[g["composition"] == co].set_index("window")
            e = real13[real13["prevalence"] == pi].set_index("window")
            assert (h.loc[with_attacks, "window_sessions"].to_numpy()
                    == e.loc[with_attacks, "window_sessions"].to_numpy()).all()
            assert (h.loc[with_attacks, "subsampled"].to_numpy()
                    == e.loc[with_attacks, "benign_subsampled"].to_numpy()).all()
            assert (h.loc[with_attacks, "prevalence"] == pi).all()
            assert (h.loc[without, "prevalence"] == 0).all() and (h.loc[without, "attacks_mean"] == 0).all()
            assert (h.loc[without, "window_sessions"].to_numpy() == w13.loc[without, "benign"].to_numpy()).all()
            assert summary["windows_subsampled"][co] == sorted(int(w) for w in e.index[e["benign_subsampled"]])
        h = g[g["composition"] == "natural"].set_index("window")
        assert np.allclose(h["prevalence"].to_numpy(), w13["natural_prevalence"].to_numpy())
        assert (h["window_sessions"].to_numpy() == w13["flows"].to_numpy()).all() and not h["subsampled"].any()
        assert (h["segment"].to_numpy() == windows["segment"].to_numpy()).all()
    assert summary["windows_subsampled"]["worst"] and summary["windows_subsampled"]["natural"] == []
    for stream in STREAMS:
        for p in POLICIES:
            for co in COMPOSITIONS:
                key = f"{stream}|{p}|{co}|"
                attacks, dismissed = draws[key + "attacks"], draws[key + "dismissed"]
                ends = np.cumsum(dismissed, axis=0)
                # no monitoring leaks the whole stream, the monitors at most that, and every alarm comes from the
                # floor on
                assert (draws[key + "none_leak"] == attacks.sum(0)).all()
                for arm in [*module.MONITORS, "naive", "window_audit"]:
                    pos = draws[key + f"{arm}_alarm_pos"]
                    assert ((pos == -1) | (pos >= floor)).all()
                    assert (draws[key + f"{arm}_leak"] <= draws[key + "none_leak"]).all()
                # the per window audit alarms at a window end
                pos = draws[key + "window_audit_alarm_pos"]
                assert all(pos[d] in ends[:, d] for d in np.flatnonzero(pos >= 0))
                for arm in module.MONITORS:
                    states = draws[key + f"{arm}_state"]
                    assert states.shape == (WINDOWS, n_draws) and set(np.unique(states)) <= {0, 1, 2}
                    assert draws[key + f"{arm}_bound"].shape == (n_draws,)
                assert (key + "burst" in draws) == (stream == "replay")
                # the windows without attacks dismiss no attack under any composition
                assert (attacks[without] == 0).all()
    sequences = claims.groupby("arm")["sequences"].first()
    assert sequences["monitor"] == 1 and sequences["monitor_week"] == 2
    assert sequences["monitor_window"] == WINDOWS and sequences["any_start"] == WINDOWS
    # the recalibration never tightens, so its mean threshold rises or stays from window to window
    for (stream, co), g in by_window[by_window["policy"] == "recalibration"].groupby(["stream", "composition"]):
        t = g.sort_values("window")["threshold_mean"].to_numpy()
        assert (np.diff(t) >= -1e-9).all()
    for arm in module.MONITORS:
        g = claims[claims["arm"] == arm]
        assert (g["miscover_rate"] <= 0.05 + 0.1).all()
        assert (g[["blocks_certified_mean", "blocks_running_mean", "blocks_violation_mean"]].sum(axis=1)
                .sub(WINDOWS).abs() < 1e-9).all()
    # the natural replay keeps every row, so a window at an offline threshold dismisses exactly the flows of its
    # histogram at or below it, with exactly its attacks there, and its population FOR is that of the simulated
    # stream
    hist = runs.load_arrays(run13.dir, "histograms")
    t13 = runs.load_arrays(run13.dir, "draws")
    for p in ["forge_simplex", "ltt_iid"]:
        t = t13[f"{p}_index"].astype(np.int64)
        assert (t >= 0).any()
        rep = f"replay|{p}|natural|"
        for w in range(WINDOWS):
            attacks_below = hist["window_family_below"][w].sum(0)
            below = hist["window_benign_below"][w] + attacks_below
            assert (draws[rep + "dismissed"][w] == np.where(t >= 0, below[np.maximum(t, 0)], 0)).all()
            assert (draws[rep + "attacks"][w] == np.where(t >= 0, attacks_below[np.maximum(t, 0)], 0)).all()
        assert np.allclose(draws[rep + "for_true"], draws[f"simulated|{p}|natural|for_true"])
    # under a thinned composition the replay keeps every benign row of a window with enough attacks, so its dismissed
    # count is at least the benign count at the threshold, and the attacks of a window follow the population FOR of
    # the window as the simulated ones do
    for co, pi in (("typical", 0.05), ("worst", 0.2)):
        for p in ["ltt_iid", "forge_simplex"]:
            sim, rep = f"simulated|{p}|{co}|", f"replay|{p}|{co}|"
            assert np.allclose(draws[rep + "for_true"], draws[sim + "for_true"])
            expected = draws[rep + "for_true"] * draws[rep + "dismissed"]
            assert np.allclose(draws[rep + "attacks"].mean(1), expected.mean(1), rtol=0.15, atol=3)
            g = by_window[(by_window["stream"] == "replay") & (by_window["policy"] == p)
                          & (by_window["composition"] == co)].set_index("window")
            assert (g["burst_max_median"] >= 0).all() and (g["burst_max_high"] >= g["burst_max_median"]).all()
            t = t13[f"{p}_index"]
            for w in g.index[~g["subsampled"]]:
                benign = np.where(t >= 0, hist["window_benign_below"][w][np.maximum(t, 0)], 0)
                assert (draws[rep + "dismissed"][w] >= benign).all()
    assert by_window.loc[by_window["stream"] == "simulated", "burst_max_median"].isna().all()
    for name in module.FIGURES:
        assert (run.dir / "figures" / f"{name}.png").exists() and (run.dir / "figures" / f"{name}.pdf").exists()
    # the pooled tables over the one seed, with the E3 table cut from the claims
    tables = seeds.pool_experiment("e24_ugr16_streams", out=tmp_path / "pooled", log=lambda s: None)
    assert {"claims", "by_window", "windows", "e3_by_policy"} <= set(tables)
    assert len(tables["claims"]) == len(claims) and len(tables["e3_by_policy"]) == len(claims)
    assert len(tables["by_window"]) == len(by_window) and len(tables["windows"]) == WINDOWS
    # the simulated stream of a run with both streams is the one of a run with the simulated stream alone, since the
    # replay spawns its seeds after it
    alone = Run(load_config(configs_dir() / "experiments" / "e24_ugr16_streams.yaml",
                            [f"source.run={run13.run_id}", *E24, "streams=[simulated]"]), root=tmp_path / "alone")
    module.run(alone.cfg, alone)
    alone.finish()
    alone_claims = runs.load_table(alone.dir, "claims")
    assert set(alone_claims["stream"]) == {"simulated"}
    simulated = claims[claims["stream"] == "simulated"].reset_index(drop=True)
    assert simulated.equals(alone_claims)
    alone_draws = runs.load_arrays(alone.dir, "draws")
    assert all(np.array_equal(alone_draws[k], draws[k]) for k in alone_draws)
