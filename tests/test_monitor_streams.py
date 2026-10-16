import numpy as np
from scipy.stats import beta

from forge import experiments, runs
from forge.certify.certificate import grid_from, plug_in
from forge.config import load_config
from forge.data.synthetic import draw_counts
from forge.experiments.monitor_streams import (ARMS, MONITORS, POLICIES, STATE_CODES, STREAMS, audited_stream,
                                               judge_stream, recalibrate, restarted_monitor)
from forge.monitor import rule
from forge.monitor.sequence import Sequence, Tables
from forge.paths import configs_dir
from forge.runs import Run

from test_for_vs_prevalence import OVERRIDES, run_e11

RATIOS = ([1.5, 2, 3, 5, 10, 30, 100], [0, 0.25, 0.5, 0.75])


def test_audited_stream_counts_and_events():
    rng = np.random.default_rng(3)
    draws, q = 400, 0.1
    dismissed = np.array([[2000] * draws, [0] * draws, [5000] * draws])
    rate = np.array([[0.05] * draws, [0.0] * draws, [0.2] * draws])
    s = audited_stream(rng, dismissed, rate, q)
    expected = dismissed * rate
    # the attacks, the audited attacks and the audited benign sessions of every window follow their rates
    assert np.allclose(s["attacks"].mean(1), expected.mean(1), rtol=0.05, atol=1)
    assert np.allclose(s["audited_attacks"].mean(1), (expected * q).mean(1), rtol=0.1, atol=1)
    assert np.allclose(s["audited_benign"].mean(1), ((dismissed - expected) * q).mean(1), rtol=0.05, atol=1)
    assert (s["attacks"][1] == 0).all() and (s["audited_attacks"][1] == 0).all()
    assert s["offsets"].tolist() == [[0] * draws, [2000] * draws, [2000] * draws, [7000] * draws]
    # a virtual first event at position 0, then the audited attacks in the order of the stream, then the padding
    ev_pos, ev_count, ev_win, valid = s["ev_pos"], s["ev_count"], s["ev_window"], s["valid"]
    assert valid[:, 0].all() and (ev_pos[:, 0] == 0).all() and (ev_count[:, 0] == 0).all()
    assert (valid.sum(1) - 1 == s["audited_attacks"].sum(0)).all()
    assert (valid[:, 1:] <= valid[:, :-1]).all()
    k = np.arange(ev_pos.shape[1])
    for d in range(draws):
        pos, count, win = ev_pos[d][valid[d]], ev_count[d][valid[d]], ev_win[d][valid[d]]
        assert (np.diff(pos) > 0).all() and pos[-1] <= 7000
        assert (np.diff(count) >= 1).all() and (count >= k[: len(count)]).all()
        # the window of an event holds its position, and the count at the last event of a window stays within the
        # attacks of the windows up to it
        ends = s["offsets"][1:, d]
        assert (win[1:] == (ends[:, None] < pos[1:][None, :]).sum(0)).all()
        cum = np.cumsum(s["attacks"][:, d])
        assert (count[1:] <= cum[win[1:]]).all()


def test_recalibration_never_tightens_and_sees_every_label_when_nothing_was_dismissed():
    rng = np.random.default_rng(4)
    eps, pi, draws, n = 0.002, 0.2, 50, 50000
    grid = beta.ppf(np.arange(1, 101) / 101, 2, 8)
    b, m = beta.cdf(grid, 2, 8), beta.cdf(grid, 6, 2)[None, :]
    drawn = draw_counts(rng, n, pi, [1.0], b, m, draws)
    attack_below = drawn.family_below.sum(1)
    window_below = drawn.benign_below + attack_below
    n_attacks = drawn.n_family.sum(1)
    full = plug_in(drawn.benign_below, drawn.n_benign, attack_below, n_attacks, eps, pi)
    assert (full > 10).all() and (full < 90).all()
    # with nothing dismissed every label is seen and the refit is the plug in threshold on the whole window
    assert (recalibrate(window_below, attack_below, n, n_attacks, np.full(draws, -1), pi, eps) == full).all()
    # the attacks dismissed at a threshold count as benign, so the refit from a threshold below the plug in one lands
    # at or beyond it, and from one above it never comes back
    assert (recalibrate(window_below, attack_below, n, n_attacks, full - 5, pi, eps) >= full).all()
    assert (recalibrate(window_below, attack_below, n, n_attacks, full + 10, pi, eps) >= full + 10).all()
    # once no attack is escalated nothing can fail and the loosest threshold wins
    assert (recalibrate(window_below, attack_below, n, n_attacks, np.full(draws, 99), pi, eps) == 99).all()


def test_judge_stream_finds_the_violating_window():
    rng = np.random.default_rng(5)
    eps, delta, q, draws = 0.02, 0.05, 0.2, 300
    dismissed = np.array([[3000] * draws, [4000] * draws, [3000] * draws])
    rate = np.array([[0.005] * draws, [0.3] * draws, [0.3] * draws])
    s = audited_stream(rng, dismissed, rate, q)
    seq = Sequence("thinning", *RATIOS, q, q, 0.025, 0.025)
    out = judge_stream(Tables(seq, q, q), s, rate, eps, delta, q)
    total = s["attacks"].sum(0)
    floor = rule.alarm_floor(eps)
    monitor = out["monitor"]
    assert monitor["alarmed"].mean() > 0.95 and (monitor["alarm_pos"][monitor["alarmed"]] >= floor).all()
    # a false alarm in the clean first block is allowed in a few draws
    assert (monitor["alarm_block"][monitor["alarmed"]] >= 1).mean() > 0.95
    assert monitor["false_alarm"].mean() < 0.1 and monitor["over_bound"].mean() < 0.1
    assert (monitor["leak"] <= total).all() and (monitor["sessions"] <= 10000).all()
    assert (monitor["miss_lower"] | monitor["miss_upper"]).mean() < 0.1
    # the state is running or certified before the alarm and violation from its block on
    state = monitor["state"]
    alarmed = monitor["alarmed"]
    assert (state[2][alarmed] == STATE_CODES["violation"]).all()
    assert (state[0] != STATE_CODES["violation"]).mean() > 0.9
    assert (out["onset_true"] == 1).all() and (out["onset_realized"] == 1).all()
    none = out["none"]
    assert (none["leak"] == total).all() and (none["sessions"] == 10000).all() and not none["alarmed"].any()
    # the per window audit alarms at the end of a block, from the second block on
    wa = out["window_audit"]
    assert wa["alarmed"].mean() > 0.9
    ends = s["offsets"][1:]
    rows = np.arange(draws)
    assert (wa["alarm_pos"][wa["alarmed"]] == ends[wa["alarm_block"][wa["alarmed"]], rows[wa["alarmed"]]]).all()
    assert (wa["alarm_block"][wa["alarmed"]] >= 1).all() and not wa["block_alarm"][0].any()
    naive = out["naive"]
    assert naive["alarmed"].mean() > 0.95 and (naive["alarm_pos"][naive["alarmed"]] >= floor).all()
    assert (out["audits"] > 0).all()
    # one segment for the whole stream is the plain monitor, and a sequence per block reads each block on its own
    one = restarted_monitor(Tables(seq, q, q), s, eps, np.array([0, 0, 0]))
    for key in ["alarmed", "alarm_pos", "leak", "sessions", "alarm_block", "false_alarm", "bound", "miss_lower",
                "miss_upper", "state"]:
        assert np.array_equal(one[key], monitor[key]), key
    assert one["sequences"] == 1
    assert (one["failed_sequences"][:, 0] == (monitor["miss_lower"] | monitor["miss_upper"])).all()
    per = restarted_monitor(Tables(seq, q, q), s, eps, np.array([0, 1, 2]))
    assert per["sequences"] == 3 and per["failed_sequences"].shape == (draws, 3)
    assert per["alarmed"].mean() > 0.9 and (per["alarm_block"][per["alarmed"]] >= 1).mean() > 0.95
    starts = s["offsets"][per["alarm_block"].clip(0), rows]
    assert (per["alarm_pos"][per["alarmed"]] - starts[per["alarmed"]] >= floor).all()
    assert per["false_alarm"].mean() < 0.1 and per["over_bound"].mean() < 0.1
    assert (per["leak"] <= total).all() and (per["failed_sequences"].mean(0) < 0.1).all()
    assert (per["state"][2][per["alarmed"]] == STATE_CODES["violation"]).all()


def test_e21_on_an_e11_run(anoshift_signal, tmp_path):
    cfg11 = load_config(configs_dir() / "experiments" / "e11_anoshift_prevalence.yaml", OVERRIDES)
    run11, _ = run_e11(cfg11, None)
    cfg = load_config(configs_dir() / "experiments" / "e21_monitor_streams.yaml",
                      [f"source.run={run11.run_id}", "monitor.epsilon=0.05", "audits.rate=0.2",
                       "compositions=[{name: typical, prevalence: 0.05}, {name: worst, prevalence: 0.2}]"])
    run = Run(cfg, root=tmp_path / "runs")
    module = experiments.get("monitor_streams")
    module.run(cfg, run)
    run.finish()
    claims = runs.load_table(run.dir, "claims")
    blocks = runs.load_table(run.dir, "blocks")
    summary = runs.load_json(run.dir, "summary")
    draws = runs.load_arrays(run.dir, "draws")
    n_draws = cfg11["simulation"]["draws"]
    assert summary["source"]["run"] == run11.run_id and summary["source"]["windows"] == 4
    assert summary["streams"] == STREAMS and summary["main_stream"] == "replay" and summary["chunk"] == 100
    assert len(claims) == 2 * len(POLICIES) * 2 * len(ARMS) and len(blocks) == 2 * len(POLICIES) * 2 * 4
    assert set(claims["policy"]) == set(POLICIES) and set(claims["arm"]) == set(ARMS)
    assert set(claims["stream"]) == set(STREAMS) and set(blocks["stream"]) == set(STREAMS)
    log = (run.dir / "log.txt").read_text()
    assert log.count("online thresholds of the source run reproduced in every window") == 2
    assert "the block histograms of the source run reproduced from the saved scores in every block" in log
    assert log.count("every attack row of a block kept with probability") == 2
    for stream in STREAMS:
        assert f"{stream}, FORGE-Monitor, largest share of draws leaving the bounds" in log
    floor = summary["alarm_floor"]
    assert floor == 20
    for stream in STREAMS:
        for p in POLICIES:
            for co in ["typical", "worst"]:
                key = f"{stream}|{p}|{co}|"
                attacks, dismissed = draws[key + "attacks"], draws[key + "dismissed"]
                ends = np.cumsum(dismissed, axis=0)
                # no monitoring leaks the whole stream, the monitor at most that, and every alarm comes from the floor
                # on
                assert (draws[key + "none_leak"] == attacks.sum(0)).all()
                assert (draws[key + "monitor_leak"] <= draws[key + "none_leak"]).all()
                for arm in [*MONITORS, "naive", "window_audit"]:
                    pos = draws[key + f"{arm}_alarm_pos"]
                    assert ((pos == -1) | (pos >= floor)).all()
                # the per window audit alarms at a block end
                pos = draws[key + "window_audit_alarm_pos"]
                hit = pos >= 0
                assert all(pos[d] in ends[:, d] for d in np.flatnonzero(hit))
                for arm in MONITORS:
                    states = draws[key + f"{arm}_state"]
                    assert states.shape == (4, n_draws) and set(np.unique(states)) <= {0, 1, 2}
                    assert (draws[key + f"{arm}_leak"] <= draws[key + "none_leak"]).all()
                assert (key + "burst" in draws) == (stream == "replay")
    # a sequence per year over the two deployment years, one per block over the four blocks
    sequences = claims.groupby("arm")["sequences"].first()
    assert sequences["monitor"] == 1 and sequences["monitor_year"] == 2 and sequences["monitor_block"] == 4
    assert summary["restart_sequences"] == {"monitor_year": 2, "monitor_block": 4}
    # the recalibration never tightens, so its mean threshold rises or stays from block to block
    for (stream, co), g in blocks[blocks["policy"] == "recalibration"].groupby(["stream", "composition"]):
        t = g.sort_values("window")["threshold_mean"].to_numpy()
        assert (np.diff(t) >= -1e-9).all()
    monitor = claims[claims["arm"] == "monitor"]
    assert (monitor["miscover_rate"] <= 0.05 + 0.1).all()
    assert (monitor[["blocks_certified_mean", "blocks_running_mean", "blocks_violation_mean"]].sum(axis=1)
            .sub(4).abs() < 1e-9).all()
    # the simulated stream of a run with both streams is the one of a run with the simulated stream alone, since the
    # replay spawns its seeds after it
    alone = Run(load_config(configs_dir() / "experiments" / "e21_monitor_streams.yaml",
                            [f"source.run={run11.run_id}", "monitor.epsilon=0.05", "audits.rate=0.2",
                             "compositions=[{name: typical, prevalence: 0.05}, {name: worst, prevalence: 0.2}]",
                             "streams=[simulated]"]), root=tmp_path / "alone")
    module.run(alone.cfg, alone)
    alone.finish()
    alone_claims = runs.load_table(alone.dir, "claims")
    assert "stream" in alone_claims and set(alone_claims["stream"]) == {"simulated"}
    simulated = claims[claims["stream"] == "simulated"].reset_index(drop=True)
    assert simulated.equals(alone_claims)
    alone_draws = runs.load_arrays(alone.dir, "draws")
    assert all(np.array_equal(alone_draws[k], draws[k]) for k in alone_draws)
    # the replay keeps every benign session of the block and thins its attacks to the prevalence, so the attacks of
    # a window at a threshold follow the population FOR of the block as the simulated ones do, and the realized FOR
    # of the whole stream sits at its population value on average
    hist = runs.load_arrays(run11.dir, "histograms")
    for co, pi in (("typical", 0.05), ("worst", 0.2)):
        for p in ["ltt_iid", "fixed_sel"]:
            sim = f"simulated|{p}|{co}|"
            rep = f"replay|{p}|{co}|"
            assert np.allclose(draws[rep + "for_true"], draws[sim + "for_true"])
            expected = draws[rep + "for_true"] * draws[rep + "dismissed"]
            assert np.allclose(draws[rep + "attacks"].mean(1), expected.mean(1), rtol=0.15, atol=3)
            g = blocks[(blocks["stream"] == "replay") & (blocks["policy"] == p) & (blocks["composition"] == co)]
            assert (g["burst_max_median"] >= 0).all() and (g["burst_max_high"] >= g["burst_max_median"]).all()
            # the window keeps every benign session, so the dismissed count is at least the benign count at the
            # threshold
            thresholds = runs.load_arrays(run11.dir, "draws")[f"{p}_index"]
            for w in range(4):
                benign = np.where(thresholds >= 0, hist["window_benign_below"][w][np.maximum(thresholds, 0)], 0)
                assert (draws[rep + "dismissed"][w] >= benign).all()
    assert blocks.loc[blocks["stream"] == "simulated", "burst_max_median"].isna().all()
    for name in module.FIGURES:
        assert (run.dir / "figures" / f"{name}.png").exists() and (run.dir / "figures" / f"{name}.pdf").exists()
