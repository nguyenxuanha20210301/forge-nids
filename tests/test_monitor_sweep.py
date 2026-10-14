import numpy as np
from scipy.stats import beta

from forge import experiments, runs
from forge.certify.certificate import for_from_rates
from forge.config import load_config
from forge.experiments.monitor_streams import judge_stream
from forge.experiments.monitor_sweep import (any_start_monitor, audited_stream, cells_of, count_at, dismissed_shares,
                                             monitor_cell, score_bins, step_monitor)
from forge.monitor import rule
from forge.monitor.sequence import Sequence, Tables
from forge.paths import configs_dir
from forge.runs import Run

from test_for_vs_prevalence import OVERRIDES, run_e11

RATIOS = ([1.5, 2, 3, 5, 10, 30, 100], [0, 0.25, 0.5, 0.75])


def synthetic_hist(g=100, n_benign=100000, n_attack=5000):
    grid = beta.ppf(np.arange(1, g + 1) / (g + 1), 2, 8)
    b, m = beta.cdf(grid, 2, 8), beta.cdf(grid, 6, 2)
    return {"selection_benign_below": np.rint(b * n_benign).astype(int),
            "selection_family_below": np.rint(m * n_attack).astype(int)[None, :],
            "selection_sizes": np.array([n_benign, n_attack])}, b, m


def test_score_bins_and_dismissed_shares():
    hist, b, m = synthetic_hist()
    bins = score_bins(hist, 0.02, 10)
    last = bins["last"]
    assert (np.diff(last) > 0).all() and last[-1] == 99 and len(last) == 10
    assert (bins["r"] >= 0).all() and (bins["r"] <= 1).all() and np.isclose(bins["mass"].sum(), 1)
    # the attack share rises toward the threshold where the attack scores sit
    assert bins["r"][-1] > bins["r"][0]
    pi = 0.05
    t = np.array([99, 40, 0, -1])
    attack, benign = dismissed_shares(b, m, pi, last, t)
    assert attack.shape == (4, 10) and np.allclose((attack + benign).sum(1)[:3], 1)
    assert np.allclose(attack.sum(1)[:3], for_from_rates(pi, m[t[:3]], b[t[:3]]))
    assert (attack[3] == 0).all() and (benign[3] == 0).all()
    # a threshold inside a bin leaves the bins above it empty
    assert (attack[1, last > 40][1:] == 0).all() and (benign[1, last > 40][1:] == 0).all()


def test_audited_stream_with_bins_follows_its_rates():
    rng = np.random.default_rng(8)
    draws, q = 400, np.array([0.05, 0.3])
    dismissed = np.array([[4000] * draws, [3000] * draws])
    attack_share = np.zeros((2, draws, 2))
    attack_share[0, :, 0], attack_share[0, :, 1] = 0.02, 0.08
    attack_share[1, :, 0], attack_share[1, :, 1] = 0.1, 0.1
    benign_share = np.zeros((2, draws, 2))
    benign_share[0, :, 0], benign_share[0, :, 1] = 0.6, 0.3
    benign_share[1, :, 0], benign_share[1, :, 1] = 0.5, 0.3
    s = audited_stream(rng, dismissed, attack_share, benign_share, q)
    expected_attacks = dismissed * attack_share.sum(2)
    expected_audited = dismissed * (attack_share * q).sum(2)
    assert np.allclose(s["attacks"].mean(1), expected_attacks.mean(1), rtol=0.05)
    assert np.allclose(s["audited_attacks"].mean(1), expected_audited.mean(1), rtol=0.1)
    assert np.allclose(s["audited_benign"].mean(1), (dismissed * (benign_share * q).sum(2)).mean(1), rtol=0.05)
    valid, ev_q, ev_win = s["valid"], s["ev_q"], s["ev_window"]
    real = valid.copy()
    real[:, 0] = False
    assert set(np.unique(ev_q[real])) <= {0.05, 0.3} and (ev_q[~real] == 1).all()
    # the bin of an audited attack follows the share of the audited attacks of its window in that bin
    first = real & (ev_win == 0)
    share_high = (ev_q[first] == 0.3).mean()
    assert abs(share_high - 0.08 * 0.3 / (0.02 * 0.05 + 0.08 * 0.3)) < 0.03


def test_monitor_cell_matches_e21_and_the_rules_hold_together():
    rng = np.random.default_rng(9)
    eps, delta, q, draws = 0.02, 0.05, 0.2, 300
    dismissed = np.array([[3000] * draws, [4000] * draws, [3000] * draws])
    rate = np.array([[0.005] * draws, [0.3] * draws, [0.3] * draws])
    attack_share, benign_share = rate[:, :, None], (1 - rate)[:, :, None]
    s = audited_stream(rng, dismissed, attack_share, benign_share, np.array([q]))
    seq = Sequence("thinning", *RATIOS, q, q, 0.025, 0.025)
    tables = Tables(seq, q, q)
    # with one bin and one segment the cell is the monitor of e21 to the draw
    out = judge_stream(tables, s, rate, eps, delta, q)["monitor"]
    cell = monitor_cell(seq, tables, s, eps, np.zeros(3, dtype=int), 0, True)
    for key in ["alarmed", "alarm_pos", "leak", "false_alarm", "bound", "miss_lower", "miss_upper", "state"]:
        assert np.array_equal(cell[key], out[key]), key
    # a delay moves the alarm and adds the attacks of the meantime
    late = monitor_cell(seq, tables, s, eps, np.zeros(3, dtype=int), 500, True)
    alarmed = late["alarmed"]
    assert np.array_equal(alarmed, cell["alarmed"])
    assert (late["alarm_pos"][alarmed] == np.minimum(cell["alarm_pos"][alarmed] + 500, 10000)).all()
    assert (late["leak"][alarmed] >= cell["leak"][alarmed]).all() and (late["leak_in_delay"][alarmed] >= 0).all()
    # the any start rule at delta over the blocks alarms no later than a sequence per block at the same level
    per = Sequence("thinning", *RATIOS, q, q, 0.025 / 3, 0.025 / 3)
    per_tables = Tables(per, q, q)
    block = monitor_cell(per, per_tables, s, eps, np.arange(3), 0, True)
    any_start = any_start_monitor(per_tables, s, eps)
    both = block["alarmed"] & any_start["alarmed"]
    assert any_start["alarmed"].mean() >= block["alarmed"].mean() - 0.02
    assert (any_start["alarm_pos"][both] <= block["alarm_pos"][both]).all()
    assert (any_start["miss_lower"] | any_start["miss_upper"]).mean() < 0.1
    assert any_start["sequences"] == 3 and (any_start["leak"] <= s["attacks"].sum(0)).all()
    # the per step target alarms on the violation and not on the clean block
    step = step_monitor(s, eps, q, 0.025, np.array([0, 1, 1]))
    assert step["alarmed"].mean() > 0.9 and (step["alarm_block"][step["alarmed"]] >= 1).mean() > 0.95
    assert step["false_alarm"].mean() < 0.1 and (step["alarm_pos"][step["alarmed"]] >= rule.alarm_floor(eps)).all()
    # score dependent audits read the bounds from the probabilities of the events and keep covering
    q2 = np.array([0.1, 0.3])
    share2 = np.stack([attack_share[..., 0] * 0.4, attack_share[..., 0] * 0.6], axis=2)
    benign2 = np.stack([benign_share[..., 0] * 0.5, benign_share[..., 0] * 0.5], axis=2)
    s2 = audited_stream(np.random.default_rng(10), dismissed, share2, benign2, q2)
    seq2 = Sequence("thinning", *RATIOS, 0.2, 0.1, 0.025, 0.025)
    scored = monitor_cell(seq2, Tables(seq2, 0.3, 0.1), s2, eps, np.array([0, 0, 1]), 0, False)
    assert scored["alarmed"].mean() > 0.9 and scored["false_alarm"].mean() < 0.1
    assert scored["failed_sequences"].mean(0).max() < 0.1 and scored["over_bound"].mean() < 0.1
    assert scored["sequences"] == 2


def test_count_at_is_exact_at_block_ends():
    rng = np.random.default_rng(11)
    draws = 50
    dismissed = np.array([[1000] * draws, [500] * draws])
    rate = np.array([[0.1] * draws, [0.2] * draws])
    s = audited_stream(rng, dismissed, rate[:, :, None], (1 - rate)[:, :, None], np.array([0.1]))
    cum = np.cumsum(s["attacks"], axis=0)
    assert np.array_equal(count_at(s, np.full(draws, 1000)), cum[0])
    assert np.array_equal(count_at(s, np.full(draws, 1500)), cum[1])
    assert np.array_equal(count_at(s, np.zeros(draws, dtype=int)), np.zeros(draws, dtype=int))
    middle = count_at(s, np.full(draws, 500))
    assert (middle >= 0).all() and (middle <= cum[0]).all() and np.allclose(middle, cum[0] / 2, atol=1)


def test_e22_on_an_e11_run(anoshift_signal, tmp_path):
    cfg11 = load_config(configs_dir() / "experiments" / "e11_anoshift_prevalence.yaml", OVERRIDES)
    run11, _ = run_e11(cfg11, None)
    cfg = load_config(configs_dir() / "experiments" / "e22_monitor_sweep.yaml",
                      [f"source.run={run11.run_id}", "monitor.epsilon=0.05", "audits.rate=0.2", "audits.floor=0.02",
                       "bins=4", "sweep.rates=[0.1]", "sweep.allocations=[{allocation: sqrt, floor: 0.02}]",
                       "sweep.delays=[200]",
                       "compositions=[{name: typical, prevalence: 0.05}, {name: worst, prevalence: 0.2}]"])
    cells = cells_of(cfg)
    assert [c["axis"] for c in cells] == ["main", "rate", "allocation", "kind", "restart", "restart", "restart",
                                          "delay", "target"]
    run = Run(cfg, root=tmp_path / "runs")
    module = experiments.get("monitor_sweep")
    module.run(cfg, run)
    run.finish()
    claims = runs.load_table(run.dir, "claims")
    summary = runs.load_json(run.dir, "summary")
    policies = cfg["policies"]
    assert len(claims) == len(policies) * 2 * len(cells)
    assert set(claims["name"]) == {c["name"] for c in cells}
    grid = runs.load_arrays(run11.dir, "histograms")["grid"]
    assert len(summary["bin_last"]) <= 4 and summary["bin_last"][-1] == len(grid) - 1
    draws = runs.load_arrays(run.dir, "draws")
    for p in policies:
        for co in ["typical", "worst"]:
            total = draws[f"{p}|{co}|main|leak"]
            for c in cells:
                key = f"{p}|{co}|{c['name']}|"
                pos = draws[key + "alarm_pos"]
                assert ((pos == -1) | (pos >= summary["alarm_floor"])).all()
    # the per step target bounds no count, every other cell holds its level per sequence
    step = claims[claims["target"] == "step"]
    assert step["bound_median"].isna().all()
    assert (claims.loc[claims["target"] == "count", "miscover_sequence_rate"] <= 0.05 + 0.1).all()
    # the audits per million follow the audit policy of the cell
    main = claims[claims["name"] == "main"]
    assert (main["audits_per_million"].between(150000, 250000)).all()
    for name in module.FIGURES:
        assert (run.dir / "figures" / f"{name}.png").exists()
    log = (run.dir / "log.txt").read_text()
    assert "largest share of sequences leaving the bounds" in log
