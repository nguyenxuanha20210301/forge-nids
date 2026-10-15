import numpy as np
import pandas as pd
import pytest

from forge import experiments, runs
from forge.config import load_config
from forge.experiments.monitor_streams import audited_stream, judge_stream
from forge.experiments.unseen_family import ARMS, MONITORS, e12_window_seed, unseen_metrics, units_of, window_size
from forge.monitor.sequence import Sequence, Tables
from forge.paths import configs_dir
from forge.runs import Run

from test_litnet_prevalence import OVERRIDES, run_e12

RATIOS = ([1.5, 2, 3, 5, 10, 30, 100], [0, 0.25, 0.5, 0.75])


def test_window_size_follows_the_real_windows_of_e12():
    # enough attacks for the prevalence, the window keeps every benign flow
    assert window_size(9800, [500], 0.02, [1.0], True) == (10000, False)
    # too few attacks, the window keeps all of them and the benign flows are subsampled, or the window is left out
    assert window_size(98000, [500], 0.02, [1.0], True) == (25000, True)
    assert window_size(98000, [500], 0.02, [1.0], False) == (0, False)
    # with two families the smaller one against its share decides, and a family of the mix with no attack leaves the
    # window out
    assert window_size(98000, [600, 200], 0.02, [0.75, 0.25], True) == (int(np.floor(200 / (0.02 * 0.25))), True)
    assert window_size(98000, [600, 0], 0.02, [0.75, 0.25], True) == (0, False)
    cfg12 = {"deployment": {"prevalences": [0.01, 0.02, 0.1], "mixes": ["window", "calibration"]}}
    # the real arm comes third in e12, so its seeds start after two arms of cells
    assert e12_window_seed(cfg12, 11, 0, 0.01) == 1 + 2 * 11 * 3 * 2
    assert e12_window_seed(cfg12, 11, 3, 0.1) == 1 + 2 * 66 + (3 * 3 + 2) * 2
    with pytest.raises(ValueError, match="no real window"):
        e12_window_seed(cfg12, 11, 0, 0.05)
    # a unit names the families of one window of a capture with benign flows, as the cuts table joins them
    segments = pd.DataFrame({"families": ["udp_flood", "land", "smurf+icmp_flood"], "filtered": [True, False, False]})
    units = units_of(["smurf+icmp_flood", "land"], segments)
    assert [u["segment"] for u in units] == [2, 1] and units[0]["families"] == ["smurf", "icmp_flood"]
    with pytest.raises(ValueError, match="no window"):
        units_of(["udp_flood"], segments)
    with pytest.raises(ValueError, match="no window"):
        units_of(["smurf"], segments)
    with pytest.raises(ValueError, match="twice"):
        units_of(["land", "land"], segments)


def test_unseen_metrics_partition_the_alarms_and_bound_the_leakage():
    rng = np.random.default_rng(6)
    eps, delta, q, draws = 0.02, 0.05, 0.2, 300
    dismissed = np.array([[3000] * draws, [4000] * draws, [3000] * draws])
    rate = np.array([[0.005] * draws, [0.3] * draws, [0.005] * draws])
    s = audited_stream(rng, dismissed, rate, q)
    seq = Sequence("thinning", *RATIOS, q, q, 0.025, 0.025)
    out = judge_stream(Tables(seq, q, q), s, rate, eps, delta, q)
    for arm in ["monitor", "window_audit", "naive", "none"]:
        r = out[arm]
        um = unseen_metrics(r, s, 1)
        parts = um["inside"].astype(int) + um["before"].astype(int) + um["after"].astype(int)
        assert (parts == r["alarmed"].astype(int)).all()
        assert (um["leak"] >= 0).all() and (um["leak"] <= s["attacks"][1]).all()
        assert (um["detect"][um["inside"]] >= 1).all() and (um["detect"][um["inside"]] <= 4000).all()
        assert (um["detect"][~um["inside"]] == -1).all()
        assert (um["share"] >= 0).all() and (um["share"] <= 1).all()
        # an alarm before the window leaks nothing of its family, none or one after it leaks all of it
        assert (um["leak"][um["before"]] == 0).all()
        assert (um["leak"][~r["alarmed"] | um["after"]] == s["attacks"][1][~r["alarmed"] | um["after"]]).all()
    monitor = unseen_metrics(out["monitor"], s, 1)
    assert monitor["inside"].mean() > 0.9 and monitor["leak"].mean() < s["attacks"][1].mean()
    # the per window audit alarms at the window's end, so the whole window is deployed and all of its attacks leak
    wa = unseen_metrics(out["window_audit"], s, 1)
    assert (wa["detect"][wa["inside"]] == 4000).all()
    assert (wa["leak"][wa["inside"]] == s["attacks"][1][wa["inside"]]).all()
    # a window before the violation is never the one with the alarm
    first = unseen_metrics(out["monitor"], s, 0)
    assert first["inside"].mean() < 0.1 and (first["after"] | ~out["monitor"]["alarmed"]).mean() > 0.9
    assert (first["leak"][first["after"]] == s["attacks"][0][first["after"]]).all()


def test_e23_on_an_e12_run(litnet_signal, tmp_path):
    cfg12 = load_config(configs_dir() / "experiments" / "e12_litnet_prevalence.yaml", OVERRIDES)
    run12, _ = run_e12(cfg12, None)
    cfg = load_config(configs_dir() / "experiments" / "e23_unseen_family.yaml",
                      [f"source.run={run12.run_id}", "monitor.epsilon=0.05", "audits.rate=0.2",
                       "units=[syn_flood, smurf]",
                       "compositions=[{name: typical, known: 0.05, unseen: 0.05}, {name: worst, known: 0.05, "
                       "unseen: 0.2}]", "simulation.trace={unit: smurf, composition: worst, policy: forge_simplex}"])
    run = Run(cfg, root=tmp_path / "runs")
    module = experiments.get("unseen_family")
    module.run(cfg, run)
    run.finish()
    summary = runs.load_json(run.dir, "summary")
    claims = runs.load_table(run.dir, "claims")
    blocks = runs.load_table(run.dir, "blocks")
    units = runs.load_table(run.dir, "units")
    windows = runs.load_table(run.dir, "windows")
    draws = runs.load_arrays(run.dir, "draws")
    hist = runs.load_arrays(run.dir, "histograms")
    log = (run.dir / "log.txt").read_text()
    policies, arms = cfg["policies"], cfg["arms"]
    n_draws = cfg12["simulation"]["draws"]
    # the udp flood capture has no window, the icmp flood capture is cut into two, which makes six windows
    assert summary["windows"] == ["Reaper worm", "SYN flood", "spam bot", "Code Red worm", "Smurf", "ICMP flood"]
    assert [u["window"] for u in summary["units"]] == [1, 4] and [u["label"] for u in summary["units"]] == \
        ["SYN flood", "Smurf"]
    # the control reproduces the split, the offline thresholds and the real windows of the source run
    assert "offline thresholds of the source run reproduced" in log and summary["source_checked"]
    assert len(claims) == 2 * 2 * 2 * len(policies) * len(arms)
    assert len(blocks) == 2 * 2 * 2 * len(policies) * 6
    assert set(claims["case"]) == {"unseen", "known"} and set(claims["arm"]) == set(arms)
    assert units["unit"].tolist() == ["all_known", "syn_flood", "smurf"]
    # the held out family leaves the training, selection and calibration parts and keeps its test attacks
    families = summary["source"]["families"]
    k_syn = families.index("syn_flood") + 1
    assert hist["syn_flood|calibration_sizes"][k_syn] == 0 and hist["syn_flood|selection_sizes"][k_syn] == 0
    assert hist["all_known|calibration_sizes"][k_syn] > 0
    assert hist["syn_flood|window_sizes"][:, k_syn].sum() == hist["all_known|window_sizes"][:, k_syn].sum() > 0
    assert units.loc[1, "train_attacks"] < units.loc[0, "train_attacks"]
    assert units.loc[1, "families"].count("syn_flood") == 0 and units.loc[0, "families"].count("syn_flood") == 1
    own = windows[(windows["unit"] == "syn_flood") & windows["unseen"]]
    assert own["segment"].tolist() == ["SYN flood"] and not own["in_claim"].iloc[0]
    assert windows[(windows["unit"] == "all_known") & windows["unseen"]].empty
    floor = summary["alarm_floor"]
    assert floor == 20
    for u in summary["units"]:
        for case in ["unseen", "known"]:
            for co in ["typical", "worst"]:
                for p in policies:
                    key = f"{u['name']}|{case}|{co}|{p}|"
                    attacks = draws[key + "attacks"]
                    assert attacks.shape == (6, n_draws)
                    assert (draws[key + "none_leak"] == attacks.sum(0)).all()
                    assert (draws[key + "none_unseen_leak"] == attacks[u["window"]]).all()
                    for arm in arms:
                        pos = draws[key + f"{arm}_alarm_pos"]
                        assert ((pos == -1) | (pos >= floor)).all()
                        assert (draws[key + f"{arm}_leak"] <= attacks.sum(0)).all()
                        assert (draws[key + f"{arm}_unseen_leak"] <= attacks[u["window"]]).all()
    # the window of the family at the unseen prevalence, the known ones at theirs, with the sizes of the real windows
    # of e12
    claims12 = runs.load_table(run12.dir, "claims")
    real = claims12[(claims12["arm"] == "real") & (claims12["mix"] == "window")]
    for row in blocks.drop_duplicates(["unit", "composition", "window"]).itertuples():
        unseen_pi = 0.2 if row.composition == "worst" else 0.05
        assert row.prevalence == (unseen_pi if row.unseen else 0.05)
        i = summary["source"]["segments"].index(row.segment)
        saved = real[(real["window"] == i) & (real["prevalence"] == row.prevalence)]["window_sessions"].unique()
        assert saved.tolist() == [row.window_sessions]
    # the known case at the typical composition is one stream for every unit, so its rows agree up to the columns
    # that speak of the family's window
    typical = claims[(claims["case"] == "known") & (claims["composition"] == "typical")]
    shared = ["alarm_rate", "false_alarm_rate", "leak_median", "sessions_median", "audits_mean"]
    for (p, arm), g in typical.groupby(["policy", "arm"]):
        assert len(g) == 2 and (g[shared].nunique() == 1).all()
    monitors = claims[claims["arm"].isin(MONITORS)]
    assert (monitors["miscover_rate"] <= 0.05 + 0.1).all()
    assert (monitors[["windows_certified_mean", "windows_running_mean", "windows_violation_mean"]].sum(axis=1)
            .sub(6).abs() < 1e-9).all()
    rates = claims[["alarm_in_unseen_rate", "alarm_before_unseen_rate", "alarm_after_unseen_rate"]].sum(axis=1)
    assert np.allclose(rates, claims["alarm_rate"])
    # the share of alarms inside the window among the violating draws exists where some draw violates
    given = claims["alarm_in_unseen_violated_rate"]
    assert (given.isna() == (claims["unseen_violated_rate"] == 0)).all()
    assert (given.dropna() >= 0).all() and (given.dropna() <= 1).all()
    assert summary["restart_sequences"] == {"monitor_window": 6, "monitor_any": 6}
    assert (claims.loc[claims["arm"] == "monitor_any", "sequences"] == 6).all()
    assert set(ARMS) >= set(arms)
    for name in module.FIGURES:
        assert (run.dir / "figures" / f"{name}.png").exists() and (run.dir / "figures" / f"{name}.pdf").exists()
    assert "largest share of draws leaving the bounds" in log


def test_e23_with_its_own_seed_refits_everything_and_is_e12_at_that_seed(litnet_signal, tmp_path):
    # the source run has seed 0. with seed 1 the scorer, the calibration sets, the windows and the coins take seed 1,
    # the control is not checked against the source run, and its thresholds are those of e12 at seed 1
    cfg12 = load_config(configs_dir() / "experiments" / "e12_litnet_prevalence.yaml", OVERRIDES)
    run12, _ = run_e12(cfg12, None)
    cfg = load_config(configs_dir() / "experiments" / "e23_unseen_family.yaml",
                      [f"source.run={run12.run_id}", "monitor.epsilon=0.05", "audits.rate=0.2", "units=[syn_flood]",
                       "compositions=[{name: typical, known: 0.05, unseen: 0.05}]",
                       "simulation.trace={unit: syn_flood, composition: typical, policy: forge_simplex}",
                       "policies=[forge_simplex, forge_online]", "arms=[monitor, none]", "seed=1"])
    run = Run(cfg, root=tmp_path / "runs")
    experiments.get("unseen_family").run(cfg, run)
    run.finish()
    log = (run.dir / "log.txt").read_text()
    summary = runs.load_json(run.dir, "summary")
    assert "seed 1 differs from the source run's 0" in log and not summary["source_checked"]
    assert summary["seed"] == 1 and summary["source_seed"] == 0
    hist = runs.load_arrays(run.dir, "histograms")
    source = runs.load_arrays(run12.dir, "draws")
    assert not np.array_equal(hist["all_known|forge_simplex_index"], source["forge_simplex_index"]) or \
        not np.array_equal(hist["all_known|counts"], source["counts"])
    cfg12_1 = load_config(configs_dir() / "experiments" / "e12_litnet_prevalence.yaml", OVERRIDES + ["seed=1"])
    run12_1, _ = run_e12(cfg12_1, tmp_path / "e12")
    twin = runs.load_arrays(run12_1.dir, "draws")
    assert np.array_equal(hist["all_known|counts"], twin["counts"])
    for p in ["forge_simplex", "ltt_iid", "conformal_fnr", "fixed_sel"]:
        assert np.array_equal(hist[f"all_known|{p}_index"], twin[f"{p}_index"])
    units = runs.load_table(run.dir, "units")
    assert units.loc[0, "best_iteration"] == runs.load_json(run12_1.dir, "scorer")["best_iteration"]
