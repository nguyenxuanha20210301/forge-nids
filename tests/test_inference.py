import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom

from forge import runs
from forge.config import load_config
from forge.eval import inference, seeds
from forge.eval.metrics import share_with_interval
from forge.paths import configs_dir

from test_seeds import fake_run
from test_ugr16_prevalence import OVERRIDES, run_e13

SUMMARY = {"dataset": "litnet2020", "epsilon": 0.001, "delta": 0.05, "pi_max": 0.1, "draws": 1000,
           "main_prevalence": 0.1}


def test_sign_test_holm_and_spread():
    test = inference.sign_test([0.1] * 20)
    assert (test["above"], test["tied"], test["below"]) == (20, 0, 0) and test["p"] == pytest.approx(2 * 0.5 ** 20)
    # a difference within rounding of zero is a tie, and ties leave the test
    test = inference.sign_test([0.0, 0.0, 1e-13, 0.2, -0.1, np.nan])
    assert (test["above"], test["tied"], test["below"]) == (1, 3, 1) and test["p"] == pytest.approx(1.0)
    assert inference.sign_test([0.0, 0.0])["p"] == 1.0 and np.isnan(inference.sign_test([np.nan])["p"])
    # Holm on hand worked values, a missing p value left out of the family
    adjusted = inference.holm([0.01, 0.04, 0.03, 0.005, np.nan])
    np.testing.assert_allclose(adjusted[:4], [0.03, 0.06, 0.06, 0.02])
    assert np.isnan(adjusted[4])
    np.testing.assert_allclose(inference.holm([0.6, 0.7]), [1.0, 1.0])
    assert inference.holm([]).shape == (0,)
    stats = inference.spread([1.0, 2.0, 3.0, 4.0, np.nan], "x")
    assert stats == pytest.approx({"x_median": 2.5, "x_low": 1.15, "x_high": 3.85, "x_min": 1.0, "x_max": 4.0})
    table = inference.adjust(pd.DataFrame({"a_p": [0.01, 0.04], "b_p": [0.03, 0.005]}), ["a", "b"])
    assert table.columns.tolist() == ["a_p", "a_p_holm", "b_p", "b_p_holm"]
    np.testing.assert_allclose(table[["a_p_holm", "b_p_holm"]].to_numpy().ravel(), [0.03, 0.06, 0.06, 0.02])


def e12_rows(seed):
    # the by_prevalence rows of an e12 run, offline FORGE and LTT on the windows RQ1 is read on and on two other cuts
    # whose numbers would move every statistic were they read
    rows = []
    for arm, subset in (("reference", "in_claim"), ("reference", "all"), ("s2", "in_claim")):
        rq1 = (arm, subset) == ("reference", "in_claim")
        for method in ("forge_simplex", "ltt_iid"):
            forge = method == "forge_simplex"
            for pi in (0.02, 0.1):
                rows.append({"arm": arm, "subset": subset, "method": method, "prevalence": pi, "mix": "window",
                             "windows": 7, "pairs": 7000,
                             "violation_rate": (0.0 if forge else 0.5 + 0.1 * seed) if rq1 else 0.9,
                             "violation_window_max": 0.0 if forge else 0.9, "refused": 0.01 * seed if forge else 0.0,
                             "dismissed_mean": (0.6 + 0.01 * seed if forge else 0.8) if rq1 else 0.1,
                             "for_median": 1e-4, "best_up_to_pi_max_dismissed": 0.75,
                             "best_at_prevalence_dismissed": 0.8, "realized_rate": 0.0 if forge else 0.2,
                             "best_every_mix_dismissed": 0.6})
    return pd.DataFrame(rows)


def test_e1_price_baselines_and_intervals_on_three_seeds(tmp_path):
    root = tmp_path / "runs"
    folders = {seed: fake_run(root, "e12_litnet_prevalence", f"20261010-00000{seed}_a",
                              {"seed": seed, "experiment": {"name": "e12_litnet_prevalence"}},
                              tables=[("by_prevalence", e12_rows(seed))], summary=SUMMARY) for seed in (0, 1, 2)}
    rows = inference.e1_seed_rows(folders, "e12_litnet_prevalence")
    # the rows of every seed are the windows RQ1 is read on, as the pooled E1 table cuts them
    assert len(rows) == 12 and (rows["dataset"] == "litnet2020").all() and set(rows["seed"]) == {0, 1, 2}
    assert rows["run_id"].tolist()[:4] == ["20261010-000000_a"] * 4 and (rows["draws"] == 1000).all()

    price = inference.e1_paired_price(rows).set_index("method")
    assert len(price) == 2 and (price["prevalence"] == 0.1).all() and (price["seeds"] == 3).all()
    forge = price.loc["forge_simplex"]
    # the yardstick dismisses 0.75 against FORGE's 0.60, 0.61 and 0.62, a price above zero in every seed
    assert forge["price_median"] == pytest.approx(0.14) and forge["price_min"] == pytest.approx(0.13)
    assert forge["price_low"] == pytest.approx(0.131) and forge["price_per_million_median"] == pytest.approx(140000)
    assert (forge["price_seeds_above"], forge["price_seeds_tied"], forge["price_seeds_below"]) == (3, 0, 0)
    assert forge["price_p"] == pytest.approx(0.25)
    # the every mix yardstick at 0.60 ties seed 0 and lies below the others
    assert (forge["price_every_mix_seeds_tied"], forge["price_every_mix_seeds_below"]) == (1, 2)
    assert forge["price_every_mix_p"] == pytest.approx(0.5)
    ltt = price.loc["ltt_iid"]
    assert ltt["price_median"] == pytest.approx(-0.05) and ltt["price_seeds_below"] == 3
    p = price[["price_p", "price_every_mix_p"]].to_numpy().ravel()
    np.testing.assert_allclose(price[["price_p_holm", "price_every_mix_p_holm"]].to_numpy().ravel(),
                               inference.holm(p))

    paired = inference.e1_paired_baselines(rows)
    assert paired["method"].tolist() == ["ltt_iid", "ltt_iid"] and paired["prevalence"].tolist() == [0.02, 0.1]
    row = paired.set_index("prevalence").loc[0.1]
    assert row["main"] and row["seeds"] == 3
    assert row["violation_diff_median"] == pytest.approx(0.6) and row["violation_diff_seeds_above"] == 3
    assert row["dismissed_diff_median"] == pytest.approx(0.19) and row["dismissed_diff_max"] == pytest.approx(0.2)
    assert row["violation_diff_p"] == pytest.approx(0.25) and "dismissed_diff_p_holm" in paired

    # the intervals count the draws of every seed, the violations of a draw being the share of its windows that break
    intervals = inference.e1_intervals(rows).set_index(["method", "prevalence"])
    expected = share_with_interval(1800, 3000)
    assert intervals.loc[("ltt_iid", 0.1), "violation_rate_ci_low"] == pytest.approx(expected["low"])
    assert intervals.loc[("ltt_iid", 0.1), "violation_rate_ci_high"] == pytest.approx(expected["high"])
    assert intervals.loc[("forge_simplex", 0.1), "refused_ci_low"] == pytest.approx(share_with_interval(30, 3000)["low"])
    tables = seeds.pool_experiment("e12_litnet_prevalence", root, log=lambda s: None)
    e1 = tables["e1_for_by_prevalence"]
    e1.insert(0, "dataset", "litnet2020")
    e1.insert(1, "epsilon", 0.001)
    table = inference.with_intervals(e1, inference.e1_intervals(rows), inference.E1_KEYS)
    assert len(table) == len(e1)
    at = table.columns.get_loc("violation_rate_pooled")
    assert table.columns[at + 1:at + 3].tolist() == ["violation_rate_ci_low", "violation_rate_ci_high"]
    for share in ("violation_rate", "refused"):
        assert (table[f"{share}_ci_low"] <= table[f"{share}_pooled"] + 1e-12).all()
        assert (table[f"{share}_pooled"] <= table[f"{share}_ci_high"] + 1e-12).all()
    pd.testing.assert_frame_equal(table[e1.columns], e1)
    # a dataset without a row at pi_max has no price
    with pytest.raises(ValueError, match="no E1 row at pi_max"):
        inference.e1_paired_price(rows.assign(pi_max=0.3))


def claims_rows(seed):
    # the claims of a run of RQ1, FORGE inside and outside its claim, LTT and the real arm, which the claim leaves out
    violations = {("forge_simplex", "reference"): ({0: 3, 1: 120, 2: 0}[seed], 10),
                  ("forge_online", "reference"): (1, 2), ("forge_simplex", "s2"): (300, 0),
                  ("forge_online", "s2"): (0, 7)}
    rows = []
    for method in ("forge_simplex", "forge_online", "ltt_iid"):
        for arm in ("reference", "s2", "real"):
            for window, segment in enumerate(["a", "b", "c"]):
                for pi in (0.1, 0.2):
                    inside = (method, arm) in violations and window < 2 and pi <= 0.1
                    v = violations[(method, arm)][window] if inside else 900
                    rows.append({"method": method, "arm": arm, "window": window, "segment": segment,
                                 "prevalence": pi, "mix": "window", "in_claim": window < 2, "draws": 1000,
                                 "violations": v, **share_with_interval(v, 1000)})
    return pd.DataFrame(rows)


def test_e1_violation_of_forge_by_seed_and_over_seeds(tmp_path):
    folders = {seed: fake_run(tmp_path, "e12_litnet_prevalence", f"20261010-00000{seed}_a", {"seed": seed},
                              tables=[("claims", claims_rows(seed))], summary=SUMMARY) for seed in (0, 1, 2)}
    cells = inference.e1_claim_cells(folders)
    # offline and online FORGE on the reference and S2 windows inside the claim up to pi_max
    assert len(cells) == 3 * 2 * 2 * 2 and set(cells["method"]) == {"forge_simplex", "forge_online"}
    assert set(cells["arm"]) == {"reference", "s2"} and (cells["prevalence"] <= 0.1).all()
    assert (cells["violations"] < 900).all()

    by_seed = inference.e1_violation_by_seed(cells)
    assert len(by_seed) == 3 * 4
    ref = by_seed[(by_seed["arm"] == "reference") & (by_seed["method"] == "forge_simplex")].set_index("seed")
    assert ref["window"].tolist() == [1, 0, 1] and ref["segment"].tolist() == ["b", "a", "b"]
    assert ref["violations"].tolist() == [10, 120, 10] and ref["above_delta"].tolist() == [False, True, False]
    assert ref.loc[1, "low"] == pytest.approx(share_with_interval(120, 1000)["low"])

    over = inference.e1_violation_over_seeds(cells).set_index(["arm", "method"])
    row = over.loc[("reference", "forge_simplex")]
    assert row["seeds"] == 3 and row["cells"] == 2
    assert (row["seeds_above_delta"], row["seeds_rate_above_delta"]) == (1, 1)
    assert row["largest_median"] == pytest.approx(0.01) and row["largest_max"] == pytest.approx(0.12)
    # the window whose share over the draws of every seed is the largest, window 0 with 123 of 3,000
    assert (row["window"], row["segment"], row["violations"], row["draws"]) == (0, "a", 123, 3000)
    assert row["rate"] == pytest.approx(0.041) and row["low"] == pytest.approx(share_with_interval(123, 3000)["low"])
    assert row["p"] == pytest.approx(binom.sf(122, 3000, 0.05))
    # Holm over the eight windows and prevalences of the table
    pooled = {("reference", "forge_simplex"): (123, 30), ("reference", "forge_online"): (3, 6),
              ("s2", "forge_simplex"): (900, 0), ("s2", "forge_online"): (0, 21)}
    p = np.array([binom.sf(v - 1, 3000, 0.05) for pair in pooled.values() for v in pair])
    adjusted = dict(zip([(k, w) for k in pooled for w in (0, 1)], inference.holm(p)))
    for key, (a, b) in pooled.items():
        worst = 0 if a >= b else 1
        assert over.loc[key, "window"] == worst
        assert over.loc[key, "p_holm"] == pytest.approx(adjusted[(key, worst)])
    assert over.loc[("s2", "forge_simplex"), "seeds_above_delta"] == 3 and over.loc[("s2", "forge_simplex"), "p"] < 1e-9


def test_e1_tables_on_an_e13_run_follow_the_run(ugr16_signal, tmp_path):
    # pi_max at the main prevalence of the fixture, which the price is read at
    cfg = load_config(configs_dir() / "experiments" / "e13_ugr16_prevalence.yaml", OVERRIDES + ["certify.pi_max=0.2"])
    run, _ = run_e13(cfg, tmp_path / "runs")
    summary = runs.load_json(run.dir, "summary")
    rows = inference.e1_seed_rows({0: run.dir}, "e13_ugr16_prevalence")
    cut = seeds.e1_table(runs.load_table(run.dir, "by_prevalence"), summary, "e13_ugr16_prevalence")
    pd.testing.assert_frame_equal(rows[cut.columns], cut)
    assert (rows["dataset"] == "ugr16").all() and (rows["epsilon"] == summary["epsilon"]).all()
    price = inference.e1_paired_price(rows)
    assert (price["seeds"] == 1).all() and len(price) == len(cut[np.isclose(cut["prevalence"], 0.2)]) > 0
    signs = price[["price_seeds_above", "price_seeds_tied", "price_seeds_below"]].sum(axis=1)
    assert (signs == 1).all()
    # the largest window share of FORGE in the claim per arm and mix is the one the run's summary holds
    by_seed = inference.e1_violation_by_seed(inference.e1_claim_cells({0: run.dir}))
    assert set(by_seed["arm"]) == {"reference", "s2"} and set(by_seed["method"]) == {"forge_simplex", "forge_online"}
    for r in by_seed.itertuples():
        largest = summary["largest_violation_up_to_pi_max"][r.arm][r.method][r.mix]
        assert (r.window, r.prevalence) == (largest["window"], largest["prevalence"])
        assert (r.rate, r.low, r.high) == pytest.approx((largest["rate"], largest["low"], largest["high"]))
    # over one seed the intervals stand around the run's own shares
    intervals = inference.e1_intervals(rows).merge(rows, on=inference.E1_KEYS)
    assert ((intervals["violation_rate_ci_low"] <= intervals["violation_rate"] + 1e-12)
            & (intervals["violation_rate"] <= intervals["violation_rate_ci_high"] + 1e-12)).all()


def e24_claims(seed):
    # the claims of a stream run of UGR16, the weekly rule leaking less than one sequence in every seed, alarming in
    # half of the draws of seed 0, in none of seed 1 and in all of seed 2
    rows = []
    for policy in ("forge_simplex", "ltt_iid"):
        for arm in ("monitor", "monitor_week", "none"):
            week, monitor = arm == "monitor_week", arm == "monitor"
            alarm = {0: 0.5, 1: 0.0, 2: 1.0}[seed] if week else 1.0 if monitor else 0.0
            rows.append({"stream": "replay", "policy": policy, "composition": "typical", "prevalence": 0.02,
                         "arm": arm, "draws": 1000, "alarm_rate": alarm, "false_alarm_rate": 0.0,
                         "onset_true_rate": 1.0, "onset_realized_rate": 1.0,
                         "leak_mean": 100.0 + seed + (50.0 if monitor else 0.0),
                         "dismissed_share_mean": 0.5 if week else 0.52 if monitor else 0.6,
                         "over_bound_rate": {0: 2 / 500, 1: 0.0, 2: 0.001}[seed] if week else 0.0 if monitor
                         else np.nan,
                         "miscover_rate": 0.01 * seed if week else 0.0 if monitor else np.nan,
                         "sequences": 3 if week else 1 if monitor else np.nan,
                         "miscover_sequence_rate": 0.002 * seed if week else 0.0 if monitor else np.nan})
    return pd.DataFrame(rows)


def test_e3_paired_arms_and_intervals_count_draws_sequences_and_alarms(tmp_path):
    folders = {seed: fake_run(tmp_path, "e24_ugr16_streams", f"20261010-00000{seed}_a", {"seed": seed},
                              tables=[("claims", e24_claims(seed))]) for seed in (0, 1, 2)}
    rows = inference.e3_seed_rows(folders, "ugr16")
    assert len(rows) == 18 and (rows["dataset"] == "ugr16").all()
    paired = inference.e3_paired_arms(rows).set_index(["policy", "arm"])
    assert len(paired) == 4 and (paired["operating"] == "monitor_week").all()
    one = paired.loc[("forge_simplex", "monitor")]
    assert one["leak_diff_median"] == pytest.approx(50.0) and one["leak_diff_seeds_above"] == 3
    assert one["dismissed_diff_median"] == pytest.approx(0.02) and one["leak_diff_p"] == pytest.approx(0.25)
    none = paired.loc[("forge_simplex", "none")]
    assert none["leak_diff_seeds_tied"] == 3 and none["leak_diff_p"] == 1.0
    assert none["dismissed_diff_median"] == pytest.approx(0.1)
    p = paired[["leak_diff_p", "dismissed_diff_p"]].to_numpy().ravel()
    np.testing.assert_allclose(paired[["leak_diff_p_holm", "dismissed_diff_p_holm"]].to_numpy().ravel(),
                               inference.holm(p))
    with pytest.raises(KeyError):
        inference.e3_paired_arms(rows.assign(dataset="kyoto"))

    intervals = inference.e3_intervals(rows).set_index(["policy", "arm"])
    week = intervals.loc[("forge_simplex", "monitor_week")]
    assert week["alarm_rate_pooled"] == pytest.approx(0.5)
    assert week["alarm_rate_ci_low"] == pytest.approx(share_with_interval(1500, 3000)["low"])
    # three alarms above the bound among the 1,500 alarms of the seeds, the seed without alarms adding none
    assert week["over_bound_rate_pooled"] == pytest.approx(3 / 1500)
    assert week["over_bound_rate_ci_high"] == pytest.approx(share_with_interval(3, 1500)["high"])
    assert week["miscover_rate_pooled"] == pytest.approx(30 / 3000)
    assert week["miscover_sequence_rate_pooled"] == pytest.approx(18 / 9000)
    assert week["miscover_sequence_rate_ci_low"] == pytest.approx(share_with_interval(18, 9000)["low"])
    nothing = intervals.loc[("forge_simplex", "none")]
    assert np.isnan(nothing["miscover_rate_pooled"]) and np.isnan(nothing["over_bound_rate_ci_low"])

    frames = {seed: e24_claims(seed) for seed in (0, 1, 2)}
    e3 = seeds.e3_table(seeds.pool(frames, ["stream", "policy", "composition", "arm"], "draws"))
    e3.insert(0, "dataset", "ugr16")
    table = inference.with_intervals(e3, inference.e3_intervals(rows), inference.E3_KEYS)
    assert len(table) == len(e3) and table["arm"].tolist() == e3["arm"].tolist()
    at = table.columns.get_loc("over_bound_rate_pooled")
    assert table.columns[at + 1:at + 3].tolist() == ["over_bound_rate_ci_low", "over_bound_rate_ci_high"]
    by = table.set_index(["policy", "arm"])
    pooled_by_draws = e3.set_index(["policy", "arm"])
    # pooled by draws the share of alarms above the bound was the mean over seeds, and a share no seed has was zero
    assert pooled_by_draws.loc[("forge_simplex", "monitor_week"), "over_bound_rate_pooled"] == pytest.approx(0.005 / 3)
    assert by.loc[("forge_simplex", "monitor_week"), "over_bound_rate_pooled"] == pytest.approx(0.002)
    assert pooled_by_draws.loc[("forge_simplex", "none"), "miscover_rate_pooled"] == 0.0
    assert np.isnan(by.loc[("forge_simplex", "none"), "miscover_rate_pooled"])
    np.testing.assert_allclose(table["alarm_rate_pooled"], e3["alarm_rate_pooled"])
