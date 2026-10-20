import numpy as np
import pytest

from forge import runs
from forge.certify.bounds import clopper_pearson_upper, dkw_margin
from forge.certify.certificate import certify, for_from_rates, group_rows, ltt_iid
from forge.config import load_config
from forge.data.synthetic import draw_counts
from forge.eval import starts
from forge.eval.windows import rates
from forge.paths import configs_dir

from test_ugr16_prevalence import OVERRIDES, run_e13


def test_rules_needed_flows_and_allowed_misses():
    summary = {"start": "share", "start_misses": 2, "start_slack": 0.05, "start_share": 0.5}
    rules = starts.rules(summary, misses=(5, 50), shares=(0.3, 0.5))
    assert [r["rule"] for r in rules] == ["feasible", "misses 5", "misses 50", "share 0.3", "share 0.5"]
    assert rules[0] == {"rule": "feasible", "start": "feasible", "start_misses": 2, "start_slack": 0.05}
    assert rules[1]["start_slack"] == starts.LIFTED and rules[4]["start_share"] == 0.5
    assert [starts.operating(r, summary) for r in rules] == [False, False, False, False, True]
    assert starts.operating(rules[0], {"start_misses": 2, "start_slack": 0.05})
    # the fewest dismissed flows with which r attacks among them pass the bound of LTT, one flow fewer fails
    for r in (0, 1, 7, 200):
        n = starts.needed_flows(r, 0.005, 0.05)
        assert clopper_pearson_upper(r, n, 0.05) <= 0.005 < clopper_pearson_upper(r, n - 1, 0.05)
    # the most misses a family of n attacks has whose bound stays at or below a rate, fractional counts included
    for n, rate in ((10740, 0.00136), (5810.4, 0.0009), (54809.0, 0.03)):
        a = starts.allowed_misses(n, rate, 0.025)
        assert clopper_pearson_upper(a, n, 0.025) <= rate < clopper_pearson_upper(a + 1, n, 0.025)
    assert starts.allowed_misses(100, 0.001, 0.025) == -1 and starts.allowed_misses(100, 0.0, 0.025) == -1


def test_start_rules_follow_certify_and_ltt_on_the_run_draws(ugr16_signal, tmp_path):
    cfg = load_config(configs_dir() / "experiments" / "e13_ugr16_prevalence.yaml", OVERRIDES)
    run, _ = run_e13(cfg, tmp_path / "runs")
    misses, shares = (5, 20, 200), (0.2, 0.5, 0.999)
    table = starts.start_rules_table({0: run.dir}, misses, shares)
    summary = runs.load_json(run.dir, "summary")
    hist = runs.load_arrays(run.dir, "histograms")
    saved = runs.load_arrays(run.dir, "draws")
    rules = starts.rules(summary, misses, shares)
    assert table["rule"].tolist() == [r["rule"] for r in rules] and (table["seed"] == 0).all()
    assert table["operating"].tolist() == [r["rule"] == "share 0.5" for r in rules]
    names = ["dos+scan11", "scan44", "nerisbotnet", "anomaly_spam"]
    assert {f"misses_{n}" for n in names} | {f"allowed_{n}" for n in names} <= set(table.columns)

    # the redrawn sets are the run's own calibration sets, drawn again from the first child of its seed
    eps, delta, pi_max, part_b = summary["epsilon"], summary["delta"], summary["pi_max"], summary["delta_benign"]
    b, m, _, _ = rates({"benign_below": hist["calibration_benign_below"],
                        "family_below": hist["calibration_family_below"], "sizes": hist["calibration_sizes"]})
    rng = np.random.default_rng(np.random.SeedSequence(cfg["seed"]).spawn(1)[0])
    cal = draw_counts(rng, summary["calibration_sessions"], summary["calibration_prevalence"], hist["shares"], b, m,
                      summary["draws"])
    assert np.array_equal(saved["counts"], np.column_stack([cal.n_benign, cal.n_family]))
    grouped = group_rows(cal.family_below, cal.n_family, hist["groups"])
    pooled = cal.family_below.sum(1)
    for rule, (_, row) in zip(rules, table.iterrows()):
        settings = {k: v for k, v in rule.items() if k != "rule"}
        forge = certify(cal.benign_below, cal.n_benign, *grouped, eps, delta, pi_max, part_b, **settings)["index"]
        ltt = ltt_iid(cal.benign_below, pooled, eps, delta, n=cal.n_benign + cal.n_family.sum(1), **settings)["index"]
        for prefix, index in (("forge", forge), ("ltt", ltt)):
            hit = index >= 0
            assert row[f"{prefix}_refused"] == pytest.approx(np.mean(~hit))
            if hit.any():
                assert row[f"{prefix}_dismissed_mean"] == pytest.approx(b[index[hit]].mean())
                assert row[f"{prefix}_dismissed_median"] == pytest.approx(np.median(b[index[hit]]))
            else:
                assert np.isnan(row[f"{prefix}_dismissed_median"])
        if row["operating"]:
            # the rule the run took gives the run's own thresholds
            assert np.array_equal(forge, saved["forge_simplex_index"]) and np.array_equal(ltt, saved["ltt_iid_index"])
            assert row["run_forge_refused"] == summary["offline_refused"]["forge_simplex"]
            assert row["forge_dismissed_mean"] == pytest.approx(row["run_forge_dismissed_mean"])
        else:
            assert np.isnan(row["run_forge_refused"])

    # the expected calibration set, every benign flow of the part and every group at its expected count
    expected = summary["calibration_sessions"] * summary["calibration_prevalence"] * hist["shares"]
    g_below, g_n = group_rows((m * expected[:, None])[None], expected[None], hist["groups"])
    n_benign = hist["calibration_sizes"][0]
    b_lower = np.clip((b * n_benign) / n_benign - dkw_margin(n_benign, part_b), 0, None)
    upper = np.stack([clopper_pearson_upper(g_below[0, k], g_n[0, k], delta - part_b) for k in range(len(names))])
    passing = np.flatnonzero(for_from_rates(pi_max, upper.max(0), b_lower) <= eps)
    assert (table["passing"] == len(passing)).all()
    assert (table["loosest_passing_benign"] == b[passing[-1]]).all()
    kappa = eps * (1 - pi_max) / ((1 - eps) * pi_max)
    for rule, (_, row) in zip(rules, table.iterrows()):
        start = row["start_index"]
        settings = {k: v for k, v in rule.items() if k != "rule"}
        index = certify((b * n_benign)[None], np.array([n_benign]), g_below, g_n, eps, delta, pi_max, part_b,
                        **settings)["index"][0]
        if index >= 0:
            assert row["expected_certified_benign"] == b[index]
        else:
            assert np.isnan(row["expected_certified_benign"])
        if start < 0:
            # a share no lower benign bound reaches leaves no start, and nothing is certified from there
            assert rule["start"] == "share" and b_lower[-1] < rule["start_share"] and index == -1
            continue
        assert row["start_benign"] == b[start] and row["binding"] == names[int(np.argmax(upper[:, start]))]
        rate = kappa * b_lower[start]
        for k, name in enumerate(names):
            allowed = row[f"allowed_{name}"]
            assert row[f"misses_{name}"] == pytest.approx(g_below[0, k, start])
            assert allowed == -1 or clopper_pearson_upper(allowed, g_n[0, k], delta - part_b) <= rate
            assert clopper_pearson_upper(allowed + 1, g_n[0, k], delta - part_b) > rate
        if rule["start"] == "share":
            assert b_lower[start] >= rule["start_share"] > b_lower[start - 1]
        elif rule["start_slack"] == starts.LIFTED and start > 0:
            # a start moved by r misses with the slack lifted is the first threshold where every group allows r,
            # when one does
            r = rule["start_misses"]
            if min(row[f"allowed_{n}"] for n in names) >= r:
                before = min(starts.allowed_misses(g_n[0, k], kappa * b_lower[start - 1], delta - part_b)
                             for k in range(len(names)))
                assert before < r
    with pytest.raises(ValueError, match="no e13 run folder"):
        starts.start_rules_table({})
