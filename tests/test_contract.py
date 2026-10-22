import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from forge import runs
from forge.certify.certificate import certify, for_from_rates, group_rows, start_settings
from forge.config import load_config
from forge.data.synthetic import draw_counts
from forge.eval import contract
from forge.paths import configs_dir
from forge.viz import paper as figures
from forge.viz.export import save_figure

from test_for_vs_prevalence import OVERRIDES as E11_OVERRIDES
from test_for_vs_prevalence import run_e11
from test_ugr16_prevalence import OVERRIDES as E13_OVERRIDES
from test_ugr16_prevalence import run_e13

AUDIT_RATE = 0.01


def kappa(eps, pi_max):
    return eps * (1 - pi_max) / ((1 - eps) * pi_max)


def check_rows(cfg, run, epsilons, pi_maxes):
    # every pair certified again on the run's own calibration sets with certify, the run's groups and its start, and
    # the run's own pair giving the run's own thresholds
    rows = contract.run_rows(run.dir, epsilons, pi_maxes, AUDIT_RATE)
    summary = runs.load_json(run.dir, "summary")
    hist = runs.load_arrays(run.dir, "histograms")
    saved = runs.load_arrays(run.dir, "draws")
    pop = contract.calibration_population(hist)
    b, m_family, shares = pop["b"], pop["m_family"], pop["shares"]
    pi_cal, delta, part_b = summary["calibration_prevalence"], summary["delta"], summary["delta_benign"]
    rng = np.random.default_rng(np.random.SeedSequence(cfg["seed"]).spawn(1)[0])
    cal = draw_counts(rng, summary["calibration_sessions"], pi_cal, shares, b, m_family, summary["draws"])
    assert np.array_equal(saved["counts"], np.column_stack([cal.n_benign, cal.n_family]))
    groups = hist["groups"]
    grouped = group_rows(cal.family_below, cal.n_family, groups)
    start = start_settings(cfg["certify"])
    dismissed = pi_cal * (shares @ m_family) + (1 - pi_cal) * b
    m_groups = np.stack([shares[groups == g] @ m_family[groups == g] / shares[groups == g].sum()
                         for g in range(groups.max() + 1)])
    assert len(rows) == len(epsilons) * len(pi_maxes) and rows["operating"].sum() == 1
    for r in rows.itertuples():
        index = certify(cal.benign_below, cal.n_benign, *grouped, r.epsilon, delta, r.pi_max, part_b, **start)["index"]
        hit, at = index >= 0, np.maximum(index, 0)
        assert r.refused == np.mean(~hit) and r.kappa == pytest.approx(kappa(r.epsilon, r.pi_max))
        assert r.benign_dismissed_mean == pytest.approx(np.where(hit, b[at], 0.0).mean())
        assert r.dismissed_mean == pytest.approx(np.where(hit, dismissed[at], 0.0).mean())
        assert r.workload_per_million == pytest.approx((1 - (1 - AUDIT_RATE) * r.dismissed_mean) * 1e6)
        worst = for_from_rates(r.pi_max, m_groups.max(0), b)
        assert r.violation_rate == np.mean(hit & (worst[at] > r.epsilon))
        safe = np.flatnonzero(for_from_rates(r.pi_max, m_groups, b).max(0) <= r.epsilon)
        assert r.best_every_mix_dismissed == pytest.approx(dismissed[safe[-1]] if len(safe) else 0.0)
        assert r.price_every_mix == pytest.approx(r.best_every_mix_dismissed - r.dismissed_mean)
        if r.operating:
            assert np.array_equal(index, saved["forge_simplex_index"])
            assert r.refused == summary["offline_refused"]["forge_simplex"]
    return rows


def test_contracts_on_ugr16_follow_certify_and_read_the_terms_through_kappa(ugr16_signal, tmp_path):
    cfg = load_config(configs_dir() / "experiments" / "e13_ugr16_prevalence.yaml", E13_OVERRIDES)
    run, _ = run_e13(cfg, tmp_path / "runs")
    # the run certifies at epsilon 0.05 and pi_max 0.3, and the pair at pi_max 0.1 with the same kappa as that one
    same = kappa(0.05, 0.3) * 0.1 / (0.9 + kappa(0.05, 0.3) * 0.1)
    epsilons, pi_maxes = (0.002, same, 0.05), (0.1, 0.3)
    rows = check_rows(cfg, run, epsilons, pi_maxes)
    assert rows.loc[rows["operating"], ["epsilon", "pi_max"]].values.tolist() == [[0.05, 0.3]]
    assert (rows["start"] == "share").all()
    # pairs with the same kappa certify the same thresholds and share the yardsticks
    one = rows[(rows["epsilon"] == 0.05) & (rows["pi_max"] == 0.3)].iloc[0]
    other = rows[np.isclose(rows["epsilon"], same) & (rows["pi_max"] == 0.1)].iloc[0]
    for col in ("refused", "dismissed_mean", "best_up_to_pi_max_dismissed", "best_every_mix_dismissed"):
        assert one[col] == pytest.approx(other[col])
    # a stricter contract dismisses no more and refuses no less
    strict = rows[(rows["epsilon"] == 0.002) & (rows["pi_max"] == 0.3)].iloc[0]
    assert strict["dismissed_mean"] <= one["dismissed_mean"] and strict["refused"] >= one["refused"]

    # pooled over one seed every number keeps its value in the median, range and pooled forms, and the figure draws
    # from the written table
    table = contract.contract_table({"ugr16": {0: run.dir}}, epsilons, pi_maxes, AUDIT_RATE)
    assert len(table) == len(rows) and (table["seeds"] == 1).all()
    for col in ("refused", "violation_rate"):
        assert np.allclose(table[f"{col}_pooled"], rows[col])
    for col in ("dismissed_mean", "workload_per_million", "price_every_mix"):
        assert np.allclose(table[f"{col}_median"], rows[col]) and np.allclose(table[f"{col}_low"], rows[col])
    path = tmp_path / "e7_contract.csv"
    table.to_csv(path, index=False)
    fig = figures.e7_dismissed_by_contract(pd.read_csv(path))
    assert [ax.get_xlabel() for ax in fig.axes[:1]] == ["$\\pi_{\\max}$, UGR'16"]
    save_figure(fig, tmp_path / "figures", "e7_dismissed_by_contract")
    plt.close("all")
    with pytest.raises(ValueError, match="no run folder"):
        contract.contract_table({"ugr16": {}})


def test_contracts_on_anoshift_take_the_mix_of_the_calibration_year(anoshift_signal, tmp_path):
    cfg = load_config(configs_dir() / "experiments" / "e11_anoshift_prevalence.yaml", E11_OVERRIDES)
    run, _ = run_e11(cfg, tmp_path / "runs")
    hist = runs.load_arrays(run.dir, "histograms")
    # e11 keeps no mix of its calibration sets, which is the natural mix of the calibration year
    assert "shares" not in hist
    sizes = hist["calibration_sizes"]
    assert np.allclose(contract.calibration_population(hist)["shares"], sizes[1:] / sizes[1:].sum())
    rows = check_rows(cfg, run, (0.01, 0.05), (0.1, 0.3))
    assert (rows["start"] == "feasible").all()
