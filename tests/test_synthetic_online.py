import numpy as np

from forge import experiments, runs
from forge.certify.certificate import for_from_rates
from forge.config import load_config
from forge.paths import configs_dir
from forge.runs import Run


def test_e14_tables_and_figures(tmp_path):
    cfg = load_config(configs_dir() / "experiments" / "e14_synthetic_online.yaml",
                      ["simulation.draws=200", "deployment.windows=12", "deployment.window_sizes=[100000, 1000000]"])
    # a third composition close to pi_max, where the window's own prevalence decides whether a threshold violates
    cfg["deployment"]["compositions"].append({"name": "near", "prevalence": 0.08, "shares": [1, 0, 0, 0]})
    run = Run(cfg, root=tmp_path)
    module = experiments.get("synthetic_online")
    module.run(cfg, run)
    run.finish()
    dep, sim = cfg["deployment"], cfg["simulation"]
    cells = len(dep["scenarios"]) * len(dep["compositions"]) * len(dep["window_sizes"])

    claims = runs.load_table(run.dir, "claims")
    assert len(claims) == cells * len(sim["methods"]) * dep["windows"]
    assert (claims["joint_violations"] >= claims["violations"]).all()
    worst = claims[claims["composition"] == "worst"]
    end = worst[(worst["scenario"] == "gradual") & (worst["window"] == 12)].set_index(["method", "window_size"])
    # once the benign scores have moved, the offline certificate fails in most draws and the online one does not
    assert (end.loc["forge_simplex", "rate"] > 0.5).all()
    assert (end.loc["forge_online", "rate"] < 0.15).all()
    jump = worst[(worst["scenario"] == "abrupt") & (worst["window"] == 11)].set_index(["method", "window_size"])
    assert (jump.loc["forge_previous", "rate"] > 0.5).all()
    # at the calibration prevalence, well below pi_max, the offline certificate keeps FOR below epsilon even after the
    # drift, so its failures come from windows near pi_max
    typical = claims[claims["composition"] == "typical"]
    assert (typical.loc[typical["method"] == "forge_simplex", "rate"] == 0).all()

    # the offline threshold does not depend on the window draws, so its violation shares follow from the saved truth
    draws = runs.load_arrays(run.dir, "draws")
    truth = runs.load_arrays(run.dir, "truth")
    offline, eps = draws["offline_index"], cfg["certify"]["epsilon"]
    offline_claims = claims[claims["method"] == "forge_simplex"]
    for i, scenario in enumerate(truth["scenarios"]):
        for co in dep["compositions"]:
            m_w = np.asarray(co["shares"], dtype=float) @ truth["families"]
            for w in range(1, dep["windows"] + 1):
                rate = for_from_rates(co["prevalence"], m_w, truth["benign_windows"][i, w])
                expected = np.mean(np.where(offline >= 0, rate[np.maximum(offline, 0)], 0.0) > eps)
                seen = offline_claims[(offline_claims["scenario"] == scenario)
                                      & (offline_claims["composition"] == co["name"]) & (offline_claims["window"] == w)]
                assert np.allclose(seen["rate"], expected)

    assert draws["online_index"].shape == (cells, dep["windows"] + 1, 200)
    summary = runs.load_json(run.dir, "summary")
    assert set(summary["claim_exceeds_delta"]) == set(sim["methods"])
    names = {p.name for p in run.figures.iterdir()}
    for name in module.FIGURES:
        assert {f"{name}.png", f"{name}.pdf", f"{name}_a4.pdf"} <= names
