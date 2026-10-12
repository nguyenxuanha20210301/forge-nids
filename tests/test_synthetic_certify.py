import numpy as np

from forge import experiments, runs
from forge.config import load_config
from forge.paths import configs_dir
from forge.runs import Run


def test_e10_tables_and_figures(tmp_path):
    cfg = load_config(configs_dir() / "experiments" / "e10_synthetic_certify.yaml",
                      ["simulation.draws=200", "simulation.sizes=[300000, 30000000]",
                       "simulation.main_size=30000000"])
    run = Run(cfg, root=tmp_path)
    module = experiments.get("synthetic_certify")
    module.run(cfg, run)
    run.finish()
    sim = cfg["simulation"]

    claims = runs.load_table(run.dir, "claims")
    assert claims["method"].tolist() == sim["methods"] * 2
    assert (claims["joint_violations"] >= claims["violations"]).all()
    assert claims.loc[claims["method"] == "ltt_iid", "floor_violations"].isna().all()
    big = claims[claims["size"] == 30000000].set_index("method")
    # at 3e7 flows the any mix certificate works, while on the grid of 200 benign quantiles the version starting at
    # the strictest threshold still refuses
    assert big.loc["forge_simplex", "refused"] < 0.5 and big.loc["forge_strict", "refused"] == 1.0
    assert big.loc["ltt_iid", "c1_rate"] > 0.5

    violations = runs.load_table(run.dir, "violations")
    assert len(violations) == 2 * len(sim["methods"]) * len(sim["prevalences"]) * (2 + len(cfg["synthetic"]["families"]))
    draws = runs.load_arrays(run.dir, "draws")
    assert draws["index"].shape == (2, len(sim["methods"]), 200)
    assert (draws["counts"].sum(-1) == np.array([300000, 30000000])[:, None]).all()

    summary = runs.load_json(run.dir, "summary")
    assert set(summary["claim_exceeds_delta"]) == set(sim["methods"])
    names = {p.name for p in run.figures.iterdir()}
    for name in module.FIGURES:
        assert {f"{name}.png", f"{name}.pdf", f"{name}_a4.pdf"} <= names
