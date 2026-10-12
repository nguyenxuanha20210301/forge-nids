import numpy as np

from forge import experiments, runs
from forge.config import load_config
from forge.data import anoshift
from forge.paths import configs_dir
from forge.runs import Run

from conftest import ANOSHIFT_YEARS


def test_e03_tables_and_figures(anoshift_source, tmp_path):
    cfg = load_config(configs_dir() / "experiments" / "e03_anoshift.yaml", ["dataset.blocks_per_year=4"])
    quiet = lambda m: None  # noqa: E731
    anoshift.download(cfg, log=quiet, expected=anoshift_source.expected, url=anoshift_source.url,
                      tar_bytes=anoshift_source.tar_bytes, tar_sha256=anoshift_source.tar_sha256)
    anoshift.prepare(cfg, log=quiet, expected=anoshift_source.expected, tar_sha256=anoshift_source.tar_sha256)
    run = Run(cfg, root=tmp_path / "runs")
    module = experiments.get("anoshift_overview")
    module.run(cfg, run)
    run.finish()

    years = runs.load_table(run.dir, "years")
    assert years["year"].tolist() == list(ANOSHIFT_YEARS)
    assert years["period"].tolist() == ["iid", "near", "far"]
    for y in years.itertuples():
        known, unknown, normal = ANOSHIFT_YEARS[y.year]
        assert (y.n_known, y.n_unknown, y.n_normal) == (known, unknown, normal)
        assert y.attack_share == (known + unknown) / (known + unknown + normal)

    blocks = runs.load_table(run.dir, "blocks")
    assert len(blocks) == 4 * len(ANOSHIFT_YEARS)
    sizes = blocks.groupby("year")["n_flows"]
    assert ((sizes.max() - sizes.min()) <= 1).all()
    assert sizes.sum().tolist() == years["n_flows"].tolist()
    assert blocks.groupby("year")["n_attacks"].sum().tolist() == years["n_attacks"].tolist()
    assert blocks.groupby("year")["n_unknown"].sum().tolist() == years["n_unknown"].tolist()

    services = runs.load_table(run.dir, "services")
    assert np.allclose(services.groupby(["year", "class"])["share"].sum(), 1)

    summary = runs.load_json(run.dir, "summary")
    assert summary["fingerprint"] == anoshift.read_manifest(cfg)["fingerprint"]
    assert summary["sessions"] == int(years["n_flows"].sum())
    names = {p.name for p in run.figures.iterdir()}
    for figure in module.FIGURES:
        assert {f"{figure}.png", f"{figure}.pdf", f"{figure}_a4.pdf"} <= names
