from forge import experiments
from forge.config import load_config
from forge.paths import configs_dir
from forge.runs import Run


def test_smoke_run(tmp_path):
    cfg = load_config(configs_dir() / "experiments" / "e00_smoke.yaml", ["smoke.n_per_window=5000", "smoke.grid_size=21"])
    run = Run(cfg, root=tmp_path)
    experiments.get("smoke").run(cfg, run)
    run.finish()
    names = {p.name for p in run.figures.iterdir()}
    assert {"for_by_window.png", "for_by_window.pdf", "for_by_window_a4.pdf"} <= names
    assert (run.results / "curves.parquet").exists()
