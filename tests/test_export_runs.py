import importlib.util
import json
import subprocess
import sys
import zipfile

import numpy as np

from forge.paths import repo_root


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, repo_root() / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_run(root, experiment, run_id, big_mb=0.0):
    folder = root / experiment / run_id
    (folder / "results").mkdir(parents=True)
    (folder / "figures").mkdir()
    (folder / "meta.json").write_text(json.dumps({"run_id": run_id, "experiment": experiment, "status": "done"}),
                                      encoding="utf-8")
    (folder / "config.yaml").write_text("seed: 0\n", encoding="utf-8")
    (folder / "results" / "claims.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    if big_mb:
        np.save(folder / "results" / "scores.npy", np.zeros(int(big_mb * 1024 * 1024 / 8)))
    return folder


def test_export_leaves_out_large_files_and_import_takes_the_zip(tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    monkeypatch.setenv("FORGE_STORAGE", str(storage))
    root = storage / "runs"
    fake_run(root, "e99", "20261010-000000_a", big_mb=1.5)
    fake_run(root, "e99", "20261010-000001_a")
    fake_run(root, "e99", "20261010-000002_a", big_mb=1.5)
    export = load_script("export_runs")
    out = export.export("e99", tmp_path / "e99.zip", max_mb=1.0, first="20261010-000001_a", log=lambda s: None)
    names = zipfile.ZipFile(out).namelist()
    assert "e99/20261010-000000_a/meta.json" not in names
    assert "e99/20261010-000001_a/results/claims.csv" in names and "e99/20261010-000002_a/meta.json" in names
    assert "e99/20261010-000002_a/results/scores.npy" not in names
    assert "e99/20261010-000002_a/left_out.txt" in names and "e99/20261010-000001_a/left_out.txt" not in names
    assert zipfile.ZipFile(out).read("e99/20261010-000002_a/left_out.txt").decode().startswith("results/scores.npy ")
    # the zip imports into another storage root as a Drive download would, and a run already there is skipped
    other = tmp_path / "other"
    monkeypatch.setenv("FORGE_STORAGE", str(other))
    fake_run(other / "runs", "e99", "20261010-000001_a")
    result = subprocess.run([sys.executable, str(repo_root() / "scripts" / "import_runs.py"), str(out)],
                            capture_output=True, text=True, env={**dict(__import__("os").environ),
                                                                 "FORGE_STORAGE": str(other)})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "skipped e99/20261010-000001_a" in result.stdout and "imported e99/20261010-000002_a" in result.stdout
    assert (other / "runs" / "e99" / "20261010-000002_a" / "results" / "claims.csv").exists()
    assert (other / "runs" / "e99" / "20261010-000002_a" / "left_out.txt").exists()
    result = subprocess.run([sys.executable, str(repo_root() / "scripts" / "export_runs.py"), "e99", "--out",
                             str(tmp_path / "all.zip")], capture_output=True, text=True,
                            env={**dict(__import__("os").environ), "FORGE_STORAGE": str(storage)})
    assert result.returncode == 0 and "3 runs" in result.stdout
