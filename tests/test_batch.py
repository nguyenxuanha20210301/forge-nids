import subprocess
import sys

import pytest

from forge import runs
from forge.batch import finished_run, override, overrides, parse_seeds, run_batch
from forge.config import load_config
from forge.paths import configs_dir, repo_root

SMOKE = configs_dir() / "experiments" / "e00_smoke.yaml"
SMALL = ["smoke.n_per_window=2000", "smoke.grid_size=11"]


def test_seed_ranges_and_overrides_are_parsed():
    assert parse_seeds(["1-3", "7", "2"]) == [1, 2, 3, 7]
    assert parse_seeds(["5"]) == [5]
    with pytest.raises(ValueError, match="ends before"):
        parse_seeds(["4-2"])
    assert override("certify.grid_size", 200) == "certify.grid_size=200"
    assert override("calibration.mix", "natural") == "calibration.mix=natural"
    assert override("split.shares", [0.45, 0.1, 0.2, 0.25]) == "split.shares=[0.45, 0.1, 0.2, 0.25]"
    assert override("deployment.subsample_benign", False) == "deployment.subsample_benign=false"
    assert override("simulation.trace", {"unit": "land", "policy": "forge_simplex"}) == \
        "simulation.trace={unit: land, policy: forge_simplex}"
    # every item parses back to the value it came from through the override rule of load_config
    sets = {"a.b": [1, "x"], "a.c": 0.005, "d": "text", "e": {"k": True}}
    cfg = load_config(SMOKE, overrides(sets))
    assert cfg["a"] == {"b": [1, "x"], "c": 0.005} and cfg["d"] == "text" and cfg["e"] == {"k": True}


def test_run_batch_runs_each_seed_once_and_goes_on_after_a_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_STORAGE", str(tmp_path / "storage"))
    lines = []
    rows = run_batch(SMOKE, [(f"seed {n}", [f"seed={n}", *SMALL]) for n in (0, 1)], log=lines.append)
    assert [r["status"] for r in rows] == ["done", "done"]
    folders = runs.run_dirs("e00_smoke", tmp_path / "storage" / "runs")
    assert [f.name for f in folders] == [r["run_id"] for r in rows]
    assert [runs.load_run_config(f)["seed"] for f in folders] == [0, 1]
    assert all(runs.load_meta(f)["status"] == "done" for f in folders)
    assert any(line.startswith("batch of 2 runs done, 0 failed, 0 skipped") for line in lines)
    # the same seeds again are skipped, a new seed runs, and a failing config does not stop the batch
    rows = run_batch(SMOKE, [("seed 1", ["seed=1", *SMALL]), ("seed 2", ["seed=2", *SMALL]),
                             ("seed 3", ["seed=3", *SMALL, "experiment.module=missing"]),
                             ("seed 5", ["seed=5", *SMALL, "smoke.prevalences=[]"])], log=lines.append)
    assert [r["status"] for r in rows] == ["skipped", "done", "failed", "failed"]
    assert rows[0]["run_id"] == folders[1].name
    assert finished_run(load_config(SMOKE, ["seed=2", *SMALL])) is not None
    assert finished_run(load_config(SMOKE, ["seed=2", "smoke.n_per_window=3000", "smoke.grid_size=11"])) is None
    # a config that fails before the run folder exists has no run id, a run that fails inside keeps its folder with
    # its status and is not reused, and a rerun makes a new folder for a finished seed
    assert rows[2]["run_id"] is None and rows[3]["run_id"] is not None
    failed = [f for f in runs.run_dirs("e00_smoke", tmp_path / "storage" / "runs") if f.name == rows[3]["run_id"]]
    assert len(failed) == 1 and runs.load_meta(failed[0])["status"] == "failed"
    assert finished_run(load_config(SMOKE, ["seed=5", *SMALL, "smoke.prevalences=[]"])) is None
    assert any(line.startswith("batch of 4 runs done, 2 failed, 1 skipped") for line in lines)
    rows = run_batch(SMOKE, [("seed 0", ["seed=0", *SMALL])], rerun=True, log=lines.append)
    assert rows[0]["status"] == "done" and rows[0]["run_id"] != folders[0].name


def test_run_seeds_and_run_sweep_scripts(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_STORAGE", str(tmp_path / "storage"))
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(
        f"inherit: [{SMOKE}]\n"
        "variant: base\n"
        "sweep:\n"
        "  - {variant: base, set: {}}\n"
        "  - {variant: grid_31, set: {smoke.grid_size: 31}}\n", encoding="utf-8")
    out = subprocess.run([sys.executable, str(repo_root() / "scripts" / "run_seeds.py"), str(SMOKE), "--seeds", "4",
                          "--set", *SMALL], capture_output=True, text=True, cwd=tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "seed 4, run " in out.stdout and "done" in out.stdout
    out = subprocess.run([sys.executable, str(repo_root() / "scripts" / "run_sweep.py"), str(sweep), "--only",
                          "grid_31", "--set", "smoke.n_per_window=2000"], capture_output=True, text=True, cwd=tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    folders = runs.run_dirs("e00_smoke", tmp_path / "storage" / "runs")
    variants = [runs.load_run_config(f).get("variant") for f in folders]
    assert variants.count("grid_31") == 1
    grid = [f for f, v in zip(folders, variants) if v == "grid_31"][0]
    cfg = runs.load_run_config(grid)
    assert cfg["smoke"]["grid_size"] == 31 and cfg["smoke"]["n_per_window"] == 2000 and "sweep" in cfg
    out = subprocess.run([sys.executable, str(repo_root() / "scripts" / "run_sweep.py"), str(sweep), "--only",
                          "nothing"], capture_output=True, text=True, cwd=tmp_path)
    assert out.returncode != 0 and "no variant named nothing" in out.stderr
