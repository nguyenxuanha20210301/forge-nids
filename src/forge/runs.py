"""one folder per run, written once and never edited afterwards.

layout of a run folder.
    config.yaml   the fully resolved config
    meta.json     git commit, start and end time, platform, package versions
    log.txt       everything the run printed
    results/      every number a table or figure needs
    figures/      png at 600 dpi, pdf, and a one figure per page A4 pdf
"""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from forge import paths
from forge.config import save_config

PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "pyarrow", "pypdf", "torch", "xgboost", "lightgbm")


def _git(*args) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=paths.repo_root(), capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _versions() -> dict:
    found = {}
    for name in PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return found


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Run:
    def __init__(self, cfg: dict, root: Path | None = None):
        self.cfg = cfg
        self.started = _now()
        commit = _git("rev-parse", "--short", "HEAD") or "nogit"
        name = cfg["experiment"]["name"]
        self.run_id = f"{self.started:%Y%m%d-%H%M%S}_{commit}"
        base = Path(root) if root else paths.runs_dir()
        self.dir = base / name / self.run_id
        self.results = self.dir / "results"
        self.figures = self.dir / "figures"
        self.results.mkdir(parents=True, exist_ok=False)
        self.figures.mkdir()
        self._log = open(self.dir / "log.txt", "a", encoding="utf-8")
        save_config(cfg, self.dir / "config.yaml")
        self.meta = {
            "run_id": self.run_id,
            "experiment": name,
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "started_utc": self.started.isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "platform": "colab" if paths.on_colab() else platform.system().lower(),
            "packages": _versions(),
            "status": "running",
        }
        self._write_meta()

    def _write_meta(self) -> None:
        with open(self.dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)

    def log(self, message: str) -> None:
        line = f"{_now():%H:%M:%S} {message}"
        print(line, flush=True)
        self._log.write(line + "\n")
        self._log.flush()

    def save_table(self, df: pd.DataFrame, name: str) -> Path:
        path = self.results / f"{name}.parquet"
        df.to_parquet(path, index=False)
        df.to_csv(self.results / f"{name}.csv", index=False)
        return path

    def save_arrays(self, name: str, **arrays) -> Path:
        path = self.results / f"{name}.npz"
        np.savez_compressed(path, **arrays)
        return path

    def save_json(self, obj: dict, name: str) -> Path:
        path = self.results / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        return path

    def finish(self, status: str = "done") -> None:
        ended = _now()
        self.meta["status"] = status
        self.meta["ended_utc"] = ended.isoformat(timespec="seconds")
        self.meta["seconds"] = round((ended - self.started).total_seconds(), 1)
        self._write_meta()
        self.log(f"run {status} in {self.meta['seconds']} s, saved to {self.dir}")
        self._log.close()


def load_table(run_dir, name: str) -> pd.DataFrame:
    return pd.read_parquet(Path(run_dir) / "results" / f"{name}.parquet")


def load_arrays(run_dir, name: str) -> dict:
    with np.load(Path(run_dir) / "results" / f"{name}.npz") as z:
        return {k: z[k] for k in z.files}


def load_json(run_dir, name: str) -> dict:
    with open(Path(run_dir) / "results" / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)
