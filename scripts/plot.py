"""redraws the figures of saved runs from their results, without rerunning anything.

    python scripts/plot.py drive/runs/e00_smoke/20261010-091500_ab12cd3
    python scripts/plot.py drive/runs/e00_smoke/20261010-091500_ab12cd3 --only for_by_window

run folders are never edited, so redrawn figures go to drive/redrawn/<experiment>/<run_id> unless --out names another folder.
"""

import argparse
from pathlib import Path

from forge import experiments, paths
from forge.config import load_config
from forge.viz.export import save_figure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=paths.storage_root() / "redrawn")
    args = parser.parse_args()

    for run_dir in args.runs:
        cfg = load_config(run_dir / "config.yaml")
        module = experiments.get(cfg["experiment"]["module"])
        out = args.out / run_dir.parent.name / run_dir.name
        for name, draw in module.FIGURES.items():
            if args.only and name not in args.only:
                continue
            save_figure(draw(run_dir), out, name)
            print(f"saved {name} to {out}")


if __name__ == "__main__":
    main()
