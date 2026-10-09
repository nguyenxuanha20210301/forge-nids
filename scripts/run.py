"""runs one experiment from a config file.

    python scripts/run.py configs/experiments/e00_smoke.yaml
    python scripts/run.py configs/experiments/e00_smoke.yaml --set seed=1 smoke.epsilon=0.005
"""

import argparse

from forge import experiments
from forge.config import load_config
from forge.runs import Run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    module = experiments.get(cfg["experiment"]["module"])
    run = Run(cfg)
    run.log(f"experiment {cfg['experiment']['name']}, run {run.run_id}")
    try:
        module.run(cfg, run)
    except BaseException:
        run.finish("failed")
        raise
    run.finish()


if __name__ == "__main__":
    main()
