"""runs one experiment config over a range of seeds, each seed a run of its own with --set seed=<n>.

    python scripts/run_seeds.py configs/experiments/e11_anoshift_prevalence.yaml --seeds 1-19
    python scripts/run_seeds.py configs/experiments/e23_unseen_family.yaml --seeds 1-4 --set audits.rate=0.005
    python scripts/run_seeds.py configs/experiments/e21_monitor_streams.yaml --seeds 0-19 --source-per-seed \
        --source-from 20261013-223628_8502bf0

every seed runs scripts/run.py in a child process, so a failed seed stops nothing else. a seed that already has a
finished run with the same resolved config in the runs folder is skipped unless --rerun is given, so the command can be
started again after a Colab session ended. with --source-per-seed the source.run of the config is set, for every seed,
to the latest finished run of the source experiment with that seed, at or after the run id of --source-from, so an
experiment that reads the thresholds of another one runs on the scorer of the same seed. the exit code is 1 when any
seed failed.
"""

import argparse
import sys

from forge.batch import parse_seeds, run_batch, source_runs
from forge.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--seeds", nargs="+", required=True, metavar="N or A-B")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides applied to every seed")
    parser.add_argument("--rerun", action="store_true", help="run a seed even when a finished run has its config")
    parser.add_argument("--source-per-seed", action="store_true",
                        help="set source.run to the run of the source experiment with the same seed")
    parser.add_argument("--source-from", default=None, metavar="RUN_ID",
                        help="the earliest run of the source experiment to take, with --source-per-seed")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    sets = {n: [f"seed={n}", *args.set] for n in seeds}
    if args.source_per_seed or args.source_from:
        cfg = load_config(args.config)
        if "source" not in cfg or "experiment" not in cfg["source"]:
            raise SystemExit("the config names no source experiment, so there is no source run to take per seed")
        for n, run_id in source_runs(cfg["source"]["experiment"], seeds, args.source_from).items():
            sets[n].insert(1, f"source.run={run_id}")
    items = [(f"seed {n}", sets[n]) for n in seeds]
    rows = run_batch(args.config, items, rerun=args.rerun)
    if any(r["status"] == "failed" for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
