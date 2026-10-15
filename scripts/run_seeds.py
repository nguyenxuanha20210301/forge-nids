"""runs one experiment config over a range of seeds, each seed a run of its own with --set seed=<n>.

    python scripts/run_seeds.py configs/experiments/e11_anoshift_prevalence.yaml --seeds 1-19
    python scripts/run_seeds.py configs/experiments/e23_unseen_family.yaml --seeds 1-4 --set audits.rate=0.005

every seed runs scripts/run.py in a child process, so a failed seed stops nothing else. a seed that already has a
finished run with the same resolved config in the runs folder is skipped unless --rerun is given, so the command can be
started again after a Colab session ended. the exit code is 1 when any seed failed.
"""

import argparse
import sys

from forge.batch import parse_seeds, run_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--seeds", nargs="+", required=True, metavar="N or A-B")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides applied to every seed")
    parser.add_argument("--rerun", action="store_true", help="run a seed even when a finished run has its config")
    args = parser.parse_args()

    items = [(f"seed {n}", [f"seed={n}", *args.set]) for n in parse_seeds(args.seeds)]
    rows = run_batch(args.config, items, rerun=args.rerun)
    if any(r["status"] == "failed" for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
