"""runs the variants of a sweep config, each variant a run of its own with the overrides the config lists for it.

    python scripts/run_sweep.py configs/experiments/e19_ablations.yaml
    python scripts/run_sweep.py configs/experiments/e19_ablations.yaml --only base grid_200

the config holds a list sweep, one entry per variant with its name and the settings it moves, for example
    sweep:
      - {variant: base, set: {}}
      - {variant: grid_200, set: {certify.grid_size: 200}}
and every run gets --set variant=<name> and those settings, so the saved config of a run names its variant. a variant
that already has a finished run with the same resolved config is skipped unless --rerun is given. the exit code is 1
when any variant failed.
"""

import argparse
import sys

from forge.batch import overrides, run_batch
from forge.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--only", nargs="*", default=None, metavar="VARIANT")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides applied to every variant")
    parser.add_argument("--rerun", action="store_true", help="run a variant even when a finished run has its config")
    args = parser.parse_args()

    sweep = load_config(args.config).get("sweep", [])
    names = [v["variant"] for v in sweep]
    if len(set(names)) != len(names) or not names:
        raise SystemExit("the config needs a list sweep with a distinct variant name in every entry")
    unknown = set(args.only or []) - set(names)
    if unknown:
        raise SystemExit(f"no variant named {', '.join(sorted(unknown))}, the config has {', '.join(names)}")
    items = [(f"variant {v['variant']}", [f"variant={v['variant']}", *overrides(v.get("set") or {}), *args.set])
             for v in sweep if not args.only or v["variant"] in args.only]
    rows = run_batch(args.config, items, rerun=args.rerun)
    if any(r["status"] == "failed" for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
