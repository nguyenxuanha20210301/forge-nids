"""the ablations of E9 side by side, one row per variant of e19 from its latest finished run, and the Blaster worm held
out under the feature sets of e19.

    python analysis/e19_ablations.py
    python analysis/e19_ablations.py --out drive/pooled/e19_ablations

the tables go to pooled/e19_ablations under the storage root unless --out names another folder, variants.csv with
every variant and blaster_unseen.csv with the Blaster rows, and the lines the plan quotes are printed.
"""

import argparse
from pathlib import Path

import pandas as pd

from forge import paths
from forge.eval import variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--base-e23", default=None, metavar="RUN_ID", help="the e23 run that stands for the base")
    args = parser.parse_args()
    out = args.out or paths.storage_root() / "pooled" / variants.E19
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    table = variants.e19_table()
    table.to_csv(out / "variants.csv", index=False)
    print(f"{len(table)} variants of {variants.E19}, written to {out / 'variants.csv'}")
    if len(table):
        shown = ["variant", "grid_thresholds", "families", "forge_benign_dismissed", "forge_miss_land",
                 "forge_simplex_dismissed", "forge_pooled_dismissed", "ltt_iid_dismissed", "best_up_to_pi_max",
                 "best_every_mix", "forge_simplex_largest_violation", "land_reference_forge_pooled",
                 "land_reference_ltt_iid", "land_real_ltt_iid", "land_calibration_first_row",
                 "land_calibration_last_row"]
        print(table[[c for c in shown if c in table]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    blaster = variants.blaster_table(base_run=args.base_e23)
    blaster.to_csv(out / "blaster_unseen.csv", index=False)
    print(f"\n{len(blaster)} feature sets with the Blaster worm held out, written to {out / 'blaster_unseen.csv'}")
    if len(blaster):
        print(blaster.drop(columns=["run_id"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
