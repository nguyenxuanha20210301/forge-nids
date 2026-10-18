"""pools the imported runs of one experiment over scorer seeds into the tables of E1 to E6, with the median and the
5% to 95% range over seeds of every number.

    python analysis/pool_seeds.py e12_litnet_prevalence
    python analysis/pool_seeds.py e23_unseen_family --out drive/pooled/e23_unseen_family
    python analysis/pool_seeds.py e11_anoshift_prevalence --from 20261013-223628_8502bf0
    python analysis/pool_seeds.py e21_monitor_streams --from <first run of the seeds>

the runs come from the runs folder of the storage root, at or after the run id given with --from, one finished run
per seed, the latest, and their configs must agree in everything but the seed and the keys named with --ignore, so
the runs of an earlier design stay out through --from. the tables go to pooled/<experiment> under
the storage root unless --out names another folder, as csv files, one per saved table of the experiment plus the E
tables, the summaries by seed and the list of runs. for e23 the control of every seed is also set against the e12 run
with the same seed, when the runs folder holds one. e21 and e22 read their thresholds from the e11 run of the same
seed, so their runs differ in the run id of the source by design, and the seed of every source run is checked
instead, when the runs folder holds it.
"""

import argparse
from pathlib import Path

import pandas as pd

from forge import paths, runs
from forge.eval import seeds


def reading(experiment: str, tables: dict) -> None:
    # the lines of each E table that the plan quotes, at the main prevalence or the typical composition
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    if "e1_for_by_prevalence" in tables:
        t = tables["e1_for_by_prevalence"]
        main = t.loc[t["main"], ["method", "mix", "prevalence"]
                     + seeds._stats(t, ["violation_rate", "dismissed_mean", "refused"])]
        print(f"\nE1, the main prevalence of the RQ1 windows over {t['seeds'].max()} seeds")
        print(main.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if "e2_by_family" in tables:
        t = tables["e2_by_family"]
        t = t[t["arm"] == "reference"]
        print(f"\nE2, the reference window of every family at the main prevalence, share of draws with FOR above "
              "epsilon")
        print(t[[c for c in ["segment", "start"] if c in t] + ["method", "in_claim"] + seeds._stats(t, ["rate"])]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if "e5_by_scorer" in tables:
        t = tables["e5_by_scorer"]
        print(f"\nE5, the certified share of every scorer at the main epsilon on the reference windows over "
              f"{t['seeds'].max()} seeds, the violation share over the windows in the claim")
        print(t[["scorer", "kind"] + seeds._stats(t, ["dismissed_mean", "refused", "best_every_mix_dismissed",
                                                         "claim_violation_rate", "benign_at_miss_hardest"])]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if "e6_by_family" in tables:
        t = tables["e6_by_family"]
        t = t[(t["case"] == "unseen") & (t["composition"] == "typical") & (t["policy"] == "forge_simplex")
              & (t["arm"].isin(["monitor", "monitor_any"]))]
        print("\nE6, the family unseen at the typical composition on the offline FORGE stream")
        print(t[[c for c in ["stream", "unit", "arm"] if c in t]
                + seeds._stats(t, ["alarm_in_unseen_rate", "unseen_leak_median", "unseen_miss_mean"])]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if "e6_windows" in tables:
        print("\nE6, the family's window with the family unseen against known, at the offline FORGE thresholds")
        t = tables["e6_windows"]
        print(t[["unit", "family"] + seeds._stats(t, ["miss_unseen", "miss_known", "auc_unseen", "auc_known"])]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if "e3_by_policy" in tables:
        t = tables["e3_by_policy"]
        t = t[t["arm"].isin(["monitor", "monitor_year", "monitor_week", "none"])]
        print(f"\nE3, the alarm and the leakage of every policy under one sequence, a sequence per year or per week "
              f"and no monitoring over {t['seeds'].max()} seeds")
        print(t[[c for c in ["stream", "composition", "policy", "arm"] if c in t]
                + seeds._stats(t, ["alarm_rate", "alarm_block_median", "leak_median", "miscover_sequence_rate"])]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if "e4_by_cell" in tables:
        t = tables["e4_by_cell"]
        t = t[t["policy"] == "conformal_fnr"]
        print(f"\nE4, every cell on the conformal FNR stream over {t['seeds'].max()} seeds")
        print(t[["name", "composition"] + seeds._stats(t, ["alarm_rate", "leak_median", "audits_per_million",
                                                           "miscover_sequence_rate"])]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--ignore", nargs="*", default=[], metavar="KEY",
                        help="config keys that may differ between the runs, such as certify.ltt_guarded_start")
    parser.add_argument("--from", dest="first", default=None, metavar="RUN_ID", help="the earliest run to pool")
    args = parser.parse_args()
    out = args.out or paths.storage_root() / "pooled" / args.experiment
    tables = seeds.pool_experiment(args.experiment, out=out, ignore=args.ignore, first=args.first)
    if args.experiment == "e23_unseen_family":
        seeds.check_e23_controls(tables["runs"].assign(
            dir=[paths.runs_dir() / args.experiment / r for r in tables["runs"]["run_id"]],
            config=[runs.load_run_config(paths.runs_dir() / args.experiment / r)
                    for r in tables["runs"]["run_id"]]))
    reading(args.experiment, tables)


if __name__ == "__main__":
    main()
