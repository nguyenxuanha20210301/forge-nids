"""the tables and figures of the paper over seeds, from the pooled tables of e11, e12, e13, e13 at the paper's
epsilon, e21, e22, e23, e24, e31 and e32 and the run folders of their seeds.

    python analysis/paper_results.py
    python analysis/paper_results.py --out results --e21 20261014-232506_44dd25a --audit-rate 0.005

the pooled tables come from analysis/pool_seeds.py under pooled/ of the storage root, and E7 reads the run folder of
every seed behind them. its monitoring stage on AnoShift reads the e21 runs of the pooled folder, or, when there is
no pooled e21 or --e21 names a run, that one run, the latest finished one unless named. the tables go to
results/tables and the figures to results/figures of the repository unless --out names another folder, the tables
first, and every figure is drawn from the written tables alone. an experiment whose pooled folder is missing is left
out with a line, and the tables and figures that need it are skipped.
"""

import argparse
from pathlib import Path

import pandas as pd

from forge import paths, runs
from forge.eval import inference, paper, seeds, starts
from forge.viz import paper as figures
from forge.viz.export import save_figure

SETTING_KEYS = ["dataset", "epsilon", "main_epsilon", "delta", "pi_max", "calibration_prevalence",
                "main_prevalence", "main_period"]


def latest_run(experiment: str, run_id: str | None) -> Path | None:
    # the named run, else the latest finished run of the experiment in the runs folder
    table = seeds.seed_runs(experiment)
    if run_id is not None:
        table = table[table["run_id"] == run_id]
    done = table[table["status"] == "done"]
    if done.empty:
        print(f"{experiment}, no finished run" + (f" {run_id}" if run_id else "") + " in the runs folder, left out")
        return None
    return Path(done["dir"].iloc[-1])


def tables_of(pooled: dict, args) -> dict:
    out = {}
    settings = []
    for name, experiment in paper.EXPERIMENTS.items():
        if name not in pooled:
            continue
        row = {"experiment": experiment, **paper.settings(pooled[name], SETTING_KEYS)}
        # an experiment that reads a source run carries the dataset of that run
        row.setdefault("dataset", paper.DATASETS.get(name))
        listed = pooled[name]["runs"]
        row.update({"seeds": len(listed), "first_run": listed["run_id"].min(), "last_run": listed["run_id"].max()})
        settings.append(row)
    out["settings"] = pd.DataFrame(settings)
    if any(name in pooled for name in paper.E1_EXPERIMENTS):
        out["e1_for_by_prevalence"] = paper.e1_table(pooled)
    out.update(paper.e2_tables(pooled))
    out.update(paper.e3_tables(pooled))
    out.update(statistics(pooled, out))
    out.update(paper.e5_tables(pooled))
    out.update(paper.e6_tables(pooled))
    certification = [paper.e7_certification(paper.run_folders(pooled[n], paper.EXPERIMENTS[n]), args.audit_rate)
                     for n in ("e11", "e12", "e13")
                     if n in pooled and paper.run_folders(pooled[n], paper.EXPERIMENTS[n])]
    if certification:
        out["e7_certification"] = pd.concat(certification, ignore_index=True)
        out["e7_frontier"] = paper.e7_frontier(out["e7_certification"])
    monitoring = []
    # the e21 runs of the pooled folder stand for the monitoring stage on AnoShift, else one run
    if args.e21 is None and "e21" in pooled and paper.run_folders(pooled["e21"], paper.EXPERIMENTS["e21"]):
        monitoring.append(paper.e7_monitoring(paper.run_folders(pooled["e21"], paper.EXPERIMENTS["e21"])))
    else:
        e21 = latest_run("e21_monitor_streams", args.e21)
        if e21 is not None:
            monitoring.append(paper.e7_monitoring({0: e21}))
            out["settings"] = pd.concat([out["settings"][out["settings"]["experiment"] != "e21_monitor_streams"],
                                         pd.DataFrame([{"experiment": "e21_monitor_streams", "dataset": "anoshift",
                                                        "seeds": 1, "first_run": e21.name, "last_run": e21.name}])],
                                        ignore_index=True)
    for n in ("e23", "e24"):
        if n in pooled and paper.run_folders(pooled[n], paper.EXPERIMENTS[n]):
            monitoring.append(paper.e7_monitoring(paper.run_folders(pooled[n], paper.EXPERIMENTS[n])))
    if monitoring:
        out["e7_monitoring"] = pd.concat(monitoring, ignore_index=True)
    return out


def statistics(pooled: dict, out: dict) -> dict:
    """the tables over seeds read from the run folders behind the pooled tables, the start rules of e13, the price,
    the baselines against FORGE and the violation of FORGE per seed in E1, the operating rule against the other arms in
    E3, and the E1 and E3 tables with the intervals of their pooled shares. an experiment with a run folder missing
    is left out of them with a line, since its counts would not be those of its pooled table."""
    folders = {}
    for name in (*paper.E1_EXPERIMENTS, "e21", "e24"):
        if name not in pooled:
            continue
        found = paper.run_folders(pooled[name], paper.EXPERIMENTS[name])
        if len(found) == len(pooled[name]["runs"]):
            folders[name] = found
        else:
            print(f"{paper.EXPERIMENTS[name]} has {len(found)} of {len(pooled[name]['runs'])} run folders, left out "
                  "of the tables over seeds")
    tables = {}
    e1 = [name for name in paper.E1_EXPERIMENTS if name in folders]
    if e1 and "e1_for_by_prevalence" in out:
        rows = pd.concat([inference.e1_seed_rows(folders[n], paper.EXPERIMENTS[n]) for n in e1], ignore_index=True)
        cells = pd.concat([inference.e1_claim_cells(folders[n]) for n in e1], ignore_index=True)
        tables["e1_for_by_prevalence"] = inference.with_intervals(out["e1_for_by_prevalence"],
                                                                  inference.e1_intervals(rows), inference.E1_KEYS)
        tables["e1_paired_price"] = inference.e1_paired_price(rows)
        tables["e1_paired_baselines"] = inference.e1_paired_baselines(rows)
        tables["e1_violation_by_seed"] = inference.e1_violation_by_seed(cells)
        tables["e1_violation_over_seeds"] = inference.e1_violation_over_seeds(cells)
    # the start rules of the certificate on the calibration parts of the e13 runs, redrawn with this machine's numpy
    if "e13" in folders:
        tables["e1_start_rules"] = starts.start_rules_table(folders["e13"])
    e3 = [name for name in ("e21", "e24") if name in folders]
    if e3 and "e3_by_policy" in out:
        rows = pd.concat([inference.e3_seed_rows(folders[n], paper.DATASETS[n]) for n in e3], ignore_index=True)
        tables["e3_by_policy"] = inference.with_intervals(out["e3_by_policy"], inference.e3_intervals(rows),
                                                          inference.E3_KEYS)
        tables["e3_paired_arms"] = inference.e3_paired_arms(rows)
    return tables


def draw(tables: dict, folder: Path) -> list[str]:
    # every figure from the written tables, read back from the folder they went to
    read = {p.stem: pd.read_csv(p) for p in sorted(folder.glob("*.csv"))}
    settings = read["settings"].set_index("experiment")
    drawn = []
    if "e1_for_by_prevalence" in read:
        t = read["e1_for_by_prevalence"]
        rows = []
        # one row per dataset at its operating target, so UGR16 at the paper's epsilon stays in the table only
        for name in ("e11", "e12", "e13"):
            experiment = paper.EXPERIMENTS[name]
            dataset = paper.DATASETS[name]
            if experiment not in settings.index or dataset not in set(t["dataset"]):
                continue
            s = settings.loc[experiment]
            data = paper.e1_rows(t[t["dataset"] == dataset], "window", float(s["epsilon"]))
            rows.append({"data": data, "dataset": dataset,
                         "summary": {k: float(s[k]) for k in ("epsilon", "delta", "calibration_prevalence", "pi_max")},
                         "label": figures.DATASET_LABELS[dataset]})
        if rows:
            drawn.append(("e1_for_by_prevalence", figures.e1_for_by_prevalence(rows)))
    if "e5_by_epsilon" in read:
        main = {paper.DATASETS[n]: float(settings.loc[paper.EXPERIMENTS[n], "main_epsilon"])
                for n in ("e31", "e32") if paper.EXPERIMENTS[n] in settings.index}
        drawn.append(("e5_dismissed_by_epsilon", figures.e5_dismissed_by_epsilon(read["e5_by_epsilon"], main)))
    if "e5_by_scorer" in read:
        drawn.append(("e5_dismissed_vs_tail", figures.e5_dismissed_vs_tail(read["e5_by_scorer"])))
    if "e6_by_family" in read and "e6_windows" in read:
        drawn.append(("e6_leakage_by_family", figures.e6_leakage_by_family(read["e6_by_family"],
                                                                           read["e6_windows"])))
    if "e7_certification" in read or "e7_monitoring" in read:
        drawn.append(("e7_workload_vs_risk", figures.e7_workload_vs_risk(
            read.get("e7_certification", pd.DataFrame()), read.get("e7_monitoring", pd.DataFrame()))))
    names = []
    for name, fig in drawn:
        save_figure(fig, folder.parent / "figures", name)
        names.append(name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None, help="the results folder, results/ of the repository")
    parser.add_argument("--pooled", type=Path, default=None, help="the pooled folder, pooled/ under the storage root")
    parser.add_argument("--e21", default=None, metavar="RUN_ID", help="the e21 run of the monitoring stage of E7")
    parser.add_argument("--audit-rate", type=float, default=0.005)
    args = parser.parse_args()
    out = args.out or paths.repo_root() / "results"
    pooled_root = args.pooled or paths.storage_root() / "pooled"
    pooled = {}
    for name, experiment in paper.EXPERIMENTS.items():
        folder = pooled_root / experiment
        if not folder.is_dir():
            print(f"{experiment} has no pooled folder under {pooled_root}, left out")
            continue
        pooled[name] = paper.read_pooled(folder)
    tables = tables_of(pooled, args)
    folder = out / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(folder / f"{name}.csv", index=False)
    print(f"wrote {', '.join(tables)} to {folder}")
    names = draw(tables, folder)
    print(f"drew {', '.join(names)} into {out / 'figures'}")
    pd.set_option("display.width", 200)
    print("\nthe runs behind every table and figure")
    print(tables["settings"][["experiment", "seeds", "first_run", "last_run"]].to_string(index=False))


if __name__ == "__main__":
    main()
