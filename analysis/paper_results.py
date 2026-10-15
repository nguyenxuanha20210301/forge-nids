"""the tables and figures of the paper over seeds, from the pooled tables of e11, e12, e23, e31 and e32 and the run
folders of their seeds.

    python analysis/paper_results.py
    python analysis/paper_results.py --out results --e21 20261014-232506_44dd25a --audit-rate 0.005

the pooled tables come from analysis/pool_seeds.py under pooled/ of the storage root, and E7 reads the run folder of
every seed behind them plus one e21 run, the latest finished one unless --e21 names it. the tables go to
results/tables and the figures to results/figures of the repository unless --out names another folder, the tables
first, and every figure is drawn from the written tables alone. an experiment whose pooled folder is missing is left
out with a line, and the tables and figures that need it are skipped.
"""

import argparse
from pathlib import Path

import pandas as pd

from forge import paths, runs
from forge.eval import paper, seeds
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
        listed = pooled[name]["runs"]
        row.update({"seeds": len(listed), "first_run": listed["run_id"].min(), "last_run": listed["run_id"].max()})
        settings.append(row)
    out["settings"] = pd.DataFrame(settings)
    if "e11" in pooled or "e12" in pooled:
        out["e1_for_by_prevalence"] = paper.e1_table(pooled)
    out.update(paper.e2_tables(pooled))
    out.update(paper.e5_tables(pooled))
    out.update(paper.e6_tables(pooled))
    certification = [paper.e7_certification(paper.run_folders(pooled[n], paper.EXPERIMENTS[n]), args.audit_rate)
                     for n in ("e11", "e12") if n in pooled and paper.run_folders(pooled[n], paper.EXPERIMENTS[n])]
    if certification:
        out["e7_certification"] = pd.concat(certification, ignore_index=True)
        out["e7_frontier"] = paper.e7_frontier(out["e7_certification"])
    monitoring = []
    e21 = latest_run("e21_monitor_streams", args.e21)
    if e21 is not None:
        monitoring.append(paper.e7_monitoring({0: e21}))
        out["settings"] = pd.concat([out["settings"], pd.DataFrame([{
            "experiment": "e21_monitor_streams", "dataset": "anoshift", "seeds": 1, "first_run": e21.name,
            "last_run": e21.name}])], ignore_index=True)
    if "e23" in pooled and paper.run_folders(pooled["e23"], paper.EXPERIMENTS["e23"]):
        monitoring.append(paper.e7_monitoring(paper.run_folders(pooled["e23"], paper.EXPERIMENTS["e23"])))
    if monitoring:
        out["e7_monitoring"] = pd.concat(monitoring, ignore_index=True)
    return out


def draw(tables: dict, folder: Path) -> list[str]:
    # every figure from the written tables, read back from the folder they went to
    read = {p.stem: pd.read_csv(p) for p in sorted(folder.glob("*.csv"))}
    settings = read["settings"].set_index("experiment")
    drawn = []
    if "e1_for_by_prevalence" in read:
        t = read["e1_for_by_prevalence"]
        rows = []
        for name in ("e11", "e12"):
            experiment = paper.EXPERIMENTS[name]
            dataset = paper.DATASETS[name]
            if experiment not in settings.index or dataset not in set(t["dataset"]):
                continue
            s = settings.loc[experiment]
            rows.append({"data": paper.e1_rows(t[t["dataset"] == dataset], "window"), "dataset": dataset,
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
