"""the tables and figures of the paper over seeds, from the pooled tables of e11, e12, e13, e13 at the paper's
epsilon, e21, e22, e23, e24, e31 and e32 and the run folders of their seeds, the contracts of E7 recertified from the
calibration counts of the e11, e12 and e13 runs, the checks of the theorems of E0 from one run each of e10, e14 and
e20, the stream of seed 0 of e21 and e24 over time, the ablations of E9 from the runs of e19, and the FOR of one
threshold against the prevalence for the introduction.

    python analysis/paper_results.py
    python analysis/paper_results.py --out results --e21 20261014-232506_44dd25a --audit-rate 0.005
    python analysis/paper_results.py --e10 <run id> --e14 <run id> --e20 <run id>

the pooled tables come from analysis/pool_seeds.py under pooled/ of the storage root, and E7 reads the run folder of
every seed behind them. its monitoring stage on AnoShift reads the e21 runs of the pooled folder, or, when there is
no pooled e21 or --e21 names a run, that one run, the latest finished one unless named. E0 reads the latest finished
run of e10, e14 and e20 in the runs folder unless --e10, --e14 or --e20 names one, and E9 the latest finished run of
every variant of e19, with the e23 run of seed 0 as the base of the Blaster rows. the tables go to
results/tables and the figures to results/figures of the repository unless --out names another folder, the tables
first, and every figure is drawn from the written tables alone. an experiment whose pooled folder is missing is left
out with a line, and the tables and figures that need it are skipped.
"""

import argparse
from pathlib import Path

import pandas as pd

from forge import paths, runs
from forge.eval import checks, contract, inference, paper, seeds, starts, variants
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
    # every pair of epsilon and pi_max certified again on the calibration sets of every seed, redrawn with this
    # machine's numpy
    contracts = {paper.DATASETS[n]: paper.run_folders(pooled[n], paper.EXPERIMENTS[n]) for n in ("e11", "e12", "e13")
                 if n in pooled}
    if any(contracts.values()):
        out["e7_contract"] = contract.contract_table(contracts, audit_rate=args.audit_rate)
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
    out.update(one_run_tables(pooled, args, out))
    out["intro_for_by_prevalence"] = paper.intro_for_by_prevalence()
    return out


def settings_row(experiment: str, dataset: str, folders: list[Path]) -> dict:
    # the settings of an experiment read from one run, with its first and last run id, for the runs read one by one
    summary = runs.load_json(folders[0], "summary")
    row = {"experiment": experiment, "dataset": dataset, "seeds": 1,
           "first_run": min(f.name for f in folders), "last_run": max(f.name for f in folders)}
    names = {"epsilon": "epsilon", "delta": "delta", "pi_max": "pi_max", "calibration_prevalence": "prevalence"}
    for column, key in names.items():
        value = summary.get(column, summary.get(key))
        if isinstance(value, (int, float)):
            row[column] = value
    return row


def one_run_tables(pooled: dict, args, out: dict) -> dict:
    """the tables read from single runs rather than from pooled seeds, the checks of the theorems of E0, the stream of
    seed 0 of e21 and e24 over time for E3 and the ablations of E9, each experiment added to the settings with its
    runs. a run that is missing leaves its tables out with a line."""
    tables, rows = {}, []
    checked = {}
    for experiment, run_id in ((checks.E10, args.e10), (checks.E14, args.e14), (checks.E20, args.e20)):
        folder = latest_run(experiment, run_id)
        if folder is not None:
            checked[experiment] = folder
            rows.append(settings_row(experiment, "synthetic", [folder]))
    if checks.E10 in checked:
        tables["e0_certify_by_size"] = checks.certify_by_size(checked[checks.E10])
        tables["e0_certify_by_prevalence"] = checks.certify_by_prevalence(checked[checks.E10])
    if checks.E14 in checked:
        tables["e0_online"] = checks.online(checked[checks.E14])
    if checks.E20 in checked:
        tables["e0_monitor"] = checks.monitor(checked[checks.E20])
        tables["e0_monitor_adaptive"] = checks.monitor_adaptive(checked[checks.E20])
        tables["e0_monitor_trace"] = checks.monitor_trace(checked[checks.E20])
    # the stream of seed 0 of every pooled monitoring experiment on a dataset of E3
    traced = {}
    for name in ("e21", "e24"):
        if name in pooled:
            found = paper.run_folders(pooled[name], paper.EXPERIMENTS[name])
            if 0 in found:
                traced[paper.DATASETS[name]] = found[0]
    if traced:
        tables["e3_bounds_over_time"] = paper.e3_bounds_over_time(traced)
    # the case study, the certificate of seed 0 of e13 on the calibration part of UGR16 and its stream through the
    # deployment in the e24 run of the same seed
    if "e13" in pooled and "e24" in pooled:
        certified = paper.run_folders(pooled["e13"], paper.EXPERIMENTS["e13"])
        streamed = paper.run_folders(pooled["e24"], paper.EXPERIMENTS["e24"])
        if 0 in certified and 0 in streamed:
            tables["case_study_certificate"] = paper.case_study_certificate(certified[0], streamed[0])
            tables["case_study_windows"] = paper.case_study_windows(certified[0], streamed[0])
            tables["case_study_stream"] = paper.case_study_stream(streamed[0])
    ablations = variants.e19_table()
    if len(ablations):
        tables["e9_ablations"] = ablations.drop(columns=["seconds"])
        rows.append(settings_row(variants.E19, "litnet2020",
                                 [paths.runs_dir() / variants.E19 / r for r in ablations["run_id"]]))
    else:
        print(f"{variants.E19} has no finished run in the runs folder, left out")
    # the Blaster worm held out under the feature sets of e19, against the e23 run of seed 0 as the base
    base = None
    if "e23" in pooled:
        found = paper.run_folders(pooled["e23"], paper.EXPERIMENTS["e23"])
        base = found[0].name if 0 in found else None
    blaster = variants.blaster_table(base_run=base)
    if len(blaster):
        tables["e9_blaster_unseen"] = blaster
        held = [paths.runs_dir() / variants.E19_BLASTER / r for r in blaster["run_id"] if r != base]
        if held:
            rows.append(settings_row(variants.E19_BLASTER, "litnet2020", held))
    if rows:
        tables["settings"] = pd.concat([out["settings"], pd.DataFrame(rows)], ignore_index=True)
    return tables


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
            drawn.append(("e1_for_by_prevalence_compact", figures.e1_for_by_prevalence_compact(rows)))
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
    if "e7_contract" in read:
        drawn.append(("e7_dismissed_by_contract", figures.e7_dismissed_by_contract(read["e7_contract"])))
    if "intro_for_by_prevalence" in read:
        drawn.append(("intro_for_by_prevalence", figures.intro_for_by_prevalence(read["intro_for_by_prevalence"],
                                                                                 read.get("e2_natural"))))
    if "e0_certify_by_prevalence" in read and "e0_certify_by_size" in read:
        drawn.append(("e0_certify", figures.e0_certify(read["e0_certify_by_prevalence"], read["e0_certify_by_size"])))
    if "e0_online" in read and "e0_monitor_trace" in read:
        drawn.append(("e0_online_and_monitor", figures.e0_online_and_monitor(read["e0_online"],
                                                                             read["e0_monitor_trace"])))
    if "e3_bounds_over_time" in read:
        drawn.append(("e3_bounds_over_time", figures.e3_bounds_over_time(read["e3_bounds_over_time"])))
    if "case_study_windows" in read and "case_study_certificate" in read:
        drawn.append(("case_study_by_clock", figures.case_study_by_clock(
            paper.case_study_figure_rows(read["case_study_windows"], read["case_study_certificate"]))))
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
    parser.add_argument("--e10", default=None, metavar="RUN_ID", help="the e10 run of the checks of Theorem 1")
    parser.add_argument("--e14", default=None, metavar="RUN_ID", help="the e14 run of the checks of Theorem 1b")
    parser.add_argument("--e20", default=None, metavar="RUN_ID", help="the e20 run of the checks of Theorems 2 and 3")
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
