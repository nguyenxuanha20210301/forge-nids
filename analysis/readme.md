# Analysis

Local scripts and notebooks that read pulled runs from `drive/runs/` and compare them. Figures that combine several runs are saved to `results/figures/` with `forge.viz.export.save_figure`.

| Script | Reads | Writes |
|---|---|---|
| `pool_seeds.py <experiment>` | the finished runs of one experiment, one per seed | the tables of E1 to E6 with medians and 5% to 95% ranges over seeds, under `drive/pooled/<experiment>/`, for `e21` and `e22` with the seed of every source run checked |
| `e19_ablations.py` | the runs of `e19_ablations` and `e19_blaster_unseen` | one row per variant and the Blaster rows, under `drive/pooled/e19_ablations/` |
| `e7_workload.py --e11 ... --e12 ... --e21 ... --e23 ...` | the runs of E1, E3 and E6 | the workload of every method at the certification and the monitoring stage and the figure `workload_vs_risk`, under `drive/pooled/e7_workload/` |
| `paper_results.py [--e21 <run_id>]` | the pooled folders of `e11`, `e12`, `e21`, `e22`, `e23`, `e31` and `e32` under `drive/pooled/` and the run folder of every seed behind them, or one `e21` run where the pooled `e21` is missing or the run is named | the tables of the paper under `results/tables/`, then the figures under `results/figures/` drawn from the written tables |
| `e12_checks.py <run_dir>` | one `e12` run | checks printed, nothing written |
| `e12_land_episodes.py <run_dir>` | one `e12` run and the processed table, on Colab | the LAND sources printed, nothing written |

What the paper uses is written to `results/` by `paper_results.py`, with the runs behind every file listed in `results/readme.md`.
