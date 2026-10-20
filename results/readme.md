# Results used in the paper

Each file here is written by `analysis/paper_results.py` from the pooled tables of the experiments over seeds and the run folders behind them. The tables under `tables/` are written first and every figure under `figures/` is drawn from them, so `python analysis/paper_results.py` reproduces the folder from the pulled runs, and `tables/settings.csv` holds the ε, δ, π_max and π_cal of every experiment with its first and last run id. The table below repeats those runs with the source runs of the experiments that read another one, and is updated with the folder.

| Experiment | Seeds | Runs |
|---|---|---|
| `e11_anoshift_prevalence` | 20 | `20261017-050051_38ae553` to `20261017-063221_38ae553` |
| `e12_litnet_prevalence` | 20 | `20261017-050513_38ae553` to `20261017-081538_38ae553` |
| `e13_ugr16_prevalence` | 20 | `20261019-015220_984f87d` to `20261019-062947_984f87d`, at the operating target of UGR16, ε = 0.005, and at the share start of the certificate |
| `e21_monitor_streams` | 20 | `20261016-173018_38ae553` to `20261016-181931_38ae553`, each on the `e11` run of its seed, `20261013-223628_8502bf0` and `20261015-041654_6cb2cb3` to `20261015-052710_6cb2cb3` |
| `e22_monitor_sweep` | 5 | `20261016-024735_3c469f1` to `20261016-030148_3c469f1` with seed 1 as `20261016-050311_c9c5de8`, each on the `e11` run of its seed as above |
| `e23_unseen_family` | 5 | `20261016-145016_38ae553` to `20261017-022706_38ae553`, the controls checked against the `e12` runs of the same seeds |
| `e24_ugr16_streams` | 20 | `20261019-120648_71e1947` to `20261020-014853_71e1947`, each on the `e13` run of its seed |
| `e31_anoshift_scorers` | 5 | `20261015-115233_6a69429` and `20261015-201004_d030bab` to `20261015-205047_d030bab` |
| `e32_litnet_scorers` | 5 | `20261015-112834_6a69429` and `20261015-184020_d030bab` to `20261015-194843_d030bab` |

The runs of `e21`, `e23` and `e24` hold two streams, the simulated one and the replay of the real order, in a `stream` column of their tables; the paper's figures and the E7 monitoring stage read the replay, the simulated stream stays beside it as the check. The runs of `e11`, `e12` and `e13` hold LTT at an estimated prevalence in a plug in and an upper bound variant, `ltt_plugin` and `ltt_upper`, beside the other methods. UGR16 runs at ε = 0.005 with π_max = 0.1, its operating target, and its certificate starts at half of the benign flows, the share start, where AnoShift and LITNET keep the feasible start; `tables/settings.csv` carries the settings of every experiment.

| File | Experiments | Content |
|---|---|---|
| `tables/settings.csv` | all | the settings every seed shares and the first and last run id per experiment |
| `tables/e1_for_by_prevalence.csv` | `e11`, `e12`, `e13` | E1, the FOR, the violation share and the dismissed share of every method by prevalence and mix on the windows of RQ1, medians and 5% to 95% ranges over seeds, the violation share also over the draws of every seed |
| `tables/e2_by_family.csv` | `e12`, `e13` | E2, the violation share of every method on the window of each family at the main prevalence, reference, S2 and real arms, a `dataset` column and the window named by its `segment` on LITNET and its `start` on UGR16 |
| `tables/e2_natural.csv` | `e12`, `e13` | E2, the real windows at their own prevalence, both datasets |
| `tables/e3_by_policy.csv` | `e21`, `e24` | E3, the alarm, the false alarms, the leakage, the audits and the share of sequences leaving the bounds of every policy under each monitoring arm and composition, per stream, over seeds |
| `tables/e4_by_cell.csv` | `e22` | E4, the alarm, the false alarms, the leakage and the audits of every policy per cell of the sweep, the audit rate, the allocation with its floor and the restart rule with its delay, over seeds |
| `tables/e5_by_scorer.csv` | `e31`, `e32` | E5, the certified share of every scorer at the main ε with the violation share in the claim and the scorer's yardstick, AUC, low tail and ECE |
| `tables/e5_by_epsilon.csv` | `e31`, `e32` | E5, the certified share of every scorer by ε on the reference and the real windows |
| `tables/e6_by_family.csv` | `e23` | E6, per held out family the alarm, the leakage, the FOR and the miss rate under every policy and arm, the family unseen and known, per stream |
| `tables/e6_windows.csv` | `e23` | E6, the AUC, the benign share dismissed and the miss rate of each family's window with the family unseen against known |
| `tables/e7_certification.csv` | `e11`, `e12`, `e13` | E7, the workload per million flows of every method and yardstick on the windows of RQ1 against its violation share, over seeds |
| `tables/e7_monitoring.csv` | `e21`, `e23`, `e24` | E7, the workload, the audits and the attacks let through of every policy under each monitoring arm on the real streams, per stream |
| `tables/e7_frontier.csv` | `e11`, `e12` | E7, the methods no other method beats on both the workload and the violation share at the main prevalence |
| `figures/e1_for_by_prevalence` | `e11`, `e12`, `e13` | the E1 figure, one row per dataset at the window's own mix |
| `figures/e5_dismissed_by_epsilon` | `e31`, `e32` | the certified share by ε per scorer with ranges, one panel per dataset |
| `figures/e5_dismissed_vs_tail` | `e31`, `e32` | the certified share at the main ε against the low tail of the hardest family and against the ECE |
| `figures/e6_leakage_by_family` | `e23` | the alarm inside the family's window and the attacks dismissed until the switch off per held out family and arm, on the replay |
| `figures/e7_workload_vs_risk` | `e11`, `e12`, `e13`, `e21`, `e23`, `e24` | the workload against the violation share at the certification stage and against the attacks let through at the monitoring stage, the monitoring stage on the replay, a sequence per year on AnoShift, one sequence on LITNET and a sequence per week on UGR16 |

Every figure comes as `<name>.png` at 600 dpi, `<name>.pdf` at paper size and `<name>_a4.pdf` filling one A4 page.
