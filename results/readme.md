# Results used in the paper

Each file here is written by `analysis/paper_results.py` from the pooled tables of the experiments over seeds and the run folders behind them, and lists its runs below. Nothing here is edited by hand. The tables under `tables/` are written first and every figure under `figures/` is drawn from them, so `python analysis/paper_results.py` reproduces the folder from the pulled runs, and `tables/settings.csv` holds the ε, δ, π_max and π_cal of every experiment with its first and last run id.

| Experiment | Seeds | Runs |
|---|---|---|
| `e11_anoshift_prevalence` | 20 | `20261013-223628_8502bf0` and `20261015-041654_6cb2cb3` to `20261015-052710_6cb2cb3` |
| `e12_litnet_prevalence` | 20 | `20261013-225331_68ab01e` and `20261015-053136_6cb2cb3` to `20261015-064722_6cb2cb3` |
| `e21_monitor_streams` | 1 | `20261014-232506_44dd25a` |
| `e23_unseen_family` | 5 | `20261015-005402_e8d1ff4` and `20261015-065054_6cb2cb3` to `20261015-075703_6cb2cb3` |
| `e31_anoshift_scorers` | 5 | `20261015-115233_6a69429` and `20261015-201004_d030bab` to `20261015-205047_d030bab` |
| `e32_litnet_scorers` | 5 | `20261015-112834_6a69429` and `20261015-184020_d030bab` to `20261015-194843_d030bab` |

| File | Experiments | Content |
|---|---|---|
| `tables/settings.csv` | all | the settings every seed shares and the first and last run id per experiment |
| `tables/e1_for_by_prevalence.csv` | `e11`, `e12` | E1, the FOR, the violation share and the dismissed share of every method by prevalence and mix on the windows of RQ1, medians and 5% to 95% ranges over seeds, the violation share also over the draws of every seed |
| `tables/e2_by_family.csv` | `e12` | E2, the violation share of every method on the window of each family at the main prevalence, reference, S2 and real arms |
| `tables/e2_natural.csv` | `e12` | E2, the real windows at their own prevalence |
| `tables/e5_by_scorer.csv` | `e31`, `e32` | E5, the certified share of every scorer at the main ε with the violation share in the claim and the scorer's yardstick, AUC, low tail and ECE |
| `tables/e5_by_epsilon.csv` | `e31`, `e32` | E5, the certified share of every scorer by ε on the reference and the real windows |
| `tables/e6_by_family.csv` | `e23` | E6, per held out family the alarm, the leakage, the FOR and the miss rate under every policy and arm, the family unseen and known |
| `tables/e6_windows.csv` | `e23` | E6, the AUC, the benign share dismissed and the miss rate of each family's window with the family unseen against known |
| `tables/e7_certification.csv` | `e11`, `e12` | E7, the workload per million flows of every method and yardstick on the windows of RQ1 against its violation share, over seeds |
| `tables/e7_monitoring.csv` | `e21`, `e23` | E7, the workload, the audits and the attacks let through of every policy under each monitoring arm on the real streams |
| `tables/e7_frontier.csv` | `e11`, `e12` | E7, the methods no other method beats on both the workload and the violation share at the main prevalence |
| `figures/e1_for_by_prevalence` | `e11`, `e12` | the E1 figure, one row per dataset at the window's own mix |
| `figures/e5_dismissed_by_epsilon` | `e31`, `e32` | the certified share by ε per scorer with ranges, one panel per dataset |
| `figures/e5_dismissed_vs_tail` | `e31`, `e32` | the certified share at the main ε against the low tail of the hardest family and against the ECE |
| `figures/e6_leakage_by_family` | `e23` | the alarm inside the family's window and the attacks dismissed until the switch off per held out family and arm |
| `figures/e7_workload_vs_risk` | `e11`, `e12`, `e21`, `e23` | the workload against the violation share at the certification stage and against the attacks let through at the monitoring stage |

Every figure comes as `<name>.png` at 600 dpi, `<name>.pdf` at paper size and `<name>_a4.pdf` filling one A4 page.
