# Plan

As of 2026-10-09. Derived from the research direction of the same date, rewritten to match the repository layout. The proposal in `docs/proposal.md` holds the argument, this file holds the order of work and where each piece lands in the code.

## Problem in one formula

A flow with score `s(x) <= t` is dismissed, everything else is escalated. The operational quantity is the false omission rate

    FOR(t; pi) = pi m(t) / (pi m(t) + (1 - pi) b(t))

with `pi` the attack prevalence, `m(t)` the miss rate on attacks and `b(t)` the dismiss rate on benign traffic. FOR increases with `pi` and `m`, decreases with `b`. The goal is the largest dismiss rate with `FOR <= eps` at confidence `1 - delta`.

## Contributions and code

| Contribution | Package | Experiments |
|---|---|---|
| C1, FOR certificate valid for every prevalence up to `pi_max` and every attack family mix in a set W | `forge.certify` | e10 to e19, RQ1 |
| C2, post-deployment monitoring with one-sided labels, random audits with known probability `q`, betting confidence sequences | `forge.monitor` | e20 to e29, RQ2 and RQ4 |
| C3, scorer and ensemble tuned for certified dismiss rate, selected by Pareto testing | `forge.models` | e30 to e39, RQ3 and RQ5 |
| C4, temporal evaluation protocol, per-window prevalence and family mix | `forge.data`, `forge.eval` | e01 to e09, used by all |

`e00_smoke` checks the run loop on synthetic scores and stays as the first thing to run on a new Colab setup.

## Research questions

- **RQ1.** At the same `eps`, how do an FNR guarantee (Şahin and Mert 2026), an i.i.d. FOR guarantee (plain LTT, as in Syed and Yasaei 2026) and the prevalence-robust FOR certificate (C1) compare in realised FOR and dismiss rate across windows of different prevalence?
- **RQ2.** Under real temporal drift (LITNET-2020 by month, AnoShift NEAR and FAR), how quickly does C2 detect a violation, at which audit rate `q`? Compared with no monitoring, ACI and DtACI, periodic recalibration, Drift2Act.
- **RQ3.** Which scorer gives the highest certified dismiss rate: calibrated single model, fixed fusion, per-sample gating, or the differentiable C3 objective?
- **RQ4.** For unseen attack families, where does the certificate fail, and how fast does C2 catch it?
- **RQ5.** Total workload, escalations plus audits, against baselines at the same safety level.

Required baselines: Şahin and Mert dual-threshold conformal deferral, Mondrian conformal, Transcending Transcend, SGR, plain LTT and CRC, conformal selection, non-exchangeable CRC, ACI and DtACI, periodic recalibration, a fixed threshold chosen on validation.

## Datasets

| Dataset | Role | Config |
|---|---|---|
| LITNET-2020 | main, about ten months, prevalence from 0.18% to 15.8% across collection periods | `configs/data/litnet2020.yaml` |
| UGR16 | real ISP traffic, built-in calibration and test periods | `configs/data/ugr16.yaml` |
| Kyoto 2006+ through AnoShift | about ten years, IID, NEAR and FAR splits, inverted prevalence | `configs/data/anoshift.yaml` |
| AIT-ADS | optional, real SOC alerts | not yet |

Not used as main datasets: Edge-IIoTset (label leak through file formatting), Gotham 2025 (no long time axis), original UNSW-NB15, ToN_IoT, BoT-IoT, CIC-IDS2017 and 2018 (documented design flaws). A corrected CIC-IDS2017 may serve as a stress test only.

## Order of work

1. Read the seven closest papers in full: Şahin and Mert 2026, Syed and Yasaei 2026, Li et al. 2605.20956, Ahmed et al. 2026, Xu et al. NeurIPS 2024, Timans et al. 2506.16416, Waudby-Smith and Ramdas JRSS-B 2024. If one of them already does C1 or C2, adjust before writing code.
2. Run a novelty search on Scopus, IEEE Xplore and Scholar with the keywords in the proposal, and check by hand the two papers not read in full (FADES, Electronics 2026, and the ResearchGate conformal IDS paper).
3. Set up the loop. `e00_smoke` runs on Colab and its figures come back to the local mirror.
4. Write the C1 proof and check it on synthetic data where the true FOR is known, `forge.data.synthetic`, experiment `e10`.
5. Build the temporal pipelines for LITNET-2020 and UGR16, `e01` and `e02`, and plot prevalence per window and per attack family.
6. Run RQ1. If C1 holds, post a preprint.
7. C2 with RQ2 and RQ4.
8. C3 with RQ3 and RQ5.

## Sentences the paper must not contain

- "We are the first to apply conformal prediction or risk control to SOC triage." Şahin and Mert 2026 did.
- "We are the first to certify the share of positives among dismissed cases." Syed and Yasaei 2026 did, for ICU alarms.
- "We find that conformal prediction breaks under drift in NIDS." Ahmed et al. 2026 showed it.
- "A competence estimator without kNN is new." META-DES 2015, Davtalab et al. 2024.
- "Combining calibrated models loses calibration, as we show." Ranjan and Gneiting 2010.
- "Coverage at a target risk level is a new metric." It is standard in selective classification.

A defensible claim: to our knowledge, the first finite-sample guarantee on the share of attacks in automatically dismissed traffic that holds under prevalence shift and is maintained after deployment when only escalated traffic is labelled.

## Open

- C2 relies on the betting confidence sequence staying valid for the running average of conditional means under arbitrary drift. The exact conditions in Waudby-Smith and Ramdas need checking before the theorem is stated.
- Target venue: IEEE TIFS or TDSC if C1 and C2 have tight theorems and the drift experiments are strong, Computers and Security or IEEE TNSM otherwise.
