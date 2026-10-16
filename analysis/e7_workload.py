"""E7, the workload of every method at the same safety level, read from the runs of E1 on both datasets and of the
monitoring on their real streams.

    python analysis/e7_workload.py --e11 <run_id> --e12 <run_id> --e21 <run_id> --e23 <run_id>
    python analysis/e7_workload.py --e12 20261013-214554_9c72cd3 --e21 20261014-232506_44dd25a

a run that is not named or not in the runs folder is left out with a line saying so. the tables go to
pooled/e7_workload under the storage root unless --out names another folder, certification.csv with the escalations
of every method on the RQ1 windows against its violation share, monitoring.csv with the escalations, the audits and
the attacks let through on the real streams under each monitoring arm, and frontier.csv with the rows no other row
beats on both the workload and the risk. the figure workload_vs_risk is saved there as well.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forge import paths
from forge.eval import paper, workload
from forge.experiments.litnet_prevalence import LABELS, MARKERS
from forge.experiments.monitor_streams import POLICY_LABELS, _policy_line
from forge.viz import layout, style
from forge.viz.export import save_figure

ARM_MARKERS = {"monitor": "o", "monitor_year": "D", "monitor_any": "D", "none": "x", "window_audit": "s",
               "naive": "^"}


def fig_workload_vs_risk(certification: pd.DataFrame, monitoring: pd.DataFrame) -> plt.Figure:
    style.use()
    datasets = list(dict.fromkeys([*certification.get("dataset", []), *monitoring.get("dataset", [])]))
    fig, axes = plt.subplots(2, max(len(datasets), 1), figsize=(style.WIDTH_DOUBLE, 5.2), squeeze=False)
    # top, the certification stage at the main prevalence with the window's own mix, the workload of every method
    # against the share of draw and window pairs with FOR above epsilon, the yardsticks as ticks. bottom, the
    # monitoring stage at the typical composition, the workload of every policy under the monitor, filled, and
    # without monitoring, hollow, against the attacks let through per million flows
    for ax, dataset in zip(axes[0], datasets):
        c = certification[(certification["dataset"] == dataset) & (certification["mix"] == "window")
                          & certification["main"]]
        for r in c.itertuples():
            if r.policy in MARKERS:
                ax.plot(r.violation_rate, r.workload_per_million / 1e3, marker=MARKERS[r.policy], linestyle="none",
                        color=style.color_for(r.policy), label=LABELS[r.policy])
            else:
                ax.axhline(r.workload_per_million / 1e3, color=style.INK, linestyle=":", linewidth=0.8,
                           label=dict(workload.YARDSTICKS)[r.policy])
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel(r"share of pairs with FOR > $\varepsilon$")
        ax.set_ylabel("workload per 1,000 flows")
        if c.empty:
            ax.axis("off")
    for ax, dataset in zip(axes[1], datasets):
        m = monitoring[(monitoring["dataset"] == dataset) & (monitoring["composition"] == "typical")]
        for r in m.itertuples():
            if r.arm not in ("monitor", "none") or r.policy not in POLICY_LABELS:
                continue
            line = _policy_line(r.policy)
            ax.plot(r.leaked_per_million, r.workload_per_million / 1e3, marker=line["marker"], linestyle="none",
                    color=line["color"], markerfacecolor=line["color"] if r.arm == "monitor" else "none",
                    label=line["label"])
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlabel("attacks let through per million flows")
        ax.set_ylabel("workload per 1,000 flows")
        if m.empty:
            ax.axis("off")
    for ax, letter in zip(axes.flat, "abcdef"):
        layout.panel_letter(ax, letter)
    filled = plt.Line2D([], [], marker="o", color=style.INK, linestyle="none")
    hollow = plt.Line2D([], [], marker="o", color=style.INK, markerfacecolor="none", linestyle="none")
    layout.legend_below(fig, axes[0, 0], [], [(filled, "under FORGE-Monitor"), (hollow, "no monitoring")])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("e11", "e12", "e21", "e23"):
        parser.add_argument(f"--{name}", default=None, metavar="RUN_ID")
    parser.add_argument("--audit-rate", type=float, default=0.005)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or paths.storage_root() / "pooled" / "e7_workload"
    out.mkdir(parents=True, exist_ok=True)
    folders = {"e11": "e11_anoshift_prevalence", "e12": "e12_litnet_prevalence", "e21": "e21_monitor_streams",
               "e23": "e23_unseen_family"}
    found = {}
    for name, experiment in folders.items():
        run_id = getattr(args, name)
        if run_id is None:
            print(f"{experiment} not named, left out")
            continue
        run_dir = paths.runs_dir() / experiment / run_id
        if not (run_dir / "results").is_dir():
            print(f"{experiment} run {run_id} is not in the runs folder, left out")
            continue
        found[name] = run_dir
    certification = pd.concat([workload.certification(found[n], args.audit_rate) for n in ("e11", "e12")
                               if n in found], ignore_index=True) if any(n in found for n in ("e11", "e12")) \
        else pd.DataFrame()
    # a run with both kinds of stream gives the one the paper reads
    monitoring = paper.paper_rows(pd.concat([workload.monitoring(found[n]) for n in ("e21", "e23") if n in found],
                                            ignore_index=True)) if any(n in found for n in ("e21", "e23")) \
        else pd.DataFrame()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    if len(certification):
        certification.to_csv(out / "certification.csv", index=False)
        shown = certification[(certification["mix"] == "window") & certification["main"]]
        print("certification stage, the reference windows at the main prevalence with the window's own mix, "
              "workload per million flows")
        print(shown[["dataset", "policy", "dismissed", "violation_rate", "safe", "escalations_per_million",
                     "audits_per_million", "workload_per_million"]]
              .to_string(index=False, float_format=lambda v: f"{v:,.4f}" if abs(v) < 10 else f"{v:,.0f}"))
        methods = shown[~shown["policy"].isin(dict(workload.YARDSTICKS))]
        front = pd.concat([workload.frontier(g, "workload_per_million", "violation_rate").assign(dataset=d)
                           for d, g in methods.groupby("dataset")], ignore_index=True)
        front.to_csv(out / "frontier.csv", index=False)
        print("\nthe frontier of the methods, the rows no other method beats on both the workload and the violation "
              "share")
        print(front[["dataset", "policy", "violation_rate", "workload_per_million"]]
              .to_string(index=False, float_format=lambda v: f"{v:,.4f}" if abs(v) < 10 else f"{v:,.0f}"))
    if len(monitoring):
        monitoring.to_csv(out / "monitoring.csv", index=False)
        shown = monitoring[monitoring["arm"].isin(["monitor", "monitor_year", "monitor_any", "none"])]
        print("\nmonitoring stage, every policy under the monitor and without it, workload per million flows")
        print(shown[["dataset", "composition", "policy", "arm", "dismissed", "escalations_per_million",
                     "audits_per_million", "workload_per_million", "leaked_per_million", "realized_for_median",
                     "alarm_rate"]]
              .to_string(index=False, float_format=lambda v: f"{v:,.4f}" if abs(v) < 10 else f"{v:,.0f}"))
    if len(certification) or len(monitoring):
        save_figure(fig_workload_vs_risk(certification, monitoring), out, "workload_vs_risk")
        print(f"\nwrote the tables and the figure workload_vs_risk to {out}")


if __name__ == "__main__":
    main()
