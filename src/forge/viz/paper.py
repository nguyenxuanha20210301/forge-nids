"""the figures of the paper over seeds, drawn from the tables forge.eval.paper writes.

every function takes the written tables and returns a figure, so the figures are redrawn from the results folder
alone. a point is the median over seeds and its bar the 5% to 95% range over seeds, a share of draws the share over
the draws of every seed.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from forge.experiments.litnet_prevalence import EVERY_LINE, LABELS, MARKERS, METHODS
from forge.experiments.monitor_streams import POLICY_LABELS, _policy_line
from forge.experiments.scorer_comparison import _line as scorer_line
from forge.experiments.unseen_family import ARM_LABELS, _arm_line
from forge.eval.inference import OPERATING_ARMS
from forge.eval.paper import paper_rows
from forge.viz import layout, prevalence, style
from forge.viz.prevalence import BEST_LINES

DATASET_LABELS = {"anoshift": "AnoShift, blocks of 2010", "litnet2020": "LITNET-2020, capture windows",
                  "ugr16": "UGR'16, windows of six hours"}
DATASET_SHORT = {"anoshift": "AnoShift", "litnet2020": "LITNET-2020", "ugr16": "UGR'16"}
DATASET_UNITS = {"anoshift": "sessions", "litnet2020": "flows", "ugr16": "flows"}
# the yardsticks of E7 in the dashes and words of the E1 figure
YARDSTICK_LINES = {name: (dash, label) for name, dash, label in [*BEST_LINES, EVERY_LINE]}
# the arms of E6 the paper shows, the one sequence, a sequence from every window, the per window audit and none
E6_ARMS = ["monitor", "monitor_any", "window_audit", "none"]
# the monitoring arm of E7 per dataset, the operating rule of the paper
E7_MONITOR_ARMS = OPERATING_ARMS
# the baselines of the tables only, conformal selection and the two at a published miss rate, since eleven methods
# in the panels over prevalences would hide each other
TABLE_ONLY = ("conformal_selection", "sahin_mert", "np_umbrella")
FIGURE_METHODS = [m for m in METHODS if m not in TABLE_ONLY]


def _method_line(method: str) -> dict:
    return {"color": style.color_for(method), "marker": MARKERS[method], "label": LABELS[method]}


def _with_columns(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    t = table.copy()
    for col in columns:
        if col not in t:
            t[col] = pd.Series(dtype=object)
    return t


def _range(table: pd.DataFrame, column: str):
    # the median with its range over seeds, or the plain column three times when no seed moved it
    if f"{column}_median" in table:
        return table[f"{column}_median"], table[f"{column}_low"], table[f"{column}_high"]
    return table[column], table[column], table[column]


def e1_for_by_prevalence(rows: list[dict]) -> plt.Figure:
    """one row per dataset, each a dict with the E1 rows of its mix as data, its settings as summary and its label.
    left the FOR of the certified pairs, middle the share of pairs with FOR above epsilon, right the share of the
    window dismissed, with the yardsticks."""
    style.use()
    methods = [m for m in FIGURE_METHODS if any(m in set(r["data"]["method"]) for r in rows)]
    drawn = [{"data": r["data"], "summary": r["summary"], "label": r["label"], "window": "reference window",
              "unit": DATASET_UNITS.get(r.get("dataset", ""), "sessions")} for r in rows]
    # the floor of the FOR axis follows the medians, since the low quantile over seeds of a FOR that is zero in some
    # seeds is an interpolation between zero and the next value and would stretch the axis
    medians = np.concatenate([r["data"]["for_median"].to_numpy(dtype=float) for r in rows])
    medians = medians[np.isfinite(medians) & (medians > 0)]
    floor = 10.0 ** np.floor(np.log10(medians.min())) if len(medians) else None
    return prevalence.for_by_prevalence_rows(drawn, methods, _method_line, [*BEST_LINES, EVERY_LINE], floor)


def e5_dismissed_by_epsilon(table: pd.DataFrame, main_epsilon: dict) -> plt.Figure:
    """the certified share of the reference windows dismissed under every scorer by epsilon, one panel per dataset,
    S-best left out since it repeats a base scorer."""
    style.use()
    datasets = [d for d in DATASET_LABELS if d in set(table["dataset"])]
    t = table[(table["arm"] == "reference") & (table["subset"] == "all") & (table["scorer"] != "s_best")]
    fig, axes = plt.subplots(1, len(datasets), figsize=(style.WIDTH_DOUBLE, 2.8), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, dataset, letter in zip(axes, datasets, "ab"):
        g = t[t["dataset"] == dataset]
        names = list(dict.fromkeys(g["scorer"]))
        for k, scorer in enumerate(names):
            h = g[g["scorer"] == scorer].sort_values("epsilon")
            line = scorer_line(scorer, h["kind"].iloc[0])
            x = layout.dodge(h["epsilon"], k, len(names), 0.012, log=True)
            mid, low, high = _range(h, "dismissed_mean")
            layout.points(ax, x, mid, low, high, **line)
        ax.axvline(main_epsilon[dataset], color=style.GRAY, linestyle="-.", linewidth=1, label=r"main $\varepsilon$")
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(style.percent)
        ax.set_xlabel(f"$\\varepsilon$, {DATASET_LABELS[dataset]}")
        layout.panel_letter(ax, letter)
    axes[0].set_ylabel("certified share of the window dismissed")
    layout.legend_below(fig, axes[-1], [], ncol=4)
    return fig


def e5_dismissed_vs_tail(table: pd.DataFrame) -> plt.Figure:
    """the certified share of every scorer at the main epsilon against the benign share dismissed at a 1% miss of
    its hardest family, left, and against its expected calibration error, right, both datasets, LITNET filled and
    AnoShift hollow."""
    style.use()
    t = table[table["scorer"] != "s_best"]
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH_DOUBLE, 2.8), sharey=True)
    faces = {"litnet2020": None, "anoshift": "white"}
    for ax, column in zip(axes, ["benign_at_miss_hardest", "ece"]):
        for r in t.itertuples():
            h = t[(t["scorer"] == r.scorer) & (t["dataset"] == r.dataset)]
            x, x_low, x_high = (v.iloc[0] for v in _range(h, column))
            y, y_low, y_high = (v.iloc[0] for v in _range(h, "dismissed_mean"))
            if not np.isfinite(x):
                continue
            line = {k: v for k, v in scorer_line(r.scorer, r.kind).items() if k != "linestyle"}
            face = faces.get(r.dataset)
            ax.errorbar(x, y, xerr=[[max(x - x_low, 0)], [max(x_high - x, 0)]],
                        yerr=[[max(y - y_low, 0)], [max(y_high - y, 0)]], linestyle="none", markersize=4,
                        elinewidth=0.8, capsize=0, markerfacecolor=face if face else line["color"], **line)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(style.percent)
    axes[0].set_xlim(0, 1)
    axes[0].xaxis.set_major_formatter(style.percent)
    axes[0].set_xlabel("benign share dismissed at a 1% miss of the hardest family")
    axes[1].set_xlabel("expected calibration error")
    axes[0].set_ylabel("certified share dismissed")
    for ax, letter in zip(axes, "ab"):
        layout.panel_letter(ax, letter)
    hollow = Line2D([], [], color=style.GRAY, marker="o", markerfacecolor="white", linestyle="none")
    filled = Line2D([], [], color=style.GRAY, marker="o", linestyle="none")
    layout.legend_below(fig, axes[0], [], [(filled, "LITNET-2020"), (hollow, "AnoShift")], ncol=4)
    return fig


def e6_leakage_by_family(table: pd.DataFrame, windows: pd.DataFrame, policy: str = "forge_simplex") -> plt.Figure:
    """per held out family, top the share of draws with the alarm inside the family's window, bottom the attacks of
    the family dismissed until the switch off, under each arm, one column per composition, the family unseen and the
    thresholds of one policy."""
    style.use()
    units = list(dict.fromkeys(windows["unit"]))
    labels = windows.drop_duplicates("unit").set_index("unit")["segment"].reindex(units)
    compositions = list(dict.fromkeys(table["composition"]))
    t = paper_rows(table)
    t = t[(t["case"] == "unseen") & (t["policy"] == policy)]
    arms = [a for a in E6_ARMS if a in set(t["arm"])]
    fig, axes = plt.subplots(2, len(compositions), figsize=(style.WIDTH_DOUBLE, 4.4), sharex=True, squeeze=False)
    x = np.arange(len(units))
    for col, composition in enumerate(compositions):
        g = t[t["composition"] == composition]
        for k, arm in enumerate(arms):
            h = g[g["arm"] == arm].set_index("unit").reindex(units)
            xs = layout.dodge(x, k, len(arms), 0.14)
            line = _arm_line(arm)
            if arm != "none":
                mid, low, high = _range(h, "alarm_in_unseen_rate")
                if "alarm_in_unseen_rate_pooled" in h:
                    mid = h["alarm_in_unseen_rate_pooled"]
                layout.points(axes[0, col], xs, mid, low, high, linestyle="none", **line)
            mid, low, high = _range(h, "unseen_leak_median")
            layout.points(axes[1, col], xs, np.maximum(mid, 1), np.maximum(low, 1), np.maximum(high, 1),
                          linestyle="none", **line)
        axes[0, col].set_ylim(-0.02, 1.02)
        axes[1, col].set_yscale("log")
        axes[1, col].set_xticks(x, labels, rotation=30, ha="right")
        axes[1, col].set_xlabel(f"held out family, {composition} composition")
    axes[0, 0].set_ylabel("share of draws with an alarm\ninside the window")
    axes[1, 0].set_ylabel("attacks of the family dismissed\nuntil the switch off")
    for ax, letter in zip(axes.flat, "abcdef"):
        layout.panel_letter(ax, letter)
    layout.legend_below(fig, axes[1, 0], [ARM_LABELS[a] for a in arms], ncol=2)
    return fig


def e7_dismissed_by_contract(table: pd.DataFrame) -> plt.Figure:
    """the share of the flows FORGE dismisses at the calibration prevalence and mix against pi_max, one line per
    epsilon and one panel per dataset, the median over seeds with the 5% to 95% range over seeds, a refused draw
    dismissing nothing, the operating contract of each dataset ringed, and on the right axis the workload the same
    share leaves, the escalated flows and the audits of the dismissed ones per 1,000 flows."""
    style.use()
    datasets = [d for d in DATASET_LABELS if d in set(table["dataset"])]
    epsilons = sorted(table["epsilon"].unique())
    # one blue from light to dark as epsilon loosens, with a marker each for gray print
    shades = np.linspace(0, len(style.SEQUENTIAL) - 1, len(epsilons)).round().astype(int)
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(style.WIDTH_DOUBLE, 2.8), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, dataset, letter in zip(axes, datasets, "abc"):
        g = table[table["dataset"] == dataset]
        for k, eps in enumerate(epsilons):
            h = g[np.isclose(g["epsilon"], eps)].sort_values("pi_max")
            x = layout.dodge(h["pi_max"], k, len(epsilons), 0.02, log=True)
            mid, low, high = _range(h, "dismissed_mean")
            layout.points(ax, x, mid, low, high, color=style.SEQUENTIAL[shades[k]], marker=markers[k % len(markers)],
                          label=f"$\\varepsilon$ = {eps:g}")
            ring = h["operating"].astype(bool).to_numpy()
            ax.plot(x[ring], mid.to_numpy()[ring], linestyle="none", marker="o", markersize=9,
                    markerfacecolor="none", markeredgecolor=style.INK, markeredgewidth=0.8)
        pi_max = sorted(g["pi_max"].unique())
        ax.set_xscale("log")
        ax.set_xticks(pi_max, [style.percent(v) for v in pi_max])
        ax.minorticks_off()
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(style.percent)
        ax.set_xlabel(f"$\\pi_{{\\max}}$, {DATASET_SHORT[dataset]}")
        layout.panel_letter(ax, letter)
    pi_cal = float(table["calibration_prevalence"].iloc[0])
    axes[0].set_ylabel(f"flows dismissed at $\\pi$ = {style.percent(pi_cal)}")
    # the workload is 1,000 (1 - (1 - q) D) per 1,000 flows for a dismissed share D and an audit rate q
    q = float(table["audit_rate"].iloc[0])
    right = axes[-1].secondary_yaxis("right", functions=(lambda d: 1000 * (1 - (1 - q) * np.asarray(d)),
                                                         lambda w: (1 - np.asarray(w) / 1000) / (1 - q)))
    right.set_ylabel("workload per 1,000 flows")
    ring = Line2D([], [], linestyle="none", marker="o", markersize=9, markerfacecolor="none",
                  markeredgecolor=style.INK, markeredgewidth=0.8)
    layout.legend_below(fig, axes[0], [], [(ring, "operating contract")], ncol=6)
    return fig


def e7_workload_vs_risk(certification: pd.DataFrame, monitoring: pd.DataFrame) -> plt.Figure:
    """top, the certification stage at the main prevalence with the window's own mix, the workload of every method
    against the share of pairs with FOR above epsilon, the yardsticks as lines. bottom, the monitoring stage at the
    typical composition, the workload of every policy under the monitor of the paper's rule, filled, and without
    monitoring, hollow, against the attacks let through per million flows. one column per dataset."""
    style.use()
    # a stage the tables lack comes as an empty frame and leaves its row of panels blank, and the monitoring stage
    # shows the stream the paper reads
    certification = _with_columns(certification, ["dataset", "mix", "main", "policy"])
    monitoring = paper_rows(_with_columns(monitoring, ["dataset", "composition", "arm", "policy"]))
    datasets = [d for d in DATASET_LABELS if d in set(certification["dataset"]) or d in set(monitoring["dataset"])]
    # the axis labels take two lines, since three datasets leave a panel too narrow for one
    fig, axes = plt.subplots(2, max(len(datasets), 1), figsize=(style.WIDTH_DOUBLE, 5.6), squeeze=False)
    sticks = {}
    drawn = set()
    for ax, dataset in zip(axes[0], datasets):
        c = certification[(certification["dataset"] == dataset) & (certification["mix"] == "window")
                          & (certification["main"] == 1)]
        for r in c.itertuples():
            h = c[c["policy"] == r.policy]
            y, y_low, y_high = (v.iloc[0] / 1e3 for v in _range(h, "workload_per_million"))
            if r.policy in YARDSTICK_LINES:
                dash, label = YARDSTICK_LINES[r.policy]
                sticks[label] = ax.axhline(y, color=style.INK, linestyle=dash, linewidth=1)
                continue
            if r.policy not in FIGURE_METHODS:
                continue
            drawn.add(r.policy)
            x = h["violation_rate_pooled"].iloc[0] if "violation_rate_pooled" in h else _range(h, "violation_rate")[0].iloc[0]
            _, x_low, x_high = (v.iloc[0] for v in _range(h, "violation_rate"))
            ax.errorbar(x, y, xerr=[[max(x - x_low, 0)], [max(x_high - x, 0)]],
                        yerr=[[max(y - y_low, 0)], [max(y_high - y, 0)]], linestyle="none", markersize=4,
                        elinewidth=0.8, capsize=0, **_method_line(r.policy))
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel(f"share of pairs with FOR > $\\varepsilon$,\n{DATASET_SHORT[dataset]}")
        ax.set_ylabel("workload per 1,000 flows")
        if c.empty:
            ax.axis("off")
    policies = {}
    for ax, dataset in zip(axes[1], datasets):
        m = monitoring[(monitoring["dataset"] == dataset) & (monitoring["composition"] == "typical")]
        monitor = E7_MONITOR_ARMS.get(dataset, "monitor")
        if monitor not in set(m["arm"]):
            monitor = "monitor"
        for r in m.itertuples():
            if r.arm not in (monitor, "none") or r.policy not in POLICY_LABELS:
                continue
            h = m[(m["policy"] == r.policy) & (m["arm"] == r.arm)]
            x, x_low, x_high = (v.iloc[0] for v in _range(h, "leaked_per_million"))
            y, y_low, y_high = (v.iloc[0] / 1e3 for v in _range(h, "workload_per_million"))
            line = _policy_line(r.policy)
            policies[r.policy] = line
            ax.errorbar(x, y, xerr=[[max(x - x_low, 0)], [max(x_high - x, 0)]],
                        yerr=[[max(y - y_low, 0)], [max(y_high - y, 0)]], linestyle="none", markersize=4,
                        elinewidth=0.8, capsize=0, markerfacecolor=line["color"] if r.arm == monitor else "white",
                        **line)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlabel(f"attacks let through per million flows,\n{DATASET_SHORT[dataset]}")
        ax.set_ylabel("workload per 1,000 flows")
        if m.empty:
            ax.axis("off")
    for ax, letter in zip(axes.flat, "abcdef"):
        layout.panel_letter(ax, letter)
    # the legend reads the top panel with the most methods, and adds the policies that only the monitoring stage
    # holds, the yardsticks and the two faces. a policy drawn at both stages keeps the words of the certification
    # stage, since the datasets name the fixed threshold's part differently
    first = max(axes[0], key=lambda ax: len(ax.get_legend_handles_labels()[1]))
    extra = [(Line2D([], [], linestyle="none", **line), line["label"]) for p, line in policies.items()
             if p not in drawn]
    extra += [(h, label) for label, h in sticks.items()]
    filled = Line2D([], [], marker="o", color=style.INK, linestyle="none")
    hollow = Line2D([], [], marker="o", color=style.INK, markerfacecolor="white", linestyle="none")
    layout.legend_below(fig, first, [], [*extra, (filled, "under FORGE-Monitor"), (hollow, "no monitoring")])
    return fig
