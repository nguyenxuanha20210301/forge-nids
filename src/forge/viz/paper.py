"""the figures of the paper over seeds, drawn from the tables forge.eval.paper writes.

every function takes the written tables and returns a figure, so the figures are redrawn from the results folder
alone. a point is the median over seeds and its bar the 5% to 95% range over seeds, a share of draws the share over
the draws of every seed.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from forge.experiments import synthetic_certify, synthetic_online
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


def intro_for_by_prevalence(curves: pd.DataFrame, natural: pd.DataFrame | None = None) -> plt.Figure:
    """the FOR of one threshold against the prevalence of deployment, one line per miss rate at one benign share
    dismissed, epsilon as a line with the prevalence each miss rate crosses it at marked, and along the foot of the
    axis the natural prevalence of every real window with attacks of the datasets the E2 table holds."""
    style.use()
    fig, ax = plt.subplots(figsize=(style.WIDTH_SINGLE, 2.7))
    rates = sorted(curves["miss_rate"].unique())
    shades = np.linspace(3, len(style.SEQUENTIAL) - 1, len(rates)).round().astype(int)
    eps = float(curves["epsilon"].iloc[0])
    for k, m in enumerate(rates):
        g = curves[curves["miss_rate"] == m].sort_values("prevalence")
        color = style.SEQUENTIAL[shades[k]]
        ax.plot(g["prevalence"], g["for"], color=color, linestyle=style.DASHES[k % len(style.DASHES)],
                linewidth=1.2, label=f"miss rate {style.percent(m)}")
        ax.plot(float(g["crossing"].iloc[0]), eps, linestyle="none", marker="o", markersize=3.5, color=color)
    ax.axhline(eps, color=style.GRAY, linestyle="--", linewidth=1, label=r"$\varepsilon$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(style.percent)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(curves["prevalence"].min(), curves["prevalence"].max())
    # room below the lowest line for the ticks of the real windows
    ax.set_ylim(float(curves["for"].min()) / 15, 2 * float(curves["for"].max()))
    ax.set_xlabel("attack prevalence in deployment")
    ax.set_ylabel("FOR of the dismissed flows")
    # one row of ticks per dataset at the foot of the axis, the windows of a dataset in its own color
    extra = []
    if natural is not None and len(natural):
        windows = natural[natural["prevalence"] > 0].drop_duplicates(["dataset", "window"])
        found = [d for d in DATASET_LABELS if d in set(windows["dataset"])]
        for k, dataset in enumerate(found):
            x = windows.loc[windows["dataset"] == dataset, "prevalence"]
            color = style.COLORS[(k + 1) % len(style.COLORS)]
            ax.vlines(x, 0.01 + 0.06 * k, 0.05 + 0.06 * k, transform=ax.get_xaxis_transform(), color=color,
                      linewidth=0.8)
            extra.append((Line2D([], [], color=color, linewidth=0.8, linestyle="none", marker="|", markersize=6),
                          f"real windows, {DATASET_SHORT[dataset]}"))
    layout.legend_below(fig, ax, [f"miss rate {style.percent(m)}" for m in rates], extra, ncol=2)
    return fig


def e0_certify(by_prevalence: pd.DataFrame, by_size: pd.DataFrame) -> plt.Figure:
    """Theorem 1 on the synthetic calibration sets of e10. a and b, the share of draws whose true FOR exceeds epsilon
    against the prevalence of deployment at the main size, with the family mix of the calibration set and with the
    worst single family. c, the share of draws violating the method's own claim, the floor included, against the
    flows in the calibration set. every share with its 95% interval over the draws."""
    style.use()
    first = by_prevalence.iloc[0]
    delta, pi_cal, pi_max = float(first["delta"]), float(first["calibration_prevalence"]), float(first["pi_max"])
    methods = [m for m in synthetic_certify.PREVALENCE_METHODS if m in set(by_prevalence["method"])]
    line = {m: {"color": style.color_for(m), "marker": synthetic_certify.MARKERS[m],
                "label": synthetic_certify.LABELS[m]} for m in synthetic_certify.METHODS}
    fig, axes = plt.subplots(1, 3, figsize=(style.WIDTH_DOUBLE, 2.7))
    for ax, mix, word, letter in zip(axes[:2], ["calibration", "any"], ["calibration mix", "worst single family"],
                                     "ab"):
        for k, method in enumerate(methods):
            g = by_prevalence[(by_prevalence["method"] == method) & (by_prevalence["mix"] == mix)]
            g = g.sort_values("prevalence")
            layout.points(ax, layout.dodge(g["prevalence"], k, len(methods), 0.045, log=True), g["rate"], g["low"],
                          g["high"], **line[method])
        ax.axhline(delta, color=style.GRAY, linestyle="--", linewidth=1, label=r"$\delta$")
        ax.axvline(pi_cal, color=style.GRAY, linestyle=":", linewidth=1, label=r"$\pi$ of the calibration set")
        ax.axvline(pi_max, color=style.GRAY, linestyle="-.", linewidth=1, label=r"$\pi_{\max}$")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(style.percent)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(f"prevalence, {word}")
        layout.panel_letter(ax, letter)
    axes[0].set_ylabel(r"share of draws with FOR > $\varepsilon$")
    sized = [m for m in synthetic_certify.METHODS if m in set(by_size["method"])]
    for k, method in enumerate(sized):
        g = by_size[by_size["method"] == method].sort_values("size")
        layout.points(axes[2], layout.dodge(g["size"], k, len(sized), 0.045, log=True), g["joint_rate"],
                      g["joint_low"], g["joint_high"], **line[method])
    axes[2].axhline(delta, color=style.GRAY, linestyle="--", linewidth=1)
    axes[2].set_xscale("log")
    axes[2].set_ylim(0, max(2 * delta, 1.15 * float(by_size["joint_high"].max())))
    axes[2].set_xlabel("flows in the calibration set")
    axes[2].set_ylabel("draws violating own claim")
    layout.panel_letter(axes[2], "c")
    extra = [] if "forge_strict" not in sized else [
        (Line2D([], [], color=style.color_for("forge_strict"), marker=synthetic_certify.MARKERS["forge_strict"],
                markersize=3.5, linewidth=1.2), synthetic_certify.LABELS["forge_strict"])]
    layout.legend_below(fig, axes[0], [synthetic_certify.LABELS[m] for m in methods], extra)
    return fig


# the drift of the benign scores of e14 the paper shows, in the words of the axis
ONLINE_SCENARIOS = {"gradual": "benign scores drifting", "abrupt": "benign scores jumping"}


def e0_online_and_monitor(online: pd.DataFrame, trace: pd.DataFrame) -> plt.Figure:
    """a and b, Theorem 1b on the windows of e14 at the main window size and the composition its figures read, the
    share of draws whose threshold has FOR above epsilon in each window, the offline certificate, the online one on
    the window's own scores and the online one of the window before, with the benign scores drifting and jumping.
    c, Theorems 2 and 3 on one stream of e20, the true count of attacks among the dismissed flows, its two bounds,
    epsilon times the dismissed flows, the start of the violation and the alarm."""
    style.use()
    fig, axes = plt.subplots(1, 3, figsize=(style.WIDTH_DOUBLE, 2.7))
    composition = synthetic_online.FIGURE_COMPOSITIONS["violation"]
    shown = online[online["main"].astype(bool) & (online["composition"] == composition)]
    methods = [m for m in synthetic_online.METHODS if m in set(shown["method"])]
    delta = float(online["delta"].iloc[0])
    for ax, (scenario, word), letter in zip(axes[:2], ONLINE_SCENARIOS.items(), "ab"):
        g = shown[shown["scenario"] == scenario]
        for k, method in enumerate(methods):
            h = g[g["method"] == method].sort_values("window")
            layout.points(ax, layout.dodge(h["window"], k, len(methods), 0.25), h["rate"], h["low"], h["high"],
                          color=style.color_for(method), marker=synthetic_online.MARKERS[method],
                          label=synthetic_online.LABELS[method])
        ax.axhline(delta, color=style.GRAY, linestyle="--", linewidth=1, label=r"$\delta$")
        if len(g) and np.isfinite(g["shift_at"].iloc[0]):
            ax.axvline(float(g["shift_at"].iloc[0]) - 0.5, color=style.GRAY, linestyle=":", linewidth=1)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(f"window, {word}")
        layout.panel_letter(ax, letter)
    axes[0].set_ylabel(r"share of draws with FOR > $\varepsilon$")
    ax = axes[2]
    scale = 1e6
    color = style.color_for("forge_simplex")
    counts = trace[trace["point"] == "count"].sort_values("position")
    audits = trace[trace["point"] == "audit"].sort_values("position")
    first = trace.iloc[0]
    n_end, eps = float(first["n_end"]), float(first["epsilon"])
    ax.step(counts["position"] / scale, counts["attacks"], where="post", color=style.INK, linewidth=1,
            label="attacks among the dismissed flows")
    x = np.r_[audits["position"], n_end] / scale
    ax.step(x, np.r_[audits["lower"], audits["lower"].iloc[-1]], where="post", color=color, linewidth=1.2,
            label="lower bound")
    ax.step(x, np.r_[audits["upper"], audits["upper"].iloc[-1]], where="post", color=color, linestyle="--",
            linewidth=1.2, label="upper bound")
    ax.plot([0, n_end / scale], [0, eps * n_end], color=style.GRAY, linestyle=":", linewidth=1,
            label=r"$\varepsilon$ times the dismissed flows")
    if int(first["violation_start"]) > 0:
        ax.axvline(int(first["violation_start"]) / scale, color=style.GRAY, linestyle="-.", linewidth=0.8)
    if int(first["alarm"]) >= 0:
        ax.axvline(int(first["alarm"]) / scale, color=color, linestyle="-.", linewidth=0.8)
    ax.set_xlim(0, n_end / scale)
    ax.set_ylim(0, 1.1 * max(float(audits["upper"].max()), float(counts["attacks"].max()), 1.0))
    ax.set_xlabel("millions of dismissed flows")
    ax.set_ylabel("attacks")
    layout.panel_letter(ax, "c")
    handles = dict(zip(*reversed(ax.get_legend_handles_labels())))
    marks = [(Line2D([], [], color=style.GRAY, linestyle=":", linewidth=1), "jump of the benign scores"),
             (Line2D([], [], color=style.GRAY, linestyle="-.", linewidth=0.8), "start of the violation"),
             (Line2D([], [], color=color, linestyle="-.", linewidth=0.8), "alarm")]
    layout.legend_below(fig, axes[0], [synthetic_online.LABELS[m] for m in methods],
                        [(h, label) for label, h in handles.items()] + marks)
    return fig


def e3_bounds_over_time(table: pd.DataFrame) -> plt.Figure:
    """one stream of the real order per dataset, offline FORGE at the typical composition under the one sequence
    over the whole stream, with the true count of attacks among the dismissed traffic, its two bounds, epsilon times the
    dismissed traffic, the first window of every period of the paper's restart rule as a dotted line and the alarm
    where the sequence fired."""
    style.use()
    datasets = [d for d in DATASET_LABELS if d in set(table["dataset"])]
    fig, axes = plt.subplots(1, max(len(datasets), 1), figsize=(style.WIDTH_DOUBLE, 2.7), squeeze=False)
    axes = axes[0]
    scale = 1e6
    color = style.color_for("forge_simplex")
    for ax, dataset, letter in zip(axes, datasets, "abc"):
        g = table[table["dataset"] == dataset]
        audits = g[g["point"] == "audit"].sort_values("position")
        windows = g[g["point"] == "window"].sort_values("position")
        first = g.iloc[0]
        n_end, eps = float(first["n_end"]), float(first["epsilon"])
        unit = DATASET_UNITS[dataset]
        # the count is known at the audited attacks and at the window ends, the bounds move at the audited attacks
        count = pd.concat([audits, windows]).sort_values("position", kind="stable")
        ax.plot(count["position"] / scale, count["attacks"], color=style.INK, linewidth=1,
                label="attacks among the dismissed traffic")
        x = np.r_[audits["position"], n_end] / scale
        ax.step(x, np.r_[audits["lower"], audits["lower"].iloc[-1]], where="post", color=color, linewidth=1.2,
                label="lower bound")
        ax.step(x, np.r_[audits["upper"], audits["upper"].iloc[-1]], where="post", color=color, linestyle="--",
                linewidth=1.2, label="upper bound")
        ax.plot([0, n_end / scale], [0, eps * n_end], color=style.GRAY, linestyle=":", linewidth=1,
                label=r"$\varepsilon$ times the dismissed traffic")
        period = windows["period"].to_numpy()
        for w in np.flatnonzero(np.r_[False, period[1:] != period[:-1]]):
            ax.axvline(windows["position"].iloc[w - 1] / scale, color=style.GRAY, linestyle=":", linewidth=0.6)
        if int(first["alarm"]) >= 0:
            ax.axvline(int(first["alarm"]) / scale, color=color, linestyle="-.", linewidth=0.8)
        ax.set_xlim(0, n_end / scale)
        ax.set_ylim(0, 1.1 * max(float(audits["upper"].max()), float(count["attacks"].max()), 1.0))
        ax.set_xlabel(f"millions of dismissed {unit}, {DATASET_SHORT[dataset]}")
        layout.panel_letter(ax, letter)
    axes[0].set_ylabel("attacks")
    extra = [(Line2D([], [], color=style.GRAY, linestyle=":", linewidth=0.6), "first window of a restart period")]
    if (table["alarm"] >= 0).any():
        extra.append((Line2D([], [], color=style.color_for("forge_simplex"), linestyle="-.", linewidth=0.8), "alarm"))
    layout.legend_below(fig, axes[0], ["attacks among the dismissed traffic", "lower bound", "upper bound"], extra,
                        ncol=2)
    return fig
