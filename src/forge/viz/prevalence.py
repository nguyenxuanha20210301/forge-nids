"""the figure over prevalences shared by the experiments of RQ1, one row per mix of attack families, with the FOR of the
certified draws, the share of draws with FOR above epsilon and the share of the window flows dismissed against the
prevalence of the window."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from forge.viz import layout, style

# the yardsticks in the panels of dismissed shares, in the color of no method, without markers, and in line styles
# that the gray lines of epsilon, delta, the calibration prevalence and pi_max leave free
BEST_LINES = [("best_up_to_pi_max", "-", r"best threshold up to $\pi_{\max}$"),
              ("best_at_prevalence", (0, (1, 1)), r"best threshold at the window's $\pi$")]


def prevalence_axis(ax, summary: dict, bottom: bool) -> None:
    ax.axvline(summary["calibration_prevalence"], color=style.GRAY, linestyle=":", linewidth=1,
               label=r"$\pi$ of the calibration set")
    ax.axvline(summary["pi_max"], color=style.GRAY, linestyle="-.", linewidth=1, label=r"$\pi_{\max}$")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(style.percent)
    ax.xaxis.set_minor_formatter(NullFormatter())
    if bottom:
        ax.set_xlabel("attack prevalence in the window")


def for_by_prevalence(data, summary: dict, methods: list[str], line, mixes: list[str], mix_labels: dict,
                      window: str, unit: str = "sessions", lines=BEST_LINES) -> plt.Figure:
    """data holds the pooled rows of one set of windows, line gives the color, marker and label of a method, window
    names the windows in the axis labels and unit what they hold. lines are the yardsticks of the dismissed panels, a
    column of data each, drawn where data holds a value of it."""
    rows = [{"data": data[data["mix"] == mix], "summary": summary, "label": mix_labels[mix], "window": window,
             "unit": unit} for mix in mixes]
    return for_by_prevalence_rows(rows, methods, line, lines)


def for_by_prevalence_rows(rows: list[dict], methods: list[str], line, lines=BEST_LINES,
                           floor: float | None = None) -> plt.Figure:
    """one row of panels per entry of rows, each a dict with the rows of one set of windows as data, the settings
    whose epsilon, delta, calibration prevalence and pi_max the lines mark as summary, the label under the FOR axis,
    and the window and unit words of the axis labels. a row with dismissed_low and dismissed_high in its data gets
    the range of the dismissed share as well. floor is the lowest FOR shown, else the lowest positive value drawn."""
    fig, axes = plt.subplots(len(rows), 3, figsize=(style.WIDTH_DOUBLE, 0.5 + 2.2 * len(rows)), sharex=True,
                             squeeze=False)
    # one row per set of windows. left, the FOR of the certified draw and window pairs, median and the 5% to 95%
    # range. middle, the share of all pairs with FOR above epsilon, a refusal counting as no violation, with the range
    # of the per window shares. right, the mean share of the window sessions dismissed, a refusal counting as none,
    # with the two best thresholds of the same windows
    yardsticks = []
    delta_line = None
    for row, spec in zip(axes, rows):
        data, summary = spec["data"], spec["summary"]
        present = [m for m in methods if m in set(data["method"])]
        for k, method in enumerate(present):
            g = data[data["method"] == method].sort_values("prevalence")
            x = layout.dodge(g["prevalence"], k, len(present), 0.035, log=True)
            layout.points(row[0], x, g["for_median"], g["for_low"], g["for_high"], **line(method))
            layout.points(row[1], x, g["violation_rate"], g["violation_window_min"], g["violation_window_max"],
                          **line(method))
            if "dismissed_low" in g and "dismissed_high" in g:
                layout.points(row[2], x, g["dismissed_mean"], g["dismissed_low"], g["dismissed_high"],
                              **line(method))
            else:
                row[2].plot(x, g["dismissed_mean"], markersize=3.5, linewidth=1.2, **line(method))
        # the yardsticks follow from the windows alone, and runs from before them lack them
        g = data[data["method"] == present[0]].sort_values("prevalence") if present else data.iloc[:0]
        drawn = [(name, dash, label) for name, dash, label in lines
                 if f"{name}_dismissed" in g and g[f"{name}_dismissed"].notna().any()]
        if drawn:
            yardsticks = [(row[2].plot(g["prevalence"], g[f"{name}_dismissed"], color=style.INK, linestyle=dash,
                                       linewidth=1, label=label)[0], label) for name, dash, label in drawn]
        row[0].axhline(summary["epsilon"], color=style.GRAY, linestyle="--", linewidth=1, label=r"$\varepsilon$")
        delta_line = row[1].axhline(summary["delta"], color=style.GRAY, linestyle="--", linewidth=1)
        row[0].set_ylabel(f"FOR in the {spec['window']}\n{spec['label']}")
        row[1].set_ylim(-0.02, 1.02)
        row[1].set_ylabel(r"share with FOR > $\varepsilon$")
        row[2].set_ylim(0, 1)
        row[2].yaxis.set_major_formatter(style.percent)
        row[2].set_ylabel(f"window {spec['unit']} dismissed")
    # a FOR of 0, where no attack of the window falls at or below the threshold, sits on the lower edge
    shown = np.concatenate([spec["data"][["for_median", "for_low", "for_high"]].to_numpy().ravel()
                            for spec in rows])
    shown = shown[np.isfinite(shown) & (shown > 0)]
    for r, (row, spec) in enumerate(zip(axes, rows)):
        for ax in row:
            prevalence_axis(ax, spec["summary"], r == len(axes) - 1)
        if len(shown):
            row[0].set_yscale("log")
            row[0].set_ylim(10.0 ** np.floor(np.log10(shown.min())) if floor is None else floor,
                            10.0 ** np.ceil(np.log10(max(shown.max(), spec["summary"]["epsilon"]))))
    for ax, letter in zip(axes.flat, "abcdefghi"):
        layout.panel_letter(ax, letter)
    # the legend reads the row that holds the most methods, since a dataset may lack one
    first = max(axes[:, 0], key=lambda ax: len(ax.get_legend_handles_labels()[1]))
    labels = [line(m)["label"] for m in methods if line(m)["label"] in first.get_legend_handles_labels()[1]]
    layout.legend_below(fig, first, labels, [(delta_line, r"$\delta$"), *yardsticks])
    return fig


def for_by_prevalence_compact(rows: list[dict], methods: list[str], line, lines=BEST_LINES) -> plt.Figure:
    """the figure of for_by_prevalence_rows without its FOR panels, so that it takes half a page. the share of pairs
    with FOR above epsilon sits above and the share of the window dismissed below, one column per entry of rows,
    whose label names the column under its prevalence axis."""
    fig, axes = plt.subplots(2, len(rows), figsize=(style.WIDTH_DOUBLE, 4.3), sharex="col", squeeze=False)
    yardsticks = []
    delta_line = None
    for c, spec in enumerate(rows):
        top, bottom = axes[0, c], axes[1, c]
        data, summary = spec["data"], spec["summary"]
        present = [m for m in methods if m in set(data["method"])]
        for k, method in enumerate(present):
            g = data[data["method"] == method].sort_values("prevalence")
            x = layout.dodge(g["prevalence"], k, len(present), 0.035, log=True)
            layout.points(top, x, g["violation_rate"], g["violation_window_min"], g["violation_window_max"],
                          **line(method))
            if "dismissed_low" in g and "dismissed_high" in g:
                layout.points(bottom, x, g["dismissed_mean"], g["dismissed_low"], g["dismissed_high"],
                              **line(method))
            else:
                bottom.plot(x, g["dismissed_mean"], markersize=3.5, linewidth=1.2, **line(method))
        g = data[data["method"] == present[0]].sort_values("prevalence") if present else data.iloc[:0]
        drawn = [(name, dash, label) for name, dash, label in lines
                 if f"{name}_dismissed" in g and g[f"{name}_dismissed"].notna().any()]
        if drawn:
            yardsticks = [(bottom.plot(g["prevalence"], g[f"{name}_dismissed"], color=style.INK, linestyle=dash,
                                       linewidth=1, label=label)[0], label) for name, dash, label in drawn]
        delta_line = top.axhline(summary["delta"], color=style.GRAY, linestyle="--", linewidth=1)
        top.set_ylim(-0.02, 1.02)
        bottom.set_ylim(0, 1)
        bottom.yaxis.set_major_formatter(style.percent)
        prevalence_axis(top, summary, False)
        prevalence_axis(bottom, summary, True)
        bottom.set_xlabel(f"attack prevalence in the window\n{spec['label']}")
    axes[0, 0].set_ylabel(r"share with FOR > $\varepsilon$")
    axes[1, 0].set_ylabel("share of the window dismissed")
    for ax, letter in zip(axes.flat, "abcdefghi"):
        layout.panel_letter(ax, letter)
    first = max(axes[0], key=lambda ax: len(ax.get_legend_handles_labels()[1]))
    labels = [line(m)["label"] for m in methods if line(m)["label"] in first.get_legend_handles_labels()[1]]
    layout.legend_below(fig, first, labels, [(delta_line, r"$\delta$"), *yardsticks])
    return fig
