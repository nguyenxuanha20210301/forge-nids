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
    fig, axes = plt.subplots(len(mixes), 3, figsize=(style.WIDTH_DOUBLE, 0.5 + 2.2 * len(mixes)), sharex=True,
                             squeeze=False)
    # one row per mix of attack families. left, the FOR of the certified draw and window pairs, median and the 5% to 95%
    # range. middle, the share of all pairs with FOR above epsilon, a refusal counting as no violation, with the range
    # of the per window shares. right, the mean share of the window sessions dismissed, a refusal counting as none,
    # with the two best thresholds of the same windows
    yardsticks = []
    for row, mix in zip(axes, mixes):
        for k, method in enumerate(methods):
            g = data[(data["method"] == method) & (data["mix"] == mix)].sort_values("prevalence")
            x = layout.dodge(g["prevalence"], k, len(methods), 0.035, log=True)
            layout.points(row[0], x, g["for_median"], g["for_low"], g["for_high"], **line(method))
            layout.points(row[1], x, g["violation_rate"], g["violation_window_min"], g["violation_window_max"],
                          **line(method))
            row[2].plot(x, g["dismissed_mean"], markersize=3.5, linewidth=1.2, **line(method))
        # the yardsticks follow from the windows alone, and runs from before them lack them
        g = data[(data["method"] == methods[0]) & (data["mix"] == mix)].sort_values("prevalence")
        drawn = [(name, dash, label) for name, dash, label in lines
                 if f"{name}_dismissed" in g and g[f"{name}_dismissed"].notna().any()]
        if drawn:
            yardsticks = [(row[2].plot(g["prevalence"], g[f"{name}_dismissed"], color=style.INK, linestyle=dash,
                                       linewidth=1, label=label)[0], label) for name, dash, label in drawn]
        row[0].axhline(summary["epsilon"], color=style.GRAY, linestyle="--", linewidth=1, label=r"$\varepsilon$")
        delta_line = row[1].axhline(summary["delta"], color=style.GRAY, linestyle="--", linewidth=1)
        row[0].set_ylabel(f"FOR in the {window}\n{mix_labels[mix]}")
        row[1].set_ylim(-0.02, 1.02)
        row[1].set_ylabel(r"share with FOR > $\varepsilon$")
        row[2].set_ylim(0, 1)
        row[2].yaxis.set_major_formatter(style.percent)
        row[2].set_ylabel(f"window {unit} dismissed")
    # a FOR of 0, where no attack of the window falls at or below the threshold, sits on the lower edge
    shown = data[["for_median", "for_low", "for_high"]].to_numpy().ravel()
    shown = shown[shown > 0]
    for r, row in enumerate(axes):
        for ax in row:
            prevalence_axis(ax, summary, r == len(axes) - 1)
        if len(shown):
            row[0].set_yscale("log")
            row[0].set_ylim(10.0 ** np.floor(np.log10(shown.min())),
                            10.0 ** np.ceil(np.log10(max(shown.max(), summary["epsilon"]))))
    for ax, letter in zip(axes.flat, "abcdefghi"):
        layout.panel_letter(ax, letter)
    layout.legend_below(fig, axes[0, 0], [line(m)["label"] for m in methods], [(delta_line, r"$\delta$"), *yardsticks])
    return fig
