"""one figure style for the whole project.

colors are assigned to methods by a fixed order, never by rank, so a method keeps its color in every figure.
"""

import matplotlib as mpl

# colorblind checked categorical order, light background
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GRAY = "#7a7a75"
# second channel for lines that would otherwise differ by color alone
DASHES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (1, 1))]
# one blue from light to dark for magnitudes such as shares, so a heatmap stays readable in gray print. the lightest
# step keeps a contrast above 2 against white, which keeps a small value apart from an empty cell
SEQUENTIAL = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

# every method of the project in one fixed order, which gives each method the same color in every experiment
METHOD_ORDER = ("forge_simplex", "forge_pooled", "ltt_iid", "forge_strict", "forge_online", "forge_previous")

# single column and double column widths in inches, close to most Elsevier and IEEE templates
WIDTH_SINGLE = 3.5
WIDTH_DOUBLE = 7.16

RC = {
    "figure.dpi": 100,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e4e4e0",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "legend.frameon": False,
    "axes.prop_cycle": mpl.cycler(color=COLORS),
}


def use() -> None:
    mpl.rcParams.update(RC)


def color_for(name: str) -> str:
    return COLORS[METHOD_ORDER.index(name) % len(COLORS)]


def percent(v, _pos=None) -> str:
    # tick label for a share, 0.001 as 0.1%, matplotlib takes the function itself as a tick formatter
    return f"{v * 100:g}%"
