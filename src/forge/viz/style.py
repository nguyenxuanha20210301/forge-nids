"""one figure style for the whole project.

colors are assigned to methods by a fixed order, never by rank, so a method keeps its color in every figure.
"""

import matplotlib as mpl

# colorblind checked categorical order, light background
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GRAY = "#7a7a75"
# second channel for lines that would otherwise differ by color alone
DASHES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (1, 1))]

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


def color_for(name: str, order: list[str]) -> str:
    # order is the fixed list of method names for the experiment
    return COLORS[order.index(name) % len(COLORS)]
