"""placement helpers shared by the figures of several experiments."""

import numpy as np


def dodge(x, k: int, n: int, step: float, log: bool = False) -> np.ndarray:
    # shifts the points of the k-th of n methods sideways, so markers and intervals do not hide each other. on a log
    # axis the shift is a factor, otherwise an offset
    shift = (k - (n - 1) / 2) * step
    x = np.asarray(x, dtype=float)
    return x * 10.0 ** shift if log else x + shift


def points(ax, x, rate, low, high, **line) -> None:
    # shares with their intervals as error bars, color, marker and label come in line
    rate, low, high = (np.asarray(v, dtype=float) for v in (rate, low, high))
    ax.errorbar(x, rate, yerr=[np.clip(rate - low, 0, None), np.clip(high - rate, 0, None)], markersize=3.5,
                linewidth=1.2, elinewidth=0.8, capsize=0, **line)


def panel_letter(ax, letter: str) -> None:
    # above the top left corner of the plot, clear of the data
    ax.text(0, 1.03, f"({letter})", transform=ax.transAxes, ha="left", va="bottom")


def legend_below(fig, ax, first: list[str], extra=(), ncol: int = 3) -> None:
    # the entries of ax labelled in first come first and in that order, then its other entries, then the extra
    # handles. matplotlib fills legend columns first, so the entries are rearranged to read row by row
    found = dict(zip(*reversed(ax.get_legend_handles_labels())))
    entries = [(found[label], label) for label in first]
    entries += [(h, label) for label, h in found.items() if label not in set(first)] + list(extra)
    rows = -(-len(entries) // ncol)
    order = [r * ncol + c for c in range(ncol) for r in range(rows) if r * ncol + c < len(entries)]
    fig.tight_layout()
    fig.legend(*zip(*[entries[i] for i in order]), loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 0.0))
