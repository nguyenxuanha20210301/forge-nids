"""flow count, attack prevalence and attack family mix of UGR'16 per window of fixed length and per test week.

the processed table holds every k-th flow of the six test weeks in time order, so a window holds the daily cycle of
the ISP at a coarser grain. the attacks the authors launched sit in the first weeks, the anomalies they found in the
real traffic come and go, and the blacklist marks addresses rather than traffic, so the three are counted apart. the
window length and the time column come from the dataset config.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from forge import runs
from forge.data import splits, ugr16
from forge.viz import layout, style
from forge.viz.export import save_figure

# the labelled flows in three groups, the attacks the authors launched, the anomalies they found in the real traffic
# with every other label, and the flows of blacklisted addresses, whose label marks the address and not the traffic
GROUPS = ["launched", "found", "blacklist"]
GROUP_LABELS = {"launched": "attacks launched by the authors", "found": "anomalies found in the traffic",
                "blacklist": "blacklisted addresses"}
GROUP_MARKERS = {"launched": "o", "found": "s", "blacklist": "^"}
BENIGN = ugr16.FAMILY_ORDER[0]


def group_of(family: str) -> str:
    if family in ugr16.INJECTED_FAMILIES:
        return "launched"
    return "blacklist" if family == "blacklist" else "found"


def _codes(column: pd.Series) -> np.ndarray:
    # pandas keeps category codes in int8 below 128 categories, so arithmetic on them wraps around without a widening
    return column.cat.codes.to_numpy().astype(np.int64)


def _group_counts(flows: pd.DataFrame, ids: np.ndarray, n: int) -> np.ndarray:
    # flows of each group per window, benign flows in column 0 and the groups after it in their order
    families = list(flows["family"].cat.categories)
    group = np.array([0 if f == BENIGN else GROUPS.index(group_of(f)) + 1 for f in families])
    k = len(GROUPS) + 1
    return np.bincount(ids * k + group[_codes(flows["family"])], minlength=n * k).reshape(n, k)


def _family_counts(flows: pd.DataFrame, ids: np.ndarray, n: int) -> tuple[np.ndarray, list[str]]:
    # attack flows of each family per window, the families in the category order of the table without benign
    attack = flows["label"].to_numpy() == 1
    families = list(flows["family"].cat.categories)
    k = len(families)
    counts = np.bincount(ids[attack] * k + _codes(flows["family"])[attack], minlength=n * k).reshape(n, k)
    keep = [i for i, f in enumerate(families) if f != BENIGN]
    return counts[:, keep], [families[i] for i in keep]


def _present(counts: np.ndarray, families: list[str]) -> list[str]:
    # the families of each row with their counts, in the category order, as one string per row
    return [", ".join(f"{f} {c}" for f, c in zip(families, row) if c > 0) for row in counts]


def window_table(flows: pd.DataFrame, t: str, length: str) -> tuple[pd.DataFrame, np.ndarray]:
    ids = splits.fixed_windows(flows[t], length)
    table = splits.fixed_window_table(flows[t], length)
    n = len(table)
    groups = _group_counts(flows, ids, n)
    table["n_flows"] = groups.sum(axis=1)
    table["n_attacks"] = groups[:, 1:].sum(axis=1)
    for j, name in enumerate(GROUPS, start=1):
        table[f"n_{name}"] = groups[:, j]
    n_flows = table["n_flows"].to_numpy()
    table["prevalence"] = np.divide(table["n_attacks"], n_flows, out=np.full(n, np.nan), where=n_flows > 0)
    # the prevalence an experiment that leaves the blacklisted addresses out of every class sees
    rest = n_flows - table["n_blacklist"].to_numpy()
    table["prevalence_without_blacklist"] = np.divide(table["n_attacks"] - table["n_blacklist"], rest,
                                                      out=np.full(n, np.nan), where=rest > 0)
    counts, families = _family_counts(flows, ids, n)
    table["families"] = _present(counts, families)
    # rows are in time order and the weeks interleave only at a boundary, so the first row of each window and week
    # pair gives the weeks in the order they appear
    codes = _codes(flows["week"])
    first = np.r_[True, (ids[1:] != ids[:-1]) | (codes[1:] != codes[:-1])]
    names = {}
    for w, c in zip(ids[first], codes[first]):
        names.setdefault(w, []).append(str(flows["week"].cat.categories[c]))
    table["weeks"] = ["+".join(dict.fromkeys(names.get(w, []))) for w in table["window"]]
    return table, ids


def family_mix(flows: pd.DataFrame, ids: np.ndarray, windows: pd.DataFrame) -> pd.DataFrame:
    # attack flows of each family in each window and their share of the attacks in that window, every window and
    # family listed, with an empty share where a window has no attacks
    counts, families = _family_counts(flows, ids, len(windows))
    total = counts.sum(axis=1, keepdims=True)
    share = np.divide(counts, total, out=np.full(counts.shape, np.nan), where=total > 0)
    return pd.DataFrame({
        "window": np.repeat(windows["window"].to_numpy(), len(families)),
        "start": np.repeat(windows["start"].to_numpy(), len(families)),
        "family": np.tile(families, len(windows)),
        "n_attacks": counts.ravel(),
        "share": share.ravel(),
    })


def week_table(flows: pd.DataFrame, t: str) -> pd.DataFrame:
    weeks = list(flows["week"].cat.categories)
    ids = _codes(flows["week"])
    g = flows.groupby("week", observed=False)
    table = pd.DataFrame({"week": weeks, "start": g[t].min().to_numpy(), "end": g[t].max().to_numpy()})
    groups = _group_counts(flows, ids, len(weeks))
    table["n_flows"] = groups.sum(axis=1)
    table["n_attacks"] = groups[:, 1:].sum(axis=1)
    for j, name in enumerate(GROUPS, start=1):
        table[f"n_{name}"] = groups[:, j]
    table["prevalence"] = table["n_attacks"] / table["n_flows"]
    counts, families = _family_counts(flows, ids, len(weeks))
    table["families"] = _present(counts, families)
    return table[table["n_flows"] > 0].sort_values("start", ignore_index=True)


def run(cfg: dict, run: runs.Run) -> None:
    d = cfg["dataset"]
    t, length = d["time_column"], d["window_freq"]
    manifest = ugr16.read_manifest(cfg)
    flows = ugr16.load(cfg, columns=[t, "week", "label", "family"])
    run.log(f"loaded {len(flows)} flows of {d['name']} {d['processed_version']}, every {manifest['sample_every']}th "
            f"of {manifest['rows_in_weeks']}, fingerprint {manifest['fingerprint']}")

    windows, ids = window_table(flows, t, length)
    run.save_table(windows, "windows")
    run.save_table(family_mix(flows, ids, windows), "family_mix")
    weeks = week_table(flows, t)
    run.save_table(weeks, "weeks")
    for w in weeks.itertuples():
        run.log(f"week {w.week}, from {w.start} to {w.end}, flows {w.n_flows}, attacks {w.n_attacks}, prevalence "
                f"{w.prevalence:.6f}, launched {w.n_launched}, found {w.n_found}, blacklist {w.n_blacklist}, "
                f"families {w.families}")
    for w in windows.itertuples():
        if w.n_flows == 0:
            run.log(f"window {w.start:%Y-%m-%d %H:%M}, no flows")
            continue
        run.log(f"window {w.start:%Y-%m-%d %H:%M}, flows {w.n_flows}, attacks {w.n_attacks}, prevalence "
                f"{w.prevalence:.6f}, launched {w.n_launched}, found {w.n_found}, blacklist {w.n_blacklist}, "
                f"weeks {w.weeks}, families {w.families}")
    filled = windows[windows["n_flows"] > 0]
    run.log(f"{len(filled)} of {len(windows)} windows of {length} hold flows")
    families = windows[[c for c in windows.columns if c.startswith("n_")]].sum()
    run.save_json({
        "dataset": d["name"],
        "processed_version": d["processed_version"],
        "fingerprint": manifest["fingerprint"],
        "processed_by_commit": manifest["code"]["commit"],
        "sample_every": manifest["sample_every"],
        "rows_in_weeks": manifest["rows_in_weeks"],
        "time_column": t,
        "window_length": length,
        "flows": int(len(flows)),
        "attacks": int(families["n_attacks"]),
        "groups": {name: int(families[f"n_{name}"]) for name in GROUPS},
        "windows": int(len(windows)),
        "windows_with_flows": int(len(filled)),
        "first": str(flows[t].min()),
        "last": str(flows[t].max()),
        "weeks": weeks["week"].tolist(),
    }, "summary")

    for name, draw in FIGURES.items():
        save_figure(draw(run.dir), run.figures, name)
        run.log(f"saved figure {name}")


def _time_axis(ax, windows: pd.DataFrame, summary: dict) -> None:
    pad = windows["end"].iloc[0] - windows["start"].iloc[0]
    ax.set_xlim(windows["start"].min() - pad, windows["end"].max() + pad)
    # a labelled tick on every monday, the first day of a test week, and a bare one on every other day
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_xlabel(f"start of the window of {summary['window_length']}, {windows['start'].iloc[0]:%Y}")


def fig_prevalence_by_window(run_dir) -> plt.Figure:
    style.use()
    windows = runs.load_table(run_dir, "windows")
    weeks = runs.load_table(run_dir, "weeks")
    summary = runs.load_json(run_dir, "summary")
    filled = windows[windows["n_flows"] > 0]
    fig, axes = plt.subplots(2, 1, figsize=(style.WIDTH_DOUBLE, 3.8), sharex=True)
    # a, the flows of each window, the daily cycle of the ISP at the grain of the sample
    axes[0].plot(filled["start"], filled["n_flows"], color=style.INK, marker="o", markersize=2, linewidth=1)
    axes[0].set_ylim(0, None)
    axes[0].set_ylabel("flows in the window")
    # b, the share of each group among the flows of the window on a log axis, the line broken where a window holds
    # none of the group
    shares = {name: filled[f"n_{name}"].to_numpy() / filled["n_flows"].to_numpy() for name in GROUPS}
    positive = np.concatenate([s[s > 0] for s in shares.values()])
    low = 10.0 ** np.floor(np.log10(positive.min())) if len(positive) else 1e-6
    for k, name in enumerate(GROUPS):
        s = shares[name]
        axes[1].plot(filled["start"], np.where(s > 0, s, np.nan), color=style.COLORS[k], linestyle=style.DASHES[k],
                     marker=GROUP_MARKERS[name], markersize=2, linewidth=1, label=GROUP_LABELS[name])
    axes[1].set_yscale("log")
    axes[1].set_ylim(low, 1)
    axes[1].yaxis.set_major_formatter(style.percent)
    axes[1].yaxis.set_minor_formatter(NullFormatter())
    axes[1].set_ylabel("share of the flows of the window")
    for ax, letter in zip(axes, "ab"):
        for start in weeks["start"].iloc[1:]:
            ax.axvline(start, color=style.GRAY, linestyle="--", linewidth=1)
        layout.panel_letter(ax, letter)
    _time_axis(axes[1], windows, summary)
    extra = [(Line2D([], [], color=style.GRAY, linestyle="--", linewidth=1), "first flow of a test week")]
    layout.legend_below(fig, axes[1], [GROUP_LABELS[name] for name in GROUPS], extra, ncol=2)
    return fig


def fig_family_mix_by_window(run_dir) -> plt.Figure:
    style.use()
    mix = runs.load_table(run_dir, "family_mix")
    summary = runs.load_json(run_dir, "summary")
    mix = mix[mix.groupby("window")["n_attacks"].transform("sum") > 0]
    families = list(dict.fromkeys(mix["family"]))
    grid = mix.pivot(index="family", columns="window", values="share").reindex(families)
    starts = mix.drop_duplicates("window").set_index("window")["start"].reindex(grid.columns)
    values = grid.to_numpy()
    fig, ax = plt.subplots(figsize=(style.WIDTH_DOUBLE, 2.9))
    # one blue for the share, a family absent from a window leaves its cell empty, white lines keep cells apart and
    # thin out where the windows are many
    cmap = LinearSegmentedColormap.from_list("share", style.SEQUENTIAL)
    mesh = ax.pcolormesh(np.ma.masked_where(~(values > 0), values), cmap=cmap, vmin=0, vmax=1, edgecolors="white",
                         linewidth=1.0 if len(starts) <= 40 else 0.3)
    ax.set_yticks(np.arange(len(grid.index)) + 0.5, [ugr16.FAMILY_LABELS.get(f, f) for f in grid.index])
    ax.invert_yaxis()
    step = max(1, len(starts) // 12)
    ax.set_xticks(np.arange(0, len(starts), step) + 0.5, [f"{s:%b %d %H:%M}" for s in starts.iloc[::step]],
                  rotation=90)
    ax.set_xlabel(f"start of the window of {summary['window_length']}, windows with attacks")
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.03)
    bar.set_label("share of the attacks in the window")
    bar.formatter = style.percent
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0)
    return fig


FIGURES = {
    "prevalence_by_window": fig_prevalence_by_window,
    "family_mix_by_window": fig_family_mix_by_window,
}
