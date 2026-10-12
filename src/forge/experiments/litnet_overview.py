"""flow count, attack prevalence and attack family mix of LITNET-2020 per calendar window and per capture.

the published dataset is eleven captures of a few minutes each, so most calendar windows hold no flows and a window
with flows holds one to three captures. empty windows stay in the tables so the gaps remain visible. the window length
and the time column come from the dataset config.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, NullFormatter

from forge import runs
from forge.data import litnet2020, splits
from forge.viz import style
from forge.viz.export import save_figure

WINDOW_NAMES = {"D": "day", "W": "week", "M": "month"}
PERCENT = FuncFormatter(lambda v, _: f"{v * 100:g}%")


def _codes(column: pd.Series) -> np.ndarray:
    # pandas keeps category codes in int8 below 128 categories, so arithmetic on them wraps around without a widening
    return column.cat.codes.to_numpy().astype(np.int64)


def window_table(flows: pd.DataFrame, t: str, freq: str) -> tuple[pd.DataFrame, np.ndarray]:
    ids = splits.assign_windows(flows[t], freq)
    table = splits.calendar_windows(flows[t], freq)
    n_flows = np.bincount(ids, minlength=len(table))
    n_attacks = np.bincount(ids, weights=flows["label"].to_numpy(), minlength=len(table)).astype(np.int64)
    table["n_flows"] = n_flows
    table["n_attacks"] = n_attacks
    table["prevalence"] = np.divide(n_attacks, n_flows, out=np.full(len(table), np.nan), where=n_flows > 0)
    # rows are in time order, so the first row of each window and capture pair gives the captures in time order
    codes = flows["capture"].cat.codes.to_numpy()
    first = np.r_[True, (ids[1:] != ids[:-1]) | (codes[1:] != codes[:-1])]
    names = {}
    for w, c in zip(ids[first], codes[first]):
        names.setdefault(w, []).append(str(flows["capture"].cat.categories[c]))
    table["captures"] = ["+".join(dict.fromkeys(names.get(w, []))) for w in table["window"]]
    table["n_captures"] = [len(set(names.get(w, []))) for w in table["window"]]
    return table, ids


def family_mix(flows: pd.DataFrame, ids: np.ndarray, windows: pd.DataFrame) -> pd.DataFrame:
    # attack flows of each family in each window and their share of the attacks in that window, every window and
    # family listed, with an empty share where a window has no attacks
    attack = flows["label"].to_numpy() == 1
    codes = _codes(flows["family"])[attack]
    k = len(litnet2020.FAMILY_ORDER)
    counts = np.bincount(ids[attack] * k + codes, minlength=len(windows) * k).reshape(len(windows), k)[:, 1:]
    total = counts.sum(axis=1, keepdims=True)
    share = np.divide(counts, total, out=np.full(counts.shape, np.nan), where=total > 0)
    families = litnet2020.ATTACK_FAMILIES
    return pd.DataFrame({
        "window": np.repeat(windows["window"].to_numpy(), len(families)),
        "start": np.repeat(windows["start"].to_numpy(), len(families)),
        "family": np.tile(families, len(windows)),
        "n_attacks": counts.ravel(),
        "share": share.ravel(),
    })


def capture_table(flows: pd.DataFrame, t: str) -> pd.DataFrame:
    g = flows.groupby("capture", observed=True)
    table = pd.DataFrame({"start": g[t].min(), "end": g[t].max(), "n_flows": g.size(), "n_attacks": g["label"].sum()})
    table["prevalence"] = table["n_attacks"] / table["n_flows"]
    attack = flows["label"].to_numpy() == 1
    names = flows["capture"].cat.categories
    k = len(litnet2020.FAMILY_ORDER)
    pairs = _codes(flows["capture"])[attack] * k + _codes(flows["family"])[attack]
    counts = np.bincount(pairs, minlength=len(names) * k).reshape(len(names), k)
    present = {str(c): "+".join(f for f, n in zip(litnet2020.FAMILY_ORDER, row) if n > 0) for c, row in zip(names, counts)}
    table = table.rename_axis("capture").reset_index()
    table["capture"] = table["capture"].astype(str)
    table["families"] = table["capture"].map(present)
    return table.sort_values("start", ignore_index=True)


def run(cfg: dict, run: runs.Run) -> None:
    d = cfg["dataset"]
    t, freq = d["time_column"], d["window_freq"]
    manifest = litnet2020.read_manifest(cfg)
    flows = litnet2020.load(cfg, columns=[t, "label", "family", "capture"])
    run.log(f"loaded {len(flows)} flows of {d['name']} {d['processed_version']}, fingerprint {manifest['fingerprint']}")

    windows, ids = window_table(flows, t, freq)
    run.save_table(windows, "windows")
    run.save_table(family_mix(flows, ids, windows), "family_mix")
    captures = capture_table(flows, t)
    run.save_table(captures, "captures")
    filled = windows[windows["n_flows"] > 0]
    for w in filled.itertuples():
        run.log(f"window {w.start:%Y-%m-%d}, flows {w.n_flows}, attacks {w.n_attacks}, "
                f"prevalence {w.prevalence:.6f}, captures {w.captures}")
    run.log(f"{len(filled)} of {len(windows)} windows hold flows")
    run.save_json({
        "dataset": d["name"],
        "processed_version": d["processed_version"],
        "fingerprint": manifest["fingerprint"],
        "processed_by_commit": manifest["code"]["commit"],
        "source_commit": manifest["source"]["commit"],
        "time_column": t,
        "window_freq": freq,
        "flows": int(len(flows)),
        "attacks": int(windows["n_attacks"].sum()),
        "windows": int(len(windows)),
        "windows_with_flows": int(len(filled)),
        "captures": int(len(captures)),
    }, "summary")

    for name, draw in FIGURES.items():
        save_figure(draw(run.dir), run.figures, name)
        run.log(f"saved figure {name}")


def fig_prevalence_by_window(run_dir) -> plt.Figure:
    style.use()
    windows = runs.load_table(run_dir, "windows")
    freq = runs.load_json(run_dir, "summary")["window_freq"]
    filled = windows[windows["n_flows"] > 0]
    positive = filled[filled["n_attacks"] > 0]
    fig, ax = plt.subplots(figsize=(style.WIDTH_SINGLE, 2.4))
    # windows are not continuous, so markers stand alone without a connecting line
    ax.plot(positive["start"], positive["prevalence"], linestyle="none", marker="o", markersize=5,
            color=style.COLORS[0], markeredgecolor="white", markeredgewidth=0.6, clip_on=False)
    ax.set_yscale("log")
    # whole decades up to 100 %, so every axis carries labelled major ticks
    low = 10.0 ** np.floor(np.log10(positive["prevalence"].min())) if len(positive) else 1e-4
    ax.set_ylim(low, 1)
    zero = filled[filled["n_attacks"] == 0]
    if len(zero):
        # zero has no place on a log axis, so windows with flows but no attacks sit on the lower edge
        ax.plot(zero["start"], np.full(len(zero), ax.get_ylim()[0]), linestyle="none", marker="o", markersize=5,
                markerfacecolor="white", markeredgecolor=style.GRAY, clip_on=False, label="no attacks")
        ax.legend()
    pad = windows["end"].iloc[0] - windows["start"].iloc[0]
    ax.set_xlim(windows["start"].min() - pad, windows["end"].max() + pad)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.yaxis.set_major_formatter(PERCENT)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(f"{WINDOW_NAMES.get(freq, 'window')} start")
    ax.set_ylabel("attack prevalence")
    return fig


def fig_family_mix_by_window(run_dir) -> plt.Figure:
    style.use()
    mix = runs.load_table(run_dir, "family_mix")
    freq = runs.load_json(run_dir, "summary")["window_freq"]
    mix = mix[mix.groupby("window")["n_attacks"].transform("sum") > 0]
    grid = mix.pivot(index="family", columns="window", values="share").reindex(litnet2020.ATTACK_FAMILIES)
    starts = mix.drop_duplicates("window").set_index("window")["start"].reindex(grid.columns)
    values = grid.to_numpy()
    fig, ax = plt.subplots(figsize=(style.WIDTH_SINGLE, 2.9))
    # one blue for the share, a family absent from a window leaves its cell empty, white lines keep cells apart
    cmap = LinearSegmentedColormap.from_list("share", style.SEQUENTIAL)
    mesh = ax.pcolormesh(np.ma.masked_where(~(values > 0), values), cmap=cmap, vmin=0, vmax=1, edgecolors="white",
                         linewidth=1.0)
    ax.set_yticks(np.arange(len(grid.index)) + 0.5, [litnet2020.FAMILY_LABELS[f] for f in grid.index])
    ax.invert_yaxis()
    step = max(1, len(starts) // 12)
    ax.set_xticks(np.arange(0, len(starts), step) + 0.5, [f"{s:%Y-%m-%d}" for s in starts.iloc[::step]], rotation=90)
    ax.set_xlabel(f"{WINDOW_NAMES.get(freq, 'window')} start")
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.03)
    bar.set_label("share of attacks in the window")
    bar.formatter = PERCENT
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0)
    return fig


FIGURES = {
    "prevalence_by_window": fig_prevalence_by_window,
    "family_mix_by_window": fig_family_mix_by_window,
}
