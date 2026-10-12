"""session count, attack share and service mix of AnoShift per year and per block of equal size within a year.

the published files carry no clock time, so each year is split into blocks of equal session count in the original row
order, blocks_per_year of them, from the dataset config. a block is a stretch of contiguous sessions and not a month,
and the years differ in size, 2006 spanning only November and December. the unknown attack type is shown apart, it is
the nearest the data comes to a second attack family. the periods are those of the AnoShift protocol.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from forge import runs
from forge.data import anoshift, splits
from forge.viz import layout, style
from forge.viz.export import save_figure

# services below this share of a class in every year are summed into one row of the service figure
SERVICE_FLOOR = 0.01
REST = "all other services"


def year_table(flows: pd.DataFrame) -> pd.DataFrame:
    counts = pd.crosstab(flows["year"], flows["attack_type"]).reindex(columns=anoshift.ATTACK_TYPES, fill_value=0)
    table = pd.DataFrame({
        "year": counts.index.to_numpy(),
        "period": [anoshift.period(int(y)) for y in counts.index],
        "n_flows": counts.sum(axis=1).to_numpy(),
        "n_attacks": (counts["known"] + counts["unknown"]).to_numpy(),
        "n_known": counts["known"].to_numpy(),
        "n_unknown": counts["unknown"].to_numpy(),
        "n_normal": counts["benign"].to_numpy(),
    })
    table["attack_share"] = table["n_attacks"] / table["n_flows"]
    table["unknown_share"] = table["n_unknown"] / table["n_flows"]
    return table


def block_table(flows: pd.DataFrame, per_year: int) -> pd.DataFrame:
    years = flows["year"].to_numpy().astype(np.int64)
    first = int(years.min())
    key = (years - first) * per_year + splits.order_blocks(years, per_year)
    size = (int(years.max()) - first + 1) * per_year
    n = np.bincount(key, minlength=size)
    attacks = np.bincount(key, weights=flows["label"].to_numpy(), minlength=size)
    unknown = np.bincount(key, weights=(flows["attack_type"] == "unknown").to_numpy(), minlength=size)
    used = np.flatnonzero(n > 0)
    table = pd.DataFrame({
        "year": first + used // per_year,
        "block": used % per_year,
        "n_flows": n[used],
        "n_attacks": attacks[used].astype(np.int64),
        "n_unknown": unknown[used].astype(np.int64),
    })
    table["attack_share"] = table["n_attacks"] / table["n_flows"]
    table["unknown_share"] = table["n_unknown"] / table["n_flows"]
    # blocks evenly spaced inside their year on the axis of the figure, which is therefore not a time axis
    table["position"] = table["year"] + (table["block"] + 0.5) / per_year
    return table


def service_table(flows: pd.DataFrame) -> pd.DataFrame:
    # sessions of each service per year and class, and their share of the sessions of that year and class
    counts = flows.groupby(["year", "label", "service"], observed=True).size().rename("n_flows").reset_index()
    counts["class"] = np.where(counts["label"] == 1, "attack", "normal")
    counts["share"] = counts["n_flows"] / counts.groupby(["year", "label"])["n_flows"].transform("sum")
    counts["service"] = counts["service"].astype(str)
    return counts[["year", "class", "service", "n_flows", "share"]]


def run(cfg: dict, run: runs.Run) -> None:
    d = cfg["dataset"]
    per_year = d["blocks_per_year"]
    manifest = anoshift.read_manifest(cfg)
    flows = anoshift.load(cfg, columns=["year", "label", "attack_type", "service"])
    run.log(f"loaded {len(flows)} sessions of {d['name']} {d['processed_version']}, fingerprint "
            f"{manifest['fingerprint']}")

    years = year_table(flows)
    blocks = block_table(flows, per_year)
    services = service_table(flows)
    run.save_table(years, "years")
    run.save_table(blocks, "blocks")
    run.save_table(services, "services")
    for y in years.itertuples():
        b = blocks[blocks["year"] == y.year]
        run.log(f"year {y.year}, period {y.period}, sessions {y.n_flows}, attack share {y.attack_share:.4f}, unknown "
                f"share {y.unknown_share:.5f}, attack share of its blocks from {b['attack_share'].min():.4f} to "
                f"{b['attack_share'].max():.4f}")
    run.save_json({
        "dataset": d["name"],
        "processed_version": d["processed_version"],
        "fingerprint": manifest["fingerprint"],
        "processed_by_commit": manifest["code"]["commit"],
        "source_tar_sha256": manifest["source"]["tar"]["sha256"],
        "blocks_per_year": per_year,
        "sessions": int(len(flows)),
        "attacks": int(years["n_attacks"].sum()),
        "years": [int(y) for y in years["year"]],
        "periods": {name: list(span) for name, span in anoshift.PERIODS.items()},
    }, "summary")

    for name, draw in FIGURES.items():
        save_figure(draw(run.dir), run.figures, name)
        run.log(f"saved figure {name}")


def fig_attack_share_by_block(run_dir) -> plt.Figure:
    style.use()
    blocks = runs.load_table(run_dir, "blocks")
    summary = runs.load_json(run_dir, "summary")
    years = summary["years"]
    fig, axes = plt.subplots(2, 1, figsize=(style.WIDTH_DOUBLE, 3.8), sharex=True)
    # a, the share of attacks of either type in each block. b, the share of the unknown type, on a log axis
    axes[0].plot(blocks["position"], blocks["attack_share"], color=style.COLORS[0], marker="o", markersize=2.5,
                 linewidth=1)
    axes[0].set_ylim(0, 1)
    axes[0].yaxis.set_major_formatter(style.percent)
    axes[0].set_ylabel("attack share")
    positive = blocks[blocks["n_unknown"] > 0]
    # the line breaks at blocks without the unknown type rather than bridging them
    share = np.where(blocks["n_unknown"] > 0, blocks["unknown_share"], np.nan)
    axes[1].plot(blocks["position"], share, color=style.COLORS[1], marker="o", markersize=2.5, linewidth=1)
    axes[1].set_yscale("log")
    low = 10.0 ** np.floor(np.log10(positive["unknown_share"].min())) if len(positive) else 1e-6
    high = 10.0 ** np.ceil(np.log10(positive["unknown_share"].max())) if len(positive) else 1.0
    axes[1].set_ylim(low, high)
    extra = [(Line2D([], [], color=style.GRAY, linestyle="--", linewidth=1), "AnoShift period boundary")]
    zero = blocks[blocks["n_unknown"] == 0]
    if len(zero):
        # zero has no place on a log axis, so blocks without the unknown type sit on the lower edge
        axes[1].plot(zero["position"], np.full(len(zero), low), linestyle="none", marker="o", markersize=3,
                     markerfacecolor="white", markeredgecolor=style.GRAY, clip_on=False)
        extra.append((Line2D([], [], linestyle="none", marker="o", markersize=3, markerfacecolor="white",
                             markeredgecolor=style.GRAY), "no unknown type in the block"))
    axes[1].yaxis.set_major_formatter(style.percent)
    axes[1].yaxis.set_minor_formatter(NullFormatter())
    axes[1].set_ylabel("unknown type share")
    for ax, letter in zip(axes, "ab"):
        for first, _ in list(summary["periods"].values())[1:]:
            ax.axvline(first, color=style.GRAY, linestyle="--", linewidth=1)
        layout.panel_letter(ax, letter)
    axes[1].set_xlim(years[0], years[-1] + 1)
    axes[1].set_xticks(np.arange(years[0], years[-1] + 2))
    axes[1].set_xlabel(f"year, each split into {summary['blocks_per_year']} blocks of equal session count")
    layout.legend_below(fig, axes[1], [], extra, ncol=2)
    return fig


def fig_service_mix_by_year(run_dir) -> plt.Figure:
    style.use()
    services = runs.load_table(run_dir, "services")
    years = runs.load_json(run_dir, "summary")["years"]
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH_DOUBLE, 3.2))
    # one blue for the share, a service absent from a year leaves its cell empty, white lines keep cells apart
    cmap = LinearSegmentedColormap.from_list("share", style.SEQUENTIAL)
    for ax, cls, letter in zip(axes, ["normal", "attack"], "ab"):
        grid = (services[services["class"] == cls].pivot_table(index="service", columns="year", values="share",
                                                               fill_value=0.0)
                .reindex(columns=years, fill_value=0.0))
        major = grid[(grid >= SERVICE_FLOOR).any(axis=1)]
        major = major.loc[major.mean(axis=1).sort_values(ascending=False).index]
        rest = grid.drop(index=major.index).sum()
        if (rest > 0).any():
            major = pd.concat([major, rest.rename(REST).to_frame().T])
        values = major.to_numpy()
        mesh = ax.pcolormesh(np.ma.masked_where(~(values > 0), values), cmap=cmap, vmin=0, vmax=1,
                             edgecolors="white", linewidth=1.0)
        ax.set_yticks(np.arange(len(major.index)) + 0.5, major.index)
        ax.invert_yaxis()
        ax.set_xticks(np.arange(len(years)) + 0.5, [str(y) for y in years], rotation=90)
        ax.set_xlabel(f"year, {cls} sessions")
        ax.tick_params(length=0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        layout.panel_letter(ax, letter)
    fig.tight_layout()
    bar = fig.colorbar(mesh, ax=axes, fraction=0.03, pad=0.02)
    bar.set_label("share of the sessions of the class in the year")
    bar.formatter = style.percent
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0)
    return fig


FIGURES = {
    "attack_share_by_block": fig_attack_share_by_block,
    "service_mix_by_year": fig_service_mix_by_year,
}
