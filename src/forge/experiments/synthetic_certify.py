"""e10, a check of Theorem 1 on synthetic scores whose true false omission rate is known in closed form.

benign scores follow one beta distribution and each attack family its own, see the config. the threshold grid holds the
true benign quantiles at levels j / (G + 1) and stands for a large D_sel. the guarantee holds given the grid, so fixing
it loses nothing. for each size of D_cal, independent calibration sets are drawn, every method certifies a threshold on
each, and the true rate of that threshold is computed at each deployment prevalence and family mix. a draw violates
when the true rate exceeds epsilon, and the share of violating draws is compared with delta.

what each method claims.
    forge_simplex   every prevalence up to pi_max and every family mix, the claim of Theorem 1
    forge_pooled    every prevalence up to pi_max with the family mix of the calibration set
    ltt_iid         the prevalence and the family mix of the calibration set
    forge_strict    the claim of forge_simplex, with the sequence started at the strictest grid threshold, which shows
                    why FORGE-Certify starts where a threshold could first pass

a FORGE claim also holds the automation floor. the figures show the FOR part, except panel a over sizes, which shows
the whole claim, as the summary does.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from scipy.stats import beta

from forge import runs
from forge.certify.bounds import clopper_pearson_lower, clopper_pearson_upper
from forge.certify.certificate import certify, fixed_sequence, for_from_rates, ltt_iid
from forge.data.synthetic import beta_cdfs, draw_counts
from forge.viz import style
from forge.viz.export import save_figure

METHODS = ["forge_simplex", "forge_pooled", "ltt_iid", "forge_strict"]
CLAIMS = {"forge_simplex": "any_mix", "forge_pooled": "calibration_mix", "ltt_iid": "iid", "forge_strict": "any_mix"}
LABELS = {
    "forge_simplex": "FORGE, any family mix",
    "forge_pooled": "FORGE, calibration mix",
    "ltt_iid": "LTT, i.i.d.",
    "forge_strict": "FORGE, from the strictest threshold",
}
MARKERS = {"forge_simplex": "o", "forge_pooled": "s", "ltt_iid": "^", "forge_strict": "D"}
# forge_strict makes the claim of forge_simplex and is compared with it in the figures over sizes
PREVALENCE_METHODS = ["forge_simplex", "forge_pooled", "ltt_iid"]


class Truth:
    # true class cdfs at the grid, from which the false omission rate of any certified threshold follows exactly
    def __init__(self, b: np.ndarray, m: np.ndarray, names: list[str], shares: np.ndarray):
        self.b = b
        self.names = names
        self.mixes = {"calibration": shares @ m, **dict(zip(names, m))}

    def rate(self, index: np.ndarray, pi: float, mix: str) -> np.ndarray:
        # true FOR of the thresholds at index, 0 where nothing is dismissed. the worst mix, any, is always a single
        # family, since the rate rises with the miss rate and the miss rate is linear in the mix
        if mix == "any":
            return np.max([self.rate(index, pi, name) for name in self.names], axis=0)
        safe = np.maximum(index, 0)
        return np.where(index >= 0, for_from_rates(pi, self.mixes[mix][safe], self.b[safe]), 0.0)

    def claimed(self, index: np.ndarray, claim: str, pi_max: float, pi_cal: float) -> np.ndarray:
        # largest true FOR over the deployments a claim covers, the rate rises with the prevalence, so pi_max is the
        # worst prevalence of the FORGE claims
        if claim == "any_mix":
            return self.rate(index, pi_max, "any")
        if claim == "calibration_mix":
            return self.rate(index, pi_max, "calibration")
        return self.rate(index, pi_cal, "calibration")

    def lowest_dismiss_rate(self, index: np.ndarray, claim: str, pi_max: float) -> np.ndarray:
        # smallest share of all flows dismissed over the deployments a FORGE claim covers. it is linear in the
        # prevalence and in the mix, so the minimum sits at prevalence 0 or pi_max and at a single family
        safe = np.maximum(index, 0)
        mixes = self.names if claim == "any_mix" else ["calibration"]
        m = np.min([self.mixes[mix][safe] for mix in mixes], axis=0)
        b = self.b[safe]
        return np.minimum(b, pi_max * m + (1 - pi_max) * b)


def certify_draws(method: str, d, c: dict) -> dict:
    # certified index, FOR bound and automation floor of one method on every calibration set of d
    args = (c["epsilon"], c["delta"], c["pi_max"], c["delta_benign"])
    pooled_below, pooled_n = d.family_below.sum(1, keepdims=True), d.n_family.sum(1, keepdims=True)
    if method == "forge_simplex":
        return certify(d.benign_below, d.n_benign, d.family_below, d.n_family, *args)
    if method == "forge_strict":
        return certify(d.benign_below, d.n_benign, d.family_below, d.n_family, *args, start="strictest")
    if method == "forge_pooled":
        return certify(d.benign_below, d.n_benign, pooled_below, pooled_n, *args)
    if method == "ltt_iid":
        out = ltt_iid(d.benign_below, pooled_below[:, 0], c["epsilon"], c["delta"])
        return {**out, "floor": np.full(out["index"].shape, np.nan)}
    raise ValueError(f"unknown method {method}")


def share_with_interval(count: int, total: int, prefix: str = "") -> dict:
    # share of draws with a two sided 95% Clopper-Pearson interval
    return {f"{prefix}rate": count / total, f"{prefix}low": clopper_pearson_lower(count, total, 0.025),
            f"{prefix}high": clopper_pearson_upper(count, total, 0.025)}


def run(cfg: dict, run: runs.Run) -> None:
    c, s, sim = cfg["certify"], cfg["synthetic"], cfg["simulation"]
    eps, delta, pi_max, pi_cal = c["epsilon"], c["delta"], c["pi_max"], s["prevalence"]
    methods, sizes, draws = sim["methods"], sim["sizes"], sim["draws"]
    if sim["main_size"] not in sizes:
        raise ValueError(f"main size {sim['main_size']} is not among the sizes {sizes}")
    names = [f["name"] for f in s["families"]]
    shares = np.array([f["share"] for f in s["families"]], dtype=float)
    benign = (s["benign"]["a"], s["benign"]["b"])
    levels = np.arange(1, c["grid_size"] + 1) / (c["grid_size"] + 1)
    grid = beta.ppf(levels, *benign)
    b, m = beta_cdfs(grid, benign, [(f["a"], f["b"]) for f in s["families"]])
    truth = Truth(b, m, names, shares)
    run.save_table(pd.DataFrame({"index": np.arange(len(grid)), "level": levels, "threshold": grid, "benign": b,
                                 **dict(zip(names, m))}), "truth")

    # the threshold each claim would certify with unlimited calibration data
    oracle = {}
    for claim in dict.fromkeys(CLAIMS[method] for method in methods):
        j = int(fixed_sequence(truth.claimed(np.arange(len(grid)), claim, pi_max, pi_cal) <= eps, 0))
        oracle[claim] = float(b[j]) if j >= 0 else 0.0
        run.log(f"claim {claim}, best grid threshold dismisses benign share {oracle[claim]:.3f}")

    # one stream per size, so adding a size leaves the draws of the others unchanged
    seeds = np.random.SeedSequence(cfg["seed"]).spawn(len(sizes))
    shape = (len(sizes), len(methods), draws)
    index = np.full(shape, -1, dtype=np.int32)
    bound = np.full(shape, np.nan)
    floor = np.full(shape, np.nan)
    counts = np.zeros((len(sizes), draws, len(names) + 1), dtype=np.int64)
    rows, claim_rows = [], []
    for i, (n, seed) in enumerate(zip(sizes, seeds)):
        d = draw_counts(np.random.default_rng(seed), n, pi_cal, shares, b, m, draws)
        counts[i] = np.column_stack([d.n_benign, d.n_family])
        for j, method in enumerate(methods):
            out = certify_draws(method, d, c)
            index[i, j], bound[i, j], floor[i, j] = out["index"], out["bound"], out["floor"]
            idx = index[i, j]
            for pi in sim["prevalences"]:
                for mix in ["calibration", "any", *names]:
                    v = int(np.sum(truth.rate(idx, pi, mix) > eps))
                    rows.append({"size": n, "method": method, "prevalence": pi, "mix": mix, "draws": draws,
                                 "violations": v, **share_with_interval(v, draws)})
            claim = CLAIMS[method]
            over = truth.claimed(idx, claim, pi_max, pi_cal) > eps
            # a FORGE claim also holds the automation floor, so a draw dismissing less than the floor violates it too
            under = (idx >= 0) & (truth.lowest_dismiss_rate(idx, claim, pi_max) < floor[i, j])
            either = over | under
            c1 = int(np.sum(truth.claimed(idx, "any_mix", pi_max, pi_cal) > eps))
            dismissed = np.where(idx >= 0, b[np.maximum(idx, 0)], 0.0)
            has_floor = claim != "iid"
            row = {
                "size": n, "method": method, "claim": claim, "draws": draws, "refused": float(np.mean(idx < 0)),
                "benign_rate_mean": float(dismissed.mean()), "benign_rate_median": float(np.median(dismissed)),
                "floor_mean": float(floor[i, j].mean()) if has_floor else np.nan,
                "violations": int(over.sum()), **share_with_interval(int(over.sum()), draws),
                "floor_violations": int(under.sum()) if has_floor else np.nan,
                "joint_violations": int(either.sum()), **share_with_interval(int(either.sum()), draws, "joint_"),
                "c1_violations": c1, **share_with_interval(c1, draws, "c1_"),
                "oracle_benign_rate": oracle[claim],
            }
            claim_rows.append(row)
            floor_part = f", floor broken in {row['floor_violations']} draws" if has_floor else ""
            run.log(f"size {n}, {method}, refused {row['refused']:.3f}, benign dismiss rate "
                    f"{row['benign_rate_mean']:.3f}, FOR claim violated in {row['rate']:.4f} of draws, 95% interval "
                    f"{row['low']:.4f} to {row['high']:.4f}{floor_part}")

    run.save_arrays("draws", sizes=np.array(sizes), methods=np.array(methods), families=np.array(names), index=index,
                    bound=bound, floor=floor, counts=counts)
    run.save_table(pd.DataFrame(rows), "violations")
    claims = pd.DataFrame(claim_rows)
    run.save_table(claims, "claims")

    # the check uses the whole claim, the FOR part and the floor together, and fails a claim only when even the lower
    # end of the interval lies above delta
    largest, exceeds = {}, {}
    for method, g in claims.groupby("method", sort=False):
        top = g.loc[g["joint_rate"].idxmax()]
        largest[method] = {"size": int(top["size"]), "rate": float(top["joint_rate"]),
                           "low": float(top["joint_low"]), "high": float(top["joint_high"])}
        exceeds[method] = bool((g["joint_low"] > delta).any())
        run.log(f"{method}, largest share of draws violating its claim {top['joint_rate']:.4f} at size "
                f"{int(top['size'])}, 95% interval {top['joint_low']:.4f} to {top['joint_high']:.4f}, delta {delta}, "
                f"above delta beyond simulation noise {exceeds[method]}")
    run.save_json({
        "epsilon": eps, "delta": delta, "delta_benign": c["delta_benign"], "pi_max": pi_max, "prevalence": pi_cal,
        "grid_size": c["grid_size"], "benign": s["benign"], "families": s["families"], "draws": draws,
        "sizes": sizes, "main_size": sim["main_size"], "methods": methods, "oracle_benign_rate": oracle,
        "largest_claim_violation": largest, "claim_exceeds_delta": exceeds,
    }, "summary")

    for name, draw in FIGURES.items():
        save_figure(draw(run.dir), run.figures, name)
        run.log(f"saved figure {name}")


def _dodge(x, k: int, n: int, step: float = 0.045) -> np.ndarray:
    # shifts the points of the k-th of n methods sideways on a log axis, so markers and intervals do not hide each other
    return np.asarray(x, dtype=float) * 10.0 ** ((k - (n - 1) / 2) * step)


def _points(ax, x, rate, low, high, method: str) -> None:
    rate, low, high = (np.asarray(v, dtype=float) for v in (rate, low, high))
    ax.errorbar(x, rate, yerr=[np.clip(rate - low, 0, None), np.clip(high - rate, 0, None)],
                color=style.color_for(method, METHODS), marker=MARKERS[method], markersize=3.5, linewidth=1.2,
                elinewidth=0.8, capsize=0, label=LABELS[method])


def _panel_letter(ax, letter: str) -> None:
    # above the top left corner of the plot, clear of the data
    ax.text(0, 1.03, f"({letter})", transform=ax.transAxes, ha="left", va="bottom")


def _legend_below(fig, ax, methods: list[str], extra=(), ncol: int = 3) -> None:
    # methods first, then the reference lines. matplotlib fills legend columns first, so the entries are rearranged to
    # read row by row
    found = dict(zip(*reversed(ax.get_legend_handles_labels())))
    entries = [(found[LABELS[m]], LABELS[m]) for m in methods]
    entries += [(h, label) for label, h in found.items() if label not in {LABELS[m] for m in methods}] + list(extra)
    rows = -(-len(entries) // ncol)
    order = [r * ncol + c for c in range(ncol) for r in range(rows) if r * ncol + c < len(entries)]
    fig.tight_layout()
    fig.legend(*zip(*[entries[i] for i in order]), loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 0.0))


def fig_violation_by_prevalence(run_dir) -> plt.Figure:
    style.use()
    table = runs.load_table(run_dir, "violations")
    summary = runs.load_json(run_dir, "summary")
    main = table[table["size"] == summary["main_size"]]
    methods = [x for x in PREVALENCE_METHODS if x in summary["methods"]]
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH_DOUBLE, 2.4), sharey=True)
    # a with the family mix of the calibration set, b with the worst single family
    for ax, mix, letter in zip(axes, ["calibration", "any"], "ab"):
        for k, method in enumerate(methods):
            g = main[(main["method"] == method) & (main["mix"] == mix)].sort_values("prevalence")
            _points(ax, _dodge(g["prevalence"], k, len(methods)), g["rate"], g["low"], g["high"], method)
        ax.axhline(summary["delta"], color=style.GRAY, linestyle="--", linewidth=1, label=r"$\delta$")
        ax.axvline(summary["prevalence"], color=style.GRAY, linestyle=":", linewidth=1,
                   label=r"$\pi$ of the calibration set")
        ax.axvline(summary["pi_max"], color=style.GRAY, linestyle="-.", linewidth=1, label=r"$\pi_{\max}$")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(style.percent)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("attack prevalence in deployment")
        _panel_letter(ax, letter)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_ylabel(r"share of draws with FOR > $\varepsilon$")
    _legend_below(fig, axes[0], methods)
    return fig


def fig_violation_and_dismiss_by_size(run_dir) -> plt.Figure:
    style.use()
    claims = runs.load_table(run_dir, "claims")
    summary = runs.load_json(run_dir, "summary")
    delta = summary["delta"]
    methods = [x for x in METHODS if x in summary["methods"]]
    fig, axes = plt.subplots(1, 3, figsize=(style.WIDTH_DOUBLE, 2.6))
    # a, each method against its whole claim. b, every method against the FOR part of Theorem 1. c, the share of
    # benign flows dismissed, a refusal counted as none
    for k, method in enumerate(methods):
        g = claims[claims["method"] == method].sort_values("size")
        x = _dodge(g["size"], k, len(methods))
        _points(axes[0], x, g["joint_rate"], g["joint_low"], g["joint_high"], method)
        _points(axes[1], x, g["c1_rate"], g["c1_low"], g["c1_high"], method)
        axes[2].plot(x, g["benign_rate_mean"], color=style.color_for(method, METHODS), marker=MARKERS[method],
                     markersize=3.5, linewidth=1.2, label=LABELS[method])
    for ax in axes[:2]:
        ax.axhline(delta, color=style.GRAY, linestyle="--", linewidth=1, label=r"$\delta$")
    # the best grid threshold of each claim, dotted in the color of the first method making that claim
    for claim, method in {CLAIMS[x]: x for x in reversed(methods)}.items():
        axes[2].axhline(summary["oracle_benign_rate"][claim], color=style.color_for(method, METHODS), linestyle=":",
                        linewidth=1)
    axes[0].set_ylim(0, max(2 * delta, 1.15 * claims["joint_high"].max()))
    axes[0].set_ylabel("draws violating own claim")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("draws violating Theorem 1")
    axes[2].set_ylim(0, 1)
    axes[2].yaxis.set_major_formatter(style.percent)
    axes[2].set_ylabel("benign flows dismissed")
    for ax, letter in zip(axes, "abc"):
        ax.set_xscale("log")
        ax.set_xlabel("flows in the calibration set")
        _panel_letter(ax, letter)
    _legend_below(fig, axes[0], methods, [(Line2D([], [], color=style.GRAY, linestyle=":", linewidth=1),
                                           "best grid threshold")])
    return fig


FIGURES = {
    "violation_by_prevalence": fig_violation_by_prevalence,
    "violation_and_dismiss_by_size": fig_violation_and_dismiss_by_size,
}
