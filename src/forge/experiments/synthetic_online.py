"""e14, a check of Theorem 1b on synthetic deployment windows whose benign scores drift away from the calibration set.

the calibration set, its attack families and the grid are those of e10 at its main size. deployment is a run of
windows, and the benign scores of each window follow their own beta distribution, which stays at the calibration one,
moves away in equal steps or jumps once, see the scenarios in the config. every attack family keeps its score
distribution, the condition of Theorem 1b. window 0 lies just before deployment and keeps the calibration benign
scores. a draw pairs one calibration set with one independent run of windows, and the true rate of every certified
threshold follows from the known distributions of the window.

what each method claims for a window.
    forge_simplex    the offline certificate of Theorem 1, one threshold for every window, valid while the benign
                     scores keep their calibration distribution
    forge_online     Theorem 1b, a threshold certified on the window's own unlabeled scores, valid whatever the benign
                     scores do
    forge_previous   the online certificate of the window before, which has no guarantee and holds only while the
                     share of flows dismissed changes little from one window to the next

a claim covers the window's FOR at its prevalence and family mix, together with the automation floor of the method.
the floors of the offline and the previous window certificates break as soon as the window's dismissed share falls
below them, which any drift larger than their DKW margin does, so the figures show the FOR part, at the worst
composition where the claims are tight, and the dismissed share at the typical composition. the log and the summary
give the FOR part and the whole claim apart. the realized share of attacks among the dismissed flows of a window is
saved too, it is a different quantity from the rate the theorems bound.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import beta

from forge import runs
from forge.certify.certificate import certify, certify_online, for_from_rates
from forge.data.synthetic import benign_path, beta_cdfs, draw_counts
from forge.eval.metrics import share_with_interval
from forge.viz import layout, style
from forge.viz.export import save_figure

METHODS = ["forge_simplex", "forge_online", "forge_previous"]
LABELS = {
    "forge_simplex": "FORGE offline, Theorem 1",
    "forge_online": "FORGE online, same window, Theorem 1b",
    "forge_previous": "FORGE online, previous window",
}
MARKERS = {"forge_simplex": "o", "forge_online": "v", "forge_previous": "P"}
# the composition each figure shows
FIGURE_COMPOSITIONS = {"violation": "worst", "dismiss": "typical"}


def run(cfg: dict, run: runs.Run) -> None:
    c, s, dep, sim = cfg["certify"], cfg["synthetic"], cfg["deployment"], cfg["simulation"]
    eps, delta, pi_max, part = c["epsilon"], c["delta"], c["pi_max"], c["delta_benign"]
    start = {"start_misses": c["start_misses"], "start_slack": c["start_slack"]}
    methods, draws, windows, sizes = sim["methods"], sim["draws"], dep["windows"], dep["window_sizes"]
    if dep["main_window_size"] not in sizes:
        raise ValueError(f"main window size {dep['main_window_size']} is not among the window sizes {sizes}")
    missing = set(FIGURE_COMPOSITIONS.values()) - {co["name"] for co in dep["compositions"]}
    if missing:
        raise ValueError(f"the figures need the compositions {', '.join(sorted(missing))}")
    shares = np.array([f["share"] for f in s["families"]], dtype=float)
    benign = (s["benign"]["a"], s["benign"]["b"])
    grid = beta.ppf(np.arange(1, c["grid_size"] + 1) / (c["grid_size"] + 1), *benign)
    b_cal, m = beta_cdfs(grid, benign, [(f["a"], f["b"]) for f in s["families"]])

    # one stream for the calibration sets and one for each scenario, composition and window size
    cells = [(sc, co, n) for sc in dep["scenarios"] for co in dep["compositions"] for n in sizes]
    seeds = np.random.SeedSequence(cfg["seed"]).spawn(1 + len(cells))
    cal = draw_counts(np.random.default_rng(seeds[0]), s["calibration_size"], s["prevalence"], shares, b_cal, m, draws)
    offline = certify(cal.benign_below, cal.n_benign, cal.family_below, cal.n_family, eps, delta, pi_max, part,
                      **start)
    run.log(f"calibration sets of {s['calibration_size']} flows, the offline certificate refused in "
            f"{np.mean(offline['index'] < 0):.3f} of draws")

    paths = {sc["name"]: benign_path(benign, (sc["end"]["a"], sc["end"]["b"]), windows, sc.get("shift_at"))
             for sc in dep["scenarios"]}
    benign_windows = {name: np.array([beta.cdf(grid, *ab) for ab in path]) for name, path in paths.items()}
    run.save_table(pd.DataFrame([{"scenario": name, "window": w, "a": ab[0], "b": ab[1]}
                                 for name, path in paths.items() for w, ab in enumerate(path)]), "benign_path")
    run.save_arrays("truth", grid=grid, benign_calibration=b_cal, families=m, scenarios=np.array(list(paths)),
                    benign_windows=np.stack(list(benign_windows.values())))

    rows = []
    online_index = np.full((len(cells), windows + 1, draws), -1, dtype=np.int16)
    for i, ((sc, co, n), seed) in enumerate(zip(cells, seeds[1:])):
        rng = np.random.default_rng(seed)
        pi_w, mix = co["prevalence"], np.asarray(co["shares"], dtype=float)
        b_w = benign_windows[sc["name"]]
        m_w = mix @ m
        true_for = for_from_rates(pi_w, m_w, b_w)
        true_dismissed = pi_w * m_w + (1 - pi_w) * b_w
        window_below = np.empty((windows + 1, draws, len(grid)), dtype=np.int64)
        attack_below = np.empty_like(window_below)
        for w in range(windows + 1):
            d = draw_counts(rng, n, pi_w, mix, b_w[w], m, draws)
            attack_below[w] = d.family_below.sum(1)
            window_below[w] = d.benign_below + attack_below[w]
        online = certify_online(window_below, np.full(window_below.shape[:-1], n), cal.family_below, cal.n_family,
                                eps, delta, pi_max, part, **start)
        online_index[i] = online["index"]
        certified = {
            "forge_simplex": (np.broadcast_to(offline["index"], online["index"].shape),
                              np.broadcast_to(offline["floor"], online["floor"].shape)),
            "forge_online": (online["index"], online["floor"]),
            # the certificate of the window before, applied to this window, window 0 has none
            "forge_previous": (np.concatenate([np.full((1, draws), -1), online["index"][:-1]]),
                               np.concatenate([np.zeros((1, draws)), online["floor"][:-1]])),
        }
        for method in methods:
            index, floor = certified[method]
            first_row = len(rows)
            for w in range(1, windows + 1):
                hit = index[w] >= 0
                safe = np.maximum(index[w], 0)
                share = np.where(hit, true_dismissed[w, safe], 0.0)
                over = np.where(hit, true_for[w, safe], 0.0) > eps
                # a draw that dismisses less of the window than its floor breaks the claim as well
                under = hit & (share < floor[w])
                either = over | under
                kept = np.take_along_axis(window_below[w], safe[:, None], 1)[:, 0]
                caught = np.take_along_axis(attack_below[w], safe[:, None], 1)[:, 0]
                realized = hit & (caught > eps * kept)
                rows.append({
                    "scenario": sc["name"], "composition": co["name"], "window_size": n, "window": w,
                    "method": method, "draws": draws, "refused": float(np.mean(~hit)),
                    "dismissed_mean": float(share.mean()),
                    "benign_dismissed_mean": float(np.where(hit, b_w[w, safe], 0.0).mean()),
                    "floor_mean": float(np.where(hit, floor[w], 0.0).mean()),
                    "violations": int(over.sum()), **share_with_interval(int(over.sum()), draws),
                    "floor_violations": int(under.sum()),
                    "joint_violations": int(either.sum()), **share_with_interval(int(either.sum()), draws, "joint_"),
                    "realized_violations": int(realized.sum()), "realized_rate": float(realized.mean()),
                })
            g = rows[first_row:]
            top = max(g, key=lambda r: r["rate"])
            run.log(f"scenario {sc['name']}, composition {co['name']}, window size {n}, {method}, FOR above epsilon "
                    f"in at most {top['rate']:.4f} of draws at window {top['window']}, floor broken in at most "
                    f"{max(r['floor_violations'] for r in g)} draws, mean share of window flows dismissed "
                    f"{np.mean([r['dismissed_mean'] for r in g]):.3f}")

    claims = pd.DataFrame(rows)
    run.save_table(claims, "claims")
    run.save_arrays("draws", offline_index=offline["index"], offline_bound=offline["bound"],
                    offline_floor=offline["floor"], online_index=online_index,
                    cell_scenarios=np.array([sc["name"] for sc, _, _ in cells]),
                    cell_compositions=np.array([co["name"] for _, co, _ in cells]),
                    cell_window_sizes=np.array([n for _, _, n in cells]),
                    counts=np.column_stack([cal.n_benign, cal.n_family]))

    # a check fails a method only when even the lower end of the interval lies above delta, for the FOR part alone
    # and for the whole claim with the floor
    largest, exceeds = {}, {}
    for method in methods:
        largest[method], exceeds[method] = {}, {}
        for scenario in paths:
            g = claims[(claims["method"] == method) & (claims["scenario"] == scenario)]
            top, whole = g.loc[g["rate"].idxmax()], g.loc[g["joint_rate"].idxmax()]
            largest[method][scenario] = {
                "window": int(top["window"]), "composition": top["composition"], "window_size": int(top["window_size"]),
                "rate": float(top["rate"]), "low": float(top["low"]), "high": float(top["high"]),
                "claim_rate": float(whole["joint_rate"]), "claim_low": float(whole["joint_low"]),
                "claim_high": float(whole["joint_high"])}
            exceeds[method][scenario] = {"for": bool((g["low"] > delta).any()),
                                         "claim": bool((g["joint_low"] > delta).any())}
            run.log(f"{method}, scenario {scenario}, largest share of draws with FOR above epsilon {top['rate']:.4f} "
                    f"at window {int(top['window'])}, composition {top['composition']}, window size "
                    f"{int(top['window_size'])}, 95% interval {top['low']:.4f} to {top['high']:.4f}, whole claim "
                    f"violated in at most {whole['joint_rate']:.4f}, above delta {delta} beyond simulation noise for "
                    f"FOR {exceeds[method][scenario]['for']} and for the whole claim "
                    f"{exceeds[method][scenario]['claim']}")
    run.save_json({
        "epsilon": eps, "delta": delta, "delta_benign": part, "pi_max": pi_max, **start, "grid_size": c["grid_size"],
        "calibration_size": s["calibration_size"], "prevalence": s["prevalence"], "benign": s["benign"],
        "families": s["families"], "windows": windows, "window_sizes": sizes,
        "main_window_size": dep["main_window_size"], "scenarios": list(paths),
        "shift_at": {sc["name"]: sc.get("shift_at") for sc in dep["scenarios"]},
        "compositions": dep["compositions"], "figure_compositions": FIGURE_COMPOSITIONS, "draws": draws,
        "methods": methods, "offline_refused": float(np.mean(offline["index"] < 0)),
        "largest_claim_violation": largest, "claim_exceeds_delta": exceeds,
    }, "summary")

    for name, draw in FIGURES.items():
        save_figure(draw(run.dir), run.figures, name)
        run.log(f"saved figure {name}")


def _points(ax, x, rate, low, high, method: str) -> None:
    layout.points(ax, x, rate, low, high, color=style.color_for(method), marker=MARKERS[method], label=LABELS[method])


def _mark_jump(ax, shift_at) -> None:
    # between the last window before the jump of the benign scores and the first one after it
    if shift_at is not None:
        ax.axvline(shift_at - 0.5, color=style.GRAY, linestyle=":", linewidth=1)


def _jump_entry(summary: dict) -> list:
    if all(v is None for v in summary["shift_at"].values()):
        return []
    return [(Line2D([], [], color=style.GRAY, linestyle=":", linewidth=1), "benign jump")]


def _window_axis(ax, windows: int) -> None:
    ax.set_xticks(sorted({1, *range(5, windows + 1, 5)}))
    ax.set_xlim(0.4, windows + 0.6)


def _flows(n: int) -> str:
    # 1000000 as 10^6 when n is a power of ten
    power = int(round(np.log10(n)))
    return f"$10^{{{power}}}$" if 10 ** power == n else f"{n:,}"


def fig_violation_by_window(run_dir) -> plt.Figure:
    style.use()
    claims = runs.load_table(run_dir, "claims")
    summary = runs.load_json(run_dir, "summary")
    delta = summary["delta"]
    methods = [x for x in METHODS if x in summary["methods"]]
    scenarios = summary["scenarios"]
    data = claims[(claims["window_size"] == summary["main_window_size"])
                  & (claims["composition"] == summary["figure_compositions"]["violation"])]
    fig, axes = plt.subplots(2, len(scenarios), figsize=(style.WIDTH_DOUBLE, 4.4), sharex=True, squeeze=False)
    # the top row shows the whole range, the bottom row the same points up to twice delta
    for col, scenario in enumerate(scenarios):
        g = data[data["scenario"] == scenario]
        for ax in axes[:, col]:
            for k, method in enumerate(methods):
                h = g[g["method"] == method].sort_values("window")
                _points(ax, layout.dodge(h["window"], k, len(methods), 0.25), h["rate"], h["low"], h["high"],
                        method)
            ax.axhline(delta, color=style.GRAY, linestyle="--", linewidth=1, label=r"$\delta$")
            _mark_jump(ax, summary["shift_at"][scenario])
            _window_axis(ax, summary["windows"])
        axes[0, col].set_ylim(-0.02, 1.02)
        axes[1, col].set_ylim(0, 2 * delta)
        axes[1, col].set_xlabel("deployment window")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"share of draws with FOR > $\varepsilon$")
    for ax, letter in zip(axes.flat, "abcdefghijkl"):
        layout.panel_letter(ax, letter)
    layout.legend_below(fig, axes[0, 0], [LABELS[m] for m in methods], _jump_entry(summary))
    return fig


def fig_dismiss_by_window(run_dir) -> plt.Figure:
    style.use()
    claims = runs.load_table(run_dir, "claims")
    summary = runs.load_json(run_dir, "summary")
    methods = [x for x in METHODS if x in summary["methods"]]
    scenarios = summary["scenarios"]
    main = summary["main_window_size"]
    data = claims[claims["composition"] == summary["figure_compositions"]["dismiss"]]
    fig, axes = plt.subplots(1, len(scenarios), figsize=(style.WIDTH_DOUBLE, 2.5), sharey=True, squeeze=False)
    axes = axes[0]
    # every method on windows of the main size, the online certificate also on the other sizes, in its color with
    # other dashes
    sizes = [n for n in summary["window_sizes"] if n != main] if "forge_online" in methods else []
    for ax, scenario, letter in zip(axes, scenarios, "abcdef"):
        g = data[data["scenario"] == scenario]
        for method in methods:
            h = g[(g["method"] == method) & (g["window_size"] == main)].sort_values("window")
            ax.plot(h["window"], h["dismissed_mean"], color=style.color_for(method), marker=MARKERS[method],
                    markersize=3.5, linewidth=1.2, label=LABELS[method])
        for n, dash in zip(sizes, style.DASHES[1:]):
            h = g[(g["method"] == "forge_online") & (g["window_size"] == n)].sort_values("window")
            ax.plot(h["window"], h["dismissed_mean"], color=style.color_for("forge_online"), linestyle=dash,
                    linewidth=1, label=f"FORGE online, windows of {_flows(n)} flows")
        _mark_jump(ax, summary["shift_at"][scenario])
        _window_axis(ax, summary["windows"])
        ax.set_xlabel("deployment window")
        layout.panel_letter(ax, letter)
    axes[0].set_ylim(0, 1.1 * data["dismissed_mean"].max())
    axes[0].yaxis.set_major_formatter(style.percent)
    axes[0].set_ylabel("window flows dismissed")
    layout.legend_below(fig, axes[0], [LABELS[m] for m in methods], _jump_entry(summary))
    return fig


FIGURES = {
    "violation_by_window": fig_violation_by_window,
    "dismiss_by_window": fig_dismiss_by_window,
}
