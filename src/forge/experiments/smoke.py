"""end to end check of the run loop on synthetic scores.

a threshold is chosen on the first window so the upper bound on its false omission rate stays below epsilon, then held fixed while the attack prevalence changes in later windows. it touches every part of the loop, config, run folder, saved results and figures, in under a minute on a CPU.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forge import runs
from forge.certify.bounds import clopper_pearson_upper
from forge.data.synthetic import scores_by_window
from forge.eval.metrics import dismiss_rate, false_omission_rate, miss_rate
from forge.viz import style
from forge.viz.export import save_figure


def run(cfg: dict, run: runs.Run) -> None:
    c = cfg["smoke"]
    windows, labels, scores = scores_by_window(c["prevalences"], c["n_per_window"], cfg["seed"])
    run.save_arrays("scores", windows=windows, labels=labels, scores=scores)
    run.log(f"generated {len(scores)} flows in {len(c['prevalences'])} windows")

    grid = np.linspace(0, 1, c["grid_size"])
    rows = []
    for w in np.unique(windows):
        m = windows == w
        for t in grid:
            rows.append({
                "window": int(w),
                "prevalence": float(labels[m].mean()),
                "threshold": float(t),
                "dismiss_rate": dismiss_rate(scores[m], t),
                "for": false_omission_rate(scores[m], labels[m], t),
                "miss_rate": miss_rate(scores[m], labels[m], t),
            })
    curves = pd.DataFrame(rows)
    run.save_table(curves, "curves")

    # largest threshold on window 0 whose upper bound on the false omission rate is below epsilon
    cal = windows == 0
    chosen = 0.0
    for t in grid:
        dismissed = scores[cal] <= t
        if clopper_pearson_upper(int(labels[cal][dismissed].sum()), int(dismissed.sum()), c["delta"]) <= c["epsilon"]:
            chosen = float(t)
    summary = {"epsilon": c["epsilon"], "delta": c["delta"], "threshold": chosen}
    run.save_json(summary, "summary")
    run.log(f"threshold {chosen:.3f} certified on window 0 at epsilon {c['epsilon']}")

    for name, draw in FIGURES.items():
        save_figure(draw(run.dir), run.figures, name)
        run.log(f"saved figure {name}")


def fig_for_by_window(run_dir) -> plt.Figure:
    style.use()
    curves = runs.load_table(run_dir, "curves")
    summary = runs.load_json(run_dir, "summary")
    t = summary["threshold"]
    at_t = curves.loc[np.isclose(curves["threshold"], t)].sort_values("window")
    fig, ax = plt.subplots(figsize=(style.WIDTH_SINGLE, 2.4))
    ax.plot(at_t["prevalence"], at_t["for"], marker="o", color=style.COLORS[0], label="fixed threshold")
    ax.axhline(summary["epsilon"], color=style.GRAY, linestyle="--", linewidth=1, label=r"target $\varepsilon$")
    ax.set_xlabel("attack prevalence")
    ax.set_ylabel("false omission rate")
    ax.legend()
    return fig


def fig_dismiss_vs_for(run_dir) -> plt.Figure:
    style.use()
    curves = runs.load_table(run_dir, "curves")
    fig, ax = plt.subplots(figsize=(style.WIDTH_SINGLE, 2.4))
    for i, (w, g) in enumerate(curves.groupby("window")):
        g = g[g["for"] > 0]
        ax.plot(g["dismiss_rate"], g["for"], color=style.COLORS[i % len(style.COLORS)],
                linestyle=style.DASHES[i % len(style.DASHES)], label=rf"$\pi$ = {g['prevalence'].iloc[0]:.3f}")
    ax.set_xlabel("dismiss rate")
    ax.set_ylabel("false omission rate")
    ax.set_yscale("log")
    ax.legend()
    return fig


FIGURES = {
    "for_by_window": fig_for_by_window,
    "dismiss_vs_for": fig_dismiss_vs_for,
}
