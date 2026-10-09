"""one module per experiment.

each module defines run(cfg, run), which computes and saves results, and FIGURES, a dict from figure name to a function that takes a run folder and returns a matplotlib figure. figures are drawn only from saved results, so scripts/plot.py can redraw any of them later without rerunning.
"""

from importlib import import_module


def get(name: str):
    return import_module(f"forge.experiments.{name}")
