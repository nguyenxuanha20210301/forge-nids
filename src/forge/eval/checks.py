"""the checks of the theorems on synthetic data with a known truth, E0 of the paper, cut from the tables the runs saved.
Theorem 1 on the calibration sets of e10, Theorem 1b on the drifting windows of e14, Theorems 2 and 3 on the streams of
dismissed flows of e20. a check is one run, since its repetitions are draws inside the run, so every table here reads
one run folder and carries the settings the run was checked at.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from forge import runs

E10, E14, E20 = "e10_synthetic_certify", "e14_synthetic_online", "e20_synthetic_monitor"
# the points at which the true count of a stream is kept, enough for the step of a figure at its printed width
TRACE_POINTS = 2000


def _with_settings(table: pd.DataFrame, summary: dict, names: dict) -> pd.DataFrame:
    # the settings of the run as columns, under the names the tables of the paper use
    t = table.copy()
    for column, key in names.items():
        t[column] = summary[key]
    return t.reset_index(drop=True)


CERTIFY_SETTINGS = {"epsilon": "epsilon", "delta": "delta", "pi_max": "pi_max", "calibration_prevalence": "prevalence"}


def certify_by_size(run_dir) -> pd.DataFrame:
    """Theorem 1 by the flows in the calibration set, one row per size and method of e10, with the share of draws whose
    threshold breaks the method's own claim on the FOR, the same with the automation floor, the share breaking the
    claim of Theorem 1 whatever the method claims, each with its 95% interval over the draws, the refusals, and the
    benign share dismissed against the best grid threshold of the claim."""
    run_dir = Path(run_dir)
    summary = runs.load_json(run_dir, "summary")
    t = runs.load_table(run_dir, "claims")
    t.insert(1, "main", t["size"] == summary["main_size"])
    return _with_settings(t, summary, CERTIFY_SETTINGS)


def certify_by_prevalence(run_dir) -> pd.DataFrame:
    """Theorem 1 by the prevalence of deployment at the main size of e10, one row per method, prevalence and family
    mix, the calibration mix, the worst single family as any, and every family alone, with the share of draws whose true
    FOR at that prevalence and mix exceeds epsilon, with its 95% interval."""
    run_dir = Path(run_dir)
    summary = runs.load_json(run_dir, "summary")
    t = runs.load_table(run_dir, "violations")
    return _with_settings(t[t["size"] == summary["main_size"]], summary, CERTIFY_SETTINGS)


def online(run_dir) -> pd.DataFrame:
    """Theorem 1b on the windows of e14 whose benign scores stay, drift or jump, one row per scenario, composition,
    window size, window and method, with the share of draws whose threshold has FOR above epsilon in the window, the
    same with the automation floor, the realized share, the refusals and the share of the window dismissed, with the
    window the benign scores jump at and whether the window size is the main one."""
    run_dir = Path(run_dir)
    summary = runs.load_json(run_dir, "summary")
    t = runs.load_table(run_dir, "claims")
    t.insert(3, "main", t["window_size"] == summary["main_window_size"])
    shift = {name: np.nan if at is None else at for name, at in summary["shift_at"].items()}
    t.insert(1, "shift_at", t["scenario"].map(shift))
    return _with_settings(t, summary, CERTIFY_SETTINGS)


MONITOR_SETTINGS = {"epsilon": "epsilon", "delta": "delta", "delta_lower": "delta_lower", "q_min": "q_min"}


def monitor(run_dir) -> pd.DataFrame:
    """Theorems 2 and 3 on the streams of e20, one row per scenario, audit rate, allocation, kind of sequence and
    label delay, with the share of draws whose true count leaves the bounds at some moment with its 95% interval, the
    alarms, true and false, the delay from the start of the violation to the alarm, the leakage at the switch off
    against the bound of Theorem 3, the certification by audit and the audits spent, with the main policy marked."""
    run_dir = Path(run_dir)
    summary = runs.load_json(run_dir, "summary")
    t = runs.load_table(run_dir, "claims")
    t.insert(4, "main", (t["rate"] == summary["main_rate"]) & (t["allocation"] == "uniform")
             & (t["kind"] == "thinning"))
    return _with_settings(t, summary, MONITOR_SETTINGS)


def monitor_adaptive(run_dir) -> pd.DataFrame:
    """Theorem 2 against the adversary of e20 who reads the past coins and the bounds before placing each attack,
    and with labels of attacks that arrive later than those of benign flows, one row per cell, at the main policy."""
    run_dir = Path(run_dir)
    summary = runs.load_json(run_dir, "summary")
    t = runs.load_table(run_dir, "adaptive")
    t.insert(1, "audit_rate", summary["main_rate"])
    return _with_settings(t, summary, MONITOR_SETTINGS)


def monitor_trace(run_dir) -> pd.DataFrame:
    """one stream of e20 at the main policy, the scenario its config names for the trace, in two kinds of rows, the
    audited attacks, where the bounds move, with the true count and both bounds there, and the true count at evenly
    spaced points of the stream. the start of the violation, the alarm and the end of the stream come as columns."""
    run_dir = Path(run_dir)
    summary = runs.load_json(run_dir, "summary")
    t = runs.load_arrays(run_dir, "trace")
    valid = t["valid"]
    audits = pd.DataFrame({"point": "audit", "position": t["ev_pos"][valid], "attacks": t["ev_g"][valid],
                           "lower": t["lcb"][valid], "upper": t["ucb"][valid], "q": t["ev_q"][valid]})
    n_end = int(t["n_end"])
    # the count rises by one at every attack, read at a grid that ends on the last flow
    grid = np.unique(np.linspace(0, n_end, min(TRACE_POINTS, n_end) + 1).round().astype(np.int64))
    counts = pd.DataFrame({"point": "count", "position": grid,
                           "attacks": np.searchsorted(np.sort(t["attacks"]), grid, side="right")})
    table = pd.concat([audits, counts], ignore_index=True)
    table.insert(0, "scenario", summary["trace"])
    table["n_end"] = n_end
    table["violation_start"] = int(t["violation_start"])
    table["alarm"] = int(t["alarm_time"])
    table["audit_rate"] = float(t["rate"])
    return _with_settings(table, summary, {"epsilon": "epsilon", "delta": "delta"})
