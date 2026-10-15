"""E7, the workload of each method at the same safety level, read from the runs of E1, E3 and E6.

the workload of a dismissal policy is what the analysts handle, the escalated flows and the audited ones, per
million flows. the certification stage gives the escalations of each method on the windows of RQ1 with its
violation share, which says whether the method is safe at the level epsilon and delta, and the monitoring stage
gives the escalations, the audits and the attacks let through on the real streams, where the monitor switches a
violating policy off and everything after the switch off is escalated. a policy is at the same safety level as FORGE
once the monitor rides on it, and the price shows in its workload.
"""

import numpy as np
import pandas as pd

from forge import runs

MILLION = 1e6
# the yardsticks of the certification stage as policies of their own, safe by construction
YARDSTICKS = [("best_up_to_pi_max", "best threshold up to pi_max"), ("best_at_prevalence", "best threshold at pi"),
              ("best_every_mix", "best threshold for every mix")]


def certification(run_dir, audit_rate: float) -> pd.DataFrame:
    """the escalations per million flows of every method on the windows RQ1 is read on, with its violation share, at
    every prevalence and mix, from an e11 or e12 run, and the yardsticks beside the methods. the audits of a monitor
    riding on the policy are added at the audit rate on the dismissed flows."""
    summary = runs.load_json(run_dir, "summary")
    table = runs.load_table(run_dir, "by_prevalence")
    dataset = summary["dataset"]
    if "subset" in table:
        rq1 = table[(table["arm"] == "reference") & (table["subset"] == "in_claim")]
    else:
        rq1 = table[(table["arm"] == "reference") & (table["period"] == summary["main_period"])]
    main = summary["main_prevalence"]
    rows = []
    for r in rq1.itertuples():
        rows.append({"dataset": dataset, "stage": "certification", "policy": r.method, "mix": r.mix,
                     "prevalence": r.prevalence, "main": r.prevalence == main, "dismissed": r.dismissed_mean,
                     "violation_rate": r.violation_rate, "refused": r.refused,
                     "safe": r.violation_rate <= summary["delta"]})
    first = rq1["method"].iloc[0] if len(rq1) else None
    for r in rq1[rq1["method"] == first].itertuples():
        for name, label in YARDSTICKS:
            share = getattr(r, f"{name}_dismissed", np.nan)
            if np.isfinite(share):
                rows.append({"dataset": dataset, "stage": "certification", "policy": name, "mix": r.mix,
                             "prevalence": r.prevalence, "main": r.prevalence == main, "dismissed": share,
                             "violation_rate": 0.0, "refused": 0.0, "safe": True})
    out = pd.DataFrame(rows)
    out["escalations_per_million"] = (1 - out["dismissed"]) * MILLION
    out["audits_per_million"] = audit_rate * out["dismissed"] * MILLION
    out["workload_per_million"] = out["escalations_per_million"] + out["audits_per_million"]
    return out


def monitoring(run_dir, control=None) -> pd.DataFrame:
    """the escalations, the audits and the attacks let through per million flows of every policy under each
    monitoring arm on the real streams of an e21 or e23 run, per composition. for e23 the streams of the control,
    with every family known, at the typical composition stand for the deployment, unless control names a unit, whose
    streams with its family unseen are read instead."""
    summary = runs.load_json(run_dir, "summary")
    claims = runs.load_table(run_dir, "claims")
    blocks = runs.load_table(run_dir, "blocks")
    if "unit" in claims:
        dataset = summary["source"]["dataset"]
        unit, case = (summary["control"], "known") if control is None else (control, "unseen")
        if control is None:
            unit = claims["unit"].iloc[0]
        claims = claims[(claims["unit"] == unit) & (claims["case"] == case)]
        blocks = blocks[(blocks["unit"] == unit) & (blocks["case"] == case)]
    else:
        dataset = "anoshift"
    sessions = blocks.drop_duplicates(["composition", "window"]).groupby("composition")["window_sessions"].sum()
    rows = []
    for r in claims.itertuples():
        total = float(sessions[r.composition])
        dismissed_share = r.dismissed_share_mean
        rows.append({"dataset": dataset, "stage": "monitoring", "composition": r.composition, "policy": r.policy,
                     "arm": r.arm, "sessions": total, "dismissed": dismissed_share,
                     "escalations_per_million": (1 - dismissed_share) * MILLION,
                     "audits_per_million": r.audits_mean / total * MILLION,
                     "leak_median": r.leak_median, "leaked_per_million": r.leak_median / total * MILLION,
                     "realized_for_median": r.realized_for_median, "alarm_rate": r.alarm_rate,
                     "excess_median": getattr(r, "excess_median", np.nan)})
    out = pd.DataFrame(rows)
    out["workload_per_million"] = out["escalations_per_million"] + out["audits_per_million"]
    return out


def frontier(table: pd.DataFrame, cost: str, risk: str) -> pd.DataFrame:
    # the rows no other row beats on both the cost and the risk, the Pareto frontier
    t = table.sort_values([risk, cost]).reset_index(drop=True)
    best = np.inf
    keep = []
    for r in t.itertuples():
        keep.append(getattr(r, cost) < best)
        best = min(best, getattr(r, cost))
    return t[keep].reset_index(drop=True)
