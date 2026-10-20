"""the start rules of the certificate on the calibration parts of the e13 runs over scorer seeds, the table of the
paper that says why the sequence on UGR16 starts at a share of the benign flows.

the fixed sequence certifies the whole path from its start, so a family whose benign-like tail puts some of its
flows among the lowest benign scores fails the test at a strict start, while looser thresholds pass, the same misses
weighing less against a larger benign share. for every run the table recomputes from the class counts of the
calibration part at the grid, with the certificate's own functions, where each start rule begins, the misses of every
family there against the count its bound allows, the thresholds that pass the bound and the loosest of them, and the
threshold certified, for FORGE and for LTT. the rules are the feasible start of the run's config, the feasible start
moved by more misses with the slack lifted, and the share start at a range of shares.

two calibration sets per run. the expected one holds every benign flow of the calibration part and every family at its
expected count, with its attacks at or below each threshold at their expected share, so its counts are not whole
numbers, which the bounds take as they are. the redrawn ones are as many sets as the run drew, from the seed stream
of the run's calibration sets, so they are the run's own sets where numpy draws the same streams and other sets of the
same law elsewhere.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from forge import runs
from forge.certify.bounds import clopper_pearson_upper
from forge.certify.certificate import _miss_upper, _share_lower, _start, fixed_sequence, for_from_rates, group_rows
from forge.data.synthetic import draw_counts
from forge.eval.windows import HIGH, LOW, rates

# the feasible start moved by these many misses with the slack lifted, and the share starts
MISSES = (5, 10, 20, 50, 100, 200)
SHARES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
# a slack that lets the moved start rise to any benign share
LIFTED = 1.0
# the draws whose starts are found together, so the bounds of every miss count up to the largest stay small in memory
CHUNK = 100


def rules(summary: dict, misses=MISSES, shares=SHARES) -> list[dict]:
    """the start rules of the table in the form start_settings gives, each with its name, the feasible start of the
    run's config first, then that start moved by more misses with the slack lifted, then the share starts."""
    base = {"start_misses": int(summary["start_misses"]), "start_slack": float(summary["start_slack"])}
    out = [{"rule": "feasible", "start": "feasible", **base}]
    out += [{"rule": f"misses {r}", "start": "feasible", "start_misses": int(r), "start_slack": LIFTED}
            for r in misses]
    out += [{"rule": f"share {s}", "start": "share", **base, "start_share": float(s)} for s in shares]
    return out


def operating(rule: dict, summary: dict) -> bool:
    # whether a rule is the start the run itself took
    if rule["start"] != summary.get("start", "feasible"):
        return False
    if rule["start"] == "share":
        return np.isclose(rule["start_share"], summary["start_share"])
    return rule["start_misses"] == summary["start_misses"] and np.isclose(rule["start_slack"], summary["start_slack"])


def allowed_misses(n, rate: float, delta: float) -> int:
    # the most misses among n attacks whose Clopper-Pearson bound at delta stays at or below rate, -1 when even no
    # miss does. the bound lies above the share of misses, so no count above rate times n can pass
    if not rate > 0:
        return -1
    r = np.arange(int(np.floor(rate * n)) + 1)
    ok = np.flatnonzero(np.asarray(clopper_pearson_upper(r, np.full(len(r), n), delta)) <= rate)
    return int(ok[-1]) if len(ok) else -1


def needed_flows(r: int, eps: float, delta: float) -> int:
    """the fewest dismissed flows with which r attacks among them pass the Clopper-Pearson bound of LTT at eps. the
    bound falls as the dismissed count grows, so a threshold passes with r attacks exactly where its dismissed count
    reaches this one, which stands for the bound of LTT with r attacks at every threshold of every draw."""
    lo, hi = r + 1, 2 * (r + 1)
    while clopper_pearson_upper(r, hi, delta) > eps:
        lo, hi = hi, 2 * hi
    # the bound passes at hi and not below lo
    while lo < hi:
        mid = (lo + hi) // 2
        if clopper_pearson_upper(r, mid, delta) <= eps:
            hi = mid
        else:
            lo = mid + 1
    return int(lo)


def _forge_starts(m_with, b_lower, eps, pi_max, rule) -> np.ndarray:
    # the start of every draw under one rule, as certify finds it, the draws taken a chunk at a time
    r = rule["start_misses"] if rule["start"] == "feasible" else 0
    out = np.empty(b_lower.shape[:-1], dtype=int)
    for lo in range(0, len(out), CHUNK):
        c = slice(lo, lo + CHUNK)
        bounds = [for_from_rates(pi_max, m[c][..., None], b_lower[c]) for m in m_with[:r + 1]]
        out[c] = _start(bounds, b_lower[c], eps, rule["start"], rule["start_slack"], rule.get("start_share"))
    return out


def _ltt_starts(dismissed, n, eps, delta, rule, needed: dict) -> np.ndarray:
    # the start of every draw under one rule, as ltt_iid finds it, with the bound of r attacks among the dismissed
    # flows passing where the dismissed count reaches the flows it needs
    share = dismissed / np.asarray(n)[:, None]
    r = rule["start_misses"] if rule["start"] == "feasible" else 0
    out = np.empty(len(dismissed), dtype=int)
    for lo in range(0, len(out), CHUNK):
        c = slice(lo, lo + CHUNK)
        bounds = [np.where(dismissed[c] >= needed[k], 0.0, 1.0) for k in range(r + 1)]
        out[c] = _start(bounds, share[c], eps, rule["start"], rule["start_slack"], rule.get("start_share"))
    return out


def _dismissed(index: np.ndarray, benign: np.ndarray, prefix: str) -> dict:
    # the refusal share and the benign share of the calibration part dismissed at the certified thresholds, the mean
    # over the certified sets as the run's summary has it, and the median and range over them
    hit = index >= 0
    shares = benign[index[hit]]
    return {f"{prefix}_refused": float(np.mean(~hit)),
            f"{prefix}_dismissed_mean": float(shares.mean()) if hit.any() else np.nan,
            f"{prefix}_dismissed_median": float(np.median(shares)) if hit.any() else np.nan,
            f"{prefix}_dismissed_low": float(np.quantile(shares, LOW)) if hit.any() else np.nan,
            f"{prefix}_dismissed_high": float(np.quantile(shares, HIGH)) if hit.any() else np.nan}


def run_table(run_dir, misses=MISSES, shares=SHARES) -> pd.DataFrame:
    """one row per start rule for one e13 run, on its expected calibration set and on its redrawn sets."""
    run_dir = Path(run_dir)
    hist = runs.load_arrays(run_dir, "histograms")
    summary = runs.load_json(run_dir, "summary")
    seed = int(runs.load_run_config(run_dir).get("seed", 0))
    eps, delta, pi_max, part_b = summary["epsilon"], summary["delta"], summary["pi_max"], summary["delta_benign"]
    part_m = delta - part_b
    kappa = eps * (1 - pi_max) / ((1 - eps) * pi_max)
    pi_cal, n_cal, draws = summary["calibration_prevalence"], int(summary["calibration_sessions"]), summary["draws"]
    shares_cal, groups = hist["shares"], hist["groups"]
    names = ["+".join(g) for g in summary["families"]]
    b_cal, m_cal, _, _ = rates({"benign_below": hist["calibration_benign_below"],
                                "family_below": hist["calibration_family_below"],
                                "sizes": hist["calibration_sizes"]})
    the_rules = rules(summary, misses, shares)
    most = max(r["start_misses"] for r in the_rules)

    # the expected calibration set, one set with the benign flows of the part and each family at its expected count
    n_benign = float(hist["calibration_sizes"][0])
    expected = n_cal * pi_cal * shares_cal
    g_below, g_n = group_rows((m_cal * expected[:, None])[None], expected[None], groups)
    e_upper, e_with = _miss_upper(g_below, g_n, part_m, most)
    e_lower = _share_lower((b_cal * n_benign)[None], np.array([n_benign]), part_b)
    e_passes = for_from_rates(pi_max, e_upper, e_lower) <= eps
    passing = np.flatnonzero(e_passes[0])
    by_group = np.stack([clopper_pearson_upper(g_below[0, k], g_n[0, k], part_m) for k in range(len(names))])

    # the redrawn sets from the stream of the run's calibration sets, the first child of its seed
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    cal = draw_counts(rng, n_cal, pi_cal, shares_cal, b_cal, m_cal, draws)
    c_below, c_n = group_rows(cal.family_below, cal.n_family, groups)
    c_upper, c_with = _miss_upper(c_below, c_n, part_m, most)
    c_lower = _share_lower(cal.benign_below, cal.n_benign, part_b)
    c_passes = for_from_rates(pi_max, c_upper, c_lower) <= eps
    pooled_below = cal.family_below.sum(1)
    dismissed = cal.benign_below + pooled_below
    n_ltt = cal.n_benign + cal.n_family.sum(1)
    ltt_passes = np.asarray(clopper_pearson_upper(pooled_below, dismissed, delta)) <= eps
    needed = {r: needed_flows(r, eps, delta) for r in range(most + 1)}

    out = []
    g = len(b_cal)
    for rule in the_rules:
        start = int(_forge_starts(e_with, e_lower, eps, pi_max, rule)[0])
        at = start < g
        index = int(fixed_sequence(e_passes, np.array([start]))[0])
        row = {"seed": seed, "run_id": run_dir.name, **{k: rule.get(k, np.nan) for k in
                                                         ["rule", "start", "start_misses", "start_slack",
                                                          "start_share"]},
               "operating": bool(operating(rule, summary)), "thresholds": g,
               "start_index": start if at else -1, "start_benign": float(b_cal[start]) if at else np.nan,
               "start_benign_lower": float(e_lower[0, start]) if at else np.nan,
               "binding": names[int(np.argmax(by_group[:, start]))] if at else None}
        for k, name in enumerate(names):
            row[f"misses_{name}"] = float(g_below[0, k, start]) if at else np.nan
            row[f"allowed_{name}"] = (allowed_misses(g_n[0, k], kappa * e_lower[0, start], part_m) if at
                                      else np.nan)
        row.update({"passing": int(len(passing)),
                    "first_passing_benign": float(b_cal[passing[0]]) if len(passing) else np.nan,
                    "loosest_passing_benign": float(b_cal[passing[-1]]) if len(passing) else np.nan,
                    "expected_certified_benign": float(b_cal[index]) if index >= 0 else np.nan,
                    "draws": int(draws)})
        forge = fixed_sequence(c_passes, _forge_starts(c_with, c_lower, eps, pi_max, rule))
        ltt = fixed_sequence(ltt_passes, _ltt_starts(dismissed, n_ltt, eps, delta, rule, needed))
        row.update(_dismissed(forge, b_cal, "forge"))
        row.update(_dismissed(ltt, b_cal, "ltt"))
        # the run's own refusal shares and certified share beside the rule it took, from its summary
        mine = row["operating"]
        row["run_forge_refused"] = summary["offline_refused"]["forge_simplex"] if mine else np.nan
        row["run_forge_dismissed_mean"] = summary["forge_calibration"]["benign_dismissed"] if mine else np.nan
        row["run_ltt_refused"] = summary["offline_refused"].get("ltt_iid", np.nan) if mine else np.nan
        out.append(row)
    return pd.DataFrame(out)


def start_rules_table(folders: dict, misses=MISSES, shares=SHARES) -> pd.DataFrame:
    """the start rules over the seeds, folders mapping a seed to an e13 run folder, one row per seed and rule."""
    if not folders:
        raise ValueError("no e13 run folder to read the start rules from")
    parts = [run_table(folder, misses, shares) for _, folder in sorted(folders.items())]
    return pd.concat(parts, ignore_index=True)
