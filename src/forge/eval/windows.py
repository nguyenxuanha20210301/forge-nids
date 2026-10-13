"""deployment windows judged on counts at the thresholds of a grid, shared by the experiments of RQ1.

a window is a population of benign flows and attack families, given by the counts of each class at or below every
threshold, a prevalence and a mix of the families. its true false omission rate and dismissed share at a threshold
follow from those counts, and each draw of a calibration set fixes one threshold per method, so a method fails in a
draw when the true rate at its threshold exceeds epsilon. the realized share of attacks among the dismissed flows comes
from a window of finite size drawn from the same population.
"""

import numpy as np

from forge.certify.certificate import counts_below, for_from_rates
from forge.eval.metrics import share_with_interval

# quantiles of the FOR over the certified draw and window pairs, pooled over windows
LOW, HIGH = 0.05, 0.95


def class_counts(score: np.ndarray, code: np.ndarray, grid: np.ndarray, n_families: int) -> dict:
    # counts at or below each threshold of the benign flows, code 0, and of each attack family, codes 1 and up, and
    # the size of each class
    return {"benign_below": counts_below(score[code == 0], grid),
            "family_below": np.stack([counts_below(score[code == k], grid) for k in range(1, n_families + 1)]),
            "sizes": np.bincount(code, minlength=n_families + 1)}


def add_counts(a: dict, b: dict) -> dict:
    return {key: a[key] + b[key] for key in a}


def rates(counts: dict) -> tuple:
    # empirical cdfs at the grid of the benign class, of each attack family and of all attacks, and the mix of the
    # families
    sizes = counts["sizes"]
    n_family = sizes[1:]
    benign = counts["benign_below"] / sizes[0]
    family = np.divide(counts["family_below"], n_family[:, None], out=np.zeros(counts["family_below"].shape),
                       where=n_family[:, None] > 0)
    attack = counts["family_below"].sum(0) / n_family.sum()
    return benign, family, attack, n_family / n_family.sum()


def mix_shares(mix: str, own: np.ndarray, calibration: np.ndarray, families: list[str]) -> np.ndarray:
    # share of each attack family among the attacks of a window, the window's own mix, the mix of the calibration set,
    # or one family alone
    if mix == "window":
        return own
    if mix == "calibration":
        return calibration
    return np.array([f == mix for f in families], dtype=float)


def best_thresholds(pi, pi_max, m_w, b_w, eps) -> dict:
    # the loosest grid threshold whose true FOR in the window stays at or below epsilon at its prevalence, and the
    # loosest that does so up to pi_max, where it is largest since the rate rises with the prevalence, with the share of
    # the window sessions each dismisses. that share grows with the threshold, and -1 dismisses nothing
    share = pi * m_w + (1 - pi) * b_w
    out = {}
    for name, at in [("best_at_prevalence", pi), ("best_up_to_pi_max", pi_max)]:
        safe = np.flatnonzero(for_from_rates(at, m_w, b_w) <= eps)
        index = int(safe[-1]) if len(safe) else -1
        out[f"{name}_index"] = index
        out[f"{name}_dismissed"] = float(share[index]) if index >= 0 else 0.0
    return out


def every_mix_threshold(pi_max, m_family, b, groups, shares, eps) -> int:
    # the loosest grid threshold whose true FOR stays at or below epsilon at pi_max for every mix the simplex claim over
    # the groups covers, each family that stands alone and each merged group at its calibration proportions, the best a
    # method could dismiss that knows the distributions and must cover every mix. -1 when no threshold is safe
    m_family, groups, shares = np.asarray(m_family), np.asarray(groups), np.asarray(shares, dtype=float)
    weights = np.stack([np.where(groups == g, shares, 0.0) for g in np.unique(groups)])
    total = weights.sum(1, keepdims=True)
    m_group = np.divide(weights, total, out=np.zeros(weights.shape), where=total > 0) @ m_family
    safe = np.flatnonzero((for_from_rates(pi_max, m_group, b) <= eps).all(0))
    return int(safe[-1]) if len(safe) else -1


def evaluate(methods, offline, online, pi, m_w, b_w, attack_below, window_below, eps, values, cell) -> list[dict]:
    # one row per method for one window, prevalence and mix. the true rate and dismissed share follow from the cdfs of
    # the window's population at the grid, the realized share of attacks among the dismissed sessions from the drawn
    # window
    true_rate = for_from_rates(pi, m_w, b_w)
    true_share = pi * m_w + (1 - pi) * b_w
    rows = []
    for method in methods:
        out = online if method == "forge_online" else offline[method]
        hit = out["index"] >= 0
        safe = np.maximum(out["index"], 0)
        rate = np.where(hit, true_rate[safe], 0.0)
        share = np.where(hit, true_share[safe], 0.0)
        over = rate > eps
        # a FORGE draw that dismisses less of the window than its floor breaks its claim as well
        under = hit & (share < out["floor"])
        caught = np.take_along_axis(attack_below, safe[:, None], 1)[:, 0]
        dismissed = np.take_along_axis(window_below, safe[:, None], 1)[:, 0]
        realized = hit & (caught > eps * dismissed)
        values[(cell["arm"], method, pi, cell["mix"], cell["window"])] = rate[hit]
        draws = len(hit)
        rows.append({
            "method": method, **cell, "refused": float(np.mean(~hit)), "dismissed_mean": float(share.mean()),
            "benign_dismissed_mean": float(np.where(hit, b_w[safe], 0.0).mean()),
            "miss_mean": float(np.where(hit, m_w[safe], 0.0).mean()),
            "for_median": float(np.median(rate[hit])) if hit.any() else np.nan,
            "violations": int(over.sum()), **share_with_interval(int(over.sum()), draws),
            "floor_violations": int(under.sum()),
            "realized_violations": int(realized.sum()), "realized_rate": float(realized.mean()),
        })
    return rows


def pooled(g, certified: np.ndarray, draws: int) -> dict:
    # the rows of one arm, method, prevalence and mix over a set of windows pooled into one, the FOR quantiles over the
    # certified draw and window pairs, the violation share over all pairs with its range over the windows
    pairs = len(g) * draws
    return {
        "windows": len(g), "pairs": pairs,
        "violation_rate": g["violations"].sum() / pairs, "violation_window_min": g["rate"].min(),
        "violation_window_max": g["rate"].max(), "refused": g["refused"].mean(),
        "dismissed_mean": g["dismissed_mean"].mean(),
        "best_at_prevalence_dismissed": g["best_at_prevalence_dismissed"].mean(),
        "best_up_to_pi_max_dismissed": g["best_up_to_pi_max_dismissed"].mean(),
        "for_median": float(np.median(certified)) if len(certified) else np.nan,
        "for_low": float(np.quantile(certified, LOW)) if len(certified) else np.nan,
        "for_high": float(np.quantile(certified, HIGH)) if len(certified) else np.nan,
        "realized_rate": g["realized_violations"].sum() / pairs,
    }
