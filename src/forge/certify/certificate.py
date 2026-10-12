"""FORGE-Certify, the dismiss threshold of C1, and the i.i.d. Learn then Test baseline.

a flow is dismissed when its score is at or below the threshold. at prevalence pi the false omission rate is
pi m / (pi m + (1 - pi) b), with m the share of attacks dismissed and b the share of benign flows dismissed. it rises
with pi and m and falls with b, so an upper bound on m and a lower bound on b, plugged in at pi_max, bound it for every
pi up to pi_max. with one miss rate bound per attack family and the largest of them, the bound covers every family mix,
at the price of splitting delta_m over the K families.

both methods take counts of calibration flows at or below each threshold of a grid t_1 < ... < t_G, so they use the
scores only through their order and give the same result after any increasing transform of the scores. thresholds are
tested in order toward looser ones and the search stops at the first failure. the first threshold whose true rate
exceeds epsilon then passes only if a bound fails at that one threshold, so no correction for the number of thresholds
is needed.

the sequence starts at the strictest threshold that could pass with no attack at or below it, not at t_1. with no
misses the miss rate bound of a family is still about ln(K / delta_m) / n_k, so where b is small the bound stays far
above epsilon whatever the attack scores are. with epsilon 1e-3, pi_max 0.1 and a grid of 200 benign quantiles, a
sequence starting at t_1 stops at once unless every family has more than about 10^5 calibration attacks. the start
depends on the benign counts, so the benign bound must hold for every threshold at once, which the one sided DKW bound
does. given the benign scores and the family sizes the start is fixed and the attack scores are still independent
draws, so the argument above holds as it is. the uniform benign bound also keeps the automation floor
(1 - pi_max) b_lower valid at the chosen threshold, which a bound holding at each threshold separately would not.
"""

import numpy as np

from forge.certify.bounds import clopper_pearson_upper, dkw_margin

STARTS = ("feasible", "strictest")


def for_from_rates(pi, m, b):
    # false omission rate from the prevalence, the miss rate and the benign dismiss rate, 0 when nothing is dismissed
    num = pi * np.asarray(m, dtype=float)
    den = num + (1 - pi) * np.asarray(b, dtype=float)
    return np.divide(num, den, out=np.zeros(np.broadcast(num, den).shape), where=den > 0)


def first_index(mask):
    # position of the first true value along the last axis, the length of that axis when there is none
    mask = np.asarray(mask, dtype=bool)
    return np.where(mask.any(-1), mask.argmax(-1), mask.shape[-1])


def fixed_sequence(passes, start):
    # walks from start toward looser thresholds and returns the last threshold passed before the first failure, -1
    # when the test at start already fails or there is no start
    passes = np.asarray(passes, dtype=bool)
    start = np.asarray(start)
    tested = np.arange(passes.shape[-1]) >= start[..., None]
    last = first_index(tested & ~passes) - 1
    return np.where(last >= start, last, -1)


def _take(values, index, fill):
    picked = np.take_along_axis(values, np.maximum(index, 0)[..., None], axis=-1)[..., 0]
    return np.where(index >= 0, picked, fill)


def certify(benign_below, n_benign, attack_below, n_attack, eps, delta, pi_max, delta_benign=None, start="feasible"):
    """FORGE-Certify on counts at or below each grid threshold.

    benign_below has shape (..., G) and attack_below (..., K, G) with one row per attack family, n_benign has shape (...)
    and n_attack (..., K). leading axes are independent calibration sets. with K families the bound covers every family
    mix, with the families summed into one row it covers only the mix of the calibration set. returns the index of the
    certified threshold, -1 when none is certified, its FOR bound and the automation floor.
    """
    if start not in STARTS:
        raise ValueError(f"start is {start}, expected one of {', '.join(STARTS)}")
    benign_below = np.asarray(benign_below)
    attack_below = np.asarray(attack_below)
    n_benign = np.asarray(n_benign)
    n_attack = np.asarray(n_attack)
    delta_benign = delta / 2 if delta_benign is None else delta_benign
    delta_family = (delta - delta_benign) / n_attack.shape[-1]

    share = np.divide(benign_below, n_benign[..., None], out=np.zeros(benign_below.shape),
                      where=n_benign[..., None] > 0)
    b_lower = np.clip(share - np.asarray(dkw_margin(n_benign, delta_benign))[..., None], 0, None)
    m_upper = clopper_pearson_upper(attack_below, n_attack[..., None], delta_family).max(-2)
    bound = for_from_rates(pi_max, m_upper, b_lower)
    if start == "feasible":
        m_zero = np.asarray(clopper_pearson_upper(np.zeros_like(n_attack), n_attack, delta_family)).max(-1)
        first = first_index(for_from_rates(pi_max, m_zero[..., None], b_lower) <= eps)
    else:
        first = np.zeros(n_benign.shape, dtype=int)
    index = fixed_sequence(bound <= eps, first)
    return {"index": index, "bound": _take(bound, index, np.nan), "floor": (1 - pi_max) * _take(b_lower, index, 0.0)}


def ltt_iid(benign_below, attack_below, eps, delta):
    """the i.i.d. baseline of RQ1, Learn then Test with an exact binomial bound on the share of attacks among the
    dismissed calibration flows.

    attack_below holds all attacks in one row, shape (..., G). the guarantee holds only when deployment traffic keeps
    the prevalence and the family mix of the calibration set.
    """
    dismissed = np.asarray(benign_below) + np.asarray(attack_below)
    upper = np.asarray(clopper_pearson_upper(attack_below, dismissed, delta))
    # thresholds that dismiss too few flows to pass even with no attack among them are skipped, as in FORGE-Certify
    first = first_index(np.asarray(clopper_pearson_upper(0, dismissed, delta)) <= eps)
    index = fixed_sequence(upper <= eps, first)
    return {"index": index, "bound": _take(upper, index, np.nan)}


def counts_below(scores, grid):
    # number of scores at or below each grid threshold
    return np.searchsorted(np.sort(np.asarray(scores)), grid, side="right")


def certify_scores(benign_scores, attack_scores, families, grid, eps, delta, pi_max, delta_benign=None,
                   start="feasible"):
    # every family present among the attack scores is one vertex of the mix simplex. a refusal returns the threshold
    # minus infinity, so nothing is dismissed
    attack_scores = np.asarray(attack_scores)
    families = np.asarray(families)
    names = np.unique(families)
    attack_below = np.stack([counts_below(attack_scores[families == f], grid) for f in names])
    n_attack = np.array([np.sum(families == f) for f in names])
    out = certify(counts_below(benign_scores, grid), len(benign_scores), attack_below, n_attack, eps, delta, pi_max,
                  delta_benign, start)
    index = int(out["index"])
    return {"threshold": float(grid[index]) if index >= 0 else -np.inf, "index": index,
            "bound": float(out["bound"]), "floor": float(out["floor"]), "families": names.tolist()}
