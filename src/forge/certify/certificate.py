"""FORGE-Certify, the dismiss threshold of C1, its online variant, and the i.i.d. Learn then Test baseline.

a flow is dismissed when its score is at or below the threshold. at prevalence pi the false omission rate is
pi m / (pi m + (1 - pi) b), with m the share of attacks dismissed and b the share of benign flows dismissed. it rises
with pi and m and falls with b, so an upper bound on m and a lower bound on b, plugged in at pi_max, bound it for every
pi up to pi_max. the miss rate of a family mix is the mix average of the family miss rates, so over all mixes the rate
is largest for the single family with the largest miss rate, and the largest of the K family bounds covers every mix.

all methods take counts of flows at or below each threshold of a grid t_1 < ... < t_G, so they use the scores only
through their order and give the same result after any increasing transform of the scores. thresholds are tested in
order toward looser ones and the search stops at the first failure. the first threshold whose true rate exceeds
epsilon then passes only if one bound fails at that one threshold, the benign bound or the bound of the family with the
largest true miss rate there. so no correction is needed for the number of thresholds, nor for the number of families,
and every family bound takes all of delta_m. a set of mixes that caps the family shares has its worst mix spread over
several families, and would need delta_m split over them.

the sequence starts at the strictest threshold that could pass with no attack at or below it, not at t_1. with no
misses the miss rate bound of a family is still about ln(1 / delta_m) / n_k, so where b is small the bound stays far
above epsilon whatever the attack scores are. with epsilon 1e-3, pi_max 0.1 and a grid of 200 benign quantiles, a
sequence starting at t_1 stops at once unless every family has more than about 80,000 calibration attacks. the start
depends on the benign counts, so the benign bound must hold for every threshold at once, which the one sided DKW bound
does. given the benign scores and the family sizes the start is fixed and the attack scores are still independent
draws, so the argument above holds as it is. the uniform benign bound also keeps the automation floor
(1 - pi_max) b_lower valid at the chosen threshold, which a bound holding at each threshold separately would not.

the online variant of Theorem 1b certifies one deployment window. its rate is also pi m / D, with D the share of all
flows of the window that are dismissed, attacks included, so pi_max times the largest family bound over a lower bound
on D bounds it for every pi up to pi_max and every mix, whatever the benign scores of the window look like. the lower
bound on D comes from the window's own unlabeled scores through the one sided DKW bound. the window's scores then take
the place of the benign scores above, they fix the start, and the attack scores of D_cal are independent of them, so
the same argument holds with probability at least 1 - delta_m - delta_window. only the attack families must keep their
score distributions between D_cal and the window.
"""

import numpy as np

from forge.certify.bounds import clopper_pearson_upper, dkw_margin

STARTS = ("feasible", "strictest")


def for_from_rates(pi, m, b):
    # false omission rate from the prevalence, the miss rate and the benign dismiss rate, 0 when nothing is dismissed
    num = pi * np.asarray(m, dtype=float)
    den = num + (1 - pi) * np.asarray(b, dtype=float)
    return np.divide(num, den, out=np.zeros(np.broadcast(num, den).shape), where=den > 0)


def for_from_dismissed(pi, m, d):
    # false omission rate from the prevalence, the miss rate and the share of all flows dismissed, attacks included.
    # capped at 1, which is also its value when nothing is dismissed, since bounds put in for m and d can exceed it
    num = pi * np.asarray(m, dtype=float)
    d = np.asarray(d, dtype=float)
    return np.minimum(np.divide(num, d, out=np.ones(np.broadcast(num, d).shape), where=d > 0), 1.0)


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


def _rest_of_delta(delta, part, name):
    # part goes to the bound on dismissed flows, the rest of delta to each family bound
    if not 0 < part < delta:
        raise ValueError(f"{name} needs 0 < {name} < delta, got {part} with delta {delta}")
    return delta - part


def _share_lower(below, n, delta_part):
    # one sided DKW lower bound on the share of n flows at or below each threshold, at every threshold at once
    below = np.asarray(below)
    n = np.asarray(n)
    share = np.divide(below, n[..., None], out=np.zeros(below.shape), where=n[..., None] > 0)
    return np.clip(share - np.asarray(dkw_margin(n, delta_part))[..., None], 0, None)


def _miss_upper(attack_below, n_attack, delta_miss):
    # the largest family bound at each threshold, and the largest bound a family has with no miss, which fixes where
    # a threshold could first pass
    attack_below = np.asarray(attack_below)
    n_attack = np.asarray(n_attack)
    m_upper = clopper_pearson_upper(attack_below, n_attack[..., None], delta_miss).max(-2)
    m_zero = np.asarray(clopper_pearson_upper(np.zeros_like(n_attack), n_attack, delta_miss)).max(-1)
    return m_upper, m_zero


def _walk(bound, bound_with_no_miss, eps, start):
    # index of the certified threshold, -1 when none is certified
    if start not in STARTS:
        raise ValueError(f"start is {start}, expected one of {', '.join(STARTS)}")
    if start == "feasible":
        first = first_index(bound_with_no_miss <= eps)
    else:
        first = np.zeros(bound.shape[:-1], dtype=int)
    return fixed_sequence(bound <= eps, first)


def certify(benign_below, n_benign, attack_below, n_attack, eps, delta, pi_max, delta_benign=None, start="feasible"):
    """FORGE-Certify on counts at or below each grid threshold.

    benign_below has shape (..., G) and attack_below (..., K, G) with one row per attack family, n_benign has shape (...)
    and n_attack (..., K). leading axes are independent calibration sets. with K families the bound covers every family
    mix, with the families summed into one row it covers only the mix of the calibration set. delta_benign goes to the
    benign bound and the rest of delta to each family bound. returns the index of the certified threshold, -1 when none
    is certified, its FOR bound and the automation floor.
    """
    delta_benign = delta / 2 if delta_benign is None else delta_benign
    m_upper, m_zero = _miss_upper(attack_below, n_attack, _rest_of_delta(delta, delta_benign, "delta_benign"))
    b_lower = _share_lower(benign_below, n_benign, delta_benign)
    bound = for_from_rates(pi_max, m_upper, b_lower)
    index = _walk(bound, for_from_rates(pi_max, m_zero[..., None], b_lower), eps, start)
    return {"index": index, "bound": _take(bound, index, np.nan), "floor": (1 - pi_max) * _take(b_lower, index, 0.0)}


def certify_online(window_below, n_window, attack_below, n_attack, eps, delta, pi_max, delta_window=None,
                   start="feasible"):
    """FORGE-Certify on one deployment window, the online variant of Theorem 1b.

    window_below has shape (..., G) and counts the window's flows at or below each threshold whatever their label, out
    of n_window flows, shape (...). attack_below and n_attack are the counts of D_cal as in certify, and they broadcast
    against the leading axes of the window, so one calibration set serves many windows. delta_window goes to the bound
    on the window's dismissed share and the rest of delta to each family bound. returns the index of the certified
    threshold, -1 when none is certified, its FOR bound and the automation floor, a lower bound on the share of the
    window's flows dismissed.
    """
    delta_window = delta / 2 if delta_window is None else delta_window
    m_upper, m_zero = _miss_upper(attack_below, n_attack, _rest_of_delta(delta, delta_window, "delta_window"))
    d_lower = _share_lower(window_below, n_window, delta_window)
    bound = for_from_dismissed(pi_max, m_upper, d_lower)
    index = _walk(bound, for_from_dismissed(pi_max, m_zero[..., None], d_lower), eps, start)
    return {"index": index, "bound": _take(bound, index, np.nan), "floor": _take(d_lower, index, 0.0)}


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


def _family_counts(attack_scores, families, grid):
    # every family present among the attack scores is one vertex of the mix simplex
    attack_scores = np.asarray(attack_scores)
    families = np.asarray(families)
    names = np.unique(families)
    below = np.stack([counts_below(attack_scores[families == f], grid) for f in names])
    return below, np.array([np.sum(families == f) for f in names]), names


def _on_grid(out, grid, names):
    # a refusal returns the threshold minus infinity, so nothing is dismissed
    index = int(out["index"])
    return {"threshold": float(grid[index]) if index >= 0 else -np.inf, "index": index,
            "bound": float(out["bound"]), "floor": float(out["floor"]), "families": names.tolist()}


def certify_scores(benign_scores, attack_scores, families, grid, eps, delta, pi_max, delta_benign=None,
                   start="feasible"):
    attack_below, n_attack, names = _family_counts(attack_scores, families, grid)
    out = certify(counts_below(benign_scores, grid), len(benign_scores), attack_below, n_attack, eps, delta, pi_max,
                  delta_benign, start)
    return _on_grid(out, grid, names)


def certify_online_scores(window_scores, attack_scores, families, grid, eps, delta, pi_max, delta_window=None,
                          start="feasible"):
    # the window's scores carry no labels, the attack scores and their families come from D_cal
    attack_below, n_attack, names = _family_counts(attack_scores, families, grid)
    out = certify_online(counts_below(window_scores, grid), len(window_scores), attack_below, n_attack, eps, delta,
                         pi_max, delta_window, start)
    return _on_grid(out, grid, names)
