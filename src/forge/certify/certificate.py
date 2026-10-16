"""FORGE-Certify, the dismiss threshold of C1, its online variant, and the baselines of RQ1.

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

the sequence does not start at t_1. with no misses the miss rate bound of a family is still about ln(1 / delta_m) / n_k,
so where b is small the bound stays far above epsilon whatever the attack scores are. with epsilon 1e-3, pi_max 0.1
and a grid of 200 benign quantiles, a sequence starting at t_1 stops at once unless every family has more than about
80,000 calibration attacks. the first threshold that could pass with no attack at or below it is the strictest
sensible start, but it passes only with no miss, so a single miss there ends the sequence at once. the start therefore
moves to the loosest threshold that could still pass with up to start_misses misses in every family, as long as it
dismisses at most start_slack more than that first threshold, and stays put where that costs more, which happens when
the attack samples are small and the first threshold is already late. the start depends on the benign counts and the
family sizes only, so the benign bound must hold for every threshold at once, which the one sided DKW bound does.
given the benign scores and the family sizes the start is fixed and the attack scores are still independent draws,
so the argument above holds as it is. the uniform benign bound also keeps the automation floor
(1 - pi_max) b_lower valid at the chosen threshold, which a bound holding at each threshold separately would not.

the online variant of Theorem 1b certifies one deployment window. its rate is also pi m / D, with D the share of all
flows of the window that are dismissed, attacks included, so pi_max times the largest family bound over a lower bound
on D bounds it for every pi up to pi_max and every mix, whatever the benign scores of the window look like. the lower
bound on D comes from the window's own unlabeled scores through the one sided DKW bound. the window's scores then take
the place of the benign scores above, they fix the start, and the attack scores of D_cal are independent of them, so
the same argument holds with probability at least 1 - delta_m - delta_window. only the attack families must keep their
score distributions between D_cal and the window.

the baselines of RQ1 besides Learn then Test claim less. the conformal threshold of Şahin and Mert bounds the miss rate
on average over the calibration draw, and the threshold of common practice reads the rate off a labeled set without
any bound.

Learn then Test at a prevalence estimated on the window is the pooled certificate above with the estimate in place of
pi_max. at a threshold t fixed on D_sel the share of a window's flows at or below it is pi m(t) + (1 - pi) b(t), so
pi = (b(t) - F_w(t)) / (b(t) - m(t)), the binary form of adjusted classify and count, read from the shares of the
calibration draw and the unlabeled window. the plug in puts the shares in. the estimate rises with b and m and
falls with F_w, so an upper bound on b from a two sided DKW event on the benign cdf of the calibration draw, an upper
Clopper-Pearson bound on m(t) and a lower one sided DKW bound on F_w(t) bound the prevalence from above, and the
certificate at that bound covers every prevalence up to it, with the lower side of the same benign event as its
benign bound and the family bound at the rest of delta. it holds under label shift with the calibration mix, since
m(t) is the miss rate of that mix, and breaks where the window's attacks sit elsewhere at t.
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


def _miss_upper(attack_below, n_attack, delta_miss, misses):
    # the largest family bound at each threshold, and for r from 0 to misses the largest bound a family has with r
    # misses, which fixes where a threshold could pass
    if misses < 0:
        raise ValueError(f"start_misses is {misses}, expected 0 or more")
    attack_below = np.asarray(attack_below)
    n_attack = np.asarray(n_attack)
    m_upper = clopper_pearson_upper(attack_below, n_attack[..., None], delta_miss).max(-2)
    m_with = [np.asarray(clopper_pearson_upper(np.full(n_attack.shape, r), n_attack, delta_miss)).max(-1)
              for r in range(misses + 1)]
    return m_upper, m_with


def _start(with_misses, share_lower, eps, start, slack):
    # with_misses[r] is the bound at each threshold when every family has r misses there. the start is the loosest
    # threshold that would pass with up to that many misses, as long as its lower bound on the dismissed share exceeds
    # the one of the first threshold that would pass with no miss by at most slack
    if start not in STARTS:
        raise ValueError(f"start is {start}, expected one of {', '.join(STARTS)}")
    shape = np.broadcast_shapes(*(b.shape for b in with_misses))
    if start == "strictest":
        return np.zeros(shape[:-1], dtype=int)
    g = shape[-1]
    share_lower = np.broadcast_to(share_lower, shape)
    starts = [first_index(b <= eps) for b in with_misses]
    first = starts[0]
    base = _take(share_lower, np.where(first < g, first, -1), np.nan)
    for s in starts[1:]:
        later = _take(share_lower, np.where(s < g, s, -1), np.inf)
        first = np.where((s < g) & (later - base <= slack), s, first)
    return first


def certify(benign_below, n_benign, attack_below, n_attack, eps, delta, pi_max, delta_benign=None, start="feasible",
            start_misses=2, start_slack=0.05):
    """FORGE-Certify on counts at or below each grid threshold.

    benign_below has shape (..., G) and attack_below (..., K, G) with one row per attack family, n_benign has shape (...)
    and n_attack (..., K). leading axes are independent calibration sets. with K families the bound covers every family
    mix, with the families summed into one row it covers only the mix of the calibration set. delta_benign goes to the
    benign bound and the rest of delta to each family bound. start_misses and start_slack set where the sequence starts,
    see above, start_misses 0 starts at the first threshold that could pass with no miss. returns the index of the
    certified threshold, -1 when none is certified, its FOR bound and the automation floor.
    """
    delta_benign = delta / 2 if delta_benign is None else delta_benign
    m_upper, m_with = _miss_upper(attack_below, n_attack, _rest_of_delta(delta, delta_benign, "delta_benign"),
                                  start_misses)
    b_lower = _share_lower(benign_below, n_benign, delta_benign)
    bound = for_from_rates(pi_max, m_upper, b_lower)
    first = _start([for_from_rates(pi_max, m[..., None], b_lower) for m in m_with], b_lower, eps, start, start_slack)
    index = fixed_sequence(bound <= eps, first)
    return {"index": index, "bound": _take(bound, index, np.nan), "floor": (1 - pi_max) * _take(b_lower, index, 0.0)}


def certify_online(window_below, n_window, attack_below, n_attack, eps, delta, pi_max, delta_window=None,
                   start="feasible", start_misses=2, start_slack=0.05):
    """FORGE-Certify on one deployment window, the online variant of Theorem 1b.

    window_below has shape (..., G) and counts the window's flows at or below each threshold whatever their label, out
    of n_window flows, shape (...). attack_below and n_attack are the counts of D_cal as in certify, and they broadcast
    against the leading axes of the window, so one calibration set serves many windows. delta_window goes to the bound
    on the window's dismissed share and the rest of delta to each family bound. the start follows the rule of certify,
    with the window's dismissed share in place of the benign one. returns the index of the certified threshold, -1 when
    none is certified, its FOR bound and the automation floor, a lower bound on the share of the window's flows
    dismissed.
    """
    return online_certifier(attack_below, n_attack, eps, delta, pi_max, delta_window, start, start_misses,
                            start_slack)(window_below, n_window)


def online_certifier(attack_below, n_attack, eps, delta, pi_max, delta_window=None, start="feasible", start_misses=2,
                     start_slack=0.05):
    """certify_online for many windows against the same D_cal.

    the bounds on the family miss rates depend on D_cal alone and take most of the time on a fine grid, so they are
    computed once here. the function returned takes window_below and n_window and returns what certify_online does.
    """
    delta_window = delta / 2 if delta_window is None else delta_window
    m_upper, m_with = _miss_upper(attack_below, n_attack, _rest_of_delta(delta, delta_window, "delta_window"),
                                  start_misses)

    def certify_window(window_below, n_window):
        d_lower = _share_lower(window_below, n_window, delta_window)
        bound = for_from_dismissed(pi_max, m_upper, d_lower)
        first = _start([for_from_dismissed(pi_max, m[..., None], d_lower) for m in m_with], d_lower, eps, start,
                       start_slack)
        index = fixed_sequence(bound <= eps, first)
        return {"index": index, "bound": _take(bound, index, np.nan), "floor": _take(d_lower, index, 0.0)}

    return certify_window


def ltt_iid(benign_below, attack_below, eps, delta, n=None, start_misses=0, start_slack=0.05):
    """the i.i.d. baseline of RQ1, Learn then Test with an exact binomial bound on the share of attacks among the
    dismissed calibration flows.

    attack_below holds all attacks in one row, shape (..., G). the guarantee holds only when deployment traffic keeps
    the prevalence and the family mix of the calibration set. the sequence starts by the rule of certify, with r
    attacks among the dismissed flows in place of r misses per family and the share of the n calibration flows
    dismissed in place of the benign bound. start_misses 0 starts at the first threshold that could pass with no attack,
    and needs no n. the start depends on the attack counts too, so the validity of either start is checked by
    simulation.
    """
    dismissed = np.asarray(benign_below) + np.asarray(attack_below)
    upper = np.asarray(clopper_pearson_upper(attack_below, dismissed, delta))
    if start_misses < 0:
        raise ValueError(f"start_misses is {start_misses}, expected 0 or more")
    if start_misses > 0 and n is None:
        raise ValueError("a start that tolerates attacks compares dismissed shares, which needs the size n of each set")
    share = np.zeros(dismissed.shape) if n is None else dismissed / np.asarray(n)[..., None]
    with_misses = [np.asarray(clopper_pearson_upper(r, dismissed, delta)) for r in range(start_misses + 1)]
    # thresholds that dismiss too few flows to pass even with no attack among them are skipped, as in FORGE-Certify
    first = _start(with_misses, share, eps, "feasible", start_slack)
    index = fixed_sequence(upper <= eps, first)
    return {"index": index, "bound": _take(upper, index, np.nan)}


def conformal_fnr(attack_below, n_attack, beta):
    """the dismiss side of the dual threshold conformal deferral of Şahin and Mert, the baseline of RQ1 with an FNR
    guarantee.

    the threshold is the loosest one with at most floor(beta (n + 1)) - 1 of the n calibration attacks at or below it. a
    new attack exchangeable with them then falls at or below it with probability at most beta, on average over the
    calibration draw. that bounds the miss rate in expectation and says nothing about the FOR. attack_below holds all
    attacks in one row, shape (..., G). returns the index of the threshold, -1 when none qualifies.
    """
    allowed = np.floor(beta * (np.asarray(n_attack) + 1)) - 1
    return fixed_sequence(np.asarray(attack_below) <= np.asarray(allowed)[..., None], 0)


def plug_in(benign_below, n_benign, attack_below, n_attack, eps, pi):
    """the threshold of common practice, the loosest one whose FOR at prevalence pi, estimated on one labeled set, stays
    at or below eps, searched from the strictest threshold to the first failure. there is no bound behind it.

    attack_below holds all attacks in one row. returns the index of the threshold, -1 when the strictest one fails.
    """
    b = np.asarray(benign_below) / np.asarray(n_benign)[..., None]
    m = np.asarray(attack_below) / np.asarray(n_attack)[..., None]
    return fixed_sequence(for_from_rates(pi, m, b) <= eps, 0)


ESTIMATE_RULES = ("ks",)


def ks_threshold(benign_share, attack_share) -> int:
    # the grid index at which the benign share and the attack share at or below the threshold differ most, the
    # point of the Kolmogorov Smirnov statistic, where the denominator of the prevalence estimate is largest
    return int(np.argmax(np.asarray(benign_share, dtype=float) - np.asarray(attack_share, dtype=float)))


def _prevalence_ratio(b, m, f):
    # (b - f) / (b - m) clipped to the unit interval, 1 where the benign share is at or below the attack share and
    # nothing can be told
    b, m, f = (np.asarray(v, dtype=float) for v in (b, m, f))
    den = b - m
    out = np.divide(b - f, den, out=np.ones(np.broadcast(b, m, f).shape), where=den > 0)
    return np.where(den > 0, np.clip(out, 0.0, 1.0), 1.0)


def estimate_prevalence(window_below, n_window, benign_below, n_benign, attack_below, n_attack, delta_benign,
                        delta_window, delta_miss) -> dict:
    """the prevalence of a window from its unlabeled scores at one threshold fixed in advance, as a plug in and as an
    upper confidence bound.

    window_below counts the window's flows at or below the threshold out of n_window, benign_below the benign flows
    of the calibration draw at or below it out of n_benign, attack_below the attacks of the calibration mix out of
    n_attack, all of shape (...). the upper bound takes the benign share up by the margin of a two sided DKW event
    at delta_benign, the miss rate up by the Clopper-Pearson bound at delta_miss and the window share down by the
    one sided DKW margin at delta_window, so it holds with probability at least 1 minus their sum under label shift
    with the calibration mix. returns the plug in and the upper bound, both in [0, 1].
    """
    window_below, n_window, benign_below, n_benign, attack_below, n_attack = (
        np.asarray(v, dtype=float) for v in (window_below, n_window, benign_below, n_benign, attack_below, n_attack))
    b = np.divide(benign_below, n_benign, out=np.zeros(np.broadcast(benign_below, n_benign).shape), where=n_benign > 0)
    m = np.divide(attack_below, n_attack, out=np.zeros(np.broadcast(attack_below, n_attack).shape), where=n_attack > 0)
    f = np.divide(window_below, n_window, out=np.zeros(np.broadcast(window_below, n_window).shape), where=n_window > 0)
    b_upper = np.minimum(b + np.asarray(dkw_margin(n_benign, delta_benign / 2)), 1.0)
    m_upper = np.asarray(clopper_pearson_upper(attack_below, n_attack, delta_miss))
    f_lower = np.clip(f - np.asarray(dkw_margin(n_window, delta_window)), 0.0, None)
    return {"plugin": _prevalence_ratio(b, m, f), "upper": _prevalence_ratio(b_upper, m_upper, f_lower)}


def estimated_certifier(benign_below, n_benign, attack_below, n_attack, eps, delta_benign, delta_miss,
                        start="feasible", start_misses=2, start_slack=0.05):
    """the pooled certificate of certify at a prevalence given per calibration set, for Learn then Test at an
    estimated prevalence.

    benign_below has shape (..., G) and attack_below the same with the attacks of the calibration mix in one row,
    n_benign and n_attack shape (...). the benign bound is the lower side of a two sided DKW event at delta_benign,
    whose upper side serves the estimate, and the family bound takes delta_miss. both depend on the calibration draw
    alone and are computed once. the function returned takes the prevalence of each calibration set, shape (...),
    and returns what certify does at that prevalence.
    """
    attack_below = np.asarray(attack_below)
    n_attack = np.asarray(n_attack)
    m_upper, m_with = _miss_upper(attack_below[..., None, :], n_attack[..., None], delta_miss, start_misses)
    b_lower = _share_lower(benign_below, n_benign, delta_benign / 2)

    def certify_at(pi):
        pi = np.asarray(pi, dtype=float)[..., None]
        bound = for_from_rates(pi, m_upper, b_lower)
        first = _start([for_from_rates(pi, m[..., None], b_lower) for m in m_with], b_lower, eps, start, start_slack)
        index = fixed_sequence(bound <= eps, first)
        return {"index": index, "bound": _take(bound, index, np.nan),
                "floor": (1 - pi[..., 0]) * _take(b_lower, index, 0.0)}

    return certify_at


def min_family_size(eps, pi_max, delta_miss, benign_share) -> int:
    # fewest attacks a family needs for a threshold whose benign lower bound is benign_share to pass with no miss of the
    # family, Proposition 3. with fewer, the bound with no miss, 1 - delta_miss^(1/n), already exceeds the miss rate
    # that the threshold allows
    kappa = eps * (1 - pi_max) / ((1 - eps) * pi_max)
    return int(np.ceil(np.log(1 / delta_miss) / -np.log1p(-kappa * benign_share)))


def family_groups(sizes, minimum) -> np.ndarray:
    """the group of each attack family in the certificate, by a rule that looks at the family sizes only, so it is
    fixed before any calibration score is seen.

    a family with at least minimum attacks forms its own group. the others are pooled into one group, which joins the
    largest family when the pool still has fewer than minimum attacks. groups are numbered from 0.
    """
    sizes = np.asarray(sizes, dtype=float)
    rare = sizes < minimum
    group = np.arange(len(sizes))
    if rare.any():
        group[rare] = np.flatnonzero(rare)[0]
        if sizes[rare].sum() < minimum and not rare.all():
            group[rare] = np.flatnonzero(~rare)[np.argmax(sizes[~rare])]
    return np.unique(group, return_inverse=True)[1]


def group_rows(below, n, groups):
    # counts at or below each threshold and family sizes summed within each group, along the family axis
    below = np.asarray(below)
    n = np.asarray(n)
    groups = np.asarray(groups)
    k = range(groups.max() + 1)
    return (np.stack([below[..., groups == g, :].sum(-2) for g in k], axis=-2),
            np.stack([n[..., groups == g].sum(-1) for g in k], axis=-1))


def counts_below(scores, grid):
    # number of scores at or below each grid threshold
    return np.searchsorted(np.sort(np.asarray(scores)), grid, side="right")


def grid_from(benign_scores, size: int, tail=()) -> np.ndarray:
    # benign scores of the selection set at the levels j / (size + 1) and at the extra levels in tail, each the smallest
    # score whose empirical cdf reaches the level, so every threshold is a score that occurs. ties among the scores
    # merge some levels into one threshold, and a level beyond the last score falls on the last score
    ordered = np.sort(benign_scores)
    levels = np.r_[np.arange(1, size + 1) / (size + 1), np.asarray(tail, dtype=float)]
    positions = np.clip(np.ceil(levels * len(ordered)).astype(int) - 1, 0, len(ordered) - 1)
    return np.unique(ordered[positions])


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
                   start="feasible", start_misses=2, start_slack=0.05):
    attack_below, n_attack, names = _family_counts(attack_scores, families, grid)
    out = certify(counts_below(benign_scores, grid), len(benign_scores), attack_below, n_attack, eps, delta, pi_max,
                  delta_benign, start, start_misses, start_slack)
    return _on_grid(out, grid, names)


def certify_online_scores(window_scores, attack_scores, families, grid, eps, delta, pi_max, delta_window=None,
                          start="feasible", start_misses=2, start_slack=0.05):
    # the window's scores carry no labels, the attack scores and their families come from D_cal
    attack_below, n_attack, names = _family_counts(attack_scores, families, grid)
    out = certify_online(counts_below(window_scores, grid), len(window_scores), attack_below, n_attack, eps, delta,
                         pi_max, delta_window, start, start_misses, start_slack)
    return _on_grid(out, grid, names)
