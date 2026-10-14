"""audit policies, the probability with which a dismissed flow goes to an analyst.

a policy fixes q for each flow before its coin is flipped, from its score bin at most, with a floor q_min, so the
monitor knows every q_i. uniform audits take one q. score dependent audits follow Proposition 5, q proportional to the
square root of r, the attack share among the dismissed flows of a score bin estimated on the selection set, which
minimizes the variance of the Horvitz Thompson estimate at a given expected number of audits. the allocation changes
how fast a violation is found, never whether the bounds hold.
"""

import numpy as np


def allocate(r, mass, budget, q_min, power):
    """audit probability per score bin, proportional to r to the given power, scaled to the budget.

    mass is the share of the dismissed flows in each bin, budget the expected audits per dismissed flow. a bin whose
    probability falls below q_min takes q_min and one above 1 takes 1, and the scale is found so that the budget is
    met exactly over the bins in between, by bisection since the mean rises with the scale. power 0 gives uniform
    audits, 0.5 the allocation of Proposition 5 and 1 audits in proportion to the attack share.
    """
    r = np.asarray(r, dtype=float)
    mass = np.asarray(mass, dtype=float)
    if r.shape != mass.shape or (r < 0).any() or (mass < 0).any():
        raise ValueError("r and mass need the same shape and no negative entry")
    if not 0 < q_min <= budget <= 1:
        raise ValueError(f"the budget needs q_min <= budget <= 1, got budget {budget} and q_min {q_min}")
    mass = mass / mass.sum()
    base = np.where(r > 0, r, 0.0) ** power

    def mean_q(scale):
        return float(np.sum(mass * np.clip(scale * base, q_min, 1.0)))

    if base.max() == 0 or mean_q(0.0) >= budget:
        return np.full(r.shape, q_min)
    lo, hi = 0.0, 1.0 / base.max()
    while mean_q(hi) < budget:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if mean_q(mid) < budget:
            lo = mid
        else:
            hi = mid
    return np.clip(hi * base, q_min, 1.0)


def bin_profile(bins: int, ratio: float):
    # a profile of attack shares over score bins of equal mass, rising by the factor ratio from the lowest scores
    # toward the threshold, as the share of all attacks in each bin
    weights = ratio ** np.arange(bins, dtype=float)
    return weights / weights.sum()
