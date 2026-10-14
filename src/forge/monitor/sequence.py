"""time uniform bounds on the number of attacks among the dismissed flows, from audits with known probabilities.

each dismissed flow i is audited with a probability q_i fixed before its coin is flipped, from its score at most, and
only an audited flow shows its label. the target is the realized count L_n of attacks among the first n dismissed
flows, whatever the sequence of flows and labels, which may depend on the past coins but not on the current one.

the construction. with one bet lam in [0, 1] the product over the dismissed flows of 1 + lam Y_i (A_i / q_i - 1) is
a nonnegative martingale, since each factor has conditional mean one given the past and the label. an unaudited attack
contributes the factor 1 - lam whatever its q_i, an audited one 1 + lam (1 / q_i - 1), and a benign flow nothing. so
the product equals (1 - lam) to the power L_n - K_n times the product over the audited attacks, with K_n the number of
audited attacks, and depends on the unknown labels through L_n only. by Ville's inequality the product at the true L_n
stays below 1 / delta at every n at once with probability at least 1 - delta, so every L at which it reaches 1 / delta
is ruled out. it falls with L, so the lower bound is the smallest L not ruled out. a bet of the opposite sign, no
larger than q_min / (1 - q_min), gives a product that rises with L and the upper bound. a mixture of bets with fixed
weights is a martingale too, which gives one sequence that adapts to the size of the violation. the bet must be the
same for every flow of a sequence, a bet that changes with the data would make the product over the unaudited attacks
depend on where they sit, which the labels do not show.

the bound of Freedman type in the proposal is kept for comparison. it uses the Horvitz Thompson estimate and an
exponential supermartingale of the increments Y_i (A_i / q_i - 1), which lie above -1 and below 1 / q_min - 1, with
the conditional variance bounded through L_n (1 - q_min) / q_min. with the same mixture over bets its log wealth is
linear in L as well, so both kinds share one search for the boundary.

in both kinds the log wealth of a bet at a hypothesis L is a + b L, with b of one sign for every bet of a side, so the
mixture is monotone in L and the boundary is found by bisection on the integers.
"""

import numpy as np

KINDS = ("thinning", "freedman")
SIDES = ("lower", "upper")
# the largest bet of the lower side. a bet of one would put everything on the audited attacks being all of them
MAX_BET = 0.9


def bets(kind, side, ratios, q, q_low):
    """the bets of a mixture, one per size of violation in ratios, each in units of epsilon.

    a bet of the lower side is growth optimal against a stream whose FOR is ratio times epsilon when the audit
    probability is q, so the mixture is within a constant factor of the best bet whatever the violation turns out to
    be. the upper side bets against streams below epsilon and must respect the floor q_low of the audit probabilities,
    which keeps every factor of the product nonnegative. the freedman kind uses the floor on both sides, since its
    increments and its variance are bounded through the floor.
    """
    if kind not in KINDS:
        raise ValueError(f"kind is {kind}, expected one of {', '.join(KINDS)}")
    if side not in SIDES:
        raise ValueError(f"side is {side}, expected one of {', '.join(SIDES)}")
    if not 0 < q_low <= q < 1:
        raise ValueError(f"the audit probabilities need 0 < q_low <= q < 1, got q {q} and q_low {q_low}")
    r = np.asarray(ratios, dtype=float)
    if side == "lower":
        if (r <= 1).any():
            raise ValueError("a bet of the lower side needs a ratio above one")
        if kind == "thinning":
            return np.minimum(q * (r - 1) / (1 - q), MAX_BET)
        return q_low * np.log(r) / (1 - q_low)
    if (r < 0).any() or (r >= 1).any():
        raise ValueError("a bet of the upper side needs a ratio from zero up to below one")
    if kind == "thinning":
        return q_low * (1 - r) / (1 - q_low)
    return np.log1p((1 - r) * q_low / (1 - q_low))


def gains(side, bet, q):
    # log factor of one audited attack with audit probability q under the thinning kind, shape (..., J). the largest
    # bet of the upper side at the floor gives minus infinity, that bet is then out of the mixture
    q = np.asarray(q, dtype=float)[..., None]
    bet = np.asarray(bet, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log1p(bet * (1 / q - 1)) if side == "lower" else np.log1p(-bet * (1 / q - 1))


def coefficients(kind, side, bet, k, stat, q_low):
    """a and b of the log wealth a + b L of each bet at a hypothesis L.

    k is the number of audited attacks, shape (...). for the thinning kind stat holds the summed log factors of the
    audited attacks per bet, shape (..., J), for the freedman kind the Horvitz Thompson estimate, shape (...).
    """
    bet = np.asarray(bet, dtype=float)
    k = np.asarray(k, dtype=float)[..., None]
    if kind == "thinning":
        stat = np.asarray(stat, dtype=float)
        slope = np.log1p(-bet) if side == "lower" else np.log1p(bet)
        return stat - k * slope, np.broadcast_to(slope, stat.shape)
    l_hat = np.asarray(stat, dtype=float)[..., None]
    c = 1 / q_low - 1
    if side == "lower":
        # Bennett, increments at most c above zero with the variance at most c per attack
        phi = (np.exp(bet * c) - 1 - bet * c) / c ** 2
        a = bet * l_hat
        return a, np.broadcast_to(-(bet + phi * c), a.shape)
    # increments of the opposite sign at most one
    slope = bet - (np.exp(bet) - 1 - bet) * c
    if (slope <= 0).any():
        raise ValueError("an upper bet of the freedman kind is too large for the floor of the audit probabilities")
    a = -bet * l_hat
    return a, np.broadcast_to(slope, a.shape)


def _value_and_slope(c, b, hypothesis):
    # log of the mixture wealth less the level at a hypothesis, with its slope. c holds the log weights and a less the
    # level per bet, so the root of the value is the boundary. a dead bet has c of minus infinity and weighs nothing
    z = c + b * np.asarray(hypothesis, dtype=float)[..., None]
    m = z.max(-1)
    e = np.exp(z - m[..., None])
    total = e.sum(-1)
    return m + np.log(total), (e * b).sum(-1) / total


def _root(c, b, start, iterations: int = 100):
    # Newton's method on the convex, monotone value from a start on the side where the value is at or above zero.
    # the tangent of a convex function lies below it, so the iterates move toward the root and never cross it
    hypothesis = np.asarray(start, dtype=float)
    for _ in range(iterations):
        value, slope = _value_and_slope(c, b, hypothesis)
        step = value / slope
        hypothesis = hypothesis - step
        if (np.abs(step) <= 1e-9 * np.maximum(1.0, np.abs(hypothesis))).all():
            break
    return hypothesis


def _settle(c, b, candidate, below, limit, up: bool):
    # moves the integer candidate until it is the first integer past the root in the direction of up and the one
    # before it is not, within the limit on the far side. the value is monotone, so both loops end
    for _ in range(10_000):
        over = _value_and_slope(c, b, candidate)[0] >= 0
        if not over.any():
            break
        candidate = candidate + np.where(over, 1 if up else -1, 0)
    for _ in range(10_000):
        inside = (candidate != limit) & (_value_and_slope(c, b, candidate - (1 if up else -1))[0] < 0)
        if not inside.any():
            break
        candidate = candidate - np.where(inside, 1 if up else -1, 0)
    return np.where(below, limit, candidate)


def lower_bound(a, b, weights, k, level):
    """smallest L at or above k at which the mixture wealth stays below exp(level), the wealth falling with L.

    a and b have shape (..., J), k shape (...), and level is log(1 / delta).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if (b >= 0).any():
        raise ValueError("the lower side needs slopes below zero")
    c = np.log(np.asarray(weights, dtype=float)) + a - level
    k = np.asarray(k, dtype=np.int64)
    # the mixture is at or above the level wherever one bet alone is, so the root lies past the last single root
    start = np.maximum((-c / b).max(-1), k)
    root = _root(c, b, start)
    candidate = np.maximum(np.floor(root).astype(np.int64) + 1, k)
    below = _value_and_slope(c, b, k)[0] < 0
    return _settle(c, b, candidate, below, k, up=True)


def upper_bound(a, b, weights, k, level, cap):
    """largest L from k up to cap at which the mixture wealth stays below exp(level), the wealth rising with L.

    cap is the number of dismissed flows, which the count of attacks cannot exceed. a bet whose a is minus infinity is
    out, and with no bet left nothing is ruled out.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if (b <= 0).any():
        raise ValueError("the upper side needs slopes above zero")
    c = np.log(np.asarray(weights, dtype=float)) + a - level
    k = np.asarray(k, dtype=np.int64)
    cap = np.broadcast_to(np.asarray(cap, dtype=np.int64), k.shape)
    if (cap < k).any():
        raise ValueError("the cap lies below the number of audited attacks")
    live = np.isfinite(c).any(-1)
    if not live.any():
        return cap.copy()
    c = np.where(live[..., None], c, 0.0)
    with np.errstate(invalid="ignore"):
        single = np.where(np.isfinite(c), -c / b, np.inf).min(-1)
    start = np.minimum(np.maximum(single, k), cap)
    root = _root(c, b, start)
    candidate = np.minimum(np.ceil(root).astype(np.int64) - 1, cap)
    candidate = np.maximum(candidate, k)
    at_cap = _value_and_slope(c, b, cap)[0] < 0
    out = _settle(c, b, candidate, at_cap, cap, up=False)
    return np.where(live, out, cap)


class Sequence:
    """one confidence sequence on the count of attacks among the dismissed flows.

    q is the mean audit probability of the policy, q_low its smallest one, and the two sides take their own delta. the
    statistics of the audited attacks are cumulated along the event axis by statistic, and lower and upper give the
    bounds after each event. tables gives both bounds by the number of audited attacks when every audit has the same
    probability, where the bounds depend on nothing else.
    """

    def __init__(self, kind, lower_ratios, upper_ratios, q, q_low, delta_lower, delta_upper):
        if not 0 < delta_lower < 1 or not 0 < delta_upper < 1:
            raise ValueError(f"the deltas need to lie in (0, 1), got {delta_lower} and {delta_upper}")
        self.kind = kind
        self.q = q
        self.q_low = q_low
        self.bet = {side: bets(kind, side, ratios, q, q_low)
                    for side, ratios in (("lower", lower_ratios), ("upper", upper_ratios))}
        self.weights = {side: np.full(len(self.bet[side]), 1 / len(self.bet[side])) for side in SIDES}
        self.level = {"lower": np.log(1 / delta_lower), "upper": np.log(1 / delta_upper)}

    def statistic(self, side, q_events, valid):
        """the statistic of each side after each event, cumulated along the last axis of q_events.

        q_events holds the audit probability of each audited attack, valid marks the events that exist, the rest are
        padding. the thinning kind returns the summed log factors per bet, shape (..., E, J), the freedman kind the
        Horvitz Thompson estimate, shape (..., E).
        """
        q_events = np.asarray(q_events, dtype=float)
        valid = np.asarray(valid, dtype=bool)
        self._check(q_events[valid])
        if self.kind == "thinning":
            return np.cumsum(np.where(valid[..., None], gains(side, self.bet[side], q_events), 0.0), axis=-2)
        return np.cumsum(np.where(valid, 1 / q_events, 0.0), axis=-1)

    def lower(self, k, stat):
        a, b = coefficients(self.kind, "lower", self.bet["lower"], k, stat, self.q_low)
        return lower_bound(a, b, self.weights["lower"], k, self.level["lower"])

    def upper(self, k, stat, cap):
        a, b = coefficients(self.kind, "upper", self.bet["upper"], k, stat, self.q_low)
        return upper_bound(a, b, self.weights["upper"], k, self.level["upper"], cap)

    def _check(self, q):
        # a bet of the upper side at a probability below the floor would turn a factor negative
        if (np.asarray(q) < self.q_low).any() or (np.asarray(q) > 1).any():
            raise ValueError(f"an audit probability lies outside [q_low, 1] with q_low {self.q_low}")

    def tables(self, q, k_max):
        # both bounds for 0 to k_max audited attacks when every audit has probability q, the upper one uncapped
        self._check(q)
        k = np.arange(k_max + 1)
        if self.kind == "thinning":
            # a dead bet has a log factor of minus infinity, which counts as nothing before the first audited attack
            with np.errstate(invalid="ignore"):
                stat = {side: np.where(k[:, None] > 0, k[:, None] * gains(side, self.bet[side], q), 0.0)
                        for side in SIDES}
        else:
            stat = {side: k / q for side in SIDES}
        return self.lower(k, stat["lower"]), self.upper(k, stat["upper"], k + 10 ** 12)


def leakage_bound(n, lcb_table, ucb_table, eps):
    """the bound of Theorem 3 on the attacks among n dismissed flows while the lower bound has not raised an alarm.

    with no alarm up to n the audited attacks number at most the largest k whose lower bound stays at or below eps n,
    and the attacks at most the upper bound at that k, both read from tables by the number of audited attacks. the
    tables must be the least favorable ones of the policy, the lower bound at its largest audit probability and the
    upper bound at its smallest.
    """
    k_star = np.searchsorted(lcb_table, eps * np.asarray(n, dtype=float), side="right") - 1
    # the last entry of the lower table says nothing about the next one, so k_star must stop before it
    if (k_star < 0).any() or (k_star + 1 >= len(lcb_table)).any() or len(ucb_table) < len(lcb_table):
        raise ValueError("the tables do not reach the number of audited attacks that n allows")
    return ucb_table[k_star]
