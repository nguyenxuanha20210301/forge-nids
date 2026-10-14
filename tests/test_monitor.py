import itertools

import numpy as np
import pytest
from scipy.special import logsumexp

from forge.data.synthetic import attack_bins, attack_positions
from forge.monitor import rule
from forge.monitor.audit import allocate, bin_profile
from forge.monitor.sequence import (KINDS, Sequence, bets, coefficients, gains, leakage_bound, lower_bound,
                                    upper_bound)

LOWER, UPPER = [1.5, 2, 3, 5, 10, 30, 100], [0, 0.25, 0.5, 0.75]


def sequence(kind, q=0.005, q_low=None, delta_lower=0.025, delta_upper=0.025):
    return Sequence(kind, LOWER, UPPER, q, q if q_low is None else q_low, delta_lower, delta_upper)


@pytest.mark.parametrize("bet", [0.3, 0.9, -0.2])
def test_the_audit_product_is_a_martingale_for_any_labels(bet):
    # exact over every outcome of the coins, the labels and audit probabilities fixed in advance. a negative bet is
    # allowed up to the floor of the probabilities over one minus the floor
    qs = np.array([0.3, 0.5, 0.2, 0.6])
    labels = np.array([1, 1, 0, 1])
    for n in range(1, 5):
        total = 0.0
        for coins in itertools.product([0, 1], repeat=n):
            prob = np.prod([q if c else 1 - q for q, c in zip(qs[:n], coins)])
            total += prob * np.prod([1 + bet * y * (c / q - 1) for q, y, c in zip(qs[:n], labels[:n], coins)])
        assert total == pytest.approx(1.0)


def test_the_product_depends_on_the_labels_through_the_count_only():
    # the log wealth from the audited attacks and the count equals the product over the flows
    rng = np.random.default_rng(0)
    qs = rng.uniform(0.1, 0.6, size=12)
    labels = rng.random(12) < 0.6
    coins = rng.random(12) < qs
    bet = np.array([0.2, 0.5])
    direct = np.prod(1 + bet[:, None] * labels * (coins / qs - 1), axis=1)
    audited = labels & coins
    stat = gains("lower", bet, qs[audited]).sum(0)
    a, b = coefficients("thinning", "lower", bet, audited.sum(), stat, qs.min())
    assert np.allclose(a + b * labels.sum(), np.log(direct))


@pytest.mark.parametrize("kind", KINDS)
def test_lower_bound_is_the_smallest_count_not_ruled_out(kind):
    seq = sequence(kind, q=0.05)
    k = np.array([0, 3, 7])
    stat = (k[:, None] * gains("lower", seq.bet["lower"], 0.05)) if kind == "thinning" else k / 0.05
    a, b = coefficients(kind, "lower", seq.bet["lower"], k, stat, 0.05)
    found = lower_bound(a, b, seq.weights["lower"], k, seq.level["lower"])
    log_w = np.log(seq.weights["lower"])
    for i in range(len(k)):
        wealth = [logsumexp(log_w + a[i] + b[i] * L) for L in range(int(k[i]), int(found[i]) + 2)]
        assert all(w >= seq.level["lower"] for w in wealth[:-2]) and wealth[-2] < seq.level["lower"]


@pytest.mark.parametrize("kind", KINDS)
def test_upper_bound_is_the_largest_count_not_ruled_out(kind):
    seq = sequence(kind, q=0.05)
    k = np.array([0, 2, 9])
    with np.errstate(invalid="ignore"):
        stat = np.where(k[:, None] > 0, k[:, None] * gains("upper", seq.bet["upper"], 0.05), 0.0) \
            if kind == "thinning" else k / 0.05
    a, b = coefficients(kind, "upper", seq.bet["upper"], k, stat, 0.05)
    found = upper_bound(a, b, seq.weights["upper"], k, seq.level["upper"], 10 ** 9)
    log_w = np.log(seq.weights["upper"])
    for i in range(len(k)):
        below = logsumexp(log_w + a[i] + b[i] * found[i])
        above = logsumexp(log_w + a[i] + b[i] * (found[i] + 1))
        assert below < seq.level["upper"] <= above
    # the cap wins when the count cannot exceed the dismissed flows
    assert (upper_bound(a, b, seq.weights["upper"], k, seq.level["upper"], k + 5) == k + 5).all()


def test_bounds_rise_with_the_audited_attacks_and_fall_with_more_audits():
    seq = sequence("thinning")
    lcb, ucb = seq.tables(0.005, 50)
    assert lcb[0] == 0 and (np.diff(lcb) >= 0).all() and (np.diff(ucb) >= 0).all() and (lcb <= ucb).all()
    # the same audited attacks mean more attacks when audits are rarer
    lcb_rare, ucb_rare = sequence("thinning", q=0.001).tables(0.001, 50)
    assert (lcb_rare[1:] >= lcb[1:]).all() and (ucb_rare >= ucb).all()
    assert lcb[1] == 1
    # a probability below the floor of the sequence is refused
    with pytest.raises(ValueError):
        seq.tables(0.001, 5)


def test_freedman_is_never_tighter_than_the_audit_martingale_with_uniform_audits():
    lcb, ucb = sequence("thinning").tables(0.005, 60)
    fl, fu = sequence("freedman").tables(0.005, 60)
    assert (fl <= lcb).all() and (fu >= ucb).all() and fu[0] > ucb[0]


def test_bets_are_growth_optimal_against_their_ratio():
    # the expected log growth per flow at a hypothesis of eps n, when the true FOR is r eps, peaks at the bet of r
    q, eps = 0.005, 1e-3
    lam = bets("thinning", "lower", [5], q, q)[0]

    def growth(b, r):
        return q * r * eps * np.log1p(b * (1 / q - 1)) + (eps - q * r * eps) * np.log1p(-b)

    grid = np.linspace(0.001, 0.2, 2000)
    assert abs(grid[np.argmax(growth(grid, 5))] - lam) < 1e-3


def test_bets_reject_impossible_ratios_and_probabilities():
    with pytest.raises(ValueError):
        bets("thinning", "lower", [1.0], 0.005, 0.005)
    with pytest.raises(ValueError):
        bets("thinning", "upper", [1.0], 0.005, 0.005)
    with pytest.raises(ValueError):
        bets("thinning", "lower", [2.0], 0.001, 0.005)
    with pytest.raises(ValueError):
        bets("other", "lower", [2.0], 0.005, 0.005)


def _coverage(kind, q, positions, draws, seed, delta_lower=0.025, delta_upper=0.025):
    # share of draws in which the count leaves the bounds at some attack, with every audit at probability q. the
    # count changes at attacks only and the bounds at audited attacks only, so every moment is checked at an attack
    rng = np.random.default_rng(seed)
    n_attacks = len(positions)
    coins = rng.random((draws, n_attacks)) < q
    k = np.cumsum(coins, axis=1)
    lcb, ucb = sequence(kind, q, delta_lower=delta_lower, delta_upper=delta_upper).tables(q, n_attacks)
    count = np.arange(1, n_attacks + 1)
    return float(((count < lcb[k]) | (count > ucb[k])).any(1).mean())


@pytest.mark.parametrize("kind", KINDS)
def test_bounds_cover_with_probability_at_least_one_minus_delta_on_a_fixed_stream(kind):
    # a burst of attacks followed by a trickle, the positions fixed so that only the coins vary. 4000 draws put the
    # 95% interval of a share at delta within about 0.007
    burst = np.arange(1, 1501)
    trickle = 1500 + np.cumsum(np.full(500, 40))
    positions = np.concatenate([burst, trickle])
    assert _coverage(kind, 0.02, positions, 4000, 1) <= 0.05 + 0.01
    # the check can fail, a lower bound at a level fit for delta 0.5 leaves the count outside far more often
    assert _coverage(kind, 0.02, positions, 4000, 1, delta_lower=0.5) > 0.1


def test_leakage_bound_covers_the_count_at_the_alarm_whenever_the_bounds_cover():
    # whenever the count stays within the bounds up to the alarm, the count at the alarm is at most the bound of
    # Theorem 3 read at the flows before it plus the attack at the alarm itself
    rng = np.random.default_rng(2)
    q, eps, draws = 0.01, 1e-3, 2000
    positions = np.cumsum(rng.geometric(5 * eps, size=4000))
    coins = rng.random((draws, len(positions))) < q
    k = np.cumsum(coins, axis=1)
    lcb, ucb = sequence("thinning", q).tables(q, len(positions))
    count = np.arange(1, len(positions) + 1)
    alarm = coins & (lcb[k] > eps * positions)
    first = np.where(alarm.any(1), alarm.argmax(1), -1)
    covered = ~((count < lcb[k]) | (count > ucb[k])).any(1)
    hit = (first >= 0) & covered
    assert hit.sum() > 1000
    n_alarm = positions[first[hit]]
    bound = leakage_bound(n_alarm - 1, lcb, ucb, eps) + 1
    assert (count[first[hit]] <= bound).all()


def test_leakage_bound_needs_tables_that_reach_the_dismissed_count():
    lcb, ucb = sequence("thinning").tables(0.005, 5)
    with pytest.raises(ValueError):
        leakage_bound(10 ** 7, lcb, ucb, 1e-3)


@pytest.mark.parametrize("power", [0, 0.5, 1])
def test_allocation_meets_the_budget_and_respects_the_floor(power):
    profile = bin_profile(10, 2.0)
    mass = np.full(10, 0.1)
    q = allocate(profile * 10 * 5e-4, mass, 0.005, 0.0005, power)
    assert np.sum(mass * q) == pytest.approx(0.005)
    assert (q >= 0.0005).all() and (q <= 1).all() and (np.diff(q) >= 0).all()
    if power == 0:
        assert np.allclose(q, 0.005)
    else:
        assert q[-1] > q[0]


def test_allocation_puts_bins_without_attacks_at_the_floor_and_rejects_a_budget_below_it():
    q = allocate(np.array([0.0, 0.0, 0.1]), np.full(3, 1 / 3), 0.01, 0.001, 0.5)
    assert q[0] == q[1] == 0.001 and np.sum(q) / 3 == pytest.approx(0.01)
    with pytest.raises(ValueError):
        allocate(np.array([0.1, 0.2]), np.array([0.5, 0.5]), 0.0001, 0.001, 0.5)


def test_bin_profile_rises_by_the_ratio_and_sums_to_one():
    p = bin_profile(4, 3.0)
    assert p.sum() == pytest.approx(1.0) and np.allclose(p[1:] / p[:-1], 3.0)


def test_attack_positions_follow_a_bernoulli_stream():
    rng = np.random.default_rng(3)
    segments = [(2000, 0.1), (1000, 0.0), (3000, 0.02)]
    positions, counts = attack_positions(rng, segments, 400)
    assert positions.shape[0] == 400 and (np.diff(positions, axis=1) >= 0).all()
    assert (positions[:, -1] >= 6001).any() or (counts == positions.shape[1]).all()
    inside = positions <= 6000
    assert (inside.sum(1) == counts).all()
    first = ((positions >= 1) & (positions <= 2000)).sum(1)
    second = ((positions > 2000) & (positions <= 3000)).sum(1)
    third = ((positions > 3000) & (positions <= 6000)).sum(1)
    assert abs(first.mean() - 200) < 4 and (second == 0).all() and abs(third.mean() - 60) < 2.5
    # no two attacks share a flow
    assert all(len(np.unique(row[row <= 6000])) == c for row, c in zip(positions, counts))
    bins = attack_bins(rng, positions.shape, [0.2, 0.8])
    assert bins.shape == positions.shape and abs((bins == 1).mean() - 0.8) < 0.02


def test_first_alarm_and_certified_stretches_on_a_hand_example():
    eps = 0.1
    positions = np.array([[0, 10, 20, 50]])
    valid = np.array([[True, True, True, False]])
    lcb = np.array([[0, 0.5, 3, 9]])
    assert rule.first_alarm(lcb, positions, valid, eps)[0] == 2
    assert rule.first_alarm(lcb, positions, np.array([[True, True, False, False]]), eps)[0] == -1
    ucb = np.array([[1.5, 4, 2.5, 0]])
    nxt = np.array([[10, 20, 101, 101]])
    # certified from the flow where eps times the count reaches the bound, 15, 40 and 25, until the next event
    start, length = rule.certified_stretches(ucb, positions, nxt, valid, eps)
    assert start.tolist() == [[0, 0, 25, 0]] and length.tolist() == [[0, 0, 76, 0]]
    start, length = rule.certified_stretches(ucb, positions, nxt, valid, eps, stop=np.array([60]))
    assert length.tolist() == [[0, 0, 35, 0]]
    assert rule.state(np.array([0, 5, 0]), np.array([4, 20, 3]), np.array([30, 30, 30]), eps).tolist() == \
        ["running", "violation", "certified"]


def test_reading_waits_for_the_earliest_pending_label():
    # no pending attack audit reads up to n less the benign delay, a pending one holds the reading before it
    assert rule.reading(100, 0, 0, 101) == 100
    assert rule.reading(100, 10, 50, 101) == 90
    assert rule.reading(100, 10, 50, 70) == 69
    assert rule.reading(5, 10, 50, 6) == 0
    with pytest.raises(ValueError):
        rule.reading(100, 50, 10, 101)
