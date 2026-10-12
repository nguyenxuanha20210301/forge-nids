import numpy as np
import pytest
from scipy.stats import beta, binom, binomtest

from forge.certify.bounds import clopper_pearson_lower, clopper_pearson_upper, dkw_margin
from forge.certify.certificate import (certify, certify_scores, counts_below, fixed_sequence, for_from_rates,
                                       ltt_iid)
from forge.data.synthetic import beta_cdfs, draw_counts, sample_flows

BENIGN = (2, 8)
FAMILIES = [(4, 3), (5, 2.5), (6, 2), (8, 2)]
SHARES = [0.1, 0.2, 0.3, 0.4]


@pytest.mark.parametrize("k, n", [(0, 1), (0, 100), (3, 1000), (50, 100), (99, 100), (100, 100), (7, 200000)])
@pytest.mark.parametrize("delta", [0.025, 0.00625])
def test_clopper_pearson_matches_the_exact_interval_of_scipy(k, n, delta):
    # a two sided exact interval at level 1 - 2 delta has the one sided bounds at delta as its ends
    ci = binomtest(k, n).proportion_ci(confidence_level=1 - 2 * delta, method="exact")
    assert clopper_pearson_upper(k, n, delta) == pytest.approx(ci.high, rel=1e-9, abs=1e-12)
    assert clopper_pearson_lower(k, n, delta) == pytest.approx(ci.low, rel=1e-9, abs=1e-12)


def test_clopper_pearson_on_arrays_and_empty_samples():
    k = np.array([[0, 5, 20], [1, 2, 3]])
    n = np.array([[20], [40]])
    upper = clopper_pearson_upper(k, n, 0.05)
    assert upper.shape == (2, 3)
    assert upper[0, 2] == 1.0 and upper[1, 1] == clopper_pearson_upper(2, 40, 0.05)
    assert isinstance(clopper_pearson_upper(2, 40, 0.05), float)
    assert clopper_pearson_upper(0, 0, 0.05) == 1.0 and clopper_pearson_lower(0, 0, 0.05) == 0.0


@pytest.mark.parametrize("p", [0.001, 0.02, 0.3, 0.9])
def test_clopper_pearson_upper_misses_with_probability_at_most_delta(p):
    # exact over every possible count, not sampled
    n, delta = 60, 0.05
    k = np.arange(n + 1)
    assert binom.pmf(k, n, p)[clopper_pearson_upper(k, n, delta) < p].sum() <= delta
    assert binom.pmf(k, n, p)[clopper_pearson_lower(k, n, delta) > p].sum() <= delta


def test_dkw_margin():
    assert np.exp(-2 * 1000 * dkw_margin(1000, 0.025) ** 2) == pytest.approx(0.025)
    assert np.isinf(dkw_margin(np.array([0, 10]), 0.025)[0])
    with pytest.raises(ValueError):
        dkw_margin(100, 0.6)


@pytest.mark.parametrize("passes, start, expected", [
    ([0, 1, 1, 0, 1], 1, 2),
    ([0, 1, 1, 0, 1], 0, -1),
    ([0, 1, 1, 0, 1], 4, 4),
    ([0, 1, 1, 0, 1], 5, -1),
    ([1, 1, 1], 0, 2),
])
def test_fixed_sequence(passes, start, expected):
    assert fixed_sequence(np.array(passes, dtype=bool), start) == expected


def test_fixed_sequence_on_many_calibration_sets():
    passes = np.array([[1, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=bool)
    assert fixed_sequence(passes, np.array([0, 1, 3])).tolist() == [1, 2, -1]


def hand_counts(misses):
    # one family of 10000 attacks and 10^6 benign flows, benign shares 0.1, 0.2, 0.3, 0.6 and 0.9 at the thresholds
    return dict(benign_below=np.array([1, 2, 3, 6, 9]) * 100000, n_benign=1000000,
                attack_below=np.array([misses]), n_attack=np.array([10000]))


def test_certify_stops_at_the_first_failure():
    counts = hand_counts([0, 0, 30, 30, 30])
    out = certify(**counts, eps=1e-3, delta=0.05, pi_max=0.1)
    share = counts["benign_below"] / counts["n_benign"] - dkw_margin(counts["n_benign"], 0.025)
    passes = for_from_rates(0.1, clopper_pearson_upper(counts["attack_below"][0], 10000, 0.025), share) <= 1e-3
    # the fourth threshold passes on its own, but the third fails first
    assert passes.tolist() == [True, True, False, True, True]
    assert out["index"] == 1
    assert out["bound"] == pytest.approx(for_from_rates(0.1, clopper_pearson_upper(0, 10000, 0.025), share[1]))
    assert out["floor"] == pytest.approx(0.9 * share[1])


def test_feasible_start_skips_thresholds_that_cannot_pass():
    # with 1000 attacks the miss rate bound is at least 3.7e-3, too much for the benign shares 0.1 to 0.3
    counts = {**hand_counts([0, 0, 0, 0, 3]), "n_attack": np.array([1000])}
    assert certify(**counts, eps=1e-3, delta=0.05, pi_max=0.1, start="strictest")["index"] == -1
    assert certify(**counts, eps=1e-3, delta=0.05, pi_max=0.1)["index"] == 3
    with pytest.raises(ValueError):
        certify(**counts, eps=1e-3, delta=0.05, pi_max=0.1, start="loosest")


def test_bound_takes_the_worst_family_at_delta_m_over_k():
    benign_below, n_benign = np.array([1, 2, 3, 6, 9]) * 100000, 1000000
    families = np.array([[0, 1, 2, 3, 9], [0, 0, 0, 5, 30]])
    n_families = np.array([20000, 8000])
    out = certify(benign_below, n_benign, families, n_families, eps=1e-3, delta=0.04, pi_max=0.2, delta_benign=0.01)
    j = int(out["index"])
    assert j >= 0
    m = max(clopper_pearson_upper(families[k, j], n_families[k], 0.03 / 2) for k in range(2))
    b = benign_below[j] / n_benign - dkw_margin(n_benign, 0.01)
    assert out["bound"] == pytest.approx(for_from_rates(0.2, m, b), rel=1e-12)
    assert out["floor"] == pytest.approx(0.8 * b, rel=1e-12)


def test_counts_below_counts_ties_as_dismissed():
    rng = np.random.default_rng(8)
    scores = rng.integers(0, 20, 500) / 10
    grid = np.array([-1.0, 0.0, 0.5, 1.0, 1.95, 3.0])
    assert counts_below(scores, grid).tolist() == (scores[:, None] <= grid).sum(0).tolist()


def test_worst_family_bound_is_stricter_than_the_pooled_one():
    benign_below, n_benign = np.array([1, 2, 3, 6, 9]) * 100000, 1000000
    families = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 4, 40]])
    n_families = np.array([20000, 2000])
    simplex = certify(benign_below, n_benign, families, n_families, eps=1e-3, delta=0.05, pi_max=0.1)
    pooled = certify(benign_below, n_benign, families.sum(0, keepdims=True), n_families.sum(keepdims=True),
                     eps=1e-3, delta=0.05, pi_max=0.1)
    assert simplex["index"] < pooled["index"]


def test_certificate_ignores_increasing_transforms_of_the_scores():
    rng = np.random.default_rng(3)
    scores, labels, family = sample_flows(rng, 400000, 0.05, SHARES, BENIGN, FAMILIES)
    grid = np.quantile(scores[labels == 0], np.arange(1, 101) / 101)
    attack = labels == 1

    def run(f):
        return certify_scores(f(scores[~attack]), f(scores[attack]), family[attack], f(grid), eps=5e-3, delta=0.05,
                              pi_max=0.1)

    plain = run(lambda s: s)
    assert plain["index"] >= 0
    for f in [lambda s: np.log(s / (1 - s)), lambda s: np.exp(8 * s) - 3, lambda s: s ** 5]:
        moved = run(f)
        assert (moved["index"], moved["bound"], moved["floor"]) == (plain["index"], plain["bound"], plain["floor"])


def test_closed_form_matches_a_large_sample():
    rng = np.random.default_rng(4)
    pi, t = 0.2, 0.3
    scores, labels, _ = sample_flows(rng, 1000000, pi, SHARES, BENIGN, FAMILIES)
    b, m = beta_cdfs(np.array([t]), BENIGN, FAMILIES)
    truth = float(for_from_rates(pi, np.array(SHARES) @ m[:, 0], b[0]))
    dismissed = scores <= t
    seen = labels[dismissed].mean()
    assert abs(seen - truth) < 4 * np.sqrt(truth * (1 - truth) / dismissed.sum())


def test_draw_counts_follow_the_cdfs():
    rng = np.random.default_rng(5)
    grid = beta.ppf(np.arange(1, 21) / 21, *BENIGN)
    b, m = beta_cdfs(grid, BENIGN, FAMILIES)
    d = draw_counts(rng, 200000, 0.05, SHARES, b, m, 400)
    assert (d.n_benign + d.n_family.sum(1) == 200000).all()
    assert (np.diff(d.benign_below, axis=1) >= 0).all() and (d.benign_below[:, -1] <= d.n_benign).all()
    assert (d.family_below[:, :, -1] <= d.n_family).all()
    assert np.allclose((d.benign_below / d.n_benign[:, None]).mean(0), b, atol=2e-3)
    assert np.allclose(d.n_family.mean(0) / (200000 * 0.05), SHARES, atol=0.01)
    assert np.allclose(d.family_below.sum(0) / d.n_family.sum(0)[:, None], m, atol=3e-3)


def violation_shares(n, pi_cal, families, shares, pi_max, draws, seed):
    # share of calibration sets whose certified threshold has a true FOR above epsilon where its claim says it holds,
    # for FORGE with the worst family claim and for LTT with the i.i.d. claim and with the worst family claim
    eps, delta = 1e-3, 0.05
    grid = beta.ppf(np.arange(1, 201) / 201, *BENIGN)
    b, m = beta_cdfs(grid, BENIGN, families)
    d = draw_counts(np.random.default_rng(seed), n, pi_cal, shares, b, m, draws)

    def worst(index, pi):
        safe = np.maximum(index, 0)
        rates = np.max([for_from_rates(pi, mk[safe], b[safe]) for mk in m], axis=0)
        return np.where(index >= 0, rates, 0.0)

    forge = certify(d.benign_below, d.n_benign, d.family_below, d.n_family, eps, delta, pi_max)["index"]
    ltt = ltt_iid(d.benign_below, d.family_below.sum(1), eps, delta)["index"]
    safe = np.maximum(ltt, 0)
    ltt_cal = np.where(ltt >= 0, for_from_rates(pi_cal, (np.array(shares) @ m)[safe], b[safe]), 0.0)
    return {"forge": np.mean(worst(forge, pi_max) > eps), "forge_refused": np.mean(forge < 0),
            "ltt_iid": np.mean(ltt_cal > eps), "ltt_at_pi_max": np.mean(worst(ltt, pi_max) > eps)}


def test_violations_stay_below_delta_and_the_check_can_fail():
    shares = violation_shares(30000000, 0.02, FAMILIES, SHARES, 0.1, draws=1500, seed=6)
    assert shares["forge_refused"] < 0.5
    assert shares["forge"] <= 0.05
    assert shares["ltt_iid"] <= 0.05
    # the same simulation catches the i.i.d. baseline at a higher prevalence, so it is able to find a violation
    assert shares["ltt_at_pi_max"] > 0.5


def test_violations_stay_below_delta_with_one_family_and_pi_max_at_calibration():
    # the tightest case, one family and no prevalence margin, where only the bounds keep the rate below epsilon
    shares = violation_shares(2000000, 0.1, [(4, 3)], [1.0], 0.1, draws=1500, seed=7)
    assert shares["forge_refused"] < 0.1
    assert shares["forge"] <= 0.05
