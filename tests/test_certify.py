import numpy as np
import pytest
from scipy.stats import beta, binom, binomtest

from forge.certify.bounds import clopper_pearson_lower, clopper_pearson_upper, dkw_margin
from forge.certify.certificate import (certify, certify_online, certify_online_scores, certify_scores, conformal_fnr,
                                       counts_below, family_groups, fixed_sequence, for_from_dismissed, for_from_rates,
                                       group_rows, ltt_iid, min_family_size, online_certifier, plug_in)
from forge.data.synthetic import benign_path, beta_cdfs, draw_counts, sample_flows

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


@pytest.mark.parametrize("fn", [certify, certify_online])
def test_start_tolerates_misses_within_the_slack(fn):
    # 10^6 flows with shares 0.01, 0.02, 0.03, 0.05, 0.08 and 0.2 at the thresholds, one family of 10^4 attacks with one
    # miss from the fourth threshold on. the fourth is the first that would pass with no miss, so that one miss ends a
    # sequence starting there. the fifth would pass with one miss and dismisses 0.03 more
    below, n, attack, n_attack = np.array([1, 2, 3, 5, 8, 20]) * 10000, 1000000, np.array([[0, 0, 0, 1, 1, 1]]), [10000]
    args = dict(eps=1e-3, delta=0.05, pi_max=0.1)
    assert fn(below, n, attack, np.array(n_attack), **args, start_misses=0)["index"] == -1
    assert fn(below, n, attack, np.array(n_attack), **args)["index"] == 5
    assert fn(below, n, attack, np.array(n_attack), **args, start_slack=0.02)["index"] == -1
    with pytest.raises(ValueError):
        fn(below, n, attack, np.array(n_attack), **args, start_misses=-1)


def test_bound_takes_the_worst_family_at_the_whole_delta_m():
    benign_below, n_benign = np.array([1, 2, 3, 6, 9]) * 100000, 1000000
    families = np.array([[0, 1, 2, 3, 9], [0, 0, 0, 5, 30]])
    n_families = np.array([20000, 8000])
    out = certify(benign_below, n_benign, families, n_families, eps=1e-3, delta=0.04, pi_max=0.2, delta_benign=0.01)
    j = int(out["index"])
    assert j >= 0
    # only the family with the largest miss rate matters at the threshold that decides, so delta_m is not split
    m = max(clopper_pearson_upper(families[k, j], n_families[k], 0.03) for k in range(2))
    b = benign_below[j] / n_benign - dkw_margin(n_benign, 0.01)
    assert out["bound"] == pytest.approx(for_from_rates(0.2, m, b), rel=1e-12)
    assert out["floor"] == pytest.approx(0.8 * b, rel=1e-12)


@pytest.mark.parametrize("delta_benign", [0.0, 0.05, 0.08])
def test_certify_rejects_a_benign_share_outside_delta(delta_benign):
    with pytest.raises(ValueError):
        certify(**hand_counts([0, 0, 0, 0, 0]), eps=1e-3, delta=0.05, pi_max=0.1, delta_benign=delta_benign)


def test_for_from_dismissed_agrees_with_the_rates_and_stops_at_one():
    pi, m, b = 0.1, np.array([0.0, 0.001, 0.02]), np.full(3, 0.3)
    assert np.allclose(for_from_dismissed(pi, m, pi * m + (1 - pi) * b), for_from_rates(pi, m, b))
    # bounds put in for m and d can give more than 1, and nothing dismissed gives 1 as well
    assert for_from_dismissed(0.1, 0.5, 0.01) == 1.0
    assert for_from_dismissed(0.1, 0.001, 0.0) == 1.0


def window_counts():
    # 10^6 unlabeled window flows, shares 0.1, 0.2, 0.3, 0.6 and 0.9 at the thresholds, and two attack families of D_cal
    return dict(window_below=np.array([1, 2, 3, 6, 9]) * 100000, n_window=1000000,
                attack_below=np.array([[0, 1, 2, 3, 9], [0, 0, 0, 5, 30]]), n_attack=np.array([20000, 8000]))


def test_online_bound_takes_the_window_share_and_the_whole_delta_m():
    counts = window_counts()
    out = certify_online(**counts, eps=1e-3, delta=0.04, pi_max=0.2, delta_window=0.01)
    # the fifth threshold fails, the second family has 30 misses there
    j = int(out["index"])
    assert j == 3
    m = max(clopper_pearson_upper(counts["attack_below"][k, j], counts["n_attack"][k], 0.03) for k in range(2))
    d = counts["window_below"][j] / counts["n_window"] - dkw_margin(counts["n_window"], 0.01)
    # the window's dismissed share stands alone in the denominator, attacks included
    assert out["bound"] == pytest.approx(0.2 * m / d, rel=1e-12)
    assert out["floor"] == pytest.approx(d, rel=1e-12)


def test_online_certificate_stops_at_the_first_failure_and_skips_impossible_starts():
    window_below, n_window = np.array([1, 2, 3, 6, 9]) * 100000, 1000000
    out = certify_online(window_below, n_window, np.array([[0, 0, 30, 30, 30]]), np.array([10000]), eps=1e-3,
                         delta=0.05, pi_max=0.1)
    # the fourth threshold passes on its own, but the third fails first
    assert out["index"] == 1
    # with 1000 attacks the bound is at least 0.1 times 3.7e-3 over the window share, too much up to a share of 0.3
    few = dict(window_below=window_below, n_window=n_window, attack_below=np.array([[0, 0, 0, 0, 9]]),
               n_attack=np.array([1000]))
    assert certify_online(**few, eps=1e-3, delta=0.05, pi_max=0.1, start="strictest")["index"] == -1
    assert certify_online(**few, eps=1e-3, delta=0.05, pi_max=0.1)["index"] == 3


@pytest.mark.parametrize("delta_window", [0.0, 0.05, 0.08])
def test_certify_online_rejects_a_window_share_outside_delta(delta_window):
    with pytest.raises(ValueError):
        certify_online(**window_counts(), eps=1e-3, delta=0.05, pi_max=0.1, delta_window=delta_window)


def test_online_certifier_matches_certify_online_window_by_window():
    # one D_cal of 300 draws against windows of other benign scores and sizes, its bounds computed once
    grid = beta.ppf(np.arange(1, 201) / 201, *BENIGN)
    b, m = beta_cdfs(grid, BENIGN, FAMILIES)
    rng = np.random.default_rng(14)
    cal = draw_counts(rng, 3000000, 0.02, SHARES, b, m, 300)
    certifier = online_certifier(cal.family_below, cal.n_family, 1e-3, 0.05, 0.1, 0.02)
    refused = []
    for window_benign, n in [((2, 8), 100000), ((2, 5), 1000000), ((3, 4), 30000)]:
        b_w, _ = beta_cdfs(grid, window_benign, FAMILIES)
        win = draw_counts(rng, n, 0.05, SHARES, b_w, m, 300)
        below, sizes = win.benign_below + win.family_below.sum(1), np.full(300, n)
        once = certifier(below, sizes)
        each = certify_online(below, sizes, cal.family_below, cal.n_family, 1e-3, 0.05, 0.1, 0.02)
        for key in ("index", "bound", "floor"):
            assert np.array_equal(once[key], each[key], equal_nan=True)
        refused.append(np.mean(once["index"] < 0))
    # the windows cover certified and refused draws alike
    assert min(refused) < 0.5 < max(refused)


def test_threshold_within_the_dkw_margin_never_passes():
    # with 1000 benign flows the margin is 0.043, so at the first threshold, which holds 1% of them, no benign share is
    # certain, and the bound must stay at 1 there whatever the attacks do
    out = certify(np.array([10, 500, 900]), 1000, np.array([[0, 0, 0]]), np.array([100000]), eps=1e-3, delta=0.05,
                  pi_max=0.1, start="strictest")
    assert out["index"] == -1


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


def test_online_certificate_ignores_increasing_transforms_of_the_scores():
    rng = np.random.default_rng(11)
    window, _, _ = sample_flows(rng, 400000, 0.05, SHARES, (2, 6), FAMILIES)
    cal, labels, family = sample_flows(rng, 400000, 0.05, SHARES, BENIGN, FAMILIES)
    attack = labels == 1
    grid = np.quantile(cal[~attack], np.arange(1, 101) / 101)

    def run(f):
        return certify_online_scores(f(window), f(cal[attack]), family[attack], f(grid), eps=5e-3, delta=0.05,
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


def test_benign_path_moves_in_steps_or_jumps_once():
    steps = benign_path((2, 8), (2, 5), 4)
    assert np.allclose(steps[:, 0], 2) and np.allclose(steps[:, 1], [8, 7.25, 6.5, 5.75, 5])
    assert np.allclose(benign_path((2, 8), (2, 5), 4, shift_at=3)[:, 1], [8, 8, 8, 5, 5])
    with pytest.raises(ValueError):
        benign_path((2, 8), (2, 5), 4, shift_at=5)


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


def online_violation_shares(window_benign, draws, seed):
    # share of draws whose online and offline certificates exceed epsilon in one window of 10^6 flows at pi_max with
    # all attacks from the family with the heaviest low tail, once the benign scores have moved to window_benign
    eps, delta, pi_max = 1e-3, 0.05, 0.1
    grid = beta.ppf(np.arange(1, 201) / 201, *BENIGN)
    b, m = beta_cdfs(grid, BENIGN, FAMILIES)
    b_w, _ = beta_cdfs(grid, window_benign, FAMILIES)
    rng = np.random.default_rng(seed)
    cal = draw_counts(rng, 30000000, 0.02, SHARES, b, m, draws)
    win = draw_counts(rng, 1000000, pi_max, [1, 0, 0, 0], b_w, m, draws)
    online = certify_online(win.benign_below + win.family_below.sum(1), 1000000, cal.family_below, cal.n_family, eps,
                            delta, pi_max)["index"]
    offline = certify(cal.benign_below, cal.n_benign, cal.family_below, cal.n_family, eps, delta, pi_max)["index"]
    rate = for_from_rates(pi_max, m[0], b_w)

    def share(index):
        return np.mean(np.where(index >= 0, rate[np.maximum(index, 0)], 0.0) > eps)

    return {"online": share(online), "online_refused": np.mean(online < 0), "offline": share(offline)}


def test_online_violations_stay_below_delta_when_the_benign_scores_drift():
    shares = online_violation_shares((2, 5), draws=1500, seed=10)
    assert shares["online_refused"] < 0.5
    assert shares["online"] <= 0.05
    # the same windows catch the offline certificate, whose benign bound no longer holds
    assert shares["offline"] > 0.5


def test_conformal_fnr_allows_floor_beta_n_plus_one_minus_one_attacks():
    # 99 attacks at beta 0.05 allow floor(5) - 1 = 4 at or below the threshold
    below = np.array([0, 2, 4, 5, 9])
    assert conformal_fnr(below, 99, 0.05) == 2
    assert conformal_fnr(np.array([[0, 2, 4, 5, 9], [5, 6, 7, 8, 9]]), np.array([99, 99]), 0.05).tolist() == [2, -1]
    # with beta (n + 1) below 1 no threshold qualifies, not even one with no attack below it
    assert conformal_fnr(below, 10, 0.05) == -1


def test_conformal_fnr_keeps_the_miss_rate_below_beta_on_average():
    # the guarantee is marginal, over the calibration draw, so the mean true miss rate of the chosen thresholds sits at
    # or below beta, close to it on a fine grid, while single draws exceed it often
    grid = beta.ppf(np.linspace(0.001, 0.999, 2000), *FAMILIES[0])
    m = beta.cdf(grid, *FAMILIES[0])
    b, _ = beta_cdfs(grid, BENIGN, FAMILIES)
    d = draw_counts(np.random.default_rng(12), 2000, 0.5, [1.0], b, m[None], 4000)
    index = conformal_fnr(d.family_below[:, 0], d.n_family[:, 0], 0.02)
    true = np.where(index >= 0, m[np.maximum(index, 0)], 0.0)
    assert 0.017 < true.mean() <= 0.02 + 3 * true.std() / np.sqrt(len(true))
    assert np.mean(true > 0.02) > 0.2


def test_plug_in_stops_at_the_first_estimate_above_epsilon():
    # at prevalence 0.1 the estimated FOR is 0, 0.0044, 0.0120 and 0.0081 at the four thresholds
    benign, attack = np.array([100, 500, 550, 950]), np.array([0, 20, 60, 70])
    assert plug_in(benign, 1000, attack, 1000, 0.01, 0.1) == 1
    assert plug_in(benign, 1000, attack, 1000, 0.013, 0.1) == 3
    # 0.0110 at the strictest threshold
    assert plug_in(benign, 1000, np.array([10, 20, 60, 70]), 1000, 0.01, 0.1) == -1


def test_min_family_size_is_where_a_threshold_with_no_miss_starts_to_pass():
    # 1,364 attacks at epsilon 1e-3, pi_max 0.1 and delta_m 0.025 for a benign share of 0.3, as in proposal 6.3
    eps, pi_max, delta_m = 1e-3, 0.1, 0.025
    n = min_family_size(eps, pi_max, delta_m, 0.3)
    assert n == 1364
    for share in [0.05, 0.3, 0.9]:
        n = min_family_size(eps, pi_max, delta_m, share)
        assert for_from_rates(pi_max, clopper_pearson_upper(0, n, delta_m), share) <= eps
        assert for_from_rates(pi_max, clopper_pearson_upper(0, n - 1, delta_m), share) > eps


@pytest.mark.parametrize("sizes, groups", [
    ([78000, 634], [0, 0]),
    ([5000, 3000], [0, 1]),
    ([5000, 600, 900], [0, 1, 1]),
    ([5000, 10, 3000, 20], [0, 0, 1, 0]),
    ([10, 20], [0, 0]),
])
def test_family_groups_pool_rare_families_and_join_the_largest_when_still_rare(sizes, groups):
    assert family_groups(sizes, 1364).tolist() == groups


def test_group_rows_sums_counts_and_sizes_within_groups():
    below = np.arange(24).reshape(2, 3, 4)
    n = np.array([[10, 20, 30], [40, 50, 60]])
    summed, sizes = group_rows(below, n, np.array([0, 1, 0]))
    assert summed.shape == (2, 2, 4) and sizes.tolist() == [[40, 20], [100, 50]]
    assert (summed[:, 0] == below[:, 0] + below[:, 2]).all() and (summed[:, 1] == below[:, 1]).all()
