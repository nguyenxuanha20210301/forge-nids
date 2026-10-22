import numpy as np
import pytest
from scipy.stats import beta, binom, binomtest

from forge.certify.bounds import clopper_pearson_lower, clopper_pearson_upper, dkw_margin
from forge.certify.certificate import (certify, certify_online, certify_online_scores, certify_scores, conformal_fnr,
                                       conformal_selection, counts_below, estimate_prevalence, estimated_certifier,
                                       family_groups, fixed_sequence, for_from_dismissed, for_from_rates, group_rows,
                                       ks_threshold, ltt_iid, min_family_size, np_umbrella, online_certifier, plug_in)
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


def test_share_start_begins_where_the_benign_bound_reaches_the_share():
    # 10^6 benign flows at shares 0.1 to 1.0, one family of 10^4 attacks with 50 flows below every threshold, a tail
    # among the lowest benign scores, and 2,000 below the last. the feasible start is the first threshold, where the
    # bound allows 45 misses against a benign share of 0.1, so the sequence fails at its first test, while from any
    # later threshold the same misses pass against a larger benign share until the last
    below, n, attack, n_attack = np.arange(1, 11) * 100000, 1000000, np.array([[50] * 9 + [2000]]), np.array([10000])
    args = dict(eps=5e-3, delta=0.05, pi_max=0.1)
    assert certify(below, n, attack, n_attack, **args)["index"] == -1
    share = below / n - dkw_margin(n, 0.025)
    assert share[4] < 0.5 <= share[5]
    for fn in (certify, certify_online):
        assert fn(below, n, attack, n_attack, **args, start="share", start_share=0.5)["index"] == 8
        # a share no lower bound reaches leaves no start, and the share must be given and lie inside (0, 1)
        assert fn(below, n, attack, n_attack, **args, start="share", start_share=0.9999)["index"] == -1
        for bad in (None, 0.0, 1.0, 1.5):
            with pytest.raises(ValueError, match="start_share"):
                fn(below, n, attack, n_attack, **args, start="share", start_share=bad)
    # the pooled estimate and the score wrappers take the same start
    at_pi = estimated_certifier(below[None, :], np.array([n]), attack, n_attack, 5e-3, 0.025, 0.025, start="share",
                                start_share=0.5)
    assert at_pi(np.array([0.1]))["index"] == [8]
    grid = np.linspace(0.1, 1.0, 10)
    benign_scores = np.repeat(grid, 100000) - 0.05
    attack_scores = np.concatenate([np.full(50, 0.0), np.full(1950, 0.95), np.full(8000, 2.0)])
    families = np.zeros(10000, dtype=int)
    out = certify_scores(benign_scores, attack_scores, families, grid, **args, start="share", start_share=0.5)
    assert out["index"] == 8


def test_ltt_share_start_begins_at_the_dismissed_share():
    # 600 attacks among the 100,600 flows dismissed at the first threshold, 0.006 of them, so the sequence from the
    # first threshold that could pass with no attack fails at once, while from the share start at half of the
    # 1,010,000 flows the same 600 attacks are 0.001 of the dismissed flows and every threshold passes
    benign, attack, n = np.arange(1, 11) * 100000, np.full(10, 600), np.array(1010000)
    assert ltt_iid(benign, attack, 5e-3, 0.05)["index"] == -1
    assert ltt_iid(benign, attack, 5e-3, 0.05, n=n, start="share", start_share=0.5)["index"] == 9
    with pytest.raises(ValueError, match="size n"):
        ltt_iid(benign, attack, 5e-3, 0.05, start="share", start_share=0.5)


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
    share = certify(d.benign_below, d.n_benign, d.family_below, d.n_family, eps, delta, pi_max, start="share",
                    start_share=0.1)["index"]
    ltt = ltt_iid(d.benign_below, d.family_below.sum(1), eps, delta)["index"]
    guarded = ltt_iid(d.benign_below, d.family_below.sum(1), eps, delta, n=d.n_benign + d.n_family.sum(1),
                      start_misses=2, start_slack=0.05)["index"]

    def at_calibration(index):
        safe = np.maximum(index, 0)
        return np.where(index >= 0, for_from_rates(pi_cal, (np.array(shares) @ m)[safe], b[safe]), 0.0)

    return {"forge": np.mean(worst(forge, pi_max) > eps), "forge_refused": np.mean(forge < 0),
            "forge_share": np.mean(worst(share, pi_max) > eps), "forge_share_refused": np.mean(share < 0),
            "ltt_iid": np.mean(at_calibration(ltt) > eps), "ltt_guarded": np.mean(at_calibration(guarded) > eps),
            "ltt_at_pi_max": np.mean(worst(ltt, pi_max) > eps)}


def test_violations_stay_below_delta_and_the_check_can_fail():
    shares = violation_shares(30000000, 0.02, FAMILIES, SHARES, 0.1, draws=1500, seed=6)
    assert shares["forge_refused"] < 0.5
    assert shares["forge"] <= 0.05
    # the share start certifies from a tenth of the benign flows on and keeps the claim as well
    assert shares["forge_share_refused"] < 0.5 and shares["forge_share"] <= 0.05
    assert shares["ltt_iid"] <= 0.05 and shares["ltt_guarded"] <= 0.05
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


def test_np_umbrella_takes_the_order_statistic_of_tong_feng_and_li():
    # their rule classifies a calibration attack as benign above the k-th of n ordered scores, with k the least whose
    # binomial tail of n at 1 - alpha from k on is at most delta, so n - k attacks may lie on the dismissed side
    n, alpha, delta = 1000, 0.01, 0.05
    k = next(k for k in range(1, n + 1) if binom.sf(k - 1, n, 1 - alpha) <= delta)
    below = np.arange(30)
    assert np_umbrella(below, n, alpha, delta) == n - k == 4
    assert clopper_pearson_upper(4, n, delta) <= alpha < clopper_pearson_upper(5, n, delta)
    # one row per calibration set, and a set whose strictest threshold already holds too many attacks refuses
    two = np.stack([below, below + 5])
    assert np_umbrella(two, np.array([n, n]), alpha, delta).tolist() == [4, -1]
    # the same counts out of more attacks allow more of them
    assert np_umbrella(below, 10 * n, alpha, delta) > 4


def test_np_umbrella_holds_the_miss_rate_with_probability_and_conformal_fnr_on_average():
    # at the same level the PAC threshold sits below the one that holds the miss rate on average, and its true miss
    # rate exceeds the level in at most delta of the draws, where the average one exceeds it often
    grid = beta.ppf(np.linspace(0.001, 0.999, 2000), *FAMILIES[0])
    m = beta.cdf(grid, *FAMILIES[0])
    b, _ = beta_cdfs(grid, BENIGN, FAMILIES)
    d = draw_counts(np.random.default_rng(13), 2000, 0.5, [1.0], b, m[None], 4000)
    umbrella = np_umbrella(d.family_below[:, 0], d.n_family[:, 0], 0.02, 0.05)
    average = conformal_fnr(d.family_below[:, 0], d.n_family[:, 0], 0.02)
    assert (umbrella >= 0).all() and (umbrella <= average).all()
    true = m[umbrella]
    assert np.mean(true > 0.02) <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / len(true))
    assert np.mean(m[average] > 0.02) > 0.2


def test_conformal_selection_is_the_benjamini_hochberg_step_on_the_window_flows():
    # p values 0.0001, 0.0002, 0.0006, 0.0041 and 0.0201 against level times the window share 0.0001, 0.0005, 0.003,
    # 0.006 and 0.01, so the step passes up to the fourth threshold
    window, attack = np.array([10, 50, 300, 600, 1000]), np.array([0, 1, 5, 40, 200])
    assert conformal_selection(window, 1000, attack, 10000, 0.01) == 3
    # the step takes the loosest passing threshold even after a failure, and none when nothing passes
    assert conformal_selection(np.array([10, 20, 900]), 1000, np.array([0, 50, 60]), 10000, 0.01) == 2
    assert conformal_selection(window, 1000, attack + 100, 10000, 0.01) == -1
    # the same step on the p values of single flows, scores rounded up to their grid threshold
    rng = np.random.default_rng(5)
    grid = np.linspace(0.05, 0.95, 19)
    for _ in range(50):
        cal = rng.beta(3, 2, size=40)
        n_cal = 400
        flows = np.r_[rng.beta(1, 6, size=300), rng.beta(3, 2, size=rng.integers(1, 40))]
        level = rng.choice([0.05, 0.1, 0.2])
        rank = np.searchsorted(grid, flows, side="left")
        p = np.where(rank < len(grid), (1 + counts_below(cal, grid)[np.minimum(rank, len(grid) - 1)]) / (n_cal + 1),
                     np.inf)
        ordered = np.sort(p)
        passing = np.flatnonzero(ordered <= level * np.arange(1, len(flows) + 1) / len(flows))
        selected = p <= level * (passing[-1] + 1) / len(flows) if len(passing) else np.zeros(len(flows), bool)
        index = conformal_selection(counts_below(flows, grid), len(flows), counts_below(cal, grid), n_cal, level)
        assert np.array_equal(selected, flows <= grid[index] if index >= 0 else np.zeros(len(flows), bool))


def test_conformal_selection_holds_the_rate_at_the_calibration_prevalence_and_not_above():
    # calibration sets and windows drawn from the same mixture are exchangeable, and the mean share of attacks among
    # the dismissed flows of the window stays at the level, while the true FOR at the threshold exceeds the level in
    # a large share of the draws, since the guarantee is an average. a window at four times the prevalence takes
    # about four times the level, the step stopping inside the grid at either prevalence
    grid = beta.ppf(np.arange(1, 201) / 201, *BENIGN)
    b, m = beta_cdfs(grid, BENIGN, FAMILIES)
    level, draws = 0.002, 3000

    def selected(pi_w):
        cal = draw_counts(np.random.default_rng(21), 20000, 0.05, SHARES, b, m, draws)
        win = draw_counts(np.random.default_rng(22), 20000, pi_w, SHARES, b, m, draws)
        attacks = win.family_below.sum(1)
        below = win.benign_below + attacks
        index = conformal_selection(below, np.full(draws, 20000), cal.family_below.sum(1),
                                    cal.n_benign + cal.n_family.sum(1), level)
        rows = np.arange(draws)
        caught, dismissed = attacks[rows, index], below[rows, index]
        share = np.divide(caught, dismissed, out=np.zeros(draws), where=(index >= 0) & (dismissed > 0))
        true = for_from_rates(pi_w, np.array(SHARES) @ m, b)[index]
        return index, share, true

    index, share, true = selected(0.05)
    assert (index >= 0).all() and np.quantile(index, 0.95) < len(grid) - 1
    assert share.mean() <= level + 3 * share.std() / np.sqrt(draws)
    assert np.mean(true > level) > 0.2
    index, share, _ = selected(0.2)
    assert np.quantile(index, 0.95) < len(grid) - 1 and share.mean() > 3 * level


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


def test_ltt_without_tolerated_attacks_starts_where_it_always_did():
    # the first threshold that could pass with no attack, then the first failure, as before the start rule took misses
    rng = np.random.default_rng(3)
    benign = np.cumsum(rng.integers(0, 4000, (300, 40)), axis=1)
    attack = np.cumsum(rng.binomial(1, 0.02, (300, 40)) * rng.integers(1, 4, (300, 40)), axis=1)
    dismissed = benign + attack
    first = np.where((clopper_pearson_upper(0, dismissed, 0.05) <= 1e-3).any(1),
                     (clopper_pearson_upper(0, dismissed, 0.05) <= 1e-3).argmax(1), 40)
    old = fixed_sequence(clopper_pearson_upper(attack, dismissed, 0.05) <= 1e-3, first)
    assert np.array_equal(ltt_iid(benign, attack, 1e-3, 0.05)["index"], old)
    assert np.array_equal(ltt_iid(benign, attack, 1e-3, 0.05, n=np.full(300, 10 ** 6), start_misses=0)["index"], old)
    with pytest.raises(ValueError, match="needs the size n"):
        ltt_iid(benign, attack, 1e-3, 0.05, start_misses=1)
    with pytest.raises(ValueError, match="start_misses"):
        ltt_iid(benign, attack, 1e-3, 0.05, n=np.full(300, 10 ** 6), start_misses=-1)


def test_ltt_start_tolerates_one_attack_among_the_first_dismissed_flows():
    # 4,525 flows dismissed at the first threshold with one attack among them, as for e12 on LITNET, end the plain
    # sequence at its start, while the guarded one starts at the second threshold, which passes with that attack and
    # dismisses 0.001 more of the 5 million flows, and walks on to the fourth
    benign = np.array([4524, 9500, 20000, 40000, 60000])
    attack = np.array([1, 1, 2, 4, 400])
    n = 5_000_000
    assert clopper_pearson_upper(1, 4525, 0.05) > 1e-3 >= clopper_pearson_upper(1, 9501, 0.05)
    assert ltt_iid(benign, attack, 1e-3, 0.05)["index"] == -1
    out = ltt_iid(benign, attack, 1e-3, 0.05, n=n, start_misses=2, start_slack=0.05)
    assert out["index"] == 3 and out["bound"] == clopper_pearson_upper(4, 40004, 0.05)
    # a slack below the extra 0.001 dismissed keeps the start, and the refusal
    assert ltt_iid(benign, attack, 1e-3, 0.05, n=n, start_misses=2, start_slack=0.0005)["index"] == -1


def test_estimate_prevalence_by_hand_and_at_its_edges():
    # benign share 0.5, attack share 0.1 and window share 0.46 at the threshold give a prevalence of 0.1
    out = estimate_prevalence(460, 1000, 500, 1000, 100, 1000, 0.01, 0.01, 0.005)
    assert np.isclose(out["plugin"], 0.1) and out["upper"] > 0.1
    b_up = 0.5 + dkw_margin(1000, 0.005)
    m_up = clopper_pearson_upper(100, 1000, 0.005)
    f_lo = 0.46 - dkw_margin(1000, 0.01)
    assert np.isclose(out["upper"], (b_up - f_lo) / (b_up - m_up))
    # a window share above the benign one gives 0, one below the attack share 1, and a benign share at or below
    # the attack share tells nothing
    out = estimate_prevalence(np.array([600, 0, 100]), 1000, 500, 1000, np.array([100, 100, 600]), 1000, 0.01,
                              0.01, 0.005)
    assert out["plugin"].tolist() == [0.0, 1.0, 1.0] and out["upper"].tolist()[1:] == [1.0, 1.0]
    b, m = beta_cdfs(beta.ppf(np.arange(1, 201) / 201, *BENIGN), BENIGN, FAMILIES)
    assert ks_threshold(b, np.array(SHARES) @ m) == int(np.argmax(b - np.array(SHARES) @ m))


def test_estimated_certifier_is_the_pooled_certificate_at_the_given_prevalence():
    grid = beta.ppf(np.arange(1, 201) / 201, *BENIGN)
    b, m = beta_cdfs(grid, BENIGN, FAMILIES)
    cal = draw_counts(np.random.default_rng(1), 3000000, 0.02, SHARES, b, m, 100)
    pooled, n = cal.family_below.sum(1), cal.n_family.sum(1)
    at = estimated_certifier(cal.benign_below, cal.n_benign, pooled, n, 1e-3, 0.01, 0.025)
    mine = at(np.full(100, 0.1))
    # the benign bound is the lower side of the two sided event at delta_benign, the family bound takes delta_miss
    same = certify(cal.benign_below, cal.n_benign, pooled[:, None], n[:, None], 1e-3, 0.03, 0.1, delta_benign=0.005)
    for key in ("index", "bound", "floor"):
        assert np.allclose(mine[key], same[key], equal_nan=True)
    assert (mine["index"] >= 0).all()
    # a lower prevalence certifies a looser threshold, and the prevalence may differ per calibration set
    lower = at(np.full(100, 0.01))["index"]
    assert (lower >= mine["index"]).all() and (lower > mine["index"]).any()
    mixed = at(np.where(np.arange(100) % 2 == 0, 0.01, 0.1))["index"]
    assert np.array_equal(mixed[::2], lower[::2]) and np.array_equal(mixed[1::2], mine["index"][1::2])


def test_upper_bound_on_the_prevalence_holds_under_label_shift_and_the_plug_in_does_not():
    # windows at prevalence 0.01 from the calibration distributions. the upper bound covers the prevalence and its
    # certificate keeps the violation share within delta while certifying far looser than pi_max, the plug in sits
    # at the prevalence and its certificate violates in many draws, since it claims nothing
    eps, draws, pi_w = 1e-3, 1500, 0.01
    grid = beta.ppf(np.arange(1, 201) / 201, *BENIGN)
    b, m = beta_cdfs(grid, BENIGN, FAMILIES)
    cal = draw_counts(np.random.default_rng(2), 3000000, 0.02, SHARES, b, m, draws)
    pooled, n = cal.family_below.sum(1), cal.n_family.sum(1)
    t = ks_threshold(b, np.array(SHARES) @ m)
    win = draw_counts(np.random.default_rng(3), 200000, pi_w, SHARES, b, m, draws)
    below = win.benign_below + win.family_below.sum(1)
    est = estimate_prevalence(below[:, t], 200000, cal.benign_below[:, t], cal.n_benign, pooled[:, t], n, 0.01,
                              0.01, 0.005)
    assert (est["upper"] >= est["plugin"]).all() and np.mean(est["upper"] < pi_w) <= 0.03
    assert abs(np.median(est["plugin"]) - pi_w) < 0.002
    at = estimated_certifier(cal.benign_below, cal.n_benign, pooled, n, eps, 0.01, 0.025)
    rate = for_from_rates(pi_w, np.array(SHARES) @ m, b)
    best = np.flatnonzero(rate <= eps)[-1]
    upper = at(est["upper"])["index"]
    plugin = at(est["plugin"])["index"]
    worst = at(np.full(draws, 0.1))["index"]
    violation_upper = np.mean(np.where(upper >= 0, rate[np.maximum(upper, 0)], 0) > eps)
    violation_plugin = np.mean(np.where(plugin >= 0, rate[np.maximum(plugin, 0)], 0) > eps)
    assert violation_upper <= 0.05 and violation_plugin >= violation_upper
    assert np.median(worst) < np.median(upper) <= np.median(plugin) <= best
    # a window whose attacks sit lower than the calibration mix at t is underestimated, since the estimate reads the
    # miss rate of the calibration mix, so both certificates lose their cover where the mix moves while the
    # certificate at pi_max keeps it
    low_family = [(4, 3), (4, 3), (4, 3), (4, 3)]
    _, m_low = beta_cdfs(grid, BENIGN, low_family)
    moved = draw_counts(np.random.default_rng(4), 200000, pi_w, SHARES, b, m_low, draws)
    below = moved.benign_below + moved.family_below.sum(1)
    est = estimate_prevalence(below[:, t], 200000, cal.benign_below[:, t], cal.n_benign, pooled[:, t], n, 0.01,
                              0.01, 0.005)
    assert np.mean(est["plugin"] < pi_w) > 0.9
    rate = for_from_rates(pi_w, np.array(SHARES) @ m_low, b)
    for index in (at(est["plugin"])["index"], at(est["upper"])["index"]):
        assert np.mean(np.where(index >= 0, rate[np.maximum(index, 0)], 0) > eps) > 0.9
    assert np.mean(np.where(worst >= 0, rate[np.maximum(worst, 0)], 0) > eps) == 0
