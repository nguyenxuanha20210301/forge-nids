"""exact binomial confidence bounds."""

from scipy.stats import beta


def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    # one sided upper bound on a binomial proportion, holds with probability at least 1 minus delta
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(1 - delta, k + 1, n - k))


def clopper_pearson_lower(k: int, n: int, delta: float) -> float:
    if n == 0 or k == 0:
        return 0.0
    return float(beta.ppf(delta, k, n - k + 1))
