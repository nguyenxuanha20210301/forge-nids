"""exact binomial confidence bounds and the one sided DKW bound on a cdf.

the binomial bounds take scalars or arrays of counts, and return a float for scalars and an array otherwise.
"""

import numpy as np
from scipy.stats import beta


def _result(out: np.ndarray):
    return out if out.ndim else float(out)


def clopper_pearson_upper(k, n, delta: float):
    # one sided upper bound on a binomial proportion, holds with probability at least 1 minus delta
    k, n = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(n, dtype=float))
    out = np.ones(k.shape)
    inside = k < n
    out[inside] = beta.ppf(1 - delta, k[inside] + 1, n[inside] - k[inside])
    return _result(out)


def clopper_pearson_lower(k, n, delta: float):
    k, n = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(n, dtype=float))
    out = np.zeros(k.shape)
    inside = (k > 0) & (n > 0)
    out[inside] = beta.ppf(delta, k[inside], n[inside] - k[inside] + 1)
    return _result(out)


def dkw_margin(n, delta: float):
    # the empirical cdf of n draws exceeds the true cdf by more than this margin at some point with probability at most
    # delta, one sided DKW with the Massart constant, which holds for any distribution and delta up to one half
    if not 0 < delta <= 0.5:
        raise ValueError(f"the one sided DKW bound needs 0 < delta <= 0.5, got {delta}")
    n = np.asarray(n, dtype=float)
    out = np.full(n.shape, np.inf)
    np.divide(np.log(1 / delta), 2 * n, out=out, where=n > 0)
    return _result(np.sqrt(out))
