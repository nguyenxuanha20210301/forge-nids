"""synthetic scores with a known ground truth, for checks where the true false omission rate must be known."""

from typing import NamedTuple

import numpy as np
from scipy.stats import beta


def scores_by_window(prevalences, n_per_window: int, seed: int, benign_ab=(2.0, 8.0), attack_ab=(6.0, 2.0)):
    # benign and attack scores come from two fixed beta distributions, only the prevalence moves
    rng = np.random.default_rng(seed)
    windows, labels, scores = [], [], []
    for w, pi in enumerate(prevalences):
        y = rng.random(n_per_window) < pi
        s = np.where(y, rng.beta(*attack_ab, n_per_window), rng.beta(*benign_ab, n_per_window))
        windows.append(np.full(n_per_window, w))
        labels.append(y.astype(np.int8))
        scores.append(s)
    return np.concatenate(windows), np.concatenate(labels), np.concatenate(scores)


class Counts(NamedTuple):
    n_benign: np.ndarray  # (draws,)
    n_family: np.ndarray  # (draws, K)
    benign_below: np.ndarray  # (draws, G)
    family_below: np.ndarray  # (draws, K, G)


def beta_cdfs(grid, benign, families):
    # true cdf of the benign class, shape (G,), and of each attack family, shape (K, G), at the grid thresholds
    return beta.cdf(grid, *benign), np.array([beta.cdf(grid, *ab) for ab in families])


def _bins(cdf):
    # probability of each interval between consecutive thresholds, with the open ends below t_1 and above t_G
    return np.diff(np.concatenate([[0.0], cdf, [1.0]]))


def draw_counts(rng, n: int, prevalence: float, shares, benign_cdf, family_cdf, draws: int) -> Counts:
    """counts at or below each grid threshold for independent calibration sets of n flows each.

    a flow is an attack with probability prevalence, and an attack belongs to family k with probability shares[k].
    given the class sizes, the counts in the intervals between thresholds are multinomial, so drawing them directly is
    exact and costs the same for any n.
    """
    shares = np.asarray(shares, dtype=float)
    sizes = rng.multinomial(n, np.concatenate([[1 - prevalence], prevalence * shares]), size=draws)
    g = len(benign_cdf)

    def below(class_sizes, cdf):
        return np.cumsum(rng.multinomial(class_sizes, _bins(cdf))[:, :g], axis=1)

    benign_below = below(sizes[:, 0], benign_cdf)
    family_below = np.stack([below(sizes[:, k + 1], cdf) for k, cdf in enumerate(family_cdf)], axis=1)
    return Counts(sizes[:, 0], sizes[:, 1:], benign_below, family_below)


def benign_path(start, end, windows: int, shift_at: int | None = None) -> np.ndarray:
    # beta parameters of the benign class in windows 0 to windows. window 0 lies before deployment and keeps start, the
    # deployment windows move to end in equal steps, or in one jump at window shift_at
    if shift_at is not None and not 1 <= shift_at <= windows:
        raise ValueError(f"shift_at is {shift_at}, expected a window from 1 to {windows}")
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    w = np.arange(windows + 1)
    u = w / windows if shift_at is None else (w >= shift_at).astype(float)
    return start + u[:, None] * (end - start)


def sample_flows(rng, n: int, prevalence: float, shares, benign, families):
    # n flows with their scores, labels and family index, the index is -1 for benign flows
    p = np.concatenate([[1 - prevalence], prevalence * np.asarray(shares, dtype=float)])
    family = rng.choice(len(p), size=n, p=p) - 1
    scores = np.empty(n)
    for k, ab in enumerate([benign, *families]):
        chosen = family == k - 1
        scores[chosen] = rng.beta(*ab, size=chosen.sum())
    return scores, (family >= 0).astype(np.int8), family
