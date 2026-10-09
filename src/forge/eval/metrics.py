"""operational metrics of a dismiss policy. a flow is dismissed when its score is at or below the threshold."""

import numpy as np


def dismiss_rate(scores: np.ndarray, t: float) -> float:
    return float(np.mean(scores <= t))


def false_omission_rate(scores: np.ndarray, labels: np.ndarray, t: float) -> float:
    # share of attacks among dismissed flows, nan when nothing is dismissed
    dismissed = scores <= t
    if not dismissed.any():
        return float("nan")
    return float(labels[dismissed].mean())


def miss_rate(scores: np.ndarray, labels: np.ndarray, t: float) -> float:
    # share of attacks that get dismissed, the false negative rate of the escalate decision
    attacks = labels == 1
    if not attacks.any():
        return float("nan")
    return float(np.mean(scores[attacks] <= t))
