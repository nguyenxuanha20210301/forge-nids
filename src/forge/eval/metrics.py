"""operational metrics of a dismiss policy. a flow is dismissed when its score is at or below the threshold."""

import numpy as np
from scipy.stats import rankdata

from forge.certify.bounds import clopper_pearson_lower, clopper_pearson_upper


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


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    # probability that an attack scores above a benign flow, a tie counting one half, from the average ranks of all
    # scores. nan when one class is empty
    attack = np.asarray(labels) == 1
    n1 = int(attack.sum())
    n0 = len(attack) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[attack].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def share_with_interval(count: int, total: int, prefix: str = "") -> dict:
    # share of draws, for example of draws violating a claim, with a two sided 95% Clopper-Pearson interval
    return {f"{prefix}rate": count / total, f"{prefix}low": clopper_pearson_lower(count, total, 0.025),
            f"{prefix}high": clopper_pearson_upper(count, total, 0.025)}
