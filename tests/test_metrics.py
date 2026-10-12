import numpy as np
import pytest

from forge.eval.metrics import auc


def test_auc_counts_a_tie_as_one_half():
    # attack 2 beats benign 1 and ties benign 2, attack 3 beats both, 3.5 of 4 pairs
    assert auc(np.array([1, 2, 2, 3]), np.array([0, 0, 1, 1])) == pytest.approx(0.875)
    assert auc(np.array([3, 2, 1]), np.array([0, 0, 1])) == 0
    assert np.isnan(auc(np.array([1.0, 2.0]), np.array([1, 1])))


def test_auc_matches_the_share_of_pairs():
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 20, 300).astype(np.float32)
    labels = (rng.random(300) < 0.4).astype(int)
    a, b = scores[labels == 1][:, None], scores[labels == 0][None, :]
    assert auc(scores, labels) == pytest.approx(np.mean((a > b) + 0.5 * (a == b)))
