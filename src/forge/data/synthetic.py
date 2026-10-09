"""synthetic scores with a known ground truth, for checks where the true false omission rate must be known."""

import numpy as np


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
