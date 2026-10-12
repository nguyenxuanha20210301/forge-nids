import numpy as np
import pandas as pd
import pytest

from forge.eval.metrics import auc
from forge.models import boosting
from forge.models.encoding import Encoder


def test_encoder_codes_only_the_categories_seen_in_training():
    train = pd.DataFrame({"x": np.array([1, -1, 3], dtype=np.int16),
                          "s": pd.Categorical(["b", "a", "b"], categories=["a", "b", "z"])})
    enc = Encoder(["x"], ["s"], {"x": -1}).fit(train)
    # z is a category of the column but occurs in no training row, so it gets no code
    assert enc.categories == {"s": ["a", "b"]}
    assert enc.columns == ["x", "s"] and enc.categorical_index == [1]
    later = pd.DataFrame({"x": np.array([5, -1], dtype=np.int16), "s": pd.Categorical(["z", "a"])})
    out = enc.transform(later)
    assert out.dtype == np.float32 and out.shape == (2, 2)
    assert out[0, 0] == 5 and np.isnan(out[1, 0])
    assert np.isnan(out[0, 1]) and out[1, 1] == 0
    # neither the order of the training rows nor the order of the categories of a later frame changes the codes
    assert Encoder(["x"], ["s"], {"x": -1}).fit(train.iloc[::-1]).categories == enc.categories
    reordered = later.assign(s=pd.Categorical(["z", "a"], categories=["z", "a"]))
    assert np.array_equal(enc.transform(reordered), out, equal_nan=True)
    plain = later.assign(s=["z", "a"])
    assert np.array_equal(enc.transform(plain), out, equal_nan=True)
    assert np.array_equal(Encoder.from_dict(enc.to_dict()).transform(later), out, equal_nan=True)
    with pytest.raises(ValueError):
        Encoder(["x"], ["s"]).transform(later)


def labeled(n, seed):
    # attacks have a higher rate and use service a more often, the bin column is noise with -1 for missing
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(np.int8)
    service = np.where((y == 1) & (rng.random(n) < 0.5), "a", rng.choice(["a", "b", "c"], n))
    return pd.DataFrame({
        "rate": np.where(y == 1, rng.normal(0.65, 0.15, n), rng.normal(0.35, 0.15, n)).astype(np.float32),
        "bin": rng.integers(-1, 20, n).astype(np.int16),
        "service": pd.Categorical(service),
    }), y


def test_boosted_scores_rank_attacks_higher_and_repeat_with_the_seed():
    (train, y_train), (valid, y_valid), (test, y_test) = labeled(4000, 0), labeled(2000, 1), labeled(2000, 2)
    enc = Encoder(["rate", "bin"], ["service"], {"bin": -1}).fit(train)
    params = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 7, "min_data_in_leaf": 20,
              "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "deterministic": True,
              "force_col_wise": True, "verbosity": -1}

    def fit(seed):
        return boosting.fit(enc.transform(train), y_train, enc.transform(valid), y_valid, params, 300, 10, seed,
                            enc.columns, enc.categorical_index)

    booster = fit(0)
    scores = boosting.score(booster, enc.transform(test))
    assert scores.dtype == np.float32 and len(scores) == len(test)
    assert auc(scores, y_test) > 0.9
    assert booster.best_iteration < 300
    assert np.array_equal(boosting.score(fit(0), enc.transform(test)), scores)
    # the margin keeps the order of the predicted probability
    probability = booster.predict(enc.transform(test), num_iteration=booster.best_iteration)
    assert (np.diff(probability[np.argsort(scores, kind="stable")]) >= -1e-6).all()
