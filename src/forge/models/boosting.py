"""gradient boosted trees on the binary attack label, LightGBM, as a base scorer.

the score is the raw margin of the model. it orders the flows as the predicted attack probability does, which is all
the certificates use, and it does not round to 0 or 1 at the low end where the dismiss thresholds sit.
"""

import lightgbm as lgb
import numpy as np


def fit(x, y, x_valid, y_valid, params: dict, rounds: int, early_stopping: int, seed: int, columns: list[str],
        categorical: list[int]) -> lgb.Booster:
    """trains on x and y, and stops once the loss on the validation set has not improved for early_stopping rounds.

    categorical holds the positions of the columns whose values are category codes, nan where missing.
    """
    train = lgb.Dataset(x, label=y, feature_name=list(columns), categorical_feature=list(categorical))
    valid = lgb.Dataset(x_valid, label=y_valid, reference=train)
    booster = lgb.train({**params, "seed": seed}, train, num_boost_round=rounds, valid_sets=[valid],
                        valid_names=["selection"], callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
    # the binned training and validation data are not needed to score
    return booster.free_dataset()


def score(booster: lgb.Booster, x: np.ndarray) -> np.ndarray:
    # raw margins of the rounds up to the best one, rounded to float32, which keeps their order
    return booster.predict(x, raw_score=True, num_iteration=booster.best_iteration).astype(np.float32)
