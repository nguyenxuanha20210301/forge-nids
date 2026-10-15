"""the base scorers of the scorer comparison behind one interface, each fit on the training rows of a dataset and
scoring any rows as a logit, so the certificate and the combiners see every scorer the same way.

the supervised scorers are LightGBM and XGBoost on the encoded matrix of the Encoder, with the categorical columns as
categories, and ExtraTrees, a two layer MLP and logistic regression on the dense matrix of Dense below. IsolationForest
is the novelty signal, fit on the benign training rows alone, and its score is the anomaly score, larger for a flow
that the trees isolate sooner. the certificate reads the scores only through their order, so a logit, a probability or
an anomaly score serve it alike.
"""

import time

import numpy as np

from forge.models import boosting

SUPERVISED = ("lightgbm", "xgboost", "extra_trees", "mlp", "logistic")
NOVELTY = ("isolation_forest",)
BASE = SUPERVISED + NOVELTY
LABELS = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "extra_trees": "ExtraTrees", "mlp": "MLP",
          "logistic": "logistic regression", "isolation_forest": "IsolationForest"}
# scorers whose probability is a logistic transform of the score, so a calibration error can be read from it
PROBABILITY = ("lightgbm", "xgboost", "extra_trees", "mlp", "logistic")
# rows scored at a time by the models of scikit-learn, which build a dense copy of what they score
CHUNK = 1_000_000
CLIP = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    # a probability of exactly 0 or 1, which a forest gives a flow all its trees agree on, keeps its order at the clip
    p = np.clip(np.asarray(p, dtype=np.float64), CLIP, 1 - CLIP)
    return np.log(p / (1 - p)).astype(np.float32)


class Dense:
    """the dense matrix of the linear and neural scorers from the encoded matrix of the Encoder.

    a numeric column takes log1p when every training value is at or above 0, since counts and bytes span orders of
    magnitude, then is standardized on the training rows, and a missing value becomes 0, the mean. a categorical column
    becomes one indicator per code seen in training, so a code unseen in training or missing gives all zeros.
    """

    def __init__(self, categorical_index: list[int]):
        self.categorical_index = list(categorical_index)
        self.log = None
        self.mean = None
        self.scale = None
        self.codes = None

    def fit(self, x: np.ndarray) -> "Dense":
        numeric = [j for j in range(x.shape[1]) if j not in self.categorical_index]
        values = x[:, numeric].astype(np.float64)
        self.log = np.array([np.nanmin(values[:, k]) >= 0 for k in range(len(numeric))]) if len(numeric) else \
            np.zeros(0, dtype=bool)
        values[:, self.log] = np.log1p(values[:, self.log])
        self.mean = np.nanmean(values, axis=0) if len(numeric) else np.zeros(0)
        scale = np.nanstd(values, axis=0) if len(numeric) else np.zeros(0)
        self.scale = np.where(scale > 0, scale, 1.0)
        self.codes = [int(np.nanmax(x[:, j])) + 1 if np.isfinite(x[:, j]).any() else 0 for j in self.categorical_index]
        return self

    @property
    def width(self) -> int:
        return len(self.mean) + sum(self.codes)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise ValueError("the dense features are used before they were fitted")
        numeric = [j for j in range(x.shape[1]) if j not in self.categorical_index]
        out = np.zeros((len(x), self.width), dtype=np.float32)
        values = x[:, numeric].astype(np.float64)
        values[:, self.log] = np.log1p(np.maximum(values[:, self.log], 0))
        values = (values - self.mean) / self.scale
        out[:, :len(numeric)] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        offset = len(numeric)
        for j, n in zip(self.categorical_index, self.codes):
            code = x[:, j]
            hit = np.isfinite(code) & (code >= 0) & (code < n)
            out[np.flatnonzero(hit), offset + code[hit].astype(int)] = 1.0
            offset += n
        return out


def _xgboost_matrix(x, y, categorical_index):
    import xgboost as xgb

    kinds = ["c" if j in categorical_index else "q" for j in range(x.shape[1])]
    return xgb.DMatrix(x, label=y, feature_types=kinds, enable_categorical=True, missing=np.nan)


def fit(name: str, x, y, x_sel, y_sel, cfg: dict, seed: int, columns: list[str], categorical_index: list[int]):
    """fits the base scorer name on x and y, with x_sel and y_sel to stop early where the model stops early, and
    returns what score needs, a model and the dense features where the model takes them."""
    if name == "lightgbm":
        return {"model": boosting.fit(x, y, x_sel, y_sel, cfg["params"], cfg["rounds"], cfg["early_stopping"], seed,
                                      columns, categorical_index)}
    if name == "xgboost":
        import xgboost as xgb

        train = _xgboost_matrix(x, y, categorical_index)
        sel = _xgboost_matrix(x_sel, y_sel, categorical_index)
        booster = xgb.train({**cfg["params"], "seed": seed}, train, num_boost_round=cfg["rounds"],
                            evals=[(sel, "selection")], early_stopping_rounds=cfg["early_stopping"],
                            verbose_eval=False)
        return {"model": booster}
    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        model = ExtraTreesClassifier(n_estimators=cfg["n_estimators"], min_samples_leaf=cfg["min_samples_leaf"],
                                     max_features=cfg.get("max_features", "sqrt"), n_jobs=-1, random_state=seed)
        # a missing value takes a code of its own below every value, which a tree can split off
        model.fit(np.nan_to_num(x, nan=-1.0), y)
        return {"model": model}
    dense = Dense(categorical_index).fit(x)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier

        model = MLPClassifier(hidden_layer_sizes=tuple(cfg["hidden"]), batch_size=cfg["batch_size"],
                              max_iter=cfg["max_iter"], learning_rate_init=cfg["learning_rate"], early_stopping=True,
                              n_iter_no_change=cfg.get("patience", 3), random_state=seed)
        model.fit(dense.transform(x), y)
        return {"model": model, "dense": dense}
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=cfg["c"], max_iter=cfg["max_iter"])
        model.fit(dense.transform(x), y)
        return {"model": model, "dense": dense}
    if name == "isolation_forest":
        from sklearn.ensemble import IsolationForest

        model = IsolationForest(n_estimators=cfg["n_estimators"], max_samples=cfg.get("max_samples", "auto"),
                                n_jobs=-1, random_state=seed)
        model.fit(dense.transform(x[np.asarray(y) == 0]))
        return {"model": model, "dense": dense}
    raise ValueError(f"unknown base scorer {name}, expected one of {', '.join(BASE)}")


def _score_chunk(name: str, fitted: dict, x: np.ndarray, categorical_index: list[int]) -> np.ndarray:
    model = fitted["model"]
    if name == "lightgbm":
        return boosting.score(model, x)
    if name == "xgboost":
        import xgboost as xgb

        kinds = ["c" if j in categorical_index else "q" for j in range(x.shape[1])]
        matrix = xgb.DMatrix(x, feature_types=kinds, enable_categorical=True, missing=np.nan)
        return model.predict(matrix, output_margin=True,
                             iteration_range=(0, model.best_iteration + 1)).astype(np.float32)
    if name == "extra_trees":
        return logit(model.predict_proba(np.nan_to_num(x, nan=-1.0))[:, 1])
    dense = fitted["dense"].transform(x)
    if name == "mlp":
        return logit(model.predict_proba(dense)[:, 1])
    if name == "logistic":
        return model.decision_function(dense).astype(np.float32)
    if name == "isolation_forest":
        return (-model.score_samples(dense)).astype(np.float32)
    raise ValueError(f"unknown base scorer {name}")


def score(name: str, fitted: dict, x: np.ndarray, categorical_index: list[int]) -> np.ndarray:
    # chunks keep the dense copy of what a model scores at a bounded size
    out = np.empty(len(x), dtype=np.float32)
    for lo in range(0, len(x), CHUNK):
        out[lo:lo + CHUNK] = _score_chunk(name, fitted, x[lo:lo + CHUNK], categorical_index)
    return out


def rounds_used(name: str, fitted: dict) -> int | None:
    # the boosting rounds up to the best one, where a model stops early
    if name == "lightgbm":
        return int(fitted["model"].best_iteration)
    if name == "xgboost":
        return int(fitted["model"].best_iteration) + 1
    if name == "mlp":
        return int(fitted["model"].n_iter_)
    return None


def timed(function, *args, **kwargs):
    # the result of a call and the seconds it took
    t0 = time.perf_counter()
    out = function(*args, **kwargs)
    return out, time.perf_counter() - t0
