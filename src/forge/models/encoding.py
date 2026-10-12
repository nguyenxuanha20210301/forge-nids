"""feature matrices for the scorers. every encoding is learned on the training set and applied unchanged to the later
sets, so nothing of the selection, calibration or deployment data reaches the features."""

import numpy as np
import pandas as pd


def _present(values: pd.Series) -> list[str]:
    # categories that occur in the rows, a categorical column may carry unused ones from other years
    if isinstance(values.dtype, pd.CategoricalDtype):
        values = values.cat.remove_unused_categories()
        return sorted(str(v) for v in values.cat.categories)
    return sorted(str(v) for v in pd.unique(values.dropna()))


def _codes(values: pd.Series, categories: list[str]) -> np.ndarray:
    # position of each value among the training categories, -1 for a missing value or a category unseen in training
    if not isinstance(values.dtype, pd.CategoricalDtype):
        values = values.astype("category")
    lookup = pd.Index(categories).get_indexer([str(v) for v in values.cat.categories])
    codes = values.cat.codes.to_numpy()
    if len(lookup) == 0:
        return np.full(len(codes), -1)
    return np.where(codes >= 0, lookup[codes], -1)


class Encoder:
    """numeric columns as they are and categorical columns as codes of the categories seen in training, in one float32
    matrix.

    missing maps a numeric column to the value that stands for a missing entry in it. that value, a missing category
    and a category first seen after training all become nan, which LightGBM treats as missing.
    """

    def __init__(self, numeric: list[str], categorical: list[str], missing: dict | None = None):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.missing = dict(missing or {})
        self.categories: dict[str, list[str]] | None = None

    @property
    def columns(self) -> list[str]:
        return self.numeric + self.categorical

    @property
    def categorical_index(self) -> list[int]:
        return list(range(len(self.numeric), len(self.columns)))

    def fit(self, frame: pd.DataFrame) -> "Encoder":
        # sorted, so the codes do not depend on the row order of the training set
        self.categories = {c: _present(frame[c]) for c in self.categorical}
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.categories is None:
            raise ValueError("the encoder is used before it was fitted")
        out = np.empty((len(frame), len(self.columns)), dtype=np.float32)
        for j, c in enumerate(self.numeric):
            raw = frame[c].to_numpy()
            out[:, j] = raw
            if c in self.missing:
                out[raw == self.missing[c], j] = np.nan
        for j, c in enumerate(self.categorical, start=len(self.numeric)):
            codes = _codes(frame[c], self.categories[c]).astype(np.float32)
            codes[codes < 0] = np.nan
            out[:, j] = codes
        return out

    def to_dict(self) -> dict:
        return {"numeric": self.numeric, "categorical": self.categorical, "missing": self.missing,
                "categories": self.categories}

    @classmethod
    def from_dict(cls, d: dict) -> "Encoder":
        encoder = cls(d["numeric"], d["categorical"], d["missing"])
        encoder.categories = d["categories"]
        return encoder
