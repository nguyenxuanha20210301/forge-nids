"""temporal splits. rows are never shuffled across time."""

import numpy as np
import pandas as pd


def _periods(timestamps, freq: str) -> pd.Series:
    periods = pd.to_datetime(pd.Series(timestamps)).dt.to_period(freq)
    if periods.isna().any():
        raise ValueError("timestamps hold missing values, which belong to no window")
    return periods


def assign_windows(timestamps, freq: str) -> np.ndarray:
    # calendar windows such as D for days, W for weeks starting on monday, M for months, counted from the
    # window of the earliest timestamp, so an empty window keeps its place and ids follow calendar time
    periods = _periods(timestamps, freq)
    codes, uniques = pd.factorize(periods, sort=True)
    offsets = np.array([(p - uniques[0]).n for p in uniques], dtype=np.int64)
    return offsets[codes]


def calendar_windows(timestamps, freq: str) -> pd.DataFrame:
    # every calendar window from the one holding the earliest timestamp to the one holding the latest,
    # empty ones included, each covering start up to but not including end
    periods = _periods(timestamps, freq)
    span = pd.period_range(periods.min(), periods.max(), freq=freq)
    return pd.DataFrame({"window": np.arange(len(span)), "start": span.start_time, "end": (span + 1).start_time})


def split_by_windows(windows: np.ndarray, roles: dict) -> dict:
    # roles maps a role such as train or calibrate to a list of window ids
    used = [w for ids in roles.values() for w in ids]
    if len(used) != len(set(used)):
        raise ValueError("a window is assigned to more than one role")
    return {role: np.isin(windows, ids) for role, ids in roles.items()}
