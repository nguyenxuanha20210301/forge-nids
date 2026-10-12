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


def order_blocks(groups, per_group: int) -> np.ndarray:
    # blocks of equal row count within each group, in row order, for data without clock time. the rows of a group must
    # sit together, and a group with fewer rows than blocks leaves some block numbers unused
    if per_group < 1:
        raise ValueError(f"per_group is {per_group}, expected 1 or more")
    groups = np.asarray(groups)
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    if len(np.unique(groups[starts])) != len(starts):
        raise ValueError("the rows of a group are not consecutive")
    lengths = np.diff(np.r_[starts, len(groups)])
    run = np.repeat(np.arange(len(starts)), lengths)
    position = np.arange(len(groups)) - starts[run]
    return position * per_group // lengths[run]
