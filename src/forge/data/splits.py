"""temporal splits. rows are never shuffled across time."""

import numpy as np
import pandas as pd


def assign_windows(timestamps: pd.Series, freq: str) -> np.ndarray:
    # consecutive calendar windows, for example freq D for days or W for weeks
    periods = pd.to_datetime(timestamps).dt.to_period(freq)
    codes, _ = pd.factorize(periods, sort=True)
    return codes


def split_by_windows(windows: np.ndarray, roles: dict) -> dict:
    # roles maps a role such as train or calibrate to a list of window ids
    used = [w for ids in roles.values() for w in ids]
    if len(used) != len(set(used)):
        raise ValueError("a window is assigned to more than one role")
    return {role: np.isin(windows, ids) for role, ids in roles.items()}
