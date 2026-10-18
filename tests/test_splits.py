import numpy as np
import pandas as pd
import pytest

from forge.data.splits import assign_windows, calendar_windows, order_blocks, split_by_windows


def stamps(*values):
    return pd.Series(pd.to_datetime(list(values)))


def test_week_starts_on_monday():
    assert assign_windows(stamps("2020-01-05 23:59:59", "2020-01-06 00:00:00"), "W").tolist() == [0, 1]


def test_empty_windows_keep_their_place():
    t = stamps("2020-01-06 08:00:00", "2020-01-27 08:00:00", "2020-01-07 09:00:00")
    assert assign_windows(t, "W").tolist() == [0, 3, 0]
    table = calendar_windows(t, "W")
    assert table["window"].tolist() == [0, 1, 2, 3]
    assert table["start"].dt.strftime("%Y-%m-%d").tolist() == ["2020-01-06", "2020-01-13", "2020-01-20", "2020-01-27"]
    assert (table["end"] - table["start"] == pd.Timedelta(days=7)).all()


def test_month_boundary():
    t = stamps("2020-01-31 23:59:59", "2020-02-01 00:00:00", "2020-04-01 00:00:00")
    assert assign_windows(t, "M").tolist() == [0, 1, 3]


def test_window_of_a_row_does_not_depend_on_row_order():
    t = stamps("2020-01-01 10:00:00", "2020-01-20 10:00:00", "2020-01-09 10:00:00", "2019-12-30 00:00:00")
    order = np.array([2, 0, 3, 1])
    assert (assign_windows(t.iloc[order], "W") == assign_windows(t, "W")[order]).all()


def test_missing_timestamp_is_rejected():
    with pytest.raises(ValueError, match="missing"):
        assign_windows(pd.Series(pd.to_datetime(["2020-01-01", None])), "W")


def test_a_window_has_one_role():
    with pytest.raises(ValueError):
        split_by_windows(np.arange(5), {"train": [0, 1], "calibrate": [1, 2]})


def test_order_blocks_split_each_group_into_equal_runs():
    groups = np.array([2006] * 10 + [2007] * 4 + [2008] * 2)
    blocks = order_blocks(groups, 4)
    assert blocks[:10].tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 3, 3]
    assert blocks[10:14].tolist() == [0, 1, 2, 3]
    assert blocks[14:].tolist() == [0, 2]
    with pytest.raises(ValueError):
        order_blocks(np.array([1, 1, 2, 1]), 2)
    with pytest.raises(ValueError):
        order_blocks(groups, 0)


def test_fixed_windows_start_at_the_same_clock_times_every_day():
    from forge.data.splits import fixed_window_table, fixed_windows

    t = stamps("2016-07-27 13:43:21", "2016-07-27 17:59:59", "2016-07-27 18:00:00", "2016-07-29 00:10:00")
    assert fixed_windows(t, "6h").tolist() == [0, 0, 1, 6]
    table = fixed_window_table(t, "6h")
    assert table["window"].tolist() == list(range(7))
    assert table["start"].dt.strftime("%Y-%m-%d %H:%M").tolist()[:3] == \
        ["2016-07-27 12:00", "2016-07-27 18:00", "2016-07-28 00:00"]
    assert (table["end"] - table["start"] == pd.Timedelta(hours=6)).all()
    assert fixed_windows(t, "1D").tolist() == [0, 0, 0, 2]
    order = np.array([3, 1, 0, 2])
    assert (fixed_windows(t.iloc[order], "6h") == fixed_windows(t, "6h")[order]).all()
    with pytest.raises(ValueError, match="missing"):
        fixed_windows(pd.Series(pd.to_datetime(["2016-07-27", None])), "6h")
    with pytest.raises(ValueError):
        fixed_windows(t, "W")
    with pytest.raises(ValueError, match="positive"):
        fixed_windows(t, "-6h")
