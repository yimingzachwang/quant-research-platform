"""Tests for src/validation/splits.py."""

import pandas as pd
import pytest
from src.validation.splits import (
    TimeSplit,
    expanding_time_splits,
    rolling_time_splits,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ten_year_index() -> pd.DatetimeIndex:
    return pd.date_range("2010-01-01", "2019-12-31", freq="B")


@pytest.fixture()
def five_year_index() -> pd.DatetimeIndex:
    return pd.date_range("2015-01-01", "2019-12-31", freq="B")


# ---------------------------------------------------------------------------
# TimeSplit.slice
# ---------------------------------------------------------------------------


def test_timesplit_slice_train() -> None:
    idx = pd.date_range("2010-01-04", periods=100, freq="B")
    split = TimeSplit(
        split_index=0,
        train_start=idx[0],
        train_end=idx[49],
        test_start=idx[50],
        test_end=idx[99],
    )
    data = pd.DataFrame({"x": range(100)}, index=idx)
    sliced = split.slice(data, period="train")
    assert sliced.index[0] == idx[0]
    assert sliced.index[-1] == idx[49]


def test_timesplit_slice_test() -> None:
    idx = pd.date_range("2010-01-04", periods=100, freq="B")
    split = TimeSplit(
        split_index=0,
        train_start=idx[0],
        train_end=idx[49],
        test_start=idx[50],
        test_end=idx[99],
    )
    data = pd.Series(range(100), index=idx)
    sliced = split.slice(data, period="test")
    assert sliced.index[0] == idx[50]
    assert sliced.index[-1] == idx[99]


def test_timesplit_slice_invalid_period() -> None:
    idx = pd.date_range("2010-01-04", periods=10, freq="B")
    split = TimeSplit(0, idx[0], idx[4], idx[5], idx[9])
    with pytest.raises(ValueError, match="period must be"):
        split.slice(pd.Series(range(10), index=idx), period="validation")


# ---------------------------------------------------------------------------
# rolling_time_splits — basic properties
# ---------------------------------------------------------------------------


def test_rolling_returns_list(ten_year_index: pd.DatetimeIndex) -> None:
    splits = rolling_time_splits(ten_year_index, train_months=36, test_months=12)
    assert isinstance(splits, list)


def test_rolling_nonempty(ten_year_index: pd.DatetimeIndex) -> None:
    splits = rolling_time_splits(ten_year_index, train_months=36, test_months=12)
    assert len(splits) > 0


def test_rolling_split_types(ten_year_index: pd.DatetimeIndex) -> None:
    splits = rolling_time_splits(ten_year_index, train_months=36, test_months=12)
    for s in splits:
        assert isinstance(s, TimeSplit)


def test_rolling_split_indices_sequential(ten_year_index: pd.DatetimeIndex) -> None:
    splits = rolling_time_splits(ten_year_index, train_months=36, test_months=12)
    for i, s in enumerate(splits):
        assert s.split_index == i


def test_rolling_no_leakage(ten_year_index: pd.DatetimeIndex) -> None:
    splits = rolling_time_splits(ten_year_index, train_months=36, test_months=12)
    for s in splits:
        assert s.train_end < s.test_start


def test_rolling_train_precedes_test(ten_year_index: pd.DatetimeIndex) -> None:
    splits = rolling_time_splits(ten_year_index, train_months=36, test_months=12)
    for s in splits:
        assert s.train_start < s.test_start
        assert s.train_end < s.test_end


def test_rolling_splits_chronological(ten_year_index: pd.DatetimeIndex) -> None:
    splits = rolling_time_splits(ten_year_index, train_months=36, test_months=12)
    for i in range(len(splits) - 1):
        assert splits[i].test_start < splits[i + 1].test_start


def test_rolling_dates_in_index(ten_year_index: pd.DatetimeIndex) -> None:
    splits = rolling_time_splits(ten_year_index, train_months=36, test_months=12)
    index_set = set(ten_year_index)
    for s in splits:
        assert s.train_start in index_set
        assert s.train_end in index_set
        assert s.test_start in index_set
        assert s.test_end in index_set


def test_rolling_empty_when_data_too_short() -> None:
    short_idx = pd.date_range("2020-01-01", periods=50, freq="B")
    splits = rolling_time_splits(short_idx, train_months=36, test_months=12)
    assert splits == []


def test_rolling_step_defaults_to_test_months(ten_year_index: pd.DatetimeIndex) -> None:
    s1 = rolling_time_splits(ten_year_index, train_months=36, test_months=12, step_months=None)
    s2 = rolling_time_splits(ten_year_index, train_months=36, test_months=12, step_months=12)
    assert len(s1) == len(s2)
    for a, b in zip(s1, s2, strict=False):
        assert a.test_start == b.test_start


def test_rolling_gap_increases_separation(ten_year_index: pd.DatetimeIndex) -> None:
    no_gap = rolling_time_splits(ten_year_index, train_months=36, test_months=12, gap_days=0)
    with_gap = rolling_time_splits(ten_year_index, train_months=36, test_months=12, gap_days=30)
    for ng, wg in zip(no_gap, with_gap, strict=False):
        # test_start should be at least as late with gap as without
        assert wg.test_start >= ng.test_start


# ---------------------------------------------------------------------------
# expanding_time_splits — basic properties
# ---------------------------------------------------------------------------


def test_expanding_returns_list(ten_year_index: pd.DatetimeIndex) -> None:
    splits = expanding_time_splits(ten_year_index, min_train_months=24, test_months=12)
    assert isinstance(splits, list)


def test_expanding_nonempty(ten_year_index: pd.DatetimeIndex) -> None:
    splits = expanding_time_splits(ten_year_index, min_train_months=24, test_months=12)
    assert len(splits) > 0


def test_expanding_no_leakage(ten_year_index: pd.DatetimeIndex) -> None:
    splits = expanding_time_splits(ten_year_index, min_train_months=24, test_months=12)
    for s in splits:
        assert s.train_end < s.test_start


def test_expanding_train_always_starts_at_data_start(ten_year_index: pd.DatetimeIndex) -> None:
    splits = expanding_time_splits(ten_year_index, min_train_months=24, test_months=12)
    first_date = ten_year_index[0]
    for s in splits:
        assert s.train_start == first_date


def test_expanding_train_grows(ten_year_index: pd.DatetimeIndex) -> None:
    splits = expanding_time_splits(ten_year_index, min_train_months=24, test_months=12)
    for i in range(len(splits) - 1):
        assert splits[i].train_end < splits[i + 1].train_end


def test_expanding_test_windows_sequential(ten_year_index: pd.DatetimeIndex) -> None:
    splits = expanding_time_splits(ten_year_index, min_train_months=24, test_months=12)
    for i in range(len(splits) - 1):
        assert splits[i].test_end < splits[i + 1].test_start


def test_expanding_dates_in_index(ten_year_index: pd.DatetimeIndex) -> None:
    splits = expanding_time_splits(ten_year_index, min_train_months=24, test_months=12)
    index_set = set(ten_year_index)
    for s in splits:
        assert s.train_start in index_set
        assert s.train_end in index_set
        assert s.test_start in index_set
        assert s.test_end in index_set


def test_expanding_empty_when_data_too_short() -> None:
    short_idx = pd.date_range("2020-01-01", periods=50, freq="B")
    splits = expanding_time_splits(short_idx, min_train_months=36, test_months=12)
    assert splits == []


# ---------------------------------------------------------------------------
# No test/train overlap between consecutive splits
# ---------------------------------------------------------------------------


def test_rolling_no_test_overlap(ten_year_index: pd.DatetimeIndex) -> None:
    splits = rolling_time_splits(ten_year_index, train_months=36, test_months=12, step_months=12)
    for i in range(len(splits) - 1):
        assert splits[i].test_end < splits[i + 1].test_start


def test_expanding_no_test_overlap(ten_year_index: pd.DatetimeIndex) -> None:
    splits = expanding_time_splits(ten_year_index, min_train_months=24, test_months=12, step_months=12)
    for i in range(len(splits) - 1):
        assert splits[i].test_end < splits[i + 1].test_start
