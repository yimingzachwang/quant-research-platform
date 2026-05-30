"""Tests for src/cleaning/timestamps.py."""

import pandas as pd
import pytest

from src.cleaning.timestamps import remove_duplicate_timestamps, sort_time_index


@pytest.fixture()
def unsorted_df() -> pd.DataFrame:
    idx = pd.to_datetime(["2023-01-03", "2023-01-01", "2023-01-02"])
    return pd.DataFrame({"close": [103.0, 101.0, 102.0]}, index=idx)


@pytest.fixture()
def df_with_dupes() -> pd.DataFrame:
    idx = pd.to_datetime(
        ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-03", "2023-01-03"]
    )
    return pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)


def test_sort_time_index_orders_ascending(unsorted_df: pd.DataFrame) -> None:
    result = sort_time_index(unsorted_df)
    assert list(result.index) == sorted(result.index)


def test_sort_time_index_returns_copy(unsorted_df: pd.DataFrame) -> None:
    original_order = list(unsorted_df.index)
    sort_time_index(unsorted_df)
    assert list(unsorted_df.index) == original_order


def test_remove_duplicates_keep_last(df_with_dupes: pd.DataFrame) -> None:
    result = remove_duplicate_timestamps(df_with_dupes, keep="last")
    assert result.index.is_unique
    # Last occurrence for 2023-01-01 is close=2.0
    assert result.loc["2023-01-01", "close"] == 2.0
    # Last occurrence for 2023-01-03 is close=5.0
    assert result.loc["2023-01-03", "close"] == 5.0


def test_remove_duplicates_keep_first(df_with_dupes: pd.DataFrame) -> None:
    result = remove_duplicate_timestamps(df_with_dupes, keep="first")
    assert result.index.is_unique
    assert result.loc["2023-01-01", "close"] == 1.0
    assert result.loc["2023-01-03", "close"] == 4.0


def test_remove_duplicates_invalid_keep(df_with_dupes: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="keep must be"):
        remove_duplicate_timestamps(df_with_dupes, keep="middle")  # type: ignore[arg-type]


def test_remove_duplicates_no_dupes_unchanged() -> None:
    idx = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    result = remove_duplicate_timestamps(df)
    assert len(result) == len(df)
