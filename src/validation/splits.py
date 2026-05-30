"""Time-aware train/test split generators.

No sklearn dependency. All splits respect chronological ordering — training
data always precedes test data. Never shuffle time series.

Two generators are provided:

- rolling_time_splits  — fixed-width sliding train window
- expanding_time_splits — expanding train window (anchored at data start)

Both snap window boundaries to the nearest available dates in the supplied
index so callers never need to worry about weekend/holiday gaps.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    """A single train/test time window.

    Attributes:
        split_index: Zero-based position in the split sequence.
        train_start: First date of the training window (inclusive).
        train_end: Last date of the training window (inclusive).
        test_start: First date of the test window (inclusive).
        test_end: Last date of the test window (inclusive).
    """

    split_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def slice(
        self,
        data: pd.DataFrame | pd.Series,
        period: str = "test",
    ) -> pd.DataFrame | pd.Series:
        """Slice `data` to the train or test window (inclusive on both ends).

        Args:
            data: DataFrame or Series with a DatetimeIndex.
            period: ``'train'`` or ``'test'``.
        """
        if period == "train":
            return data.loc[self.train_start : self.train_end]
        if period == "test":
            return data.loc[self.test_start : self.test_end]
        raise ValueError(f"period must be 'train' or 'test', got {period!r}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _snap_left(ts: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    """Return the first index date >= ts."""
    pos = index.searchsorted(ts, side="left")
    pos = min(pos, len(index) - 1)
    return index[pos]


def _snap_right(ts: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    """Return the last index date <= ts."""
    pos = index.searchsorted(ts, side="right") - 1
    pos = max(pos, 0)
    return index[pos]


# ---------------------------------------------------------------------------
# Public split generators
# ---------------------------------------------------------------------------


def rolling_time_splits(
    index: pd.DatetimeIndex,
    train_months: int,
    test_months: int,
    step_months: int | None = None,
    gap_days: int = 0,
) -> list[TimeSplit]:
    """Generate rolling (fixed-width) train/test time splits.

    Each split advances the window by ``step_months`` calendar months.
    The training window width is always ``train_months`` months.

    Example with train=60mo, test=12mo, step=12mo, gap=0::

        split 0: train=[2000-01, 2004-12], test=[2005-01, 2005-12]
        split 1: train=[2001-01, 2005-12], test=[2006-01, 2006-12]
        ...

    Args:
        index: DatetimeIndex of available data (must be monotonic).
        train_months: Training window length in calendar months.
        test_months: Test window length in calendar months.
        step_months: Advance per split in months (defaults to test_months).
        gap_days: Calendar days between train_end and test_start.
                  Use > 0 to add a buffer against leakage for strategies
                  with prediction horizons (e.g. 21 days for monthly signals).

    Returns:
        Ordered list of TimeSplit objects. Empty list if no valid splits fit
        in the data.
    """
    if step_months is None:
        step_months = test_months

    train_offset = pd.DateOffset(months=train_months)
    test_offset = pd.DateOffset(months=test_months)
    step_offset = pd.DateOffset(months=step_months)
    gap = pd.Timedelta(days=gap_days)

    data_end = index[-1]
    splits: list[TimeSplit] = []
    split_idx = 0
    train_start_cal = index[0]

    while True:
        train_end_cal = train_start_cal + train_offset - pd.Timedelta(days=1)
        test_start_cal = train_end_cal + pd.Timedelta(days=1) + gap
        test_end_cal = test_start_cal + test_offset - pd.Timedelta(days=1)

        if test_end_cal > data_end:
            break

        ts_train_start = _snap_left(train_start_cal, index)
        ts_train_end = _snap_right(train_end_cal, index)
        ts_test_start = _snap_left(test_start_cal, index)
        ts_test_end = _snap_right(test_end_cal, index)

        # Require non-empty windows and strict chronological ordering
        if (
            ts_train_start <= ts_train_end
            and ts_test_start <= ts_test_end
            and ts_train_end < ts_test_start
        ):
            splits.append(
                TimeSplit(
                    split_index=split_idx,
                    train_start=ts_train_start,
                    train_end=ts_train_end,
                    test_start=ts_test_start,
                    test_end=ts_test_end,
                )
            )
            split_idx += 1

        train_start_cal = train_start_cal + step_offset

    return splits


def expanding_time_splits(
    index: pd.DatetimeIndex,
    min_train_months: int,
    test_months: int,
    step_months: int | None = None,
    gap_days: int = 0,
) -> list[TimeSplit]:
    """Generate expanding (anchored) train/test time splits.

    The train window always starts at the first date in ``index`` and grows
    with each split; each test window is a fixed-length block sliding forward.

    Example with min_train=24mo, test=12mo, step=12mo::

        split 0: train=[2000-01, 2001-12], test=[2002-01, 2002-12]
        split 1: train=[2000-01, 2002-12], test=[2003-01, 2003-12]
        ...

    Args:
        index: DatetimeIndex of available data (must be monotonic).
        min_train_months: Minimum training window in calendar months.
        test_months: Test window length in calendar months.
        step_months: Advance per split in months (defaults to test_months).
        gap_days: Calendar days between train_end and test_start.

    Returns:
        Ordered list of TimeSplit objects.
    """
    if step_months is None:
        step_months = test_months

    test_offset = pd.DateOffset(months=test_months)
    step_offset = pd.DateOffset(months=step_months)
    gap = pd.Timedelta(days=gap_days)

    data_start = index[0]
    data_end = index[-1]
    splits: list[TimeSplit] = []
    split_idx = 0

    test_start_cal = data_start + pd.DateOffset(months=min_train_months) + gap

    while True:
        test_end_cal = test_start_cal + test_offset - pd.Timedelta(days=1)

        if test_end_cal > data_end:
            break

        train_end_cal = test_start_cal - gap - pd.Timedelta(days=1)

        ts_train_start = _snap_left(data_start, index)
        ts_train_end = _snap_right(train_end_cal, index)
        ts_test_start = _snap_left(test_start_cal, index)
        ts_test_end = _snap_right(test_end_cal, index)

        if (
            ts_train_start <= ts_train_end
            and ts_test_start <= ts_test_end
            and ts_train_end < ts_test_start
        ):
            splits.append(
                TimeSplit(
                    split_index=split_idx,
                    train_start=ts_train_start,
                    train_end=ts_train_end,
                    test_start=ts_test_start,
                    test_end=ts_test_end,
                )
            )
            split_idx += 1

        test_start_cal = test_start_cal + step_offset

    return splits
