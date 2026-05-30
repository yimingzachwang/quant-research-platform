"""Tests for src.ml.diagnostics.turnover."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.ml.diagnostics.turnover import (
    average_turnover,
    signal_turnover,
    turnover_by_split,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B")


def _make_weights(n_dates: int = 10, n_assets: int = 3,
                  seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = _make_dates(n_dates)
    assets = [f"A{i}" for i in range(n_assets)]
    raw = np.abs(rng.standard_normal((n_dates, n_assets)))
    w = raw / raw.sum(axis=1, keepdims=True)
    return pd.DataFrame(w, index=dates, columns=assets)


# ---------------------------------------------------------------------------
# signal_turnover
# ---------------------------------------------------------------------------

class TestSignalTurnover:
    def test_returns_series(self):
        w = _make_weights()
        to = signal_turnover(w)
        assert isinstance(to, pd.Series)

    def test_first_row_is_nan(self):
        w = _make_weights()
        to = signal_turnover(w)
        assert pd.isna(to.iloc[0])

    def test_named_turnover(self):
        w = _make_weights()
        to = signal_turnover(w)
        assert to.name == "turnover"

    def test_length_matches_input(self):
        w = _make_weights(n_dates=15)
        to = signal_turnover(w)
        assert len(to) == 15

    def test_zero_change_gives_zero_turnover(self):
        dates = _make_dates(5)
        w = pd.DataFrame(
            np.tile([0.5, 0.3, 0.2], (5, 1)),
            index=dates, columns=["A", "B", "C"]
        )
        to = signal_turnover(w)
        # All rows after the first should be 0.0
        assert to.iloc[1:].max() == 0.0

    def test_full_rotation_gives_max_turnover(self):
        dates = _make_dates(3)
        w = pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            index=dates, columns=["A", "B"]
        )
        to = signal_turnover(w)
        # Each rotation: |1-0| + |0-1| = 2.0
        assert abs(to.iloc[1] - 2.0) < 1e-9
        assert abs(to.iloc[2] - 2.0) < 1e-9

    def test_nan_weights_treated_as_zero(self):
        dates = _make_dates(3)
        w = pd.DataFrame(
            [[np.nan, 1.0], [0.5, 0.5], [1.0, 0.0]],
            index=dates, columns=["A", "B"]
        )
        to = signal_turnover(w)
        # Row 1: |0.5 - 0.0| + |0.5 - 1.0| = 1.0
        assert abs(to.iloc[1] - 1.0) < 1e-9

    def test_values_nonnegative(self):
        w = _make_weights(n_dates=20)
        to = signal_turnover(w).dropna()
        assert (to >= 0.0).all()

    def test_index_preserved(self):
        w = _make_weights(n_dates=10)
        to = signal_turnover(w)
        assert to.index.equals(w.index)

    def test_single_row_first_is_nan(self):
        dates = _make_dates(1)
        w = pd.DataFrame([[0.5, 0.5]], index=dates, columns=["A", "B"])
        to = signal_turnover(w)
        assert pd.isna(to.iloc[0])


# ---------------------------------------------------------------------------
# average_turnover
# ---------------------------------------------------------------------------

class TestAverageTurnover:
    def test_returns_float(self):
        w = _make_weights()
        result = average_turnover(w)
        assert isinstance(result, float)

    def test_zero_change_gives_zero_average(self):
        dates = _make_dates(5)
        w = pd.DataFrame(
            np.tile([0.5, 0.5], (5, 1)),
            index=dates, columns=["A", "B"]
        )
        assert average_turnover(w) == 0.0

    def test_single_row_is_nan(self):
        dates = _make_dates(1)
        w = pd.DataFrame([[1.0, 0.0]], index=dates, columns=["A", "B"])
        assert np.isnan(average_turnover(w))

    def test_empty_dataframe_is_nan(self):
        w = pd.DataFrame(columns=["A", "B"])
        assert np.isnan(average_turnover(w))

    def test_value_nonnegative(self):
        w = _make_weights(n_dates=30)
        assert average_turnover(w) >= 0.0

    def test_consistent_with_signal_turnover(self):
        w = _make_weights(n_dates=20)
        to = signal_turnover(w).dropna()
        expected = float(to.mean())
        assert abs(average_turnover(w) - expected) < 1e-9


# ---------------------------------------------------------------------------
# turnover_by_split
# ---------------------------------------------------------------------------

class TestTurnoverBySplit:
    def test_empty_input_returns_empty_df(self):
        result = turnover_by_split([])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert set(result.columns) == {
            "mean_turnover", "max_turnover", "std_turnover", "n_periods"
        }

    def test_returns_dataframe(self):
        splits = [_make_weights(n_dates=10), _make_weights(n_dates=10, seed=1)]
        result = turnover_by_split(splits)
        assert isinstance(result, pd.DataFrame)

    def test_row_count_matches_splits(self):
        splits = [_make_weights(n_dates=10, seed=i) for i in range(4)]
        result = turnover_by_split(splits)
        assert len(result) == 4

    def test_indexed_by_split(self):
        splits = [_make_weights(n_dates=10, seed=i) for i in range(3)]
        result = turnover_by_split(splits)
        assert list(result.index) == [0, 1, 2]

    def test_required_columns(self):
        splits = [_make_weights(n_dates=10)]
        result = turnover_by_split(splits)
        for col in ["mean_turnover", "max_turnover", "std_turnover", "n_periods"]:
            assert col in result.columns

    def test_n_periods_correct(self):
        w = _make_weights(n_dates=10)
        result = turnover_by_split([w])
        # First row of signal_turnover is NaN → 9 valid periods
        assert result.loc[0, "n_periods"] == 9

    def test_mean_turnover_nonnegative(self):
        splits = [_make_weights(n_dates=15, seed=i) for i in range(3)]
        result = turnover_by_split(splits)
        assert (result["mean_turnover"].dropna() >= 0.0).all()

    def test_max_geq_mean(self):
        splits = [_make_weights(n_dates=15, seed=i) for i in range(3)]
        result = turnover_by_split(splits)
        clean = result.dropna()
        assert (clean["max_turnover"] >= clean["mean_turnover"]).all()

    def test_single_row_split_nan_stats(self):
        dates = _make_dates(1)
        w = pd.DataFrame([[1.0, 0.0]], index=dates, columns=["A", "B"])
        result = turnover_by_split([w])
        assert result.loc[0, "n_periods"] == 0
        assert np.isnan(result.loc[0, "mean_turnover"])

    def test_std_nan_for_single_period(self):
        # 2 rows → 1 valid turnover period → std with ddof=1 is NaN
        dates = _make_dates(2)
        w = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=dates, columns=["A", "B"])
        result = turnover_by_split([w])
        assert np.isnan(result.loc[0, "std_turnover"])


# ---------------------------------------------------------------------------
# Package import
# ---------------------------------------------------------------------------

def test_import_from_src_ml():
    from src.ml import average_turnover, signal_turnover, turnover_by_split
    assert callable(average_turnover)
    assert callable(signal_turnover)
    assert callable(turnover_by_split)
