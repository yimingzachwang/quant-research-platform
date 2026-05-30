"""Tests for src.ml.diagnostics.prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.diagnostics.prediction import (
    information_coefficient,
    prediction_correlation,
    prediction_quantiles,
    rolling_directional_accuracy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B")


# ---------------------------------------------------------------------------
# prediction_correlation
# ---------------------------------------------------------------------------

class TestPredictionCorrelation:
    def test_perfect_correlation(self):
        idx = _make_dates(50)
        s = pd.Series(np.arange(50, dtype=float), index=idx)
        r = prediction_correlation(s, s)
        assert abs(r - 1.0) < 1e-9

    def test_perfect_negative_correlation(self):
        idx = _make_dates(50)
        actual = pd.Series(np.arange(50, dtype=float), index=idx)
        predicted = pd.Series(-np.arange(50, dtype=float), index=idx)
        r = prediction_correlation(actual, predicted)
        assert abs(r - (-1.0)) < 1e-9

    def test_returns_float(self):
        idx = _make_dates(30)
        a = pd.Series(np.random.randn(30), index=idx)
        p = pd.Series(np.random.randn(30), index=idx)
        r = prediction_correlation(a, p)
        assert isinstance(r, float)

    def test_range_minus1_to_1(self):
        idx = _make_dates(100)
        a = pd.Series(np.random.randn(100), index=idx)
        p = pd.Series(np.random.randn(100), index=idx)
        r = prediction_correlation(a, p)
        assert -1.0 <= r <= 1.0

    def test_nan_handling(self):
        idx = _make_dates(10)
        a = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0,
                        6.0, 7.0, 8.0, 9.0, 10.0], index=idx)
        p = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0,
                        6.0, 7.0, 8.0, 9.0, 10.0], index=idx)
        r = prediction_correlation(a, p)
        assert not np.isnan(r)


# ---------------------------------------------------------------------------
# information_coefficient
# ---------------------------------------------------------------------------

class TestInformationCoefficient:
    def _panel(self, n_dates: int = 20, n_assets: int = 5,
               seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(seed)
        dates = _make_dates(n_dates)
        assets = [f"A{i}" for i in range(n_assets)]
        actual = pd.DataFrame(rng.standard_normal((n_dates, n_assets)),
                              index=dates, columns=assets)
        predicted = actual + rng.standard_normal((n_dates, n_assets)) * 0.5
        return actual, predicted

    def test_returns_series(self):
        actual, predicted = self._panel()
        ic = information_coefficient(actual, predicted)
        assert isinstance(ic, pd.Series)

    def test_index_is_dates(self):
        actual, predicted = self._panel()
        ic = information_coefficient(actual, predicted)
        assert isinstance(ic.index, pd.DatetimeIndex)

    def test_series_name_is_ic(self):
        actual, predicted = self._panel()
        ic = information_coefficient(actual, predicted)
        assert ic.name == "IC"

    def test_length_matches_dates(self):
        actual, predicted = self._panel(n_dates=20, n_assets=5)
        ic = information_coefficient(actual, predicted)
        # All 20 dates have >= 5 observations (default min_observations=5)
        assert len(ic.dropna()) == 20

    def test_perfect_rank_correlation(self):
        dates = _make_dates(10)
        assets = ["A", "B", "C", "D", "E"]
        actual = pd.DataFrame(
            np.tile(np.arange(5, dtype=float), (10, 1)),
            index=dates, columns=assets
        )
        predicted = actual.copy()
        ic = information_coefficient(actual, predicted)
        # Perfect rank agreement → IC = 1.0 for every date
        assert (ic.dropna() - 1.0).abs().max() < 1e-9

    def test_min_observations_filters_dates(self):
        dates = _make_dates(5)
        assets = ["A", "B"]
        actual = pd.DataFrame(np.ones((5, 2)), index=dates, columns=assets)
        predicted = actual.copy()
        # min_observations=5 but only 2 assets → all NaN
        ic = information_coefficient(actual, predicted, min_observations=5)
        assert ic.isna().all()

    def test_values_in_range(self):
        actual, predicted = self._panel(n_dates=30, n_assets=10)
        ic = information_coefficient(actual, predicted)
        clean = ic.dropna()
        assert (clean >= -1.0).all() and (clean <= 1.0).all()

    def test_mismatched_columns_uses_intersection(self):
        dates = _make_dates(10)
        actual = pd.DataFrame(
            np.random.randn(10, 3), index=dates, columns=["A", "B", "C"]
        )
        predicted = pd.DataFrame(
            np.random.randn(10, 2), index=dates, columns=["A", "B"]
        )
        ic = information_coefficient(actual, predicted)
        assert isinstance(ic, pd.Series)
        assert len(ic) > 0

    def test_mismatched_index_uses_intersection(self):
        dates_a = _make_dates(15)
        dates_p = _make_dates(10)
        actual = pd.DataFrame(
            np.random.randn(15, 4), index=dates_a, columns=["A", "B", "C", "D"]
        )
        predicted = pd.DataFrame(
            np.random.randn(10, 4), index=dates_p, columns=["A", "B", "C", "D"]
        )
        ic = information_coefficient(actual, predicted)
        assert len(ic) <= 10


# ---------------------------------------------------------------------------
# rolling_directional_accuracy
# ---------------------------------------------------------------------------

class TestRollingDirectionalAccuracy:
    def test_perfect_direction_match(self):
        idx = _make_dates(30)
        s = pd.Series(np.sin(np.arange(30) * 0.5), index=idx)
        rda = rolling_directional_accuracy(s, s, window=5)
        clean = rda.dropna()
        # Perfect match → 1.0 everywhere (excluding zeros)
        assert (clean[clean.notna()] == 1.0).all()

    def test_perfect_direction_mismatch(self):
        idx = _make_dates(30)
        actual = pd.Series(np.sin(np.arange(30) * 0.5), index=idx)
        predicted = -actual
        rda = rolling_directional_accuracy(actual, predicted, window=5)
        clean = rda.dropna()
        assert (clean == 0.0).all()

    def test_returns_series(self):
        idx = _make_dates(30)
        a = pd.Series(np.random.randn(30), index=idx)
        p = pd.Series(np.random.randn(30), index=idx)
        rda = rolling_directional_accuracy(a, p, window=10)
        assert isinstance(rda, pd.Series)

    def test_first_window_minus_one_is_nan(self):
        idx = _make_dates(20)
        a = pd.Series(np.random.randn(20), index=idx)
        p = pd.Series(np.random.randn(20), index=idx)
        rda = rolling_directional_accuracy(a, p, window=10)
        assert rda.iloc[:9].isna().all()

    def test_values_in_range(self):
        idx = _make_dates(50)
        a = pd.Series(np.random.randn(50), index=idx)
        p = pd.Series(np.random.randn(50), index=idx)
        rda = rolling_directional_accuracy(a, p, window=10)
        clean = rda.dropna()
        assert (clean >= 0.0).all() and (clean <= 1.0).all()

    def test_raises_on_window_less_than_1(self):
        idx = _make_dates(10)
        a = pd.Series(np.ones(10), index=idx)
        p = pd.Series(np.ones(10), index=idx)
        with pytest.raises(ValueError):
            rolling_directional_accuracy(a, p, window=0)

    def test_nan_propagation(self):
        idx = _make_dates(20)
        a = pd.Series([np.nan] + list(np.random.randn(19)), index=idx)
        p = pd.Series(np.random.randn(20), index=idx)
        rda = rolling_directional_accuracy(a, p, window=5)
        # NaN pair is dropped via inner-join → result is 19 rows
        assert len(rda) == 19
        # First 4 rows still NaN (rolling window not yet filled)
        assert rda.iloc[:4].isna().all()

    def test_index_preserved(self):
        idx = _make_dates(25)
        a = pd.Series(np.random.randn(25), index=idx)
        p = pd.Series(np.random.randn(25), index=idx)
        rda = rolling_directional_accuracy(a, p, window=5)
        assert rda.index.equals(idx)


# ---------------------------------------------------------------------------
# prediction_quantiles
# ---------------------------------------------------------------------------

class TestPredictionQuantiles:
    def test_returns_series(self):
        idx = _make_dates(50)
        preds = pd.Series(np.random.randn(50), index=idx)
        q = prediction_quantiles(preds)
        assert isinstance(q, pd.Series)

    def test_quantile_labels_range(self):
        idx = _make_dates(100)
        preds = pd.Series(np.random.randn(100), index=idx)
        q = prediction_quantiles(preds, n_quantiles=5)
        valid = q.dropna()
        assert valid.min() >= 1
        assert valid.max() <= 5

    def test_default_10_quantiles(self):
        idx = _make_dates(200)
        preds = pd.Series(np.linspace(-1, 1, 200), index=idx)
        q = prediction_quantiles(preds)
        unique_q = set(int(v) for v in q.dropna())
        assert unique_q == set(range(1, 11))

    def test_index_preserved(self):
        idx = _make_dates(50)
        preds = pd.Series(np.random.randn(50), index=idx)
        q = prediction_quantiles(preds)
        assert q.index.equals(idx)

    def test_raises_on_n_quantiles_lt_2(self):
        idx = _make_dates(20)
        preds = pd.Series(np.random.randn(20), index=idx)
        with pytest.raises(ValueError):
            prediction_quantiles(preds, n_quantiles=1)

    def test_series_named_quantile(self):
        idx = _make_dates(50)
        preds = pd.Series(np.random.randn(50), index=idx)
        q = prediction_quantiles(preds)
        assert q.name == "quantile"

    def test_nan_handling(self):
        idx = _make_dates(50)
        vals = np.random.randn(50)
        vals[5] = np.nan
        preds = pd.Series(vals, index=idx)
        q = prediction_quantiles(preds)
        # NaN input should produce NaN output at that position
        assert pd.isna(q.iloc[5])


# ---------------------------------------------------------------------------
# Package import
# ---------------------------------------------------------------------------

def test_import_from_src_ml():
    from src.ml import (
        information_coefficient,
        prediction_correlation,
        prediction_quantiles,
        rolling_directional_accuracy,
    )
    assert callable(information_coefficient)
    assert callable(prediction_correlation)
    assert callable(prediction_quantiles)
    assert callable(rolling_directional_accuracy)
