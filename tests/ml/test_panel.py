"""Tests for src.ml.panel — cross-sectional panel utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.ml.panel import build_panel_feature_matrix, compute_cross_sectional_ic

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_prices(n_dates: int = 300, tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or ["SPY", "QQQ", "IWM"]
    rng = np.random.default_rng(42)
    idx = pd.date_range("2018-01-01", periods=n_dates, freq="B")
    data = np.cumprod(1 + rng.normal(0.0002, 0.01, (n_dates, len(tickers))), axis=0) * 100
    return pd.DataFrame(data, index=idx, columns=tickers)


def _simple_feature_fn_builder(feature_name: str = "momentum"):
    """Returns a builder that produces a single momentum feature for any ticker."""
    def builder(ticker: str) -> dict:
        def _fn(prices: pd.DataFrame, t: str = ticker) -> pd.Series:
            return prices[t].pct_change(20)
        return {feature_name: _fn}
    return builder


# ---------------------------------------------------------------------------
# build_panel_feature_matrix
# ---------------------------------------------------------------------------


class TestBuildPanelFeatureMatrix:
    def test_returns_multiindex_dataframe(self):
        prices = _make_prices()
        builder = _simple_feature_fn_builder()
        result = build_panel_feature_matrix(prices, builder, ["SPY", "QQQ"])
        assert isinstance(result, pd.DataFrame)
        assert result.index.nlevels == 2
        assert result.index.names == ["date", "asset"]

    def test_column_names_preserved(self):
        prices = _make_prices()
        builder = _simple_feature_fn_builder("my_feature")
        result = build_panel_feature_matrix(prices, builder, ["SPY"])
        assert "my_feature" in result.columns

    def test_n_rows_equals_n_dates_times_n_tickers(self):
        n_dates = 100
        tickers = ["SPY", "QQQ", "IWM"]
        prices = _make_prices(n_dates=n_dates, tickers=tickers)
        builder = _simple_feature_fn_builder()
        result = build_panel_feature_matrix(prices, builder, tickers)
        assert len(result) == n_dates * len(tickers)

    def test_asset_level_contains_all_tickers(self):
        prices = _make_prices()
        tickers = ["SPY", "QQQ", "IWM"]
        builder = _simple_feature_fn_builder()
        result = build_panel_feature_matrix(prices, builder, tickers)
        asset_values = result.index.get_level_values("asset").unique().tolist()
        assert sorted(asset_values) == sorted(tickers)

    def test_empty_tickers_returns_empty_dataframe(self):
        prices = _make_prices()
        builder = _simple_feature_fn_builder()
        result = build_panel_feature_matrix(prices, builder, [])
        assert result.empty

    def test_sorted_by_date_asset(self):
        prices = _make_prices(n_dates=50)
        tickers = ["QQQ", "SPY"]  # intentionally out of alphabetical order
        builder = _simple_feature_fn_builder()
        result = build_panel_feature_matrix(prices, builder, tickers)
        dates = result.index.get_level_values("date")
        assert dates.is_monotonic_increasing

    def test_dropna_on_result_removes_warmup(self):
        prices = _make_prices(n_dates=100)
        tickers = ["SPY", "QQQ"]
        builder = _simple_feature_fn_builder()
        result = build_panel_feature_matrix(prices, builder, tickers)
        clean = result.dropna()
        assert len(clean) < len(result)
        assert clean.notna().all().all()

    def test_multiple_features_per_ticker(self):
        prices = _make_prices()
        tickers = ["SPY", "QQQ"]

        def multi_builder(ticker: str) -> dict:
            def _mom(prices: pd.DataFrame, t: str = ticker) -> pd.Series:
                return prices[t].pct_change(20)
            def _vol(prices: pd.DataFrame, t: str = ticker) -> pd.Series:
                return prices[t].pct_change().rolling(21).std()
            return {"mom": _mom, "vol": _vol}

        result = build_panel_feature_matrix(prices, multi_builder, tickers)
        assert set(result.columns) == {"mom", "vol"}


# ---------------------------------------------------------------------------
# compute_cross_sectional_ic
# ---------------------------------------------------------------------------


class TestComputeCrossSectionalIC:
    def _make_pred_actual(self, n_dates: int = 100, n_assets: int = 5) -> tuple:
        rng = np.random.default_rng(0)
        idx = pd.date_range("2019-01-01", periods=n_dates, freq="B")
        tickers = [f"A{i}" for i in range(n_assets)]
        pred = pd.DataFrame(rng.standard_normal((n_dates, n_assets)), index=idx, columns=tickers)
        actual = pd.DataFrame(rng.standard_normal((n_dates, n_assets)), index=idx, columns=tickers)
        return pred, actual

    def test_returns_series(self):
        pred, actual = self._make_pred_actual()
        result = compute_cross_sectional_ic(pred, actual)
        assert isinstance(result, pd.Series)

    def test_length_equals_n_common_dates(self):
        pred, actual = self._make_pred_actual(n_dates=100)
        result = compute_cross_sectional_ic(pred, actual)
        assert len(result) == 100

    def test_values_bounded_minus_one_to_one(self):
        pred, actual = self._make_pred_actual(n_dates=200)
        result = compute_cross_sectional_ic(pred, actual)
        assert (result >= -1.0).all() and (result <= 1.0).all()

    def test_perfect_ranking_gives_ic_one(self):
        rng = np.random.default_rng(7)
        idx = pd.date_range("2020-01-01", periods=50, freq="B")
        tickers = ["A", "B", "C", "D"]
        # Perfect: predictions exactly equal actual returns
        data = pd.DataFrame(rng.standard_normal((50, 4)), index=idx, columns=tickers)
        result = compute_cross_sectional_ic(data, data)
        assert (result - 1.0).abs().max() < 1e-10

    def test_reversed_ranking_gives_ic_minus_one(self):
        rng = np.random.default_rng(8)
        idx = pd.date_range("2020-01-01", periods=50, freq="B")
        tickers = ["A", "B", "C", "D"]
        data = pd.DataFrame(rng.standard_normal((50, 4)), index=idx, columns=tickers)
        result = compute_cross_sectional_ic(data, -data)
        assert (result + 1.0).abs().max() < 1e-10

    def test_min_assets_filter(self):
        pred, actual = self._make_pred_actual(n_dates=50, n_assets=5)
        # Require 10 assets per date — 5 available, so all dates filtered
        result = compute_cross_sectional_ic(pred, actual, min_assets=10)
        assert result.empty

    def test_partial_overlap_dates(self):
        idx_pred = pd.date_range("2020-01-01", periods=60, freq="B")
        idx_act = pd.date_range("2020-03-01", periods=60, freq="B")
        tickers = ["A", "B", "C"]
        rng = np.random.default_rng(1)
        pred = pd.DataFrame(rng.standard_normal((60, 3)), index=idx_pred, columns=tickers)
        actual = pd.DataFrame(rng.standard_normal((60, 3)), index=idx_act, columns=tickers)
        result = compute_cross_sectional_ic(pred, actual)
        common = idx_pred.intersection(idx_act)
        assert len(result) == len(common)

    def test_empty_dataframes_return_empty_series(self):
        result = compute_cross_sectional_ic(pd.DataFrame(), pd.DataFrame())
        assert result.empty

    def test_random_predictions_mean_ic_near_zero(self):
        """Uncorrelated predictions should give mean IC close to zero."""
        pred, actual = self._make_pred_actual(n_dates=500, n_assets=10)
        result = compute_cross_sectional_ic(pred, actual)
        assert abs(float(result.mean())) < 0.15  # loose bound for random
