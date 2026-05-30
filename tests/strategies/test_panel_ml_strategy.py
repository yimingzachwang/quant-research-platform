"""Tests for src.strategies.panel_ml_strategy — PanelMLStrategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.strategies.panel_ml_strategy import PanelMLStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prices(n_dates: int = 350, tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or ["SPY", "QQQ", "IWM", "EEM"]
    rng = np.random.default_rng(42)
    idx = pd.date_range("2018-01-01", periods=n_dates, freq="B")
    data = np.cumprod(1 + rng.normal(0.0002, 0.01, (n_dates, len(tickers))), axis=0) * 100
    return pd.DataFrame(data, index=idx, columns=tickers)


def _momentum_builder(ticker: str) -> dict:
    """Simple momentum feature builder for any ticker."""
    def _fn(prices: pd.DataFrame, t: str = ticker) -> pd.Series:
        return prices[t].pct_change(20)
    return {"mom_20": _fn}


def _make_label_fn(tickers: list[str], horizon: int = 5):
    from src.ml.labels import ranking_target
    available = tickers
    def _fn(prices: pd.DataFrame, t: list = available, h: int = horizon) -> pd.DataFrame:
        cols = [x for x in t if x in prices.columns]
        return ranking_target(prices[cols], h)
    return _fn


def _top_n_signal_fn(n: int):
    from src.ml.signals.prediction import top_n_weights
    return lambda preds, _n=n: top_n_weights(preds, _n)


def _make_strategy(tickers: list[str], top_n: int = 2, horizon: int = 5) -> PanelMLStrategy:
    from src.ml.models.linear import RidgeRegressionModel
    model = RidgeRegressionModel(alpha=1.0)
    return PanelMLStrategy(
        model=model,
        tickers=tickers,
        feature_fn_builder=_momentum_builder,
        label_fn=_make_label_fn(tickers, horizon),
        horizon=horizon,
        signal_fn=_top_n_signal_fn(top_n),
    )


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestPanelMLStrategyConstructor:
    def test_name_contains_model_name(self):
        s = _make_strategy(["SPY", "QQQ"])
        assert "PanelMLStrategy" in s.name
        assert "Ridge" in s.name

    def test_params_include_tickers(self):
        tickers = ["SPY", "QQQ", "IWM"]
        s = _make_strategy(tickers)
        p = s.params()
        assert p["tickers"] == tickers
        assert p["n_tickers"] == 3

    def test_not_fitted_initially(self):
        s = _make_strategy(["SPY", "QQQ"])
        assert not s._is_fitted


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------


class TestPanelMLStrategyFit:
    def test_fit_sets_is_fitted(self):
        tickers = ["SPY", "QQQ", "IWM"]
        prices = _make_prices(tickers=tickers)
        s = _make_strategy(tickers)
        s.fit(prices)
        assert s._is_fitted

    def test_fit_raises_without_sufficient_data(self):
        tickers = ["SPY", "QQQ"]
        prices = _make_prices(n_dates=5, tickers=tickers)  # far too few rows
        s = _make_strategy(tickers)
        with pytest.raises(ValueError):
            s.fit(prices)

    def test_fit_can_be_called_multiple_times(self):
        tickers = ["SPY", "QQQ"]
        prices = _make_prices(tickers=tickers)
        s = _make_strategy(tickers)
        s.fit(prices)
        s.fit(prices)  # second fit should overwrite, no error
        assert s._is_fitted


# ---------------------------------------------------------------------------
# generate_weights()
# ---------------------------------------------------------------------------


class TestPanelMLStrategyGenerateWeights:
    def test_generate_weights_before_fit_raises(self):
        tickers = ["SPY", "QQQ"]
        prices = _make_prices(tickers=tickers)
        s = _make_strategy(tickers)
        with pytest.raises(RuntimeError, match="fit"):
            s.generate_weights(prices)

    def test_output_shape_is_date_by_asset(self):
        tickers = ["SPY", "QQQ", "IWM", "EEM"]
        prices = _make_prices(tickers=tickers)
        s = _make_strategy(tickers, top_n=2)
        s.fit(prices)
        weights = s.generate_weights(prices)
        assert weights.shape == (len(prices), len(tickers))

    def test_output_columns_match_tickers(self):
        tickers = ["SPY", "QQQ", "IWM"]
        prices = _make_prices(tickers=tickers)
        s = _make_strategy(tickers)
        s.fit(prices)
        weights = s.generate_weights(prices)
        assert list(weights.columns) == tickers

    def test_output_index_matches_prices_index(self):
        tickers = ["SPY", "QQQ"]
        prices = _make_prices(tickers=tickers)
        s = _make_strategy(tickers)
        s.fit(prices)
        weights = s.generate_weights(prices)
        assert (weights.index == prices.index).all()

    def test_active_rows_sum_to_one(self):
        tickers = ["SPY", "QQQ", "IWM", "EEM"]
        prices = _make_prices(tickers=tickers)
        s = _make_strategy(tickers, top_n=2)
        s.fit(prices)
        weights = s.generate_weights(prices)
        # Non-warmup rows should sum to 1.0
        active = weights.abs().sum(axis=1) > 1e-8
        row_sums = weights[active].sum(axis=1)
        assert (row_sums - 1.0).abs().max() < 1e-8

    def test_top_n_constraint_honored(self):
        tickers = ["SPY", "QQQ", "IWM", "EEM", "TLT"]
        prices = _make_prices(tickers=tickers)
        top_n = 2
        s = _make_strategy(tickers, top_n=top_n)
        s.fit(prices)
        weights = s.generate_weights(prices)
        active = weights.abs().sum(axis=1) > 1e-8
        n_held_per_row = (weights[active] > 1e-8).sum(axis=1)
        assert (n_held_per_row <= top_n).all()

    def test_warmup_rows_are_flat(self):
        tickers = ["SPY", "QQQ"]
        prices = _make_prices(n_dates=100, tickers=tickers)
        s = _make_strategy(tickers, horizon=5)
        s.fit(prices)
        weights = s.generate_weights(prices)
        # First ~20 rows (momentum warmup) should all be zero
        assert (weights.iloc[:20].abs().sum(axis=1) < 1e-8).any()


# ---------------------------------------------------------------------------
# Walk-forward compatibility
# ---------------------------------------------------------------------------


class TestPanelMLStrategyWalkForward:
    def test_walk_forward_validation_runs(self):
        from src.experiments.factory import ValidationConfig, build_validation_splits
        from src.validation.walk_forward import run_walk_forward_validation

        tickers = ["SPY", "QQQ", "IWM"]
        prices = _make_prices(n_dates=500, tickers=tickers)
        s = _make_strategy(tickers, top_n=2, horizon=5)

        val_cfg = ValidationConfig(type="rolling", parameters={"train_months": 12, "test_months": 3, "gap_days": 0})
        splits = build_validation_splits(val_cfg, prices.index)
        assert splits  # sanity

        wf = run_walk_forward_validation(prices=prices, strategy=s, splits=splits)
        assert wf.n_splits > 0

    def test_each_split_produces_weights(self):
        from src.experiments.factory import ValidationConfig, build_validation_splits
        from src.validation.walk_forward import run_walk_forward_validation

        tickers = ["SPY", "QQQ", "IWM", "EEM"]
        prices = _make_prices(n_dates=500, tickers=tickers)
        s = _make_strategy(tickers, top_n=2, horizon=5)

        val_cfg = ValidationConfig(type="rolling", parameters={"train_months": 12, "test_months": 3, "gap_days": 0})
        splits = build_validation_splits(val_cfg, prices.index)
        wf = run_walk_forward_validation(prices=prices, strategy=s, splits=splits)

        for split_result in wf.splits:
            assert not split_result.weights.empty
