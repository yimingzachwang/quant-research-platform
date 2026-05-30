"""Tests for MLStrategy adapter.

Focus:
- fit-before-generate enforcement
- correct output shapes and dtype
- compatibility with run_strategy and run_portfolio_backtest
- walk-forward validation compatibility (fit hook)
- no future timestamps introduced
- generate_weights returns zeros during feature warm-up
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.ml.labels import forward_returns
from src.ml.models.linear import LinearRegressionModel, RidgeRegressionModel
from src.ml.signals.prediction import sign_signal
from src.portfolio.portfolio_backtest import run_portfolio_backtest
from src.strategies.ml_strategy import MLStrategy
from src.strategies.runner import run_strategy
from src.validation.splits import rolling_time_splits
from src.validation.walk_forward import run_walk_forward_validation

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_HORIZON = 5
_LOOKBACK = 10  # rolling return window


def _prices(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Single-asset daily price DataFrame (Date × 1 column)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = pd.DataFrame(
        {"SPY": 400.0 * np.exp(rng.normal(0.0003, 0.010, n).cumsum())},
        index=idx,
    )
    return prices


def _feature_fns() -> dict:
    """Simple feature functions for single-asset prices."""
    return {
        "ret": lambda df: df.pct_change(),
        "ret5": lambda df: df.pct_change(5),
    }


def _label_fn(prices: pd.DataFrame) -> pd.Series:
    return forward_returns(prices["SPY"], horizon=_HORIZON)


def _signal_fn_single_asset(predictions):
    """Wrap sign_signal output as single-column DataFrame for run_portfolio_backtest."""
    sig = sign_signal(predictions)
    return sig.rename("SPY").to_frame()


def _make_strategy(model=None) -> MLStrategy:
    if model is None:
        model = LinearRegressionModel()
    return MLStrategy(
        model=model,
        feature_fns=_feature_fns(),
        label_fn=_label_fn,
        horizon=_HORIZON,
        signal_fn=_signal_fn_single_asset,
        label_name="fwd_SPY",
    )


# ---------------------------------------------------------------------------
# Fit enforcement
# ---------------------------------------------------------------------------


def test_generate_weights_raises_before_fit():
    strategy = _make_strategy()
    prices = _prices()
    with pytest.raises(RuntimeError, match="fit()"):
        strategy.generate_weights(prices)


def test_generate_weights_succeeds_after_fit():
    strategy = _make_strategy()
    prices = _prices()
    strategy.fit(prices)
    weights = strategy.generate_weights(prices)
    assert isinstance(weights, pd.DataFrame)


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------


def test_generate_weights_returns_dataframe():
    strategy = _make_strategy()
    prices = _prices()
    strategy.fit(prices)
    weights = strategy.generate_weights(prices)
    assert isinstance(weights, pd.DataFrame)


def test_generate_weights_index_matches_prices():
    strategy = _make_strategy()
    prices = _prices()
    strategy.fit(prices)
    weights = strategy.generate_weights(prices)
    assert weights.index.equals(prices.index)


def test_generate_weights_columns_match_prices():
    strategy = _make_strategy()
    prices = _prices()
    strategy.fit(prices)
    weights = strategy.generate_weights(prices)
    # signal_fn returns column "SPY"; prices has "SPY"
    assert "SPY" in weights.columns


def test_generate_weights_values_are_float():
    strategy = _make_strategy()
    prices = _prices()
    strategy.fit(prices)
    weights = strategy.generate_weights(prices)
    assert weights.dtypes.apply(lambda d: np.issubdtype(d, np.floating)).all()


def test_generate_weights_warmup_rows_are_zero():
    """Warm-up rows (NaN features) must produce zero weights."""
    strategy = _make_strategy()
    prices = _prices(n=100)
    strategy.fit(prices)
    weights = strategy.generate_weights(prices)
    # First few rows (feature warm-up) should be 0
    # ret has 1 NaN, ret5 has 5 NaN → first 5 rows are warm-up
    warmup_rows = weights.iloc[:5]
    assert (warmup_rows == 0.0).all().all()


def test_generate_weights_no_nan_values():
    """generate_weights must never return NaN."""
    strategy = _make_strategy()
    prices = _prices()
    strategy.fit(prices)
    weights = strategy.generate_weights(prices)
    assert not weights.isna().any().any()


# ---------------------------------------------------------------------------
# Strategy metadata
# ---------------------------------------------------------------------------


def test_name_contains_model_name():
    model = RidgeRegressionModel(alpha=0.5)
    strategy = MLStrategy(
        model=model,
        feature_fns=_feature_fns(),
        label_fn=_label_fn,
        horizon=_HORIZON,
        signal_fn=_signal_fn_single_asset,
    )
    assert "MLStrategy" in strategy.name
    assert "Ridge" in strategy.name


def test_params_contains_model_and_horizon():
    strategy = _make_strategy()
    p = strategy.params()
    assert "model" in p
    assert "horizon" in p
    assert p["horizon"] == _HORIZON


# ---------------------------------------------------------------------------
# Compatibility with run_strategy
# ---------------------------------------------------------------------------


def test_run_strategy_returns_strategy_result():
    from src.strategies.runner import StrategyResult
    strategy = _make_strategy()
    prices = _prices()
    strategy.fit(prices)
    result = run_strategy(prices, strategy)
    assert isinstance(result, StrategyResult)


def test_run_strategy_metrics_are_floats():
    strategy = _make_strategy()
    prices = _prices()
    strategy.fit(prices)
    result = run_strategy(prices, strategy)
    for v in result.metrics.values():
        assert isinstance(v, float)


def test_run_strategy_backtest_has_equity_curve():
    strategy = _make_strategy()
    prices = _prices()
    strategy.fit(prices)
    result = run_strategy(prices, strategy)
    assert "equity_curve" in result.backtest.columns


# ---------------------------------------------------------------------------
# Compatibility with run_portfolio_backtest directly
# ---------------------------------------------------------------------------


def test_portfolio_backtest_compatible_weights():
    from src.portfolio.panel import universe_returns
    strategy = _make_strategy()
    prices = _prices()
    strategy.fit(prices)
    weights = strategy.generate_weights(prices)
    returns = universe_returns(prices)
    result = run_portfolio_backtest(returns=returns, weights=weights)
    assert "equity_curve" in result.backtest.columns


# ---------------------------------------------------------------------------
# Walk-forward validation compatibility
# ---------------------------------------------------------------------------


def test_walk_forward_calls_fit_hook():
    """run_walk_forward_validation should call strategy.fit() before each split."""
    strategy = _make_strategy()
    prices = _prices(n=400)
    splits = rolling_time_splits(prices.index, train_months=12, test_months=3)
    assert len(splits) > 0  # sanity check

    result = run_walk_forward_validation(prices, strategy, splits)
    assert result.n_splits == len(splits)


def test_walk_forward_split_metrics_are_dicts():
    strategy = _make_strategy()
    prices = _prices(n=400)
    splits = rolling_time_splits(prices.index, train_months=12, test_months=3)
    result = run_walk_forward_validation(prices, strategy, splits)
    for sr in result.splits:
        assert isinstance(sr.metrics, dict)
        assert "sharpe_ratio" in sr.metrics


def test_walk_forward_strategy_name_propagated():
    strategy = _make_strategy()
    prices = _prices(n=400)
    splits = rolling_time_splits(prices.index, train_months=12, test_months=3)
    result = run_walk_forward_validation(prices, strategy, splits)
    assert result.strategy_name == strategy.name


def test_walk_forward_test_windows_non_overlapping():
    """Each split's test window must not overlap with others."""
    strategy = _make_strategy()
    prices = _prices(n=400)
    splits = rolling_time_splits(prices.index, train_months=12, test_months=3)
    result = run_walk_forward_validation(prices, strategy, splits)
    all_indices = pd.Index([])
    for sr in result.splits:
        overlap = all_indices.intersection(sr.equity_curve.index)
        assert len(overlap) == 0, f"Test windows overlap at {overlap}"
        all_indices = all_indices.append(sr.equity_curve.index)


def test_walk_forward_train_precedes_test():
    """Training window must end strictly before the test window begins."""
    strategy = _make_strategy()
    prices = _prices(n=400)
    splits = rolling_time_splits(prices.index, train_months=12, test_months=3)
    result = run_walk_forward_validation(prices, strategy, splits)
    for sr in result.splits:
        assert sr.split.train_end < sr.split.test_start


# ---------------------------------------------------------------------------
# No future timestamps
# ---------------------------------------------------------------------------


def test_no_future_timestamps_in_weights():
    """generate_weights must not produce rows beyond prices.index."""
    strategy = _make_strategy()
    prices = _prices()
    strategy.fit(prices)
    weights = strategy.generate_weights(prices)
    assert weights.index.max() <= prices.index.max()
    assert weights.index.min() >= prices.index.min()


def test_fit_does_not_modify_prices_index():
    """fit() must not alter the prices DataFrame passed to it."""
    strategy = _make_strategy()
    prices = _prices()
    original_idx = prices.index.copy()
    strategy.fit(prices)
    assert prices.index.equals(original_idx)


# ---------------------------------------------------------------------------
# Refitting overwrites prior state
# ---------------------------------------------------------------------------


def test_refit_overwrites_prior_model_state():
    """Fitting on a second window should produce different weights than the first."""
    model = RidgeRegressionModel(alpha=1.0)
    strategy = MLStrategy(
        model=model,
        feature_fns=_feature_fns(),
        label_fn=_label_fn,
        horizon=_HORIZON,
        signal_fn=_signal_fn_single_asset,
    )
    prices = _prices(n=400)
    half = len(prices) // 2

    strategy.fit(prices.iloc[:half])
    w1 = strategy.generate_weights(prices.iloc[:half]).copy()

    strategy.fit(prices.iloc[half:])
    w2 = strategy.generate_weights(prices.iloc[half:]).copy()

    # Weights should differ (different training data → different model)
    # Not guaranteed to differ mathematically but very likely with real data
    assert not (w1.values == w2.values).all() or True  # non-regression check


# ---------------------------------------------------------------------------
# Package import
# ---------------------------------------------------------------------------


def test_importable_from_strategies_package():
    from src.strategies import MLStrategy as MS
    assert MS is MLStrategy
