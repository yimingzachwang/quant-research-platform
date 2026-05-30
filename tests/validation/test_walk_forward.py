"""Tests for src/validation/walk_forward.py."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.baselines import EqualWeightStrategy
from src.strategies.momentum_rotation import MomentumRotationStrategy
from src.validation.splits import rolling_time_splits, expanding_time_splits
from src.validation.walk_forward import (
    SplitResult,
    WalkForwardResult,
    run_walk_forward_validation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def prices() -> pd.DataFrame:
    idx = pd.date_range("2010-01-01", "2019-12-31", freq="B")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "A": 100 * np.cumprod(1 + rng.normal(0.001, 0.01, len(idx))),
            "B": 100 * np.cumprod(1 + rng.normal(0.000, 0.01, len(idx))),
            "C": 100 * np.cumprod(1 + rng.normal(0.002, 0.01, len(idx))),
        },
        index=idx,
    )


@pytest.fixture()
def rolling_splits(prices: pd.DataFrame):
    return rolling_time_splits(prices.index, train_months=36, test_months=12)


@pytest.fixture()
def wf_result(prices: pd.DataFrame, rolling_splits):
    strategy = EqualWeightStrategy()
    return run_walk_forward_validation(prices, strategy, rolling_splits)


# ---------------------------------------------------------------------------
# WalkForwardResult structure
# ---------------------------------------------------------------------------


def test_run_returns_wf_result(wf_result: WalkForwardResult) -> None:
    assert isinstance(wf_result, WalkForwardResult)


def test_wf_result_n_splits_matches(
    wf_result: WalkForwardResult, rolling_splits: list
) -> None:
    assert wf_result.n_splits == len(rolling_splits)


def test_wf_result_strategy_name(wf_result: WalkForwardResult) -> None:
    assert wf_result.strategy_name == EqualWeightStrategy().name


def test_split_results_are_split_result_type(wf_result: WalkForwardResult) -> None:
    for sr in wf_result.splits:
        assert isinstance(sr, SplitResult)


# ---------------------------------------------------------------------------
# SplitResult contents
# ---------------------------------------------------------------------------


def test_split_result_has_metrics(wf_result: WalkForwardResult) -> None:
    for sr in wf_result.splits:
        assert isinstance(sr.metrics, dict)
        assert "sharpe_ratio" in sr.metrics
        assert "annualized_return" in sr.metrics


def test_split_result_equity_curve_is_series(wf_result: WalkForwardResult) -> None:
    for sr in wf_result.splits:
        assert isinstance(sr.equity_curve, pd.Series)


def test_split_result_equity_starts_near_1(wf_result: WalkForwardResult) -> None:
    for sr in wf_result.splits:
        # First value is (1 + r_0) so should be close to 1.0 but not exactly
        assert abs(sr.equity_curve.iloc[0] - 1.0) < 0.1


def test_split_result_weights_is_dataframe(wf_result: WalkForwardResult) -> None:
    for sr in wf_result.splits:
        assert isinstance(sr.weights, pd.DataFrame)


def test_split_result_equity_within_test_window(wf_result: WalkForwardResult) -> None:
    for sr in wf_result.splits:
        assert sr.equity_curve.index[0] >= sr.split.test_start
        assert sr.equity_curve.index[-1] <= sr.split.test_end


def test_split_result_weights_within_test_window(wf_result: WalkForwardResult) -> None:
    for sr in wf_result.splits:
        assert sr.weights.index[0] >= sr.split.test_start
        assert sr.weights.index[-1] <= sr.split.test_end


# ---------------------------------------------------------------------------
# No leakage: equity only covers test period
# ---------------------------------------------------------------------------


def test_no_leakage_equity_before_train_end(wf_result: WalkForwardResult) -> None:
    for sr in wf_result.splits:
        # Equity curve must not include dates from training period
        assert (sr.equity_curve.index >= sr.split.test_start).all()


def test_no_leakage_no_future_data(
    prices: pd.DataFrame, rolling_splits: list
) -> None:
    strategy = EqualWeightStrategy()
    result = run_walk_forward_validation(prices, strategy, rolling_splits)
    for sr in result.splits:
        # weights should not extend beyond test_end
        assert sr.weights.index[-1] <= sr.split.test_end


# ---------------------------------------------------------------------------
# Empty splits
# ---------------------------------------------------------------------------


def test_empty_splits_returns_empty_result(prices: pd.DataFrame) -> None:
    result = run_walk_forward_validation(prices, EqualWeightStrategy(), splits=[])
    assert result.n_splits == 0
    assert result.splits == []


# ---------------------------------------------------------------------------
# Expanding splits
# ---------------------------------------------------------------------------


def test_expanding_splits_work(prices: pd.DataFrame) -> None:
    splits = expanding_time_splits(prices.index, min_train_months=24, test_months=12)
    result = run_walk_forward_validation(prices, EqualWeightStrategy(), splits)
    assert result.n_splits == len(splits)


# ---------------------------------------------------------------------------
# Transaction costs reduce net return
# ---------------------------------------------------------------------------


def test_cost_reduces_net_return(
    prices: pd.DataFrame, rolling_splits: list
) -> None:
    strategy = EqualWeightStrategy()
    no_cost = run_walk_forward_validation(prices, strategy, rolling_splits, transaction_cost_bps=0.0)
    with_cost = run_walk_forward_validation(prices, strategy, rolling_splits, transaction_cost_bps=20.0)

    for nc_sr, wc_sr in zip(no_cost.splits, with_cost.splits):
        nc_ret = nc_sr.metrics["annualized_return"]
        wc_ret = wc_sr.metrics["annualized_return"]
        assert wc_ret <= nc_ret + 1e-10
