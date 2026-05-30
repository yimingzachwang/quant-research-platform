"""Tests for src/validation/stability.py."""

import numpy as np
import pandas as pd
import pytest
from src.strategies.baselines import EqualWeightStrategy
from src.validation.splits import rolling_time_splits
from src.validation.stability import (
    parameter_robustness_summary,
    rolling_sharpe_by_split,
    split_metrics_table,
    summarize_stability,
)
from src.validation.walk_forward import WalkForwardResult, run_walk_forward_validation

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    idx = pd.date_range("2010-01-01", "2019-12-31", freq="B")
    rng = np.random.default_rng(77)
    return pd.DataFrame(
        {
            "A": 100 * np.cumprod(1 + rng.normal(0.001, 0.01, len(idx))),
            "B": 100 * np.cumprod(1 + rng.normal(0.000, 0.01, len(idx))),
        },
        index=idx,
    )


@pytest.fixture(scope="module")
def wf_result(prices: pd.DataFrame) -> WalkForwardResult:
    splits = rolling_time_splits(prices.index, train_months=36, test_months=12)
    return run_walk_forward_validation(prices, EqualWeightStrategy(), splits)


# ---------------------------------------------------------------------------
# split_metrics_table
# ---------------------------------------------------------------------------


def test_split_metrics_table_returns_dataframe(wf_result: WalkForwardResult) -> None:
    t = split_metrics_table(wf_result)
    assert isinstance(t, pd.DataFrame)


def test_split_metrics_table_row_count(wf_result: WalkForwardResult) -> None:
    t = split_metrics_table(wf_result)
    assert len(t) == wf_result.n_splits


def test_split_metrics_table_has_sharpe(wf_result: WalkForwardResult) -> None:
    t = split_metrics_table(wf_result)
    assert "sharpe_ratio" in t.columns


def test_split_metrics_table_has_date_cols(wf_result: WalkForwardResult) -> None:
    t = split_metrics_table(wf_result)
    for col in ["train_start", "train_end", "test_start", "test_end"]:
        assert col in t.columns


def test_split_metrics_table_empty_result() -> None:
    empty = WalkForwardResult(strategy_name="test", splits=[])
    t = split_metrics_table(empty)
    assert t.empty


# ---------------------------------------------------------------------------
# summarize_stability
# ---------------------------------------------------------------------------


def test_summarize_stability_returns_dict(wf_result: WalkForwardResult) -> None:
    s = summarize_stability(wf_result)
    assert isinstance(s, dict)


def test_summarize_stability_has_n_splits(wf_result: WalkForwardResult) -> None:
    s = summarize_stability(wf_result)
    assert s["n_splits"] == wf_result.n_splits


def test_summarize_stability_required_keys(wf_result: WalkForwardResult) -> None:
    s = summarize_stability(wf_result)
    for key in ["mean_sharpe", "std_sharpe", "hit_rate_positive_sharpe",
                "mean_annualized_return", "mean_max_drawdown"]:
        assert key in s


def test_summarize_stability_hit_rate_in_range(wf_result: WalkForwardResult) -> None:
    s = summarize_stability(wf_result)
    assert 0.0 <= s["hit_rate_positive_sharpe"] <= 1.0


def test_summarize_stability_max_dd_nonpositive(wf_result: WalkForwardResult) -> None:
    s = summarize_stability(wf_result)
    assert s["mean_max_drawdown"] <= 0.0 + 1e-9
    assert s["worst_max_drawdown"] <= 0.0 + 1e-9


def test_summarize_stability_empty_result() -> None:
    empty = WalkForwardResult(strategy_name="test", splits=[])
    s = summarize_stability(empty)
    assert s["n_splits"] == 0


# ---------------------------------------------------------------------------
# rolling_sharpe_by_split
# ---------------------------------------------------------------------------


def test_rolling_sharpe_by_split_returns_series(wf_result: WalkForwardResult) -> None:
    s = rolling_sharpe_by_split(wf_result)
    assert isinstance(s, pd.Series)


def test_rolling_sharpe_by_split_length(wf_result: WalkForwardResult) -> None:
    s = rolling_sharpe_by_split(wf_result)
    assert len(s) == wf_result.n_splits


def test_rolling_sharpe_by_split_index_is_test_starts(wf_result: WalkForwardResult) -> None:
    s = rolling_sharpe_by_split(wf_result)
    expected = [sr.split.test_start for sr in wf_result.splits]
    assert list(s.index) == expected


# ---------------------------------------------------------------------------
# parameter_robustness_summary
# ---------------------------------------------------------------------------


def test_robustness_summary_returns_dataframe(prices: pd.DataFrame) -> None:
    splits = rolling_time_splits(prices.index, train_months=36, test_months=12)
    results = {
        "equal_weight": run_walk_forward_validation(prices, EqualWeightStrategy(), splits),
    }
    df = parameter_robustness_summary(results)
    assert isinstance(df, pd.DataFrame)


def test_robustness_summary_row_per_strategy(prices: pd.DataFrame) -> None:
    from src.strategies.baselines import BuyAndHoldStrategy
    splits = rolling_time_splits(prices.index, train_months=36, test_months=12)
    results = {
        "ew": run_walk_forward_validation(prices, EqualWeightStrategy(), splits),
        "bh": run_walk_forward_validation(prices, BuyAndHoldStrategy(), splits),
    }
    df = parameter_robustness_summary(results)
    assert set(df.index) == {"ew", "bh"}


def test_robustness_summary_has_mean_and_std(prices: pd.DataFrame) -> None:
    splits = rolling_time_splits(prices.index, train_months=36, test_months=12)
    results = {"ew": run_walk_forward_validation(prices, EqualWeightStrategy(), splits)}
    df = parameter_robustness_summary(results)
    assert "mean" in df.columns
    assert "std" in df.columns


def test_robustness_summary_invalid_metric(prices: pd.DataFrame) -> None:
    splits = rolling_time_splits(prices.index, train_months=36, test_months=12)
    results = {"ew": run_walk_forward_validation(prices, EqualWeightStrategy(), splits)}
    df = parameter_robustness_summary(results, metric="nonexistent")
    # Should return NaN row rather than raise
    assert df.loc["ew", "mean"] != df.loc["ew", "mean"]  # isnan check
