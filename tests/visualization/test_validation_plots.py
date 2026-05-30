"""Smoke tests for src/visualization/validation_plots.py."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from src.strategies.baselines import BuyAndHoldStrategy, EqualWeightStrategy
from src.validation.splits import rolling_time_splits
from src.validation.walk_forward import WalkForwardResult, run_walk_forward_validation
from src.visualization.validation_plots import (
    plot_metric_stability,
    plot_split_sharpes,
    plot_train_vs_test,
    plot_walk_forward_equity,
    plot_walk_forward_stitched,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    idx = pd.date_range("2010-01-01", "2019-12-31", freq="B")
    rng = np.random.default_rng(55)
    return pd.DataFrame(
        {
            "A": 100 * np.cumprod(1 + rng.normal(0.001, 0.01, len(idx))),
            "B": 100 * np.cumprod(1 + rng.normal(0.000, 0.01, len(idx))),
            "C": 100 * np.cumprod(1 + rng.normal(0.002, 0.01, len(idx))),
        },
        index=idx,
    )


@pytest.fixture(scope="module")
def wf_result(prices: pd.DataFrame) -> WalkForwardResult:
    splits = rolling_time_splits(prices.index, train_months=36, test_months=12)
    return run_walk_forward_validation(prices, EqualWeightStrategy(), splits)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_plot_walk_forward_equity_returns_figure(wf_result: WalkForwardResult) -> None:
    fig = plot_walk_forward_equity(wf_result)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_walk_forward_stitched_returns_figure(wf_result: WalkForwardResult) -> None:
    fig = plot_walk_forward_stitched(wf_result)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_split_sharpes_returns_figure(wf_result: WalkForwardResult) -> None:
    fig = plot_split_sharpes(wf_result)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_metric_stability_returns_figure(wf_result: WalkForwardResult) -> None:
    fig = plot_metric_stability(wf_result, metric="annualized_return")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_train_vs_test_returns_figure(wf_result: WalkForwardResult) -> None:
    fig = plot_train_vs_test(wf_result, metric="sharpe_ratio")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_metric_stability_invalid_metric(wf_result: WalkForwardResult) -> None:
    with pytest.raises(ValueError, match="not found"):
        plot_metric_stability(wf_result, metric="nonexistent_metric")


def test_plot_walk_forward_equity_empty_result() -> None:
    empty = WalkForwardResult(strategy_name="test", splits=[])
    fig = plot_walk_forward_equity(empty)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_walk_forward_stitched_empty_result() -> None:
    empty = WalkForwardResult(strategy_name="test", splits=[])
    fig = plot_walk_forward_stitched(empty)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_walk_forward_equity_save(wf_result: WalkForwardResult, tmp_path) -> None:
    out = tmp_path / "wf_equity.png"
    fig = plot_walk_forward_equity(wf_result, save_path=str(out))
    assert out.exists()
    plt.close(fig)


def test_plot_walk_forward_stitched_save(wf_result: WalkForwardResult, tmp_path) -> None:
    out = tmp_path / "wf_stitched.png"
    fig = plot_walk_forward_stitched(wf_result, save_path=str(out))
    assert out.exists()
    plt.close(fig)


def test_plot_split_sharpes_save(wf_result: WalkForwardResult, tmp_path) -> None:
    out = tmp_path / "split_sharpes.png"
    fig = plot_split_sharpes(wf_result, save_path=str(out))
    assert out.exists()
    plt.close(fig)


def test_plot_metric_stability_sharpe(wf_result: WalkForwardResult) -> None:
    fig = plot_metric_stability(wf_result, metric="sharpe_ratio")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_train_vs_test_with_second_result(
    prices: pd.DataFrame, wf_result: WalkForwardResult
) -> None:
    splits = rolling_time_splits(prices.index, train_months=36, test_months=12)
    train_wf = run_walk_forward_validation(prices, BuyAndHoldStrategy(), splits)
    fig = plot_train_vs_test(wf_result, metric="sharpe_ratio", train_results=train_wf)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
