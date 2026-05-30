"""Smoke tests for src/visualization/comparison_plots.py.

Tests only verify that functions return plt.Figure without crashing.
No assertions on visual content (pixel-level or layout testing is out of scope).
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from src.strategies.baselines import BuyAndHoldStrategy, EqualWeightStrategy
from src.strategies.comparison import compare_strategies, metrics_table
from src.visualization.comparison_plots import (
    plot_metric_comparison,
    plot_metrics_table,
    plot_strategy_drawdowns,
    plot_strategy_equity_curves,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=150, freq="B")
    rng = np.random.default_rng(123)
    return pd.DataFrame(
        {
            "A": 100 * np.cumprod(1 + rng.normal(0.001, 0.01, 150)),
            "B": 100 * np.cumprod(1 + rng.normal(0.000, 0.01, 150)),
        },
        index=idx,
    )


@pytest.fixture(scope="module")
def results(prices: pd.DataFrame) -> dict:
    strategies = [BuyAndHoldStrategy(), EqualWeightStrategy()]
    return compare_strategies(prices, strategies)


@pytest.fixture(scope="module")
def table(results: dict) -> pd.DataFrame:
    return metrics_table(results)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_plot_equity_curves_returns_figure(results: dict) -> None:
    fig = plot_strategy_equity_curves(results)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_drawdowns_returns_figure(results: dict) -> None:
    fig = plot_strategy_drawdowns(results)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_metric_comparison_returns_figure(table: pd.DataFrame) -> None:
    fig = plot_metric_comparison(table, metric="sharpe_ratio")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_metrics_table_returns_figure(table: pd.DataFrame) -> None:
    fig = plot_metrics_table(table)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_metric_comparison_invalid_metric(table: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="not in DataFrame columns"):
        plot_metric_comparison(table, metric="nonexistent")


def test_plot_equity_curves_single_strategy(prices: pd.DataFrame) -> None:
    results = compare_strategies(prices, [BuyAndHoldStrategy()])
    fig = plot_strategy_equity_curves(results, title="Single")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_with_save_path(results: dict, tmp_path) -> None:
    out = tmp_path / "equity.png"
    fig = plot_strategy_equity_curves(results, save_path=str(out))
    assert out.exists()
    plt.close(fig)
