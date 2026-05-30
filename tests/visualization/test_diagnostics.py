"""Tests for src/visualization/diagnostics.py and the five new portfolio plot functions."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from src.visualization.diagnostics import (
    compute_concentration_metrics,
    compute_turnover,
    rolling_average_correlation,
)
from src.visualization.portfolio_plots import (
    plot_asset_contribution,
    plot_rolling_correlation,
    plot_turnover,
    plot_weight_concentration,
    plot_weight_heatmap,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def weights() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    rng = np.random.default_rng(7)
    raw = rng.random((200, 3))
    raw = raw / raw.sum(axis=1, keepdims=True)
    return pd.DataFrame(raw, index=idx, columns=["A", "B", "C"])


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    rng = np.random.default_rng(8)
    return pd.DataFrame(rng.normal(0.001, 0.01, (200, 3)), index=idx, columns=["A", "B", "C"])


# ---------------------------------------------------------------------------
# compute_turnover
# ---------------------------------------------------------------------------


def test_compute_turnover_returns_series(weights: pd.DataFrame) -> None:
    to = compute_turnover(weights)
    assert isinstance(to, pd.Series)


def test_compute_turnover_length(weights: pd.DataFrame) -> None:
    to = compute_turnover(weights)
    assert len(to) == len(weights)


def test_compute_turnover_first_row_is_nan(weights: pd.DataFrame) -> None:
    to = compute_turnover(weights)
    assert np.isnan(to.iloc[0])


def test_compute_turnover_nonnegative(weights: pd.DataFrame) -> None:
    to = compute_turnover(weights).dropna()
    assert (to >= 0).all()


def test_compute_turnover_constant_weights_is_zero() -> None:
    idx = pd.date_range("2021-01-01", periods=10, freq="B")
    w = pd.DataFrame({"A": 0.5, "B": 0.5}, index=idx)
    to = compute_turnover(w).dropna()
    assert (to.abs() < 1e-12).all()


# ---------------------------------------------------------------------------
# compute_concentration_metrics
# ---------------------------------------------------------------------------


def test_concentration_returns_dataframe(weights: pd.DataFrame) -> None:
    m = compute_concentration_metrics(weights)
    assert isinstance(m, pd.DataFrame)


def test_concentration_columns(weights: pd.DataFrame) -> None:
    m = compute_concentration_metrics(weights)
    assert set(m.columns) == {"hhi", "max_weight", "effective_n"}


def test_concentration_hhi_bounds(weights: pd.DataFrame) -> None:
    m = compute_concentration_metrics(weights)
    assert (m["hhi"] >= 0).all()
    assert (m["hhi"] <= 1.0 + 1e-9).all()


def test_concentration_max_weight_bounds(weights: pd.DataFrame) -> None:
    m = compute_concentration_metrics(weights)
    assert (m["max_weight"] >= 0).all()
    assert (m["max_weight"] <= 1.0 + 1e-9).all()


def test_concentration_effective_n_min_1(weights: pd.DataFrame) -> None:
    m = compute_concentration_metrics(weights).dropna()
    assert (m["effective_n"] >= 1.0 - 1e-9).all()


def test_concentration_equal_weight_effective_n() -> None:
    idx = pd.date_range("2021-01-01", periods=5, freq="B")
    w = pd.DataFrame({"A": 1/3, "B": 1/3, "C": 1/3}, index=idx)
    m = compute_concentration_metrics(w)
    assert (abs(m["effective_n"] - 3.0) < 1e-9).all()


def test_concentration_all_in_one_asset() -> None:
    idx = pd.date_range("2021-01-01", periods=5, freq="B")
    w = pd.DataFrame({"A": 1.0, "B": 0.0}, index=idx)
    m = compute_concentration_metrics(w)
    assert (abs(m["hhi"] - 1.0) < 1e-9).all()
    assert (abs(m["effective_n"] - 1.0) < 1e-9).all()


# ---------------------------------------------------------------------------
# rolling_average_correlation
# ---------------------------------------------------------------------------


def test_rolling_avg_corr_returns_series(returns: pd.DataFrame) -> None:
    s = rolling_average_correlation(returns, window=30)
    assert isinstance(s, pd.Series)


def test_rolling_avg_corr_length(returns: pd.DataFrame) -> None:
    s = rolling_average_correlation(returns, window=30)
    assert len(s) == len(returns)


def test_rolling_avg_corr_range(returns: pd.DataFrame) -> None:
    s = rolling_average_correlation(returns, window=30).dropna()
    assert (s >= -1.0 - 1e-9).all()
    assert (s <= 1.0 + 1e-9).all()


def test_rolling_avg_corr_single_column() -> None:
    idx = pd.date_range("2021-01-01", periods=50, freq="B")
    r = pd.DataFrame({"A": np.random.default_rng(1).normal(size=50)}, index=idx)
    s = rolling_average_correlation(r, window=20)
    assert s.isna().all()


def test_rolling_avg_corr_perfect_corr() -> None:
    idx = pd.date_range("2021-01-01", periods=100, freq="B")
    x = pd.Series(range(100), dtype=float)
    r = pd.DataFrame({"A": x, "B": x}, index=idx)
    s = rolling_average_correlation(r, window=20).dropna()
    assert (abs(s - 1.0) < 1e-9).all()


# ---------------------------------------------------------------------------
# Plot smoke tests (return plt.Figure, don't crash)
# ---------------------------------------------------------------------------


def test_plot_weight_heatmap_returns_figure(weights: pd.DataFrame) -> None:
    fig = plot_weight_heatmap(weights)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_turnover_returns_figure(weights: pd.DataFrame) -> None:
    fig = plot_turnover(weights)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_weight_concentration_returns_figure(weights: pd.DataFrame) -> None:
    fig = plot_weight_concentration(weights)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_asset_contribution_returns_figure(
    returns: pd.DataFrame, weights: pd.DataFrame
) -> None:
    fig = plot_asset_contribution(returns, weights)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_rolling_correlation_returns_figure(returns: pd.DataFrame) -> None:
    fig = plot_rolling_correlation(returns, window=30)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_weight_heatmap_save(weights: pd.DataFrame, tmp_path) -> None:
    out = tmp_path / "heatmap.png"
    fig = plot_weight_heatmap(weights, save_path=str(out))
    assert out.exists()
    plt.close(fig)


def test_plot_turnover_save(weights: pd.DataFrame, tmp_path) -> None:
    out = tmp_path / "turnover.png"
    fig = plot_turnover(weights, save_path=str(out))
    assert out.exists()
    plt.close(fig)


def test_plot_weight_concentration_save(weights: pd.DataFrame, tmp_path) -> None:
    out = tmp_path / "concentration.png"
    fig = plot_weight_concentration(weights, save_path=str(out))
    assert out.exists()
    plt.close(fig)


def test_plot_asset_contribution_save(
    returns: pd.DataFrame, weights: pd.DataFrame, tmp_path
) -> None:
    out = tmp_path / "contribution.png"
    fig = plot_asset_contribution(returns, weights, save_path=str(out))
    assert out.exists()
    plt.close(fig)


def test_plot_rolling_correlation_save(returns: pd.DataFrame, tmp_path) -> None:
    out = tmp_path / "rolling_corr.png"
    fig = plot_rolling_correlation(returns, window=30, save_path=str(out))
    assert out.exists()
    plt.close(fig)


def test_plot_turnover_custom_title(weights: pd.DataFrame) -> None:
    fig = plot_turnover(weights, title="Custom Title")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_rolling_correlation_single_column() -> None:
    idx = pd.date_range("2021-01-01", periods=80, freq="B")
    r = pd.DataFrame({"A": np.random.default_rng(2).normal(size=80)}, index=idx)
    fig = plot_rolling_correlation(r, window=20)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
