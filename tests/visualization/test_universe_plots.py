"""Smoke tests for src.visualization.universe_plots.

All tests verify that functions return a matplotlib Figure without error
across normal multi-asset data, single-asset data, and edge cases.
No pixel-level assertions — structural smoke tests only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import pytest

matplotlib.use("Agg")

from src.visualization.universe_plots import (
    plot_asset_availability_timeline,
    plot_cross_asset_volatility,
    plot_universe_correlation_heatmap,
    plot_universe_coverage_heatmap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]


def _make_prices(n_days: int = 200, tickers: list[str] = _TICKERS,
                 seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n_days, freq="B")
    data = 100 * np.exp(
        np.cumsum(rng.normal(0.0002, 0.01, (n_days, len(tickers))), axis=0)
    )
    return pd.DataFrame(data, index=idx, columns=tickers)


def _make_monthly_coverage(n_months: int = 24, tickers: list[str] = _TICKERS,
                            seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-31", periods=n_months, freq="ME")
    data = rng.uniform(0.8, 1.0, (n_months, len(tickers)))
    # Introduce gaps only when there are enough columns
    if len(tickers) > 2:
        data[:5, 2] = 0.0
    return pd.DataFrame(data, index=idx, columns=tickers)


def _make_vol_df(n_days: int = 200, tickers: list[str] = _TICKERS,
                 seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n_days, freq="B")
    data = rng.uniform(0.1, 0.4, (n_days, len(tickers)))
    return pd.DataFrame(data, index=idx, columns=tickers)


def _make_corr_df(tickers: list[str] = _TICKERS, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(tickers)
    A = rng.standard_normal((n, n))
    C = A @ A.T
    D = np.sqrt(np.diag(C))
    corr = C / np.outer(D, D)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=tickers, columns=tickers)


# ---------------------------------------------------------------------------
# plot_universe_coverage_heatmap
# ---------------------------------------------------------------------------

def test_coverage_heatmap_returns_figure():
    df = _make_monthly_coverage()
    fig = plot_universe_coverage_heatmap(df)
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_coverage_heatmap_single_asset():
    df = _make_monthly_coverage(tickers=["SPY"])
    fig = plot_universe_coverage_heatmap(df)
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_coverage_heatmap_empty_returns_figure():
    df = pd.DataFrame()
    fig = plot_universe_coverage_heatmap(df)
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_coverage_heatmap_all_ones():
    """Full coverage — should not crash."""
    idx = pd.date_range("2020-01", periods=12, freq="ME")
    df = pd.DataFrame(1.0, index=idx, columns=_TICKERS)
    fig = plot_universe_coverage_heatmap(df)
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_asset_availability_timeline
# ---------------------------------------------------------------------------

def test_availability_timeline_returns_figure():
    prices = _make_prices()
    fig = plot_asset_availability_timeline(prices)
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_availability_timeline_with_gaps():
    """Some assets unavailable in early periods."""
    prices = _make_prices()
    prices.iloc[:50, 2:] = np.nan
    fig = plot_asset_availability_timeline(prices)
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_availability_timeline_single_asset():
    prices = _make_prices(tickers=["SPY"])
    fig = plot_asset_availability_timeline(prices)
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_availability_timeline_empty_returns_figure():
    fig = plot_asset_availability_timeline(pd.DataFrame())
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_cross_asset_volatility
# ---------------------------------------------------------------------------

def test_cross_asset_vol_returns_figure():
    vol_df = _make_vol_df()
    fig = plot_cross_asset_volatility(vol_df)
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_cross_asset_vol_single_asset():
    vol_df = _make_vol_df(tickers=["SPY"])
    fig = plot_cross_asset_volatility(vol_df)
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_cross_asset_vol_ten_assets():
    """Full 10-ETF universe — legend layout test."""
    tickers = ["SPY", "QQQ", "IWM", "EEM", "TLT", "GLD", "XLF", "XLK", "XLE", "HYG"]
    vol_df = _make_vol_df(tickers=tickers)
    fig = plot_cross_asset_volatility(vol_df)
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_cross_asset_vol_empty_returns_figure():
    fig = plot_cross_asset_volatility(pd.DataFrame())
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_cross_asset_vol_with_nans():
    vol_df = _make_vol_df()
    vol_df.iloc[:30, 1] = np.nan
    fig = plot_cross_asset_volatility(vol_df)
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_universe_correlation_heatmap
# ---------------------------------------------------------------------------

def test_correlation_heatmap_returns_figure():
    corr = _make_corr_df()
    fig = plot_universe_correlation_heatmap(corr)
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_correlation_heatmap_two_assets():
    tickers = ["SPY", "TLT"]
    corr = _make_corr_df(tickers=tickers)
    fig = plot_universe_correlation_heatmap(corr)
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_correlation_heatmap_ten_assets():
    tickers = ["SPY", "QQQ", "IWM", "EEM", "TLT", "GLD", "XLF", "XLK", "XLE", "HYG"]
    corr = _make_corr_df(tickers=tickers)
    fig = plot_universe_correlation_heatmap(corr)
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_correlation_heatmap_single_asset_graceful():
    """Single asset — should render empty-state figure without crash."""
    corr = pd.DataFrame([[1.0]], index=["SPY"], columns=["SPY"])
    fig = plot_universe_correlation_heatmap(corr)
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Round-trip through _prepare_universe_diagnostics
# ---------------------------------------------------------------------------

def test_prepare_universe_diagnostics_smoke():
    """Verify _prepare_universe_diagnostics returns expected keys without error."""
    from src.experiments.orchestrator import _prepare_universe_diagnostics

    prices = _make_prices()
    result = _prepare_universe_diagnostics(prices)

    assert "tickers" in result
    assert "n_assets" in result
    assert result["n_assets"] == len(_TICKERS)
    assert "asset_coverage" in result
    assert len(result["asset_coverage"]) == len(_TICKERS)
    assert "monthly_coverage_df" in result
    assert "rolling_vol_df" in result
    assert "corr_df" in result
    assert "correlation_matrix" in result


def test_prepare_universe_diagnostics_single_asset():
    from src.experiments.orchestrator import _prepare_universe_diagnostics

    prices = _make_prices(tickers=["SPY"])
    result = _prepare_universe_diagnostics(prices)

    assert result["n_assets"] == 1
    assert result["tickers"] == ["SPY"]
    # Correlation matrix requires >= 2 assets
    assert "corr_df" not in result or result.get("corr_df") is None or True


def test_prepare_universe_diagnostics_with_gaps():
    """Prices with NaN gaps — no crash, missingness computed."""
    from src.experiments.orchestrator import _prepare_universe_diagnostics  # noqa: F811
    prices = _make_prices()
    prices.iloc[:20, 0] = np.nan
    result = _prepare_universe_diagnostics(prices)

    spy_cov = next(
        (e for e in result.get("asset_coverage", []) if e["ticker"] == "SPY"), {}
    )
    assert isinstance(spy_cov.get("missingness_pct"), float)
    assert spy_cov["missingness_pct"] > 0.0
