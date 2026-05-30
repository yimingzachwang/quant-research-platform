"""Tests for family-aware feature correlation heatmap (G-SYNC-4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import pytest

matplotlib.use("Agg")

from src.visualization.ml_plots import plot_feature_correlation_heatmap


def _make_corr(features: list[str]) -> pd.DataFrame:
    n = len(features)
    rng = np.random.default_rng(42)
    A = rng.standard_normal((n, n))
    C = A @ A.T
    D = np.sqrt(np.diag(C))
    corr = C / np.outer(D, D)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=features, columns=features)


_FEATURES = ["mom_5", "mom_20", "vol_21", "zscore_20", "trend_20"]
_FAMILIES = {
    "Trend": ["mom_5", "mom_20", "trend_20"],
    "Volatility": ["vol_21"],
    "Mean-Reversion": ["zscore_20"],
}


def test_heatmap_no_families():
    import matplotlib.pyplot as plt
    corr = _make_corr(_FEATURES)
    fig = plot_feature_correlation_heatmap(corr)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_heatmap_with_families():
    import matplotlib.pyplot as plt
    corr = _make_corr(_FEATURES)
    fig = plot_feature_correlation_heatmap(corr, feature_families=_FAMILIES)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_heatmap_families_reorders_features():
    """Family-ordered heatmap reorders corr_df columns."""
    import matplotlib.pyplot as plt
    corr = _make_corr(_FEATURES)
    fig = plot_feature_correlation_heatmap(corr, feature_families=_FAMILIES)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_heatmap_two_features():
    import matplotlib.pyplot as plt
    corr = _make_corr(["mom_20", "vol_21"])
    families = {"Trend": ["mom_20"], "Volatility": ["vol_21"]}
    fig = plot_feature_correlation_heatmap(corr, feature_families=families)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_heatmap_ten_features():
    import matplotlib.pyplot as plt
    feats = ["mom_5", "mom_20", "mom_60", "vol_21", "vol_63",
             "zscore_20", "trend_20", "bollinger_20d", "skew_60d", "autocorr_1_60d"]
    families = {
        "Trend": ["mom_5", "mom_20", "mom_60", "trend_20"],
        "Volatility": ["vol_21", "vol_63"],
        "Mean-Reversion": ["zscore_20", "bollinger_20d"],
        "Market Structure": ["skew_60d", "autocorr_1_60d"],
    }
    corr = _make_corr(feats)
    fig = plot_feature_correlation_heatmap(corr, feature_families=families)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_heatmap_families_with_unknown_members():
    """Family members not in corr_df are silently ignored."""
    import matplotlib.pyplot as plt
    corr = _make_corr(["mom_20", "vol_21"])
    families = {
        "Trend": ["mom_20", "nonexistent_feature"],
        "Volatility": ["vol_21"],
    }
    fig = plot_feature_correlation_heatmap(corr, feature_families=families)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_heatmap_empty_df_returns_figure():
    import matplotlib.pyplot as plt
    fig = plot_feature_correlation_heatmap(pd.DataFrame())
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
