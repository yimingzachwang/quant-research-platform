"""Smoke tests for the G2 feature family IC visualization.

Tests verify that plot_feature_family_ic returns a Figure without error
across normal data, single-family, empty data, and unknown-family cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import pytest

matplotlib.use("Agg")

from src.visualization.ml_plots import plot_feature_family_ic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FEATURE_NAMES = ["mom_5", "mom_20", "vol_21", "zscore_20", "trend_20"]
_N_SPLITS = 6

_FAMILIES = {
    "Trend": ["mom_5", "mom_20", "trend_20"],
    "Volatility": ["vol_21"],
    "Mean-Reversion": ["zscore_20"],
}


def _make_ic_df(
    n_splits: int = _N_SPLITS,
    features: list[str] = _FEATURE_NAMES,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.uniform(-0.1, 0.1, (n_splits, len(features)))
    return pd.DataFrame(data, index=range(n_splits), columns=features)


# ---------------------------------------------------------------------------
# Normal cases
# ---------------------------------------------------------------------------

def test_feature_family_ic_returns_figure():
    import matplotlib.pyplot as plt
    df = _make_ic_df()
    fig = plot_feature_family_ic(df, _FAMILIES)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_feature_family_ic_with_split_labels():
    import matplotlib.pyplot as plt
    df = _make_ic_df()
    labels = ["2016-01", "2017-01", "2018-01", "2019-01", "2020-01", "2021-01"]
    fig = plot_feature_family_ic(df, _FAMILIES, split_labels=labels)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_feature_family_ic_single_family():
    import matplotlib.pyplot as plt
    df = _make_ic_df(features=["mom_5", "mom_20"])
    families = {"Trend": ["mom_5", "mom_20"]}
    fig = plot_feature_family_ic(df, families)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_feature_family_ic_five_families():
    import matplotlib.pyplot as plt
    feats = ["mom_5", "vol_21", "zscore_20", "skew_60d", "cross_sec_rank"]
    families = {
        "Trend": ["mom_5"],
        "Volatility": ["vol_21"],
        "Mean-Reversion": ["zscore_20"],
        "Market Structure": ["skew_60d"],
        "Relative Strength": ["cross_sec_rank"],
    }
    df = _make_ic_df(features=feats)
    fig = plot_feature_family_ic(df, families)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_feature_family_ic_with_nans():
    import matplotlib.pyplot as plt
    df = _make_ic_df()
    df.iloc[2, 0] = float("nan")
    fig = plot_feature_family_ic(df, _FAMILIES)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_feature_family_ic_mismatched_split_labels_falls_back():
    """When split_labels length doesn't match n_splits, fall back to 'S{i}' labels."""
    import matplotlib.pyplot as plt
    df = _make_ic_df()
    labels = ["2016-01", "2017-01"]  # Too short
    fig = plot_feature_family_ic(df, _FAMILIES, split_labels=labels)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_feature_family_ic_empty_df_returns_figure():
    import matplotlib.pyplot as plt
    fig = plot_feature_family_ic(pd.DataFrame(), _FAMILIES)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_feature_family_ic_no_matching_members_returns_figure():
    import matplotlib.pyplot as plt
    df = _make_ic_df()
    # None of the family members are in df.columns
    families = {"Trend": ["nonexistent_feature_a", "nonexistent_feature_b"]}
    fig = plot_feature_family_ic(df, families)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_feature_family_ic_single_split():
    import matplotlib.pyplot as plt
    df = _make_ic_df(n_splits=1)
    fig = plot_feature_family_ic(df, _FAMILIES)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_feature_family_ic_all_positive_ic():
    import matplotlib.pyplot as plt
    df = _make_ic_df()
    df.iloc[:, :] = 0.05
    fig = plot_feature_family_ic(df, _FAMILIES)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_feature_family_ic_all_negative_ic():
    import matplotlib.pyplot as plt
    df = _make_ic_df()
    df.iloc[:, :] = -0.03
    fig = plot_feature_family_ic(df, _FAMILIES)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
