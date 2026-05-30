"""Tests for src/portfolio/allocation.py."""

import numpy as np
import pandas as pd
import pytest
from src.portfolio.allocation import equal_weight, resample_weights_to_daily, volatility_scaled


@pytest.fixture()
def mask_top2() -> pd.DataFrame:
    """3-asset mask where A and B are always selected."""
    idx = pd.date_range("2021-01-01", periods=20, freq="B")
    return pd.DataFrame(
        {"A": True, "B": True, "C": False},
        index=idx,
    )


@pytest.fixture()
def mask_all() -> pd.DataFrame:
    """4-asset mask where all assets are always selected."""
    idx = pd.date_range("2021-01-01", periods=20, freq="B")
    return pd.DataFrame(True, index=idx, columns=["A", "B", "C", "D"])


@pytest.fixture()
def mask_none() -> pd.DataFrame:
    """All-False mask — no asset selected."""
    idx = pd.date_range("2021-01-01", periods=5, freq="B")
    return pd.DataFrame(False, index=idx, columns=["A", "B"])


# ---------------------------------------------------------------------------
# equal_weight
# ---------------------------------------------------------------------------


def test_equal_weight_sums_to_one(mask_top2: pd.DataFrame) -> None:
    w = equal_weight(mask_top2)
    assert ((w.sum(axis=1) - 1.0).abs() < 1e-9).all()


def test_equal_weight_value(mask_top2: pd.DataFrame) -> None:
    w = equal_weight(mask_top2)
    assert ((w["A"] - 0.5).abs() < 1e-9).all()
    assert ((w["B"] - 0.5).abs() < 1e-9).all()
    assert (w["C"] == 0.0).all()


def test_equal_weight_all_selected(mask_all: pd.DataFrame) -> None:
    w = equal_weight(mask_all)
    assert ((w.sum(axis=1) - 1.0).abs() < 1e-9).all()
    assert ((w - 0.25).abs() < 1e-9).all().all()


def test_equal_weight_no_selection_is_zero(mask_none: pd.DataFrame) -> None:
    w = equal_weight(mask_none)
    assert (w == 0.0).all().all()


def test_equal_weight_non_negative(mask_top2: pd.DataFrame) -> None:
    w = equal_weight(mask_top2)
    assert (w >= 0).all().all()


# ---------------------------------------------------------------------------
# volatility_scaled
# ---------------------------------------------------------------------------


@pytest.fixture()
def returns_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2021-01-01", periods=200, freq="B")
    return pd.DataFrame(rng.normal(0, 0.01, (200, 3)), index=idx,
                        columns=["A", "B", "C"])


def test_volatility_scaled_sums_to_one(returns_df: pd.DataFrame) -> None:
    idx = returns_df.index
    mask = pd.DataFrame(True, index=idx, columns=returns_df.columns)
    w = volatility_scaled(mask, returns_df, window=20)
    valid_rows = w.sum(axis=1)
    # After warm-up all selected rows should sum to ~1
    assert ((valid_rows.iloc[20:] - 1.0).abs() < 1e-6).all()


def test_volatility_scaled_max_weight_cap(returns_df: pd.DataFrame) -> None:
    idx = returns_df.index
    mask = pd.DataFrame(True, index=idx, columns=returns_df.columns)
    w = volatility_scaled(mask, returns_df, window=20, max_weight=0.5)
    assert (w.iloc[20:] <= 0.5 + 1e-9).all().all()


def test_volatility_scaled_unselected_zero(returns_df: pd.DataFrame) -> None:
    idx = returns_df.index
    mask = pd.DataFrame(
        {"A": True, "B": True, "C": False}, index=idx
    )
    w = volatility_scaled(mask, returns_df, window=20)
    assert (w["C"] == 0.0).all()


# ---------------------------------------------------------------------------
# resample_weights_to_daily
# ---------------------------------------------------------------------------


def test_resample_weights_to_daily_shape() -> None:
    monthly_idx = pd.date_range("2021-01-31", periods=6, freq="ME")
    weights = pd.DataFrame({"A": 0.5, "B": 0.5}, index=monthly_idx)
    daily_idx = pd.date_range("2021-01-01", "2021-06-30", freq="B")
    daily = resample_weights_to_daily(weights, daily_idx)
    assert len(daily) == len(daily_idx)


def test_resample_weights_to_daily_no_nan_after_first_signal() -> None:
    monthly_idx = pd.date_range("2021-01-31", periods=3, freq="ME")
    weights = pd.DataFrame({"A": [0.6, 0.4, 0.5], "B": [0.4, 0.6, 0.5]}, index=monthly_idx)
    daily_idx = pd.date_range("2021-01-31", "2021-03-31", freq="B")
    daily = resample_weights_to_daily(weights, daily_idx)
    # From the first rebalance date onward there should be no NaN
    assert not daily.isna().any().any()
