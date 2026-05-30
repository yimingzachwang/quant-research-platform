"""Tests for src/features/returns.py."""

import numpy as np
import pandas as pd
import pytest

from src.features.returns import compute_cumulative_returns, compute_log_returns, compute_returns


@pytest.fixture()
def prices() -> pd.Series:
    return pd.Series([100.0, 110.0, 99.0, 108.9], name="close")


def test_compute_returns_first_is_nan(prices: pd.Series) -> None:
    r = compute_returns(prices)
    assert np.isnan(r.iloc[0])


def test_compute_returns_values(prices: pd.Series) -> None:
    r = compute_returns(prices)
    assert r.iloc[1] == pytest.approx(0.10)
    assert r.iloc[2] == pytest.approx((99 - 110) / 110)


def test_compute_log_returns_first_is_nan(prices: pd.Series) -> None:
    lr = compute_log_returns(prices)
    assert np.isnan(lr.iloc[0])


def test_compute_log_returns_values(prices: pd.Series) -> None:
    lr = compute_log_returns(prices)
    assert lr.iloc[1] == pytest.approx(np.log(110 / 100))


def test_compute_cumulative_returns_starts_at_zero() -> None:
    r = pd.Series([0.10, 0.05, -0.03])
    cr = compute_cumulative_returns(r)
    assert cr.iloc[0] == pytest.approx(0.10)


def test_compute_cumulative_returns_compound() -> None:
    r = pd.Series([0.10, 0.10])
    cr = compute_cumulative_returns(r)
    assert cr.iloc[1] == pytest.approx(1.10 * 1.10 - 1)


def test_compute_cumulative_returns_all_zero() -> None:
    r = pd.Series([0.0, 0.0, 0.0])
    cr = compute_cumulative_returns(r)
    assert (cr == 0.0).all()
