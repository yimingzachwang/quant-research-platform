"""Tests for src/features/volatility.py."""

import math

import numpy as np
import pandas as pd
import pytest

from src.features.volatility import ewm_volatility, rolling_volatility

_ANN = math.sqrt(252)


@pytest.fixture()
def const_returns() -> pd.Series:
    """Constant 1% daily returns — volatility should be zero."""
    return pd.Series([0.01] * 30)


@pytest.fixture()
def varied_returns() -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0, 0.01, 60))


def test_rolling_vol_nan_before_window(varied_returns: pd.Series) -> None:
    vol = rolling_volatility(varied_returns, window=20, annualize=False)
    assert vol.iloc[:19].isna().all()
    assert vol.iloc[19:].notna().all()


def test_rolling_vol_constant_returns(const_returns: pd.Series) -> None:
    vol = rolling_volatility(const_returns, window=10, annualize=False)
    # std of a constant series is 0
    assert (vol.dropna().abs() < 1e-12).all()


def test_rolling_vol_annualization(varied_returns: pd.Series) -> None:
    raw = rolling_volatility(varied_returns, window=20, annualize=False)
    ann = rolling_volatility(varied_returns, window=20, annualize=True)
    ratio = (ann / raw).dropna()
    assert ratio.iloc[0] == pytest.approx(_ANN)


def test_rolling_vol_name() -> None:
    s = pd.Series([0.01] * 30)
    assert rolling_volatility(s, 20).name == "vol_20d_ann"
    assert rolling_volatility(s, 20, annualize=False).name == "vol_20d"


def test_ewm_vol_no_leading_nan(varied_returns: pd.Series) -> None:
    vol = ewm_volatility(varied_returns, span=20, annualize=False)
    # EWM std requires at least 2 observations; bar 0 is NaN, bar 1+ are not
    assert vol.iloc[1:].notna().all()


def test_ewm_vol_annualization(varied_returns: pd.Series) -> None:
    raw = ewm_volatility(varied_returns, span=20, annualize=False)
    ann = ewm_volatility(varied_returns, span=20, annualize=True)
    ratio = ann / raw
    assert ratio.iloc[-1] == pytest.approx(_ANN)
