"""Tests for src/backtesting/metrics.py."""

import math

import numpy as np
import pandas as pd
import pytest
from src.backtesting.metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    compute_metrics,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
)

_PERIODS = 252


@pytest.fixture()
def flat_positive() -> pd.Series:
    """Constant 0.1% daily returns — no drawdown, positive Sharpe."""
    return pd.Series(0.001, index=range(_PERIODS))


@pytest.fixture()
def zero_returns() -> pd.Series:
    return pd.Series(0.0, index=range(_PERIODS))


@pytest.fixture()
def random_returns() -> pd.Series:
    rng = np.random.default_rng(7)
    return pd.Series(rng.normal(0.0003, 0.01, _PERIODS))


# --- annualized_return ---


def test_annualized_return_positive(flat_positive: pd.Series) -> None:
    r = annualized_return(flat_positive)
    # CAGR of 0.1%/day over 252 days ≈ 1.001^252 - 1 ≈ 28.4%
    assert r == pytest.approx((1.001**252) - 1, rel=1e-6)


def test_annualized_return_zero(zero_returns: pd.Series) -> None:
    assert annualized_return(zero_returns) == pytest.approx(0.0)


def test_annualized_return_empty() -> None:
    assert math.isnan(annualized_return(pd.Series([], dtype=float)))


def test_annualized_return_all_nan() -> None:
    s = pd.Series([float("nan")] * 10)
    assert math.isnan(annualized_return(s))


# --- annualized_volatility ---


def test_annualized_volatility_positive(random_returns: pd.Series) -> None:
    vol = annualized_volatility(random_returns)
    expected = random_returns.std() * math.sqrt(_PERIODS)
    assert vol == pytest.approx(expected, rel=1e-6)


def test_annualized_volatility_zero(zero_returns: pd.Series) -> None:
    assert annualized_volatility(zero_returns) == pytest.approx(0.0, abs=1e-12)


def test_annualized_volatility_single_obs() -> None:
    assert math.isnan(annualized_volatility(pd.Series([0.01])))


# --- sharpe_ratio ---


def test_sharpe_ratio_positive_returns_positive_sharpe(flat_positive: pd.Series) -> None:
    sr = sharpe_ratio(flat_positive)
    assert sr > 0.0


def test_sharpe_ratio_zero_returns(zero_returns: pd.Series) -> None:
    # All zeros → std=0 → NaN Sharpe
    assert math.isnan(sharpe_ratio(zero_returns))


def test_sharpe_ratio_symmetry(random_returns: pd.Series) -> None:
    neg = -random_returns
    assert sharpe_ratio(random_returns) == pytest.approx(-sharpe_ratio(neg), rel=1e-6)


def test_sharpe_ratio_risk_free_reduces_sharpe(random_returns: pd.Series) -> None:
    # Use random (non-constant) returns so std > 0 and Sharpe is defined
    sr_no_rf = sharpe_ratio(random_returns, risk_free_rate=0.0)
    sr_with_rf = sharpe_ratio(random_returns, risk_free_rate=0.10)
    assert sr_with_rf < sr_no_rf


# --- max_drawdown ---


def test_max_drawdown_zero_for_monotone_positive(flat_positive: pd.Series) -> None:
    assert max_drawdown(flat_positive) == pytest.approx(0.0, abs=1e-10)


def test_max_drawdown_negative_for_losses() -> None:
    r = pd.Series([0.10, -0.20, 0.05])
    mdd = max_drawdown(r)
    assert mdd < 0.0


def test_max_drawdown_known_value() -> None:
    # equity: 1.0 → 1.1 → 0.88 → 0.924
    # peak at 1.1, trough at 0.88 → DD = (0.88 - 1.1) / 1.1 = -0.2
    r = pd.Series([0.10, -0.20, 0.05])
    mdd = max_drawdown(r)
    assert mdd == pytest.approx((0.88 - 1.1) / 1.1, rel=1e-6)


def test_max_drawdown_empty() -> None:
    assert max_drawdown(pd.Series([], dtype=float)) == 0.0


def test_max_drawdown_always_non_positive(random_returns: pd.Series) -> None:
    assert max_drawdown(random_returns) <= 0.0


# --- calmar_ratio ---


def test_calmar_ratio_zero_drawdown_is_nan(flat_positive: pd.Series) -> None:
    assert math.isnan(calmar_ratio(flat_positive))


def test_calmar_ratio_sign() -> None:
    # 220 days up, 32 days mild dip — net positive CAGR, non-zero drawdown
    r = pd.Series([0.001] * 220 + [-0.003] * 32)
    cr = calmar_ratio(r)
    assert cr > 0.0


# --- hit_rate ---


def test_hit_rate_all_positive(flat_positive: pd.Series) -> None:
    assert hit_rate(flat_positive) == pytest.approx(1.0)


def test_hit_rate_all_negative() -> None:
    s = pd.Series([-0.01] * 50)
    assert hit_rate(s) == pytest.approx(0.0)


def test_hit_rate_half() -> None:
    s = pd.Series([0.01, -0.01] * 50)
    assert hit_rate(s) == pytest.approx(0.5)


def test_hit_rate_empty() -> None:
    assert math.isnan(hit_rate(pd.Series([], dtype=float)))


# --- compute_metrics ---


def test_compute_metrics_keys(random_returns: pd.Series) -> None:
    m = compute_metrics(random_returns)
    expected = {
        "annualized_return", "annualized_volatility", "sharpe_ratio",
        "max_drawdown", "calmar_ratio", "hit_rate",
    }
    assert set(m.keys()) == expected


def test_compute_metrics_values_match_individual(random_returns: pd.Series) -> None:
    m = compute_metrics(random_returns)
    assert m["annualized_return"] == pytest.approx(annualized_return(random_returns))
    assert m["max_drawdown"] == pytest.approx(max_drawdown(random_returns))
    assert m["hit_rate"] == pytest.approx(hit_rate(random_returns))
