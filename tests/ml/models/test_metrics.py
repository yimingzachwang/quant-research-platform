"""Tests for pure metric functions.

Focus: deterministic values on known inputs, NaN handling, edge cases.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.ml.models.metrics import (
    correlation_coefficient,
    directional_accuracy,
    mae,
    mse,
    r2_score,
    rmse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _series(vals, start="2020-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(vals), freq="B")
    return pd.Series(vals, index=idx, dtype="float64")


# ---------------------------------------------------------------------------
# MSE / RMSE / MAE
# ---------------------------------------------------------------------------


def test_mse_perfect_prediction():
    y = _series([1.0, 2.0, 3.0, 4.0])
    assert mse(y, y) == pytest.approx(0.0)


def test_mse_known_value():
    y_true = _series([0.0, 0.0, 0.0])
    y_pred = _series([1.0, 2.0, 3.0])
    # MSE = (1 + 4 + 9) / 3 = 14/3
    assert mse(y_true, y_pred) == pytest.approx(14.0 / 3.0)


def test_mse_symmetric():
    a = _series([1.0, 2.0, 3.0])
    b = _series([2.0, 3.0, 4.0])
    assert mse(a, b) == pytest.approx(mse(b, a))


def test_mse_all_nan_returns_nan():
    y = _series([float("nan"), float("nan")])
    result = mse(y, y)
    assert math.isnan(result)


def test_mse_partial_nan_drops_pairs():
    y_true = _series([1.0, float("nan"), 3.0])
    y_pred = _series([1.0, 999.0, 3.0])
    # Only rows 0 and 2 are valid; MSE = 0
    assert mse(y_true, y_pred) == pytest.approx(0.0)


def test_rmse_is_sqrt_of_mse():
    a = _series([0.0, 0.0, 0.0])
    b = _series([1.0, 2.0, 3.0])
    assert rmse(a, b) == pytest.approx(math.sqrt(mse(a, b)))


def test_mae_perfect_prediction():
    y = _series([1.0, 2.0, 3.0])
    assert mae(y, y) == pytest.approx(0.0)


def test_mae_known_value():
    y_true = _series([0.0, 0.0, 0.0])
    y_pred = _series([1.0, 2.0, 3.0])
    # MAE = (1 + 2 + 3) / 3 = 2.0
    assert mae(y_true, y_pred) == pytest.approx(2.0)


def test_mae_all_nan_returns_nan():
    y = _series([float("nan")])
    assert math.isnan(mae(y, y))


# ---------------------------------------------------------------------------
# R² score
# ---------------------------------------------------------------------------


def test_r2_perfect_prediction():
    y = _series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert r2_score(y, y) == pytest.approx(1.0)


def test_r2_mean_prediction_is_zero():
    y_true = _series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = _series([3.0, 3.0, 3.0, 3.0, 3.0])  # constant = mean
    assert r2_score(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)


def test_r2_zero_variance_returns_zero():
    y_true = _series([5.0, 5.0, 5.0])
    y_pred = _series([1.0, 2.0, 3.0])
    # ss_tot = 0 → return 0.0 (not nan, not divide by zero)
    assert r2_score(y_true, y_pred) == 0.0


def test_r2_all_nan_returns_nan():
    y = _series([float("nan"), float("nan")])
    assert math.isnan(r2_score(y, y))


def test_r2_negative_for_terrible_prediction():
    y_true = _series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = _series([5.0, 4.0, 3.0, 2.0, 1.0])  # anti-correlated
    assert r2_score(y_true, y_pred) < 0.0


# ---------------------------------------------------------------------------
# Correlation coefficient
# ---------------------------------------------------------------------------


def test_correlation_perfect_positive():
    y = _series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert correlation_coefficient(y, y) == pytest.approx(1.0)


def test_correlation_perfect_negative():
    y_true = _series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = _series([5.0, 4.0, 3.0, 2.0, 1.0])
    assert correlation_coefficient(y_true, y_pred) == pytest.approx(-1.0)


def test_correlation_one_pair_returns_nan():
    y = _series([1.0])
    assert math.isnan(correlation_coefficient(y, y))


def test_correlation_all_nan_returns_nan():
    y = _series([float("nan"), float("nan")])
    assert math.isnan(correlation_coefficient(y, y))


def test_correlation_between_minus_one_and_one():
    rng = np.random.default_rng(0)
    n = 50
    y_true = _series(rng.standard_normal(n))
    y_pred = _series(rng.standard_normal(n))
    c = correlation_coefficient(y_true, y_pred)
    assert -1.0 <= c <= 1.0


# ---------------------------------------------------------------------------
# Directional accuracy
# ---------------------------------------------------------------------------


def test_directional_accuracy_perfect():
    y_true = _series([0.02, -0.01, 0.03, -0.02])
    y_pred = _series([0.01, -0.03, 0.02, -0.01])  # same signs
    assert directional_accuracy(y_true, y_pred) == pytest.approx(1.0)


def test_directional_accuracy_zero():
    y_true = _series([0.02, -0.01, 0.03, -0.02])
    y_pred = _series([-0.01, 0.03, -0.02, 0.01])  # opposite signs
    assert directional_accuracy(y_true, y_pred) == pytest.approx(0.0)


def test_directional_accuracy_half():
    y_true = _series([0.01, -0.01, 0.01, -0.01])
    y_pred = _series([0.01, 0.01, -0.01, -0.01])  # 2/4 correct
    assert directional_accuracy(y_true, y_pred) == pytest.approx(0.5)


def test_directional_accuracy_excludes_zeros():
    # Rows with y_true=0 or y_pred=0 are excluded (undefined direction)
    y_true = _series([0.0, 0.01, -0.01])
    y_pred = _series([0.01, 0.01, 0.01])
    # Row 0 excluded (y_true=0). Row 1 correct. Row 2 incorrect. → 0.5
    assert directional_accuracy(y_true, y_pred) == pytest.approx(0.5)


def test_directional_accuracy_all_zeros_returns_nan():
    y_true = _series([0.0, 0.0, 0.0])
    y_pred = _series([0.01, 0.02, 0.03])
    assert math.isnan(directional_accuracy(y_true, y_pred))


def test_directional_accuracy_nan_pairs_dropped():
    y_true = _series([0.01, float("nan"), -0.01])
    y_pred = _series([0.02, 0.03, -0.02])
    # Only rows 0 and 2 valid; both correct → 1.0
    assert directional_accuracy(y_true, y_pred) == pytest.approx(1.0)


def test_directional_accuracy_all_nan_returns_nan():
    y = _series([float("nan"), float("nan")])
    assert math.isnan(directional_accuracy(y, y))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_all_metrics_deterministic():
    rng = np.random.default_rng(42)
    y_t = _series(rng.standard_normal(50))
    y_p = _series(rng.standard_normal(50))
    for fn in (mse, rmse, mae, r2_score, correlation_coefficient, directional_accuracy):
        assert fn(y_t, y_p) == fn(y_t, y_p)
