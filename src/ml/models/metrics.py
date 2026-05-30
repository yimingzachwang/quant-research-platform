"""Pure evaluation metrics for ML model outputs.

All functions are pure: same inputs → same output, no side effects.
NaN pairs are dropped before computation via _align_dropna().
Functions return float("nan") when no valid pairs remain after NaN removal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean squared error after dropping NaN-aligned pairs."""
    t, p = _align_dropna(y_true, y_pred)
    if len(t) == 0:
        return float("nan")
    return float(np.mean((t - p) ** 2))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute error after dropping NaN-aligned pairs."""
    t, p = _align_dropna(y_true, y_pred)
    if len(t) == 0:
        return float("nan")
    return float(np.mean(np.abs(t - p)))


def r2_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Coefficient of determination (R²).

    Returns 0.0 when y_true has zero variance (avoids divide-by-zero).
    Returns float("nan") when no valid pairs remain.
    """
    t, p = _align_dropna(y_true, y_pred)
    if len(t) == 0:
        return float("nan")
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def correlation_coefficient(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Pearson correlation between predictions and actuals.

    Returns float("nan") when fewer than 2 valid pairs remain.
    """
    t, p = _align_dropna(y_true, y_pred)
    if len(t) < 2:
        return float("nan")
    return float(np.corrcoef(t.to_numpy(), p.to_numpy())[0, 1])


def directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Fraction of rows where sign(prediction) == sign(actual).

    Definition: sign(pred_t) == sign(actual_t) after removing rows where
    either is zero or NaN.  Zero-valued rows are excluded because the
    directional call is undefined when the true return or the prediction
    is exactly zero.

    Returns float("nan") when no valid rows remain after filtering.
    """
    t, p = _align_dropna(y_true, y_pred)
    # Remove rows where either is exactly zero (ambiguous direction)
    valid = (t != 0) & (p != 0)
    t, p = t[valid], p[valid]
    if len(t) == 0:
        return float("nan")
    return float((np.sign(t.to_numpy()) == np.sign(p.to_numpy())).mean())


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _align_dropna(
    y_true: pd.Series, y_pred: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Inner-join on index and drop rows where either is NaN."""
    df = pd.DataFrame({"t": y_true, "p": y_pred}).dropna()
    return df["t"], df["p"]
