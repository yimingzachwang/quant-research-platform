"""Configurable weighting policies for selected asset baskets.

Pure functions — no fitting, no I/O, no hidden state.

Each policy accepts raw prediction scores and a boolean selection mask and
returns a Date × Asset weight DataFrame.  All operations are strictly
row-wise (timestamp-local) to preserve walk-forward chronology integrity.

Public API
----------
apply_weighting_policy(scores, mask, scheme, prediction_normalization, temperature)
    Route to the correct weighting scheme.  Delegates "equal_weight" to
    src.portfolio.allocation.equal_weight — no logic is duplicated.

Supported schemes
-----------------
equal_weight        : 1/N per selected asset (default; backward-compatible baseline)
zscore_softmax      : softmax over row-wise z-scored prediction scores within selection
confidence_weighted : proportional to clipped-positive raw prediction scores
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio.allocation import equal_weight

VALID_WEIGHTING_SCHEMES: frozenset[str] = frozenset({
    "equal_weight",
    "zscore_softmax",
    "confidence_weighted",
})

VALID_PREDICTION_NORMALIZATIONS: frozenset[str] = frozenset({"none", "zscore"})


def apply_weighting_policy(
    scores: pd.DataFrame,
    mask: pd.DataFrame,
    scheme: str = "equal_weight",
    prediction_normalization: str = "none",
    temperature: float | None = None,
) -> pd.DataFrame:
    """Apply a named weighting policy to a selected asset basket.

    All schemes operate strictly row-wise — no information crosses dates
    and no future data is accessed.  Walk-forward chronology is preserved.

    Args:
        scores: Date × Asset raw prediction score DataFrame.
        mask: Date × Asset boolean selection mask (True = selected).
        scheme: Weighting scheme name.
            "equal_weight"        — 1/N for selected assets (default)
            "zscore_softmax"      — softmax over row-wise z-scored scores within mask
            "confidence_weighted" — proportional to clipped-positive scores within mask
        prediction_normalization: Row-wise pre-processing before weighting.
            "none"   — use raw scores (default)
            "zscore" — row-wise z-score normalization before weighting
        temperature: Softmax temperature (>0).  Higher values → more uniform weights.
            Relevant for zscore_softmax; ignored for other schemes.

    Returns:
        Date × Asset weight DataFrame.  Rows with selected assets sum to 1.0;
        rows with no selected assets sum to 0.0.

    Raises:
        ValueError: If scheme or prediction_normalization is unrecognized.
    """
    if scheme not in VALID_WEIGHTING_SCHEMES:
        raise ValueError(
            f"Unknown weighting scheme {scheme!r}. "
            f"Available: {sorted(VALID_WEIGHTING_SCHEMES)}"
        )

    if scheme == "equal_weight":
        return equal_weight(mask)

    processed = _apply_prediction_normalization(scores, prediction_normalization)

    if scheme == "zscore_softmax":
        temp = float(temperature) if temperature is not None else 1.0
        return _zscore_softmax(processed, mask, temperature=temp)

    if scheme == "confidence_weighted":
        return _confidence_weighted(processed, mask)

    raise ValueError(f"Unhandled scheme {scheme!r}.")


# ---------------------------------------------------------------------------
# Internal implementations
# ---------------------------------------------------------------------------


def _apply_prediction_normalization(
    scores: pd.DataFrame,
    normalization: str,
) -> pd.DataFrame:
    """Row-wise prediction score normalization.

    Args:
        scores: Date × Asset score DataFrame.
        normalization: "none" (pass-through) or "zscore" (row-wise standardize).

    Returns:
        Normalized score DataFrame with the same shape and index.

    Raises:
        ValueError: If normalization method is unrecognized.
    """
    if normalization == "none" or not normalization:
        return scores
    if normalization == "zscore":
        row_mean = scores.mean(axis=1)
        row_std = scores.std(axis=1).replace(0.0, float("nan"))
        return scores.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0.0)
    raise ValueError(
        f"Unknown prediction_normalization {normalization!r}. "
        f"Available: {sorted(VALID_PREDICTION_NORMALIZATIONS)}"
    )


def _zscore_softmax(
    scores: pd.DataFrame,
    mask: pd.DataFrame,
    temperature: float = 1.0,
) -> pd.DataFrame:
    """Softmax weighting within the selection mask using scaled scores.

    Non-selected assets are excluded from softmax mass (probability = 0).
    Row-max subtraction ensures numerical stability across arbitrary score scales.

    Row-wise invariant: only cross-sectional information from the current date
    is used — no cross-date or cross-fold normalization.

    Args:
        scores: Date × Asset score DataFrame (may already be z-scored by caller).
        mask: Date × Asset boolean selection mask.
        temperature: Softmax temperature (must be > 0).  T → 0 concentrates all
            weight on the top asset; T → ∞ approaches equal weighting.

    Returns:
        Date × Asset weight DataFrame.  Rows sum to 1.0 for rows with selections.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    bool_mask = mask.astype(bool)
    masked_scores = scores.where(bool_mask, other=float("-inf"))
    scaled = masked_scores / temperature

    # Row-max subtraction for numerical stability (ignores -inf entries)
    row_max = scaled.replace(float("-inf"), float("nan")).max(axis=1).fillna(0.0)
    stable = scaled.sub(row_max, axis=0)

    exp_scores = np.exp(stable).where(bool_mask, other=0.0)
    row_sum = exp_scores.sum(axis=1).replace(0.0, float("nan"))
    return exp_scores.div(row_sum, axis=0).fillna(0.0)


def _confidence_weighted(
    scores: pd.DataFrame,
    mask: pd.DataFrame,
) -> pd.DataFrame:
    """Proportional weighting by clipped-positive prediction scores.

    Weights are proportional to max(score, 0) within the selected basket.
    Rows where all selected assets have non-positive scores fall back to
    equal_weight for that row (prevents silent zero-weight allocations).

    Row-wise invariant: only cross-sectional information from the current date
    is used — no cross-date or cross-fold normalization.

    Args:
        scores: Date × Asset score DataFrame.
        mask: Date × Asset boolean selection mask.

    Returns:
        Date × Asset weight DataFrame.  Rows with positive scores sum to 1.0.
    """
    bool_mask = mask.astype(bool)
    masked = scores.where(bool_mask, other=0.0).clip(lower=0.0)
    row_sum = masked.sum(axis=1)
    has_signal = row_sum > 0.0

    weights = masked.div(row_sum.replace(0.0, float("nan")), axis=0).fillna(0.0)

    # Equal-weight fallback for rows with no positive prediction within selection
    if (~has_signal).any():
        fallback = equal_weight(mask)
        weights.loc[~has_signal] = fallback.loc[~has_signal]

    return weights
