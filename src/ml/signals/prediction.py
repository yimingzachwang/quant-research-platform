"""Pure functions that translate PredictionSeries outputs into portfolio-compatible
signals and weights.

All functions are stateless.  No fitting, no orchestration, no I/O.

Dependency direction: this module depends on src.ml.contracts (PredictionSeries)
and reuses src.portfolio.ranking + src.portfolio.allocation for weight
construction.  Nothing in src.portfolio imports from src.ml.

Single-asset functions (require pd.Series predictions):
    sign_signal       — np.sign of predictions: +1, 0, -1
    threshold_signal  — binary 1/0 above a threshold

Cross-sectional / panel functions (require pd.DataFrame predictions):
    top_n_weights         — equal-weight top-N by predicted score
    long_short_weights    — long top-N, short bottom-N; market-neutral
    normalize_to_weights  — proportional positive weights summing to 1
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.contracts import PredictionSeries
from src.portfolio.allocation import equal_weight
from src.portfolio.ranking import rank_assets, select_bottom_n, select_top_n

# ---------------------------------------------------------------------------
# Single-asset signal functions  (predictions.values must be pd.Series)
# ---------------------------------------------------------------------------


def sign_signal(predictions: PredictionSeries) -> pd.Series:
    """Convert continuous predictions to a directional signal via np.sign.

    Returns +1.0 where predictions are positive, -1.0 where negative, 0.0
    where zero or NaN.  Compatible with run_backtest (single-asset) after
    conversion to a one-column DataFrame for run_portfolio_backtest.

    Args:
        predictions: PredictionSeries whose values field is a pd.Series.

    Returns:
        float64 Series with the same index as predictions.values.

    Raises:
        TypeError: If predictions.values is not a pd.Series.
    """
    _require_series(predictions, "sign_signal")
    result = np.sign(predictions.values).astype("float64")
    result.name = f"sign({predictions.model_name})"
    return result


def threshold_signal(
    predictions: PredictionSeries,
    threshold: float = 0.0,
) -> pd.Series:
    """Convert predictions to a binary long/flat signal using a fixed threshold.

    Returns 1.0 where predictions.values > threshold, 0.0 elsewhere.
    Useful for probability outputs from LogisticRegressionModel (threshold=0.5)
    or for regression outputs (threshold=0.0 is equivalent to sign_signal >= 0).

    Args:
        predictions: PredictionSeries whose values field is a pd.Series.
        threshold: Decision boundary; default 0.0.

    Returns:
        float64 Series with the same index as predictions.values.

    Raises:
        TypeError: If predictions.values is not a pd.Series.
    """
    _require_series(predictions, "threshold_signal")
    result = (predictions.values > threshold).astype("float64")
    result.name = f"threshold({predictions.model_name},t={threshold})"
    return result


# ---------------------------------------------------------------------------
# Cross-sectional / panel functions  (predictions.values must be pd.DataFrame)
# ---------------------------------------------------------------------------


def top_n_weights(predictions: PredictionSeries, n: int) -> pd.DataFrame:
    """Select top-N assets by predicted score per row and equal-weight them.

    Delegates ranking and allocation to existing portfolio utilities:
        rank_assets → select_top_n → equal_weight

    Args:
        predictions: PredictionSeries whose values field is a pd.DataFrame
            with shape (Date × Asset) containing predicted scores.
        n: Number of assets to select per date.

    Returns:
        Date × Asset weight DataFrame.  Rows sum to 1.0 where N valid assets
        exist; rows with fewer than N valid predictions weight all available
        assets equally.  Flat rows (no valid predictions) are all zeros.

    Raises:
        TypeError: If predictions.values is not a pd.DataFrame.
        ValueError: If n < 1.
    """
    _require_dataframe(predictions, "top_n_weights")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    ranks = rank_assets(predictions.values, ascending=False)
    mask = select_top_n(ranks, n=n)
    return equal_weight(mask)


def long_short_weights(
    predictions: PredictionSeries,
    n_long: int,
    n_short: int,
) -> pd.DataFrame:
    """Long the top-N and short the bottom-N assets; dollar-neutral.

    Long weights = +1/n_long per selected long asset.
    Short weights = -1/n_short per selected short asset.
    Net exposure = 0.0; gross exposure = 2.0.

    Delegates to rank_assets, select_top_n, select_bottom_n, and equal_weight
    from src.portfolio — no allocation logic is duplicated here.

    Args:
        predictions: PredictionSeries whose values field is a pd.DataFrame
            with shape (Date × Asset) containing predicted scores.
        n_long: Number of assets to hold long.
        n_short: Number of assets to hold short.

    Returns:
        Date × Asset weight DataFrame with positive (long) and negative
        (short) values.  Each row sums to approximately 0.0.

    Raises:
        TypeError: If predictions.values is not a pd.DataFrame.
        ValueError: If n_long < 1 or n_short < 1.
    """
    _require_dataframe(predictions, "long_short_weights")
    if n_long < 1:
        raise ValueError(f"n_long must be >= 1, got {n_long}")
    if n_short < 1:
        raise ValueError(f"n_short must be >= 1, got {n_short}")

    ranks = rank_assets(predictions.values, ascending=False)
    long_mask = select_top_n(ranks, n=n_long)
    short_mask = select_bottom_n(ranks, n=n_short)

    # equal_weight produces rows summing to 1.0 on each side
    long_w = equal_weight(long_mask)
    short_w = equal_weight(short_mask)

    return long_w - short_w


def normalize_to_weights(predictions: PredictionSeries) -> pd.DataFrame:
    """Normalize positive prediction scores proportionally to portfolio weights.

    Negative values are clipped to zero before normalization.  Rows where all
    predictions are non-positive become flat (all zeros).

    Args:
        predictions: PredictionSeries whose values field is a pd.DataFrame
            with shape (Date × Asset) containing predicted scores.

    Returns:
        Date × Asset weight DataFrame.  Each row with at least one positive
        prediction sums to 1.0; all-non-positive rows are all zeros.

    Raises:
        TypeError: If predictions.values is not a pd.DataFrame.
    """
    _require_dataframe(predictions, "normalize_to_weights")
    clipped = predictions.values.clip(lower=0.0)
    row_sum = clipped.sum(axis=1).replace(0.0, float("nan"))
    return clipped.div(row_sum, axis=0).fillna(0.0)


# ---------------------------------------------------------------------------
# Internal guards
# ---------------------------------------------------------------------------


def _require_series(predictions: PredictionSeries, fn_name: str) -> None:
    if not isinstance(predictions.values, pd.Series):
        raise TypeError(
            f"{fn_name} requires predictions.values to be pd.Series; "
            f"got {type(predictions.values).__name__}. "
            "For panel predictions use top_n_weights, long_short_weights, "
            "or normalize_to_weights."
        )


def _require_dataframe(predictions: PredictionSeries, fn_name: str) -> None:
    if not isinstance(predictions.values, pd.DataFrame):
        raise TypeError(
            f"{fn_name} requires predictions.values to be pd.DataFrame "
            f"(Date × Asset); got {type(predictions.values).__name__}. "
            "For single-asset predictions use sign_signal or threshold_signal."
        )
