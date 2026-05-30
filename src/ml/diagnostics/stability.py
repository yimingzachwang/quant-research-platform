"""Pure functions for analysing ML model stability across walk-forward splits.

All functions are stateless — no fitting, no I/O, no side effects.

split_metric_table delegates to src.validation.stability.split_metrics_table
to avoid duplicating that logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.stability import split_metrics_table
from src.validation.walk_forward import WalkForwardResult


def split_metric_table(wf_result: WalkForwardResult) -> pd.DataFrame:
    """Convert a WalkForwardResult into a split-by-split metric DataFrame.

    Thin delegation to src.validation.stability.split_metrics_table — kept
    here so that ML diagnostic workflows have a single import location for
    all diagnostic utilities.

    Returns:
        DataFrame indexed by split_index with columns: train_start,
        train_end, test_start, test_end, and all metric keys
        (annualized_return, sharpe_ratio, max_drawdown, etc.).
        Empty DataFrame if wf_result has no splits.
    """
    return split_metrics_table(wf_result)


def coefficient_stability(
    coefficients: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise coefficient stability across walk-forward splits.

    Input is a Date/Split × Feature DataFrame where each row is a set of
    model coefficients fitted on one training window.  This is the natural
    output of collecting linear model weights across walk-forward splits.

    Output statistics per feature:
        mean            — average coefficient across splits
        std             — standard deviation across splits
        sign_consistency — fraction of splits where sign matches sign(mean)
        min             — minimum coefficient seen
        max             — maximum coefficient seen

    Args:
        coefficients: DataFrame of shape (n_splits, n_features).  Index is
                      typically the split index or training end date.
                      Each cell is the fitted coefficient for one feature
                      in one split.

    Returns:
        DataFrame indexed by feature name with columns:
        mean, std, sign_consistency, min, max.
        Empty if input is empty.

    Raises:
        ValueError: If coefficients is not a 2-D DataFrame.
    """
    if coefficients.empty:
        return pd.DataFrame(
            columns=["mean", "std", "sign_consistency", "min", "max"]
        )

    stats: dict[str, dict[str, float]] = {}
    for col in coefficients.columns:
        vals = coefficients[col].dropna()
        if len(vals) == 0:
            stats[col] = {
                "mean": float("nan"),
                "std": float("nan"),
                "sign_consistency": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
            continue

        mean_val = float(vals.mean())
        std_val = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
        # Sign consistency: fraction of splits where sign matches sign(mean)
        # If mean is 0, sign is undefined — fall back to NaN
        if mean_val == 0.0:
            sign_consistency = float("nan")
        else:
            dominant_sign = np.sign(mean_val)
            sign_consistency = float((np.sign(vals) == dominant_sign).mean())

        stats[col] = {
            "mean": mean_val,
            "std": std_val,
            "sign_consistency": sign_consistency,
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    result = pd.DataFrame(stats).T
    result.index.name = "feature"
    return result


def prediction_drift(
    predictions: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling mean of raw predictions.

    Captures whether the model is systematically biased long or short
    over time, and whether that bias shifts — a symptom of regime change
    or model instability.

    Distinct from rolling_directional_accuracy (which measures sign hits).
    This measures the level and direction of prediction magnitude.

    Args:
        predictions: Time-indexed series of model predictions.
        window:      Rolling window in periods.

    Returns:
        pd.Series of rolling mean predictions, same index as input.
        First (window - 1) values are NaN.

    Raises:
        ValueError: If window < 1.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    result = predictions.rolling(window, min_periods=window).mean()
    result.name = f"prediction_drift_{window}d"
    return result
