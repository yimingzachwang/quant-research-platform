"""Pure functions for evaluating prediction quality.

All functions are stateless — no fitting, no I/O, no side effects.
NaN values are dropped before computation; index alignment is done
explicitly before any metric is computed.

Reuses:
    src.ml.models.metrics.correlation_coefficient — for prediction_correlation
    scipy.stats.spearmanr                          — for IC (research extra)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.models.metrics import correlation_coefficient


def prediction_correlation(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    """Pearson correlation between actual and predicted values.

    Inner-joins on index and drops NaN pairs before computation.
    Returns float("nan") when fewer than 2 valid pairs remain.

    Delegates to src.ml.models.metrics.correlation_coefficient so the
    definition stays consistent with the rest of the ML evaluation layer.

    Args:
        actual:    Observed target values (pd.Series, DatetimeIndex).
        predicted: Model predictions aligned to the same index.

    Returns:
        Pearson correlation coefficient in [-1, 1], or float("nan").
    """
    return correlation_coefficient(actual, predicted)


def information_coefficient(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    min_observations: int = 5,
) -> pd.Series:
    """Cross-sectional Spearman rank IC per timestamp.

    For each date (row) computes the Spearman rank correlation between
    actual cross-sectional returns and predicted cross-sectional scores.
    Dates with fewer than ``min_observations`` valid pairs are returned as NaN.

    Alignment: the two DataFrames are inner-joined on both the index
    (dates) and columns (assets) before computation.

    Args:
        actual:           Date × Asset DataFrame of realised returns.
        predicted:        Date × Asset DataFrame of model predictions.
        min_observations: Minimum non-NaN assets per date required to
                          compute IC.  Default 5.

    Returns:
        Date-indexed pd.Series of Spearman IC values in [-1, 1].
        NaN on dates with insufficient observations.
    """
    from scipy.stats import spearmanr  # research extra

    # Align on common dates and common assets
    common_idx = actual.index.intersection(predicted.index)
    common_cols = actual.columns.intersection(predicted.columns)
    act = actual.loc[common_idx, common_cols]
    pred = predicted.loc[common_idx, common_cols]

    ic_values: dict[pd.Timestamp, float] = {}
    for date in common_idx:
        a_row = act.loc[date]
        p_row = pred.loc[date]
        # Keep only rows where both are non-NaN
        valid = a_row.notna() & p_row.notna()
        if valid.sum() < min_observations:
            ic_values[date] = float("nan")
            continue
        corr, _ = spearmanr(p_row[valid].to_numpy(), a_row[valid].to_numpy())
        ic_values[date] = float(corr)

    result = pd.Series(ic_values, name="IC")
    result.index.name = actual.index.name or "date"
    return result


def rolling_directional_accuracy(
    actual: pd.Series,
    predicted: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling hit-rate: fraction of periods where sign(pred) == sign(actual).

    Inner-joins on index before computation.  Rows where either value is
    zero or NaN are excluded from the hit-rate denominator (undefined direction).

    Args:
        actual:    Observed return series.
        predicted: Prediction series aligned to the same index.
        window:    Rolling window length in periods.

    Returns:
        pd.Series of rolling directional accuracy in [0, 1].
        First (window - 1) values are NaN (insufficient history).

    Raises:
        ValueError: If window < 1.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    # Align and drop NaN pairs
    df = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()

    # Directional hit: 1.0 when signs match and neither is zero
    valid = (df["actual"] != 0) & (df["predicted"] != 0)
    hits = pd.Series(np.nan, index=df.index, dtype="float64")
    hits[valid] = (
        np.sign(df.loc[valid, "actual"].to_numpy())
        == np.sign(df.loc[valid, "predicted"].to_numpy())
    ).astype("float64")

    # Rolling mean — NaN rows contribute NaN (min_periods enforces minimum)
    result = hits.rolling(window, min_periods=window).mean()
    result.name = f"rolling_da_{window}d"
    return result


def prediction_quantiles(
    predictions: pd.Series,
    n_quantiles: int = 10,
) -> pd.Series:
    """Assign each prediction to a quantile bin (1 = lowest, n = highest).

    Useful for decile analysis: compare average realised returns across
    quantile bins to evaluate whether higher predictions correspond to
    higher actual returns.

    NaN predictions are assigned NaN quantile labels.

    Args:
        predictions: Series of model predictions.
        n_quantiles: Number of equal-frequency bins.  Default 10 (deciles).

    Returns:
        Integer-labelled pd.Series (1 through n_quantiles) with the same
        index as ``predictions``.  Ties are resolved by assigning to the
        lower quantile.

    Raises:
        ValueError: If n_quantiles < 2.
    """
    if n_quantiles < 2:
        raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")

    labels = list(range(1, n_quantiles + 1))
    # duplicates="drop" handles flat predictions without raising
    result = pd.qcut(predictions, q=n_quantiles, labels=labels, duplicates="drop")
    return result.astype("Int64").rename("quantile")
