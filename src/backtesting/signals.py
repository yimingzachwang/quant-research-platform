"""Deterministic signal construction helpers.

All functions return pd.Series with values in {-1, 0, +1} or continuous
exposure weights.  Signals are NOT lagged here — the backtest engine is
responsible for applying the one-period lag that prevents look-ahead bias.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.features.volatility import rolling_volatility


def long_only_signal(condition: pd.Series) -> pd.Series:
    """Return 1 where ``condition`` is True, 0 elsewhere.

    Args:
        condition: Boolean Series (or integer-castable).
    """
    result = condition.astype(float).clip(lower=0, upper=1)
    result.name = "long_only_signal"
    return result


def signal_from_threshold(
    series: pd.Series,
    threshold: float,
    direction: str = "above",
) -> pd.Series:
    """Return a binary signal based on a fixed threshold.

    Args:
        series: Input time series.
        threshold: The cutoff value.
        direction: ``"above"`` → 1 when series > threshold;
                   ``"below"`` → 1 when series < threshold.
    """
    if direction not in {"above", "below"}:
        msg = f"direction must be 'above' or 'below', got {direction!r}"
        raise ValueError(msg)
    mask = series > threshold if direction == "above" else series < threshold
    result = mask.astype(float)
    result.name = f"threshold_signal_{direction}_{threshold}"
    return result


def crossover_signal(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """Return sign(fast - slow): +1 (fast above), -1 (fast below), 0 (equal).

    Args:
        fast: Faster-moving series (e.g. short SMA).
        slow: Slower-moving series (e.g. long SMA).
    """
    result = pd.Series(
        np.sign(fast.values - slow.values),
        index=fast.index,
        name="crossover_signal",
    )
    return result


def volatility_target_signal(
    signal: pd.Series,
    returns: pd.Series,
    target_vol: float,
    window: int,
    max_leverage: float = 1.0,
) -> pd.Series:
    """Scale a raw signal to target a given annualized realized volatility.

    The resulting weight is:

        w_t = signal_t * (target_vol / realized_vol_t)

    capped to [-max_leverage, +max_leverage].  NaN realized volatility
    (warm-up window) is propagated as NaN — the engine treats NaN as flat.

    Args:
        signal: Raw signal series (typically in {-1, 0, +1}).
        returns: Asset return series used to estimate realized vol.
        target_vol: Target annualized volatility (e.g. 0.10 for 10%).
        window: Look-back window for realized volatility.
        max_leverage: Absolute cap on the resulting weight.
    """
    realized = rolling_volatility(returns, window=window, annualize=True)
    # Avoid division by zero; where realized_vol is 0 leave weight as NaN
    realized = realized.replace(0.0, float("nan"))
    weight = signal * (target_vol / realized)
    weight = weight.clip(lower=-max_leverage, upper=max_leverage)
    weight.name = f"vol_target_{target_vol}_signal"
    return weight
