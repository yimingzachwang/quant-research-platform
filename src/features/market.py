"""Cross-asset market sensitivity features.

These features require two return series — an asset series and a market
reference series — and must be computed within feature callables that have
access to the full price panel.
"""

from __future__ import annotations

import pandas as pd


def rolling_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Rolling OLS beta of asset returns against market returns.

    Computed as Cov(r_asset, r_market; window) / Var(r_market; window).
    A beta of 1.0 means the asset moves in lockstep with the market.
    Beta > 1 indicates amplified market sensitivity; beta < 1 indicates
    defensive positioning relative to the reference index.

    In a cross-sectional ranking framework, rolling beta captures the
    time-varying *systematic exposure* of each asset — distinguishing assets
    that are currently high-beta (momentum amplifiers in bull markets) from
    low-beta / defensive assets that are relatively insensitive to market moves.

    Args:
        asset_returns: Period return series for the asset.
        market_returns: Period return series for the market reference.
        window: Look-back window in trading days.
    """
    cov = asset_returns.rolling(window).cov(market_returns)
    var = market_returns.rolling(window).var()
    denom = var.replace(0, float("nan"))
    result = cov / denom
    result.name = f"beta_{window}d"
    return result
