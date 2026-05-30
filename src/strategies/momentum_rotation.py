"""Momentum rotation strategy.

Ranks assets by trailing price momentum, selects the top N, and equal-weights
them.  Rebalances at a fixed calendar frequency (default: month-end).

Timing is look-ahead-safe:
    - Momentum computed from prices[t-lookback] to prices[t].
    - Signal observable at rebalance date t.
    - Weight forward-filled to daily; backtest engine shifts by 1 day.
    - Position therefore enters at open of day t+1.

This module contains no I/O, no plotting, and no reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.portfolio.allocation import equal_weight, resample_weights_to_daily
from src.portfolio.panel import universe_momentum
from src.portfolio.ranking import rank_assets, select_top_n
from src.strategies.base import Strategy


@dataclass
class MomentumRotationStrategy(Strategy):
    """Equal-weight rotation into the top-N momentum ETFs.

    Parameters:
        lookback: Look-back window in trading days for momentum calculation.
            Momentum at date t = price[t] / price[t - lookback] - 1.
        top_n: Number of assets to hold in each rebalance period.
        rebalance_freq: Pandas offset alias for the rebalance calendar
            (default 'ME' = month-end).  Common alternatives: 'QE' (quarter),
            'W-FRI' (weekly on Friday).
    """

    lookback: int = 252
    top_n: int = 3
    rebalance_freq: str = "ME"

    def generate_weights(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute daily portfolio weights from price history.

        Steps:
            1. Compute trailing momentum for every asset.
            2. Sample momentum at each rebalance date.
            3. Rank assets cross-sectionally; select top N.
            4. Equal-weight the selected basket.
            5. Forward-fill periodic weights to the full daily index.

        Args:
            data: Date × Asset close price DataFrame with a DatetimeIndex.
                Assets with NaN momentum (i.e., with fewer than ``lookback``
                days of history) are unrankable and will not be selected.

        Returns:
            Date × Asset daily weight DataFrame.  Rows sum to 1 once the
            warm-up period has passed; rows during warm-up are all zeros.
        """
        # Step 1 — trailing momentum (Date × Asset, NaN during warm-up)
        momentum = universe_momentum(data, window=self.lookback)

        # Step 2 — sample at rebalance calendar dates
        periodic_momentum = momentum.resample(self.rebalance_freq).last()

        # Step 3 — cross-sectional rank; higher momentum → higher rank
        ranks = rank_assets(periodic_momentum, ascending=False)
        mask = select_top_n(ranks, n=self.top_n)

        # Step 4 — equal weight within selected basket
        periodic_weights = equal_weight(mask)

        # Step 5 — expand to daily (forward-fill from each rebalance date)
        daily_weights = resample_weights_to_daily(periodic_weights, data.index)

        return daily_weights

    @property
    def name(self) -> str:
        return (
            f"MomentumRotation"
            f"(lookback={self.lookback},top_n={self.top_n},"
            f"freq={self.rebalance_freq})"
        )

    def params(self) -> dict[str, Any]:
        return {
            "lookback": self.lookback,
            "top_n": self.top_n,
            "rebalance_freq": self.rebalance_freq,
        }
