"""Baseline strategies for benchmarking.

These are the minimum hurdles every more complex strategy must beat.
They are intentionally simple, transparent, and have no look-ahead.

BuyAndHoldStrategy  — static allocation, never rebalances.
EqualWeightStrategy — equal weight across all assets, periodic rebalance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.portfolio.allocation import resample_weights_to_daily
from src.strategies.base import Strategy


@dataclass
class BuyAndHoldStrategy(Strategy):
    """Static portfolio allocation that never rebalances.

    Parameters:
        weights: Optional dict mapping asset name → weight (e.g. {"SPY": 0.6,
            "TLT": 0.4}).  Weights need not sum to 1; the values are used
            as-is.  Assets not present in the dict receive weight 0.
            If None (default), 100% is allocated to the first asset in the
            price DataFrame.
    """

    weights: dict[str, float] | None = None

    def generate_weights(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.weights is None:
            # 100% first asset, 0 for all others
            row = {col: (1.0 if i == 0 else 0.0) for i, col in enumerate(data.columns)}
        else:
            row = {col: float(self.weights.get(col, 0.0)) for col in data.columns}

        # Broadcast the constant weight vector across all dates
        return pd.DataFrame(row, index=data.index)

    @property
    def name(self) -> str:
        if self.weights is None:
            return "BuyAndHold(first_asset)"
        label = "/".join(f"{k}={v:.0%}" for k, v in self.weights.items())
        return f"BuyAndHold({label})"

    def params(self) -> dict[str, Any]:
        return {"weights": self.weights}


@dataclass
class EqualWeightStrategy(Strategy):
    """Equal-weight portfolio with periodic rebalancing.

    At each rebalance date every asset receives weight 1/N, where N is the
    number of columns in the price DataFrame.  Between rebalance dates the
    weights are forward-filled (held constant).  The first rebalance is
    synthetic on the first trading day so the portfolio is invested from
    day 1.

    Parameters:
        rebalance_freq: Pandas offset alias for the rebalance calendar.
            Default 'ME' (month-end).  Use 'QE' for quarterly, 'W-FRI' for
            weekly Friday rebalances.
    """

    rebalance_freq: str = "ME"

    def generate_weights(self, data: pd.DataFrame) -> pd.DataFrame:
        n = len(data.columns)
        equal = 1.0 / n

        # Rebalance dates from the calendar — last day of each period
        periodic_idx = data.resample(self.rebalance_freq).last().index

        # Include the first trading day so the portfolio is invested immediately
        all_dates = periodic_idx.union(data.index[:1]).sort_values()

        periodic_weights = pd.DataFrame(equal, index=all_dates, columns=data.columns)

        return resample_weights_to_daily(periodic_weights, data.index)

    @property
    def name(self) -> str:
        return f"EqualWeight(freq={self.rebalance_freq})"

    def params(self) -> dict[str, Any]:
        return {"rebalance_freq": self.rebalance_freq}
