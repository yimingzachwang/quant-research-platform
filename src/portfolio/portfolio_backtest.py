"""Vectorized portfolio backtest engine.

Timing convention (identical to single-asset engine — prevents look-ahead):
    weights computed at close of day t
    →  positions held from close of day t to close of day t+1
    →  returns realized at close of day t+1

Implementation: applied_weights = weights.shift(1)

The result is a lightweight dataclass carrying:
    - backtest  : daily time-series DataFrame
    - weights   : the lagged weight matrix that was actually applied
    - metrics   : dict of scalar performance metrics
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtesting.metrics import compute_metrics


@dataclass
class PortfolioBacktestResult:
    """Output of run_portfolio_backtest().

    Attributes:
        backtest:  Daily DataFrame with columns gross_return, turnover,
                   transaction_cost, net_return, equity_curve, drawdown.
        weights:   Date × Asset DataFrame of the lagged weights actually used.
        metrics:   Scalar performance metrics computed on net_return.
    """

    backtest: pd.DataFrame
    weights: pd.DataFrame
    metrics: dict[str, float]


def run_portfolio_backtest(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    transaction_cost_bps: float = 0.0,
) -> PortfolioBacktestResult:
    """Run a vectorized multi-asset portfolio backtest.

    Args:
        returns: Date × Asset return DataFrame.
        weights: Date × Asset weight DataFrame.  Weights need not sum to
            exactly 1; gross exposure is determined by the actual row sums.
            NaN weights are treated as 0 (flat in that asset).
        transaction_cost_bps: One-way transaction cost in basis points,
            applied to each unit of absolute portfolio weight change.

    Returns:
        PortfolioBacktestResult with backtest DataFrame, lagged weights, and
        scalar metrics.

    Notes:
        - Index is the intersection of returns and weights indices, sorted.
        - The first row always has zero weights (look-ahead prevention).
        - Turnover is summed across all assets: Σ |Δw_i| per period.
    """
    # Align on common index
    common = returns.index.intersection(weights.index).sort_values()

    # Align columns: use assets present in both DataFrames
    assets = returns.columns.intersection(weights.columns)
    r = returns.loc[common, assets].fillna(0.0)
    w = weights.loc[common, assets].fillna(0.0)

    # Lag weights by one period — critical look-ahead prevention
    w_lagged = w.shift(1).fillna(0.0)

    # Portfolio gross return: weighted sum of asset returns
    gross_return = (w_lagged * r).sum(axis=1)

    # Turnover: total absolute weight change across all assets
    weight_change = w_lagged.diff().abs()
    weight_change.iloc[0] = w_lagged.iloc[0].abs()  # first row: from 0 to initial weight
    turnover = weight_change.sum(axis=1)

    transaction_cost = turnover * (transaction_cost_bps / 10_000)

    net_return = gross_return - transaction_cost

    equity_curve = (1.0 + net_return).cumprod()
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak

    backtest = pd.DataFrame(
        {
            "gross_return": gross_return,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "net_return": net_return,
            "equity_curve": equity_curve,
            "drawdown": drawdown,
        },
        index=common,
    )

    metrics = compute_metrics(net_return)

    return PortfolioBacktestResult(
        backtest=backtest,
        weights=w_lagged,
        metrics=metrics,
    )
