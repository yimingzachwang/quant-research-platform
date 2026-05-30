"""Strategy runner: executes a strategy and returns structured results.

Single entry point: run_strategy(prices, strategy, cost_bps) → StrategyResult

The runner is responsible for:
    1. Computing returns from prices.
    2. Calling strategy.generate_weights(prices).
    3. Running the portfolio backtest (which applies weights.shift(1)).
    4. Packaging results into StrategyResult.

It is NOT responsible for:
    - File I/O
    - Plotting
    - Logging
    - Experiment metadata
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.portfolio.panel import universe_returns
from src.portfolio.portfolio_backtest import run_portfolio_backtest
from src.strategies.base import Strategy


@dataclass
class StrategyResult:
    """Structured output from run_strategy().

    Attributes:
        strategy_name: Human-readable strategy identifier.
        weights:       Date × Asset weight DataFrame (lagged — as applied).
        backtest:      Daily time-series with gross_return, turnover,
                       transaction_cost, net_return, equity_curve, drawdown.
        metrics:       Scalar performance metrics dict.
    """

    strategy_name: str
    weights: pd.DataFrame
    backtest: pd.DataFrame
    metrics: dict[str, float]


def run_strategy(
    prices: pd.DataFrame,
    strategy: Strategy,
    transaction_cost_bps: float = 0.0,
) -> StrategyResult:
    """Execute a strategy on a price matrix and return structured results.

    Args:
        prices: Date × Asset close price DataFrame with a DatetimeIndex.
            No NaN values expected (caller cleans first if needed).
        strategy: Any Strategy subclass with a generate_weights() method.
        transaction_cost_bps: One-way transaction cost in basis points,
            applied to each unit of absolute weight change per period.

    Returns:
        StrategyResult containing the lagged applied weights, full backtest
        time-series DataFrame, and scalar performance metrics.

    Notes:
        - Look-ahead prevention is handled inside run_portfolio_backtest
          via weights.shift(1).  The strategy generates signals at time t;
          the position is entered at time t+1.
        - The first row of backtest always has zero return (flat before the
          first signal propagates through the lag).
    """
    weights = strategy.generate_weights(prices)
    returns = universe_returns(prices)

    result = run_portfolio_backtest(
        returns=returns,
        weights=weights,
        transaction_cost_bps=transaction_cost_bps,
    )

    return StrategyResult(
        strategy_name=strategy.name,
        weights=result.weights,
        backtest=result.backtest,
        metrics=result.metrics,
    )
