"""Vectorized backtest engine.

Core function: run_backtest(returns, signal, ...) -> pd.DataFrame

Timing convention (critical — prevents look-ahead bias):
    signal computed at close of day t
    →  position held from close of day t to close of day t+1
    →  return realized at close of day t+1

Implementation: position = signal.shift(1)
The first period always has no position (flat/cash).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.core import ExperimentContext


def run_backtest(
    returns: pd.Series,
    signal: pd.Series,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Run a fully vectorized single-asset backtest.

    Args:
        returns: Asset period returns aligned to a DatetimeIndex.
        signal: Signal series at the same frequency.  Values may be
            discrete {-1, 0, +1} or continuous exposure weights.
            NaN signal → flat (0 position) for that period.
        transaction_cost_bps: One-way transaction cost in basis points
            applied to every unit of absolute position change.

    Returns:
        DataFrame with one row per period, columns:

        position          — signal lagged by 1 period (look-ahead-safe)
        gross_return      — position * asset_return
        turnover          — |Δposition| per period
        transaction_cost  — turnover * cost_bps / 10_000
        net_return        — gross_return - transaction_cost
        equity_curve      — cumulative (1 + net_return), starting at 1.0
        drawdown          — (equity_curve - peak) / peak  (≤ 0)

    Notes:
        - Index is the intersection of ``returns`` and ``signal`` indices,
          sorted ascending.
        - The first row always has position=0 (no prior signal), so
          gross_return and net_return are 0.0 on that row.
    """
    # Align on common index
    common = returns.index.intersection(signal.index).sort_values()
    r = returns.loc[common]
    s = signal.loc[common]

    # Lag signal by one period — this is the critical look-ahead guard
    position = s.shift(1).fillna(0.0)

    gross_return = position * r

    # Turnover: absolute position change per period
    turnover = position.diff().abs().fillna(0.0)

    cost = turnover * (transaction_cost_bps / 10_000)

    net_return = gross_return - cost

    equity_curve = (1.0 + net_return).cumprod()

    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak

    result = pd.DataFrame(
        {
            "position": position,
            "gross_return": gross_return,
            "turnover": turnover,
            "transaction_cost": cost,
            "net_return": net_return,
            "equity_curve": equity_curve,
            "drawdown": drawdown,
        },
        index=common,
    )
    return result


# ---------------------------------------------------------------------------
# Legacy skeleton — preserved for ExperimentContext wiring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestResult:
    """Structured output from a historical simulation."""

    context: ExperimentContext
    artifacts: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


class BacktestEngine:
    """Thin orchestration wrapper around run_backtest.

    Intended for future wiring with ExperimentContext, feature pipelines,
    and signal generators.  Direct use of run_backtest() is preferred for
    interactive research.
    """

    def run(self, context: ExperimentContext) -> BacktestResult:
        """Run a backtest for an experiment context.

        TODO: Wire data loading, feature generation, signal generation,
        and run_backtest() for each symbol in context.universe.
        """
        return BacktestResult(context=context)
