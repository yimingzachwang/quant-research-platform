"""Minimal vectorized backtesting layer."""

from src.backtesting.engine import BacktestEngine, BacktestResult, run_backtest
from src.backtesting.metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    compute_metrics,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
)
from src.backtesting.portfolio import compute_exposure, compute_turnover, position_sizing
from src.backtesting.signals import (
    crossover_signal,
    long_only_signal,
    signal_from_threshold,
    volatility_target_signal,
)

__all__ = [
    # engine
    "run_backtest",
    "BacktestResult",
    "BacktestEngine",
    # signals
    "long_only_signal",
    "signal_from_threshold",
    "crossover_signal",
    "volatility_target_signal",
    # metrics
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "calmar_ratio",
    "hit_rate",
    "compute_metrics",
    # portfolio
    "compute_turnover",
    "compute_exposure",
    "position_sizing",
]
