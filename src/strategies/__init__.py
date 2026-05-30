"""Strategy research layer."""

from src.strategies.base import Strategy
from src.strategies.baselines import BuyAndHoldStrategy, EqualWeightStrategy
from src.strategies.comparison import compare_strategies, metrics_table, rank_strategies
from src.strategies.ml_strategy import MLStrategy
from src.strategies.momentum_rotation import MomentumRotationStrategy
from src.strategies.runner import StrategyResult, run_strategy

# Legacy placeholder — preserved for compatibility
from src.strategies.placeholder import StrategySpec

__all__ = [
    # base
    "Strategy",
    # implementations
    "BuyAndHoldStrategy",
    "EqualWeightStrategy",
    "MomentumRotationStrategy",
    "MLStrategy",
    # runner
    "StrategyResult",
    "run_strategy",
    # comparison
    "compare_strategies",
    "metrics_table",
    "rank_strategies",
    # legacy
    "StrategySpec",
]
