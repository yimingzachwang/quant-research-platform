"""Walk-forward validation runner.

Evaluates a strategy on each test window of a set of time splits without
any look-ahead leakage.

Leakage prevention contract
----------------------------
For each split the strategy receives prices up to and including
``split.test_end`` — this is the maximum amount of data consistent with
when the evaluation period ends.  Critically, *no prices beyond test_end*
are ever passed, so future returns cannot influence weight generation.

The ``run_portfolio_backtest`` function (and therefore ``run_strategy``)
applies ``weights.shift(1)`` before multiplying returns, so the position
entered on day ``t`` only uses information available at close of day ``t-1``.

Future ML compatibility
-----------------------
Strategies may optionally implement a ``fit(train_data)`` method.  If
present, it is called on the training slice before ``generate_weights`` is
called on the full window.  Traditional strategies (no ``fit``) work without
any changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.backtesting.metrics import compute_metrics
from src.strategies.base import Strategy
from src.strategies.runner import run_strategy
from src.validation.splits import TimeSplit

# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------


@dataclass
class SplitResult:
    """Performance on one test window.

    Attributes:
        split: The TimeSplit that defines the train/test dates.
        strategy_name: Canonical name of the strategy used.
        metrics: Standard metrics computed over the test period.
        equity_curve: Cumulative return starting at 1.0 over test period.
        weights: Daily weights during test period.
    """

    split: TimeSplit
    strategy_name: str
    metrics: dict[str, float]
    equity_curve: pd.Series
    weights: pd.DataFrame


@dataclass
class WalkForwardResult:
    """Aggregated result of a full walk-forward validation run.

    Attributes:
        strategy_name: Canonical name of the strategy.
        splits: Ordered list of per-split results.
    """

    strategy_name: str
    splits: list[SplitResult] = field(default_factory=list)

    @property
    def n_splits(self) -> int:
        return len(self.splits)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_walk_forward_validation(
    prices: pd.DataFrame,
    strategy: Strategy,
    splits: list[TimeSplit],
    transaction_cost_bps: float = 0.0,
) -> WalkForwardResult:
    """Run walk-forward validation of a strategy over a set of time splits.

    For each split:
    1. Optionally calls ``strategy.fit(train_prices)`` if the method exists
       (for future ML strategies).
    2. Runs the strategy on ``prices[:test_end]`` to generate weights.
    3. Extracts the test-period slice of the backtest.
    4. Computes metrics solely on the test window.

    No data from after ``split.test_end`` is ever visible to the strategy.

    Args:
        prices: Date × Asset price DataFrame.
        strategy: Strategy instance implementing ``generate_weights()``.
        splits: Time splits from ``rolling_time_splits`` or
            ``expanding_time_splits``.
        transaction_cost_bps: One-way transaction cost in basis points.

    Returns:
        WalkForwardResult with per-split metrics, equity curves, and weights.
    """
    split_results: list[SplitResult] = []

    for split in splits:
        # Strict no-leakage: strategy sees only data up to test_end
        prices_window = prices.loc[:split.test_end]

        # Optional fit hook for future ML strategies
        if hasattr(strategy, "fit"):
            train_prices = prices.loc[split.train_start : split.train_end]
            strategy.fit(train_prices)  # type: ignore[attr-defined]

        result = run_strategy(
            prices_window, strategy, transaction_cost_bps=transaction_cost_bps
        )

        # Extract test-period slice only
        test_backtest = result.backtest.loc[split.test_start : split.test_end]
        test_weights = result.weights.loc[split.test_start : split.test_end]

        net_returns = test_backtest["net_return"]

        # Equity curve anchored at 1.0 for the test period
        equity_curve = (1.0 + net_returns).cumprod()
        equity_curve.name = "equity"

        test_metrics = compute_metrics(net_returns)

        split_results.append(
            SplitResult(
                split=split,
                strategy_name=strategy.name,
                metrics=test_metrics,
                equity_curve=equity_curve,
                weights=test_weights,
            )
        )

    return WalkForwardResult(
        strategy_name=strategy.name,
        splits=split_results,
    )
