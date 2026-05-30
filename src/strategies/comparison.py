"""Strategy comparison utilities.

compare_strategies()  — run a list of strategies, return a keyed result dict.
metrics_table()       — flatten result metrics into a tidy comparison DataFrame.
rank_strategies()     — rank strategies by a chosen metric.

These are pure research utilities: no I/O, no plotting, no side effects.
The caller decides what to do with the outputs.
"""

from __future__ import annotations

import pandas as pd

from src.strategies.base import Strategy
from src.strategies.runner import StrategyResult, run_strategy


def compare_strategies(
    prices: pd.DataFrame,
    strategies: list[Strategy],
    transaction_cost_bps: float = 0.0,
) -> dict[str, StrategyResult]:
    """Run multiple strategies on the same price history.

    Executes each strategy sequentially and returns results keyed by
    strategy name.  No parallelism, no caching.

    Args:
        prices: Date × Asset close price DataFrame (shared universe).
        strategies: List of Strategy instances to evaluate.
        transaction_cost_bps: One-way cost in basis points, applied uniformly
            to all strategies.

    Returns:
        Dict mapping strategy name → StrategyResult.  Insertion order
        matches the order of ``strategies``.

    Raises:
        ValueError: If two strategies share the same name (ambiguous keys).
    """
    seen: set[str] = set()
    for s in strategies:
        if s.name in seen:
            msg = (
                f"Duplicate strategy name {s.name!r}.  "
                "Rename one instance before comparing."
            )
            raise ValueError(msg)
        seen.add(s.name)

    return {
        s.name: run_strategy(prices, s, transaction_cost_bps=transaction_cost_bps)
        for s in strategies
    }


def metrics_table(results: dict[str, StrategyResult]) -> pd.DataFrame:
    """Build a tidy metrics comparison DataFrame.

    Args:
        results: Output of compare_strategies() — or any dict mapping
            strategy name → StrategyResult.

    Returns:
        DataFrame with one row per strategy and one column per metric.
        Columns: annualized_return, annualized_volatility, sharpe_ratio,
        max_drawdown, calmar_ratio, hit_rate.
        Index: strategy names.
    """
    rows = {name: r.metrics for name, r in results.items()}
    df = pd.DataFrame(rows).T
    df.index.name = "strategy"
    return df


def rank_strategies(
    results: dict[str, StrategyResult],
    by: str = "sharpe_ratio",
    ascending: bool = False,
) -> pd.DataFrame:
    """Return the metrics table sorted by a chosen metric.

    Args:
        results: Output of compare_strategies().
        by: Column name to sort by.  Must be a key in strategy metrics dicts.
        ascending: Sort direction.  Default False (highest value first).

    Returns:
        Sorted metrics DataFrame with an added 'rank' column.
    """
    table = metrics_table(results)
    if by not in table.columns:
        msg = f"Metric {by!r} not found.  Available: {list(table.columns)}"
        raise ValueError(msg)
    table = table.sort_values(by, ascending=ascending)
    table.insert(0, "rank", range(1, len(table) + 1))
    return table
