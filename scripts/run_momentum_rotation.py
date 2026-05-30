"""Run the MomentumRotationStrategy on a 7-ETF universe.

Pipeline:
    load prices → strategy.generate_weights() → run_strategy() → save_experiment()

Outputs saved to:
    results/experiments/momentum_rotation_<date>/
        metadata.json
        metrics.json
        equity_curve.parquet
        weights.parquet

Usage:
    python -m scripts.run_momentum_rotation
    python -m scripts.run_momentum_rotation --lookback 252 --top-n 3 --cost-bps 5
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.experiments.results import ExperimentResult, save_experiment
from src.portfolio.alignment import align_prices, load_universe
from src.strategies.momentum_rotation import MomentumRotationStrategy
from src.strategies.runner import run_strategy

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

SYMBOLS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD"]
DEFAULT_LOOKBACK = 252
DEFAULT_TOP_N = 3
DEFAULT_REBALANCE_FREQ = "ME"
DEFAULT_COST_BPS = 5.0
OUTPUT_BASE = Path("results/experiments")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(
    lookback: int = DEFAULT_LOOKBACK,
    top_n: int = DEFAULT_TOP_N,
    rebalance_freq: str = DEFAULT_REBALANCE_FREQ,
    cost_bps: float = DEFAULT_COST_BPS,
) -> None:
    experiment_name = f"momentum_rotation_{date.today().isoformat()}"

    # 1. Load and align prices
    print(f"Loading universe: {SYMBOLS}")
    universe = load_universe(SYMBOLS, frequency="1d", source="yfinance")
    prices = align_prices(universe, join="inner")
    print(
        f"  {prices.shape[0]} trading days × {prices.shape[1]} assets  "
        f"({prices.index[0].date()} → {prices.index[-1].date()})"
    )

    # 2. Build strategy
    strategy = MomentumRotationStrategy(
        lookback=lookback,
        top_n=top_n,
        rebalance_freq=rebalance_freq,
    )
    print(f"\nStrategy: {strategy.name}")

    # 3. Run strategy + backtest
    print("Running backtest...")
    result = run_strategy(prices, strategy, transaction_cost_bps=cost_bps)

    # 4. Print summary
    print("\n── Performance metrics ──────────────────────────")
    for k, v in result.metrics.items():
        print(f"  {k:<28} {v:>8.4f}")

    # 5. Package into ExperimentResult and save
    experiment = ExperimentResult(
        experiment_name=experiment_name,
        strategy_name=result.strategy_name,
        parameters={**strategy.params(), "transaction_cost_bps": cost_bps},
        metrics=result.metrics,
        weights=result.weights,
        equity_curve=result.backtest["equity_curve"],
        returns=result.backtest["net_return"],
    )

    out_path = save_experiment(experiment, output_dir=OUTPUT_BASE)
    print(f"\nOutputs saved to: {out_path.resolve()}")
    print("  metadata.json")
    print("  metrics.json")
    print("  equity_curve.parquet")
    print("  weights.parquet")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run momentum rotation strategy.")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--rebalance-freq", type=str, default=DEFAULT_REBALANCE_FREQ)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        lookback=args.lookback,
        top_n=args.top_n,
        rebalance_freq=args.rebalance_freq,
        cost_bps=args.cost_bps,
    )
