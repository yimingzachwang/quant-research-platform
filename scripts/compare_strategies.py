"""Compare BuyAndHold, EqualWeight, and MomentumRotation on a 7-ETF universe.

Pipeline:
    load prices → compare_strategies() → metrics_table() → save plots + artifacts

Outputs saved to:
    results/comparisons/
        metrics.csv
        equity_curves.png
        drawdowns.png
        sharpe_comparison.png
        metrics_table.png

Usage:
    python -m scripts.compare_strategies
    python -m scripts.compare_strategies --cost-bps 5 --lookback 252 --top-n 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


from src.portfolio.alignment import align_prices, load_universe
from src.strategies.baselines import BuyAndHoldStrategy, EqualWeightStrategy
from src.strategies.comparison import compare_strategies, metrics_table, rank_strategies
from src.strategies.momentum_rotation import MomentumRotationStrategy
from src.visualization import apply_research_style, save_figure
from src.visualization.comparison_plots import (
    plot_metric_comparison,
    plot_metrics_table,
    plot_strategy_drawdowns,
    plot_strategy_equity_curves,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYMBOLS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD"]
OUTPUT_DIR = Path("results/comparisons")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(
    cost_bps: float = 5.0,
    lookback: int = 252,
    top_n: int = 3,
    rebalance_freq: str = "ME",
) -> None:
    # 1. Load universe
    print(f"Loading universe: {SYMBOLS}")
    universe = load_universe(SYMBOLS, frequency="1d", source="yfinance")
    prices = align_prices(universe, join="inner")
    print(
        f"  {prices.shape[0]} trading days × {prices.shape[1]} assets  "
        f"({prices.index[0].date()} → {prices.index[-1].date()})"
    )

    # 2. Define strategies
    strategies = [
        BuyAndHoldStrategy(weights={"SPY": 1.0}),
        EqualWeightStrategy(rebalance_freq=rebalance_freq),
        MomentumRotationStrategy(
            lookback=lookback,
            top_n=top_n,
            rebalance_freq=rebalance_freq,
        ),
    ]
    names = [s.name for s in strategies]
    print("\nStrategies:\n" + "\n".join(f"  {n}" for n in names))

    # 3. Run comparison
    print("\nRunning backtests...")
    results = compare_strategies(prices, strategies, transaction_cost_bps=cost_bps)

    # 4. Build metrics table
    table = metrics_table(results)
    ranked = rank_strategies(results, by="sharpe_ratio")

    print("\n── Metrics comparison ───────────────────────────────────────────")
    print(
        table.to_string(
            float_format=lambda v: f"{v:8.4f}",
        )
    )

    print("\n── Ranked by Sharpe ─────────────────────────────────────────────")
    print(ranked.to_string(float_format=lambda v: f"{v:8.4f}"))

    # 5. Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_research_style()

    # metrics.csv — tidy, machine-readable
    table.to_csv(OUTPUT_DIR / "metrics.csv")
    print("\nSaved: metrics.csv")

    # equity_curves.png
    fig = plot_strategy_equity_curves(
        results,
        title=f"Strategy Comparison — Equity Curves ({cost_bps:.0f} bps cost)",
    )
    save_figure(fig, OUTPUT_DIR / "equity_curves.png")
    print("Saved: equity_curves.png")

    # drawdowns.png
    fig = plot_strategy_drawdowns(
        results,
        title="Strategy Comparison — Drawdowns",
    )
    save_figure(fig, OUTPUT_DIR / "drawdowns.png")
    print("Saved: drawdowns.png")

    # sharpe_comparison.png
    fig = plot_metric_comparison(
        table,
        metric="sharpe_ratio",
        title="Sharpe Ratio Comparison",
    )
    save_figure(fig, OUTPUT_DIR / "sharpe_comparison.png")
    print("Saved: sharpe_comparison.png")

    # metrics_table.png
    fig = plot_metrics_table(table, title="Strategy Metrics Summary")
    save_figure(fig, OUTPUT_DIR / "metrics_table.png")
    print("Saved: metrics_table.png")

    print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare momentum vs baselines.")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--rebalance-freq", type=str, default="ME")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        cost_bps=args.cost_bps,
        lookback=args.lookback,
        top_n=args.top_n,
        rebalance_freq=args.rebalance_freq,
    )
