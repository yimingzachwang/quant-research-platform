"""Walk-forward validation script for MomentumRotationStrategy.

Loads the ETF universe, generates rolling and expanding time splits,
runs walk-forward validation, prints stability diagnostics, and saves
plots to results/validation/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")

import pandas as pd

from src.portfolio.alignment import load_universe, align_prices
from src.strategies.baselines import EqualWeightStrategy
from src.strategies.momentum_rotation import MomentumRotationStrategy
from src.validation.splits import expanding_time_splits, rolling_time_splits
from src.validation.stability import (
    parameter_robustness_summary,
    split_metrics_table,
    summarize_stability,
)
from src.validation.walk_forward import run_walk_forward_validation
from src.visualization.styles import apply_research_style
from src.visualization.validation_plots import (
    plot_metric_stability,
    plot_split_sharpes,
    plot_walk_forward_equity,
    plot_walk_forward_stitched,
)

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD", "XLF", "XLK"]
TRAIN_MONTHS = 36
TEST_MONTHS = 12
STEP_MONTHS = 12
OUTPUT_DIR = Path("results/validation")


def main() -> None:
    apply_research_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading universe …")
    universe = load_universe(TICKERS)
    prices = align_prices(universe)
    print(f"  Price data: {prices.index[0].date()} → {prices.index[-1].date()} "
          f"({len(prices)} trading days, {len(prices.columns)} assets)")

    # ------------------------------------------------------------------
    # Generate splits
    # ------------------------------------------------------------------
    rolling_splits = rolling_time_splits(
        prices.index,
        train_months=TRAIN_MONTHS,
        test_months=TEST_MONTHS,
        step_months=STEP_MONTHS,
    )
    expanding_splits = expanding_time_splits(
        prices.index,
        min_train_months=TRAIN_MONTHS,
        test_months=TEST_MONTHS,
        step_months=STEP_MONTHS,
    )
    print(f"  Rolling splits: {len(rolling_splits)},  "
          f"Expanding splits: {len(expanding_splits)}")

    # ------------------------------------------------------------------
    # Run walk-forward — MomentumRotation
    # ------------------------------------------------------------------
    momentum_strategy = MomentumRotationStrategy(lookback=252, top_n=3, rebalance_freq="ME")
    print(f"\nRunning walk-forward: {momentum_strategy.name} …")
    wf_momentum = run_walk_forward_validation(prices, momentum_strategy, rolling_splits)

    # ------------------------------------------------------------------
    # Run walk-forward — EqualWeight (benchmark)
    # ------------------------------------------------------------------
    ew_strategy = EqualWeightStrategy()
    print(f"Running walk-forward: {ew_strategy.name} …")
    wf_ew = run_walk_forward_validation(prices, ew_strategy, rolling_splits)

    # ------------------------------------------------------------------
    # Print stability summaries
    # ------------------------------------------------------------------
    print("\n─── MomentumRotation Walk-Forward Stability ───")
    stats = summarize_stability(wf_momentum)
    print(f"  Splits              : {stats['n_splits']}")
    print(f"  Mean OOS Sharpe     : {stats['mean_sharpe']:.3f} ± {stats['std_sharpe']:.3f}")
    print(f"  Hit rate (Sharpe>0) : {stats['hit_rate_positive_sharpe']:.0%}")
    print(f"  Mean ann. return    : {stats['mean_annualized_return']:.1%}")
    print(f"  Mean max drawdown   : {stats['mean_max_drawdown']:.1%}")
    print(f"  Worst max drawdown  : {stats['worst_max_drawdown']:.1%}")

    print("\n─── Per-Split Metrics (MomentumRotation) ───")
    table = split_metrics_table(wf_momentum)
    display_cols = ["test_start", "test_end", "annualized_return", "sharpe_ratio", "max_drawdown"]
    print(table[display_cols].to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n─── Parameter Robustness Summary ───")
    robustness = parameter_robustness_summary(
        {"MomentumRotation": wf_momentum, "EqualWeight": wf_ew},
        metric="sharpe_ratio",
    )
    print(robustness.to_string(float_format=lambda x: f"{x:.3f}"))

    # ------------------------------------------------------------------
    # Save plots
    # ------------------------------------------------------------------
    figures: list[tuple[str, plt.Figure]] = [
        ("momentum_equity_curves.png", plot_walk_forward_equity(
            wf_momentum,
            title="MomentumRotation — Walk-Forward Equity Curves",
            save_path=str(OUTPUT_DIR / "momentum_equity_curves.png"),
        )),
        ("momentum_stitched_equity.png", plot_walk_forward_stitched(
            wf_momentum,
            title="MomentumRotation — Stitched OOS Equity",
            save_path=str(OUTPUT_DIR / "momentum_stitched_equity.png"),
        )),
        ("momentum_split_sharpes.png", plot_split_sharpes(
            wf_momentum,
            title="MomentumRotation — OOS Sharpe by Split",
            save_path=str(OUTPUT_DIR / "momentum_split_sharpes.png"),
        )),
        ("momentum_return_stability.png", plot_metric_stability(
            wf_momentum,
            metric="annualized_return",
            title="MomentumRotation — OOS Annual Return by Split",
            save_path=str(OUTPUT_DIR / "momentum_return_stability.png"),
        )),
    ]

    for fname, fig in figures:
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR / fname}")

    print(f"\nAll outputs written to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
