"""Portfolio diagnostics script.

Loads the core ETF universe, runs MomentumRotationStrategy, then generates
five diagnostic plots and prints a concise summary to stdout.

Output directory: results/portfolio_diagnostics/
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import matplotlib
from src.portfolio.alignment import align_prices, load_universe
from src.portfolio.panel import universe_returns
from src.strategies.momentum_rotation import MomentumRotationStrategy
from src.strategies.runner import run_strategy
from src.visualization.diagnostics import (
    compute_concentration_metrics,
    compute_turnover,
    rolling_average_correlation,
)
from src.visualization.portfolio_plots import (
    plot_asset_contribution,
    plot_rolling_correlation,
    plot_turnover,
    plot_weight_concentration,
    plot_weight_heatmap,
)
from src.visualization.styles import apply_research_style

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD", "XLF", "XLK"]
OUTPUT_DIR = Path("results/portfolio_diagnostics")
LOOKBACK = 252
TOP_N = 3
REBALANCE_FREQ = "ME"
ROLLING_CORR_WINDOW = 60


def main() -> None:
    apply_research_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading universe …")
    universe = load_universe(TICKERS)
    prices = align_prices(universe)
    returns = universe_returns(prices)

    # ------------------------------------------------------------------
    # Run strategy
    # ------------------------------------------------------------------
    strategy = MomentumRotationStrategy(
        lookback=LOOKBACK, top_n=TOP_N, rebalance_freq=REBALANCE_FREQ
    )
    print(f"Running {strategy.name} …")
    result = run_strategy(prices, strategy)
    weights = result.weights

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    plots = [
        ("weight_heatmap.png", plot_weight_heatmap(weights, save_path=str(OUTPUT_DIR / "weight_heatmap.png"))),
        ("turnover.png", plot_turnover(weights, save_path=str(OUTPUT_DIR / "turnover.png"))),
        ("concentration.png", plot_weight_concentration(weights, save_path=str(OUTPUT_DIR / "concentration.png"))),
        ("asset_contribution.png", plot_asset_contribution(
            returns, weights, save_path=str(OUTPUT_DIR / "asset_contribution.png"))),
        ("rolling_correlation.png", plot_rolling_correlation(
            returns, window=ROLLING_CORR_WINDOW, save_path=str(OUTPUT_DIR / "rolling_correlation.png"))),
    ]

    import matplotlib.pyplot as plt
    for fname, fig in plots:
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR / fname}")

    # ------------------------------------------------------------------
    # Print diagnostics summary
    # ------------------------------------------------------------------
    print("\n─── Portfolio Diagnostics Summary ───")

    to = compute_turnover(weights).dropna()
    print(f"Turnover   avg={to.mean():.3f}  p95={to.quantile(0.95):.3f}")

    conc = compute_concentration_metrics(weights)
    print(f"HHI        avg={conc['hhi'].mean():.3f}  max={conc['hhi'].max():.3f}")
    print(f"EffectiveN avg={conc['effective_n'].mean():.2f}  min={conc['effective_n'].min():.2f}")
    print(f"MaxWeight  avg={conc['max_weight'].mean():.1%}  max={conc['max_weight'].max():.1%}")

    avg_corr = rolling_average_correlation(returns, window=ROLLING_CORR_WINDOW).dropna()
    print(f"AvgCorr    mean={avg_corr.mean():.3f}  p95={avg_corr.quantile(0.95):.3f}")

    metrics = result.metrics
    print("\n─── Strategy Performance ───")
    print(f"Annualized return : {metrics.get('annualized_return', float('nan')):.1%}")
    print(f"Sharpe ratio      : {metrics.get('sharpe_ratio', float('nan')):.2f}")
    print(f"Max drawdown      : {metrics.get('max_drawdown', float('nan')):.1%}")

    print(f"\nAll outputs written to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
