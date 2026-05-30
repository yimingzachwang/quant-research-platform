"""Run a single experiment and persist all artefacts.

Demonstrates the Phase D0 experiment lifecycle:
  1. Build an ExperimentSpec (typed, hashable config)
  2. Load data and run a strategy
  3. Save artefacts with save_run() (result + config + plots)
  4. Register in the local experiment registry
  5. Print a concise summary

Output directory: results/experiments/<experiment_name>/
Registry:         results/experiments/registry.json
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")

from src.experiments.config import ExperimentSpec, experiment_hash
from src.experiments.registry import ExperimentRegistry
from src.experiments.results import ExperimentResult
from src.experiments.tracking import save_run
from src.portfolio.alignment import align_prices, load_universe
from src.strategies.momentum_rotation import MomentumRotationStrategy
from src.strategies.runner import run_strategy
from src.visualization.backtest_plots import plot_equity_and_drawdown
from src.visualization.styles import apply_research_style

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPERIMENT_NAME = "momentum_rotation_d0"
TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD", "XLF", "XLK"]
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
REBALANCE_FREQ = "ME"
LOOKBACK = 252
TOP_N = 3
OUTPUT_DIR = Path("results/experiments")
REGISTRY_PATH = OUTPUT_DIR / "registry.json"


def main() -> None:
    apply_research_style()

    # ------------------------------------------------------------------
    # 1. Define the experiment spec
    # ------------------------------------------------------------------
    spec = ExperimentSpec(
        experiment_name=EXPERIMENT_NAME,
        strategy_name=f"MomentumRotation(lookback={LOOKBACK},top_n={TOP_N})",
        universe=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        rebalance_frequency=REBALANCE_FREQ,
        parameters={"lookback": LOOKBACK, "top_n": TOP_N, "rebalance_freq": REBALANCE_FREQ},
        tags=["momentum", "etf", "rotation"],
        description="Momentum rotation across 7 core ETFs — Phase D0 baseline.",
    )
    print(f"Experiment : {spec.experiment_name}")
    print(f"Hash       : {experiment_hash(spec)}")

    # ------------------------------------------------------------------
    # 2. Load data and run strategy
    # ------------------------------------------------------------------
    print("Loading universe …")
    universe = load_universe(TICKERS)
    prices = align_prices(universe)

    strategy = MomentumRotationStrategy(
        lookback=LOOKBACK, top_n=TOP_N, rebalance_freq=REBALANCE_FREQ
    )
    print(f"Running {strategy.name} …")
    sr = run_strategy(prices, strategy)

    result = ExperimentResult(
        experiment_name=EXPERIMENT_NAME,
        strategy_name=strategy.name,
        parameters=spec.parameters,
        metrics=sr.metrics,
        weights=sr.weights,
        equity_curve=sr.backtest["gross_return"].add(1).cumprod(),
        returns=sr.backtest["net_return"],
        created_at=datetime.now(UTC),
    )

    # ------------------------------------------------------------------
    # 3. Generate plots
    # ------------------------------------------------------------------
    fig = plot_equity_and_drawdown(result.returns, title=f"{strategy.name} — Equity & Drawdown")

    # ------------------------------------------------------------------
    # 4. Save artefacts
    # ------------------------------------------------------------------
    out = save_run(
        result,
        spec=spec,
        output_dir=OUTPUT_DIR,
        plots={"equity_and_drawdown": fig},
    )
    plt.close(fig)
    print(f"Saved to   : {out}")

    # ------------------------------------------------------------------
    # 5. Register
    # ------------------------------------------------------------------
    registry = ExperimentRegistry(REGISTRY_PATH)
    exp_id = registry.register(result, spec=spec, path=out)
    print(f"Registered : {exp_id}")

    # ------------------------------------------------------------------
    # 6. Print summary
    # ------------------------------------------------------------------
    print("\n─── Metrics ───")
    for k, v in result.metrics.items():
        print(f"  {k:<28}: {v:.4f}")


if __name__ == "__main__":
    main()
