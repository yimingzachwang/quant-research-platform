"""Load multiple saved experiments and compare their performance.

Demonstrates the Phase D0 comparison workflow:
  1. Load experiments from registry or explicit paths
  2. Print metrics table and ranking
  3. Optionally save comparison plots

Usage:
    python scripts/compare_experiments.py
"""
# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from src.experiments.comparison import metrics_table, rank_experiments
from src.experiments.registry import ExperimentRegistry
from src.experiments.results import load_experiment
from src.strategies.runner import StrategyResult
from src.visualization.comparison_plots import (
    plot_metric_comparison,
    plot_metrics_table,
    plot_strategy_equity_curves,
)
from src.visualization.styles import apply_research_style

REGISTRY_PATH = Path("results/experiments/registry.json")
OUTPUT_DIR = Path("results/comparisons")


def _build_mock_strategy_results(experiments):
    """Convert ExperimentResult dict to StrategyResult-compatible dict for viz."""
    out = {}
    for label, exp in experiments.items():
        out[label] = StrategyResult(
            strategy_name=label,
            weights=exp.weights,
            backtest=pd.DataFrame({"net_return": exp.returns}),
            metrics=exp.metrics,
        )
    return out


def main() -> None:
    apply_research_style()

    # ------------------------------------------------------------------
    # Load experiments from registry
    # ------------------------------------------------------------------
    registry = ExperimentRegistry(REGISTRY_PATH)
    entries = registry.load()

    if not entries:
        print("No experiments registered yet. Run scripts/run_experiment.py first.")
        return

    print(f"Found {len(entries)} experiment(s) in registry.\n")

    # Load all registered experiments
    experiments = {}
    for entry in entries:
        path = Path(entry["path"])
        if path.is_dir():
            experiments[entry["experiment_name"]] = load_experiment(path)
        else:
            print(f"  Warning: path not found for {entry['experiment_name']}: {path}")

    if not experiments:
        print("No experiment artefacts found on disk.")
        return

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print("─── Metrics Table ───")
    table = metrics_table(experiments)
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n─── Ranking by Sharpe ───")
    ranked = rank_experiments(experiments, by="sharpe_ratio")
    print(ranked[["rank", "sharpe_ratio", "annualized_return", "max_drawdown"]].to_string(
        float_format=lambda x: f"{x:.4f}"
    ))

    # ------------------------------------------------------------------
    # Save comparison plots (if more than one experiment)
    # ------------------------------------------------------------------
    if len(experiments) > 1:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        sr_dict = _build_mock_strategy_results(experiments)

        fig1 = plot_strategy_equity_curves(
            sr_dict, title="Experiment Equity Curves",
            save_path=str(OUTPUT_DIR / "equity_curves.png"),
        )
        plt.close(fig1)

        fig2 = plot_metric_comparison(
            table, metric="sharpe_ratio", title="Sharpe Ratio Comparison",
            save_path=str(OUTPUT_DIR / "sharpe_comparison.png"),
        )
        plt.close(fig2)

        fig3 = plot_metrics_table(
            table, title="Experiment Metrics",
            save_path=str(OUTPUT_DIR / "metrics_table.png"),
        )
        plt.close(fig3)

        print(f"\nPlots saved to: {OUTPUT_DIR}/")
    else:
        print("\n(Only one experiment loaded — skipping multi-experiment plots)")


if __name__ == "__main__":
    main()
