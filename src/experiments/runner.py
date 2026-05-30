"""Experiment runner skeleton."""

from __future__ import annotations

from pathlib import Path

from src.backtesting import BacktestEngine, BacktestResult
from src.core import ExperimentContext
from src.experiments.config import ExperimentConfig, load_experiment_config
from src.experiments.tracking import ExperimentTracker


class ExperimentRunner:
    """Coordinates one reproducible experiment run."""

    def __init__(
        self,
        backtest_engine: BacktestEngine | None = None,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        self._backtest_engine = backtest_engine or BacktestEngine()
        self._tracker = tracker or ExperimentTracker(enabled=False)

    def run(self, context: ExperimentContext) -> BacktestResult:
        """Run the configured research workflow and return structured results."""
        tracking_run = self._tracker.start_run(context)
        try:
            result = self._backtest_engine.run(context)
            result.artifacts["tracking_run_id"] = tracking_run.run_id
            self._tracker.log_metrics(result.metrics)
            self._tracker.log_artifact_manifest(result.artifacts)
            return result
        finally:
            self._tracker.end_run()

    def run_config(self, config: ExperimentConfig | str | Path) -> BacktestResult:
        """Load and run a config-driven experiment."""
        experiment_config = (
            load_experiment_config(config) if isinstance(config, str | Path) else config
        )
        return self.run(experiment_config.context)
