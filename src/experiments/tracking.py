"""Experiment tracking: filesystem-first save/load + optional MLflow adapter.

Two layers live here:

save_run / load_run (Phase D0 — filesystem-first)
    Persist an experiment result alongside its ExperimentSpec and any
    matplotlib plots.  Output folder layout::

        <output_dir>/<experiment_name>/
            metadata.json         ← run provenance
            config.json           ← ExperimentSpec (if provided)
            metrics.json          ← scalar performance metrics
            equity_curve.parquet
            returns.parquet
            weights.parquet
            predictions.parquet   ← optional, for future ML models
            plots/                ← PNG exports (caller-supplied)
            diagnostics/          ← reserved for future diagnostic artefacts

ExperimentTracker / TrackingRun (legacy MLflow adapter)
    No-op by default; only activates when ``enabled=True`` and mlflow is
    installed.  Kept for backward compatibility with ExperimentRunner.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.core import ExperimentContext
from src.experiments.results import ExperimentResult, load_experiment, save_experiment

if TYPE_CHECKING:
    from src.experiments.config import ExperimentSpec


# ---------------------------------------------------------------------------
# Phase D0: filesystem-first save / load
# ---------------------------------------------------------------------------


def save_run(
    result: ExperimentResult,
    spec: ExperimentSpec | None = None,
    output_dir: str | Path = Path("results/experiments"),
    plots: dict[str, Any] | None = None,
    predictions: pd.DataFrame | None = None,
) -> Path:
    """Persist a full experiment run to the filesystem.

    Delegates core persistence to ``save_experiment()`` (metadata.json,
    metrics.json, equity_curve / returns / weights parquet), then adds:

    - ``config.json`` if ``spec`` is provided.
    - PNG files for any matplotlib Figures in ``plots``.
    - ``predictions.parquet`` if ``predictions`` is provided.
    - Empty ``plots/`` and ``diagnostics/`` subdirectories.

    Existing files are overwritten — re-saving the same experiment name
    produces a clean, reproducible snapshot.

    Args:
        result:     ExperimentResult from a strategy run.
        spec:       Optional ExperimentSpec describing the experiment config.
        output_dir: Root directory under which the experiment folder sits.
        plots:      Optional dict mapping filename (without extension) →
                    matplotlib Figure.  Each figure is saved as PNG.
        predictions: Optional Date × Feature DataFrame for future ML use.

    Returns:
        Path to the experiment output folder.
    """
    out = save_experiment(result, output_dir=output_dir)

    if spec is not None:
        spec.save_config(out / "config.json")

    if predictions is not None:
        predictions.to_parquet(out / "predictions.parquet")

    plots_dir = out / "plots"
    plots_dir.mkdir(exist_ok=True)
    (out / "diagnostics").mkdir(exist_ok=True)

    if plots:
        for name, fig in plots.items():
            fname = name if name.endswith(".png") else f"{name}.png"
            fig.savefig(plots_dir / fname, bbox_inches="tight",
                        dpi=getattr(fig, "_save_dpi", 150))

    return out


def load_run(
    path: str | Path,
) -> tuple[ExperimentResult, ExperimentSpec | None]:
    """Load a full experiment run saved by ``save_run()``.

    Args:
        path: Path to the experiment folder.

    Returns:
        Tuple of ``(ExperimentResult, ExperimentSpec | None)``.
        The spec is ``None`` when ``config.json`` was not saved.

    Raises:
        FileNotFoundError: If the folder or required files are missing.
    """
    from src.experiments.config import ExperimentSpec

    folder = Path(path)
    result = load_experiment(folder)

    spec: ExperimentSpec | None = None
    config_path = folder / "config.json"
    if config_path.exists():
        spec = ExperimentSpec.load_config(config_path)

    return result, spec


@dataclass(frozen=True)
class TrackingRun:
    """Reference to a tracking run."""

    run_id: str
    tracking_uri: str | None = None


class ExperimentTracker:
    """Minimal MLflow-backed tracking wrapper with a no-op fallback."""

    def __init__(self, enabled: bool = False, tracking_uri: str | None = None) -> None:
        self.enabled = enabled
        self.tracking_uri = tracking_uri

    def start_run(self, context: ExperimentContext) -> TrackingRun:
        """Start or describe a run for an experiment context."""
        if not self.enabled:
            return TrackingRun(run_id=context.experiment_id, tracking_uri=self.tracking_uri)

        import mlflow

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        active_run = mlflow.start_run(run_name=context.experiment_id)
        mlflow.log_params(
            {
                "experiment_id": context.experiment_id,
                "universe": context.universe.name,
                "horizon": context.horizon.name,
                "start": context.date_range.start.isoformat(),
                "end": context.date_range.end.isoformat(),
            }
        )
        return TrackingRun(run_id=active_run.info.run_id, tracking_uri=self.tracking_uri)

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        """Log metrics when tracking is enabled."""
        if not self.enabled:
            return

        import mlflow

        mlflow.log_metrics(dict(metrics))

    def log_artifact_manifest(self, manifest: Mapping[str, Any]) -> None:
        """Placeholder hook for artifact manifest logging."""
        if not self.enabled:
            return

        import mlflow

        mlflow.log_dict(dict(manifest), "artifact_manifest.json")

    def end_run(self) -> None:
        """End the active tracking run when tracking is enabled."""
        if not self.enabled:
            return

        import mlflow

        mlflow.end_run()
