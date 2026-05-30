"""Experiment result dataclass and filesystem persistence.

ExperimentResult is a plain dataclass that holds everything produced by one
experiment run.  save_experiment() writes it to a deterministic folder
structure using only the local filesystem — no databases, no MLflow.
load_experiment() reconstructs an ExperimentResult from the same folder.

Output layout (relative to the caller-supplied output_dir):

    <output_dir>/
        <experiment_name>/
            metadata.json       ← name, strategy, params, timestamp
            metrics.json        ← scalar performance metrics
            equity_curve.parquet
            returns.parquet
            weights.parquet

All parquet files use the default snappy compression.  JSON files are
human-readable with 2-space indentation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ExperimentResult:
    """Everything produced by one experiment run.

    Attributes:
        experiment_name: Unique name for this run (used as folder name).
        strategy_name:   Human-readable strategy identifier.
        parameters:      Strategy hyperparameters (serialisable dict).
        metrics:         Scalar performance metrics (annualized return etc.).
        weights:         Date × Asset applied weight DataFrame.
        equity_curve:    Portfolio equity curve Series (starts at 1.0).
        returns:         Portfolio net return Series.
        created_at:      UTC timestamp of the run.
    """

    experiment_name: str
    strategy_name: str
    parameters: dict[str, Any]
    metrics: dict[str, float]
    weights: pd.DataFrame
    equity_curve: pd.Series
    returns: pd.Series
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def save_experiment(
    result: ExperimentResult,
    output_dir: str | Path = Path("results/experiments"),
) -> Path:
    """Persist an ExperimentResult to a deterministic folder structure.

    Creates ``<output_dir>/<experiment_name>/`` and writes four files:
        metadata.json, metrics.json, equity_curve.parquet, weights.parquet.

    Existing files are overwritten — re-running an experiment with the same
    name produces a clean, reproducible snapshot.

    Args:
        result:     The ExperimentResult to persist.
        output_dir: Base directory under which the experiment folder is created.
            Defaults to ``results/experiments/``.

    Returns:
        Path to the experiment output folder.
    """
    out = Path(output_dir) / result.experiment_name
    out.mkdir(parents=True, exist_ok=True)

    # metadata.json — human-readable run provenance
    metadata: dict[str, Any] = {
        "experiment_name": result.experiment_name,
        "strategy_name": result.strategy_name,
        "parameters": _json_safe(result.parameters),
        "created_at": result.created_at.isoformat(),
    }
    _write_json(out / "metadata.json", metadata)

    # metrics.json — scalar performance summary
    _write_json(out / "metrics.json", _json_safe(result.metrics))

    # equity_curve.parquet
    result.equity_curve.rename("equity_curve").to_frame().to_parquet(
        out / "equity_curve.parquet"
    )

    # returns.parquet
    result.returns.rename("net_return").to_frame().to_parquet(out / "returns.parquet")

    # weights.parquet
    result.weights.to_parquet(out / "weights.parquet")

    return out


def load_experiment(path: str | Path) -> ExperimentResult:
    """Load a saved ExperimentResult from a folder written by save_experiment().

    Args:
        path: Path to the experiment folder (the directory that contains
            metadata.json, metrics.json, equity_curve.parquet, etc.).

    Returns:
        ExperimentResult populated from the saved artifacts.

    Raises:
        FileNotFoundError: If the folder or any required file is missing.
    """
    folder = Path(path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Experiment folder not found: {folder}")

    with (folder / "metadata.json").open(encoding="utf-8") as f:
        metadata = json.load(f)

    with (folder / "metrics.json").open(encoding="utf-8") as f:
        metrics: dict[str, float] = json.load(f)

    equity_curve = pd.read_parquet(folder / "equity_curve.parquet")["equity_curve"]
    weights = pd.read_parquet(folder / "weights.parquet")
    returns = pd.read_parquet(folder / "returns.parquet")["net_return"]

    return ExperimentResult(
        experiment_name=metadata["experiment_name"],
        strategy_name=metadata["strategy_name"],
        parameters=metadata["parameters"],
        metrics=metrics,
        weights=weights,
        equity_curve=equity_curve,
        returns=returns,
        created_at=datetime.fromisoformat(metadata["created_at"]),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _json_safe(obj: Any) -> Any:
    """Recursively coerce numpy/pandas scalar types to JSON-native Python."""
    # Import here to avoid a hard numpy dependency at module level
    try:
        import numpy as np  # type: ignore[import-untyped]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj
