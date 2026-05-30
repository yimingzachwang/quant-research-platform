"""Loads persisted experiment artefacts from the filesystem.

Reads metadata.json, metrics.json, and optionally time-series parquet files
directly from the results/experiments/ tree.  Does not invoke the quantitative
execution engine — all state is resolved from pre-computed, persisted output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.orchestration.utils.filesystem import (
    equity_curve_path,
    metadata_path,
    metrics_path,
    returns_path,
    weights_path,
)
from src.orchestration.utils.serialization import load_json, load_parquet, load_series_parquet


def load_experiment_metadata(
    experiment_name: str,
    base: Path | str | None = None,
) -> dict[str, Any] | None:
    """Load metadata.json for an experiment."""
    return load_json(metadata_path(experiment_name, base))


def load_experiment_metrics(
    experiment_name: str,
    base: Path | str | None = None,
) -> dict[str, float] | None:
    return load_json(metrics_path(experiment_name, base))


def load_equity_curve(
    experiment_name: str,
    base: Path | str | None = None,
) -> pd.Series | None:
    return load_series_parquet(equity_curve_path(experiment_name, base))


def load_returns(
    experiment_name: str,
    base: Path | str | None = None,
) -> pd.Series | None:
    return load_series_parquet(returns_path(experiment_name, base))


def load_weights(
    experiment_name: str,
    base: Path | str | None = None,
) -> pd.DataFrame | None:
    return load_parquet(weights_path(experiment_name, base))


def load_experiment_bundle(
    experiment_name: str,
    base: Path | str | None = None,
    include_timeseries: bool = False,
) -> dict[str, Any]:
    """Load metadata + metrics (+ optionally time-series) into one dict.

    Setting ``include_timeseries=False`` (default) keeps the return value
    JSON-serialisable and avoids loading large parquet files.
    """
    bundle: dict[str, Any] = {
        "metadata": load_experiment_metadata(experiment_name, base),
        "metrics": load_experiment_metrics(experiment_name, base),
    }
    if include_timeseries:
        bundle["equity_curve"] = load_equity_curve(experiment_name, base)
        bundle["returns"] = load_returns(experiment_name, base)
        bundle["weights"] = load_weights(experiment_name, base)
    return bundle
