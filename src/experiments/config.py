"""Configuration loading for reproducible experiments.

Two config abstractions live here:

ExperimentSpec (Phase D0)
    Typed, hashable, serializable configuration for a research experiment.
    Supports deterministic ``experiment_hash()`` and full round-trip JSON
    serialization.  Intended as the primary config object for new experiments.

ExperimentConfig / load_experiment_config (legacy YAML scaffold)
    YAML-driven config parsed into an ExperimentContext.  Kept for backward
    compatibility with existing experiment YAML files and tests.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

from src.core import DateRange, ExperimentContext, Horizon, Universe


# ---------------------------------------------------------------------------
# Phase D0: ExperimentSpec — typed, hashable experiment configuration
# ---------------------------------------------------------------------------


@dataclass
class ExperimentSpec:
    """Typed configuration for a research experiment.

    Attributes:
        experiment_name: Human-readable unique identifier for this experiment.
        strategy_name:   Name of the strategy class or variant being tested.
        universe:        Ordered list of ticker symbols in the test universe.
        start_date:      First date of the evaluation period (ISO 'YYYY-MM-DD').
        end_date:        Last date of the evaluation period (ISO 'YYYY-MM-DD').
        rebalance_frequency: Pandas offset alias (e.g. 'ME', 'QE', 'W').
        parameters:      Strategy hyperparameters — must be JSON-serializable.
        tags:            Optional list of metadata labels for registry queries.
        description:     Optional free-text description of the experiment intent.
    """

    experiment_name: str
    strategy_name: str
    universe: list[str]
    start_date: str
    end_date: str
    rebalance_frequency: str
    parameters: dict[str, Any]
    tags: list[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Export to a plain, JSON-serializable dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "strategy_name": self.strategy_name,
            "universe": list(self.universe),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "rebalance_frequency": self.rebalance_frequency,
            "parameters": _json_safe(self.parameters),
            "tags": list(self.tags),
            "description": self.description,
        }

    def save_config(self, path: str | Path) -> None:
        """Write the spec to a JSON file at ``path``."""
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentSpec:
        """Reconstruct an ExperimentSpec from a ``to_dict()`` output."""
        return cls(
            experiment_name=data["experiment_name"],
            strategy_name=data["strategy_name"],
            universe=list(data["universe"]),
            start_date=data["start_date"],
            end_date=data["end_date"],
            rebalance_frequency=data["rebalance_frequency"],
            parameters=data.get("parameters", {}),
            tags=list(data.get("tags", [])),
            description=data.get("description", ""),
        )

    @classmethod
    def load_config(cls, path: str | Path) -> ExperimentSpec:
        """Load a spec from a JSON file written by ``save_config()``."""
        with Path(path).open(encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def experiment_hash(spec: ExperimentSpec) -> str:
    """Compute a stable, deterministic SHA-256 hash of experiment configuration.

    Only the fields that define the experiment's identity are hashed —
    ``tags`` and ``description`` are excluded as they are metadata.
    Parameters are JSON-serialized with sorted keys to guarantee
    cross-session determinism.

    Returns:
        First 12 hex characters of the SHA-256 digest.
    """
    payload = {
        "experiment_name": spec.experiment_name,
        "strategy_name": spec.strategy_name,
        "universe": sorted(spec.universe),
        "start_date": spec.start_date,
        "end_date": spec.end_date,
        "rebalance_frequency": spec.rebalance_frequency,
        "parameters": _json_safe(spec.parameters),
    }
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


def _json_safe(obj: Any) -> Any:
    """Recursively coerce non-JSON-native types to their JSON equivalents."""
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


@dataclass(frozen=True)
class ExperimentConfig:
    """Parsed experiment manifest with raw config preserved."""

    path: Path
    context: ExperimentContext
    raw: Mapping[str, Any]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML document from disk."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML config must be a mapping: {config_path}")
    return loaded


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment YAML file into an experiment context."""
    config_path = Path(path)
    raw = load_yaml(config_path)
    experiment = raw.get("experiment", raw)
    if not isinstance(experiment, Mapping):
        raise ValueError("Experiment section must be a mapping.")

    universe = _load_universe(experiment["universe"], base_dir=config_path.parent)
    horizon_raw = experiment["horizon"]
    date_range_raw = experiment["date_range"]

    context = ExperimentContext(
        experiment_id=str(experiment.get("id", experiment["name"])),
        name=str(experiment["name"]),
        created_at=_parse_datetime(experiment.get("created_at")),
        universe=universe,
        horizon=Horizon(name=str(horizon_raw["name"]), periods=int(horizon_raw["periods"])),
        date_range=DateRange(
            start=_parse_date(date_range_raw["start"]),
            end=_parse_date(date_range_raw["end"]),
        ),
        config=raw,
    )
    return ExperimentConfig(path=config_path, context=context, raw=raw)


def _load_universe(value: Any, base_dir: Path) -> Universe:
    if isinstance(value, str) and value.endswith((".yaml", ".yml")):
        path = Path(value)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
            if not path.exists():
                path = Path(value)
        raw = load_yaml(path)
        universe_raw = raw.get("universe", raw)
        symbols = universe_raw.get("symbols", [])
        return Universe(
            name=str(universe_raw["name"]),
            symbols=tuple(str(symbol) for symbol in symbols),
            description=universe_raw.get("description"),
        )

    if isinstance(value, Mapping):
        symbols = value.get("symbols", [])
        return Universe(
            name=str(value["name"]),
            symbols=tuple(str(symbol) for symbol in symbols),
            description=value.get("description"),
        )

    return Universe(name=str(value), symbols=())


def _parse_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if value is None:
        return datetime.now(UTC)
    text = str(value).replace("Z", "+00:00")
    return datetime.fromisoformat(text)
