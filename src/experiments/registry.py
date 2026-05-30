"""Lightweight filesystem-backed experiment registry.

The registry is a single JSON file — ``registry.json`` — that lives inside
the experiments output directory.  Each entry records enough metadata to
find, filter, and compare experiments without re-loading the full artefacts.

Registry entry schema::

    {
        "experiment_id":   "momentum_rotation_abc123def456",
        "experiment_name": "momentum_rotation",
        "config_hash":     "abc123def456",
        "timestamp":       "2026-05-22T12:00:00+00:00",
        "strategy_name":   "MomentumRotation(lookback=252,...)",
        "tags":            ["momentum", "etf"],
        "metrics_summary": {"sharpe_ratio": 0.65, "annualized_return": 0.082},
        "path":            "results/experiments/momentum_rotation"
    }

No databases.  Human-readable.  Easy to inspect with any text editor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.experiments.results import ExperimentResult

if TYPE_CHECKING:
    from src.experiments.config import ExperimentSpec


_SUMMARY_METRICS = ("annualized_return", "sharpe_ratio", "max_drawdown", "calmar_ratio")


def _make_entry(
    result: ExperimentResult,
    spec: ExperimentSpec | None,
    path: Path,
) -> dict[str, Any]:
    from src.experiments.config import experiment_hash

    config_hash = experiment_hash(spec) if spec is not None else ""
    experiment_id = f"{result.experiment_name}_{config_hash}" if config_hash else result.experiment_name

    metrics_summary = {k: v for k, v in result.metrics.items() if k in _SUMMARY_METRICS}

    return {
        "experiment_id": experiment_id,
        "experiment_name": result.experiment_name,
        "config_hash": config_hash,
        "timestamp": result.created_at.isoformat(),
        "strategy_name": result.strategy_name,
        "tags": list(spec.tags) if spec is not None else [],
        "metrics_summary": metrics_summary,
        "path": str(path),
    }


class ExperimentRegistry:
    """Append-and-query interface over a flat JSON experiment index.

    Args:
        registry_path: Path to ``registry.json``.  Created automatically on
            first write if it does not exist.
    """

    def __init__(
        self,
        registry_path: str | Path = Path("results/experiments/registry.json"),
    ) -> None:
        self._path = Path(registry_path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def register(
        self,
        result: ExperimentResult,
        spec: ExperimentSpec | None = None,
        path: Path | None = None,
    ) -> str:
        """Add or update an experiment entry in the registry.

        If an entry with the same ``experiment_name`` already exists it is
        replaced so that re-running an experiment produces a fresh record.

        Args:
            result: The ExperimentResult to register.
            spec:   Optional ExperimentSpec for richer metadata and hashing.
            path:   Filesystem path where artefacts were saved.  Auto-derived
                    from result.experiment_name when omitted.

        Returns:
            The ``experiment_id`` of the registered entry.
        """
        if path is None:
            path = self._path.parent / result.experiment_name

        experiments = self._load_raw()
        entry = _make_entry(result, spec, path)

        # Replace existing entry with same id; different id = new entry (append-safe)
        experiments = [e for e in experiments if e.get("experiment_id") != entry["experiment_id"]]
        experiments.append(entry)

        self._save_raw(experiments)
        return entry["experiment_id"]

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(self) -> list[dict[str, Any]]:
        """Return all registry entries, newest first."""
        entries = self._load_raw()
        return sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True)

    def latest(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the ``n`` most recently registered experiments."""
        return self.load()[:n]

    def query(
        self,
        strategy_name: str | None = None,
        tags: list[str] | None = None,
        min_sharpe: float | None = None,
    ) -> list[dict[str, Any]]:
        """Filter registry entries by strategy name, tags, and/or Sharpe threshold.

        Args:
            strategy_name: Exact or substring match against ``strategy_name``.
            tags:          Entry must contain ALL listed tags.
            min_sharpe:    Minimum Sharpe ratio in ``metrics_summary``.

        Returns:
            List of matching registry entries, newest first.
        """
        results = self.load()

        if strategy_name is not None:
            results = [e for e in results if strategy_name in e.get("strategy_name", "")]

        if tags:
            results = [
                e for e in results
                if all(t in e.get("tags", []) for t in tags)
            ]

        if min_sharpe is not None:
            results = [
                e for e in results
                if e.get("metrics_summary", {}).get("sharpe_ratio", float("-inf")) >= min_sharpe
            ]

        return results

    def get(self, experiment_name: str) -> dict[str, Any] | None:
        """Return the registry entry for ``experiment_name``, or None."""
        for entry in self.load():
            if entry.get("experiment_name") == experiment_name:
                return entry
        return None

    def remove(self, experiment_name: str) -> bool:
        """Delete an entry by experiment_name.  Returns True if found."""
        experiments = self._load_raw()
        filtered = [e for e in experiments if e.get("experiment_name") != experiment_name]
        if len(filtered) == len(experiments):
            return False
        self._save_raw(filtered)
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_raw(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        with self._path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

    def _save_raw(self, experiments: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(experiments, f, indent=2)


# ---------------------------------------------------------------------------
# Module-level convenience helpers
# ---------------------------------------------------------------------------


def register_experiment(
    result: ExperimentResult,
    spec: ExperimentSpec | None = None,
    path: Path | None = None,
    registry_path: str | Path = Path("results/experiments/registry.json"),
) -> str:
    """One-call shortcut: register an experiment with a default registry."""
    return ExperimentRegistry(registry_path).register(result, spec=spec, path=path)


def load_registry(
    registry_path: str | Path = Path("results/experiments/registry.json"),
) -> list[dict[str, Any]]:
    """Load all entries from the default (or specified) registry."""
    return ExperimentRegistry(registry_path).load()


def query_registry(
    strategy_name: str | None = None,
    tags: list[str] | None = None,
    min_sharpe: float | None = None,
    registry_path: str | Path = Path("results/experiments/registry.json"),
) -> list[dict[str, Any]]:
    """Query the registry with optional filters."""
    return ExperimentRegistry(registry_path).query(
        strategy_name=strategy_name, tags=tags, min_sharpe=min_sharpe
    )


def latest_experiments(
    n: int = 10,
    registry_path: str | Path = Path("results/experiments/registry.json"),
) -> list[dict[str, Any]]:
    """Return the n most recently registered experiments."""
    return ExperimentRegistry(registry_path).latest(n)
