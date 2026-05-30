"""Comparison utilities for ExperimentResult objects.

Analogous to ``src/strategies/comparison.py`` (which compares StrategyResults)
but operates on saved ExperimentResult artefacts.  Intended for post-hoc
analysis across multiple persisted experiments.

All functions accept plain Python dicts or ExperimentResult objects and return
pandas DataFrames — easy to inspect, export, or pass to visualization helpers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.experiments.results import ExperimentResult, load_experiment

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_experiments(
    experiments: dict[str, ExperimentResult] | list[ExperimentResult],
) -> dict[str, ExperimentResult]:
    """Validate and normalise a collection of ExperimentResults.

    Accepts either a pre-keyed dict or a bare list.  List items are keyed
    by ``experiment_name``; raises on duplicate names.

    Args:
        experiments: Dict mapping label → ExperimentResult, or a plain list.

    Returns:
        Ordered dict preserving insertion order.

    Raises:
        ValueError: On duplicate experiment names.
    """
    if isinstance(experiments, list):
        mapping: dict[str, ExperimentResult] = {}
        for exp in experiments:
            if exp.experiment_name in mapping:
                raise ValueError(
                    f"Duplicate experiment name {exp.experiment_name!r}. "
                    "Pass a dict with explicit labels to resolve."
                )
            mapping[exp.experiment_name] = exp
        return mapping

    # dict path — check for duplicate values (same object under different keys is fine,
    # but two keys mapping to the same experiment_name would be confusing)
    return dict(experiments)


def metrics_table(
    experiments: dict[str, ExperimentResult] | list[ExperimentResult],
) -> pd.DataFrame:
    """Build a metrics DataFrame from a collection of ExperimentResults.

    Returns:
        DataFrame indexed by experiment label with one column per metric.
        Columns: annualized_return, annualized_volatility, sharpe_ratio,
        max_drawdown, calmar_ratio, hit_rate (whatever the result.metrics dicts
        contain — no fixed schema assumed).
    """
    resolved = compare_experiments(experiments)
    rows = {label: result.metrics for label, result in resolved.items()}
    df = pd.DataFrame(rows).T
    df.index.name = "experiment"
    return df


def rank_experiments(
    experiments: dict[str, ExperimentResult] | list[ExperimentResult],
    by: str = "sharpe_ratio",
    ascending: bool = False,
) -> pd.DataFrame:
    """Return a ranked metrics table sorted by ``by``.

    Args:
        experiments: Experiment collection (dict or list).
        by:          Metric column to sort by.
        ascending:   Sort direction (default False = best first for Sharpe/return).

    Returns:
        Metrics DataFrame with an added ``rank`` column (1 = best).

    Raises:
        ValueError: If ``by`` is not a column in the metrics table.
    """
    table = metrics_table(experiments)
    if by not in table.columns:
        raise ValueError(
            f"Metric {by!r} not found in experiment metrics. "
            f"Available: {list(table.columns)}"
        )
    table = table.sort_values(by, ascending=ascending)
    table.insert(0, "rank", range(1, len(table) + 1))
    return table


def load_and_compare(
    paths: list[str | Path],
    labels: list[str] | None = None,
) -> dict[str, ExperimentResult]:
    """Load experiments from disk and return a labelled dict for comparison.

    Args:
        paths:  List of paths to experiment folders (saved by save_experiment
                or save_run).
        labels: Optional list of display labels.  Defaults to each folder's
                stem name.

    Returns:
        Dict mapping label → ExperimentResult.

    Raises:
        FileNotFoundError: If any path does not exist.
        ValueError: If ``labels`` length differs from ``paths``.
    """
    if labels is not None and len(labels) != len(paths):
        raise ValueError(
            f"labels length ({len(labels)}) must match paths length ({len(paths)})"
        )
    result: dict[str, ExperimentResult] = {}
    for i, p in enumerate(paths):
        label = labels[i] if labels is not None else Path(p).stem
        result[label] = load_experiment(p)
    return result


def metrics_delta(
    baseline: ExperimentResult,
    candidate: ExperimentResult,
) -> dict[str, float]:
    """Compute absolute metric differences: candidate − baseline.

    Returns:
        Dict mapping metric name → (candidate_value − baseline_value).
        Only metrics present in both experiments are included.
    """
    shared = set(baseline.metrics) & set(candidate.metrics)
    return {k: candidate.metrics[k] - baseline.metrics[k] for k in sorted(shared)}
