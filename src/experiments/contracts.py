"""Artefact version constants and advisory contracts for experiment directories.

Constants describe the expected layout of a saved experiment directory.
check_artefact_dir() is advisory: it returns a list of violations and never
raises — callers decide whether to warn, error, or proceed.
"""

from __future__ import annotations

from pathlib import Path

# Bump when the artefact schema changes in a breaking way.
ARTEFACT_VERSION = "1"

# Files that must be present for a directory to be a valid experiment artefact.
REQUIRED_ARTEFACTS = ("metadata.json", "metrics.json")

# Core data files written by save_run().
CORE_ARTEFACTS = ("equity_curve.parquet", "returns.parquet", "weights.parquet")

# Config artefact written by Phase D0 / legacy scripts.
CONFIG_ARTEFACTS = ("config.json",)

# Normalised config artefact written by the Phase D1 orchestrator.
D1_ARTEFACTS = ("normalized_config.json",)

# Sub-directories created under the experiment root.
ARTEFACT_DIRS = ("plots", "diagnostics")

# ML-specific artefacts written by the F3 orchestrator for version "2" experiments.
ML_ARTEFACTS = ("ml_provenance.json",)

# Diagnostic artefacts written into diagnostics/ by the orchestrator.
# split_metrics.json is written for all experiments that run walk-forward validation.
# ml_diagnostics.json is written for version "2" (ML) experiments only.
DIAGNOSTICS_ARTEFACTS = ("split_metrics.json", "ml_diagnostics.json")


def check_diagnostics_dir(path: str | Path) -> list[str]:
    """Advisory checker for the diagnostics sub-directory.

    Only checks for the presence of the diagnostics/ directory and its known
    artefacts.  Callers decide whether missing artefacts are a violation.

    Args:
        path: Path to the experiment root directory.

    Returns:
        List of human-readable violation strings.  Empty list = all present.
    """
    p = Path(path)
    violations: list[str] = []

    if not p.is_dir():
        violations.append(f"Directory does not exist: {p}")
        return violations

    diag_dir = p / "diagnostics"
    if not diag_dir.is_dir():
        violations.append("diagnostics/ subdirectory not found")
        return violations

    for name in DIAGNOSTICS_ARTEFACTS:
        if not (diag_dir / name).exists():
            violations.append(f"Missing diagnostic artefact: diagnostics/{name}")

    return violations


def check_ml_artefacts(path: str | Path) -> list[str]:
    """Advisory checker for ML-specific artefacts (version "2" experiments).

    Args:
        path: Path to the experiment directory to inspect.

    Returns:
        List of human-readable violation strings.  Empty list means all
        ML artefacts are present.
    """
    p = Path(path)
    violations: list[str] = []

    if not p.is_dir():
        violations.append(f"Directory does not exist: {p}")
        return violations

    for name in ML_ARTEFACTS:
        if not (p / name).exists():
            violations.append(f"Missing ML artefact: {name}")

    return violations


def check_artefact_dir(path: str | Path) -> list[str]:
    """Return a list of advisory violations for an experiment directory.

    Only REQUIRED_ARTEFACTS are checked — their absence is considered a
    violation.  All other constants describe expected but optional content.

    Args:
        path: Path to the experiment directory to inspect.

    Returns:
        List of human-readable violation strings.  An empty list means
        all required artefacts are present.  The directory not existing
        is itself reported as a violation rather than raising.
    """
    p = Path(path)
    violations: list[str] = []

    if not p.is_dir():
        violations.append(f"Directory does not exist: {p}")
        return violations

    for name in REQUIRED_ARTEFACTS:
        if not (p / name).exists():
            violations.append(f"Missing required artefact: {name}")

    return violations
