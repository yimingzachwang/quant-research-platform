"""Experiment orchestration, configuration, tracking, and comparison."""

# Phase F3: ML experiment config (version "2" schema)
# Phase D0: comparison utilities
from src.experiments.comparison import (
    compare_experiments,
    load_and_compare,
    metrics_delta,
    metrics_table,
    rank_experiments,
)

# Phase D0: typed experiment spec + hashing
# Legacy YAML config scaffold (keep for backward compat with existing tests)
from src.experiments.config import (
    ExperimentConfig,
    ExperimentSpec,
    experiment_hash,
    load_experiment_config,
    load_yaml,
)

# Phase D1: config I/O (load, validate, normalize)
from src.experiments.config_io import load_config, normalize_config, validate_config

# Phase D3: contracts, composite helpers
from src.experiments.contracts import (
    ARTEFACT_VERSION,
    DIAGNOSTICS_ARTEFACTS,
    ML_ARTEFACTS,
    REQUIRED_ARTEFACTS,
    check_artefact_dir,
    check_diagnostics_dir,
    check_ml_artefacts,
)

# Phase D1: factory layer
from src.experiments.factory import (
    UniverseSpec,
    ValidationConfig,
    available_strategies,
    build_experiment_spec,
    build_strategy,
    build_universe_spec,
    build_validation_config,
    build_validation_splits,
)
from src.experiments.ml_config import (
    FeatureEntry,
    FeatureSpec,
    LabelSpec,
    MLExperimentSpec,
    ModelSpec,
    SignalSpec,
    build_ml_experiment_spec,
    ml_experiment_hash,
    normalize_ml_config,
    validate_ml_config,
)

# Phase F3: ML experiment factory
from src.experiments.ml_factory import (
    build_feature_fns,
    build_label_fn,
    build_ml_strategy,
    build_model,
    build_signal_fn,
)

# Phase D1: orchestrator
from src.experiments.orchestrator import (
    ExperimentRun,
    format_run_summary,
    run_and_report,
    run_experiment_from_config,
)

# Phase D0: experiment registry
from src.experiments.registry import (
    ExperimentRegistry,
    latest_experiments,
    load_registry,
    query_registry,
    register_experiment,
)

# Phase C1/C2: strategy-based experiment persistence (keep for backward compat)
from src.experiments.results import ExperimentResult, load_experiment, save_experiment
from src.experiments.runner import ExperimentRunner

# Phase D0: filesystem-first tracking
from src.experiments.tracking import (
    ExperimentTracker,
    TrackingRun,
    load_run,
    save_run,
)

__all__ = [
    # Phase F3 — ML config (version "2" schema)
    "FeatureEntry",
    "FeatureSpec",
    "LabelSpec",
    "ModelSpec",
    "SignalSpec",
    "MLExperimentSpec",
    "ml_experiment_hash",
    "validate_ml_config",
    "normalize_ml_config",
    "build_ml_experiment_spec",
    # Phase F3 — ML factory
    "build_feature_fns",
    "build_label_fn",
    "build_model",
    "build_signal_fn",
    "build_ml_strategy",
    # Phase D1 — config I/O
    "load_config",
    "validate_config",
    "normalize_config",
    # Phase D1 — factory
    "UniverseSpec",
    "ValidationConfig",
    "build_strategy",
    "build_universe_spec",
    "build_validation_config",
    "build_experiment_spec",
    "build_validation_splits",
    "available_strategies",
    # Phase D1 — orchestrator
    "ExperimentRun",
    "run_experiment_from_config",
    # Phase D3 — contracts
    "ARTEFACT_VERSION",
    "ML_ARTEFACTS",
    "REQUIRED_ARTEFACTS",
    "DIAGNOSTICS_ARTEFACTS",
    "check_artefact_dir",
    "check_diagnostics_dir",
    "check_ml_artefacts",
    # Phase D3 — composite helpers
    "format_run_summary",
    "run_and_report",
    # Phase D0 — config
    "ExperimentSpec",
    "experiment_hash",
    # Phase D0 — tracking
    "save_run",
    "load_run",
    # Phase D0 — registry
    "ExperimentRegistry",
    "register_experiment",
    "load_registry",
    "query_registry",
    "latest_experiments",
    # Phase D0 — comparison
    "compare_experiments",
    "metrics_table",
    "rank_experiments",
    "load_and_compare",
    "metrics_delta",
    # Phase C1/C2 — persistence
    "ExperimentResult",
    "save_experiment",
    "load_experiment",
    # Legacy YAML scaffold
    "ExperimentConfig",
    "ExperimentRunner",
    "ExperimentTracker",
    "TrackingRun",
    "load_experiment_config",
    "load_yaml",
]
