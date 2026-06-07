"""ML experiment configuration specs for F3 — Reproducible ML Experiment Framework.

Defines typed configuration objects, validation, normalization, and deterministic
hashing for ML experiments using the version "2" config schema.

Version "2" = strict superset of version "1" base sections (universe, date_range,
validation, execution, output) with strategy replaced by four ML-specific sections:
    features, labels, model, signal

Existing version "1" configs are unaffected — the orchestrator routes on version.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Type registries
# ---------------------------------------------------------------------------

_VALID_FEATURE_TYPES: frozenset[str] = frozenset({
    "momentum",
    "rolling_volatility",
    "rolling_zscore",
    "sma",
    "ema",
    "compute_returns",
    "trend_strength",
    "downside_volatility",
    "vol_of_vol",
    "vol_percentile",
    "bollinger_distance",
    "rolling_skewness",
    "rolling_autocorrelation",
    # Phase H-1 additions
    "trend_persistence",
    "breakout_strength",
    "drawdown_distance",
    "vol_compression",
    "rolling_beta",
    "risk_adjusted_momentum",
})

_VALID_LABEL_TYPES: frozenset[str] = frozenset({
    "forward_returns",
    "binary_direction",
    "volatility_target",
    "ranking_target",
})

_VALID_MODEL_TYPES: frozenset[str] = frozenset({
    "LinearRegression",
    "RidgeRegression",
    "LassoRegression",
    "ElasticNetRegression",
    "LogisticRegression",
})

_VALID_SIGNAL_TYPES: frozenset[str] = frozenset({
    "sign",
    "threshold",
    "top_n",
    "long_short",
    "normalize",
})

_VALID_WEIGHTING_SCHEMES: frozenset[str] = frozenset({
    "equal_weight",
    "zscore_softmax",
    "confidence_weighted",
})

_VALID_PREDICTION_NORMALIZATIONS: frozenset[str] = frozenset({"none", "zscore"})

# Required params per feature type
_FEATURE_REQUIRED_PARAMS: dict[str, frozenset[str]] = {
    "momentum": frozenset({"lookback"}),
    "rolling_volatility": frozenset({"window"}),
    "rolling_zscore": frozenset({"window"}),
    "sma": frozenset({"window"}),
    "ema": frozenset({"span"}),
    "compute_returns": frozenset(),
    "trend_strength": frozenset({"window"}),
    "downside_volatility": frozenset({"window"}),
    "vol_of_vol": frozenset(),
    "vol_percentile": frozenset(),
    "bollinger_distance": frozenset({"window"}),
    "rolling_skewness": frozenset({"window"}),
    "rolling_autocorrelation": frozenset({"window"}),
    # Phase H-1 additions — market_ticker and vol_window have defaults
    "trend_persistence": frozenset({"window"}),
    "breakout_strength": frozenset({"window"}),
    "drawdown_distance": frozenset({"window"}),
    "vol_compression": frozenset({"short_window", "long_window"}),
    "rolling_beta": frozenset({"window"}),
    "risk_adjusted_momentum": frozenset({"mom_window"}),
}

_LABEL_REQUIRED_PARAMS: dict[str, frozenset[str]] = {
    "forward_returns": frozenset({"horizon"}),
    "binary_direction": frozenset({"horizon"}),
    "volatility_target": frozenset({"horizon"}),
    "ranking_target": frozenset({"horizon"}),
}

_SIGNAL_REQUIRED_PARAMS: dict[str, frozenset[str]] = {
    "sign": frozenset(),
    "threshold": frozenset(),
    "top_n": frozenset({"n"}),
    "long_short": frozenset({"n_long", "n_short"}),
    "normalize": frozenset(),
}

# Panel-only types (require multiple tickers; factory raises "not yet implemented")
PANEL_LABEL_TYPES: frozenset[str] = frozenset({"ranking_target"})
PANEL_SIGNAL_TYPES: frozenset[str] = frozenset({"top_n", "long_short", "normalize"})

_VALID_VALIDATION_TYPES: frozenset[str] = frozenset({"rolling", "expanding", "none"})

_OUTPUT_DEFAULTS: dict[str, Any] = {
    "base_dir": "results/experiments",
    "registry_path": "results/experiments/registry.json",
    "register": True,
    "save_plots": True,
}


# ---------------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FeatureEntry:
    """A single feature specification."""

    name: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.type, "params": dict(self.params)}


@dataclass
class FeatureSpec:
    """Collection of features for a single-ticker (single-asset) workflow."""

    ticker: str
    entries: list[FeatureEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "entries": [e.to_dict() for e in self.entries],
        }


@dataclass
class LabelSpec:
    """Label generation specification."""

    type: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "params": dict(self.params)}


@dataclass
class ModelSpec:
    """ML model specification."""

    type: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "params": dict(self.params)}


@dataclass
class SignalSpec:
    """Signal/weight conversion specification."""

    type: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "params": dict(self.params)}


@dataclass
class WeightingSpec:
    """Portfolio weighting policy within the selected asset basket."""

    scheme: str = "equal_weight"
    prediction_normalization: str = "none"
    temperature: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "prediction_normalization": self.prediction_normalization,
            "temperature": self.temperature,
        }

    def is_default(self) -> bool:
        """True iff this spec is behaviourally identical to the equal-weight baseline."""
        return self.scheme == "equal_weight"


@dataclass
class PortfolioConstructionSpec:
    """Portfolio construction policy: weighting scheme and parameters."""

    weighting: WeightingSpec = field(default_factory=WeightingSpec)

    def to_dict(self) -> dict[str, Any]:
        return {"weighting": self.weighting.to_dict()}


@dataclass
class MLExperimentSpec:
    """Full ML experiment specification for version "2" configs."""

    name: str
    universe: list[str]
    start_date: str
    end_date: str
    features: FeatureSpec
    labels: LabelSpec
    model: ModelSpec
    signal: SignalSpec
    portfolio_construction: PortfolioConstructionSpec = field(
        default_factory=PortfolioConstructionSpec
    )
    tags: list[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "universe": list(self.universe),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "features": self.features.to_dict(),
            "labels": self.labels.to_dict(),
            "model": self.model.to_dict(),
            "signal": self.signal.to_dict(),
            "portfolio_construction": self.portfolio_construction.to_dict(),
            "tags": list(self.tags),
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# Deterministic hashing
# ---------------------------------------------------------------------------


def ml_experiment_hash(spec: MLExperimentSpec) -> str:
    """Deterministic 12-char SHA-256 hash of an MLExperimentSpec.

    Excludes tags and description (metadata only).  All dict fields use
    sorted keys for cross-session determinism.

    Returns:
        First 12 hex characters of the SHA-256 digest.
    """
    payload = {
        "name": spec.name,
        "universe": sorted(spec.universe),
        "start_date": spec.start_date,
        "end_date": spec.end_date,
        "features": _json_safe(spec.features.to_dict()),
        "labels": _json_safe(spec.labels.to_dict()),
        "model": _json_safe(spec.model.to_dict()),
        "signal": _json_safe(spec.signal.to_dict()),
    }
    # Include portfolio_construction only when non-default to preserve hashes for
    # existing experiments that used equal-weight and predated this config block.
    if not spec.portfolio_construction.weighting.is_default():
        payload["portfolio_construction"] = _json_safe(
            spec.portfolio_construction.to_dict()
        )
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_ml_config(cfg: dict[str, Any]) -> None:
    """Validate a version "2" ML experiment config completely.

    Checks base sections (name, universe, date_range, optional validation/
    execution/output) and ML-specific sections (features, labels, model, signal).
    Raises ValueError on the first violation found.  Does not modify cfg.

    Args:
        cfg: Raw config dict with version == "2".

    Raises:
        ValueError: On any structural violation.
    """
    # --- Base sections ---
    name = cfg.get("name")
    if not name or not isinstance(name, str) or not name.strip():
        raise ValueError("Config requires a non-empty string 'name'.")

    universe = cfg.get("universe")
    if not isinstance(universe, dict):
        raise ValueError("'universe' must be a mapping with a 'tickers' key.")
    tickers = universe.get("tickers")
    if (
        not tickers
        or not isinstance(tickers, list)
        or not all(isinstance(t, str) and t.strip() for t in tickers)
    ):
        raise ValueError("'universe.tickers' must be a non-empty list of non-empty strings.")

    date_range = cfg.get("date_range")
    if not isinstance(date_range, dict):
        raise ValueError("'date_range' must be a mapping with 'start' and 'end' keys.")
    start = date_range.get("start")
    end = date_range.get("end")
    if not start or not end:
        raise ValueError("'date_range.start' and 'date_range.end' are required.")
    if not isinstance(start, str) or not isinstance(end, str):
        raise ValueError("'date_range.start' and 'date_range.end' must be ISO date strings.")
    if start >= end:
        raise ValueError(
            f"'date_range.start' ({start!r}) must be before 'date_range.end' ({end!r})."
        )

    # Optional: validation
    validation = cfg.get("validation")
    if validation is not None:
        if not isinstance(validation, dict):
            raise ValueError("'validation' must be a mapping.")
        vtype = validation.get("type", "none")
        if vtype not in _VALID_VALIDATION_TYPES:
            raise ValueError(
                f"Unknown validation type {vtype!r}. "
                f"Available: {sorted(_VALID_VALIDATION_TYPES)}"
            )
        vparams = validation.get("parameters", {})
        if vparams is not None and not isinstance(vparams, dict):
            raise ValueError("'validation.parameters' must be a mapping.")
        if vtype != "none" and isinstance(vparams, dict):
            for key in ("train_months", "test_months"):
                val = vparams.get(key)
                if val is not None and (not isinstance(val, int) or val <= 0):
                    raise ValueError(
                        f"'validation.parameters.{key}' must be a positive integer, "
                        f"got {val!r}."
                    )

    # Optional: execution
    execution = cfg.get("execution")
    if execution is not None:
        if not isinstance(execution, dict):
            raise ValueError("'execution' must be a mapping.")
        cost = execution.get("transaction_cost_bps")
        if cost is not None and (not isinstance(cost, (int, float)) or cost < 0):
            raise ValueError(
                f"'execution.transaction_cost_bps' must be a non-negative number, "
                f"got {cost!r}."
            )

    # Optional: output
    output = cfg.get("output")
    if output is not None and not isinstance(output, dict):
        raise ValueError("'output' must be a mapping.")

    # --- ML sections ---
    _validate_features(cfg)
    _validate_labels(cfg)
    _validate_model(cfg)
    _validate_signal(cfg)
    _validate_portfolio_construction(cfg)


def _validate_features(cfg: dict[str, Any]) -> None:
    features = cfg.get("features")
    if not isinstance(features, dict):
        raise ValueError("'features' must be a mapping with 'ticker' and 'entries' keys.")
    ticker = features.get("ticker")
    if not ticker or not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("'features.ticker' must be a non-empty string.")
    entries = features.get("entries")
    if not isinstance(entries, list) or len(entries) == 0:
        raise ValueError("'features.entries' must be a non-empty list.")
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"'features.entries[{i}]' must be a mapping.")
        ename = entry.get("name")
        if not ename or not isinstance(ename, str) or not ename.strip():
            raise ValueError(f"'features.entries[{i}].name' must be a non-empty string.")
        etype = entry.get("type")
        if etype not in _VALID_FEATURE_TYPES:
            raise ValueError(
                f"'features.entries[{i}].type' {etype!r} is unknown. "
                f"Available: {sorted(_VALID_FEATURE_TYPES)}"
            )
        eparams = entry.get("params", {})
        if not isinstance(eparams, dict):
            raise ValueError(f"'features.entries[{i}].params' must be a mapping.")
        required = _FEATURE_REQUIRED_PARAMS.get(etype, frozenset())
        for req in required:
            if req not in eparams:
                raise ValueError(
                    f"'features.entries[{i}].params' is missing required key "
                    f"'{req}' for feature type {etype!r}."
                )


def _validate_labels(cfg: dict[str, Any]) -> None:
    labels = cfg.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("'labels' must be a mapping with a 'type' key.")
    ltype = labels.get("type")
    if ltype not in _VALID_LABEL_TYPES:
        raise ValueError(
            f"'labels.type' {ltype!r} is unknown. "
            f"Available: {sorted(_VALID_LABEL_TYPES)}"
        )
    lparams = labels.get("params", {})
    if not isinstance(lparams, dict):
        raise ValueError("'labels.params' must be a mapping.")
    required = _LABEL_REQUIRED_PARAMS.get(ltype, frozenset())
    for req in required:
        if req not in lparams:
            raise ValueError(
                f"'labels.params' is missing required key '{req}' for label type {ltype!r}."
            )
    horizon = lparams.get("horizon")
    if horizon is not None and (not isinstance(horizon, int) or horizon < 1):
        raise ValueError(
            f"'labels.params.horizon' must be a positive integer, got {horizon!r}."
        )


def _validate_model(cfg: dict[str, Any]) -> None:
    model = cfg.get("model")
    if not isinstance(model, dict):
        raise ValueError("'model' must be a mapping with a 'type' key.")
    mtype = model.get("type")
    if mtype not in _VALID_MODEL_TYPES:
        raise ValueError(
            f"'model.type' {mtype!r} is unknown. "
            f"Available: {sorted(_VALID_MODEL_TYPES)}"
        )
    mparams = model.get("params", {})
    if mparams is not None and not isinstance(mparams, dict):
        raise ValueError("'model.params' must be a mapping.")


def _validate_portfolio_construction(cfg: dict[str, Any]) -> None:
    pc = cfg.get("portfolio_construction")
    if pc is None:
        return  # optional; defaults applied in normalize_ml_config
    if not isinstance(pc, dict):
        raise ValueError("'portfolio_construction' must be a mapping.")
    w = pc.get("weighting")
    if w is None:
        return
    if not isinstance(w, dict):
        raise ValueError("'portfolio_construction.weighting' must be a mapping.")
    scheme = w.get("scheme", "equal_weight")
    if scheme not in _VALID_WEIGHTING_SCHEMES:
        raise ValueError(
            f"'portfolio_construction.weighting.scheme' {scheme!r} is unknown. "
            f"Available: {sorted(_VALID_WEIGHTING_SCHEMES)}"
        )
    pred_norm = w.get("prediction_normalization", "none")
    if pred_norm not in _VALID_PREDICTION_NORMALIZATIONS:
        raise ValueError(
            f"'portfolio_construction.weighting.prediction_normalization' {pred_norm!r} "
            f"is unknown. Available: {sorted(_VALID_PREDICTION_NORMALIZATIONS)}"
        )
    temperature = w.get("temperature")
    if temperature is not None and (
        not isinstance(temperature, (int, float)) or float(temperature) <= 0
    ):
        raise ValueError(
            f"'portfolio_construction.weighting.temperature' must be a positive number "
            f"or null, got {temperature!r}."
        )


def _validate_signal(cfg: dict[str, Any]) -> None:
    signal = cfg.get("signal")
    if not isinstance(signal, dict):
        raise ValueError("'signal' must be a mapping with a 'type' key.")
    stype = signal.get("type")
    if stype not in _VALID_SIGNAL_TYPES:
        raise ValueError(
            f"'signal.type' {stype!r} is unknown. "
            f"Available: {sorted(_VALID_SIGNAL_TYPES)}"
        )
    sparams = signal.get("params", {})
    if sparams is not None and not isinstance(sparams, dict):
        raise ValueError("'signal.params' must be a mapping.")
    sparams = sparams or {}
    required = _SIGNAL_REQUIRED_PARAMS.get(stype, frozenset())
    for req in required:
        if req not in sparams:
            raise ValueError(
                f"'signal.params' is missing required key '{req}' for signal type {stype!r}."
            )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_ml_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a canonical copy of a version "2" config with all defaults filled.

    Handles the full version "2" config (base sections + ML sections).
    Never mutates the input.  Called instead of normalize_config() for v2.

    Args:
        cfg: Raw version "2" config dict, already validated by validate_ml_config().

    Returns:
        New dict with all defaults applied.
    """
    out = copy.deepcopy(cfg)

    # Version and metadata
    out["version"] = "2"
    out.setdefault("description", "")
    if not isinstance(out.get("tags"), list):
        out["tags"] = list(out.get("tags", []))

    # validation
    if "validation" not in out or not isinstance(out["validation"], dict):
        out["validation"] = {"type": "none", "parameters": {}}
    else:
        out["validation"].setdefault("type", "none")
        out["validation"].setdefault("parameters", {})
        vtype = out["validation"]["type"]
        vparams = out["validation"]["parameters"]
        if vtype in ("rolling", "expanding"):
            vparams.setdefault("gap_days", 0)
            if "step_months" not in vparams and "test_months" in vparams:
                vparams["step_months"] = vparams["test_months"]

    # execution
    if "execution" not in out or not isinstance(out["execution"], dict):
        out["execution"] = {"transaction_cost_bps": 0.0}
    else:
        out["execution"].setdefault("transaction_cost_bps", 0.0)
        out["execution"]["transaction_cost_bps"] = float(
            out["execution"]["transaction_cost_bps"]
        )

    # output
    if "output" not in out or not isinstance(out["output"], dict):
        out["output"] = dict(_OUTPUT_DEFAULTS)
    else:
        for key, default in _OUTPUT_DEFAULTS.items():
            out["output"].setdefault(key, default)

    # features entries — ensure params dict present
    if "features" in out and isinstance(out["features"], dict):
        for entry in out["features"].get("entries", []):
            if isinstance(entry, dict):
                entry.setdefault("params", {})

    # labels, model, signal — ensure params dict present
    for section in ("labels", "model", "signal"):
        if section in out and isinstance(out[section], dict):
            out[section].setdefault("params", {})

    # portfolio_construction — fill defaults so downstream code has a complete dict
    if "portfolio_construction" not in out or not isinstance(
        out["portfolio_construction"], dict
    ):
        out["portfolio_construction"] = {
            "weighting": {
                "scheme": "equal_weight",
                "prediction_normalization": "none",
                "temperature": None,
            }
        }
    else:
        w = out["portfolio_construction"].setdefault("weighting", {})
        w.setdefault("scheme", "equal_weight")
        w.setdefault("prediction_normalization", "none")
        w.setdefault("temperature", None)

    return out


# ---------------------------------------------------------------------------
# Spec construction from normalized config
# ---------------------------------------------------------------------------


def build_ml_experiment_spec(cfg: dict[str, Any]) -> MLExperimentSpec:
    """Construct an MLExperimentSpec from a fully normalized version "2" config.

    Args:
        cfg: Normalized version "2" config dict (output of normalize_ml_config()).

    Returns:
        MLExperimentSpec ready for hashing and registry storage.
    """
    features_cfg = cfg["features"]
    entries = [
        FeatureEntry(
            name=e["name"],
            type=e["type"],
            params=dict(e.get("params", {})),
        )
        for e in features_cfg.get("entries", [])
    ]
    feature_spec = FeatureSpec(ticker=features_cfg["ticker"], entries=entries)

    labels_cfg = cfg["labels"]
    label_spec = LabelSpec(
        type=labels_cfg["type"],
        params=dict(labels_cfg.get("params", {})),
    )

    model_cfg = cfg["model"]
    model_spec = ModelSpec(
        type=model_cfg["type"],
        params=dict(model_cfg.get("params", {})),
    )

    signal_cfg = cfg["signal"]
    signal_spec = SignalSpec(
        type=signal_cfg["type"],
        params=dict(signal_cfg.get("params", {})),
    )

    pc_cfg = cfg.get("portfolio_construction") or {}
    w_cfg = pc_cfg.get("weighting") or {}
    weighting_spec = WeightingSpec(
        scheme=str(w_cfg.get("scheme", "equal_weight")),
        prediction_normalization=str(w_cfg.get("prediction_normalization", "none")),
        temperature=w_cfg.get("temperature"),
    )
    portfolio_construction_spec = PortfolioConstructionSpec(weighting=weighting_spec)

    dr = cfg["date_range"]
    return MLExperimentSpec(
        name=cfg["name"],
        universe=list(cfg["universe"]["tickers"]),
        start_date=str(dr["start"]),
        end_date=str(dr["end"]),
        features=feature_spec,
        labels=label_spec,
        model=model_spec,
        signal=signal_spec,
        portfolio_construction=portfolio_construction_spec,
        tags=list(cfg.get("tags") or []),
        description=cfg.get("description", ""),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _json_safe(obj: Any) -> Any:
    """Recursively coerce non-JSON-native types to their JSON equivalents."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Public schema accessors (used by config-introspection and draft layers)
# ---------------------------------------------------------------------------


def get_valid_feature_types() -> frozenset[str]:
    """Return the authoritative set of valid feature type strings."""
    return _VALID_FEATURE_TYPES


def get_feature_required_params() -> dict[str, frozenset[str]]:
    """Return the required params mapping per feature type."""
    return dict(_FEATURE_REQUIRED_PARAMS)


def get_valid_model_types() -> frozenset[str]:
    """Return the authoritative set of valid model type strings."""
    return _VALID_MODEL_TYPES


def get_valid_label_types() -> frozenset[str]:
    """Return the authoritative set of valid label type strings."""
    return _VALID_LABEL_TYPES


def get_valid_signal_types() -> frozenset[str]:
    """Return the authoritative set of valid signal type strings."""
    return _VALID_SIGNAL_TYPES


def get_valid_weighting_schemes() -> frozenset[str]:
    """Return the authoritative set of valid weighting scheme strings."""
    return _VALID_WEIGHTING_SCHEMES


def get_valid_prediction_normalizations() -> frozenset[str]:
    """Return the authoritative set of valid prediction normalisation strings."""
    return _VALID_PREDICTION_NORMALIZATIONS
