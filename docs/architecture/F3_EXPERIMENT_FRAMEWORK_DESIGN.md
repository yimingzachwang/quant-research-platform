# F3 — Reproducible ML Experiment Framework: Design Specification

**Status:** DESIGN ONLY — no code, no scaffolding, no implementation.
**Grounded in:** Repository state as of 2026-05-23 (1001 tests passing).
**Author:** Architecture design session, Phase F3 pre-implementation review.

---

## 1. Executive Summary

### Purpose

F3 is the framework layer that makes the ML research stack reproducible, config-driven, and artifact-complete. Phases E0–F2 built a functionally correct ML stack with stable contracts, tested diagnostics, and a working walk-forward integration. What those phases deliberately deferred was the ability to describe an ML experiment in a config file, run it deterministically, and reconstruct every step of it from saved artifacts weeks later.

F3 bridges that gap: it extends the existing D1 config/orchestrator/artifact system to support ML experiments without replacing or duplicating any of the systems already in place.

### Problems Solved

1. **ML experiments have no config path.** The D1 orchestrator (`run_experiment_from_config`) only supports `MomentumRotation`, `EqualWeight`, and `BuyAndHold`. `MLStrategy` exists but cannot be described in a YAML file, built by the factory layer, or registered in the experiment registry.

2. **Feature and label construction is code-only.** A research notebook can construct a `SupervisedDataset` by calling `build_feature_matrix` and `forward_returns` with hand-written Python, but there is no config representation of which features or which label function was used, so a saved artifact has no record of how its data was constructed.

3. **ML predictions are never saved.** `save_run()` in `tracking.py` accepts a `predictions` DataFrame parameter (line 53), but the D1 orchestrator never passes it. The `diagnostics/` subdirectory created by `save_run()` is always empty.

4. **Provenance is incomplete for ML runs.** `metadata.json` does not record the dataset hash, feature spec, label spec, model parameters, or git commit. The `config_hash` field in the registry and provenance sidecar is `None` by design (see `report_builder.py` line 192: _"config_hash is not currently persisted in metadata.json; record None honestly"_).

5. **Registry identity is name-based, not content-based.** `ExperimentRegistry.register()` replaces any prior entry with the same `experiment_name`. Two runs with different hyperparameters under the same name silently overwrite each other. For ML research, where experiments vary along model type, feature set, horizon, and signal function, this is a reproducibility hazard.

### What F3 Does Not Attempt To Solve

- Hyperparameter search or tuning (grid search, Bayesian optimization, random search).
- Distributed or parallel experiment execution.
- Database-backed experiment storage or query systems (MLflow, W&B, Neptune).
- Deep learning model support or GPU infrastructure.
- Real-time or near-real-time inference.
- Model serving, deployment, or production endpoints.
- Automatic feature engineering or selection.
- Risk-model integration or live execution.

These are deferred not because they are unimportant, but because the current platform is a research backtesting system and none of those capabilities are presently needed to do the core work.

---

## 2. Current System Analysis

### 2.1 Experiment Infrastructure (D0–D3)

The experiment layer contains four overlapping tracks that represent progressive additions rather than a unified design:

**D0 (`results.py`, `tracking.py`):** `ExperimentResult` is a plain dataclass (experiment name, strategy name, parameters, metrics, weights, equity curve, returns, created_at). `save_experiment()` writes five files (`metadata.json`, `metrics.json`, three parquet files). `save_run()` adds `config.json`, optional `predictions.parquet`, `plots/`, and `diagnostics/`. The `diagnostics/` directory is always empty — it was created as a reserved slot with no current consumers.

**D1 (`config_io.py`, `factory.py`, `orchestrator.py`):** YAML/JSON config → validate → normalize → factory → run → save → register. The schema is hardcoded in `config_io.py`: `_VALID_STRATEGY_TYPES = {"MomentumRotation", "EqualWeight", "BuyAndHold"}`. The factory's `_STRATEGY_REGISTRY` mirrors this exactly. Both lists must be updated atomically. `MLStrategy` appears in neither.

**D2 (`report_builder.py`):** Read-only report generator. Reads `metadata.json`, `metrics.json`, discovers PNGs in `plots/`, produces markdown and HTML. Does not know about ML predictions, diagnostic artifacts, or the `diagnostics/` subdirectory.

**D3 (`contracts.py`):** `ARTEFACT_VERSION = "1"`, `REQUIRED_ARTEFACTS = ("metadata.json", "metrics.json")`, advisory `check_artefact_dir()`. Stable constants. No schema awareness beyond presence/absence of the two required files.

**Registry (`registry.py`):** Flat JSON, one entry per `experiment_name`. `register()` replaces on name match, so re-running overwrites. `_SUMMARY_METRICS` is hardcoded to four keys: `annualized_return`, `sharpe_ratio`, `max_drawdown`, `calmar_ratio`. An ML run that produces different metric keys (e.g., IC, directional accuracy) has no path to the summary.

### 2.2 Current ML Workflow

The ML stack is stable and tested but is used exclusively through hand-written Python, not config files. A researcher today writes:

```python
feature_fns = {
    "mom_21": lambda p: momentum(p["SPY"], 21),
    "vol_21": lambda p: rolling_volatility(p["SPY"], 21),
}
label_fn = lambda p: forward_returns(p["SPY"], horizon=5)
model = RidgeRegressionModel(alpha=1.0)
signal_fn = lambda preds: sign_signal(preds).rename("SPY").to_frame()
strategy = MLStrategy(model, feature_fns, label_fn, horizon=5, signal_fn=signal_fn)
```

There is no YAML equivalent of this construction. The feature callables are closures that capture state (the asset name "SPY", the lookback 21) in Python but leave no declarative record. Nothing in this workflow reaches `save_run()`, so no artifacts are written.

### 2.3 Artifact Persistence

Current artifacts from a D1 run:

```
results/experiments/momentum_rotation_d1/
    metadata.json               ← name, strategy, params, timestamp
    metrics.json                ← 6 scalar metrics
    equity_curve.parquet
    returns.parquet
    weights.parquet
    config.json                 ← ExperimentSpec JSON
    normalized_config.json      ← normalized YAML/JSON as JSON
    raw_config.yaml             ← verbatim copy of source config
    plots/
        equity_and_drawdown.png
        walk_forward_stitched.png
        split_sharpes.png
    diagnostics/                ← empty
```

Three things are absent that F3 must address:
1. No `predictions.parquet` (the hook exists in `save_run()` but is never used).
2. No ML diagnostic artifacts in `diagnostics/` (the directory is empty by design today).
3. No feature/label/model provenance record.

### 2.4 Orchestration Limitations

`run_experiment_from_config()` contains 12 numbered steps (documented in its module docstring). Steps 1–5 are generic (load, validate, normalize, factory, data). Steps 6–12 are strategy-specific. An ML experiment requires several additional steps between step 5 and step 6:
- Build feature matrix
- Build label series
- Construct `SupervisedDataset`
- Optionally run `run_walk_forward_predictions` (the raw prediction loop) separately from `run_walk_forward_validation` (the Strategy-level loop)

These are absent from the current orchestrator and cannot be expressed in the current config schema.

### 2.5 Reproducibility Gaps

| Gap | Location | Severity |
|-----|----------|----------|
| `MLStrategy` not in factory registry | `factory.py:33` | Blocking |
| No ML config sections (features, labels, model, signal) | `config_io.py` | Blocking |
| Registry replaces by name, not by content hash | `registry.py:108` | High |
| `config_hash` is always `None` in provenance sidecar | `report_builder.py:192` | High |
| Predictions never written during orchestrated runs | `orchestrator.py` | High |
| `diagnostics/` always empty | `tracking.py:88` | Medium |
| No git commit in provenance | `tracking.py` / `report_builder.py` | Medium |
| `dataset_hash()` hashes metadata only, not feature spec | `datasets.py:124` | Medium |
| No model artifact (fitted weights not persisted) | `src/ml/models/` | Medium |
| `_SUMMARY_METRICS` hardcoded to non-ML keys | `registry.py:36` | Low |

---

## 3. F3 Design Principles

### 3.1 No Duplicate Frameworks

The existing strategy/backtesting/walk-forward/artifact stack must not be duplicated. F3 extends the D1 pipeline, not replaces it. Concretely:
- `run_walk_forward_validation` is not superseded — it remains the authoritative out-of-sample evaluator for `MLStrategy`.
- `save_run()` is not superseded — it gains new callers but no competing save path.
- `ExperimentRegistry` is extended, not replaced — new fields are added to entries, not a new registry created.
- `generate_experiment_report()` is extended — new artefact types become discoverable sections, not a second report generator.

### 3.2 Deterministic Orchestration

Given the same config file, the same dataset state, and the same model hyperparameters, running the orchestrator twice must produce byte-identical artifacts (modulo floating-point randomness in model training, which should be made deterministic via seeding). Every step must be stateless: no hidden mutable globals, no session state between runs.

### 3.3 Explicit Lineage

Every artifact must carry enough metadata to trace back to its inputs. The lineage chain is:

```
git_commit → config_hash → dataset_hash → feature_hash → label_hash
→ model_params_hash → split_config → prediction_hash → metrics
```

No link in this chain may be implicit. If a link cannot be computed deterministically, the provenance record must record `null` explicitly (as `config_hash` does today) rather than silently omitting it.

### 3.4 Config Normalization

Config normalization must remain a pure function: same input → same output, no I/O, no side effects. The existing `normalize_config()` contract must hold for any new sections added in F3. Default filling must be explicit and documented in code, not inferred.

### 3.5 Composability

F3 types are composable building blocks, not monolithic specs. A `FeatureSpec` can be used without a `ModelSpec`. A `LabelSpec` can be used independently of a `ValidationSpec`. This composability allows incremental testing of each layer in isolation.

### 3.6 Artifact-Driven Reproducibility

A saved experiment directory is the ground truth for what happened. Reproducibility means: given the directory, a researcher can reconstruct every step by re-running the same config against the same data. The directory must contain enough information to do this without relying on memory, notebooks, or undocumented code.

### 3.7 Read-Only Reporting

The D2 reporting contract (`generate_experiment_report()` never recomputes, only reads) is permanent. F3 extends what `load_experiment_artefacts()` discovers, but never gives the report generator write access to experiment state.

### 3.8 Explicit Temporal Semantics

All F3 config nodes that involve time must express their temporal assumptions explicitly:
- `horizon` is always in trading periods (not calendar days).
- `shift(-horizon)` is the only valid forward label mechanism.
- Split boundaries must be expressed as dates, not integer indices.
- Feature lookbacks must be documented as periods on the `FeatureSpec`.

No temporal assumption may be inferred from defaults without a config-level declaration.

---

## 4. Proposed Experiment Object Model

This section defines the first-class typed concepts for F3. None of these exist in the codebase today in the form described here. They are proposed extensions, not implementations.

### 4.1 FeatureSpec

**Responsibility:** Declare which features to build, with what parameters, producing what column names.

**What it is NOT:** A feature registry, a DAG, or a dynamic feature store. It is a simple declarative list of feature name → callable key, with enough metadata to hash the construction.

**Proposed fields:**
```
name: str                    # human-readable label, e.g. "momentum_vol_21"
features: list[FeatureEntry] # ordered list (order matters for column naming)
version: str                 # "1" — bump when semantics change
```

Where `FeatureEntry` contains:
```
key: str                     # becomes build_feature_matrix key
type: str                    # maps to a registered feature function
parameters: dict[str, Any]   # lookback, etc.
```

**Ownership boundary:** `FeatureSpec` describes construction intent. The Python callable that executes the computation is resolved by a registry (see §5 config schema). `FeatureSpec` itself does not hold a reference to any callable — it is serializable to JSON.

**Serialization:** Full round-trip to/from JSON. All parameter values must be JSON-native types.

**Lineage:** `feature_hash(spec: FeatureSpec) -> str` — SHA-256 of the normalized JSON representation (sorted keys), 12 hex characters, consistent with `experiment_hash()` and `dataset_hash()` conventions.

**Relationship to existing code:** `build_feature_matrix()` in `src/ml/feature_matrix.py` accepts `dict[str, Callable]`. The factory layer resolves `FeatureSpec` entries into that dict. `FeatureSpec` sits above `build_feature_matrix()` — it is a config concept, not a computation concept.

---

### 4.2 LabelSpec

**Responsibility:** Declare which label function to use, with what horizon and parameters.

**What it is NOT:** A label generation library. It is a pointer into the existing `src/ml/labels.py` functions.

**Proposed fields:**
```
name: str                    # human-readable, e.g. "fwd_return_5d"
type: str                    # "forward_returns" | "binary_direction_label"
                             # | "volatility_target" | "ranking_target"
horizon: int                 # trading periods — stored here, not inferred
parameters: dict[str, Any]   # any additional params (currently none needed)
```

**Key invariant:** `horizon` on `LabelSpec` must equal `horizon` passed to `build_supervised_dataset()`. The factory layer is responsible for this consistency — it reads `horizon` from `LabelSpec` once and passes it to both `label_fn` and `build_supervised_dataset`. No two code paths may independently hold the horizon value.

**Lineage:** `label_hash(spec: LabelSpec) -> str` — SHA-256 of normalized JSON. Both `type` and `horizon` are included in the hash payload. Changing the horizon changes the hash — this is correct and intentional.

**Relationship to existing code:** `src/ml/labels.py` contains `forward_returns`, `binary_direction_label`, `volatility_target`, `ranking_target`. `LabelSpec.type` maps directly to these function names. The factory resolves `LabelSpec.type` → callable at run time. No new label functions are introduced by F3.

---

### 4.3 ModelSpec

**Responsibility:** Declare which model class to instantiate, with what hyperparameters.

**What it is NOT:** A hyperparameter search space, a model registry, or a training pipeline.

**Proposed fields:**
```
type: str                    # "LinearRegression" | "Ridge" | "Lasso"
                             # | "ElasticNet" | "LogisticRegression"
parameters: dict[str, Any]   # hyperparameters, e.g. {"alpha": 1.0}
random_seed: int | None      # for reproducibility in stochastic models
name: str | None             # optional human label; defaults to type + params
```

**Ownership boundary:** `ModelSpec` knows nothing about features or labels. It describes only model construction. The factory resolves `ModelSpec` → instantiated `BaseMLModel` object.

**The model itself is not serialized.** A fitted model's internal state (sklearn estimator weights) is not persisted in F3. Reproducibility is achieved by re-fitting from the config, not by pickling. This is a deliberate choice: Python pickle is not version-stable, and sklearn model serialization has historically caused reproducibility problems across library versions.

**Lineage:** `model_hash(spec: ModelSpec) -> str` — SHA-256 of `type` + sorted parameters + `random_seed`. The `name` field is excluded from the hash (metadata only), consistent with how `experiment_hash()` excludes `tags` and `description`.

**Relationship to existing code:** `src/ml/models/` contains `LinearRegressionModel`, `RidgeRegressionModel`, `LassoRegressionModel`, `ElasticNetRegressionModel`, `LogisticRegressionModel`. All satisfy `BaseMLModel`. `ModelSpec.type` maps to these class names. The factory's `_MODEL_REGISTRY` mirrors this mapping, analogous to the existing `_STRATEGY_REGISTRY`.

---

### 4.4 SignalSpec

**Responsibility:** Declare which signal translation function converts `PredictionSeries` to portfolio weights, and with what parameters.

**What it is NOT:** A signal library, a position-sizing system, or a risk model.

**Proposed fields:**
```
type: str                    # "sign" | "threshold" | "top_n"
                             # | "long_short" | "normalize"
parameters: dict[str, Any]   # e.g. {"threshold": 0.0} or {"n": 3}
```

**Ownership boundary:** `SignalSpec` describes the conversion from predictions to weights. It does not describe the model that produced the predictions or the strategy that applies them. The factory resolves `SignalSpec` → a `Callable[[PredictionSeries], pd.DataFrame]` for injection into `MLStrategy`.

**Important constraint:** `sign_signal` and `threshold_signal` require `predictions.values` to be `pd.Series` (single-asset). `top_n_weights`, `long_short_weights`, and `normalize_to_weights` require `pd.DataFrame` (panel). The config schema must enforce consistency between `LabelSpec.type`, the universe size, and `SignalSpec.type`. Specifically:
- Panel signal functions (`top_n`, `long_short`, `normalize`) are only valid when the universe contains multiple assets.
- Series signal functions (`sign`, `threshold`) are only valid when the universe contains exactly one asset, or when the model produces a single-asset prediction.

This constraint cannot be checked purely within `SignalSpec` — it requires cross-spec validation at the orchestrator level.

**Lineage:** `signal_hash(spec: SignalSpec) -> str` — SHA-256 of type + sorted parameters.

**Relationship to existing code:** `src/ml/signals/prediction.py` contains all five functions. `SignalSpec.type` maps to them directly.

---

### 4.5 ValidationSpec

**Responsibility:** Declare the walk-forward split configuration.

**What it is NOT:** A new validation system. This is a thin wrapper over the existing `ValidationConfig` in `factory.py`.

**This concept already exists** as `ValidationConfig` in `factory.py`. F3 promotes it to a first-class named object in the config schema. The existing fields (`type`, `parameters.train_months`, `parameters.test_months`, `parameters.step_months`, `parameters.gap_days`) are preserved exactly. No new validation logic is introduced.

**Proposed rename:** `ValidationConfig` → `ValidationSpec` for naming consistency with the other spec types. The existing `build_validation_config()` and `build_validation_splits()` factory functions remain unchanged and accept the renamed object.

**Lineage:** `validation_hash(spec: ValidationSpec) -> str` — SHA-256 of type + sorted parameters.

---

### 4.6 MLExperimentSpec

**Responsibility:** The top-level config object for an ML experiment. Extends `ExperimentSpec` rather than replacing it.

**Design choice — extension vs. replacement:**

`ExperimentSpec` in `config.py` covers: `experiment_name`, `strategy_name`, `universe`, `start_date`, `end_date`, `rebalance_frequency`, `parameters`, `tags`, `description`. For an ML experiment, `strategy_name` would be `"MLStrategy"` and `parameters` would be insufficient to hold the full ML specification.

Two design options:
- **Option A (Composition):** A new `MLExperimentSpec` dataclass that holds an `ExperimentSpec` plus the four new specs (`FeatureSpec`, `LabelSpec`, `ModelSpec`, `SignalSpec`).
- **Option B (Subclass):** `MLExperimentSpec` extends `ExperimentSpec` with additional fields.

**Recommendation: Option A (Composition)**

Composition is preferred because `ExperimentSpec` is a frozen-like dataclass used for hashing and registry storage. Subclassing it would require changing `experiment_hash()`, `to_dict()`, `from_dict()`, and `load_config()` — a broad change. Composition wraps without modifying: the `ExperimentSpec` inside `MLExperimentSpec` continues to describe the orchestration-level identity, while the ML-specific specs describe the research configuration.

**Proposed fields:**
```
base: ExperimentSpec         # carries name, universe, dates, tags
feature_spec: FeatureSpec
label_spec: LabelSpec
model_spec: ModelSpec
signal_spec: SignalSpec
validation_spec: ValidationSpec
```

**Lineage:** `ml_experiment_hash(spec: MLExperimentSpec) -> str` — SHA-256 of `experiment_hash(spec.base)` + `feature_hash` + `label_hash` + `model_hash` + `signal_hash` + `validation_hash`. Changes to any component change the composite hash. This is the artifact-level identity for ML experiment runs.

**Serialization:** Full round-trip JSON. Each sub-spec serializes independently. The combined JSON is the canonical config artifact saved alongside ML experiment results.

---

## 5. Proposed Config Schema Philosophy

### 5.1 Current Schema (D1, version "1")

The D1 schema has seven top-level keys:
```yaml
version: "1"
name: string
description: string
tags: list[string]
universe:
  tickers: list[string]
date_range:
  start: ISO-date
  end: ISO-date
strategy:
  type: "MomentumRotation" | "EqualWeight" | "BuyAndHold"
  parameters: dict
validation:
  type: "rolling" | "expanding" | "none"
  parameters: dict
execution:
  transaction_cost_bps: float
output:
  base_dir: string
  registry_path: string
  register: bool
  save_plots: bool
```

### 5.2 Proposed F3 ML Config Extension (version "2")

A new schema version "2" adds three required sections for ML experiments and one optional extension to `output`. Non-ML experiments continue to use version "1" without modification.

**The existing `validate_config()` and `normalize_config()` must continue to validate version "1" configs without change.** Version "2" configs are handled by a new parallel path: `validate_ml_config()` and `normalize_ml_config()`.

---

#### Example: Single-Asset ML Experiment

```yaml
version: "2"
name: ridge_spy_fwd5_v1
description: "Ridge regression on SPY with momentum and vol features, 5-day forward return label."
tags:
  - ml
  - single-asset
  - regression

universe:
  tickers:
    - SPY

date_range:
  start: "2015-01-01"
  end: "2024-12-31"

features:
  name: momentum_vol_21
  version: "1"
  entries:
    - key: mom_21
      type: momentum
      parameters:
        lookback: 21
    - key: vol_21
      type: rolling_volatility
      parameters:
        lookback: 21

labels:
  name: fwd_return_5d
  type: forward_returns
  horizon: 5

model:
  type: Ridge
  parameters:
    alpha: 1.0
  random_seed: 42

signal:
  type: sign
  parameters: {}

validation:
  type: rolling
  parameters:
    train_months: 36
    test_months: 12
    gap_days: 5

execution:
  transaction_cost_bps: 5.0

output:
  base_dir: results/experiments
  registry_path: results/experiments/registry.json
  register: true
  save_plots: true
  save_predictions: true
  save_diagnostics: true
```

---

#### Example: Panel (Cross-Sectional) ML Experiment

```yaml
version: "2"
name: ridge_etf_panel_top3
description: "Cross-sectional ridge regression, long top-3 ETFs by predicted 21-day return."
tags:
  - ml
  - panel
  - long-only

universe:
  tickers:
    - SPY
    - QQQ
    - IWM
    - TLT
    - GLD
    - XLF
    - XLK

date_range:
  start: "2016-01-01"
  end: "2024-12-31"

features:
  name: cross_section_mom_vol
  version: "1"
  entries:
    - key: mom_63
      type: momentum
      parameters:
        lookback: 63
    - key: vol_21
      type: rolling_volatility
      parameters:
        lookback: 21

labels:
  name: rank_target_21d
  type: ranking_target
  horizon: 21

model:
  type: Ridge
  parameters:
    alpha: 0.5
  random_seed: 42

signal:
  type: top_n
  parameters:
    n: 3

validation:
  type: rolling
  parameters:
    train_months: 48
    test_months: 12
    gap_days: 5

execution:
  transaction_cost_bps: 5.0

output:
  base_dir: results/experiments
  save_plots: true
  save_predictions: true
  save_diagnostics: true
```

---

#### Example: Walk-Forward Predictions Only (No Backtest)

```yaml
version: "2"
name: logistic_spy_direction_wfp
description: "Walk-forward prediction run only — no backtest, just out-of-sample predictions."
tags:
  - ml
  - predictions-only

universe:
  tickers:
    - SPY

date_range:
  start: "2018-01-01"
  end: "2024-12-31"

features:
  name: sma_features
  version: "1"
  entries:
    - key: mom_5
      type: momentum
      parameters:
        lookback: 5
    - key: mom_21
      type: momentum
      parameters:
        lookback: 21

labels:
  name: direction_5d
  type: binary_direction_label
  horizon: 5

model:
  type: LogisticRegression
  parameters:
    C: 1.0
    max_iter: 1000
  random_seed: 0

signal:
  type: threshold
  parameters:
    threshold: 0.5

validation:
  type: rolling
  parameters:
    train_months: 24
    test_months: 6

execution:
  transaction_cost_bps: 5.0

output:
  save_predictions: true
  save_diagnostics: true
  save_plots: false
  run_backtest: false     # skip backtest; save predictions only
```

---

### 5.3 Normalization Rules for Version "2"

`normalize_ml_config()` must follow the same pattern as `normalize_config()`:
- Never mutates the input dict.
- `copy.deepcopy()` at the start.
- Fills default fields with documented, explicit defaults.
- Fills `features.version` to `"1"` if absent.
- Fills `labels.parameters` to `{}` if absent.
- Fills `model.random_seed` to `null` if absent.
- Fills `signal.parameters` to `{}` if absent.
- Inherits all version "1" normalization for `validation`, `execution`, `output`.
- Adds new `output` defaults: `save_predictions: true`, `save_diagnostics: true`, `run_backtest: true`.

### 5.4 Validation Rules for Version "2"

`validate_ml_config()` must validate:
- All version "1" rules still apply to shared sections.
- `features.entries` is a non-empty list.
- Each feature entry has `key`, `type`, and `parameters`.
- `features.entries[*].type` must be in `_VALID_FEATURE_TYPES` (registered feature function names).
- `labels.type` must be in `_VALID_LABEL_TYPES = {"forward_returns", "binary_direction_label", "volatility_target", "ranking_target"}`.
- `labels.horizon` must be a positive integer.
- `model.type` must be in `_VALID_MODEL_TYPES`.
- `signal.type` must be in `_VALID_SIGNAL_TYPES`.
- Cross-spec consistency: panel signal types (`top_n`, `long_short`, `normalize`) require `len(universe.tickers) > 1` and `labels.type == "ranking_target"`.
- Series signal types (`sign`, `threshold`) require `len(universe.tickers) == 1` OR the model produces a single-asset prediction (which requires `universe.tickers` to be a single-element list in the current linear/logistic models).

### 5.5 Versioning and Schema Evolution

**Version "1" is frozen.** No new fields may be added to it. Backward compatibility is maintained by continuing to parse version "1" through the existing `validate_config()` / `normalize_config()` path. This is a hard constraint — changes to version "1" parsing would break existing saved configs.

**Version "2" introduces ML sections.** It is a superset of version "1" in terms of shared sections (`universe`, `date_range`, `execution`, `output`, `validation`). Non-ML experiments remain on version "1".

**Future versions** should be introduced only when a breaking schema change is necessary. A breaking change is one that changes the semantics of an existing field, not one that adds new optional fields. New optional fields with defaults are non-breaking and can be added to version "2" without bumping to version "3".

**`ARTEFACT_VERSION` in `contracts.py`** must be bumped from `"1"` to `"2"` when F3 introduces ML-specific artifact files into the standard save layout. This signals to `check_artefact_dir()` that a new layout is expected.

### 5.6 Tradeoffs

**YAML expressiveness vs. type safety:** YAML is human-readable and allows inline comments, but Python callables cannot be serialized to YAML. The design resolves this by using type strings (`"Ridge"`, `"forward_returns"`, `"top_n"`) resolved by factory registries. This is the same pattern used successfully in D1 for `strategy.type`. The tradeoff is that registries must be kept in sync with actual implementations — but this is already true for the strategy registry and has been manageable.

**Feature function parameters as dicts:** Feature parameters (`{"lookback": 21}`) are stored as plain JSON dicts. This means no type checking of parameter values at config load time — a lookback of `"twenty-one"` would not be caught until the feature callable is invoked. Mitigation: add per-feature-type parameter validation in `validate_ml_config()` for known feature types.

**Panel vs. single-asset complexity:** The cross-spec consistency check (signal type vs. universe size) is awkward to validate purely in config terms. The simpler alternative is to defer this check to the factory layer (which has access to all specs simultaneously) rather than to the config validator (which validates each section in isolation). This is acceptable — validation is not all-or-nothing.

---

## 6. Experiment Lineage and Provenance

### 6.1 What Should Be Hashed

The principle is: **hash what uniquely identifies the experiment's computational identity, not what describes its presentation.** Concretely:

| Component | Hash inputs | Excluded from hash |
|-----------|-------------|-------------------|
| `ExperimentSpec` | name, strategy_name, universe (sorted), start_date, end_date, rebalance_frequency, parameters (sorted keys) | tags, description |
| `FeatureSpec` | version, sorted list of (key, type, sorted parameters) | name |
| `LabelSpec` | type, horizon, sorted parameters | name |
| `ModelSpec` | type, sorted parameters, random_seed | human-readable name |
| `SignalSpec` | type, sorted parameters | — |
| `ValidationSpec` | type, sorted parameters | — |
| `MLExperimentSpec` | all six component hashes concatenated in fixed order | timestamps, output paths |

**Why sort parameters?** Parameter dict order is not semantically meaningful and varies between YAML parsers and Python versions. SHA-256 of `json.dumps(..., sort_keys=True)` is the existing convention in `experiment_hash()` and `dataset_hash()` and must be preserved.

**Why exclude `name` and `tags` and `description`?** These are metadata annotations. Two experiments with identical computational specifications but different names, tags, or descriptions are computationally identical. Hashing them would cause two names for the same run to appear as different experiments, defeating reproducibility detection.

### 6.2 What Should NOT Be Hashed

- **Runtime timestamps** (`created_at`): change every run, defeat hash stability.
- **Output paths** (`base_dir`, `registry_path`): filesystem-local, not computational.
- **Git commit hash**: important for provenance but not part of the experiment specification — the same code may be committed multiple times, or a run may be done in a dirty tree. Record it separately in provenance metadata; do not include it in the experiment hash.
- **Feature values** (the actual data in `SupervisedDataset.X`): `dataset_hash()` deliberately hashes metadata only. Re-hashing data values would be expensive and is not reproducibility-relevant — the same config applied to the same registered dataset produces the same data. Record the dataset manifest hash instead.
- **Fitted model weights**: not hashed; not persisted. The model is re-fit from config at reproduction time.

### 6.3 Provenance Metadata Record

Each ML experiment run should write a `ml_provenance.json` alongside `metadata.json`:

```json
{
  "artefact_version": "2",
  "ml_experiment_hash": "abc123def456",
  "feature_hash": "abc123def456",
  "label_hash": "def456789abc",
  "model_hash": "789abc123def",
  "signal_hash": "123def456789",
  "validation_hash": "456789abcdef",
  "dataset_hash": "6789abcdef12",
  "git_commit": "a1b2c3d4e5f6...",
  "git_dirty": true,
  "created_at": "2026-05-23T12:00:00+00:00",
  "python_version": "3.11.x",
  "sklearn_version": "1.x.x",
  "pandas_version": "2.x.x"
}
```

**`git_commit`:** obtained via `subprocess.run(["git", "rev-parse", "HEAD"], ...)`. If git is unavailable or the directory is not a git repository, record `null` explicitly.

**`git_dirty`:** `git status --porcelain` non-empty output → `true`. Dirty trees are common in research; recording the flag is more useful than blocking the run.

**Library versions:** `sklearn.__version__`, `pd.__version__`, `np.__version__`. These affect model outputs even with the same random seed, so they are essential provenance data.

**`dataset_hash`:** computed from the `SupervisedDataset` built during the run, using the existing `dataset_hash()` function. This records exactly which aligned (X, y) pair was used for training.

### 6.4 Registry Entry Extension

The existing registry entry format stores `config_hash` (currently empty string or None). For F3 ML runs, this field is populated with `ml_experiment_hash`. The `_SUMMARY_METRICS` list should be extended or made dynamic:

```python
_ML_SUMMARY_METRICS = (
    "annualized_return", "sharpe_ratio", "max_drawdown",
    "calmar_ratio", "mean_ic", "directional_accuracy",
)
```

`mean_ic` and `directional_accuracy` may not be present for every run; the registry entry builder should use `.get()` with silent skipping rather than raising on missing keys — the same approach used for existing metrics.

### 6.5 Artifact Versioning

`ARTEFACT_VERSION` in `contracts.py` must be bumped to `"2"` when F3 is implemented. `REQUIRED_ARTEFACTS` should remain `("metadata.json", "metrics.json")` — these are the minimal required files for any experiment type. ML-specific files (`ml_provenance.json`, `predictions.parquet`) are present only in ML runs and checked separately by a new `check_ml_artefact_dir()` advisory function.

---

## 7. Artifact Layout Design

### 7.1 Current Layout (D1, ARTEFACT_VERSION "1")

```
results/experiments/<experiment_name>/
    metadata.json
    metrics.json
    config.json
    normalized_config.json
    raw_config.yaml
    equity_curve.parquet
    returns.parquet
    weights.parquet
    plots/
        equity_and_drawdown.png
        walk_forward_stitched.png
        split_sharpes.png
    diagnostics/                    ← empty today
```

### 7.2 Proposed ML Experiment Layout (ARTEFACT_VERSION "2")

```
results/experiments/<experiment_name>/
    metadata.json                   ← unchanged (name, strategy, params, timestamp)
    metrics.json                    ← unchanged + optional ML metrics (mean_ic, etc.)
    ml_provenance.json              ← NEW: all hashes, git commit, lib versions
    config.json                     ← ExperimentSpec base (unchanged format)
    ml_config.json                  ← NEW: full MLExperimentSpec as JSON
    normalized_config.json          ← normalized version "2" config
    raw_config.yaml                 ← verbatim copy of source config (unchanged)
    equity_curve.parquet            ← unchanged (absent if run_backtest: false)
    returns.parquet                 ← unchanged (absent if run_backtest: false)
    weights.parquet                 ← unchanged (absent if run_backtest: false)
    predictions.parquet             ← NEW: concatenated walk-forward predictions
    diagnostics/
        ic_series.parquet           ← NEW: information_coefficient() output
        turnover_series.parquet     ← NEW: signal_turnover() output
        split_metrics.parquet       ← NEW: split_metric_table() output (if wf run)
        coefficient_stability.parquet ← NEW: coefficient_stability() output (if linear)
    plots/
        equity_and_drawdown.png     ← existing (if run_backtest)
        walk_forward_stitched.png   ← existing (if walk-forward)
        split_sharpes.png           ← existing (if walk-forward)
        prediction_vs_actual.png    ← NEW: plot_prediction_vs_actual()
        prediction_distribution.png ← NEW: plot_prediction_distribution()
        ic_series.png               ← NEW: plot_information_coefficient()
        split_metric_stability.png  ← NEW: plot_split_metric_stability()
        signal_turnover.png         ← NEW: plot_signal_turnover()
```

### 7.3 Design Rationale for Diagnostic Sub-Layout

The `diagnostics/` subdirectory was created empty by `save_run()` (line 88 of `tracking.py`) but never populated. F3 populates it with the outputs of `src/ml/diagnostics/` functions — all of which produce `pd.Series` or `pd.DataFrame` objects that are directly serializable to parquet.

The diagnostic artifacts are written in parquet rather than JSON because:
- They are time-indexed Series/DataFrames that benefit from parquet's column storage.
- They are consumed programmatically (by future comparison tools and the report generator), not by humans directly.
- Parquet roundtrip is lossless for float64 data (unlike JSON float serialization).

The diagnostic artifacts are optional: a run with `save_diagnostics: false` in the config omits them. `check_artefact_dir()` does not check for their presence.

### 7.4 Predictions Artifact Design

`predictions.parquet` is the output of `concatenate_predictions(wf)` — the stitched out-of-sample prediction series from all walk-forward test windows. It is a `pd.Series` (single-asset) or `pd.DataFrame` (panel) with a `DatetimeIndex`.

The save format must preserve the index name and dtype. For a `pd.Series` named `"target"`, save as a single-column DataFrame with that column name. For a `pd.DataFrame`, save directly. Load by detecting the shape.

This file already has a slot in `save_run()` (the optional `predictions` parameter), but the D1 orchestrator never passes it. F3 changes the orchestrator to compute and pass it.

### 7.5 Multiple Runs Under the Same Name

The current `save_experiment()` overwrites files on name collision (line 79 of `results.py`: `out.mkdir(parents=True, exist_ok=True)` followed by explicit file writes). This is intentional for D1 research experiments but problematic for ML research where a researcher runs dozens of parameter variants.

**Proposed approach for F3:** Add an optional `run_id` suffix mechanism. When `output.deduplicate: true` is set in the config, the output directory is `<experiment_name>_<ml_experiment_hash[:8]>/` instead of `<experiment_name>/`. This produces a new directory for each distinct configuration, while the registry uses the full `ml_experiment_hash` as the deduplication key rather than `experiment_name`.

The default (`deduplicate: false`) preserves the current behavior for compatibility.

---

## 8. ML Workflow Integration

### 8.1 How F3 Extends the D1 Orchestrator

The D1 orchestrator's 12 steps (documented in `orchestrator.py`) remain unchanged for version "1" configs. For version "2" configs, the orchestrator follows an extended sequence:

```
1.  load_config(path)               → raw dict (unchanged)
2.  validate_ml_config(raw)         → raises on ML schema errors (new)
3.  normalize_ml_config(raw)        → fills ML defaults (new)
4a. build_ml_experiment_spec(norm)  → MLExperimentSpec (new)
4b. build_feature_fns(spec.features)→ dict[str, Callable] (new)
4c. build_label_fn(spec.labels)     → Callable (new)
4d. build_model(spec.model)         → BaseMLModel (new)
4e. build_signal_fn(spec.signal)    → Callable (new)
4f. build_ml_strategy(...)          → MLStrategy (new, wraps existing)
4g. build_validation_splits(...)    → list[TimeSplit] (unchanged)
5.  load_universe / align_prices    → prices DataFrame (unchanged)
6.  [optional] run_strategy         → StrategyResult (unchanged, if run_backtest)
7.  run_walk_forward_validation     → WalkForwardResult (unchanged)
7b. run_walk_forward_predictions    → WalkForwardPredictions (new step, uses ml.pipelines)
7c. concatenate_predictions         → PredictionSeries (new step)
8.  build_ExperimentResult          → ExperimentResult (unchanged, add ML metrics)
9.  compute_ml_diagnostics          → ic_series, turnover, split_metrics (new)
10. build_plots                     → dict of figures (extended with ML plots)
11. save_run + write_ml_artifacts   → persists all artifacts (extended)
12. write_dual_configs + ml_provenance → config + provenance artifacts (extended)
13. registry.register               → with ml_experiment_hash (extended)
```

**Key integration point:** Step 7 (`run_walk_forward_validation`) runs `MLStrategy` through the existing walk-forward runner unchanged. Step 7b runs the raw prediction pipeline (`run_walk_forward_predictions` from `src/ml/pipelines/walk_forward.py`) on the same splits. Both pipelines operate on the same data; they are not parallel implementations — they serve different purposes. The strategy-level runner produces `WalkForwardResult` (per-split backtested performance). The prediction-level runner produces `WalkForwardPredictions` (per-split raw predictions for diagnostic use).

### 8.2 SupervisedDataset Integration

`SupervisedDataset` is in-memory only and is not persisted (by design — see `datasets.py` docstring). F3 does not change this. The dataset is reconstructed at run time from config. The `dataset_hash()` output is written to `ml_provenance.json` as a structural identity check, not as a persistence mechanism.

The factory layer constructs the `SupervisedDataset` at orchestrator step 4 (before data slicing). This dataset is passed into `MLStrategy.fit()` indirectly: `MLStrategy.fit(train_prices)` internally calls `build_feature_matrix` and the `label_fn`, which are the same callables used to build the dataset. There is no redundant computation — the feature and label functions are the shared construction pathway.

### 8.3 BaseMLModel Integration

`BaseMLModel` is a `runtime_checkable` Protocol. F3 does not change it. The factory builds a concrete model instance from `ModelSpec.type` → registered class → instantiate with `ModelSpec.parameters`. The same model instance is passed to both `MLStrategy` (for the strategy-level walk-forward) and to `run_walk_forward_predictions` (for the raw prediction pipeline) if both are run.

**Warning:** Running both pipelines with the same model instance means that refitting one mutates the shared state, causing the second pipeline's predictions to be based on the wrong training data. The orchestrator must use two separate model instances (or run them sequentially, re-instantiating the model between them). This is a subtle correctness constraint that must be documented in the orchestrator code.

### 8.4 Validation Splits Integration

`build_validation_splits(val_config, prices.index)` in `factory.py` is unchanged. The same `list[TimeSplit]` is passed to both `run_walk_forward_validation` (existing) and `run_walk_forward_predictions` (new). The split semantics are identical — there is no need for separate split configurations for the strategy runner and the prediction runner.

### 8.5 Diagnostics Integration

The F3 diagnostics step (step 9 above) calls:
- `information_coefficient(actual, predicted)` using the concatenated predictions and the label series for the corresponding test periods.
- `signal_turnover(weights)` using the walk-forward weights from `WalkForwardResult`.
- `split_metric_table(wf_result)` using the `WalkForwardResult` from the strategy runner.
- `coefficient_stability(coefficients)` — only for linear models that expose fitted coefficients. This requires adding an optional `get_coefficients() -> pd.Series` method to the linear model classes, or checking `hasattr(model._model, 'coef_')` at the orchestrator level.

All diagnostic functions are pure. No new computation is introduced by calling them in the orchestrator — they are already tested and stable.

### 8.6 Portfolio Backtest Integration

`run_portfolio_backtest()` applies `weights.shift(1)` before computing returns. This lag is applied to the weights produced by `MLStrategy.generate_weights()`, which already handles the warm-up period correctly by returning zeros for early rows. F3 introduces no changes to the backtesting stack. The integration point is identical to D1: the orchestrator passes `MLStrategy` to `run_strategy()`, which calls `run_portfolio_backtest()` internally.

---

## 9. Reporting and Visualization Integration

### 9.1 Extending `load_experiment_artefacts()`

The current `load_experiment_artefacts()` function discovers PNGs in `plots/` and loads `metadata.json`, `metrics.json`, and the config file. For F3, it should additionally:
- Check for `predictions.parquet` and, if present, load or flag it.
- Check for `ml_provenance.json` and, if present, load it into a new `ml_provenance` field on `ExperimentArtefacts`.
- Check for parquet files in `diagnostics/` and expose their paths (not load them) as a `diagnostic_files` list.

The extension to `ExperimentArtefacts` should use `| None` optional fields to preserve backward compatibility with version "1" artifacts:

```python
@dataclass
class ExperimentArtefacts:
    # ... existing fields unchanged ...
    ml_provenance: dict[str, Any] | None  # None for v1 artifacts
    diagnostic_files: list[Path]          # empty list for v1 artifacts
```

### 9.2 Extending the Report Generator

`generate_experiment_report()` is extended to emit an additional "ML Diagnostics" section in the markdown report when `ml_provenance` is present. This section includes:
- Lineage hashes table (feature, label, model, signal hashes).
- Git commit and library versions.
- Summary IC and directional accuracy statistics.
- Links to ML diagnostic plots (which are now present in `plots/`).

The D2 architectural constraint — **no recomputation, no data loading** — is strictly preserved. The report reads `ml_provenance.json` and discovers PNG files; it does not re-run diagnostics.

### 9.3 Experiment Comparison

The existing `comparison.py` (`compare_experiments`, `metrics_table`, `rank_experiments`) operates on loaded `ExperimentResult` objects. For ML experiments, `metrics.json` can contain additional keys (mean IC, directional accuracy). The comparison functions use `.get()` with defaults and are agnostic to metric names — they will naturally include ML metrics in comparison tables without modification.

A future extension (F3.3 or later) could add `compare_ml_experiments()` that loads `ml_provenance.json` alongside results and produces a lineage-aware comparison table. This is not part of the core F3 scope.

### 9.4 Future Narrative Report Possibilities

The current D2 report is static HTML/markdown from templates. A future extension could generate narrative text by feeding the structured `ml_provenance.json` and `metrics.json` to an LLM-backed summarizer (within the `src/llm/` module boundary). This is deferred to a later phase. The constraint remains: any narrative generation is a read-only consumer of saved artifacts, never a producer of new computation.

---

## 10. AI-Agent Integration Strategy

### 10.1 What Agents Are Allowed To Do

The `src/agents/` module currently contains only interfaces and placeholders. F3 should be designed so that future agents can interact with the ML experiment framework through well-defined typed interfaces.

**Permitted agent operations:**

- **Config generation:** An agent reads a research question (e.g., "test momentum features with Ridge on the full ETF universe") and produces a valid version "2" YAML config. The config is written to disk; the agent does not invoke the orchestrator directly. A human reviews and runs it.

- **Artifact summarization:** An agent reads `ml_provenance.json`, `metrics.json`, `diagnostics/*.parquet`, and produces a natural-language summary. Read-only: no mutation of artifacts, no re-computation.

- **Diagnostics interpretation:** An agent reads IC series, turnover series, and split metrics and flags anomalies (e.g., IC significantly negative after split 3, indicating possible regime shift). Advisory output only — no automated remediation.

- **Experiment proposal:** An agent reads the current `registry.json` and proposes new experiments based on gaps (e.g., no experiments with `labels.type == "volatility_target"`, no high-alpha experiments in the 2020–2022 period). Proposals are returned as typed config dicts; the agent does not write configs or run experiments autonomously.

### 10.2 What Agents Are Forbidden From Doing

- **Bypassing contracts:** An agent must not call `model.fit(dataset)` directly, construct `SupervisedDataset` outside the factory layer, or invoke `run_walk_forward_validation` without going through the config/orchestrator path.

- **Hidden orchestration:** An agent must not spawn sub-processes, schedule background jobs, or maintain session state between calls. Every agent action is a single stateless function call.

- **Direct mutation of datasets:** An agent must not write to parquet files, modify registry JSON, or delete experiment directories. Write operations belong to the orchestrator and save functions, not to agents.

- **Implicit runtime state:** An agent must not hold a reference to a fitted model, a loaded dataset, or a live registry handle between calls. All inputs to agent functions come through typed arguments; all outputs are typed return values.

### 10.3 Agent Interface Boundaries

The clean boundary for agent integration is the config layer: agents produce configs; configs drive orchestrators; orchestrators produce artifacts; agents read artifacts. This is a one-way data pipeline with no feedback loop that bypasses the computation layer.

```
Agent → config (YAML/JSON)
                    ↓
          orchestrator (runs computation)
                    ↓
          artifacts (saved to disk)
                    ↑
          Agent ← reads artifacts (read-only)
```

---

## 11. Explicit Anti-Goals

### 11.1 Distributed Orchestration

Celery, Dask, Ray, Prefect, and similar frameworks are deferred. The current platform runs single-machine, in-process. Adding distributed execution would require:
- Task serialization and deserialization (models, datasets).
- Result aggregation across workers.
- Failure handling and retry logic.
- Network and process isolation.

None of these are needed for the current research use case, which involves running a few dozen experiments sequentially over hours, not thousands of experiments in parallel over minutes.

### 11.2 Hyperparameter Sweeps

Grid search, random search, and Bayesian optimization are deferred. These require:
- A search space definition (a new config abstraction not currently designed).
- Multiple experiment runs as a coordinated batch.
- A results aggregator that compares across parameter variants.
- Protection against implicit data leakage through repeated evaluation on the same test period.

The last point is the critical one. Running 100 hyperparameter configurations against the same test period is a form of overfitting — the selected parameters are tuned to in-sample test performance. F3 does not introduce the infrastructure needed to do this correctly (e.g., nested cross-validation, held-out final test sets). Introducing sweep infrastructure without those safeguards would actively harm research validity.

### 11.3 Autonomous Agents

No agent in the current or planned architecture may autonomously decide to run an experiment, modify a config, or write artifacts without a human review step. Autonomous execution requires a level of trust in agent judgment that is not yet established, and the consequences of a misconfigured ML experiment (a leaked test period, an incorrect horizon) are subtle and hard to detect after the fact.

### 11.4 Deep Learning Pipelines

PyTorch, TensorFlow, JAX, and similar frameworks are deferred. The current model layer (`src/ml/models/`) uses sklearn, which provides sufficient expressiveness for the intended use cases (linear/logistic regression on low-frequency ETF signals). Deep learning introduces:
- GPU dependency and CUDA configuration.
- Training loop orchestration (epochs, batches, early stopping).
- Model serialization complexity (checkpoints, state dicts).
- Significantly longer training times incompatible with quick research iteration.

The `BaseMLModel` Protocol is intentionally framework-agnostic. Deep learning models can implement it in the future without any changes to F3.

### 11.5 Real-Time Execution

Live trading, streaming data, and order management systems are outside the scope of the entire platform, not just F3. The backtesting and walk-forward systems assume end-of-day prices and batch processing.

### 11.6 Model Serving

Exposing trained models as REST APIs or microservices is deferred. Models are research tools, not production services. The appropriate deployment pattern for this codebase is scripts and notebooks, not always-on servers.

### 11.7 Feature Stores

Precomputed, versioned, registry-backed feature datasets are on the roadmap (`docs/architecture/roadmap.md`: "Add feature/label/split manifests linked to dataset manifests") but are not part of F3. F3 computes features on-the-fly at run time from `FeatureSpec` configurations. A feature store would reduce computation time for large universes and long histories, but adds significant infrastructure complexity.

### 11.8 DAG Schedulers

Airflow, Luigi, and similar DAG execution frameworks are deferred. The orchestrator is a single Python function, which is sufficient for the current use case. DAGs add operational complexity (scheduler process, worker processes, monitoring, retry policies) that is not justified for batch research scripts.

### 11.9 MLFlow-Style Infrastructure

MLFlow (or similar tools) provides experiment tracking UI, artifact storage, model registry, and model serving. It is deferred because:
- It introduces a server dependency (either local or remote).
- The filesystem-first approach is sufficient for single-researcher use.
- The `ExperimentTracker` / `TrackingRun` no-op adapter in `tracking.py` already reserves this integration point for the future.
- The current team size does not require collaborative experiment sharing through a web UI.

---

## 12. Architectural Risk Analysis

### 12.1 YAML Explosion Risk

**Risk:** Adding `features`, `labels`, `model`, and `signal` sections multiplies the config complexity. A researcher attempting to configure a 7-feature cross-sectional model with 5 hyperparameters faces 30+ lines of YAML, of which a typo in a nested key silently fails.

**Mitigation:**
- Strict validation in `validate_ml_config()` catches typos and missing required fields before the run starts.
- Example configs (like those in §5.2) should be committed to `configs/experiments/` and serve as copy-paste templates.
- Feature type names must exactly match registered function names — no fuzzy matching.
- The `validate_config()` error messages (already implemented in `config_io.py`) should be used as the model for new ML validation: specific, actionable, pointing to the offending key.
- Limit default filling to genuinely optional fields; never silently fill fields that affect computation.

### 12.2 Orchestration Duplication Risk

**Risk:** The D1 orchestrator (`orchestrator.py`) and the ML prediction pipeline (`src/ml/pipelines/walk_forward.py`) both run walk-forward loops. If F3 introduces a third loop (or a modified orchestrator), three separate orchestration paths will exist, each with slightly different semantics.

**Mitigation:**
- F3 must not add a third walk-forward loop. The ML prediction pipeline (`run_walk_forward_predictions`) runs once; the strategy-level runner (`run_walk_forward_validation`) runs once. They operate on the same splits.
- The F3 orchestrator is an extension of the D1 orchestrator (version "2" config path), not a parallel implementation. The `run_experiment_from_config()` function detects the schema version and routes accordingly:

```python
if _schema_version(raw_cfg) == "2":
    return _run_ml_experiment(raw_cfg, source_path)
else:
    return _run_traditional_experiment(raw_cfg, source_path)
```

Where `_run_traditional_experiment` is exactly the current implementation, unchanged.

### 12.3 Feature-Lineage Complexity

**Risk:** `FeatureSpec` describes feature construction intent, but Python callables (lambdas, closures) are used to execute it. If the same `FeatureSpec` produces different features depending on which registered function is currently installed, the hash no longer guarantees reproducibility.

**Mitigation:**
- The feature function registry must be version-controlled in code. Changes to a registered function's behavior must bump the function's version in `FeatureSpec`.
- `FeatureSpec.version` applies to the spec schema; individual feature entries should optionally carry a `fn_version` field that records which implementation version was used.
- Recommended policy: treat feature function implementations as immutable once committed. New implementations get new type names (`"momentum_v2"` rather than modifying `"momentum"`).
- This is a policy constraint, not an implementation constraint. F3 should document it, not enforce it programmatically.

### 12.4 Hidden State Risk in MLStrategy

**Risk:** `MLStrategy` is stateful — it holds a fitted model in `self._model` and a `self._is_fitted` flag. If the same `MLStrategy` instance is passed to both `run_walk_forward_validation` and `run_walk_forward_predictions`, the model's internal state after one run may affect the other.

**Mitigation (documented in §8.3):** The F3 orchestrator must instantiate two separate model objects when running both pipelines. The factory's `build_model(spec)` function must be called twice, producing independent instances. This is a correctness requirement that must be explicitly enforced in the orchestrator implementation and documented in the factory function's docstring.

### 12.5 Overengineering Risk

**Risk:** This design document describes many new types and multiple schema versions. There is a real risk that implementing all of F3 at once produces a large, complex codebase addition that is harder to test and harder to debug than the incremental approach.

**Mitigation:** The phased implementation plan in §13 is specifically designed to deliver F3 in small, independently testable increments. Each increment ships with tests. No increment introduces abstractions that are not immediately used.

### 12.6 Experiment Sprawl Risk

**Risk:** Once ML experiments can be run via config, a researcher may create dozens of variants with small differences, filling the registry with noise and making it difficult to identify which experiments are significant.

**Mitigation:**
- The `deduplicate` option (§7.5) produces unique directories per config hash, making it easy to enumerate distinct runs.
- Registry query capabilities (`query()` by tags, strategy, min Sharpe) already exist and can be used to filter.
- Discipline over naming conventions (`tags: ["pilot"]` vs `tags: ["final"]`) is a research practice issue, not a framework issue. F3 can recommend tag conventions in its documentation without enforcing them.

### 12.7 Cross-Spec Validation Complexity

**Risk:** The consistency check between `SignalSpec.type` and `LabelSpec.type` and universe size (§5.4) is a cross-spec constraint that does not fit cleanly into per-section validation. If it is implemented incompletely or inconsistently, researchers will get cryptic runtime errors (e.g., "TypeError: top_n_weights requires pd.DataFrame predictions") instead of clear config errors.

**Mitigation:** Implement cross-spec validation as a separate function `validate_ml_config_consistency(spec: MLExperimentSpec)` that takes the fully constructed spec and checks all cross-spec constraints in one place. This function is called after the per-section validation passes. Errors from this function are formatted the same way as per-section errors: specific key paths, expected values, actual values.

---

## 13. Recommended Incremental Implementation Plan

### F3.1 — Config Schema Extension (No ML run yet)

**Scope:** Extend the D1 config schema to version "2". No orchestrator changes. No new artifact types.

**Deliverables:**
- `validate_ml_config(raw: dict) -> None` in `config_io.py`.
- `normalize_ml_config(raw: dict) -> dict` in `config_io.py`.
- `_VALID_FEATURE_TYPES`, `_VALID_LABEL_TYPES`, `_VALID_MODEL_TYPES`, `_VALID_SIGNAL_TYPES` registries.
- Cross-spec validation function `validate_ml_config_consistency`.
- Example version "2" configs in `configs/experiments/`.
- Tests for all new validation and normalization functions (covering valid, invalid, and edge cases).

**Does not change:** orchestrator, factory, artifact layout, registry.

**Why first?** The config layer is the foundation. If the schema is wrong, every downstream component will be wrong. Testing the schema in isolation before wiring it to computation is the safest order.

---

### F3.2 — Factory Extension (MLStrategy via config)

**Scope:** Extend the factory layer to support version "2" configs. `MLStrategy` becomes buildable from config. No artifact changes yet.

**Deliverables:**
- `build_feature_fns(feature_spec: dict) -> dict[str, Callable]` in `factory.py`.
- `build_label_fn(label_spec: dict) -> Callable` in `factory.py`.
- `build_model(model_spec: dict) -> BaseMLModel` in `factory.py`.
- `build_signal_fn(signal_spec: dict) -> Callable[[PredictionSeries], pd.DataFrame]` in `factory.py`.
- `build_ml_strategy(feature_fns, label_fn, model, signal_fn, horizon, label_name) -> MLStrategy` in `factory.py`.
- `build_ml_experiment_spec(norm_cfg) -> MLExperimentSpec` — composing all sub-specs.
- Hash functions: `feature_hash`, `label_hash`, `model_hash`, `signal_hash`, `ml_experiment_hash` in `config.py`.
- `MLStrategy` added to `_STRATEGY_REGISTRY` (for identification; factory builds it via new path).
- Tests for all factory functions with round-trip validation.

**Does not change:** orchestrator, artifact layout, registry, reporting.

---

### F3.3 — Orchestrator Extension (ML runs produce artifacts)

**Scope:** Extend `run_experiment_from_config()` to detect version "2" configs and run the full ML pipeline. New artifact types are written.

**Deliverables:**
- Version routing in `run_experiment_from_config()`: `_run_ml_experiment()` for version "2".
- `_run_ml_experiment()` implementation: full 13-step sequence from §8.1.
- `ml_provenance.json` written alongside `metadata.json`.
- `predictions.parquet` written to experiment directory (using existing `save_run()` hook).
- Diagnostic parquets written to `diagnostics/`.
- New ML plot calls (6 plot functions from `ml_plots.py`) added to `_build_plots()`.
- `ARTEFACT_VERSION` bumped to `"2"` in `contracts.py`.
- `check_ml_artefact_dir()` advisory function in `contracts.py`.
- Tests: integration test running a full ML experiment from a version "2" YAML config, checking that all expected files exist.

**Does not change:** existing D1 orchestrator path (version "1" configs unchanged), reporting.

---

### F3.4 — Registry and Provenance Extension

**Scope:** Make the registry aware of ML experiments. Populate `config_hash` field in provenance. Add `ml_experiment_hash` to registry entries.

**Deliverables:**
- `register()` in `ExperimentRegistry` extended to accept optional `ml_experiment_hash`.
- `_SUMMARY_METRICS` replaced with dynamic metric inclusion (uses `.get()` on metrics dict).
- `config_hash` populated from `ml_experiment_hash` when available.
- `ml_provenance.json` path exposed in registry entries (for future report linking).
- Tests for extended registry behavior.

**Does not change:** reporting, schema, factory, orchestrator (except registry call).

---

### F3.5 — Reporting Extension (ML Diagnostics Section)

**Scope:** Extend D2 report generation to include ML diagnostics when `ml_provenance.json` is present.

**Deliverables:**
- `ExperimentArtefacts` dataclass extended with `ml_provenance` and `diagnostic_files` fields.
- `load_experiment_artefacts()` updated to discover ML artifact files.
- `render_report()` in `markdown.py` extended with an optional "ML Diagnostics" section.
- Provenance sidecar updated to include `ml_experiment_hash`.
- Tests: report generation from a saved ML experiment directory, verifying ML section is present.

**Does not change:** schema, factory, orchestrator, registry.

---

### F3.6 — Deduplication and Identity Management (Optional)

**Scope:** Add `deduplicate` output option for ML experiments. Registry identity shifts from name-based to content-hash-based.

**Deliverables:**
- `output.deduplicate: bool` config option (default `false`).
- When `true`: output directory is `<experiment_name>_<hash[:8]>/`.
- Registry `register()` deduplication key is `ml_experiment_hash` instead of `experiment_name`.
- `ExperimentRegistry.query()` extended to filter by `ml_experiment_hash`.
- Tests for both `deduplicate: true` and `deduplicate: false` behavior.

**Note:** This phase is listed as optional because the current behavior (name-based, overwrite-on-rerun) is acceptable for single-researcher use and avoids directory proliferation for exploratory work. F3.6 becomes necessary only when multiple variants of the same named experiment need to coexist.

---

### Implementation Sequencing Summary

| Phase | Dependencies | Artifacts | Tests |
|-------|-------------|-----------|-------|
| F3.1 — Schema | None | Config validation | Validation/normalization unit tests |
| F3.2 — Factory | F3.1 | Spec objects | Factory unit + integration tests |
| F3.3 — Orchestrator | F3.1, F3.2 | All ML artifacts | End-to-end integration test |
| F3.4 — Registry | F3.3 | Registry entries | Registry unit tests |
| F3.5 — Reporting | F3.3 | ML report section | Report generation test |
| F3.6 — Dedup | F3.3, F3.4 | Directory layout | Dedup behavior tests |

Each phase is independently shippable. The platform remains fully functional between phases — version "1" configs run correctly at every point in the sequence.

---

## Appendix: Key Existing File References

All claims in this document are grounded in the following files, read during this design session:

| Claim | File | Key Lines |
|-------|------|-----------|
| `_VALID_STRATEGY_TYPES` hardcoded | `src/experiments/config_io.py` | Line 24 |
| `_STRATEGY_REGISTRY` hardcoded | `src/experiments/factory.py` | Lines 32–36 |
| `predictions` slot exists but unused | `src/experiments/tracking.py` | Lines 47–53 |
| `diagnostics/` always empty | `src/experiments/tracking.py` | Line 88 |
| `config_hash` is None by design | `src/reporting/report_builder.py` | Line 192 |
| Registry replaces by name | `src/experiments/registry.py` | Line 108 |
| `_SUMMARY_METRICS` hardcoded | `src/experiments/registry.py` | Line 36 |
| `dataset_hash` hashes metadata only | `src/ml/datasets.py` | Lines 124–157 |
| `BaseMLModel` Protocol | `src/ml/models/base.py` | Lines 17–47 |
| `MLStrategy.fit` hook | `src/strategies/ml_strategy.py` | Lines 141–158 |
| Walk-forward fit hook | `src/validation/walk_forward.py` | Module docstring |
| `run_walk_forward_predictions` | `src/ml/pipelines/walk_forward.py` | Lines 38–81 |
| `signal_fn` injection pattern | `src/strategies/ml_strategy.py` | Lines 52–91 |
| Diagnostic functions exist | `src/ml/diagnostics/` | All three modules |
| ML plot functions exist | `src/visualization/ml_plots.py` | Lines 41–354 |
| `save_run` signature | `src/experiments/tracking.py` | Lines 47–95 |
| `generate_experiment_report` read-only | `src/reporting/report_builder.py` | Module docstring |
| `ARTEFACT_VERSION = "1"` | `src/experiments/contracts.py` | Line 13 |
| D1 schema version "1" | `src/experiments/config_io.py` | Line 23 |
