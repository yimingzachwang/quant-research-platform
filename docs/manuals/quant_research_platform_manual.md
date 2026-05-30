# Quant Research Platform Manual

Status: code-derived architecture and methodology manual  
Last updated: 2026-05-23

This manual documents the implemented quantitative research platform as it
exists in the repository. It is not a roadmap disguised as current state.
Where systems are partial or planned, they are labelled explicitly.

## 1. Platform Overview

The repository is a research-first quantitative ETF platform. Its main purpose
is to support deterministic, leakage-aware, reproducible research workflows
over daily ETF data. It is not a live trading system and does not contain
broker integration, order management, production risk, or a full execution
simulator.

The platform philosophy is infrastructure before alpha. The codebase first
establishes contracts for data, features, labels, predictions, strategies,
backtests, validation, experiments, visualisation, reporting, and future
agent workflows. This allows research ideas to move through a controlled
pipeline rather than remaining notebook-only experiments.

The strongest design priorities are:

- reproducible artefacts and deterministic identifiers
- explicit temporal semantics
- no look-ahead leakage in model evaluation or backtests
- modular subsystem boundaries
- vectorised pandas workflows for low/mid-frequency research
- read-only visualisation and reporting
- future AI-agent compatibility through typed contracts rather than implicit
  natural-language execution

Current implemented capabilities include registry-backed data loading,
profile-driven ingestion, feature functions, portfolio alignment and
allocation helpers, vectorised single-asset and multi-asset backtests,
walk-forward validation, baseline ML model wrappers, single-asset ML
experiment orchestration, ML diagnostics, plotting utilities, and static
report generation.

Several systems remain partial or planned. Generic `src/models`, `src/signals`,
`src/risk`, `src/execution`, and `src/evaluation` are placeholder/protocol
areas. Panel ML experiment execution is schema-recognised in places but not
implemented. `src/llm` is experimental and not integrated into the deterministic
ingestion or experiment pipeline.

## 2. System Architecture

The platform is organised as layered research infrastructure:

```text
configs/
  -> src/data/
  -> src/cleaning/
  -> src/features/
  -> src/ml/
  -> src/strategies/
  -> src/portfolio/
  -> src/backtesting/
  -> src/validation/
  -> src/experiments/
  -> src/visualization/
  -> src/reporting/
```

The dependency structure is intentionally conservative. Upstream layers own
data access and pure transformations. Midstream layers compose those
transformations into research objects such as strategies, predictions, and
weights. Downstream layers evaluate, persist, visualise, and report results.

Key architecture diagrams are stored under `docs/architecture/`:

- `platform_architecture_full.mmd`
- `research_pipeline_flow.mmd`
- `ml_experiment_lifecycle.mmd`
- `validation_workflow.mmd`
- `artefact_persistence_flow.mmd`
- `visualization_reporting_architecture.mmd`
- `dependency_subsystem_graph.mmd`

### Subsystem Boundaries

`src/data` owns ingestion, manifests, registry, storage, validation reports,
and registry-backed loading. It should not contain strategy, ML, or
backtesting logic.

`src/features` owns deterministic transformations. Feature functions should
not load data, generate labels, fit models, or persist artefacts.

`src/ml` owns supervised dataset construction, label generation, model
wrappers, prediction contracts, prediction-to-weight translation, ML
walk-forward prediction helpers, and ML diagnostics. It reuses the existing
feature, portfolio, strategy, and validation layers rather than replacing
them.

`src/strategies` owns price-to-weight generation. Strategies emit target
weights. They do not apply execution lag or compute final performance.

`src/backtesting` and `src/portfolio/portfolio_backtest.py` own realised
performance simulation, including timing, transaction costs, equity curves,
drawdowns, and metrics.

`src/validation` owns chronological train/test design and split-level
evaluation.

`src/experiments` owns config-driven orchestration and artefact persistence.
Factory functions remain pure; orchestrators own I/O.

`src/visualization` and `src/reporting` are read-only consumers of computed
results and saved artefacts.

### Stateless Design And Artefacts

The code favours stateless functions where possible. Features, labels,
metrics, diagnostics, report renderers, and factory functions are mostly pure.
Stateful objects are used where they encode real platform state: registries,
storage, engines, experiment runs, model wrappers, and strategies.

Artefacts are filesystem-based and inspectable. Data is parquet. Registries
are JSON. Experiment outputs are JSON, parquet, and PNG. Reports are markdown
and optional HTML, with provenance sidecars.

## 3. Research Pipeline

The core research flow is:

```text
Data
-> Cleaning
-> Features
-> ML/Datasets
-> Signals
-> Strategies
-> Backtesting
-> Validation
-> Diagnostics
-> Visualisation
-> Reporting
-> Artefact Persistence
```

### Data

Purpose: acquire and materialise external market or macro data.

Key abstractions:

- `DataRequest`
- `DatasetProfileConfig`
- `DataAgentV1Config`
- `DatasetUpdateEngine`
- `DatasetManifest`
- `DatasetQuery`
- `DatasetRegistry`
- `DataStorage`
- `DatasetValidator`

Guarantees:

- request expansion is typed
- request hashes are deterministic
- duplicate requests can be skipped
- materialised datasets are registered with manifest metadata
- loader resolution is registry-driven, not path-inferred

Extension points:

- add downloader adapters behind existing request contracts
- extend manifest fields carefully if new dataset families need them
- generalise registry queries before adding any new registry system

### Cleaning

Purpose: perform explicit, deterministic data hygiene operations.

Key modules:

- `src/cleaning/timestamps.py`
- `src/cleaning/numeric.py`
- `src/cleaning/missing.py`
- `src/cleaning/validation.py`

Guarantees:

- helpers are small and testable
- severe data issues are not silently turned into research assumptions
- cleaning remains separate from strategy and ML logic

### Features

Purpose: compute deterministic transformations used by strategies or ML
datasets.

Key modules:

- `src/features/returns.py`
- `src/features/momentum.py`
- `src/features/rolling.py`
- `src/features/volatility.py`
- `src/features/trend.py`
- `src/features/normalization.py`

Guarantees:

- functions are stateless pandas transformations
- rolling-window warm-up NaNs remain visible
- feature functions do not create forward labels

### ML/Datasets

Purpose: build leakage-aware supervised learning objects.

Key abstractions:

- `SupervisedDataset`
- `PredictionSeries`
- `BaseMLModel`
- `WalkForwardPredictions`

Major functions:

- `build_feature_matrix`
- `align_features_and_labels`
- `build_supervised_dataset`
- `dataset_hash`
- `forward_returns`
- `binary_direction_label`
- `volatility_target`
- `ranking_target`

Guarantees:

- labels use explicit `shift(-horizon)`
- feature/label alignment happens at a named boundary
- models receive aligned, NaN-free datasets
- splits are owned by validation, not by model wrappers

### Signals And Strategies

Purpose: convert research signals or predictions into target weights.

Implemented:

- `Strategy`
- `BuyAndHoldStrategy`
- `EqualWeightStrategy`
- `MomentumRotationStrategy`
- `MLStrategy`
- `sign_signal`
- `threshold_signal`
- `top_n_weights`
- `long_short_weights`
- `normalize_to_weights`

Guarantees:

- strategies emit weights, not performance claims
- signal translators are pure
- MLStrategy adapts models to the existing strategy interface

### Backtesting

Purpose: evaluate realised performance under explicit timing and cost
assumptions.

Key functions:

- `run_backtest`
- `run_portfolio_backtest`
- `compute_metrics`

Guarantees:

- single-asset signals are lagged with `signal.shift(1)`
- portfolio weights are lagged with `weights.shift(1)`
- transaction costs are turnover-based
- metrics are computed on net returns

### Validation

Purpose: evaluate out-of-sample stability through chronological splits.

Key abstractions:

- `TimeSplit`
- `WalkForwardResult`
- `SplitResult`

Major functions:

- `rolling_time_splits`
- `expanding_time_splits`
- `run_walk_forward_validation`
- `split_metrics_table`
- `summarize_stability`
- `parameter_robustness_summary`

Guarantees:

- train windows precede test windows
- optional fit hooks receive train data only
- metrics are computed on held-out test windows

### Diagnostics

Purpose: inspect prediction quality, split stability, turnover, and model
behaviour beyond aggregate returns.

Implemented:

- prediction correlation
- information coefficient
- rolling directional accuracy
- prediction quantiles
- coefficient stability
- prediction drift
- signal turnover
- turnover by split

### Visualisation

Purpose: create diagnostic figures from computed results.

Guarantee:

- plotting functions consume already-computed inputs and return figures
- visualisation does not own ingestion, features, backtests, validation, or
  reporting

### Reporting

Purpose: render saved experiment artefacts into markdown and optional HTML.

Implemented:

- `ExperimentArtefacts`
- `ReportPaths`
- `ResearchReportSpec`
- `render_report`
- `generate_experiment_report`

Guarantee:

- reports are read-only with respect to experiment artefacts
- reporting does not rerun experiments or recompute metrics

## 4. Data Infrastructure

The data layer centres on typed requests and manifest-backed lineage.

`DataRequest` is the ingestion contract. A profile such as
`configs/data/daily_prices.yaml` expands into one request per symbol. The
request is then downloaded, standardised, validated, persisted, and registered.

`DatasetManifest` records the materialised dataset identity: dataset ID, name,
symbol, data type, source, frequency, date range, schema version, storage path,
row count, timestamp, and request hash. `hash_request` uses stable JSON
serialisation and SHA-256 over request fields plus schema version.

`DatasetRegistry` persists manifests to
`data/external/registry/datasets.json`. `DatasetQuery` formalises exact-match
lookup. `load_dataset` resolves a query through the registry before reading
parquet. It raises explicit exceptions for missing, ambiguous, schema
incompatible, malformed, or column-incomplete datasets.

Parquet is used for local raw and processed data because it is efficient,
portable, and inspectable. The platform does not currently implement a
database-backed dataset registry or cloud object storage abstraction.

Cleaning is deliberately conservative. The repository contains timestamp,
numeric, missing-data, and validation helpers, and the data validator checks
basic structural integrity before persistence.

Known limitation: `adjusted_close` is declared in V1 schema/config context,
but runtime OHLCV standardisation/validation is not fully wired around it.

## 5. Feature Engineering System

The feature system is explicit and function-oriented. This is a deliberate
engineering decision: in a research codebase, hidden feature pipelines can
make temporal assumptions difficult to audit.

Implemented feature families include:

- simple, log, and cumulative returns
- momentum
- rolling z-score/rank/min-max transforms
- rolling volatility
- SMA, EMA, and trend strength
- normalisation helpers

Feature functions operate on `pd.Series` or `pd.DataFrame` inputs and return
pandas outputs. Warm-up periods are represented as NaN rather than silently
filled. This allows the ML alignment boundary to decide which rows are valid.

`build_feature_matrix` composes feature functions into a wide matrix. It does
not reimplement feature logic and does not drop rows. Multi-column outputs are
prefixed to avoid name collisions.

Current extension point: materialised feature datasets are not yet registered
through the dataset registry. Future feature materialisation should extend the
existing manifest/query approach rather than introduce a parallel feature
store.

## 6. Backtesting Engine

The backtesting layer is vectorised and timing-aware.

For single-asset research, `run_backtest` accepts returns, a signal series,
and transaction costs. It aligns inputs on common timestamps, applies:

```python
position = signal.shift(1).fillna(0.0)
```

It then computes gross return, turnover, cost, net return, equity curve, and
drawdown.

For multi-asset research, `run_portfolio_backtest` accepts a Date x Asset
return matrix and Date x Asset target weight matrix. It aligns rows and
columns, applies:

```python
w_lagged = weights.shift(1).fillna(0.0)
```

It then computes weighted gross returns, turnover, transaction costs, net
returns, equity curve, drawdown, and scalar metrics.

This shift is not an implementation detail. It is the central look-ahead
prevention rule for execution timing. A weight produced at time `t` cannot
earn the return at time `t`; it is applied to the next period.

Transaction costs are currently simple one-way turnover costs in basis points.
This is a research approximation, not a production execution simulator.

## 7. Multi-Asset Research Layer

The multi-asset layer lives primarily in `src/portfolio`.

Implemented responsibilities:

- load registered datasets for a symbol universe
- align prices into Date x Asset panels
- compute aligned returns
- compute panel features
- rank assets
- select top or bottom assets
- allocate equal or volatility-scaled weights
- resample periodic weights to daily weights
- run portfolio backtests

The layer supports cross-asset workflows such as momentum rotation. It also
supports ML signal conversion for panel predictions through allocation helpers
used by `src.ml.signals`.

Current limitations:

- no portfolio optimiser
- no constraint solver
- no covariance model
- no live risk system
- no execution-aware order lifecycle

Future portfolio optimisation should consume predictions, scores, risk
estimates, and constraints, then emit target weights. It should not load data
directly or run its own backtest.

## 8. Validation Framework

The validation framework is chronological by design. Financial time series
are not IID samples, so random k-fold splitting is avoided. Random folds can
leak future regimes into training and produce misleading estimates of
out-of-sample performance.

Implemented split types:

- rolling windows: fixed-width training windows and fixed-width test windows
- expanding windows: training starts at the beginning and grows through time

Both split generators snap boundaries to available dates and enforce strict
chronological ordering. Optional `gap_days` can insert a buffer between train
and test periods.

`run_walk_forward_validation` evaluates one split at a time. For strategies
with a `fit(train_prices)` method, the fit hook receives only the training
slice. The strategy is then run on prices truncated at the split's `test_end`.
Metrics are computed only on the held-out test window.

This design supports both traditional strategies and ML strategies without
creating separate validation engines.

## 9. Machine Learning Research Infrastructure

The ML system is implemented as a research layer that wraps the existing
platform rather than replacing it.

### Dataset Construction

Labels are built in `src/ml/labels.py` and all forward-looking labels use
explicit `shift(-horizon)`. The resulting final rows are NaN because the
future is unavailable.

`align_features_and_labels` is the canonical alignment boundary. It
inner-joins features and labels on index and drops rows with any NaN values.
`build_supervised_dataset` wraps this into a `SupervisedDataset`.

`dataset_hash` hashes structural metadata only: feature names, label name,
horizon, index range, and shapes. It does not hash data values.

### Model Abstraction

`BaseMLModel` is a runtime-checkable protocol with `fit(dataset)` and
`predict(X) -> PredictionSeries`. Implemented wrappers include linear
regression, ridge, lasso, elastic net, and logistic regression. Logistic
regression returns class-1 probabilities.

Model wrappers are thin. They assume alignment and NaN removal have already
happened. They do not generate labels, create splits, shift time, or mutate
index semantics.

### Prediction And Signal Abstraction

`PredictionSeries` supports both `pd.Series` and `pd.DataFrame` values.
Single-asset translators include `sign_signal` and `threshold_signal`.
Panel translators include `top_n_weights`, `long_short_weights`, and
`normalize_to_weights`, though panel experiment orchestration is not yet
implemented in F3.

`MLStrategy` adapts a model to the existing `Strategy` interface. It builds
features, labels, and a supervised dataset during `fit`, then builds features,
predicts, applies a signal function, and emits weights during
`generate_weights`.

### F3 ML Experiment Support

Implemented F3 components include:

- version `"2"` ML config validation in `src/experiments/ml_config.py`
- typed ML specs: feature, label, model, signal, experiment
- deterministic `ml_experiment_hash`
- pure ML factory functions in `src/experiments/ml_factory.py`
- version routing in `run_experiment_from_config`
- single-asset ML experiment execution
- `ml_provenance.json`
- split metrics and ML diagnostics persistence

Current F3 limitation:

- panel labels/signals such as `ranking_target`, `top_n`, `long_short`, and
  `normalize` are schema-recognised but raise `ValueError` at factory time
  because panel experiments are not yet implemented.

## 10. Diagnostics And Analytics

Diagnostics extend evaluation beyond scalar performance metrics.

Implemented prediction diagnostics:

- `prediction_correlation`: Pearson correlation after index alignment and
  NaN-pair removal
- `information_coefficient`: cross-sectional Spearman IC per timestamp
- `rolling_directional_accuracy`: rolling sign-hit rate
- `prediction_quantiles`: quantile assignment for prediction analysis

Implemented stability diagnostics:

- `split_metric_table`: delegation to validation stability tables
- `coefficient_stability`: mean, standard deviation, sign consistency, min,
  and max by feature coefficient
- `prediction_drift`: rolling mean of predictions

Implemented turnover diagnostics:

- `signal_turnover`
- `average_turnover`
- `turnover_by_split`

These diagnostics matter because attractive aggregate returns can hide unstable
coefficients, inconsistent prediction ranking, high turnover, or regime
dependence.

## 11. Visualisation Architecture

The visualisation subsystem is read-only. Its job is to turn computed results
into diagnostic figures, not to compute those results.

Implemented plot families include:

- equity and drawdown plots
- rolling backtest diagnostics
- signal and position plots
- return distributions and heatmaps
- portfolio weights, correlations, concentration, turnover, and contribution
- strategy comparison figures
- walk-forward validation plots
- ML prediction, IC, coefficient stability, split metric, and turnover plots

Plotting functions are reusable and suitable for report-quality figures, but
they are intentionally not a dashboard framework. They accept already-computed
series, frames, or result objects and return matplotlib figures.

This separation protects reproducibility. A figure should not secretly rerun
a strategy, fetch data, or recompute model outputs.

## 12. Experiment Persistence

Experiment persistence is filesystem-based. `ExperimentResult` carries
strategy name, parameters, metrics, weights, equity curve, returns, and
creation time. `save_run` persists structured artefacts to an experiment
directory.

Current artefacts can include:

- `metadata.json`
- `metrics.json`
- weights parquet
- returns parquet
- equity curve parquet
- raw config
- normalised config
- plots
- diagnostics JSON
- `ml_provenance.json` for v2 ML experiments

`ExperimentRegistry` records saved runs in a local registry file. The
orchestrator writes raw and normalised config artefacts so future readers can
see both original user input and resolved defaults.

Reproducibility guarantees are practical rather than absolute:

- configs are validated and normalised
- artefacts are saved in deterministic locations under configured output dirs
- ML experiment hashes are deterministic over structural spec fields
- reporting consumes saved artefacts rather than recomputing runs

## 13. Reporting System

The reporting system is implemented under `src/reporting`.

Current capabilities:

- load saved experiment artefacts
- discover figures
- load optional ML provenance and diagnostics sidecars
- render markdown through `render_report`
- render optional HTML through a minimal converter
- copy figures into report output directories
- write provenance JSON
- write a frontend-facing report manifest
- govern section inclusion with `ResearchReportSpec`

`ResearchReportSpec` is implemented as a frozen dataclass. Current presets:

- `FULL_DEMO_REPORT`
- `COMPACT_REPORT`
- `DIAGNOSTICS_REPORT`
- `AUDIT_REPORT`

The spec controls sections such as summary, metadata, configuration, metrics,
ML analysis, validation, diagnostics, figures, and provenance. The renderer
remains fixed-scope and deterministic; the spec decides which sections are
included.

Partially implemented/future:

- richer configurable report modes are emerging through `ResearchReportSpec`
- PDF output is not implemented in package code
- frontend dashboards are not implemented
- report templates/plugins are intentionally absent

## 14. Extension Architecture

This section describes future integration points. These are not current
implemented systems unless stated above.

### Frontend Dashboards

Future dashboards should consume report manifests, saved artefacts, metrics,
figures, and registry metadata. They should not run research logic directly.

### LLM-Driven Orchestration

Future agents should produce typed configs, review artefacts, summarise
diagnostics, or propose next experiments. Agents should not bypass data
requests, registry loading, validation splits, or experiment artefact
persistence.

### Live Trading Adapters

Live trading is not implemented. If ever added, it should be a separate
adapter layer downstream of validated research workflows, with explicit
execution and risk contracts. It should not be added to the current research
backtesting code.

### Portfolio Optimisation

Optimisers should consume predictions, expected return estimates, risk
estimates, constraints, and previous weights, then emit target weights.
Backtesting should remain responsible for execution lag and costs.

### Distributed Execution

Distributed execution is not implemented. It should be deferred until local
deterministic workflows are stable and there is a measurable workload that
requires distribution.

## 15. Design Principles And Engineering Decisions

The architecture is shaped by research integrity.

### Stateless Transforms

Features, labels, metrics, diagnostics, and report renderers are mostly
stateless functions. Stateless code is easier to test, easier to reuse, and
less likely to hide research state.

### Explicit Alignment

Alignment happens at named boundaries: registry query resolution, feature/label
alignment, return/weight alignment in backtests, and split slicing in
validation. Hidden alignment inside model wrappers or plotting functions is
avoided.

### Explicit Temporal Shifts

Forward-looking labels use `shift(-horizon)`. Backtests use `shift(1)`.
These are visible, testable, and documented.

### Deterministic Artefacts

The system uses typed configs, normalised configs, manifests, hashes,
filesystem artefacts, and report provenance to make research inspectable.

### Modularity

Each layer owns a narrow responsibility. Data does not own alpha. Features do
not own labels. Models do not own splits. Strategies do not own execution
timing. Reports do not recompute experiments.

### Validation-First Research

The platform privileges chronological evaluation over convenient random
splits. Walk-forward validation is treated as a core research primitive, not
an optional afterthought.

### Avoided Complexity

The repository intentionally avoids ORMs, service layers, plugin frameworks,
async orchestration, distributed execution, hidden caches, and microservice
patterns. Those are future possibilities only if the current deterministic
local architecture becomes insufficient.

