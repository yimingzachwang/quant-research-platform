# Research Systems Architecture Audit

Generated: 2026-05-26

## Executive Finding

This platform is best represented as a filesystem-first quantitative research publication and experiment infrastructure. The strongest real capabilities are config-driven experiment execution, temporal validation, deterministic feature/model specification, persisted diagnostic sidecars, generated research reports, and frontend-facing report manifests. The platform has enough real implementation to support a narrative around research visibility, reproducibility, chronology preservation, and validation discipline.

It should not be represented as a live trading system, deployed institutional platform, cloud architecture, autonomous AI hedge fund, or fully general multi-asset ML research engine. Several layers remain scaffolded or partially implemented, especially risk, execution simulation, generic evaluation, panel ML workflows, and production data freshness operations.

## Evidence Base

Primary implementation files inspected:

- `src/experiments/orchestrator.py`
- `src/experiments/config_io.py`
- `src/experiments/ml_config.py`
- `src/experiments/ml_factory.py`
- `src/experiments/results.py`
- `src/experiments/registry.py`
- `src/experiments/tracking.py`
- `src/reporting/report_builder.py`
- `src/reporting/markdown.py`
- `src/reporting/report_spec.py`
- `src/reporting/consistency.py`
- `src/validation/splits.py`
- `src/validation/walk_forward.py`
- `src/validation/stability.py`
- `src/portfolio/portfolio_backtest.py`
- `src/portfolio/alignment.py`
- `src/strategies/runner.py`
- `src/strategies/ml_strategy.py`
- `src/data/engine.py`
- `src/data/manifest.py`
- `src/data/loaders/storage.py`
- `src/data/validators/dataset_validator.py`
- `src/data/registry/json_registry.py`
- `src/llm/translator.py`
- `src/agents/interfaces.py`

Representative generated artefacts inspected:

- `results/experiments/canonical_ml_showcase/`
- `results/experiments/example_momentum_rotation/`
- `reports/markdown/canonical_ml_showcase.md`
- `reports/markdown/canonical_ml_showcase_manifest.json`
- `reports/markdown/canonical_ml_showcase_provenance.json`
- `reports/markdown/example_momentum_rotation_provenance.json`
- `reports/figures/canonical_ml_showcase/`

## Part A - Architectural Audit

### 1. Subsystem Inventory

#### Configuration Layer

Purpose: declarative experiment, data, reporting, universe, portfolio, risk, model, and agent specification.

Major files:

- `configs/experiments/canonical_ml_showcase.yaml`
- `configs/experiments/momentum_rotation.yaml`
- `configs/experiments/equal_weight.yaml`
- `configs/data/daily_prices.yaml`
- `configs/universes/core_etfs.yaml`
- `configs/reports/default_markdown.yaml`
- `src/experiments/config_io.py`
- `src/experiments/ml_config.py`

Implemented responsibilities:

- Load YAML/JSON configs.
- Validate required experiment fields.
- Normalize missing defaults into canonical structures.
- Route version `1` strategy experiments and version `2` ML experiments.
- Hash ML experiment specs deterministically through `ml_experiment_hash`.

Inputs:

- Raw YAML/JSON experiment configs.
- Universe, date range, validation, execution, output, feature, label, model, and signal sections.

Outputs:

- Normalized dictionaries.
- Typed experiment specs.
- Persisted `raw_config.yaml` and `normalized_config.json` per run.
- ML provenance hash in `ml_provenance.json` for version `2` experiments.

Research role:

- Makes experiments inspectable before execution.
- Separates research intent from implementation.
- Supports reproducibility by preserving both the original config and normalized execution contract.

Maturity:

- Implemented for v1 strategy experiments and v2 single-asset ML experiments.
- Panel ML schema is partially represented but intentionally not implemented in the factory.

#### Data Layer

Purpose: dataset ingestion, storage, validation, manifesting, and universe loading.

Major files:

- `src/data/engine.py`
- `src/data/manifest.py`
- `src/data/loaders/storage.py`
- `src/data/validators/dataset_validator.py`
- `src/data/registry/json_registry.py`
- `src/portfolio/alignment.py`
- `configs/data/daily_prices.yaml`

Implemented responsibilities:

- Profile-driven dataset updates through `DatasetUpdateEngine`.
- Request hashing through `hash_request`.
- Raw and processed parquet storage.
- Validation reports for required columns, empty datasets, duplicate timestamps, monotonic ordering, missing timestamps, and NaN ratios.
- Dataset registry entries.
- Universe loading and price alignment for experiment execution.

Inputs:

- Dataset profile configs.
- Vendor downloader results.
- Universe symbol lists.

Outputs:

- Raw parquet extracts under `data/raw/`.
- Processed parquet datasets under `data/processed/`.
- Validation JSON reports under `data/external/metadata/`.
- Dataset registry JSON under `data/external/registry/`.
- Aligned price panels for experiments.

Research role:

- Provides a reproducible path from requested market data to validated canonical datasets.
- Makes dataset identity and request provenance explicit.

Maturity:

- Data Agent V1 is implemented structurally.
- Experiment orchestration currently loads local datasets through the dataset loader and alignment utilities; live vendor access is not required for existing generated reports.
- No claim should be made that full data freshness monitoring or point-in-time vendor revision management is production complete.

#### Feature and ML Research Layer

Purpose: deterministic feature construction, label generation, model fitting, prediction, and signal conversion.

Major files:

- `src/features/`
- `src/features/families.py`
- `src/ml/feature_matrix.py`
- `src/ml/datasets.py`
- `src/ml/labels.py`
- `src/ml/models/linear.py`
- `src/ml/models/logistic.py`
- `src/ml/signals/prediction.py`
- `src/strategies/ml_strategy.py`
- `src/experiments/ml_factory.py`

Implemented responsibilities:

- Build feature callables from config.
- Build forward-return, binary-direction, and volatility labels.
- Build linear/logistic model wrappers.
- Adapt ML models into the existing `Strategy` interface.
- Convert predictions into single-asset sign or threshold weights.
- Persist feature registry, feature summary, feature correlations, feature families, alignment diagnostics, ML model diagnostics, and ML provenance.

Inputs:

- Price panel.
- Feature specs.
- Label specs.
- Model specs.
- Signal specs.

Outputs:

- Feature matrix.
- Supervised dataset.
- Predictions.
- Portfolio weights.
- Diagnostic JSON sidecars.
- ML diagnostic figures.

Research role:

- Makes model behavior visible beyond final performance metrics.
- Preserves feature definitions and label horizons.
- Supports feature-family and coefficient stability storytelling.

Maturity:

- Implemented for single-asset ML workflows.
- Panel ranking targets and panel signals are schema-valid but explicitly deferred.

#### Strategy, Portfolio, and Backtesting Layer

Purpose: translate research signals into lagged applied weights and net-of-cost performance series.

Major files:

- `src/strategies/runner.py`
- `src/strategies/momentum_rotation.py`
- `src/strategies/ml_strategy.py`
- `src/portfolio/portfolio_backtest.py`
- `src/portfolio/allocation.py`
- `src/portfolio/ranking.py`
- `src/backtesting/metrics.py`

Implemented responsibilities:

- Strategy execution through `run_strategy`.
- Vectorized portfolio backtest.
- Weight lagging through `weights.shift(1)` to prevent look-ahead.
- Transaction cost application in basis points.
- Standard performance metrics: annualized return, annualized volatility, Sharpe ratio, max drawdown, Calmar ratio, hit rate.
- Portfolio turnover and allocation history figures.

Inputs:

- Price matrix.
- Strategy object.
- Transaction cost setting.

Outputs:

- Daily backtest time series.
- Lagged applied weights.
- Scalar metrics.
- Equity curve, returns, weights parquet files.
- Portfolio and performance figures.

Research role:

- Makes execution timing explicit.
- Ties reported metrics to actual applied weights rather than raw signals.
- Creates a repeatable historical simulation boundary.

Maturity:

- Implemented for vectorized research backtests.
- Execution layer under `src/execution/` is still placeholder; do not imply order lifecycle simulation, fill modeling, or broker integration.

#### Validation Layer

Purpose: chronological train/test splitting, walk-forward evaluation, and stability summarization.

Major files:

- `src/validation/splits.py`
- `src/validation/walk_forward.py`
- `src/validation/stability.py`

Implemented responsibilities:

- Rolling and expanding time splits.
- Chronological train/test windows with optional gap days.
- Walk-forward validation over strategy or ML strategy objects.
- Fit hook for ML strategies per training window.
- Test-window-only metric computation.
- Split-level metrics and stability summaries.
- Walk-forward equity curves and validation figures.

Inputs:

- Datetime index.
- Validation config.
- Prices.
- Strategy.

Outputs:

- `TimeSplit` objects.
- `WalkForwardResult`.
- `diagnostics/split_metrics.json`.
- `diagnostics/wf_equity_curves.json`.
- Validation figures, including stitched OOS equity, split Sharpes, train vs test Sharpe, and walk-forward timeline.

Research role:

- Preserves chronology and makes overfitting/regime dependence visible.
- Provides the strongest implementation-backed evidence for validation rigor.

Maturity:

- Implemented and tested.
- It evaluates historical research workflows; it is not a live monitoring system.

#### Artefact Persistence Layer

Purpose: persist experiment outputs, diagnostics, provenance, reports, and registry entries in a human-inspectable filesystem structure.

Major files:

- `src/experiments/results.py`
- `src/experiments/tracking.py`
- `src/experiments/registry.py`
- `src/experiments/orchestrator.py`
- `results/experiments/canonical_ml_showcase/`

Implemented responsibilities:

- Save core experiment outputs: metadata, metrics, equity curve, returns, weights.
- Save configs: raw and normalized.
- Save plots and plot index.
- Save diagnostics and research sidecars.
- Save ML provenance.
- Register experiments in `registry.json`.

Representative persisted structure:

- `metadata.json`
- `metrics.json`
- `equity_curve.parquet`
- `returns.parquet`
- `weights.parquet`
- `raw_config.yaml`
- `normalized_config.json`
- `ml_provenance.json`
- `diagnostics/*.json`
- `research/*.json`
- `plots/*.png`
- `plots/plot_index.json`

Research role:

- Turns an experiment into an inspectable dossier.
- Preserves intermediate states that are usually lost in notebooks.
- Provides frontend-readable primitives.

Maturity:

- Implemented and present in generated experiments.
- `ExperimentTracker` has optional MLflow support but is no-op by default; filesystem persistence is the primary implemented tracking mechanism.

#### Reporting Layer

Purpose: generate reproducible research reports from saved artefacts without recomputing experiment results.

Major files:

- `src/reporting/report_builder.py`
- `src/reporting/markdown.py`
- `src/reporting/html.py`
- `src/reporting/report_spec.py`
- `src/reporting/consistency.py`
- `scripts/generate_report.py`
- `reports/markdown/canonical_ml_showcase.md`
- `reports/html/canonical_ml_showcase.html`

Implemented responsibilities:

- Load saved experiment artefacts.
- Copy figures into report output.
- Render markdown and optional HTML.
- Write report provenance sidecar.
- Write frontend-facing manifest JSON.
- Govern section inclusion through immutable `ResearchReportSpec` presets.
- Surface publication consistency warnings when active research components are omitted.

Inputs:

- Saved experiment directory.
- Report preset.

Outputs:

- Markdown report.
- HTML report.
- Report provenance JSON.
- Frontend manifest JSON.
- Copied report figures.

Research role:

- Converts an experiment into a research publication object.
- Separates report generation from experiment recomputation.
- Makes chronology, validation, diagnostics, figures, and provenance available to the frontend.

Maturity:

- Implemented and actively used in generated reports.

#### Visualization Layer

Purpose: generate research figures from already-computed experiment and diagnostic data.

Major files:

- `src/visualization/backtest_plots.py`
- `src/visualization/portfolio_plots.py`
- `src/visualization/validation_plots.py`
- `src/visualization/ml_plots.py`
- `src/visualization/universe_plots.py`
- `results/experiments/canonical_ml_showcase/plots/plot_index.json`
- `reports/figures/canonical_ml_showcase/`

Implemented responsibilities:

- Performance figures.
- Portfolio allocation and turnover figures.
- Validation figures.
- ML diagnostic figures.
- Feature and universe diagnostic figures.
- Semantic plot ordering, groups, importance levels, and captions in `plot_index.json`.

Research role:

- Makes intermediate research evidence visible.
- Supports a frontend narrative around figures as evidence, not decoration.

Maturity:

- Implemented for the canonical ML showcase and momentum examples.

#### AI and Agent Readiness Layer

Purpose: define structured agent roles and limited AI-assisted request translation.

Major files:

- `src/agents/interfaces.py`
- `agents/`
- `configs/ai_agents/research_reviewer.yaml`
- `skills/`
- `src/llm/translator.py`
- `src/llm/schemas.py`

Implemented responsibilities:

- Agent task protocol.
- Role descriptions and ownership boundaries.
- Placeholder research reviewer config requiring structured artefacts.
- Natural language data request translator using the OpenAI API.

Research role:

- Provides role boundaries and structured handoff concepts.
- Supports an honest claim that the repository is AI-agent-ready around structured artefacts.

Maturity:

- Partially implemented.
- Do not claim autonomous research orchestration, autonomous portfolio management, or AI-generated alpha production.
- Honest frontend language: "AI-assisted orchestration readiness" or "structured agent handoff surfaces."

### 2. Dependency and Relationship Mapping

Actual execution relationship for v1 strategy experiments:

Config file -> load/validate/normalize -> factory builds strategy, universe spec, validation spec, experiment spec -> load local universe data -> align prices -> run strategy -> optional walk-forward validation -> build experiment result -> build plots -> save artefacts -> write configs and diagnostics -> register experiment -> optional report generation.

Actual execution relationship for v2 ML experiments:

Config file -> validate/normalize ML config -> build ML experiment spec -> build feature functions, label function, model, signal function -> build ML strategy -> load and align prices -> fit strategy on full period -> run strategy -> optional walk-forward validation with a fresh ML strategy instance -> prepare feature, ML, and universe diagnostics -> build plots -> save artefacts -> write ML provenance, configs, diagnostics, research sidecars -> register experiment -> optional report generation.

Report relationship:

Saved experiment directory -> load metadata, metrics, configs, diagnostics, research sidecars, plot index, figures -> copy figures -> render markdown -> render HTML -> write provenance -> write frontend manifest.

Data ingestion relationship:

Dataset profile config -> ingestion config and universe config -> data requests -> request hashes -> downloader -> standardizer -> validator -> raw parquet, processed parquet, validation report -> dataset registry.

### 3. Persistence Structure

The platform has two major persistence surfaces.

Experiment persistence:

- `results/experiments/<experiment_name>/metadata.json`
- `results/experiments/<experiment_name>/metrics.json`
- `results/experiments/<experiment_name>/equity_curve.parquet`
- `results/experiments/<experiment_name>/returns.parquet`
- `results/experiments/<experiment_name>/weights.parquet`
- `results/experiments/<experiment_name>/raw_config.yaml`
- `results/experiments/<experiment_name>/normalized_config.json`
- `results/experiments/<experiment_name>/ml_provenance.json`
- `results/experiments/<experiment_name>/diagnostics/*.json`
- `results/experiments/<experiment_name>/research/*.json`
- `results/experiments/<experiment_name>/plots/*.png`
- `results/experiments/<experiment_name>/plots/plot_index.json`
- `results/experiments/registry.json`

Report persistence:

- `reports/markdown/<experiment_name>.md`
- `reports/markdown/<experiment_name>_manifest.json`
- `reports/markdown/<experiment_name>_provenance.json`
- `reports/html/<experiment_name>.html`
- `reports/figures/<experiment_name>/*.png`

Dataset persistence:

- `data/raw/<source>/<data_type>/<symbol>/<timestamped_extract>.parquet`
- `data/processed/<family>/<symbol>/<frequency>.parquet`
- `data/external/metadata/*_validation.json`
- `data/external/registry/datasets.json`

### 4. Reporting Structure

The reporting system is implemented as an archival renderer, not a live dashboard.

Report generation is read-only with respect to saved experiment artefacts. It does not recompute results, reload data, refit models, or rerun validation. This is important for frontend positioning: reports represent a frozen experiment state.

Implemented report sections include:

- Summary
- Research Thesis and Methodology
- Universe Construction and Coverage
- Data Infrastructure
- Feature Engineering
- Backtesting Methodology
- Portfolio Construction Process
- ML Model Behaviour
- Performance Metrics
- Walk-Forward Validation
- Failure Analysis
- Diagnostics Appendix
- Metadata
- Configuration
- Figures
- Provenance

The manifest provides frontend-ready fields:

- experiment name
- experiment type
- report spec
- tags
- markdown/html/provenance paths
- figure paths
- metrics summary
- rendered sections
- validation verdict
- plot index
- figure hierarchy
- feature/diagnostic capability booleans

## Part B - Research Lifecycle Audit

### 1. Chronological Research Lifecycle

The real lifecycle implemented by the platform is:

1. Research intent is declared in config.
2. Config is loaded, validated, and normalized.
3. Typed specs and strategy/model components are constructed.
4. Local dataset panels are loaded and aligned.
5. Feature and label datasets are built for ML experiments.
6. Strategy or ML model is fitted where applicable.
7. Signals are converted into weights.
8. Portfolio backtest applies lagged weights and transaction costs.
9. Walk-forward validation is executed over chronological splits.
10. Diagnostics are computed from already-available run objects.
11. Core artefacts, diagnostics, research sidecars, plots, configs, and provenance are persisted.
12. Experiment registry is updated.
13. Report generation reads the frozen experiment directory.
14. Markdown, HTML, provenance, manifest, and frontend figures are written.

### 2. Stage-by-Stage Explanation

#### Stage 1 - Config Declaration

The experiment begins as a YAML/JSON contract. In `canonical_ml_showcase.yaml`, the declared materials include universe, date range, feature list, label horizon, model type, signal type, validation window, transaction cost, and output paths.

Observability point:

- Raw config is copied into the experiment artefact directory as `raw_config.yaml`.

#### Stage 2 - Validation and Normalization

The platform separates raw config loading, structural validation, and normalization. Defaults are filled only after validation, and normalized configs are persisted.

Observability point:

- `normalized_config.json` exposes the exact execution contract used by the run.

#### Stage 3 - Component Construction

Factories build strategies, universe specs, validation specs, ML features, labels, models, and signal functions. The factories are intentionally pure: they do not load data or write files.

Observability point:

- ML component identity is persisted in `ml_provenance.json`.

#### Stage 4 - Data Loading and Alignment

Experiments load local datasets through the dataset loader, then align prices into a Date x Asset panel. The default alignment policy is inner join.

Observability point:

- `research/data_summary.json`
- `diagnostics/universe_coverage.json`
- universe coverage and volatility figures where available.

#### Stage 5 - Feature and Label Construction

For v2 ML experiments, feature callables generate a feature matrix, labels generate target series, and alignment removes warm-up/label-incomplete rows.

Observability point:

- `research/feature_registry.json`
- `research/feature_summary.json`
- `research/alignment_diagnostics.json`
- `research/feature_correlations.json`
- `research/feature_families.json`

#### Stage 6 - Model Fitting and Signal Generation

`MLStrategy.fit()` builds a supervised dataset and fits the configured model. `generate_weights()` builds features, predicts, and converts predictions to weights.

Observability point:

- `diagnostics/ml_model_diagnostics.json`
- prediction distribution figure
- prediction vs actual figure
- residual diagnostics
- coefficient stability/evolution figures

#### Stage 7 - Portfolio Backtest

The strategy runner computes returns, obtains weights, and runs the portfolio backtest. The backtest shifts weights by one period before applying returns, preventing same-period look-ahead.

Observability point:

- `weights.parquet`
- `returns.parquet`
- `equity_curve.parquet`
- `metrics.json`
- `diagnostics/backtest_diagnostics.json`
- allocation, turnover, rolling Sharpe, volatility, equity/drawdown figures.

#### Stage 8 - Walk-Forward Validation

If validation is enabled, rolling or expanding time splits are generated. For ML strategies, fitting occurs only on the train slice. The strategy sees data only up to `test_end`, and metrics are computed only on the test window.

Observability point:

- `diagnostics/split_metrics.json`
- `diagnostics/wf_equity_curves.json`
- walk-forward stitched equity figure
- split Sharpe figure
- train vs test Sharpe figure
- walk-forward timeline figure.

#### Stage 9 - Research and Diagnostic Persistence

The orchestrator writes diagnostics after the experiment has already run. These are consumers of computed objects, not hidden reruns.

Observability point:

- research sidecars in `research/`
- diagnostics sidecars in `diagnostics/`
- semantic plot index in `plots/plot_index.json`.

#### Stage 10 - Report Publication

Report generation reads the saved experiment directory and renders markdown/HTML. It copies figures and writes manifest/provenance sidecars.

Observability point:

- `reports/markdown/*`
- `reports/html/*`
- `reports/markdown/*_manifest.json`
- `reports/markdown/*_provenance.json`
- `reports/figures/*`.

### 3. Research Observability Points

Best frontend evidence surfaces:

- Raw and normalized configs.
- ML provenance hash and component specs.
- Feature registry and feature families.
- Alignment diagnostics.
- Feature correlations.
- Split metrics and walk-forward windows.
- Backtest diagnostics.
- Universe coverage diagnostics.
- Signal transitions.
- Plot index with semantic grouping and captions.
- Report manifest with rendered sections and figure hierarchy.

### 4. Candidate Frontend Lifecycle Narrative

Use this structure:

Configured hypothesis -> validated experiment contract -> aligned research dataset -> feature and label construction -> model or signal formation -> lagged portfolio simulation -> chronological walk-forward validation -> persisted diagnostics -> generated research dossier -> frontend rendering.

Avoid "raw data to dashboard." The platform is closer to "research claim to reproducible dossier."

### 5. Suggested Lifecycle Visual Grouping

Group 1: Experiment Contract

- raw config
- normalized config
- typed specs
- ML hash

Group 2: Research Construction

- dataset alignment
- feature registry
- label horizon
- model/signal wiring

Group 3: Historical Evaluation

- lagged weights
- transaction costs
- metrics
- drawdowns

Group 4: Chronological Validation

- rolling/expanding splits
- train/test windows
- split metrics
- OOS equity

Group 5: Research Publication

- diagnostics JSON
- figures
- report
- manifest
- provenance.

## Part C - Research Visibility Audit

### 1. Research Visibility Inventory

Implemented visibility surfaces:

- Experiment metadata: `metadata.json`
- Performance metrics: `metrics.json`
- Applied weights: `weights.parquet`
- Returns: `returns.parquet`
- Equity curve: `equity_curve.parquet`
- Raw execution config: `raw_config.yaml`
- Normalized execution config: `normalized_config.json`
- ML component provenance: `ml_provenance.json`
- Data summary: `research/data_summary.json`
- Signal transitions: `research/signal_transitions.json`
- Feature registry: `research/feature_registry.json`
- Feature summary: `research/feature_summary.json`
- Feature families: `research/feature_families.json`
- Feature correlations: `research/feature_correlations.json`
- Alignment diagnostics: `research/alignment_diagnostics.json`
- Backtest diagnostics: `diagnostics/backtest_diagnostics.json`
- Split metrics: `diagnostics/split_metrics.json`
- Walk-forward equity curves: `diagnostics/wf_equity_curves.json`
- ML diagnostics: `diagnostics/ml_diagnostics.json`
- ML model diagnostics: `diagnostics/ml_model_diagnostics.json`
- Universe coverage: `diagnostics/universe_coverage.json`
- Semantic figure index: `plots/plot_index.json`
- Report manifest: `reports/markdown/*_manifest.json`
- Report provenance: `reports/markdown/*_provenance.json`

### 2. Artefact Persistence Inventory

Core persisted evidence:

- Filesystem snapshots under `results/experiments/<experiment_name>/`.
- Markdown/HTML publication outputs under `reports/`.
- JSON sidecars for diagnostics and provenance.
- Parquet files for time-series outputs.
- PNG figures for visual diagnostics.
- Registry JSON for experiment discovery.

Not persisted or only partially persisted:

- Full in-memory feature matrices are not saved as parquet in the canonical ML run.
- Fitted model binary artefacts are not saved.
- Random seed state is not centrally recorded, though current linear/logistic workflows are mostly deterministic by construction.
- Config hash in report provenance is currently null, even though experiment registry computes config hashes and ML provenance stores `ml_hash`.

### 3. Intermediate Research Evidence Map

Research issue -> persisted evidence:

- Data coverage -> `data_summary.json`, `universe_coverage.json`, universe coverage heatmap.
- Feature definitions -> `feature_registry.json`, `ml_provenance.json`.
- Feature availability and warm-up loss -> `feature_summary.json`, `alignment_diagnostics.json`.
- Feature collinearity -> `feature_correlations.json`, feature correlation heatmap.
- Feature family structure -> `feature_families.json`, family-aware plots where generated.
- Model coefficient behavior -> `ml_model_diagnostics.json`, coefficient stability/evolution/sign heatmap figures.
- Prediction quality -> rolling IC, directional accuracy, prediction vs actual, residual diagnostics.
- Signal behavior -> `ml_diagnostics.json`, `signal_transitions.json`, signal turnover figure.
- Portfolio implementation -> `weights.parquet`, allocation history, portfolio turnover.
- Performance path -> `equity_curve.parquet`, equity/drawdown, rolling Sharpe, rolling volatility.
- Validation chronology -> `split_metrics.json`, `wf_equity_curves.json`, walk-forward timeline.
- Report publication state -> report manifest and provenance JSON.

### 4. Suggested Frontend Research Visibility Narrative

The platform does not ask the viewer to trust a single final metric. It exposes the chain of evidence behind a research result: config, dataset coverage, feature construction, alignment loss, prediction behavior, portfolio transition, chronological validation, diagnostic figures, and a generated report manifest.

This is the most authentic differentiator. It is supported by actual artefacts, not only architecture notes.

### 5. Candidate Inputs for Research Visibility Diagram

Nodes:

- Experiment config
- Normalized config
- ML provenance
- Data summary
- Feature registry
- Alignment diagnostics
- Feature correlations
- Model diagnostics
- Backtest diagnostics
- Split metrics
- Walk-forward equity
- Plot index
- Report manifest
- Report provenance

Relationships:

- Config creates normalized execution contract.
- Execution contract produces run outputs.
- Run outputs produce diagnostics and research sidecars.
- Diagnostics and figures feed report builder.
- Report builder produces frontend manifest and provenance.

Visual focus:

- Evidence trail, not pipeline complexity.
- Show sidecars as visible research states.
- Use chronological bands: before run, during run, after run, publication.

