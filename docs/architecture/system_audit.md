# System Audit

Audit basis: repository inspection of source, tests, configs, data registry,
results artifacts, and context docs.

## Findings

### 1. Adjusted Close Schema Mismatch

Status: PARTIAL.

Evidence:

- `configs/data/data_agent_v1.yaml` lists `adjusted_close` in the OHLCV
  canonical schema.
- `src/data/transformers/standardizers.py::OHLCVStandardizer.canonical_columns`
  does not include `adjusted_close`.
- `src/data/validators/dataset_validator.py` does not require
  `adjusted_close`.

Risk: documentation/config claims a column that runtime data may not contain.

### 2. Duplicate Data Registry Concepts

Status: PARTIAL/LEGACY.

Evidence:

- Manifest path: `DatasetManifest`, `DatasetRegistry`,
  `data/external/registry/datasets.json`.
- Legacy path: `DatasetMetadata`, `JsonDatasetRegistry`,
  `src/data/update_engine/engine.py`.

Risk: future code may register or query different metadata shapes.

### 3. Backtest Orchestration Placeholder

Status: PLACEHOLDER.

Evidence: `BacktestEngine.run()` returns `BacktestResult(context=context)` with
empty artifacts and metrics.

Risk: callers may assume `BacktestEngine` is the canonical engine when the real
implemented path is `run_backtest`, `run_portfolio_backtest`, or `run_strategy`.

### 4. Coexisting Experiment Paths

Status: PARTIAL.

Evidence:

- Legacy `ExperimentRunner` and `ExperimentContext`.
- D0 `ExperimentSpec`, `save_run`, `ExperimentRegistry`.
- D1 `run_experiment_from_config`.
- D2 reporting.

Risk: multiple valid entrypoints can confuse contributors. Current docs should
point to `run_from_config.py` for config-driven research runs.

### 5. Placeholder Packages Beside Implemented ML Package

Status: PARTIAL.

Evidence:

- `src/ml` contains implemented contracts, datasets, labels, sklearn wrappers,
  metrics, and walk-forward prediction pipeline.
- `src/models` contains generic protocol/no-op placeholders.

Risk: "models" vs "ml" ownership may be unclear.

### 6. LLM Translator Is Experimental

Status: EXPERIMENTAL.

Evidence:

- `src/llm/translator.py` calls OpenAI directly at import/runtime.
- It returns `LLMDataRequest`, not current `DataRequest`.
- No tests found for `src/llm`.

Risk: bypassing validated ingestion contracts if promoted without adapter tests.

### 7. Documentation / Script Interface Drift

Status: PARTIAL.

Evidence:

- `README.md` shows `python scripts/run_experiment.py --config ...`.
- `scripts/run_experiment.py` does not parse `--config`; it uses constants
  embedded in the script.
- `scripts/run_from_config.py` is the implemented config-file entrypoint.

Risk: onboarding users may run the wrong command for config-driven
experiments.

### 8. Placeholder Risk/Execution/Evaluation

Status: PLACEHOLDER.

Evidence: interfaces and no-op implementations exist, but no real risk,
execution simulator, or evaluator implementation was found.

Risk: architecture docs must not describe these as implemented systems.

## Positive Architectural Evidence

- Explicit anti-lookahead lagging in single-asset and portfolio backtests.
- Chronological split generation with train/test ordering.
- Registry-backed dataset loading with clear errors.
- Filesystem-first experiment artifacts.
- Reporting is read-only over saved artifacts.
- Broad test coverage across data, features, backtesting, portfolio, strategies,
  validation, ML, experiments, visualization, and reporting.
