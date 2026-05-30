# Experiment Tracking

Status: STABLE/PARTIAL.

## Implemented Tracks

The experiment layer has multiple coexisting paths:

1. Legacy `ExperimentContext`/`ExperimentRunner` scaffold.
2. D0 filesystem-first experiment result persistence.
3. D1 config-driven experiment orchestration.
4. D2 static report generation from saved artifacts.

## D0 Persistence

`ExperimentResult` contains experiment name, strategy name, parameters,
metrics, applied weights, equity curve, returns, and creation timestamp.

`save_experiment()` writes:

- `metadata.json`
- `metrics.json`
- `equity_curve.parquet`
- `returns.parquet`
- `weights.parquet`

`save_run()` adds:

- `config.json` when an `ExperimentSpec` is provided.
- `predictions.parquet` when predictions are provided.
- `plots/`
- `diagnostics/`

## D1 Orchestration

`run_experiment_from_config(path)`:

1. Loads raw YAML/JSON.
2. Validates config structure.
3. Normalizes defaults.
4. Builds strategy, universe spec, validation config, and experiment spec.
5. Loads data through portfolio alignment helpers.
6. Runs strategy.
7. Optionally runs walk-forward validation.
8. Builds `ExperimentResult`.
9. Generates plots.
10. Saves artifacts.
11. Writes raw and normalized configs.
12. Registers the run in `ExperimentRegistry`.

## Experiment Registry

`ExperimentRegistry` stores a flat JSON file with experiment name, hash,
timestamp, strategy name, tags, summary metrics, and path.

## Reporting

`generate_experiment_report()` reads saved experiment artifacts, copies saved
figures, renders markdown/HTML, and writes provenance JSON. It does not rerun
experiments or recompute metrics.

## Limitations

- Registry replaces by experiment name, not by full immutable run ID.
- ML predictions can be saved by `save_run()`, but D1 does not currently pass
  predictions through.
- MLflow adapter exists but is optional/no-op unless enabled and installed.

