# run_backtest

## Purpose

Run a config-driven historical simulation through the platform entry points.

## Expected Inputs

- Experiment config path
- Data availability confirmation
- Backtest assumptions
- Output directory or tracking URI

## Expected Outputs

- Backtest result object
- Metrics dictionary
- Artifact manifest
- MLflow run reference when enabled

## Workflow

1. Validate the experiment config.
2. Confirm universe, date range, horizon, and rebalance policy.
3. Run `scripts/run_experiment.py`.
4. Inspect metrics, warnings, and artifact manifest.
5. Hand off to `evaluate_strategy`.

## Guardrails

- Do not interpret placeholder metrics as alpha evidence.
- Fail closed when required data or assumptions are missing.
