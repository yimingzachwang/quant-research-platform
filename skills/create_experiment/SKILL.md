# create_experiment

## Purpose

Create a reproducible config-driven experiment manifest.

## Expected Inputs

- Experiment name and owner
- Universe
- Date range
- Horizon
- Data, feature, model, portfolio, risk, and report configs

## Expected Outputs

- YAML experiment config
- Experiment context preview
- Required validation checklist

## Workflow

1. Generate an experiment ID using `exp_<topic>_<yyyymmdd>_<short_hash>`.
2. Reference existing modular configs where possible.
3. Make tracking and artifact paths explicit.
4. Validate required fields before running.
