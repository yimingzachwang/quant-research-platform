# Test Plan: src.experiments.config

## Scope

Validate experiment YAML parsing into typed context objects.

## Test Type

Unit and contract.

## Cases

- Loads required fields from example config.
- Rejects missing date range.
- Preserves nested config metadata.

## Fixtures

Static YAML files under `configs/experiments/`.

## Residual Risk

Does not validate live data availability.
