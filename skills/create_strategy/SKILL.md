# create_strategy

## Purpose

Create the scaffold for a new strategy idea without implementing alpha logic.

## Expected Inputs

- Strategy name and ID
- Research hypothesis
- Universe config path
- Horizon
- Feature and signal families
- Risk and portfolio constraints

## Expected Outputs

- Strategy config draft
- Research brief
- Placeholder strategy module
- Test plan

## Workflow

1. Validate naming follows `strat_<asset_scope>_<signal_family>_<horizon>`.
2. Create or update a config under `configs/experiments/`.
3. Add a placeholder module under `src/strategies/`.
4. Record assumptions, exclusions, and required data.
5. Hand off feature work to `create_feature` and simulation work to `run_backtest`.

## Guardrails

- Do not add production trading logic.
- Do not claim expected performance.
- Keep hypotheses falsifiable.
