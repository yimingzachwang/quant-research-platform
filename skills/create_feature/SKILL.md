# create_feature

## Purpose

Create a leakage-aware feature definition and placeholder implementation.

## Expected Inputs

- Feature ID
- Data dependencies
- Lookback window
- Alignment rules
- Missing data handling

## Expected Outputs

- Feature config
- Feature class or function placeholder
- Unit or contract tests
- Documentation of leakage assumptions

## Workflow

1. Confirm the feature ID follows `feat_<data_family>_<transform>_<window>`.
2. Define required raw fields and calendar alignment.
3. Add a placeholder under `src/features/` if needed.
4. Add tests using deterministic toy data.
5. Document NaN, lagging, and point-in-time behavior.
