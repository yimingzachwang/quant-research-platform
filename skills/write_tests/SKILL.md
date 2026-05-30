# write_tests

## Purpose

Create focused tests for infrastructure contracts and future strategy components.

## Expected Inputs

- Module path
- Behavior or contract under test
- Required fixtures
- Risk level

## Expected Outputs

- Pytest tests
- Minimal fixtures
- Coverage notes
- Any residual test gaps

## Workflow

1. Identify whether the test is unit, contract, integration, or smoke.
2. Prefer deterministic toy data.
3. Avoid live data and network calls.
4. Assert behavior, not implementation details.
5. Run `pytest` and relevant static checks.
