# Testing

Testing is organized around contract confidence rather than strategy performance.

## Test Types

- Unit tests: deterministic behavior within a single function or class.
- Contract tests: module boundary behavior for data, features, signals, models, backtests, and reports.
- Smoke tests: CLI and config loading checks.
- Integration tests: optional tests that connect real datasets or tracking systems.

## Rules

- Do not use live vendor data in default tests.
- Do not require network access in default tests.
- Keep toy data small and explicit.
- Test reproducibility controls before evaluating alpha logic.
