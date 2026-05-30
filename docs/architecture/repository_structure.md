# Repository Structure

This map reflects the visible repository structure at audit time.

```text
configs/        Declarative YAML for data, experiments, portfolio, reports, risk, MLflow.
data/           Local data lake: raw, processed, features, external registry/metadata.
docs/           Human and agent-facing documentation.
experiments/    Notebook-like or script-like research experiments.
results/        Generated metrics, plots, experiment artifacts, comparison outputs.
scripts/        Thin CLI-style entrypoints.
src/            Reusable package code imported as `src.*`.
tests/          Unit, contract, and smoke tests.
agents/         Role descriptions for AI-agent collaboration.
skills/         Agent skill documentation.
```

## Important `src/` Packages

| Package | Status | Role |
|---|---:|---|
| `src/data` | STABLE/PARTIAL | Data requests, configs, ingestion engine, storage, validation, registry, loaders |
| `src/cleaning` | STABLE | Local cleaning and OHLCV validation helpers |
| `src/features` | STABLE | Pure feature transformations |
| `src/backtesting` | STABLE/PARTIAL | Vectorized backtest functions, metrics, signal helpers, placeholder engine |
| `src/portfolio` | STABLE/PARTIAL | Alignment, panel features, ranking, allocation, portfolio backtest |
| `src/strategies` | STABLE | Strategy contracts and baseline/momentum implementations |
| `src/validation` | STABLE | Chronological split generation and walk-forward validation |
| `src/ml` | EXPERIMENTAL/STABLE | Supervised dataset contracts, labels, models, prediction pipeline |
| `src/experiments` | STABLE/PARTIAL | Config IO, factory, orchestration, persistence, registry |
| `src/visualization` | STABLE | Read-only matplotlib plots |
| `src/reporting` | STABLE | Static markdown/HTML report generation from saved artifacts |
| `src/llm` | EXPERIMENTAL | Natural-language data request translator |
| `src/models` | PLACEHOLDER | Generic model protocols/no-op implementations |
| `src/signals` | PLACEHOLDER | Generic signal protocol/no-op implementation |
| `src/risk` | PLACEHOLDER | Risk protocol/no-op implementation |
| `src/execution` | PLACEHOLDER | Execution/cost protocols/no-op implementation |
| `src/evaluation` | PLACEHOLDER | Evaluation protocol/no-op implementation |

## Known Import/Ownership Notes

- Current scripts import `src.portfolio.alignment` and `src.portfolio.panel`;
  no `src.portfolio.universe` or `src.portfolio.returns` files exist.
- `src/data/engine.py` is the current profile ingestion engine.
  `src/data/update_engine/engine.py` remains as a legacy metadata path.
- `src/ml` is an actual implemented research layer; `src/models` remains a
  separate placeholder protocol package.

