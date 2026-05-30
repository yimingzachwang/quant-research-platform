# System Overview

Repository source of truth: files under `src/`, `tests/`, `configs/`, `data/`,
`scripts/`, `results/`, and existing project docs inspected on 2026-05-23.

## Platform Type

This repository is a quantitative ETF research platform with AI-agent-facing
documentation and workflow scaffolding. It is research-first, filesystem-based,
and deterministic where practical. It is not a live trading system.

The implemented system supports local dataset ingestion, registry-backed
loading, deterministic feature functions, vectorized backtests, simple
portfolio construction, strategy comparison, chronological validation, early
ML research contracts, visualization, experiment persistence, and static
reporting.

## Philosophy

- Reproducibility before performance claims.
- Vectorized research before complex simulation.
- Explicit timing assumptions and one-period signal/weight lagging.
- Filesystem artifacts and JSON/parquet outputs over hidden state.
- Registry metadata as the source of truth for persisted datasets.
- Visualization is read-only and consumes computed artifacts.
- LLM/agent code should produce typed configs, requests, or summaries, not
  bypass deterministic contracts.

## Capability Matrix

| Capability | Status | Evidence |
|---|---:|---|
| Data request contracts | STABLE | `src/data/contracts/requests.py`, tests in `tests/data/` |
| Profile ingestion engine | STABLE/PARTIAL | `src/data/engine.py`, `configs/data/*.yaml`; adjusted-close mismatch remains |
| Registry-backed dataset loading | STABLE | `src/data/manifest.py`, `src/data/registry/dataset_registry.py`, `src/data/loaders/dataset_loader.py` |
| Legacy data metadata path | PARTIAL | `src/data/models/metadata.py`, `src/data/registry/json_registry.py` |
| Cleaning utilities | STABLE | `src/cleaning/*`, `tests/cleaning/*` |
| Feature functions | STABLE | `src/features/*`, `tests/features/*` |
| Signal helper functions | STABLE | `src/backtesting/signals.py` |
| `src/signals` package | PLACEHOLDER | `src/signals/interfaces.py`, `src/signals/placeholders.py` |
| Strategy layer | STABLE | `src/strategies/*`, `tests/strategies/*` |
| Single-asset backtest function | STABLE | `src/backtesting/engine.py::run_backtest` |
| `BacktestEngine` orchestration class | PLACEHOLDER | returns empty `BacktestResult` from context |
| Portfolio construction utilities | STABLE/PARTIAL | `src/portfolio/*`; no optimizer or constraints engine |
| Walk-forward validation | STABLE | `src/validation/*`, `tests/validation/*` |
| ML datasets/contracts/linear models | EXPERIMENTAL/STABLE | `src/ml/*`, `tests/ml/*`; early research layer, not production model platform |
| Model/risk/evaluation/execution top-level packages | PLACEHOLDER | `src/models`, `src/risk`, `src/evaluation`, `src/execution` |
| Experiment artifacts and registry | STABLE | `src/experiments/results.py`, `tracking.py`, `registry.py` |
| Config-driven experiment orchestration | STABLE/PARTIAL | `src/experiments/config_io.py`, `factory.py`, `orchestrator.py` |
| Static reporting | STABLE | `src/reporting/report_builder.py`, `markdown.py`, `html.py` |
| Visualization subsystem | STABLE | `src/visualization/*`, `tests/visualization/*` |
| LLM natural-language data requests | EXPERIMENTAL | `src/llm/*`, `scripts/run_nl_data_request.py` |
| Live trading | PLANNED/FUTURE | no broker/execution implementation found |

## Intended Workflow

1. Ingest/update datasets from `configs/data/*.yaml`.
2. Load registered datasets through `load_dataset()` or portfolio alignment
   helpers.
3. Compute features, labels, signals, or strategy weights with pure functions.
4. Run vectorized backtests with explicit costs and one-period lagging.
5. Validate chronologically using rolling/expanding splits.
6. Persist experiment artifacts under `results/experiments/`.
7. Generate plots and static markdown/HTML reports from saved artifacts.

## Future Direction

Future work is documented in context and scaffold files, but not all of it is
implemented. The likely direction is a registry-driven research system that can
connect raw datasets, feature datasets, labels, model predictions, strategy
outputs, validation results, reports, and agent review notes through explicit
metadata.

