# Quant Research Platform

Full-stack Python infrastructure for systematic ETF research — typed data contracts, leakage-safe ML pipelines, walk-forward validation, AI-assisted research orchestration, and a production-grade API layer — engineered for reproducibility at every layer.

[![CI](https://github.com/USERNAME/quant-research-platform/actions/workflows/ci.yaml/badge.svg)](https://github.com/USERNAME/quant-research-platform/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-2271%20passing-brightgreen.svg)](tests/)

---

## Research Showcase

Three fully executed experiments demonstrate the platform's research lifecycle from hypothesis formation to validated diagnostic output. Each report covers data infrastructure, feature engineering, model methodology, walk-forward validation, IC regime analysis, and structured risk factor analysis. Sharpe ratios are included in the reports; they are not the point.

**[Canonical ML Showcase](reports/markdown/canonical_ml_showcase.md)** · SPY · 2013–2024  
Single-asset Ridge regression trained on a 7-feature taxonomy spanning momentum, realised volatility, z-score normalisation, and trend strength. Demonstrates the complete supervised ML research cycle: leakage-safe feature alignment, L2-regularised model training, walk-forward validation across rolling 48-month windows, information coefficient regime decomposition, and cross-split coefficient stability analysis.

**[Canonical ML Multi-Asset](reports/markdown/canonical_ml_multi_asset.md)** · 9-ETF cross-sectional universe · 2013–2024  
Cross-sectional rank prediction across US equities, international equities, rates, commodities, and sectors (SPY, QQQ, IWM, EEM, TLT, GLD, XLF, XLK, XLE) using a 13-feature set extending the single-asset taxonomy with rolling market beta, risk-adjusted momentum, volatility compression ratio, trend persistence, drawdown distance from peak, and breakout proximity. Demonstrates cross-sectional label construction, top-N signal translation, and multi-asset walk-forward diagnostics.

**[Momentum Rotation](reports/markdown/example_momentum_rotation.md)** · 7-ETF cross-sectional universe · 2015–2024  
Rule-based trailing 252-day momentum signal with month-end rebalancing across equity, fixed income, and commodity ETFs. Demonstrates signal construction, rebalance policy design, regime-conditional performance characterisation, and a structured risk factor analysis covering momentum crash exposure, concentration risk, lookback sensitivity, and transaction cost drag.

> Controlled allocation research (softmax temperature study, 4-arm design) is documented in [`reports/allocation_study/`](reports/allocation_study/). Signal geometry analysis (Ridge regularisation α sweep, 4-arm design) is documented in [`reports/signal_geometry/`](reports/signal_geometry/).

---

## What This Platform Is

This repository is a research engineering platform for systematic, reproducible quantitative research on ETF universes. It implements the full research lifecycle — from raw data ingestion and schema validation through feature engineering, ML model training, portfolio construction, backtesting, walk-forward validation, and publication-quality reporting — as a layered, typed, testable Python codebase. The source tree spans 216 files and 33,671 lines of production code; the test suite spans 124 files, 23,621 lines, and 2,271 passing tests.

The platform is designed around reproducibility by construction. Every experiment is defined by a declarative YAML configuration, executed against a versioned data registry, and produces a provenance-tracked artefact set: a metrics JSON, ML diagnostic outputs, walk-forward stability summaries, and a rendered research report with embedded figures. A research session that cannot be reproduced from its configuration and registry state is not a valid research session.

This is not a trading system. The repository contains no broker integration, order management, live position tracking, or production risk controls. It is the research infrastructure that precedes those systems — the platform on which strategy hypotheses are formed, tested against historical data, validated out-of-sample, and reviewed before any deployment decision is made.

---

## Platform Architecture

The platform is organised as seven layered subsystems. Each layer consumes typed inputs from the layer below and exposes typed interfaces to the layer above. No layer reaches past its immediate dependency boundary — this constraint is enforced by automated tests.

```
┌──────────────────────────────────────────────────────────────────────┐
│  API Layer                                                           │
│  FastAPI bridge · Research API · typed request/response schemas      │
│  HTTP transport only — no business logic in routing                  │
├──────────────────────────────────────────────────────────────────────┤
│  Orchestration Layer                                                 │
│  LLM review engine · config synthesis · session management           │
│  intent parsing · research routing · context building · retrieval    │
├──────────────────────────────────────────────────────────────────────┤
│  Reporting · Visualization                                           │
│  Provenance-tracked markdown/HTML reports · IC diagnostics           │
│  Publication-quality figures · coefficient and regime plots          │
├──────────────────────────────────────────────────────────────────────┤
│  Experiment · Backtesting                                            │
│  YAML config orchestration · MLflow tracking · artefact persistence  │
│  Vectorized historical simulation · walk-forward evaluation          │
├──────────────────────────────────────────────────────────────────────┤
│  ML · Portfolio · Strategies                                         │
│  Model training/inference · walk-forward pipelines · diagnostics     │
│  Cross-sectional portfolio construction · allocation · constraints   │
├──────────────────────────────────────────────────────────────────────┤
│  Features · Cleaning · Signals                                       │
│  Leakage-safe deterministic transformations · family taxonomy        │
│  Timestamp normalisation · missing value policy · signal translation │
├──────────────────────────────────────────────────────────────────────┤
│  Data Layer                                                          │
│  Typed contracts · dataset registry · download adapters              │
│  Schema validation · freshness checks · DatasetQuery resolution      │
└──────────────────────────────────────────────────────────────────────┘
```

> Mermaid source diagrams for each layer boundary, the full system dependency graph, ML experiment lifecycle, and validation workflow are maintained in [`docs/architecture/`](docs/architecture/).

**Repository structure:**

```text
quant-research-platform/
├── agents/          # AI research agent role definitions and collaboration protocols
├── configs/         # Declarative YAML for data, features, models, experiments, portfolios
├── data/            # Local data lake (raw · processed · features · external — contents gitignored)
├── docker/
├── docs/
│   ├── architecture/        # Layer design, extension patterns, Mermaid system diagrams
│   ├── ai_orchestration/    # Orchestration layer design and rationale
│   ├── diagrams/            # Rendered system diagrams
│   ├── manuals/             # Platform manual, reporting framework, visualization philosophy
│   ├── DESIGN_PRINCIPLES.md
│   └── VISUALIZATION_PHILOSOPHY.md
├── reports/         # Executed experiment output (figures · markdown · HTML · provenance JSON)
│   ├── allocation_study/
│   ├── figures/
│   ├── html/
│   ├── markdown/
│   └── signal_geometry/
├── results/         # Runtime artefact storage (gitignored; see results/README.md)
├── scripts/         # Runnable research workflow entrypoints
├── skills/          # Reusable AI-agent procedures for repeatable research tasks
├── src/
│   ├── api/             # FastAPI bridge and HTTP routers
│   ├── backtesting/     # Vectorized backtesting and performance evaluation
│   ├── cleaning/        # Data cleaning and normalisation
│   ├── data/            # Data contracts, registry, download adapters, loaders
│   ├── experiments/     # Config loading, orchestration, MLflow tracking hooks
│   ├── features/        # Deterministic, leakage-safe feature transformations
│   ├── ml/              # Models, walk-forward pipelines, diagnostics, signals
│   ├── orchestration/   # AI research orchestration layer (see below)
│   ├── portfolio/       # Cross-sectional portfolio construction and allocation
│   ├── reporting/       # Artefact-driven report generation
│   ├── signals/         # Signal construction and translation
│   ├── strategies/      # Strategy definitions and cross-strategy comparison
│   ├── validation/      # Walk-forward splits and out-of-sample stability diagnostics
│   └── visualization/   # Publication-quality figure rendering
├── tests/           # 124 test files mirroring the src/ structure
├── .github/
├── AGENTS.md
├── Dockerfile
├── docker-compose.yaml
└── pyproject.toml
```

---

## AI Orchestration Layer

`src/orchestration/` implements AI-assisted research workflows over the quantitative research infrastructure. It is not an autonomous system. Every action is human-initiated, every output is typed and validated, and every state change is recorded as a session event. The layer sits between LLM reasoning and the research pipeline — providing structure, not autonomy.

| Subsystem | Role |
|---|---|
| **LLM Review Engine** | Generates structured research reviews of experiment output: signal quality assessment, methodology gap identification, suggested refinements. Output follows a typed schema (`ReviewSchema`). Supports multiple LLM providers via a unified interface. |
| **Comparison Engine** | Produces structured comparative analysis across two or more experiment runs, normalising for universe and feature differences. |
| **Config Synthesis** | Translates LLM-proposed experiment modifications into draft configuration objects (`ExperimentDraft`). Drafts pass through schema validation, hash verification, human approval, and YAML rendering before any configuration is written. |
| **Session Management** | Tracks a research iteration as a typed event sequence — every review generated, draft created, draft approved, and config rendered is a session event with a timestamp and structured payload. Sessions are persisted and reloadable. |
| **Intent Parser** | Parses natural-language research requests into typed intent objects (`ExperimentIntent`, `ComparisonIntent`, `ReviewIntent`, `RefinementIntent`) using few-shot LLM prompting with example-grounded schema enforcement. |
| **Research Router** | Dispatches parsed intents to the appropriate research workflow and returns structured results with elapsed time and error context. |
| **Context Builder** | Constructs structured experiment context from artefacts for LLM consumption — metrics, configuration, feature registry, walk-forward diagnostics, and failure mode signals. Includes specialised summarisers for ML diagnostics and validation outputs. |
| **Evolution Tracker** | Records how experiments evolve across research iterations — which features were added or removed, which hyperparameters changed, and what the LLM review found at each step. |
| **Retrieval** | Finds similar historical experiments by feature overlap, universe composition, or model class — used to provide comparative context to LLM review without manual cross-referencing. |

The FastAPI bridge (`src/api/`) exposes the orchestration layer over HTTP. Routers call only Research API functions — they do not import orchestration utilities, quant engine internals, or experiment execution logic directly. This boundary is enforced by parametrized architectural tests that verify the import graph of every router module at test time, not at review time.

See [`docs/ai_orchestration/overview.md`](docs/ai_orchestration/overview.md) for design rationale, the session lifecycle model, and the draft approval workflow.

---

## Engineering Standards

**Test suite.** 2,271 tests across 124 files — unit, contract, integration, and architectural boundary tests. Test distribution spans every platform layer: data (7), features (9), ML (14), orchestration (15), experiments (13), API (8), visualization (11), and more. Test count is comparable to source line count: the discipline is not bolted on.

**Architectural enforcement.** `tests/api/test_api_non_coupling.py` parametrizes over all five router modules and programmatically verifies that no router imports quant engine internals, executes experiment logic, calls lineage registration, or spawns background tasks. Architectural constraints are enforced by the test suite, not by convention.

**Static analysis.** `ruff` for linting and import ordering · `black` for formatting · `mypy` in strict mode across all `src/` packages · `pre-commit` runs the full check sequence before every commit.

**CI.** Every push and pull request triggers ruff, black, and the full pytest suite on Python 3.11 via GitHub Actions.

---

## Research Design Principles

**Leakage prevention by construction.** Features are pure functions of prices available at or before the observation date. Labels are constructed separately, aligned to features with explicit temporal lags, and never touched during feature computation. The data pipeline enforces a strict chronological contract: no future information enters any training window at any stage.

**Walk-forward validation over in-sample fitting.** Experiments are evaluated using rolling or expanding time splits with held-out out-of-sample test windows. In-sample metrics are diagnostics. The walk-forward artefacts — split-level metrics tables, IC stability summaries, regime-conditional performance — are the primary evaluation deliverables.

**Provenance-tracked, reproducible experiments.** Every experiment execution produces a provenance sidecar: configuration hash, data registry state, artefact manifest, and run timestamp. Experiment reports embed their own manifest. A result that cannot be traced to a specific configuration and data registry version is not a valid result.

**Methodology as product.** Research reports open with an explicit hypothesis, provide economic rationale for each feature, document the model choice, enumerate known failure modes, and scope the analysis explicitly. A Sharpe ratio without this context is not research — it is a number.

---

## Setup

Clone the repository and create a local Python environment:

```bash
git clone https://github.com/USERNAME/quant-research-platform.git
cd quant-research-platform
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[research,dev]"
pre-commit install
```

Copy the environment template and supply API keys for the LLM orchestration layer:

```bash
cp .env.example .env
# Edit .env — add OPENAI_API_KEY and/or ANTHROPIC_API_KEY
```

Run the full check suite:

```bash
ruff check .
black --check .
pytest
```

Run a config-driven experiment end-to-end:

```bash
python scripts/run_from_config.py configs/experiments/canonical_ml_showcase.yaml --report --preset canonical
```

Use absolute imports throughout:

```python
from src.data.loaders import load_dataset
from src.features import compute_returns, rolling_volatility
from src.ml.models.linear import RidgeRegressionModel
from src.orchestration.api import research_api
```

---

## Docker

Build and run the research environment in a container:

```bash
docker compose run --rm research
```

Start MLflow experiment tracking locally:

```bash
docker compose up mlflow
```

---

## Data Infrastructure

Market data is sourced from Yahoo Finance via `yfinance` and persisted to local Parquet files under `data/processed/`. The data layer is built around a typed registry: every dataset is registered with a manifest covering source, frequency, schema, content hash, and last-updated timestamp. All downstream code loads data via `DatasetQuery` against the registry rather than by direct file path — this decouples the research stack from storage layout and is a prerequisite for reproducibility across machines and collaborators.

Data directories (`data/raw/`, `data/processed/`, `data/features/`, `data/external/`) are gitignored. The repository ships directory placeholders and the registry schema; data is populated locally using the provided scripts:

```bash
python scripts/update_dataset.py --config configs/data/daily_prices.yaml
python scripts/validate_data.py
python scripts/list_datasets.py
```

Raw data is treated as immutable once downloaded. Cleaning, normalisation, and feature computation produce outputs in separate paths and do not modify source files.
