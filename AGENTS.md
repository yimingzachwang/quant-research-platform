# AGENTS.md

This repository is an institutional-style, AI-assisted quantitative research platform for ETF research and systematic trading. It is architecture-first: no production alpha logic should be added until data contracts, experiment controls, validation, and review workflows are in place.

## Architecture

The platform is organized as layered research infrastructure:

- `configs/`: declarative YAML for universes, data, features, models, portfolios, risk, reports, MLflow, and experiments.
- `src/`: reusable Python package with typed interfaces and placeholder implementations. Use absolute imports such as `from src.data import DataAgent`.
- `scripts/`: thin command-line entry points that call package code.
- `experiments/`: experiment manifests, run notes, and artifact references.
- `skills/`: reusable AI-agent skills for repeatable research and engineering tasks.
- `agents/`: subagent role definitions and collaboration boundaries.
- `docs/`: human-readable system design, workflows, standards, and decision records.
- `tests/`: unit, contract, and smoke tests.

Primary package boundaries:

- `data` and `ingestion`: external and internal data access, schema validation, dataset metadata, and freshness checks.
- `features`: leakage-aware deterministic feature transformations.
- `signals`: conversion of features and model outputs into tradable research signals.
- `models`: model training, inference contracts, validation schemes, and artifact metadata.
- `portfolio`: portfolio construction, constraints, rebalancing policies, and target weights.
- `execution`: transaction costs, fill simulation, and order lifecycle assumptions.
- `backtesting`: historical orchestration across data, features, signals, portfolio, execution, and evaluation.
- `evaluation`: performance, risk, robustness, turnover, and diagnostics.
- `risk`: ex-ante and ex-post risk analysis.
- `reporting`: reproducible research reports and experiment summaries.
- `experiments`: config loading, context creation, MLflow tracking hooks, and workflow orchestration.
- `agents`: reusable agent task contracts and handoff metadata under `src/agents`; orchestration role docs remain in top-level `agents/`.

## Coding Standards

- Use Python 3.11+ with `src/` layout.
- Prefer dataclasses, protocols, and explicit domain objects for cross-module contracts.
- Keep entry points thin; put business logic in `src/`.
- Do not implement alpha strategies in infrastructure scaffolding.
- Keep functions deterministic where possible and pass configuration explicitly.
- Avoid hidden global state. Any run-level state must be carried by `ExperimentContext`.
- Use precise names: `feature`, `signal`, `forecast`, `target_weight`, `fill`, and `metric` are not interchangeable.
- Use structured parsers for configs and data contracts. Avoid ad hoc string parsing.
- Add comments only for non-obvious decisions, assumptions, or risk controls.

## Testing Standards

- Every module should have at least smoke or contract tests before production use.
- Test interfaces with minimal fake implementations before connecting real vendors or compute backends.
- Use pytest for unit and contract tests.
- Add regression tests for experiment configuration parsing and artifact naming.
- Tests must not rely on live market data, live broker APIs, or network availability unless explicitly marked.
- Strategy research tests should separate infrastructure correctness from alpha performance claims.

## Tooling Standards

- `ruff` is used for linting and import ordering.
- `black` is used for formatting.
- `pytest` is used for tests.
- `mypy` should remain strict for package code as the platform matures.
- `pre-commit` runs formatting and static checks before commits.
- Docker is used for reproducible local and CI execution.
- MLflow records experiment metadata, parameters, metrics, and artifacts.

## Workflow Expectations

1. Start with a config in `configs/experiments/`.
2. Validate universe, date range, data dependencies, and leakage assumptions.
3. Run through package orchestration rather than notebook-only logic.
4. Persist run metadata and artifacts using stable identifiers.
5. Evaluate performance, risk, turnover, capacity assumptions, and failure modes.
6. Generate a report that states assumptions, limitations, and next steps.
7. Request agent review for modules outside your ownership boundary.

## Agent Responsibilities

- `research_manager`: owns research briefs, priorities, acceptance criteria, and cross-agent handoffs.
- `quant_analyst`: owns hypotheses, feature ideas, signal definitions, diagnostics, and economic interpretation.
- `data_engineer`: owns data ingestion, schemas, quality checks, lineage, and dataset freshness.
- `ml_engineer`: owns model training contracts, validation design, model artifacts, and MLflow integration.
- `backtesting_agent`: owns simulation assumptions, cost models, portfolio transitions, and backtest correctness.
- `reporting_agent`: owns reports, experiment summaries, visual diagnostics, and investment committee narratives.
- `infra_agent`: owns packaging, CI, Docker, pre-commit, dependency hygiene, and operational reproducibility.

Agents should write down assumptions, modify only their owned areas unless coordinated, and leave clear handoff notes in PRs or experiment logs.

## Module Ownership

- Data ingestion and schemas: `data_engineer`
- Feature engineering and signal research: `quant_analyst`
- Model training and validation: `ml_engineer`
- Backtesting, execution, and portfolio plumbing: `backtesting_agent`
- Evaluation and risk diagnostics: `quant_analyst`, `backtesting_agent`
- Reports and presentation artifacts: `reporting_agent`
- Tooling, CI, containers, and repository hygiene: `infra_agent`
- Experiment lifecycle and prioritization: `research_manager`

## Naming Conventions

- Python packages and modules: `snake_case`.
- Classes: `PascalCase`.
- Functions, methods, variables: `snake_case`.
- Experiment IDs: `exp_<topic>_<yyyymmdd>_<short_hash>`.
- Strategy IDs: `strat_<asset_scope>_<signal_family>_<horizon>`.
- Feature IDs: `feat_<data_family>_<transform>_<window>`.
- Model IDs: `model_<family>_<target>_<horizon>`.
- Config files: descriptive `snake_case.yaml`.
- Artifact paths: `artifacts/<experiment_id>/<run_id>/...`.

## Guardrails

- No look-ahead data, survivorship assumptions, or silent restatements.
- No live trading or broker execution in this scaffold.
- No performance claims without reproducible configs and diagnostics.
- No notebook-only research promotion without package-level implementation and tests.
- No secrets in configs; use environment variables or secret managers.

Documentation reports are external artefacts, not repository state.

Agents must NOT modify:
- docs/context/
- governance files
- workflow instructions
- architecture memory
- handoff state

unless explicitly requested.

Repository-inspection reports should instead be generated as standalone
deliverables outside the operational documentation tree.
