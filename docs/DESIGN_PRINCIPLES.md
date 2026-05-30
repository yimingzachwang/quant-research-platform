# Design Principles

## Research Discipline

- Reproducibility before claims.
- Leakage prevention by construction: lag signals/weights, preserve
  chronological validation, keep target construction separate, and record
  assumptions.
- Prefer vectorized pandas research workflows for low/mid-frequency ETF work.
- Promote reusable logic into `src/`; notebooks and scripts should remain
  orchestration or exploration layers.

## Simplicity Rules

- Prefer small functions, dataclasses, Pydantic models, and explicit pandas
  transformations.
- Keep CLIs thin and inspectable.
- Keep storage filesystem-based until a real need appears.
- Treat registry metadata as the source of truth for persisted datasets.
- Extend existing abstractions before adding new ones.

## Anti-Overengineering Rules

- Do not add ORMs, database migrations, service layers, plugin systems,
  dependency-injection frameworks, or async orchestration for current needs.
- Do not create parallel request schemas, registries, storage resolvers, or
  validation systems.
- Do not hide strategy logic inside data loading, plotting, or reporting.
- Do not make visualization mutate data or recompute core analytics.

## Boundary Rules

- Data owns ingestion, validation, storage, manifests, registry, and loading.
- Features own deterministic transformations and lookback semantics.
- Portfolio owns alignment, ranking, allocation, and multi-asset weights.
- Strategies generate weights; backtests evaluate weights.
- Validation owns chronological split design and out-of-sample diagnostics.
- Backtesting owns timing assumptions, transaction costs, equity curves, and
  metrics.
- Experiments own reproducible run configuration, orchestration, persistence,
  and registry metadata.
- Reporting owns presentation of saved artifacts only; it must not rerun
  experiments or recompute metrics.
- Visualization consumes computed results and returns/saves figures.
- LLM/agent code may translate intent or summarize artifacts but must not bypass
  deterministic contracts.

## Documentation Discipline

- Keep `docs/context/` aligned with the repository whenever major systems
  change.
- Mark functionality as implemented, partial, or planned.
- Keep docs concise, technical, and grounded in files that exist.
