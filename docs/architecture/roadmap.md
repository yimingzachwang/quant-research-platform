# Roadmap

This roadmap separates implemented reality from plausible next steps.

## Near Term

- Fix current script import drift and keep scripts aligned with
  `src.portfolio.alignment` / `src.portfolio.panel`.
- Resolve the adjusted-close inconsistency between config schema and runtime
  OHLCV standardization/validation.
- Consolidate manifest registry and legacy metadata registry semantics.
- Integrate ML predictions with experiment artifacts and reporting only after
  contracts remain stable.
- Materialize feature datasets through the existing registry path.

## Medium Term

- Add feature/label/split manifests linked to dataset manifests.
- Add model artifact persistence and model-run provenance.
- Expand risk and evaluation beyond placeholders.
- Add richer strategy diagnostics and capacity/turnover analysis.
- Extend reporting from single-experiment reports to registry-driven summaries.

## Long Term / Future

- Agent review workflows over structured artifacts.
- Model-driven signal generation and portfolio integration.
- Optional execution simulation beyond turnover costs.
- Live trading only after research, validation, risk, and execution boundaries
  are mature.

## Deferred

- Database registries.
- Distributed orchestration.
- Broker integrations.
- Autonomous trading.
- Performance claims without reproducible configs and diagnostics.

