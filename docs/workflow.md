# Research Workflow

## Experiment Lifecycle

1. Define the research question and hypothesis.
2. Select the ETF universe and prediction horizon.
3. Build a point-in-time dataset with explicit data availability assumptions.
4. Generate features using reusable feature definitions.
5. Train or configure a signal generator.
6. Convert signals into portfolio targets.
7. Simulate execution with explicit cost and timing assumptions.
8. Run the backtest and persist artifacts.
9. Evaluate performance, robustness, and diagnostics.
10. Review results with human and AI-assisted analysis.

## Reproducibility Requirements

Each experiment should eventually persist:

- Config snapshot
- Code version
- Data version or source manifest
- Universe definition
- Feature definitions
- Model parameters
- Backtest assumptions
- Metrics and diagnostics
- Generated reports

## Leakage-Aware Practices

- Lag features by default unless a feature explicitly proves same-day availability.
- Separate training, validation, and test periods.
- Record rebalance timing and execution timing.
- Treat survivorship bias and ETF inception dates as first-class data concerns.
- Keep target construction separate from feature construction.

## AI-Agent Touchpoints

AI agents should eventually assist by reading structured artifacts from `experiments` and `evaluation`, then producing summaries, diagnostics, and follow-up research suggestions.

The core platform should remain deterministic without agents.

