# Experiment Framework

Experiments are YAML manifests that compose modular configs.

The minimal lifecycle is:

1. Load `configs/experiments/<experiment>.yaml`.
2. Resolve universe, data, feature, model, portfolio, risk, backtest, report, and tracking references.
3. Build an `ExperimentContext`.
4. Run the configured workflow through `ExperimentRunner`.
5. Track parameters, metrics, and artifacts through MLflow when enabled.
6. Generate a report with assumptions, limitations, and next steps.

The current framework validates architecture and reproducibility. It does not implement alpha logic.
