# Visualization Philosophy

Visualization is a first-class research subsystem, but it is read-only. It
communicates diagnostics; it does not own computation.

## Implemented

- Backtest plots: equity curves, drawdowns, rolling Sharpe, rolling volatility.
- Signal plots: SMA strategy and position-state diagnostics.
- Distribution plots: return distributions and monthly heatmaps.
- Portfolio plots: weights, rolling weights, weight lines, correlations,
  concentration, turnover, contribution, and heatmaps.
- Strategy comparison plots: equity overlays, drawdown overlays, metric bars,
  and metric tables.
- Validation plots: walk-forward equity curves, stitched out-of-sample equity,
  split Sharpe bars, metric stability, and train/test comparison.
- Style utilities and `save_figure(...)`.
- Demo/result images exist under `results/`.

## Rules

- Plotting functions must accept already-computed series/dataframes/results.
- Do not perform data ingestion, feature generation, strategy execution,
  validation, or backtesting inside plotting functions.
- Return matplotlib figure objects where practical.
- Save figures only when an explicit path/helper is provided.
- Keep plots deterministic for the same input data.
- Keep visual defaults suitable for research review: legible, restrained, and
  diagnostic rather than decorative.

## Preferred Artifacts

- Equity and drawdown plots.
- Rolling risk/Sharpe diagnostics.
- Return distribution and heatmap diagnostics.
- Signal and position-state plots.
- Portfolio weight, turnover, concentration, contribution, and correlation
  diagnostics.
- Strategy comparison figures and metrics tables.
- Walk-forward validation and stability figures.

## Planned Integration

- D1 experiment workflows save plots alongside metrics and configs.
- D2 reports consume saved figures and structured metrics without
  recomputation.
- AI agents may summarize visualization artifacts but should not scrape
  notebooks or regenerate analytics.
