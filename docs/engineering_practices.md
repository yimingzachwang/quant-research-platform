# Engineering Practices

## Repository Practices

- Keep reusable logic in `src/` and import it with absolute `src.*` imports.
- Keep notebooks thin and exploratory.
- Prefer configuration-driven experiments over hard-coded parameters.
- Use typed interfaces for component boundaries.
- Add tests around contracts before adding complex implementations.

## Research Practices

- State the hypothesis before running the experiment.
- Define the universe and horizon explicitly.
- Track all assumptions that affect tradability.
- Evaluate turnover, transaction costs, exposure, drawdown, and stability, not only returns.
- Prefer boring, reproducible workflows over clever one-off scripts.

## Future Tooling Candidates

- `pandas`, `numpy`, `pyarrow` for data and storage
- `scikit-learn`, `lightgbm`, `xgboost` for modelling
- `matplotlib`, `plotly` for visualization
- MLflow or Weights & Biases for experiment tracking
- Ray or Dask for scale-out research jobs
- FastAPI for later service boundaries
- Docker for reproducible environments
