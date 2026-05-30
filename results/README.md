# Results Directory

Generated research outputs belong here:

- `backtests/`: backtest artifacts and summaries
- `experiments/`: per-experiment artefact trees (metrics, equity curves, plots, provenance)
- `plots/`: generated charts and figures
- `metrics/`: metric snapshots and diagnostics
- `logs/`: local run logs

All contents are gitignored and must be generated locally by running experiments.

---

## Test Suite — Artefact-Dependent Failures

**179 tests in this repository require locally generated artefacts to pass.** These tests are not broken; they test the orchestration and retrieval layer against real experiment outputs that do not exist in a fresh checkout.

### Affected test files

| File | Failure count | Root cause |
|---|---|---|
| `tests/integration/test_workflow.py` | 96 | `DatasetNotFoundError` — requires `data/processed/ohlcv/` market data |
| `tests/experiments/test_ml_orchestrator.py` | 18 | Missing `results/experiments/canonical_ml_showcase/` artefacts |
| `tests/orchestration/test_retrieval.py` | 13 | Missing `results/experiments/canonical_ml_multi_asset/` artefacts |
| `tests/orchestration/test_research_api.py` | 13 | Missing `results/experiments/` registry entries |
| `tests/orchestration/test_context_builder.py` | 13 | Missing ML diagnostic artefacts |
| `tests/orchestration/test_comparison_engine.py` | 11 | Missing `results/experiments/canonical_ml_*` artefacts |
| `tests/orchestration/test_registry.py` | 6 | Missing experiment registry entries |
| `tests/orchestration/test_rendering.py` | 4 | Missing artefact paths for report generation |
| `tests/orchestration/test_iteration.py` | 3 | Missing artefacts for LLM iteration context |
| `tests/orchestration/test_evolution.py` | 2 | Missing experiment lineage artefacts |

### Resolution

Run the two canonical experiments to generate the required artefacts:

```bash
python scripts/update_dataset.py --config configs/data/daily_prices.yaml
python scripts/run_from_config.py configs/experiments/canonical_ml_showcase.yaml --report --preset canonical
python scripts/run_from_config.py configs/experiments/canonical_ml_multi_asset.yaml --report --preset canonical
```

After running these, the 179 artefact-dependent tests will pass. The remaining 2,092 tests pass on a fresh checkout without any data or artefacts.
