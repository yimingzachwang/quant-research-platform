"""Tests for save_run / load_run in src/experiments/tracking.py."""

import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from src.experiments.config import ExperimentSpec
from src.experiments.results import ExperimentResult
from src.experiments.tracking import load_run, save_run

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def spec() -> ExperimentSpec:
    return ExperimentSpec(
        experiment_name="tracking_test",
        strategy_name="EqualWeight",
        universe=["SPY", "TLT"],
        start_date="2020-01-01",
        end_date="2023-12-31",
        rebalance_frequency="ME",
        parameters={"rebalance_freq": "ME"},
        tags=["baseline"],
    )


@pytest.fixture()
def result() -> ExperimentResult:
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    rng = np.random.default_rng(42)
    net_ret = pd.Series(rng.normal(0.0003, 0.01, 60), index=idx)
    equity = (1.0 + net_ret).cumprod()
    weights = pd.DataFrame({"SPY": 0.5, "TLT": 0.5}, index=idx)
    return ExperimentResult(
        experiment_name="tracking_test",
        strategy_name="EqualWeight",
        parameters={"rebalance_freq": "ME"},
        metrics={"annualized_return": 0.07, "sharpe_ratio": 0.55,
                 "max_drawdown": -0.04, "annualized_volatility": 0.10,
                 "calmar_ratio": 1.75, "hit_rate": 0.52},
        weights=weights,
        equity_curve=equity,
        returns=net_ret,
        created_at=datetime(2026, 5, 22, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# save_run — folder structure
# ---------------------------------------------------------------------------


def test_save_run_returns_path(result: ExperimentResult, tmp_path: Path) -> None:
    out = save_run(result, output_dir=tmp_path)
    assert isinstance(out, Path)
    assert out.is_dir()


def test_save_run_creates_required_files(result: ExperimentResult, tmp_path: Path) -> None:
    out = save_run(result, output_dir=tmp_path)
    for fname in ["metadata.json", "metrics.json", "equity_curve.parquet",
                  "returns.parquet", "weights.parquet"]:
        assert (out / fname).exists(), f"Missing: {fname}"


def test_save_run_creates_plots_dir(result: ExperimentResult, tmp_path: Path) -> None:
    out = save_run(result, output_dir=tmp_path)
    assert (out / "plots").is_dir()


def test_save_run_creates_diagnostics_dir(result: ExperimentResult, tmp_path: Path) -> None:
    out = save_run(result, output_dir=tmp_path)
    assert (out / "diagnostics").is_dir()


def test_save_run_without_spec_no_config_json(result: ExperimentResult, tmp_path: Path) -> None:
    out = save_run(result, spec=None, output_dir=tmp_path)
    assert not (out / "config.json").exists()


def test_save_run_with_spec_writes_config_json(
    result: ExperimentResult, spec: ExperimentSpec, tmp_path: Path
) -> None:
    out = save_run(result, spec=spec, output_dir=tmp_path)
    assert (out / "config.json").exists()
    with (out / "config.json").open() as f:
        data = json.load(f)
    assert data["experiment_name"] == spec.experiment_name


def test_save_run_with_plots(result: ExperimentResult, tmp_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    out = save_run(result, output_dir=tmp_path, plots={"equity_curve": fig})
    plt.close(fig)
    assert (out / "plots" / "equity_curve.png").exists()


def test_save_run_with_predictions(result: ExperimentResult, tmp_path: Path) -> None:
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    preds = pd.DataFrame({"SPY": np.random.default_rng(1).normal(size=10)}, index=idx)
    out = save_run(result, output_dir=tmp_path, predictions=preds)
    assert (out / "predictions.parquet").exists()


# ---------------------------------------------------------------------------
# save_run — idempotency / overwrite
# ---------------------------------------------------------------------------


def test_save_run_is_idempotent(result: ExperimentResult, tmp_path: Path) -> None:
    save_run(result, output_dir=tmp_path)
    save_run(result, output_dir=tmp_path)  # overwrite — must not raise
    assert (tmp_path / result.experiment_name / "metrics.json").exists()


# ---------------------------------------------------------------------------
# load_run — roundtrip
# ---------------------------------------------------------------------------


def test_load_run_without_spec(result: ExperimentResult, tmp_path: Path) -> None:
    out = save_run(result, spec=None, output_dir=tmp_path)
    loaded_result, loaded_spec = load_run(out)
    assert isinstance(loaded_result, ExperimentResult)
    assert loaded_spec is None


def test_load_run_with_spec(
    result: ExperimentResult, spec: ExperimentSpec, tmp_path: Path
) -> None:
    out = save_run(result, spec=spec, output_dir=tmp_path)
    loaded_result, loaded_spec = load_run(out)
    assert loaded_spec is not None
    assert isinstance(loaded_spec, ExperimentSpec)
    assert loaded_spec.experiment_name == spec.experiment_name


def test_load_run_result_metrics(
    result: ExperimentResult, tmp_path: Path
) -> None:
    out = save_run(result, output_dir=tmp_path)
    loaded, _ = load_run(out)
    for k, v in result.metrics.items():
        assert abs(loaded.metrics[k] - v) < 1e-10


def test_load_run_spec_roundtrip(
    result: ExperimentResult, spec: ExperimentSpec, tmp_path: Path
) -> None:
    out = save_run(result, spec=spec, output_dir=tmp_path)
    _, loaded_spec = load_run(out)
    assert loaded_spec.strategy_name == spec.strategy_name
    assert loaded_spec.universe == spec.universe
    assert loaded_spec.parameters == spec.parameters
    assert loaded_spec.tags == spec.tags


def test_load_run_missing_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path / "nonexistent")
