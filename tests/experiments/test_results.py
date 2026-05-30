"""Tests for src/experiments/results.py."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiments.results import ExperimentResult, save_experiment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_result() -> ExperimentResult:
    idx = pd.date_range("2021-01-01", periods=50, freq="B")
    rng = np.random.default_rng(1)
    net_ret = pd.Series(rng.normal(0.0003, 0.01, 50), index=idx, name="net_return")
    equity = (1.0 + net_ret).cumprod()
    weights = pd.DataFrame(
        {"A": 0.5, "B": 0.5}, index=idx
    )
    return ExperimentResult(
        experiment_name="test_exp",
        strategy_name="TestStrategy",
        parameters={"lookback": 20, "top_n": 2},
        metrics={"annualized_return": 0.08, "sharpe_ratio": 0.6},
        weights=weights,
        equity_curve=equity,
        returns=net_ret,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# save_experiment — filesystem structure
# ---------------------------------------------------------------------------


def test_save_creates_experiment_folder(
    sample_result: ExperimentResult, tmp_path: Path
) -> None:
    out = save_experiment(sample_result, output_dir=tmp_path)
    assert out.is_dir()
    assert out.name == sample_result.experiment_name


def test_save_returns_path_to_experiment_folder(
    sample_result: ExperimentResult, tmp_path: Path
) -> None:
    out = save_experiment(sample_result, output_dir=tmp_path)
    assert out == tmp_path / sample_result.experiment_name


def test_save_writes_all_four_files(
    sample_result: ExperimentResult, tmp_path: Path
) -> None:
    out = save_experiment(sample_result, output_dir=tmp_path)
    assert (out / "metadata.json").exists()
    assert (out / "metrics.json").exists()
    assert (out / "equity_curve.parquet").exists()
    assert (out / "weights.parquet").exists()


# ---------------------------------------------------------------------------
# metadata.json content
# ---------------------------------------------------------------------------


def test_metadata_json_is_valid(
    sample_result: ExperimentResult, tmp_path: Path
) -> None:
    out = save_experiment(sample_result, output_dir=tmp_path)
    with (out / "metadata.json").open() as f:
        meta = json.load(f)
    assert meta["experiment_name"] == sample_result.experiment_name
    assert meta["strategy_name"] == sample_result.strategy_name
    assert meta["parameters"] == sample_result.parameters
    assert "created_at" in meta


def test_metadata_created_at_is_iso_string(
    sample_result: ExperimentResult, tmp_path: Path
) -> None:
    out = save_experiment(sample_result, output_dir=tmp_path)
    with (out / "metadata.json").open() as f:
        meta = json.load(f)
    # Should parse back to a datetime without error
    dt = datetime.fromisoformat(meta["created_at"])
    assert dt == sample_result.created_at


# ---------------------------------------------------------------------------
# metrics.json content
# ---------------------------------------------------------------------------


def test_metrics_json_is_valid(
    sample_result: ExperimentResult, tmp_path: Path
) -> None:
    out = save_experiment(sample_result, output_dir=tmp_path)
    with (out / "metrics.json").open() as f:
        metrics = json.load(f)
    assert metrics["annualized_return"] == pytest.approx(0.08)
    assert metrics["sharpe_ratio"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Parquet roundtrip
# ---------------------------------------------------------------------------


def test_equity_curve_parquet_roundtrip(
    sample_result: ExperimentResult, tmp_path: Path
) -> None:
    out = save_experiment(sample_result, output_dir=tmp_path)
    loaded = pd.read_parquet(out / "equity_curve.parquet")["equity_curve"]
    pd.testing.assert_series_equal(loaded, sample_result.equity_curve, check_names=False, check_freq=False)


def test_weights_parquet_roundtrip(
    sample_result: ExperimentResult, tmp_path: Path
) -> None:
    out = save_experiment(sample_result, output_dir=tmp_path)
    loaded = pd.read_parquet(out / "weights.parquet")
    pd.testing.assert_frame_equal(loaded, sample_result.weights, check_freq=False)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_save_is_idempotent(
    sample_result: ExperimentResult, tmp_path: Path
) -> None:
    save_experiment(sample_result, output_dir=tmp_path)
    save_experiment(sample_result, output_dir=tmp_path)  # overwrite — must not raise
    with (tmp_path / sample_result.experiment_name / "metrics.json").open() as f:
        metrics = json.load(f)
    assert "annualized_return" in metrics


# ---------------------------------------------------------------------------
# Numpy scalar serialisation
# ---------------------------------------------------------------------------


def test_numpy_scalars_in_metrics_serialise(tmp_path: Path) -> None:
    idx = pd.date_range("2021-01-01", periods=10, freq="B")
    import numpy as np
    result = ExperimentResult(
        experiment_name="np_test",
        strategy_name="S",
        parameters={"n": np.int64(3)},
        metrics={"r": float(np.float64(0.05))},
        weights=pd.DataFrame({"A": 1.0}, index=idx),
        equity_curve=pd.Series(np.ones(10), index=idx),
        returns=pd.Series(np.zeros(10), index=idx),
    )
    out = save_experiment(result, output_dir=tmp_path)
    with (out / "metadata.json").open() as f:
        meta = json.load(f)
    assert meta["parameters"]["n"] == 3
