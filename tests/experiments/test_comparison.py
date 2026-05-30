"""Tests for src/experiments/comparison.py."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiments.comparison import (
    compare_experiments,
    load_and_compare,
    metrics_delta,
    metrics_table,
    rank_experiments,
)
from src.experiments.results import ExperimentResult, save_experiment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result(name: str, sharpe: float = 0.6, ret: float = 0.08) -> ExperimentResult:
    idx = pd.date_range("2020-01-01", periods=30, freq="B")
    rng = np.random.default_rng(hash(name) % 2**32)
    net_ret = pd.Series(rng.normal(0.0003, 0.01, 30), index=idx)
    equity = (1.0 + net_ret).cumprod()
    weights = pd.DataFrame({"A": 0.5, "B": 0.5}, index=idx)
    return ExperimentResult(
        experiment_name=name,
        strategy_name=f"Strategy_{name}",
        parameters={},
        metrics={"annualized_return": ret, "sharpe_ratio": sharpe,
                 "max_drawdown": -0.05, "annualized_volatility": 0.10,
                 "calmar_ratio": abs(ret) / 0.05, "hit_rate": 0.52},
        weights=weights,
        equity_curve=equity,
        returns=net_ret,
        created_at=datetime(2026, 5, 22, tzinfo=UTC),
    )


@pytest.fixture()
def three_results() -> dict[str, ExperimentResult]:
    return {
        "exp_a": _make_result("exp_a", sharpe=1.0, ret=0.10),
        "exp_b": _make_result("exp_b", sharpe=0.6, ret=0.07),
        "exp_c": _make_result("exp_c", sharpe=-0.1, ret=-0.02),
    }


# ---------------------------------------------------------------------------
# compare_experiments
# ---------------------------------------------------------------------------


def test_compare_dict_passthrough(three_results: dict) -> None:
    out = compare_experiments(three_results)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"exp_a", "exp_b", "exp_c"}


def test_compare_list_uses_experiment_name() -> None:
    results = [_make_result("r1"), _make_result("r2")]
    out = compare_experiments(results)
    assert "r1" in out
    assert "r2" in out


def test_compare_list_duplicate_name_raises() -> None:
    r1 = _make_result("dup")
    r2 = _make_result("dup")
    with pytest.raises(ValueError, match="Duplicate"):
        compare_experiments([r1, r2])


# ---------------------------------------------------------------------------
# metrics_table
# ---------------------------------------------------------------------------


def test_metrics_table_returns_dataframe(three_results: dict) -> None:
    t = metrics_table(three_results)
    assert isinstance(t, pd.DataFrame)


def test_metrics_table_index_is_labels(three_results: dict) -> None:
    t = metrics_table(three_results)
    assert set(t.index) == set(three_results.keys())


def test_metrics_table_has_sharpe_column(three_results: dict) -> None:
    t = metrics_table(three_results)
    assert "sharpe_ratio" in t.columns


def test_metrics_table_values_correct(three_results: dict) -> None:
    t = metrics_table(three_results)
    assert t.loc["exp_a", "sharpe_ratio"] == pytest.approx(1.0)
    assert t.loc["exp_c", "sharpe_ratio"] == pytest.approx(-0.1)


def test_metrics_table_accepts_list() -> None:
    results = [_make_result("p"), _make_result("q")]
    t = metrics_table(results)
    assert len(t) == 2


# ---------------------------------------------------------------------------
# rank_experiments
# ---------------------------------------------------------------------------


def test_rank_experiments_returns_dataframe(three_results: dict) -> None:
    r = rank_experiments(three_results, by="sharpe_ratio")
    assert isinstance(r, pd.DataFrame)


def test_rank_experiments_has_rank_column(three_results: dict) -> None:
    r = rank_experiments(three_results, by="sharpe_ratio")
    assert "rank" in r.columns


def test_rank_experiments_rank_starts_at_1(three_results: dict) -> None:
    r = rank_experiments(three_results, by="sharpe_ratio")
    assert r["rank"].iloc[0] == 1


def test_rank_experiments_descending_sharpe(three_results: dict) -> None:
    r = rank_experiments(three_results, by="sharpe_ratio", ascending=False)
    sharpes = r["sharpe_ratio"].values
    assert (sharpes[:-1] >= sharpes[1:]).all()


def test_rank_experiments_ascending(three_results: dict) -> None:
    r = rank_experiments(three_results, by="sharpe_ratio", ascending=True)
    sharpes = r["sharpe_ratio"].values
    assert (sharpes[:-1] <= sharpes[1:]).all()


def test_rank_experiments_invalid_metric_raises(three_results: dict) -> None:
    with pytest.raises(ValueError, match="not found"):
        rank_experiments(three_results, by="nonexistent_metric")


# ---------------------------------------------------------------------------
# load_and_compare
# ---------------------------------------------------------------------------


def test_load_and_compare_returns_dict(tmp_path: Path) -> None:
    r1 = _make_result("exp_x")
    r2 = _make_result("exp_y")
    p1 = save_experiment(r1, output_dir=tmp_path)
    p2 = save_experiment(r2, output_dir=tmp_path)
    out = load_and_compare([p1, p2])
    assert isinstance(out, dict)
    assert len(out) == 2


def test_load_and_compare_with_labels(tmp_path: Path) -> None:
    r1 = _make_result("a")
    p1 = save_experiment(r1, output_dir=tmp_path)
    out = load_and_compare([p1], labels=["my_label"])
    assert "my_label" in out


def test_load_and_compare_labels_mismatch_raises(tmp_path: Path) -> None:
    r1 = _make_result("a")
    p1 = save_experiment(r1, output_dir=tmp_path)
    with pytest.raises(ValueError, match="labels length"):
        load_and_compare([p1], labels=["x", "y"])


def test_load_and_compare_nonexistent_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_and_compare([tmp_path / "ghost"])


# ---------------------------------------------------------------------------
# metrics_delta
# ---------------------------------------------------------------------------


def test_metrics_delta_returns_dict() -> None:
    baseline = _make_result("base", sharpe=0.5, ret=0.06)
    candidate = _make_result("new", sharpe=0.8, ret=0.10)
    delta = metrics_delta(baseline, candidate)
    assert isinstance(delta, dict)


def test_metrics_delta_sharpe_positive(three_results: dict) -> None:
    baseline = three_results["exp_c"]
    candidate = three_results["exp_a"]
    delta = metrics_delta(baseline, candidate)
    assert delta["sharpe_ratio"] > 0


def test_metrics_delta_only_shared_keys() -> None:
    r1 = _make_result("r1")
    r1.metrics["extra_metric"] = 1.0
    r2 = _make_result("r2")
    delta = metrics_delta(r1, r2)
    assert "extra_metric" not in delta
