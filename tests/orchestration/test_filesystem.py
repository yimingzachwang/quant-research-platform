"""Tests for orchestration.utils.filesystem path resolution."""

from pathlib import Path

import pytest

from src.orchestration.utils.filesystem import (
    all_diagnostics_paths,
    config_path,
    diagnostics_dir,
    equity_curve_path,
    experiment_root,
    experiments_root,
    list_experiments,
    llm_context_path,
    llm_review_path,
    metadata_path,
    metrics_path,
    ml_provenance_path,
    plot_index_path,
    plot_path,
    plots_dir,
    report_figures_dir,
    report_manifest_path,
    research_dir,
    returns_path,
    weights_path,
)

_BASE = Path("results/experiments")
_NAME = "test_exp"


def test_experiment_root():
    assert experiment_root(_NAME) == _BASE / _NAME


def test_experiment_root_custom_base(tmp_path):
    assert experiment_root(_NAME, tmp_path) == tmp_path / _NAME


def test_metadata_path():
    assert metadata_path(_NAME) == _BASE / _NAME / "metadata.json"


def test_metrics_path():
    assert metrics_path(_NAME) == _BASE / _NAME / "metrics.json"


def test_equity_curve_path():
    assert equity_curve_path(_NAME) == _BASE / _NAME / "equity_curve.parquet"


def test_returns_path():
    assert returns_path(_NAME) == _BASE / _NAME / "returns.parquet"


def test_weights_path():
    assert weights_path(_NAME) == _BASE / _NAME / "weights.parquet"


def test_ml_provenance_path():
    assert ml_provenance_path(_NAME) == _BASE / _NAME / "ml_provenance.json"


def test_diagnostics_dir():
    assert diagnostics_dir(_NAME) == _BASE / _NAME / "diagnostics"


def test_research_dir():
    assert research_dir(_NAME) == _BASE / _NAME / "research"


def test_plots_dir():
    assert plots_dir(_NAME) == _BASE / _NAME / "plots"


def test_plot_index_path():
    assert plot_index_path(_NAME) == _BASE / _NAME / "plots" / "plot_index.json"


def test_plot_path():
    assert plot_path(_NAME, "equity_curve") == _BASE / _NAME / "plots" / "equity_curve.png"


def test_all_diagnostics_paths_keys():
    paths = all_diagnostics_paths(_NAME)
    expected_keys = {
        "backtest_diagnostics", "ml_diagnostics", "ml_model_diagnostics",
        "split_metrics", "universe_coverage", "wf_equity_curves",
        "alignment_diagnostics", "data_summary", "feature_correlations",
        "feature_families", "feature_registry", "feature_summary",
        "signal_transitions",
    }
    assert set(paths.keys()) == expected_keys


def test_all_diagnostics_paths_types():
    paths = all_diagnostics_paths(_NAME)
    for key, path in paths.items():
        assert isinstance(path, Path), f"{key} should be a Path"
        assert path.suffix == ".json", f"{key} should be .json"


def test_llm_context_path():
    p = llm_context_path(_NAME)
    assert p.name == "llm_context.json"
    assert _NAME in str(p)


def test_llm_review_path():
    p = llm_review_path(_NAME)
    assert p.name == "llm_review.json"


def test_report_manifest_path():
    p = report_manifest_path(_NAME)
    assert p.name == f"{_NAME}_manifest.json"


def test_list_experiments_real(tmp_path):
    """list_experiments finds dirs with metadata.json."""
    for name in ("exp_a", "exp_b", "not_exp"):
        d = tmp_path / name
        d.mkdir()
        if name != "not_exp":
            (d / "metadata.json").write_text("{}")

    found = list_experiments(tmp_path)
    assert "exp_a" in found
    assert "exp_b" in found
    assert "not_exp" not in found


def test_list_experiments_empty(tmp_path):
    assert list_experiments(tmp_path) == []
