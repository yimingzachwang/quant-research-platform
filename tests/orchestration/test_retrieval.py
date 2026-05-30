"""Tests for orchestration.retrieval against real experiment artefacts."""

from pathlib import Path

import pytest

from src.orchestration.retrieval.artefact_retriever import (
    _infer_type,
    list_artefacts,
    retrieve,
    retrieve_many,
)
from src.orchestration.retrieval.diagnostics_retriever import (
    load_all_diagnostics,
    load_ml_model_diagnostics,
    load_split_metrics,
)
from src.orchestration.retrieval.plot_retriever import (
    get_plot_index,
    get_primary_plots,
    list_plot_stems,
    plot_exists,
)
from src.orchestration.retrieval.manifest_retriever import (
    get_rendered_sections,
    load_manifest,
)

_CANONICAL = "canonical_ml_multi_asset"


# ---------------------------------------------------------------------------
# _infer_type — regression for leading-dot bug (Path.suffix includes ".")
# ---------------------------------------------------------------------------


def test_infer_type_parquet():
    assert _infer_type(Path("equity_curve.parquet")) == "parquet"


def test_infer_type_json():
    assert _infer_type(Path("metrics.json")) == "json"


def test_infer_type_yaml():
    assert _infer_type(Path("config.yaml")) == "yaml"


def test_infer_type_png():
    assert _infer_type(Path("plot.png")) == "png"


def test_infer_type_unknown():
    assert _infer_type(Path("data.csv")) == "unknown"


# ---------------------------------------------------------------------------
# Artefact retriever
# ---------------------------------------------------------------------------


def test_retrieve_metadata():
    meta = retrieve(_CANONICAL, "metadata")
    assert meta is not None
    assert "experiment_name" in meta


def test_retrieve_metrics():
    m = retrieve(_CANONICAL, "metrics")
    assert m is not None
    assert "sharpe_ratio" in m


def test_retrieve_ml_model_diagnostics():
    d = retrieve(_CANONICAL, "ml_model_diagnostics")
    assert d is not None
    assert "ic_summary" in d


def test_retrieve_nonexistent_key():
    result = retrieve(_CANONICAL, "nonexistent_artefact_xyz")
    assert result is None


def test_retrieve_many():
    result = retrieve_many(_CANONICAL, ["metadata", "metrics"])
    assert "metadata" in result
    assert "metrics" in result


def test_retrieve_many_skips_missing():
    result = retrieve_many(_CANONICAL, ["metadata", "not_a_real_key"])
    assert "metadata" in result
    assert "not_a_real_key" not in result


def test_list_artefacts_canonical():
    artefacts = list_artefacts(_CANONICAL)
    keys = [a.key for a in artefacts]
    assert "metadata" in keys
    assert "metrics" in keys
    assert "ml_model_diagnostics" in keys


def test_list_artefacts_existing():
    artefacts = list_artefacts(_CANONICAL)
    existing = [a for a in artefacts if a.exists]
    assert len(existing) > 5


# ---------------------------------------------------------------------------
# Diagnostics retriever
# ---------------------------------------------------------------------------


def test_load_ml_model_diagnostics():
    d = load_ml_model_diagnostics(_CANONICAL)
    assert d is not None
    assert "ic_summary" in d
    assert "coefficient_stability_summary" in d


def test_load_split_metrics():
    d = load_split_metrics(_CANONICAL)
    assert d is not None
    assert "summary" in d
    assert "splits" in d


def test_load_all_diagnostics():
    all_d = load_all_diagnostics(_CANONICAL)
    assert "ml_model_diagnostics" in all_d
    assert "split_metrics" in all_d
    assert "feature_families" in all_d


# ---------------------------------------------------------------------------
# Plot retriever
# ---------------------------------------------------------------------------


def test_get_plot_index():
    plots = get_plot_index(_CANONICAL)
    assert len(plots) > 0
    names = [p.name for p in plots]
    assert "equity_and_drawdown" in names or "allocation_history" in names


def test_get_primary_plots():
    primary = get_primary_plots(_CANONICAL)
    assert all(p.importance == "primary" for p in primary)
    assert len(primary) > 0


def test_list_plot_stems():
    stems = list_plot_stems(_CANONICAL)
    assert len(stems) > 0
    assert all(isinstance(s, str) for s in stems)


def test_plot_exists_known():
    assert plot_exists(_CANONICAL, "equity_and_drawdown")


def test_plot_exists_missing():
    assert not plot_exists(_CANONICAL, "nonexistent_plot_xyz_abc")


# ---------------------------------------------------------------------------
# Manifest retriever
# ---------------------------------------------------------------------------


def test_load_manifest():
    m = load_manifest(_CANONICAL)
    assert m is not None
    assert "sections_rendered" in m


def test_get_rendered_sections():
    sections = get_rendered_sections(_CANONICAL)
    assert len(sections) > 0
    assert "Performance Metrics" in sections or "ML Model Behaviour" in sections
