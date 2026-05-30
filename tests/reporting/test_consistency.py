"""Tests for src.reporting.consistency — G-SYNC-5 validation checks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.reporting.consistency import (
    ConsistencyReport,
    ConsistencyWarning,
    validate_report_consistency,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_artefacts(
    feature_registry=None,
    feature_summary=None,
    feature_correlations=None,
    feature_families=None,
    ml_model_diagnostics=None,
):
    """Build a minimal mock ExperimentArtefacts."""
    a = MagicMock()
    a.feature_registry = feature_registry
    a.feature_summary = feature_summary
    a.feature_correlations = feature_correlations
    a.feature_families = feature_families
    a.ml_model_diagnostics = ml_model_diagnostics
    return a


# ---------------------------------------------------------------------------
# ConsistencyReport
# ---------------------------------------------------------------------------

def test_consistency_report_empty():
    cr = ConsistencyReport()
    assert not cr.has_warnings
    assert cr.n_warnings == 0
    assert cr.as_markdown() == ""


def test_consistency_report_with_warnings():
    cr = ConsistencyReport(warnings=[
        ConsistencyWarning("warning", "test", "Something is wrong")
    ])
    assert cr.has_warnings
    assert cr.n_warnings == 1


def test_consistency_report_as_markdown():
    cr = ConsistencyReport(warnings=[
        ConsistencyWarning("warning", "feature_omission", "Feature x missing"),
        ConsistencyWarning("info", "unclaimed_figure", "Figure y not embedded"),
    ])
    md = cr.as_markdown()
    assert "feature_omission" in md
    assert "Feature x missing" in md
    assert "unclaimed_figure" in md


# ---------------------------------------------------------------------------
# validate_report_consistency — no-op on empty artefacts
# ---------------------------------------------------------------------------

def test_validate_empty_artefacts():
    a = _make_artefacts()
    cr = validate_report_consistency(a, {})
    assert isinstance(cr, ConsistencyReport)


def test_validate_no_feature_registry():
    a = _make_artefacts(feature_registry=None)
    cr = validate_report_consistency(a, {}, set())
    assert not cr.has_warnings


def test_validate_clean_artefacts():
    """Registry and correlations have identical features — no warnings."""
    fr = {"features": [{"name": "mom_20"}, {"name": "vol_21"}], "n_features": 2}
    fc = {"features": ["mom_20", "vol_21"]}
    fs = {"features": {"mom_20": {"mean": 0.01}, "vol_21": {"mean": 0.1}}}
    a = _make_artefacts(feature_registry=fr, feature_correlations=fc, feature_summary=fs)
    cr = validate_report_consistency(a, {})
    # No divergence warnings
    divergence = [w for w in cr.warnings if w.category == "feature_set_divergence"]
    assert not divergence


def test_validate_feature_set_divergence():
    """Feature in registry but not in correlations should trigger warning."""
    fr = {"features": [{"name": "mom_20"}, {"name": "new_feat"}], "n_features": 2}
    fc = {"features": ["mom_20"]}  # missing new_feat
    a = _make_artefacts(feature_registry=fr, feature_correlations=fc)
    cr = validate_report_consistency(a, {})
    divergence = [w for w in cr.warnings if w.category == "feature_set_divergence"]
    assert len(divergence) >= 1
    assert any("new_feat" in w.message for w in divergence)


# ---------------------------------------------------------------------------
# Feature family IC checks
# ---------------------------------------------------------------------------

def test_validate_family_ic_absent_warns():
    """Multiple families but no feature_family_ic figure should warn."""
    ff = {"families": {"Trend": ["mom_20"], "Volatility": ["vol_21"]}}
    a = _make_artefacts(feature_families=ff)
    cr = validate_report_consistency(a, {})  # no figures
    ic_warnings = [w for w in cr.warnings if w.category == "family_ic_absent"]
    assert len(ic_warnings) == 1


def test_validate_family_ic_present_no_warn():
    """When feature_family_ic figure exists, no family_ic_absent warning."""
    ff = {"families": {"Trend": ["mom_20"], "Volatility": ["vol_21"]}}
    a = _make_artefacts(feature_families=ff)
    fig_map = {"feature_family_ic": Path("some/path.png")}
    cr = validate_report_consistency(a, fig_map)
    ic_warnings = [w for w in cr.warnings if w.category == "family_ic_absent"]
    assert not ic_warnings


def test_validate_single_family_no_warn():
    """Single family doesn't require family IC figure."""
    ff = {"families": {"Trend": ["mom_20", "mom_60"]}}
    a = _make_artefacts(feature_families=ff)
    cr = validate_report_consistency(a, {})
    ic_warnings = [w for w in cr.warnings if w.category == "family_ic_absent"]
    assert not ic_warnings


# ---------------------------------------------------------------------------
# Unclaimed figure checks
# ---------------------------------------------------------------------------

def test_validate_unclaimed_important_figure_is_info():
    """An important figure available but not claimed should produce info warning."""
    a = _make_artefacts()
    available = {"feature_correlation_heatmap": Path("some/fig.png")}
    cr = validate_report_consistency(a, available, claimed=set())
    unclaimed = [w for w in cr.warnings if w.category == "unclaimed_figure"]
    assert len(unclaimed) >= 1
    assert all(w.severity == "info" for w in unclaimed)


def test_validate_claimed_figure_no_warn():
    """A figure that is claimed should not generate unclaimed warning."""
    a = _make_artefacts()
    available = {"feature_correlation_heatmap": Path("some/fig.png")}
    claimed = {"feature_correlation_heatmap"}
    cr = validate_report_consistency(a, available, claimed=claimed)
    unclaimed = [w for w in cr.warnings if w.category == "unclaimed_figure"]
    assert not unclaimed
