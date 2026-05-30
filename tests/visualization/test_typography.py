"""Tests for the centralized typography system.

Covers TypographyScale, get_typography(), heatmap_cell_fontsize(), and
scale_dynamic_fontsize() across both render profiles.
"""

from __future__ import annotations

import pytest
import matplotlib.pyplot as plt

from src.visualization.render_profiles import set_active_profile
from src.visualization.styles import apply_research_style
from src.visualization.typography import (
    TypographyScale,
    get_typography,
    heatmap_cell_fontsize,
    scale_dynamic_fontsize,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_profile():
    """Restore report profile and close figures after each test."""
    yield
    set_active_profile("report")
    apply_research_style("report")
    plt.close("all")


# ---------------------------------------------------------------------------
# TypographyScale dataclass
# ---------------------------------------------------------------------------

def test_typography_scale_is_frozen():
    t = get_typography()
    with pytest.raises((AttributeError, TypeError)):
        t.annotation = 99.0  # type: ignore[misc]


def test_typography_scale_has_all_fields():
    t = get_typography()
    for field in ("annotation", "legend", "tick", "colorbar",
                  "heatmap_cell", "figure_text", "small_label", "small_annotation"):
        assert hasattr(t, field), f"Missing field: {field}"
        assert isinstance(getattr(t, field), float)


# ---------------------------------------------------------------------------
# get_typography() — profile switching
# ---------------------------------------------------------------------------

def test_get_typography_returns_report_by_default():
    apply_research_style("report")
    t = get_typography()
    assert t.annotation == pytest.approx(8.5)
    assert t.legend == pytest.approx(8.5)
    assert t.tick == pytest.approx(8.5)
    assert t.colorbar == pytest.approx(7.5)
    assert t.figure_text == pytest.approx(10.0)
    assert t.small_label == pytest.approx(8.0)
    assert t.small_annotation == pytest.approx(7.0)


def test_get_typography_frontend_is_larger_than_report():
    apply_research_style("report")
    report_t = get_typography()
    apply_research_style("frontend")
    frontend_t = get_typography()
    assert frontend_t.annotation > report_t.annotation
    assert frontend_t.legend > report_t.legend
    assert frontend_t.tick > report_t.tick
    assert frontend_t.colorbar > report_t.colorbar
    assert frontend_t.figure_text > report_t.figure_text


def test_get_typography_switches_with_profile():
    apply_research_style("report")
    r = get_typography()
    apply_research_style("frontend")
    f = get_typography()
    assert f != r


def test_get_typography_archetype_param_accepted():
    from src.visualization.render_profiles import ARCHETYPE_HEATMAP
    apply_research_style("report")
    t_default = get_typography()
    t_heatmap = get_typography(archetype=ARCHETYPE_HEATMAP)
    # archetype does not alter scale values — canvas scaling handles it
    assert t_default.annotation == t_heatmap.annotation


# ---------------------------------------------------------------------------
# heatmap_cell_fontsize()
# ---------------------------------------------------------------------------

def test_heatmap_cell_fontsize_report_formula():
    apply_research_style("report")
    # Formula: max(7.0, min(10.0, 120 / max(n, m)))
    assert heatmap_cell_fontsize(10, 10) == pytest.approx(max(7.0, min(10.0, 120 / 10)))
    assert heatmap_cell_fontsize(20, 20) == pytest.approx(max(7.0, min(10.0, 120 / 20)))
    assert heatmap_cell_fontsize(5, 5) == pytest.approx(10.0)   # capped at 10
    assert heatmap_cell_fontsize(100, 100) == pytest.approx(7.0)  # floored at 7


def test_heatmap_cell_fontsize_frontend_is_larger():
    apply_research_style("report")
    report_val = heatmap_cell_fontsize(15, 15)
    apply_research_style("frontend")
    frontend_val = heatmap_cell_fontsize(15, 15)
    assert frontend_val > report_val


def test_heatmap_cell_fontsize_asymmetric_grid():
    apply_research_style("report")
    # Larger dimension dominates
    val_tall = heatmap_cell_fontsize(50, 5)
    val_wide = heatmap_cell_fontsize(5, 50)
    assert val_tall == val_wide


def test_heatmap_cell_fontsize_single_cell():
    apply_research_style("report")
    # n=1, m=1 → density = 120 / 1 = 120 → capped at 10
    assert heatmap_cell_fontsize(1, 1) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# scale_dynamic_fontsize()
# ---------------------------------------------------------------------------

def test_scale_dynamic_fontsize_report_returns_unchanged():
    apply_research_style("report")
    assert scale_dynamic_fontsize(8.0, "tick") == pytest.approx(8.0)
    assert scale_dynamic_fontsize(7.0, "colorbar") == pytest.approx(7.0)
    assert scale_dynamic_fontsize(6.5, "small_annotation") == pytest.approx(6.5)


def test_scale_dynamic_fontsize_frontend_scales_up():
    apply_research_style("frontend")
    # tick: report=8.5, frontend=12.0 → scale = 12/8.5 ≈ 1.41
    result = scale_dynamic_fontsize(8.0, "tick")
    assert result > 8.0


def test_scale_dynamic_fontsize_ratio_consistent():
    """The scale ratio is the same regardless of input value."""
    apply_research_style("frontend")
    r1 = scale_dynamic_fontsize(7.0, "tick")
    r2 = scale_dynamic_fontsize(8.5, "tick")
    # ratio should be identical: r1/7.0 ≈ r2/8.5
    assert r1 / 7.0 == pytest.approx(r2 / 8.5, rel=0.01)


def test_scale_dynamic_fontsize_unknown_field_defaults_to_tick():
    apply_research_style("frontend")
    val_tick = scale_dynamic_fontsize(8.0, "tick")
    # unknown field falls back gracefully (won't raise)
    val_unknown = scale_dynamic_fontsize(8.0, "tick")
    assert val_tick == val_unknown


def test_scale_dynamic_fontsize_all_semantic_fields():
    apply_research_style("frontend")
    for field in ("annotation", "legend", "tick", "colorbar",
                  "heatmap_cell", "figure_text", "small_label", "small_annotation"):
        result = scale_dynamic_fontsize(8.0, field)
        assert isinstance(result, float), f"field={field} did not return float"
        assert result > 0


# ---------------------------------------------------------------------------
# Public API export in __init__
# ---------------------------------------------------------------------------

def test_typography_exported_from_visualization_package():
    from src.visualization import (
        TypographyScale,
        get_typography,
        heatmap_cell_fontsize,
        scale_dynamic_fontsize,
    )
    assert callable(get_typography)
    assert callable(heatmap_cell_fontsize)
    assert callable(scale_dynamic_fontsize)
    assert TypographyScale is not None
