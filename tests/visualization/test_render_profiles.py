"""Tests for the rendering profile infrastructure."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
from src.visualization.render_profiles import (
    ARCHETYPE_BAR,
    ARCHETYPE_CONTRIBUTION,
    ARCHETYPE_DEFAULT,
    ARCHETYPE_HEATMAP,
    ARCHETYPE_ROLLING,
    ARCHETYPE_STACKED,
    ARCHETYPE_TIMELINE,
    RENDER_PROFILES,
    apply_render_profile,
    get_active_profile,
    get_archetype_hint,
    get_dpi_save,
    get_figsize_scale,
    get_line_width_scale,
    get_render_profile,
    set_active_profile,
)
from src.visualization.styles import apply_research_style, make_figure

# ---------------------------------------------------------------------------
# Fixtures — ensure clean state between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_profile():
    """Reset active profile and matplotlib RC to report baseline after each test."""
    yield
    set_active_profile("report")
    apply_research_style("report")
    plt.close("all")


# ---------------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------------

def test_render_profiles_contains_report_and_frontend():
    assert "report" in RENDER_PROFILES
    assert "frontend" in RENDER_PROFILES


def test_render_profiles_have_required_keys():
    for name, p in RENDER_PROFILES.items():
        assert "rc" in p, f"Profile {name!r} missing 'rc'"
        assert "figsize_scale" in p, f"Profile {name!r} missing 'figsize_scale'"
        assert "dpi_save" in p, f"Profile {name!r} missing 'dpi_save'"
        assert "line_width_scale" in p, f"Profile {name!r} missing 'line_width_scale'"
        assert "archetype_overrides" in p, f"Profile {name!r} missing 'archetype_overrides'"


# ---------------------------------------------------------------------------
# Active profile state
# ---------------------------------------------------------------------------

def test_default_active_profile_is_report():
    set_active_profile("report")
    assert get_active_profile() == "report"


def test_set_active_profile_frontend():
    set_active_profile("frontend")
    assert get_active_profile() == "frontend"


def test_set_active_profile_invalid_raises():
    with pytest.raises(ValueError, match="Unknown render profile"):
        set_active_profile("pitch_deck")


# ---------------------------------------------------------------------------
# get_render_profile
# ---------------------------------------------------------------------------

def test_get_render_profile_report_returns_dict():
    p = get_render_profile("report")
    assert isinstance(p, dict)
    assert p["figsize_scale"] == 1.0


def test_get_render_profile_frontend_returns_dict():
    p = get_render_profile("frontend")
    assert isinstance(p, dict)
    assert p["figsize_scale"] > 1.0


def test_get_render_profile_none_uses_active_profile():
    set_active_profile("frontend")
    p = get_render_profile(None)
    assert p["figsize_scale"] > 1.0


def test_get_render_profile_invalid_raises():
    with pytest.raises(ValueError, match="Unknown render profile"):
        get_render_profile("dashboard")


# ---------------------------------------------------------------------------
# apply_render_profile
# ---------------------------------------------------------------------------

def test_apply_render_profile_frontend_raises_legend_fontsize():
    apply_render_profile("frontend")
    assert mpl.rcParams["legend.fontsize"] == 13


def test_apply_render_profile_frontend_raises_title_fontsize():
    apply_render_profile("frontend")
    assert mpl.rcParams["axes.titlesize"] == 20


def test_apply_render_profile_frontend_raises_label_fontsize():
    apply_render_profile("frontend")
    assert mpl.rcParams["axes.labelsize"] == 16


def test_apply_render_profile_frontend_raises_tick_fontsize():
    apply_render_profile("frontend")
    assert mpl.rcParams["xtick.labelsize"] == 13
    assert mpl.rcParams["ytick.labelsize"] == 13


def test_apply_render_profile_frontend_raises_line_width():
    apply_render_profile("frontend")
    assert mpl.rcParams["lines.linewidth"] == 2.0


def test_apply_render_profile_frontend_raises_save_dpi():
    apply_render_profile("frontend")
    assert mpl.rcParams["savefig.dpi"] == 250


def test_apply_render_profile_report_has_no_rc_overrides():
    # Report profile has an empty rc dict — base RC is the authority
    p = get_render_profile("report")
    assert p["rc"] == {}


def test_apply_render_profile_sets_active_profile():
    apply_render_profile("frontend")
    assert get_active_profile() == "frontend"


# ---------------------------------------------------------------------------
# apply_research_style profile integration
# ---------------------------------------------------------------------------

def test_apply_research_style_default_is_report():
    apply_research_style()
    assert get_active_profile() == "report"


def test_apply_research_style_frontend_sets_rc():
    apply_research_style("frontend")
    assert mpl.rcParams["axes.titlesize"] == 20
    assert mpl.rcParams["legend.fontsize"] == 13
    assert mpl.rcParams["savefig.dpi"] == 250


def test_apply_research_style_report_does_not_override_base():
    apply_research_style("report")
    # Base RC has title size 11
    assert mpl.rcParams["axes.titlesize"] == 11


def test_apply_research_style_frontend_sets_active_profile():
    apply_research_style("frontend")
    assert get_active_profile() == "frontend"


# ---------------------------------------------------------------------------
# get_figsize_scale
# ---------------------------------------------------------------------------

def test_figsize_scale_report_is_one():
    assert get_figsize_scale("report") == 1.0


def test_figsize_scale_frontend_is_greater_than_one():
    scale = get_figsize_scale("frontend")
    assert scale > 1.0


def test_figsize_scale_heatmap_archetype_frontend():
    scale = get_figsize_scale("frontend", ARCHETYPE_HEATMAP)
    assert scale > get_figsize_scale("frontend")


def test_figsize_scale_heatmap_archetype_report_unchanged():
    # Report has no archetype overrides
    scale_default = get_figsize_scale("report", ARCHETYPE_DEFAULT)
    scale_heatmap = get_figsize_scale("report", ARCHETYPE_HEATMAP)
    assert scale_default == scale_heatmap == 1.0


def test_figsize_scale_unknown_archetype_falls_back_to_base():
    scale = get_figsize_scale("frontend", "nonexistent_archetype")
    assert scale == get_figsize_scale("frontend", ARCHETYPE_DEFAULT)


# ---------------------------------------------------------------------------
# get_dpi_save
# ---------------------------------------------------------------------------

def test_dpi_save_report_is_200():
    assert get_dpi_save("report") == 200


def test_dpi_save_frontend_is_250():
    assert get_dpi_save("frontend") == 250


# ---------------------------------------------------------------------------
# get_line_width_scale
# ---------------------------------------------------------------------------

def test_line_width_scale_report_is_one():
    assert get_line_width_scale("report") == 1.0


def test_line_width_scale_frontend_is_greater_than_one():
    assert get_line_width_scale("frontend") > 1.0


def test_line_width_scale_rolling_archetype_frontend():
    base = get_line_width_scale("frontend")
    rolling = get_line_width_scale("frontend", ARCHETYPE_ROLLING)
    assert rolling >= base


# ---------------------------------------------------------------------------
# get_archetype_hint
# ---------------------------------------------------------------------------

def test_archetype_hint_timeline_frontend_has_external_legend():
    hint = get_archetype_hint(ARCHETYPE_TIMELINE, "frontend")
    assert hint.get("external_legend") is True


def test_archetype_hint_stacked_frontend_has_external_legend():
    hint = get_archetype_hint(ARCHETYPE_STACKED, "frontend")
    assert hint.get("external_legend") is True


def test_archetype_hint_bar_frontend_has_annotation_padding():
    hint = get_archetype_hint(ARCHETYPE_BAR, "frontend")
    assert "annotation_padding" in hint


def test_archetype_hint_contribution_frontend_has_external_legend():
    hint = get_archetype_hint(ARCHETYPE_CONTRIBUTION, "frontend")
    assert hint.get("external_legend") is True


def test_archetype_hint_report_is_empty_for_all_archetypes():
    for archetype in [ARCHETYPE_HEATMAP, ARCHETYPE_TIMELINE, ARCHETYPE_STACKED,
                      ARCHETYPE_BAR, ARCHETYPE_ROLLING, ARCHETYPE_CONTRIBUTION]:
        hint = get_archetype_hint(archetype, "report")
        assert hint == {}, f"Report profile should have no hints for {archetype!r}"


def test_archetype_hint_unknown_archetype_is_empty_dict():
    hint = get_archetype_hint("unknown_archetype", "frontend")
    assert hint == {}


def test_archetype_hint_returns_copy_not_reference():
    hint = get_archetype_hint(ARCHETYPE_TIMELINE, "frontend")
    hint["injected"] = True
    # Original profile must not be mutated
    assert "injected" not in get_archetype_hint(ARCHETYPE_TIMELINE, "frontend")


# ---------------------------------------------------------------------------
# make_figure scales with active profile
# ---------------------------------------------------------------------------

def test_make_figure_report_profile_uses_exact_dimensions():
    apply_research_style("report")
    w, h = 12.0, 4.0
    fig, _ = make_figure(width=w, height=h)
    fw, fh = fig.get_size_inches()
    assert abs(fw - w) < 1e-6
    assert abs(fh - h) < 1e-6
    plt.close(fig)


def test_make_figure_frontend_profile_scales_dimensions():
    apply_research_style("frontend")
    w, h = 12.0, 4.0
    fig, _ = make_figure(width=w, height=h)
    fw, fh = fig.get_size_inches()
    scale = get_figsize_scale("frontend")
    assert abs(fw - w * scale) < 1e-5
    assert abs(fh - h * scale) < 1e-5
    plt.close(fig)


def test_make_figure_frontend_figures_are_larger_than_report():
    apply_research_style("report")
    fig_r, _ = make_figure(width=12.0, height=4.0)
    w_r, h_r = fig_r.get_size_inches()
    plt.close(fig_r)

    apply_research_style("frontend")
    fig_f, _ = make_figure(width=12.0, height=4.0)
    w_f, h_f = fig_f.get_size_inches()
    plt.close(fig_f)

    assert w_f > w_r
    assert h_f > h_r


def test_make_figure_default_height_scales_with_profile():
    apply_research_style("frontend")
    fig, _ = make_figure()
    _, fh = fig.get_size_inches()
    from src.visualization.styles import FIG_HEIGHT_STANDARD
    scale = get_figsize_scale("frontend")
    assert abs(fh - FIG_HEIGHT_STANDARD * scale) < 1e-5
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API via __init__
# ---------------------------------------------------------------------------

def test_public_api_exports_profile_functions():
    import src.visualization as viz
    assert hasattr(viz, "apply_render_profile")
    assert hasattr(viz, "get_active_profile")
    assert hasattr(viz, "get_render_profile")
    assert hasattr(viz, "set_active_profile")
    assert hasattr(viz, "get_archetype_hint")
    assert hasattr(viz, "RENDER_PROFILES")


def test_public_api_exports_archetype_constants():
    import src.visualization as viz
    assert viz.ARCHETYPE_HEATMAP == "heatmap"
    assert viz.ARCHETYPE_TIMELINE == "timeline"
    assert viz.ARCHETYPE_STACKED == "stacked_area"
    assert viz.ARCHETYPE_BAR == "bar_chart"
    assert viz.ARCHETYPE_ROLLING == "rolling_diagnostics"
    assert viz.ARCHETYPE_CONTRIBUTION == "contribution"
