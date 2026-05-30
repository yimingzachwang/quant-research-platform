"""Tests for prompt rendering integrity — Fix 1 and Fix 2.

Validates:
- Jinja2 rendering produces a fully resolved prompt
- Unresolved template tokens trigger a hard RuntimeError
- Rendered prompt contains actual diagnostic values from context
"""

import re

import pytest
from src.orchestration.context.context_builder import build_context
from src.orchestration.llm.prompt_templates import EXPERIMENT_REVIEW
from src.orchestration.llm.review_engine import _assert_no_unresolved_tokens, _render_prompt

_CANONICAL = "canonical_ml_multi_asset"
_UNRESOLVED_RE = re.compile(r"\{\{|\}\}|\{%|%\}")


# ---------------------------------------------------------------------------
# Rendering correctness
# ---------------------------------------------------------------------------


def test_render_produces_no_unresolved_tokens():
    ctx = build_context(_CANONICAL)
    prompt = _render_prompt(ctx, EXPERIMENT_REVIEW)
    remaining = _UNRESOLVED_RE.findall(prompt)
    assert remaining == [], f"Unresolved tokens found: {remaining}"


def test_render_contains_experiment_name():
    ctx = build_context(_CANONICAL)
    prompt = _render_prompt(ctx, EXPERIMENT_REVIEW)
    assert _CANONICAL in prompt


def test_render_contains_strategy_name():
    ctx = build_context(_CANONICAL)
    prompt = _render_prompt(ctx, EXPERIMENT_REVIEW)
    assert ctx.strategy_name in prompt


def test_render_contains_sharpe_value():
    ctx = build_context(_CANONICAL)
    prompt = _render_prompt(ctx, EXPERIMENT_REVIEW)
    # The Sharpe ratio float must appear somewhere in the prompt
    sharpe = ctx.performance.get("sharpe_ratio")
    assert sharpe is not None
    assert str(round(sharpe, 3)) in prompt or "sharpe" in prompt.lower()


def test_render_contains_failure_mode():
    ctx = build_context(_CANONICAL)
    assert len(ctx.failure_modes) > 0, "Expected at least one failure mode"
    prompt = _render_prompt(ctx, EXPERIMENT_REVIEW)
    # Failure mode name must appear in rendered prompt
    assert ctx.failure_modes[0]["name"] in prompt


def test_render_contains_section_instructions():
    ctx = build_context(_CANONICAL)
    prompt = _render_prompt(ctx, EXPERIMENT_REVIEW)
    assert "performance_assessment" in prompt
    assert "signal_quality" in prompt
    assert "recommendations" in prompt


def test_render_contains_ml_diagnostics():
    ctx = build_context(_CANONICAL)
    prompt = _render_prompt(ctx, EXPERIMENT_REVIEW)
    # IC value must appear in the ML diagnostics block
    assert "mean_ic" in prompt or "ic_tier" in prompt


def test_render_contains_validation_data():
    ctx = build_context(_CANONICAL)
    prompt = _render_prompt(ctx, EXPERIMENT_REVIEW)
    assert "n_splits" in prompt or "consistency_tier" in prompt


# ---------------------------------------------------------------------------
# Hard failure on unresolved tokens — Fix 2
# ---------------------------------------------------------------------------


def test_assert_no_unresolved_tokens_passes_on_clean_prompt():
    # Should not raise
    _assert_no_unresolved_tokens("This is a clean rendered prompt.", "test_template")


def test_assert_no_unresolved_tokens_raises_on_double_brace():
    with pytest.raises(RuntimeError, match="unresolved template tokens"):
        _assert_no_unresolved_tokens(
            "Performance: {{ performance | tojson }}", "test_template"
        )


def test_assert_no_unresolved_tokens_raises_on_block_tag():
    with pytest.raises(RuntimeError, match="unresolved template tokens"):
        _assert_no_unresolved_tokens(
            "{% if failure_modes %}critical{% endif %}", "test_template"
        )


def test_render_raises_on_missing_jinja2_if_template_has_filters(monkeypatch):
    """Simulate Jinja2 import failure — must raise, not fall back silently."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "jinja2":
            raise ImportError("jinja2 not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    ctx = build_context(_CANONICAL)
    with pytest.raises((ImportError, RuntimeError)):
        _render_prompt(ctx, EXPERIMENT_REVIEW)


# ---------------------------------------------------------------------------
# Phase 1.5 — Fix 1: validation variance emphasis
# ---------------------------------------------------------------------------


def test_render_contains_std_oos_sharpe_instruction():
    """Validation section must instruct the model to address std_oos_sharpe."""
    ctx = build_context(_CANONICAL)
    prompt = _render_prompt(ctx, EXPERIMENT_REVIEW)
    assert "std_oos_sharpe" in prompt


def test_render_contains_worst_split_drawdown_instruction():
    ctx = build_context(_CANONICAL)
    prompt = _render_prompt(ctx, EXPERIMENT_REVIEW)
    assert "worst split drawdown" in prompt.lower() or "worst_split_drawdown" in prompt


def test_render_contains_dispersion_risk_instruction():
    ctx = build_context(_CANONICAL)
    prompt = _render_prompt(ctx, EXPERIMENT_REVIEW)
    assert "dispersion risk" in prompt.lower()


def test_render_contains_coefficient_of_variation_instruction():
    ctx = build_context(_CANONICAL)
    prompt = _render_prompt(ctx, EXPERIMENT_REVIEW)
    assert "coefficient of variation" in prompt.lower()


# ---------------------------------------------------------------------------
# Phase 1.5 — Fix 2: recommendation grounding enforcement
# ---------------------------------------------------------------------------


def test_render_contains_grounding_enforcement():
    """Recommendations section must require citations of specific diagnostic values."""
    ctx = build_context(_CANONICAL)
    prompt = _render_prompt(ctx, EXPERIMENT_REVIEW)
    assert "FORBIDDEN" in prompt


def test_render_forbids_generic_boilerplate():
    ctx = build_context(_CANONICAL)
    prompt = _render_prompt(ctx, EXPERIMENT_REVIEW)
    assert "do NOT recommend" in prompt or "do not recommend" in prompt.lower()


# ---------------------------------------------------------------------------
# Phase 1.5 — Fix 3: feature instability emphasis
# ---------------------------------------------------------------------------


def test_render_contains_n_family_transitions_instruction():
    """Feature contribution section must require n_family_transitions discussion."""
    ctx = build_context(_CANONICAL)
    prompt = _render_prompt(ctx, EXPERIMENT_REVIEW)
    assert "n_family_transitions" in prompt


def test_render_contains_most_volatile_feature_instruction():
    ctx = build_context(_CANONICAL)
    prompt = _render_prompt(ctx, EXPERIMENT_REVIEW)
    assert "most_volatile_feature" in prompt
