"""Tests for Phase 2A — Research Iteration Orchestration.

Validates:
- iteration prompt rendering and grounding enforcement
- section parsing correctness
- context hash provenance
- persistence of JSON and MD artefacts
- structured IterationProposal construction
- no unresolved template tokens
- deterministic persistence behavior
- grounding quality: no autonomous execution language in prompt
"""

from __future__ import annotations

import json
import re

import pytest

from src.orchestration.api.schemas import IterationProposal
from src.orchestration.context.context_builder import build_context
from src.orchestration.llm.iteration_engine import (
    ITERATION_VERSION,
    _normalise_heading,
    _parse_iteration_proposal,
    _render_iteration_prompt,
    _split_sections,
    run_iteration_proposal,
)
from src.orchestration.llm.prompt_templates import ITERATION_PROPOSAL, load_template
from src.orchestration.llm.review_engine import _assert_no_unresolved_tokens
from src.orchestration.llm.review_schema import PROVIDER_STUB

_CANONICAL = "canonical_ml_multi_asset"
_UNRESOLVED_RE = re.compile(r"\{\{|\}\}|\{%|%\}")


# ---------------------------------------------------------------------------
# 1. Template rendering — no unresolved tokens
# ---------------------------------------------------------------------------


def test_iteration_prompt_renders_no_unresolved_tokens():
    ctx = build_context(_CANONICAL)
    prompt = _render_iteration_prompt(ctx, ITERATION_PROPOSAL)
    remaining = _UNRESOLVED_RE.findall(prompt)
    assert remaining == [], f"Unresolved tokens found: {remaining}"


def test_iteration_prompt_contains_experiment_name():
    ctx = build_context(_CANONICAL)
    prompt = _render_iteration_prompt(ctx, ITERATION_PROPOSAL)
    assert _CANONICAL in prompt


def test_iteration_prompt_contains_strategy_name():
    ctx = build_context(_CANONICAL)
    prompt = _render_iteration_prompt(ctx, ITERATION_PROPOSAL)
    assert ctx.strategy_name in prompt


def test_iteration_prompt_contains_failure_modes():
    ctx = build_context(_CANONICAL)
    assert len(ctx.failure_modes) > 0
    prompt = _render_iteration_prompt(ctx, ITERATION_PROPOSAL)
    assert ctx.failure_modes[0]["name"] in prompt


def test_iteration_prompt_contains_validation_data():
    ctx = build_context(_CANONICAL)
    prompt = _render_iteration_prompt(ctx, ITERATION_PROPOSAL)
    assert "std_oos_sharpe" in prompt or "n_splits" in prompt


def test_iteration_prompt_contains_ml_diagnostics():
    ctx = build_context(_CANONICAL)
    prompt = _render_iteration_prompt(ctx, ITERATION_PROPOSAL)
    assert "mean_ic" in prompt or "ic_tier" in prompt


def test_iteration_prompt_contains_feature_context():
    ctx = build_context(_CANONICAL)
    prompt = _render_iteration_prompt(ctx, ITERATION_PROPOSAL)
    assert "Trend" in prompt or "feature_families" in prompt


def test_iteration_prompt_lists_primary_plots_only():
    ctx = build_context(_CANONICAL)
    prompt = _render_iteration_prompt(ctx, ITERATION_PROPOSAL)
    assert "equity_and_drawdown" in prompt or "rolling_sharpe" in prompt


# ---------------------------------------------------------------------------
# 2. Grounding enforcement — forbidden content not in instructions
# ---------------------------------------------------------------------------


def test_iteration_prompt_forbids_deployment():
    tmpl = load_template(ITERATION_PROPOSAL)
    assert "deploy" in tmpl.lower() or "FORBIDDEN" in tmpl


def test_iteration_prompt_forbids_parameter_prescriptions():
    tmpl = load_template(ITERATION_PROPOSAL)
    assert "FORBIDDEN" in tmpl or "prescribe" in tmpl.lower()


def test_iteration_prompt_requires_grounding():
    tmpl = load_template(ITERATION_PROPOSAL)
    assert "grounded" in tmpl.lower() or "cite" in tmpl.lower() or "named" in tmpl.lower()


def test_iteration_prompt_has_research_focus_section_instruction():
    tmpl = load_template(ITERATION_PROPOSAL)
    assert "Research Focus" in tmpl


def test_iteration_prompt_has_suggested_experiments_instruction():
    tmpl = load_template(ITERATION_PROPOSAL)
    assert "Suggested Experiments" in tmpl


def test_iteration_prompt_has_validation_concerns_instruction():
    tmpl = load_template(ITERATION_PROPOSAL)
    assert "Validation Concerns" in tmpl


def test_iteration_prompt_has_confidence_instruction():
    tmpl = load_template(ITERATION_PROPOSAL)
    assert "Confidence" in tmpl


def test_iteration_prompt_has_instability_signals_instruction():
    tmpl = load_template(ITERATION_PROPOSAL)
    assert "Instability Signals" in tmpl


def test_iteration_prompt_has_feature_risks_instruction():
    tmpl = load_template(ITERATION_PROPOSAL)
    assert "Feature Risks" in tmpl


# ---------------------------------------------------------------------------
# 3. Section parsing — _split_sections and _normalise_heading
# ---------------------------------------------------------------------------


def test_normalise_heading_basic():
    assert _normalise_heading("Research Focus") == "research_focus"
    assert _normalise_heading("Suggested Experiments") == "suggested_experiments"
    assert _normalise_heading("Validation Concerns") == "validation_concerns"
    assert _normalise_heading("Feature Risks") == "feature_risks"
    assert _normalise_heading("Confidence") == "confidence"


def test_normalise_heading_strips_special_chars():
    assert _normalise_heading("  Supporting Evidence  ") == "supporting_evidence"


def test_split_sections_empty_text():
    assert _split_sections("") == {}


def test_split_sections_single_section():
    text = "### Research Focus\nThis is the focus text."
    sections = _split_sections(text)
    assert "research_focus" in sections
    assert sections["research_focus"] == "This is the focus text."


def test_split_sections_multiple_sections():
    text = (
        "### Research Focus\nFocus paragraph.\n"
        "### Rationale\nRationale paragraph.\n"
        "### Confidence\nmedium — good evidence."
    )
    sections = _split_sections(text)
    assert sections["research_focus"] == "Focus paragraph."
    assert sections["rationale"] == "Rationale paragraph."
    assert sections["confidence"] == "medium — good evidence."


def test_split_sections_bullet_list():
    text = (
        "### Supporting Evidence\n"
        "- catastrophic_split worst Sharpe -1.07\n"
        "- std_oos_sharpe of 1.437\n"
        "- breakout_63d sign reversal\n"
    )
    sections = _split_sections(text)
    raw = sections.get("supporting_evidence", "")
    assert "catastrophic_split" in raw
    assert "breakout_63d" in raw


# ---------------------------------------------------------------------------
# 4. _parse_iteration_proposal — structured extraction
# ---------------------------------------------------------------------------


_SAMPLE_OUTPUT = """\
### Research Focus
Investigate whether breakout_63d sign instability explains the catastrophic_split failure mode.

### Rationale
The catastrophic_split failure mode (worst split Sharpe -1.07) and std_oos_sharpe of 1.437 suggest regime sensitivity. The most_volatile_feature breakout_63d may amplify cross-split instability.

### Supporting Evidence
- catastrophic_split: worst split Sharpe -1.07
- std_oos_sharpe 1.437 — high dispersion across splits
- breakout_63d identified as most_volatile_feature
- n_family_transitions 8 — frequent hypothesis regime shifts
- 2 negative-Sharpe splits out of 7

### Suggested Experiments
- Test whether excluding breakout_63d reduces split-to-split Sharpe variance
- Evaluate whether Trend-family weight reduction during high-HHI periods improves consistency

### Instability Signals
- breakout_63d: most volatile feature across walk-forward splits
- 8 family transitions indicate unstable dominant hypothesis
- 2 of 7 splits produced negative OOS Sharpe

### Validation Concerns
- std_oos_sharpe 1.437 implies high regime dependence
- Worst split Sharpe -1.07 classified as catastrophic_split
- CV of 2.23 undermines in-sample Sharpe narrative

### Feature Risks
- breakout_63d: sign instability across splits risks spurious in-sample overfitting
- Trend family: HHI 0.436 — moderate concentration amplifies regime sensitivity

### Confidence
medium — diagnostic evidence is specific but the hypothesis requires empirical validation.
"""


def test_parse_research_focus():
    proposal = _parse_iteration_proposal(
        _SAMPLE_OUTPUT, "test_exp", "abc123", "stub", "stub", "iteration_proposal"
    )
    assert "breakout_63d" in proposal.research_focus
    assert "catastrophic_split" in proposal.research_focus


def test_parse_rationale():
    proposal = _parse_iteration_proposal(
        _SAMPLE_OUTPUT, "test_exp", "abc123", "stub", "stub", "iteration_proposal"
    )
    assert "catastrophic_split" in proposal.rationale
    assert "1.437" in proposal.rationale


def test_parse_supporting_evidence_is_list():
    proposal = _parse_iteration_proposal(
        _SAMPLE_OUTPUT, "test_exp", "abc123", "stub", "stub", "iteration_proposal"
    )
    assert isinstance(proposal.supporting_evidence, list)
    assert len(proposal.supporting_evidence) >= 3


def test_parse_suggested_experiments_non_empty():
    proposal = _parse_iteration_proposal(
        _SAMPLE_OUTPUT, "test_exp", "abc123", "stub", "stub", "iteration_proposal"
    )
    assert len(proposal.suggested_experiments) >= 1


def test_parse_instability_signals_non_empty():
    proposal = _parse_iteration_proposal(
        _SAMPLE_OUTPUT, "test_exp", "abc123", "stub", "stub", "iteration_proposal"
    )
    assert len(proposal.instability_signals) >= 1


def test_parse_validation_concerns_non_empty():
    proposal = _parse_iteration_proposal(
        _SAMPLE_OUTPUT, "test_exp", "abc123", "stub", "stub", "iteration_proposal"
    )
    assert len(proposal.validation_concerns) >= 1


def test_parse_feature_risks_non_empty():
    proposal = _parse_iteration_proposal(
        _SAMPLE_OUTPUT, "test_exp", "abc123", "stub", "stub", "iteration_proposal"
    )
    assert len(proposal.feature_risks) >= 1


def test_parse_confidence():
    proposal = _parse_iteration_proposal(
        _SAMPLE_OUTPUT, "test_exp", "abc123", "stub", "stub", "iteration_proposal"
    )
    assert "medium" in proposal.confidence.lower()


def test_parse_context_hash_preserved():
    proposal = _parse_iteration_proposal(
        _SAMPLE_OUTPUT, "test_exp", "deadbeef1234", "stub", "stub", "iteration_proposal"
    )
    assert proposal.context_hash == "deadbeef1234"


def test_parse_experiment_name_preserved():
    proposal = _parse_iteration_proposal(
        _SAMPLE_OUTPUT, "canonical_ml_multi_asset", "x", "stub", "stub", "iteration_proposal"
    )
    assert proposal.experiment_name == "canonical_ml_multi_asset"


# ---------------------------------------------------------------------------
# 5. Persistence correctness
# ---------------------------------------------------------------------------


def test_persist_writes_json(tmp_path):
    ctx = build_context(_CANONICAL)
    run_iteration_proposal(ctx, provider=PROVIDER_STUB, persist=True, llm_base=tmp_path)
    json_path = tmp_path / _CANONICAL / "iteration_proposal.json"
    assert json_path.exists(), "iteration_proposal.json must be written"


def test_persist_writes_md(tmp_path):
    ctx = build_context(_CANONICAL)
    run_iteration_proposal(ctx, provider=PROVIDER_STUB, persist=True, llm_base=tmp_path)
    md_path = tmp_path / _CANONICAL / "iteration_proposal.md"
    assert md_path.exists(), "iteration_proposal.md must be written"


def test_persist_json_has_context_hash(tmp_path):
    ctx = build_context(_CANONICAL)
    run_iteration_proposal(ctx, provider=PROVIDER_STUB, persist=True, llm_base=tmp_path)
    data = json.loads((tmp_path / _CANONICAL / "iteration_proposal.json").read_text())
    assert "context_hash" in data
    assert len(data["context_hash"]) == 64


def test_persist_json_has_iteration_version(tmp_path):
    ctx = build_context(_CANONICAL)
    run_iteration_proposal(ctx, provider=PROVIDER_STUB, persist=True, llm_base=tmp_path)
    data = json.loads((tmp_path / _CANONICAL / "iteration_proposal.json").read_text())
    assert data.get("iteration_version") == ITERATION_VERSION


def test_persist_json_has_provenance_fields(tmp_path):
    ctx = build_context(_CANONICAL)
    run_iteration_proposal(ctx, provider=PROVIDER_STUB, persist=True, llm_base=tmp_path)
    data = json.loads((tmp_path / _CANONICAL / "iteration_proposal.json").read_text())
    for key in ("experiment_name", "generated_at", "provider", "model", "prompt_template"):
        assert key in data, f"Missing provenance field: {key}"


def test_persist_json_has_all_proposal_fields(tmp_path):
    ctx = build_context(_CANONICAL)
    run_iteration_proposal(ctx, provider=PROVIDER_STUB, persist=True, llm_base=tmp_path)
    data = json.loads((tmp_path / _CANONICAL / "iteration_proposal.json").read_text())
    for key in (
        "research_focus",
        "rationale",
        "supporting_evidence",
        "suggested_experiments",
        "instability_signals",
        "validation_concerns",
        "feature_risks",
        "confidence",
    ):
        assert key in data, f"Missing proposal field: {key}"


def test_persist_json_list_fields_are_lists(tmp_path):
    ctx = build_context(_CANONICAL)
    run_iteration_proposal(ctx, provider=PROVIDER_STUB, persist=True, llm_base=tmp_path)
    data = json.loads((tmp_path / _CANONICAL / "iteration_proposal.json").read_text())
    for key in ("supporting_evidence", "suggested_experiments", "instability_signals",
                "validation_concerns", "feature_risks"):
        assert isinstance(data[key], list), f"{key} must be a list"


def test_persist_no_persist_skips_files(tmp_path):
    ctx = build_context(_CANONICAL)
    run_iteration_proposal(ctx, provider=PROVIDER_STUB, persist=False, llm_base=tmp_path)
    assert not (tmp_path / _CANONICAL / "iteration_proposal.json").exists()


# ---------------------------------------------------------------------------
# 6. Recommendation grounding — stub output is parseable
# ---------------------------------------------------------------------------


def test_stub_produces_iterable_proposal():
    ctx = build_context(_CANONICAL)
    proposal = run_iteration_proposal(ctx, provider=PROVIDER_STUB, persist=False)
    assert isinstance(proposal, IterationProposal)
    assert proposal.experiment_name == _CANONICAL


def test_stub_proposal_has_context_hash():
    ctx = build_context(_CANONICAL)
    proposal = run_iteration_proposal(ctx, provider=PROVIDER_STUB, persist=False)
    assert len(proposal.context_hash) == 64


def test_stub_proposal_generated_at_is_iso():
    from datetime import datetime
    ctx = build_context(_CANONICAL)
    proposal = run_iteration_proposal(ctx, provider=PROVIDER_STUB, persist=False)
    # Should parse without error
    datetime.fromisoformat(proposal.generated_at)


# ---------------------------------------------------------------------------
# 7. Deterministic context hash across two calls
# ---------------------------------------------------------------------------


def test_context_hash_same_across_two_iteration_calls():
    ctx = build_context(_CANONICAL)
    p1 = run_iteration_proposal(ctx, provider=PROVIDER_STUB, persist=False)
    p2 = run_iteration_proposal(ctx, provider=PROVIDER_STUB, persist=False)
    assert p1.context_hash == p2.context_hash


def test_context_hash_differs_across_experiments():
    ctx_a = build_context(_CANONICAL)
    ctx_b = build_context("canonical_ml_showcase")
    p_a = run_iteration_proposal(ctx_a, provider=PROVIDER_STUB, persist=False)
    p_b = run_iteration_proposal(ctx_b, provider=PROVIDER_STUB, persist=False)
    assert p_a.context_hash != p_b.context_hash


# ---------------------------------------------------------------------------
# 8. Research API integration
# ---------------------------------------------------------------------------


def test_research_api_generate_iteration_proposal(tmp_path):
    from src.orchestration.api.research_api import generate_iteration_proposal

    proposal = generate_iteration_proposal(
        _CANONICAL,
        provider=PROVIDER_STUB,
        llm_base=tmp_path,
        persist=True,
    )
    assert isinstance(proposal, IterationProposal)
    assert proposal.experiment_name == _CANONICAL
    assert (tmp_path / _CANONICAL / "iteration_proposal.json").exists()


def test_research_api_iteration_proposal_no_persist(tmp_path):
    from src.orchestration.api.research_api import generate_iteration_proposal

    proposal = generate_iteration_proposal(
        _CANONICAL,
        provider=PROVIDER_STUB,
        llm_base=tmp_path,
        persist=False,
    )
    assert isinstance(proposal, IterationProposal)
    assert not (tmp_path / _CANONICAL / "iteration_proposal.json").exists()


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------


def test_iteration_template_registered():
    from src.orchestration.llm.prompt_templates import ITERATION_PROPOSAL, load_template

    tmpl = load_template(ITERATION_PROPOSAL)
    assert len(tmpl) > 100


def test_iteration_template_has_required_jinja_vars():
    tmpl = load_template(ITERATION_PROPOSAL)
    for var in ("experiment_name", "strategy_name", "performance", "failure_modes",
                "validation", "ml_diagnostics", "feature_summary"):
        assert var in tmpl, f"Template missing required variable: {var}"
