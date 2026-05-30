"""Tests for orchestration.llm: stub provider and review schema."""

import json

import pytest

from src.orchestration.llm.llm_interface import call_llm, LLMResponse
from src.orchestration.llm.review_schema import (
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI,
    PROVIDER_STUB,
    ALL_SECTIONS,
)
from src.orchestration.llm.prompt_templates import load_template, EXPERIMENT_REVIEW


def test_stub_provider_returns_response():
    resp = call_llm("test prompt", provider=PROVIDER_STUB)
    assert isinstance(resp, LLMResponse)
    assert resp.provider == PROVIDER_STUB
    assert "STUB" in resp.text
    assert resp.model == "stub"


def test_stub_provider_no_external_calls():
    # If this reaches an external API it would hang or fail — stub must be local
    resp = call_llm("some prompt", provider=PROVIDER_STUB, model="custom-stub")
    assert resp.provider == PROVIDER_STUB
    assert resp.model == "custom-stub"


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        call_llm("prompt", provider="made_up_provider")


def test_all_sections_not_empty():
    assert len(ALL_SECTIONS) > 0
    for section in ALL_SECTIONS:
        assert isinstance(section, str)
        assert section != ""


def test_load_experiment_review_template():
    tmpl = load_template(EXPERIMENT_REVIEW)
    assert "experiment_name" in tmpl
    assert "performance" in tmpl
    assert "failure_modes" in tmpl


def test_load_nonexistent_template_raises():
    with pytest.raises(KeyError):
        load_template("nonexistent_template_xyz")


# ---------------------------------------------------------------------------
# Phase 1.5 — Fix 4: context hash provenance in persisted review
# ---------------------------------------------------------------------------


def test_persist_review_includes_context_hash(tmp_path):
    """Persisted llm_review.json must contain a 64-char SHA256 context_hash."""
    from src.orchestration.context.context_builder import build_context
    from src.orchestration.llm.review_engine import run_review

    ctx = build_context("canonical_ml_multi_asset")
    run_review(ctx, provider=PROVIDER_STUB, persist=True, llm_base=tmp_path)

    review_file = tmp_path / "canonical_ml_multi_asset" / "llm_review.json"
    assert review_file.exists(), "llm_review.json must be created"
    data = json.loads(review_file.read_text())
    assert "context_hash" in data, "context_hash must be present in persisted review"
    assert len(data["context_hash"]) == 64, "context_hash must be a 64-char hex SHA256"
    assert all(c in "0123456789abcdef" for c in data["context_hash"])


def test_compute_context_hash_is_deterministic():
    """_compute_context_hash must return the same value for the same context."""
    from src.orchestration.context.context_builder import build_context
    from src.orchestration.llm.review_engine import _compute_context_hash

    ctx = build_context("canonical_ml_multi_asset")
    h1 = _compute_context_hash(ctx)
    h2 = _compute_context_hash(ctx)
    assert h1 == h2


def test_compute_context_hash_changes_on_different_experiment():
    """Different experiments must produce different context hashes."""
    from src.orchestration.context.context_builder import build_context
    from src.orchestration.llm.review_engine import _compute_context_hash

    h_multi = _compute_context_hash(build_context("canonical_ml_multi_asset"))
    h_showcase = _compute_context_hash(build_context("canonical_ml_showcase"))
    assert h_multi != h_showcase
