"""Shared fixtures and stub factories for API bridge tests.

All backend functions are mocked — no disk I/O, no LLM calls, no quant engine.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.orchestration.api.schemas import (
    ComparativeReview,
    ExperimentSummary,
    IterationProposal,
    LLMReviewOutput,
)
from src.orchestration.config_generation.draft_schema import (
    DraftChange,
    DraftValidationResult,
    ExperimentDraft,
)
from src.orchestration.session.session_schema import ResearchSession, SessionStatus


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# Stub factories
# ---------------------------------------------------------------------------


def make_stub_session(**kwargs) -> ResearchSession:
    defaults = dict(
        session_id="test-session-id",
        research_goal="Explore regularisation sweep",
        root_experiment="exp_a",
        active_experiment="exp_a",
        status=SessionStatus.ACTIVE,
        created_at="2026-05-30T00:00:00+00:00",
        updated_at="2026-05-30T00:00:00+00:00",
    )
    defaults.update(kwargs)
    return ResearchSession(**defaults)


def make_stub_review(**kwargs) -> LLMReviewOutput:
    defaults = dict(
        experiment_name="exp_a",
        provider="stub",
        model="stub-model",
        prompt_template="review",
        review_text="Strong OOS stability observed.",
        sections={"executive_summary": "Solid result."},
        generated_at="2026-05-30T00:00:00+00:00",
    )
    defaults.update(kwargs)
    return LLMReviewOutput(**defaults)


def make_stub_proposal(**kwargs) -> IterationProposal:
    defaults = dict(
        experiment_name="exp_a",
        generated_at="2026-05-30T00:00:00+00:00",
        context_hash="abc123",
        research_focus="Regularisation sweep",
        rationale="Alpha compression suggests lower regularisation.",
        supporting_evidence=["IC is stable"],
        suggested_experiments=["exp_a_ridge_low"],
        instability_signals=["High HHI"],
        validation_concerns=["Short backtest"],
        feature_risks=["Momentum crowding"],
        confidence="medium",
    )
    defaults.update(kwargs)
    return IterationProposal(**defaults)


def make_stub_draft(approved: bool = False, **kwargs) -> ExperimentDraft:
    defaults = dict(
        draft_id="draft-001",
        draft_hash="abc123def456",
        base_experiment="exp_a",
        source_proposal_hash="prop123",
        proposed_name="exp_a_v2",
        changes=[
            DraftChange(
                section="model",
                field="params.alpha",
                current_value=0.1,
                proposed_value=0.5,
                rationale="Reduce overfitting.",
            )
        ],
        generated_at="2026-05-30T00:00:00+00:00",
        approved=approved,
        approved_at="2026-05-30T00:01:00+00:00" if approved else None,
    )
    defaults.update(kwargs)
    return ExperimentDraft(**defaults)


def make_stub_summary(**kwargs) -> ExperimentSummary:
    defaults = dict(
        experiment_name="exp_a",
        strategy_name="MLStrategy",
        created_at="2026-05-30T00:00:00+00:00",
        tags=["ml", "ridge"],
        sharpe_ratio=1.25,
        annualized_return=0.14,
        max_drawdown=-0.08,
    )
    defaults.update(kwargs)
    return ExperimentSummary(**defaults)


def make_stub_comparison(**kwargs) -> ComparativeReview:
    defaults = dict(
        baseline_experiment="exp_a",
        candidate_experiment="exp_b",
        generated_at="2026-05-30T00:00:00+00:00",
        context_hash="cmp123",
        overall_assessment="Candidate shows marginal improvement.",
        validation_changes=["OOS Sharpe +0.1"],
        instability_changes=[],
        feature_behavior_changes=[],
        robustness_changes=[],
        failure_mode_changes=[],
        key_tradeoffs=["Higher drawdown"],
        research_progression_summary="Minor step forward.",
        confidence="medium",
    )
    defaults.update(kwargs)
    return ComparativeReview(**defaults)


def make_stub_validation(is_valid: bool = True) -> DraftValidationResult:
    errors = [] if is_valid else ["model.params.alpha must be between 0 and 1"]
    return DraftValidationResult(is_valid=is_valid, errors=errors)
