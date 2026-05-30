"""Tests for Phase 2B-A — Comparative Research Review Orchestration.

Validates:
- comparative prompt rendering and grounding enforcement
- no unresolved template tokens
- comparison payload correctness (deltas, failure mode diffs)
- section parsing
- persistence of JSON and MD artefacts
- deterministic context hashing
- forbidden language checks
- research API integration
"""

from __future__ import annotations

import json
import re

import pytest
from src.orchestration.api.schemas import ComparativeReview
from src.orchestration.context.context_builder import build_context
from src.orchestration.llm.comparison_engine import (
    COMPARISON_VERSION,
    _build_comparative_payload,
    _compute_comparison_hash,
    _delta,
    _parse_comparative_review,
    _render_comparative_prompt,
    _split_sections,
    run_comparative_review,
)
from src.orchestration.llm.prompt_templates import COMPARATIVE_REVIEW, load_template
from src.orchestration.llm.review_schema import PROVIDER_STUB

_BASELINE = "canonical_ml_multi_asset"
_CANDIDATE = "canonical_ml_showcase"
_UNRESOLVED_RE = re.compile(r"\{\{|\}\}|\{%|%\}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def baseline_ctx():
    return build_context(_BASELINE)


@pytest.fixture(scope="module")
def candidate_ctx():
    return build_context(_CANDIDATE)


@pytest.fixture(scope="module")
def comparative_payload(baseline_ctx, candidate_ctx):
    return _build_comparative_payload(baseline_ctx, candidate_ctx)


# ---------------------------------------------------------------------------
# 1. Template rendering — no unresolved tokens
# ---------------------------------------------------------------------------


def test_comparative_prompt_renders_no_unresolved_tokens(baseline_ctx, candidate_ctx, comparative_payload):
    prompt = _render_comparative_prompt(baseline_ctx, candidate_ctx, comparative_payload, COMPARATIVE_REVIEW)
    remaining = _UNRESOLVED_RE.findall(prompt)
    assert remaining == [], f"Unresolved tokens: {remaining}"


def test_comparative_prompt_contains_both_experiment_names(baseline_ctx, candidate_ctx, comparative_payload):
    prompt = _render_comparative_prompt(baseline_ctx, candidate_ctx, comparative_payload, COMPARATIVE_REVIEW)
    assert _BASELINE in prompt
    assert _CANDIDATE in prompt


def test_comparative_prompt_contains_baseline_failure_modes(baseline_ctx, candidate_ctx, comparative_payload):
    prompt = _render_comparative_prompt(baseline_ctx, candidate_ctx, comparative_payload, COMPARATIVE_REVIEW)
    for fm in baseline_ctx.failure_modes:
        assert fm["name"] in prompt


def test_comparative_prompt_contains_candidate_failure_modes(baseline_ctx, candidate_ctx, comparative_payload):
    prompt = _render_comparative_prompt(baseline_ctx, candidate_ctx, comparative_payload, COMPARATIVE_REVIEW)
    for fm in candidate_ctx.failure_modes:
        assert fm["name"] in prompt


def test_comparative_prompt_contains_delta_payload(baseline_ctx, candidate_ctx, comparative_payload):
    prompt = _render_comparative_prompt(baseline_ctx, candidate_ctx, comparative_payload, COMPARATIVE_REVIEW)
    assert "mean_oos_sharpe" in prompt
    assert "mean_hhi" in prompt


def test_comparative_prompt_contains_validation_data(baseline_ctx, candidate_ctx, comparative_payload):
    prompt = _render_comparative_prompt(baseline_ctx, candidate_ctx, comparative_payload, COMPARATIVE_REVIEW)
    assert "std_oos_sharpe" in prompt or "n_negative_sharpe_splits" in prompt


# ---------------------------------------------------------------------------
# 2. Template grounding enforcement
# ---------------------------------------------------------------------------


def test_template_forbids_deployment():
    tmpl = load_template(COMPARATIVE_REVIEW)
    assert "FORBIDDEN" in tmpl
    assert "deploy" in tmpl.lower()


def test_template_forbids_ranking_language():
    tmpl = load_template(COMPARATIVE_REVIEW)
    forbidden = tmpl.lower()
    assert "superior" in forbidden or "outperform" in forbidden or "ranking" in forbidden


def test_template_requires_tradeoff_framing():
    tmpl = load_template(COMPARATIVE_REVIEW)
    assert "tradeoff" in tmpl.lower() or "cost" in tmpl.lower()


def test_template_requires_validation_first():
    tmpl = load_template(COMPARATIVE_REVIEW)
    assert "Validation Changes" in tmpl


def test_template_requires_failure_mode_evolution():
    tmpl = load_template(COMPARATIVE_REVIEW)
    assert "Failure Mode Changes" in tmpl


def test_template_has_all_required_sections():
    tmpl = load_template(COMPARATIVE_REVIEW)
    required = [
        "Overall Assessment",
        "Validation Changes",
        "Instability Changes",
        "Feature Behavior Changes",
        "Robustness Changes",
        "Failure Mode Changes",
        "Key Tradeoffs",
        "Research Progression Summary",
        "Confidence",
    ]
    for section in required:
        assert section in tmpl, f"Template missing section: {section}"


def test_template_has_required_jinja_vars():
    tmpl = load_template(COMPARATIVE_REVIEW)
    for var in ("baseline_experiment", "candidate_experiment", "baseline", "candidate",
                "metric_comparison", "failure_mode_comparison", "ml_comparison", "feature_comparison"):
        assert var in tmpl, f"Template missing variable: {var}"


# ---------------------------------------------------------------------------
# 3. Comparative payload correctness
# ---------------------------------------------------------------------------


def test_payload_has_all_top_level_keys(comparative_payload):
    for key in ("metric_comparison", "failure_mode_comparison", "ml_comparison",
                "feature_comparison", "validation_comparison", "universe_comparison"):
        assert key in comparative_payload, f"Missing payload key: {key}"


def test_metric_comparison_has_sharpe_delta(comparative_payload):
    mc = comparative_payload["metric_comparison"]
    assert "sharpe_ratio" in mc
    delta = mc["sharpe_ratio"]["delta"]
    assert delta is not None
    # showcase Sharpe (0.695) > multi_asset Sharpe (0.548): delta ~+0.147
    assert delta > 0


def test_metric_comparison_has_drawdown_pct(comparative_payload):
    # max_drawdown is stored as max_drawdown_pct (string) in the performance summary
    mc = comparative_payload["metric_comparison"]
    assert "max_drawdown_pct" in mc
    assert mc["max_drawdown_pct"]["baseline"] is not None
    assert mc["max_drawdown_pct"]["candidate"] is not None


def test_failure_mode_comparison_baseline_only(comparative_payload):
    fmc = comparative_payload["failure_mode_comparison"]
    assert "baseline_only" in fmc
    # severe_drawdown is only in multi_asset
    assert "severe_drawdown" in fmc["baseline_only"]


def test_failure_mode_comparison_candidate_only(comparative_payload):
    fmc = comparative_payload["failure_mode_comparison"]
    assert "candidate_only" in fmc
    # poor_oos_consistency and high_split_sharpe_variance are only in showcase
    assert "poor_oos_consistency" in fmc["candidate_only"]


def test_failure_mode_comparison_shared(comparative_payload):
    fmc = comparative_payload["failure_mode_comparison"]
    assert "shared" in fmc
    # catastrophic_split is in both
    assert "catastrophic_split" in fmc["shared"]


def test_ml_comparison_ic_delta(comparative_payload):
    ml = comparative_payload["ml_comparison"]
    assert ml.get("available") is True
    delta = ml["mean_ic"]["delta"]
    assert delta is not None
    # showcase IC (0.488) > multi_asset IC (0.145)
    assert delta > 0


def test_feature_comparison_hhi_delta(comparative_payload):
    fc = comparative_payload["feature_comparison"]
    assert "mean_hhi" in fc
    delta = fc["mean_hhi"]["delta"]
    assert delta is not None
    # showcase HHI (0.577) > multi_asset HHI (0.437)
    assert delta > 0


def test_feature_comparison_transitions_delta(comparative_payload):
    fc = comparative_payload["feature_comparison"]
    delta = fc["n_family_transitions"]["delta"]
    assert delta is not None
    # showcase has more transitions (17 vs 8)
    assert delta > 0


def test_feature_comparison_names_volatile_features(comparative_payload):
    fc = comparative_payload["feature_comparison"]
    mvf = fc["most_volatile_feature"]
    assert mvf["baseline"] == "breakout_63d"
    assert mvf["candidate"] == "mom_60"


def test_validation_comparison_oos_sharpe_delta(comparative_payload):
    vc = comparative_payload["validation_comparison"]
    delta = vc["mean_oos_sharpe"]["delta"]
    assert delta is not None
    # showcase mean OOS (-0.32) < multi_asset (0.645): delta negative
    assert delta < 0


def test_universe_comparison_n_assets(comparative_payload):
    uc = comparative_payload["universe_comparison"]
    assert uc["n_assets"]["baseline"] == 9
    assert uc["n_assets"]["candidate"] == 1


# ---------------------------------------------------------------------------
# 4. _delta helper
# ---------------------------------------------------------------------------


def test_delta_basic():
    assert _delta(0.5, 0.7) == pytest.approx(0.2, abs=1e-4)


def test_delta_negative():
    assert _delta(0.7, 0.3) == pytest.approx(-0.4, abs=1e-4)


def test_delta_none_inputs():
    assert _delta(None, 0.5) is None
    assert _delta(0.5, None) is None
    assert _delta(None, None) is None


# ---------------------------------------------------------------------------
# 5. Section parsing
# ---------------------------------------------------------------------------


_SAMPLE_COMPARATIVE = """\
### Overall Assessment
The candidate shows higher in-sample Sharpe but severe OOS degradation.

### Validation Changes
- mean_oos_sharpe fell from 0.645 to -0.320, a delta of -0.965
- n_negative_sharpe_splits increased from 2 to 4
- hit_rate dropped from 71.43% to 42.86%, triggering poor_oos_consistency

### Instability Changes
- std_oos_sharpe improved from 1.437 to 1.274
- n_family_transitions rose from 8 to 17

### Feature Behavior Changes
- mean_hhi rose from 0.437 to 0.577, shifting from moderately_concentrated to highly_concentrated
- most_volatile_feature changed from breakout_63d to mom_60

### Robustness Changes
- max_drawdown improved from -45.0% to -34.1%, removing severe_drawdown
- catastrophic_split persists in both experiments

### Failure Mode Changes
- resolved: severe_drawdown — removed by reduced drawdown depth
- gained: poor_oos_consistency — OOS hit rate fell below 50%
- gained: high_split_sharpe_variance — split CV exceeded 3.0
- persistent: catastrophic_split — worst split Sharpe remains below -1.0

### Key Tradeoffs
- drawdown vs OOS consistency: candidate reduces max drawdown at cost of OOS hit rate
- IC vs OOS Sharpe: candidate's IC 0.488 coexists with mean OOS Sharpe -0.32

### Research Progression Summary
The transition reveals an overfitting pattern — single-asset concentration produces higher in-sample IC
but collapses OOS consistency.

### Confidence
medium - strong IC and validation evidence but universe difference limits comparability.
"""


def test_split_sections_overall():
    sections = _split_sections(_SAMPLE_COMPARATIVE)
    assert "overall_assessment" in sections
    assert "candidate" in sections["overall_assessment"].lower()


def test_split_sections_validation_changes():
    sections = _split_sections(_SAMPLE_COMPARATIVE)
    assert "validation_changes" in sections
    assert "mean_oos_sharpe" in sections["validation_changes"]


def test_split_sections_failure_mode_changes():
    sections = _split_sections(_SAMPLE_COMPARATIVE)
    assert "failure_mode_changes" in sections


def test_split_sections_key_tradeoffs():
    sections = _split_sections(_SAMPLE_COMPARATIVE)
    assert "key_tradeoffs" in sections


def test_parse_comparative_review_overall():
    review = _parse_comparative_review(
        _SAMPLE_COMPARATIVE, _BASELINE, _CANDIDATE,
        "abc123", "stub", "stub", COMPARATIVE_REVIEW
    )
    assert isinstance(review, ComparativeReview)
    assert "OOS" in review.overall_assessment or "Sharpe" in review.overall_assessment


def test_parse_comparative_review_validation_changes():
    review = _parse_comparative_review(
        _SAMPLE_COMPARATIVE, _BASELINE, _CANDIDATE,
        "abc123", "stub", "stub", COMPARATIVE_REVIEW
    )
    assert isinstance(review.validation_changes, list)
    assert len(review.validation_changes) >= 2
    assert any("mean_oos_sharpe" in item for item in review.validation_changes)


def test_parse_comparative_review_failure_mode_changes():
    review = _parse_comparative_review(
        _SAMPLE_COMPARATIVE, _BASELINE, _CANDIDATE,
        "abc123", "stub", "stub", COMPARATIVE_REVIEW
    )
    assert len(review.failure_mode_changes) >= 2
    all_text = " ".join(review.failure_mode_changes)
    assert "severe_drawdown" in all_text or "poor_oos_consistency" in all_text


def test_parse_comparative_review_key_tradeoffs():
    review = _parse_comparative_review(
        _SAMPLE_COMPARATIVE, _BASELINE, _CANDIDATE,
        "abc123", "stub", "stub", COMPARATIVE_REVIEW
    )
    assert len(review.key_tradeoffs) >= 1


def test_parse_comparative_review_research_progression():
    review = _parse_comparative_review(
        _SAMPLE_COMPARATIVE, _BASELINE, _CANDIDATE,
        "abc123", "stub", "stub", COMPARATIVE_REVIEW
    )
    assert review.research_progression_summary != ""


def test_parse_comparative_review_confidence():
    review = _parse_comparative_review(
        _SAMPLE_COMPARATIVE, _BASELINE, _CANDIDATE,
        "abc123", "stub", "stub", COMPARATIVE_REVIEW
    )
    assert "medium" in review.confidence.lower()


def test_parse_preserves_experiment_names():
    review = _parse_comparative_review(
        _SAMPLE_COMPARATIVE, _BASELINE, _CANDIDATE,
        "abc123", "stub", "stub", COMPARATIVE_REVIEW
    )
    assert review.baseline_experiment == _BASELINE
    assert review.candidate_experiment == _CANDIDATE


def test_parse_preserves_context_hash():
    review = _parse_comparative_review(
        _SAMPLE_COMPARATIVE, _BASELINE, _CANDIDATE,
        "deadbeef1234567890", "stub", "stub", COMPARATIVE_REVIEW
    )
    assert review.context_hash == "deadbeef1234567890"


# ---------------------------------------------------------------------------
# 6. Deterministic hashing
# ---------------------------------------------------------------------------


def test_comparison_hash_is_deterministic(baseline_ctx, candidate_ctx):
    h1 = _compute_comparison_hash(baseline_ctx, candidate_ctx)
    h2 = _compute_comparison_hash(baseline_ctx, candidate_ctx)
    assert h1 == h2


def test_comparison_hash_is_64_chars(baseline_ctx, candidate_ctx):
    h = _compute_comparison_hash(baseline_ctx, candidate_ctx)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_comparison_hash_order_dependent(baseline_ctx, candidate_ctx):
    h_ab = _compute_comparison_hash(baseline_ctx, candidate_ctx)
    h_ba = _compute_comparison_hash(candidate_ctx, baseline_ctx)
    assert h_ab != h_ba


# ---------------------------------------------------------------------------
# 7. Persistence correctness
# ---------------------------------------------------------------------------


def test_persist_writes_json(tmp_path, baseline_ctx, candidate_ctx):
    run_comparative_review(baseline_ctx, candidate_ctx, provider=PROVIDER_STUB,
                           persist=True, comparisons_base=tmp_path)
    json_path = tmp_path / f"{_BASELINE}__vs__{_CANDIDATE}" / "comparative_review.json"
    assert json_path.exists()


def test_persist_writes_md(tmp_path, baseline_ctx, candidate_ctx):
    run_comparative_review(baseline_ctx, candidate_ctx, provider=PROVIDER_STUB,
                           persist=True, comparisons_base=tmp_path)
    md_path = tmp_path / f"{_BASELINE}__vs__{_CANDIDATE}" / "comparative_review.md"
    assert md_path.exists()


def test_persist_json_has_context_hash(tmp_path, baseline_ctx, candidate_ctx):
    run_comparative_review(baseline_ctx, candidate_ctx, provider=PROVIDER_STUB,
                           persist=True, comparisons_base=tmp_path)
    json_path = tmp_path / f"{_BASELINE}__vs__{_CANDIDATE}" / "comparative_review.json"
    data = json.loads(json_path.read_text())
    assert "context_hash" in data
    assert len(data["context_hash"]) == 64


def test_persist_json_has_comparison_version(tmp_path, baseline_ctx, candidate_ctx):
    run_comparative_review(baseline_ctx, candidate_ctx, provider=PROVIDER_STUB,
                           persist=True, comparisons_base=tmp_path)
    json_path = tmp_path / f"{_BASELINE}__vs__{_CANDIDATE}" / "comparative_review.json"
    data = json.loads(json_path.read_text())
    assert data.get("comparison_version") == COMPARISON_VERSION


def test_persist_json_has_provenance_fields(tmp_path, baseline_ctx, candidate_ctx):
    run_comparative_review(baseline_ctx, candidate_ctx, provider=PROVIDER_STUB,
                           persist=True, comparisons_base=tmp_path)
    json_path = tmp_path / f"{_BASELINE}__vs__{_CANDIDATE}" / "comparative_review.json"
    data = json.loads(json_path.read_text())
    for field in ("baseline_experiment", "candidate_experiment", "generated_at",
                  "provider", "model", "prompt_template"):
        assert field in data, f"Missing provenance field: {field}"


def test_persist_json_has_all_review_fields(tmp_path, baseline_ctx, candidate_ctx):
    run_comparative_review(baseline_ctx, candidate_ctx, provider=PROVIDER_STUB,
                           persist=True, comparisons_base=tmp_path)
    json_path = tmp_path / f"{_BASELINE}__vs__{_CANDIDATE}" / "comparative_review.json"
    data = json.loads(json_path.read_text())
    for field in ("overall_assessment", "validation_changes", "instability_changes",
                  "feature_behavior_changes", "robustness_changes", "failure_mode_changes",
                  "key_tradeoffs", "research_progression_summary", "confidence"):
        assert field in data, f"Missing review field: {field}"


def test_persist_json_list_fields_are_lists(tmp_path, baseline_ctx, candidate_ctx):
    run_comparative_review(baseline_ctx, candidate_ctx, provider=PROVIDER_STUB,
                           persist=True, comparisons_base=tmp_path)
    json_path = tmp_path / f"{_BASELINE}__vs__{_CANDIDATE}" / "comparative_review.json"
    data = json.loads(json_path.read_text())
    for field in ("validation_changes", "instability_changes", "feature_behavior_changes",
                  "robustness_changes", "failure_mode_changes", "key_tradeoffs"):
        assert isinstance(data[field], list), f"{field} must be a list"


def test_persist_no_persist_skips_files(tmp_path, baseline_ctx, candidate_ctx):
    run_comparative_review(baseline_ctx, candidate_ctx, provider=PROVIDER_STUB,
                           persist=False, comparisons_base=tmp_path)
    assert not (tmp_path / f"{_BASELINE}__vs__{_CANDIDATE}" / "comparative_review.json").exists()


# ---------------------------------------------------------------------------
# 8. Stub produces ComparativeReview
# ---------------------------------------------------------------------------


def test_stub_produces_comparative_review(baseline_ctx, candidate_ctx):
    review = run_comparative_review(baseline_ctx, candidate_ctx,
                                    provider=PROVIDER_STUB, persist=False)
    assert isinstance(review, ComparativeReview)
    assert review.baseline_experiment == _BASELINE
    assert review.candidate_experiment == _CANDIDATE


def test_stub_review_has_context_hash(baseline_ctx, candidate_ctx):
    review = run_comparative_review(baseline_ctx, candidate_ctx,
                                    provider=PROVIDER_STUB, persist=False)
    assert len(review.context_hash) == 64


def test_stub_review_generated_at_is_iso(baseline_ctx, candidate_ctx):
    from datetime import datetime
    review = run_comparative_review(baseline_ctx, candidate_ctx,
                                    provider=PROVIDER_STUB, persist=False)
    datetime.fromisoformat(review.generated_at)


# ---------------------------------------------------------------------------
# 9. Research API integration
# ---------------------------------------------------------------------------


def test_research_api_run_comparative_review(tmp_path):
    from src.orchestration.api.research_api import run_llm_comparative_review

    review = run_llm_comparative_review(
        _BASELINE, _CANDIDATE,
        provider=PROVIDER_STUB,
        comparisons_base=tmp_path,
        persist=True,
    )
    assert isinstance(review, ComparativeReview)
    json_path = tmp_path / f"{_BASELINE}__vs__{_CANDIDATE}" / "comparative_review.json"
    assert json_path.exists()


def test_research_api_no_persist(tmp_path):
    from src.orchestration.api.research_api import run_llm_comparative_review

    review = run_llm_comparative_review(
        _BASELINE, _CANDIDATE,
        provider=PROVIDER_STUB,
        comparisons_base=tmp_path,
        persist=False,
    )
    assert isinstance(review, ComparativeReview)
    assert not (tmp_path / f"{_BASELINE}__vs__{_CANDIDATE}" / "comparative_review.json").exists()


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------


def test_comparative_review_template_registered():
    tmpl = load_template(COMPARATIVE_REVIEW)
    assert len(tmpl) > 200


def test_comparative_review_template_is_jinja2():
    tmpl = load_template(COMPARATIVE_REVIEW)
    assert "{{" in tmpl and "}}" in tmpl
