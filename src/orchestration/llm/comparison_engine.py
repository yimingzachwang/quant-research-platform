"""Comparison engine: builds comparative context, renders prompt, calls LLM, parses, persists.

Mirrors iteration_engine and review_engine architecture.
Reuses rendering guards, context hashing, and persistence philosophy.
Does not recompute quantitative research — reads only from existing semantic contexts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.orchestration.api.schemas import ComparativeReview, LLMContext
from src.orchestration.llm.llm_interface import call_llm
from src.orchestration.llm.prompt_templates import COMPARATIVE_REVIEW, load_template
from src.orchestration.llm.review_engine import _assert_no_unresolved_tokens
from src.orchestration.llm.review_schema import PROVIDER_ANTHROPIC
from src.orchestration.utils.filesystem import (
    comparative_review_json_path,
    comparative_review_md_path,
)
from src.orchestration.utils.serialization import dump_json

logger = logging.getLogger(__name__)

COMPARISON_VERSION = "1.0"

_LIST_SECTIONS = {
    "validation_changes",
    "instability_changes",
    "feature_behavior_changes",
    "robustness_changes",
    "failure_mode_changes",
    "key_tradeoffs",
}


def run_comparative_review(
    baseline_ctx: LLMContext,
    candidate_ctx: LLMContext,
    provider: str = PROVIDER_ANTHROPIC,
    model: str | None = None,
    template: str = COMPARATIVE_REVIEW,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    persist: bool = True,
    comparisons_base: Path | str | None = None,
    base_url: str | None = None,
) -> ComparativeReview:
    """Generate a structured comparative review from two pre-built LLMContexts.

    Args:
        baseline_ctx:     LLMContext for the baseline experiment.
        candidate_ctx:    LLMContext for the candidate experiment.
        provider:         LLM provider identifier.
        model:            Provider model ID override.
        template:         Prompt template name.
        max_tokens:       Completion token budget.
        temperature:      Sampling temperature.
        persist:          If True, write JSON + MD artefacts to results/comparisons/.
        comparisons_base: Override for the comparisons output directory.
        base_url:         Optional base URL for OpenAI-compatible local endpoints.

    Returns:
        ComparativeReview with parsed sections and full provenance.
    """
    comparative_payload = _build_comparative_payload(baseline_ctx, candidate_ctx)
    context_hash = _compute_comparison_hash(baseline_ctx, candidate_ctx)
    prompt = _render_comparative_prompt(
        baseline_ctx, candidate_ctx, comparative_payload, template
    )

    response = call_llm(
        prompt=prompt,
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        base_url=base_url,
    )

    review = _parse_comparative_review(
        text=response.text,
        baseline_experiment=baseline_ctx.experiment_name,
        candidate_experiment=candidate_ctx.experiment_name,
        context_hash=context_hash,
        provider=response.provider,
        model=response.model,
        template=template,
    )

    if persist:
        _persist_comparative_review(review, response.text, comparisons_base)

    return review


# ---------------------------------------------------------------------------
# Comparative payload construction
# ---------------------------------------------------------------------------


def _build_comparative_payload(
    baseline: LLMContext,
    candidate: LLMContext,
) -> dict[str, Any]:
    """Build pre-computed delta payload from two semantic contexts.

    Reads only from existing context summaries — no quantitative recomputation.
    """
    return {
        "metric_comparison": _compare_performance(baseline.performance, candidate.performance),
        "failure_mode_comparison": _compare_failure_modes(
            baseline.failure_modes, candidate.failure_modes
        ),
        "ml_comparison": _compare_ml(baseline.ml_diagnostics, candidate.ml_diagnostics),
        "feature_comparison": _compare_features(baseline.ml_diagnostics, candidate.ml_diagnostics),
        "validation_comparison": _compare_validation(baseline.validation, candidate.validation),
        "universe_comparison": _compare_universe(baseline.universe_summary, candidate.universe_summary),
    }


def _compare_performance(
    base_perf: dict[str, Any],
    cand_perf: dict[str, Any],
) -> dict[str, Any]:
    # max_drawdown is stored as max_drawdown_pct (string) in the performance summary
    scalar_keys = ["sharpe_ratio", "calmar_ratio", "return_to_vol"]
    result: dict[str, Any] = {}
    for key in scalar_keys:
        bv = base_perf.get(key)
        cv = cand_perf.get(key)
        result[key] = {
            "baseline": bv,
            "candidate": cv,
            "delta": _delta(bv, cv),
        }
    # Pass-through tier/pct fields for direct LLM reading
    for key in ("sharpe_tier", "drawdown_severity", "annualized_return_pct",
                "annualized_volatility_pct", "max_drawdown_pct", "hit_rate_pct"):
        bv = base_perf.get(key)
        cv = cand_perf.get(key)
        if bv is not None or cv is not None:
            result[key] = {"baseline": bv, "candidate": cv}
    return result


def _compare_validation(
    base_val: dict[str, Any],
    cand_val: dict[str, Any],
) -> dict[str, Any]:
    scalar_keys = ["mean_oos_sharpe", "std_oos_sharpe", "n_negative_sharpe_splits", "n_splits"]
    result: dict[str, Any] = {}
    for key in scalar_keys:
        bv = base_val.get(key)
        cv = cand_val.get(key)
        result[key] = {
            "baseline": bv,
            "candidate": cv,
            "delta": _delta(bv, cv),
        }
    for key in ("hit_rate_positive_sharpe_pct", "worst_split_drawdown_pct",
                "mean_oos_return_pct", "consistency_tier"):
        bv = base_val.get(key)
        cv = cand_val.get(key)
        if bv is not None or cv is not None:
            result[key] = {"baseline": bv, "candidate": cv}
    return result


def _compare_failure_modes(
    base_fms: list[dict[str, Any]],
    cand_fms: list[dict[str, Any]],
) -> dict[str, Any]:
    base_names = {fm["name"] for fm in base_fms}
    cand_names = {fm["name"] for fm in cand_fms}
    return {
        "baseline_only": sorted(base_names - cand_names),
        "candidate_only": sorted(cand_names - base_names),
        "shared": sorted(base_names & cand_names),
        "baseline_count": len(base_names),
        "candidate_count": len(cand_names),
    }


def _compare_ml(
    base_ml: dict[str, Any],
    cand_ml: dict[str, Any],
) -> dict[str, Any]:
    if not base_ml.get("available") and not cand_ml.get("available"):
        return {"available": False}

    result: dict[str, Any] = {"available": True}

    # IC
    base_ic = base_ml.get("ic", {})
    cand_ic = cand_ml.get("ic", {})
    bv, cv = base_ic.get("mean_ic"), cand_ic.get("mean_ic")
    result["mean_ic"] = {"baseline": bv, "candidate": cv, "delta": _delta(bv, cv)}
    result["ic_tier"] = {"baseline": base_ic.get("ic_tier"), "candidate": cand_ic.get("ic_tier")}

    # Directional accuracy
    base_da = base_ml.get("directional_accuracy", {})
    cand_da = cand_ml.get("directional_accuracy", {})
    result["da_tier"] = {"baseline": base_da.get("tier"), "candidate": cand_da.get("tier")}
    result["da_pct"] = {"baseline": base_da.get("value_pct"), "candidate": cand_da.get("value_pct")}

    # Coefficient stability
    base_cs = base_ml.get("coefficient_stability", {})
    cand_cs = cand_ml.get("coefficient_stability", {})
    for key in ("n_features", "n_stable_features", "n_sign_reversal_features"):
        bv = base_cs.get(key)
        cv = cand_cs.get(key)
        result[key] = {"baseline": bv, "candidate": cv, "delta": _delta(bv, cv)}

    return result


def _compare_features(
    base_ml: dict[str, Any],
    cand_ml: dict[str, Any],
) -> dict[str, Any]:
    if not base_ml.get("available") and not cand_ml.get("available"):
        return {"available": False}

    base_fc = base_ml.get("feature_contributions", {})
    cand_fc = cand_ml.get("feature_contributions", {})

    bv = base_fc.get("mean_hhi")
    cv = cand_fc.get("mean_hhi")
    bnt = base_fc.get("n_family_transitions")
    cnt = cand_fc.get("n_family_transitions")

    return {
        "dominant_family": {
            "baseline": base_fc.get("dominant_family"),
            "candidate": cand_fc.get("dominant_family"),
        },
        "mean_hhi": {"baseline": bv, "candidate": cv, "delta": _delta(bv, cv)},
        "concentration_tier": {
            "baseline": base_fc.get("concentration_tier"),
            "candidate": cand_fc.get("concentration_tier"),
        },
        "n_family_transitions": {
            "baseline": bnt,
            "candidate": cnt,
            "delta": _delta(bnt, cnt),
        },
        "most_volatile_feature": {
            "baseline": base_fc.get("most_volatile_feature"),
            "candidate": cand_fc.get("most_volatile_feature"),
        },
    }


def _compare_universe(
    base_univ: dict[str, Any],
    cand_univ: dict[str, Any],
) -> dict[str, Any]:
    return {
        "n_assets": {
            "baseline": base_univ.get("n_assets"),
            "candidate": cand_univ.get("n_assets"),
        },
        "baseline_tickers": base_univ.get("asset_tickers", []),
        "candidate_tickers": cand_univ.get("asset_tickers", []),
        "coverage": {
            "baseline": base_univ.get("mean_coverage_pct"),
            "candidate": cand_univ.get("mean_coverage_pct"),
        },
    }


def _delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    try:
        return round(float(b) - float(a), 4)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_comparative_prompt(
    baseline: LLMContext,
    candidate: LLMContext,
    comparative_payload: dict[str, Any],
    template_name: str,
) -> str:
    """Render the Jinja2 comparative prompt. Raises on unresolved tokens."""
    from jinja2 import Environment, StrictUndefined

    raw_template = load_template(template_name)

    ctx_dict = {
        "baseline_experiment": baseline.experiment_name,
        "candidate_experiment": candidate.experiment_name,
        "generated_at": datetime.now(UTC).isoformat(),
        "baseline": {
            "performance": baseline.performance,
            "validation": baseline.validation,
            "ml_diagnostics": baseline.ml_diagnostics,
            "failure_modes": baseline.failure_modes,
            "universe_summary": baseline.universe_summary,
            "feature_summary": baseline.feature_summary,
        },
        "candidate": {
            "performance": candidate.performance,
            "validation": candidate.validation,
            "ml_diagnostics": candidate.ml_diagnostics,
            "failure_modes": candidate.failure_modes,
            "universe_summary": candidate.universe_summary,
            "feature_summary": candidate.feature_summary,
        },
        **comparative_payload,
    }

    env = Environment(undefined=StrictUndefined)
    env.filters["tojson"] = lambda v, indent=None: json.dumps(v, indent=indent, default=str)
    tmpl = env.from_string(raw_template)
    rendered = tmpl.render(**ctx_dict)

    _assert_no_unresolved_tokens(rendered, template_name)
    return rendered


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_comparative_review(
    text: str,
    baseline_experiment: str,
    candidate_experiment: str,
    context_hash: str,
    provider: str,
    model: str,
    template: str,
) -> ComparativeReview:
    sections = _split_sections(text)

    def _text(key: str) -> str:
        return sections.get(key, "").strip()

    def _bullets(key: str) -> list[str]:
        raw = sections.get(key, "")
        return [
            line.lstrip("-").strip()
            for line in raw.splitlines()
            if line.strip().startswith("-") and line.lstrip("-").strip()
        ]

    return ComparativeReview(
        baseline_experiment=baseline_experiment,
        candidate_experiment=candidate_experiment,
        generated_at=datetime.now(UTC).isoformat(),
        context_hash=context_hash,
        overall_assessment=_text("overall_assessment"),
        validation_changes=_bullets("validation_changes"),
        instability_changes=_bullets("instability_changes"),
        feature_behavior_changes=_bullets("feature_behavior_changes"),
        robustness_changes=_bullets("robustness_changes"),
        failure_mode_changes=_bullets("failure_mode_changes"),
        key_tradeoffs=_bullets("key_tradeoffs"),
        research_progression_summary=_text("research_progression_summary"),
        confidence=_text("confidence"),
        provider=provider,
        model=model,
        prompt_template=template,
    )


def _split_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current_key: str | None = None
    buffer: list[str] = []
    for line in text.split("\n"):
        if line.startswith("###"):
            if current_key is not None:
                sections[current_key] = "\n".join(buffer).strip()
            heading = line.lstrip("#").strip()
            current_key = re.sub(r"[^a-z0-9]+", "_", heading.lower()).strip("_")
            buffer = []
        else:
            if current_key is not None:
                buffer.append(line)
    if current_key is not None:
        sections[current_key] = "\n".join(buffer).strip()
    return sections


# ---------------------------------------------------------------------------
# Provenance hashing
# ---------------------------------------------------------------------------


def _compute_comparison_hash(baseline: LLMContext, candidate: LLMContext) -> str:
    """SHA256 of both serialised contexts concatenated — deterministic provenance."""
    from src.orchestration.context.context_builder import _context_to_dict

    combined = {
        "baseline": _context_to_dict(baseline),
        "candidate": _context_to_dict(candidate),
    }
    raw = json.dumps(combined, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _persist_comparative_review(
    review: ComparativeReview,
    raw_text: str,
    comparisons_base: Path | str | None,
) -> None:
    json_path = comparative_review_json_path(
        review.baseline_experiment, review.candidate_experiment, comparisons_base
    )
    md_path = comparative_review_md_path(
        review.baseline_experiment, review.candidate_experiment, comparisons_base
    )

    payload = {
        "baseline_experiment": review.baseline_experiment,
        "candidate_experiment": review.candidate_experiment,
        "generated_at": review.generated_at,
        "context_hash": review.context_hash,
        "provider": review.provider,
        "model": review.model,
        "prompt_template": review.prompt_template,
        "comparison_version": COMPARISON_VERSION,
        "overall_assessment": review.overall_assessment,
        "validation_changes": review.validation_changes,
        "instability_changes": review.instability_changes,
        "feature_behavior_changes": review.feature_behavior_changes,
        "robustness_changes": review.robustness_changes,
        "failure_mode_changes": review.failure_mode_changes,
        "key_tradeoffs": review.key_tradeoffs,
        "research_progression_summary": review.research_progression_summary,
        "confidence": review.confidence,
    }
    dump_json(payload, json_path)
    logger.info("Comparative review (JSON) persisted to %s", json_path)

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(raw_text, encoding="utf-8")
    logger.info("Comparative review (MD) persisted to %s", md_path)
