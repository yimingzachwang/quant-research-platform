"""Iteration engine: renders iteration prompt, calls LLM, parses and persists proposal.

Mirrors the review engine's architecture — reuses the same rendering guard,
context hash, and persistence philosophy.  Does not duplicate orchestration logic.

Advisory only: the researcher remains the decision-maker.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

from src.orchestration.api.schemas import IterationProposal, LLMContext
from src.orchestration.llm.llm_interface import call_llm
from src.orchestration.llm.prompt_templates import ITERATION_PROPOSAL, load_template
from src.orchestration.llm.review_engine import (
    _assert_no_unresolved_tokens,
    _compute_context_hash,
)
from src.orchestration.llm.review_schema import PROVIDER_ANTHROPIC
from src.orchestration.utils.filesystem import (
    iteration_proposal_json_path,
    iteration_proposal_md_path,
)
from src.orchestration.utils.serialization import dump_json

logger = logging.getLogger(__name__)

ITERATION_VERSION = "1.0"

# Canonical section keys matching the prompt template headers
SECTION_RESEARCH_FOCUS = "research_focus"
SECTION_RATIONALE = "rationale"
SECTION_SUPPORTING_EVIDENCE = "supporting_evidence"
SECTION_SUGGESTED_EXPERIMENTS = "suggested_experiments"
SECTION_INSTABILITY_SIGNALS = "instability_signals"
SECTION_VALIDATION_CONCERNS = "validation_concerns"
SECTION_FEATURE_RISKS = "feature_risks"
SECTION_CONFIDENCE = "confidence"

_LIST_SECTIONS = {
    SECTION_SUPPORTING_EVIDENCE,
    SECTION_SUGGESTED_EXPERIMENTS,
    SECTION_INSTABILITY_SIGNALS,
    SECTION_VALIDATION_CONCERNS,
    SECTION_FEATURE_RISKS,
}


def run_iteration_proposal(
    context: LLMContext,
    provider: str = PROVIDER_ANTHROPIC,
    model: str | None = None,
    template: str = ITERATION_PROPOSAL,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    persist: bool = True,
    llm_base: Path | str | None = None,
    base_url: str | None = None,
) -> IterationProposal:
    """Generate a structured research iteration proposal from a pre-built LLMContext.

    Args:
        context:     LLMContext from context_builder.build_context().
        provider:    LLM provider ("openai", "anthropic", or "stub").
        model:       Provider model ID override.
        template:    Prompt template name (default: iteration_proposal).
        max_tokens:  Completion token budget.
        temperature: Sampling temperature.
        persist:     If True, write iteration_proposal.json and .md to results/llm_reviews/.
        llm_base:    Override for output directory.
        base_url:    Optional base URL for OpenAI-compatible local endpoints.

    Returns:
        IterationProposal with parsed sections and provenance metadata.
    """
    prompt = _render_iteration_prompt(context, template)
    context_hash = _compute_context_hash(context)

    response = call_llm(
        prompt=prompt,
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        base_url=base_url,
    )

    proposal = _parse_iteration_proposal(
        text=response.text,
        experiment_name=context.experiment_name,
        context_hash=context_hash,
        provider=response.provider,
        model=response.model,
        template=template,
    )

    if persist:
        _persist_iteration_proposal(proposal, response.text, llm_base)

    return proposal


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_iteration_prompt(context: LLMContext, template_name: str) -> str:
    """Render the Jinja2 iteration prompt template with context data.

    Raises RuntimeError if Jinja2 is unavailable or unresolved tokens survive.
    """
    from jinja2 import Environment, StrictUndefined

    raw_template = load_template(template_name)

    ctx_dict = {
        "experiment_name": context.experiment_name,
        "strategy_name": context.strategy_name,
        "tags": context.tags,
        "created_at": context.created_at,
        "performance": context.performance,
        "validation": context.validation,
        "ml_diagnostics": context.ml_diagnostics,
        "failure_modes": context.failure_modes,
        "feature_summary": context.feature_summary,
        "universe_summary": context.universe_summary,
        "available_plots": context.available_plots,
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


def _parse_iteration_proposal(
    text: str,
    experiment_name: str,
    context_hash: str,
    provider: str,
    model: str,
    template: str,
) -> IterationProposal:
    """Parse LLM output into a structured IterationProposal.

    Splits on ### headers, normalises section names, extracts bullet items
    for list-valued fields, and raw text for prose fields.
    """
    sections = _split_sections(text)

    def _text(key: str) -> str:
        return sections.get(key, "").strip()

    def _bullets(key: str) -> list[str]:
        raw = sections.get(key, "")
        items = []
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("-"):
                item = line.lstrip("-").strip()
                if item:
                    items.append(item)
        return items

    return IterationProposal(
        experiment_name=experiment_name,
        generated_at=datetime.now(UTC).isoformat(),
        context_hash=context_hash,
        research_focus=_text(SECTION_RESEARCH_FOCUS),
        rationale=_text(SECTION_RATIONALE),
        supporting_evidence=_bullets(SECTION_SUPPORTING_EVIDENCE),
        suggested_experiments=_bullets(SECTION_SUGGESTED_EXPERIMENTS),
        instability_signals=_bullets(SECTION_INSTABILITY_SIGNALS),
        validation_concerns=_bullets(SECTION_VALIDATION_CONCERNS),
        feature_risks=_bullets(SECTION_FEATURE_RISKS),
        confidence=_text(SECTION_CONFIDENCE),
        provider=provider,
        model=model,
        prompt_template=template,
    )


def _split_sections(text: str) -> dict[str, str]:
    """Split LLM output on ### headings into a keyed dict."""
    sections: dict[str, str] = {}
    current_key: str | None = None
    buffer: list[str] = []

    for line in text.split("\n"):
        if line.startswith("###"):
            if current_key is not None:
                sections[current_key] = "\n".join(buffer).strip()
            heading = line.lstrip("#").strip()
            current_key = _normalise_heading(heading)
            buffer = []
        else:
            if current_key is not None:
                buffer.append(line)

    if current_key is not None:
        sections[current_key] = "\n".join(buffer).strip()

    return sections


def _normalise_heading(heading: str) -> str:
    """Convert a markdown heading to snake_case section key."""
    return re.sub(r"[^a-z0-9]+", "_", heading.lower()).strip("_")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _persist_iteration_proposal(
    proposal: IterationProposal,
    raw_text: str,
    llm_base: Path | str | None,
) -> None:
    json_path = iteration_proposal_json_path(proposal.experiment_name, llm_base)
    md_path = iteration_proposal_md_path(proposal.experiment_name, llm_base)

    payload = {
        "experiment_name": proposal.experiment_name,
        "generated_at": proposal.generated_at,
        "context_hash": proposal.context_hash,
        "provider": proposal.provider,
        "model": proposal.model,
        "prompt_template": proposal.prompt_template,
        "iteration_version": ITERATION_VERSION,
        "research_focus": proposal.research_focus,
        "rationale": proposal.rationale,
        "supporting_evidence": proposal.supporting_evidence,
        "suggested_experiments": proposal.suggested_experiments,
        "instability_signals": proposal.instability_signals,
        "validation_concerns": proposal.validation_concerns,
        "feature_risks": proposal.feature_risks,
        "confidence": proposal.confidence,
    }
    dump_json(payload, json_path)
    logger.info("Iteration proposal (JSON) persisted to %s", json_path)

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(raw_text, encoding="utf-8")
    logger.info("Iteration proposal (MD) persisted to %s", md_path)
