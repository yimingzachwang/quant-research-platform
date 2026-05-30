"""Review engine: loads context, renders prompt, calls LLM, persists output.

This is the single integration point that connects the structured context
builder to the LLM interface and produces a persisted LLMReviewOutput.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from src.orchestration.api.schemas import LLMContext, LLMReviewOutput
from src.orchestration.llm.llm_interface import call_llm
from src.orchestration.llm.prompt_templates import EXPERIMENT_REVIEW, load_template
from src.orchestration.llm.review_schema import (
    PROVIDER_ANTHROPIC,
    REVIEW_VERSION,
)
from src.orchestration.utils.filesystem import llm_review_path
from src.orchestration.utils.serialization import dump_json

logger = logging.getLogger(__name__)


def run_review(
    context: LLMContext,
    provider: str = PROVIDER_ANTHROPIC,
    model: str | None = None,
    template: str = EXPERIMENT_REVIEW,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    persist: bool = True,
    llm_base: Path | str | None = None,
    base_url: str | None = None,
) -> LLMReviewOutput:
    """Run one LLM review pass against a pre-built LLMContext.

    Args:
        context:     LLMContext from context_builder.build_context().
        provider:    "anthropic", "openai", or "stub".
        model:       Provider model ID override.
        template:    Prompt template name (default: experiment_review).
        max_tokens:  Completion token budget.
        temperature: Sampling temperature.
        persist:     If True, write llm_review.json to results/llm_reviews/.
        llm_base:    Override for llm_review output directory.
        base_url:    Optional base URL for OpenAI-compatible local endpoints.

    Returns:
        LLMReviewOutput with review text and extracted sections.
    """
    prompt = _render_prompt(context, template)
    context_hash = _compute_context_hash(context)
    response = call_llm(
        prompt=prompt,
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        base_url=base_url,
    )

    sections = _extract_sections(response.text)
    flags = _extract_flags(context)

    output = LLMReviewOutput(
        experiment_name=context.experiment_name,
        provider=response.provider,
        model=response.model,
        prompt_template=template,
        review_text=response.text,
        sections=sections,
        flags=flags,
        generated_at=datetime.now(UTC).isoformat(),
        token_usage=response.usage,
    )

    if persist:
        _persist_review(output, llm_base, context_hash=context_hash)

    return output


_UNRESOLVED_MARKERS = ("{{", "}}", "{%", "%}")


def _render_prompt(context: LLMContext, template_name: str) -> str:
    """Render the Jinja2 prompt template with context data.

    Raises RuntimeError if Jinja2 is unavailable or if any template tokens
    remain unresolved after rendering — unresolved syntax must never reach
    the inference layer.
    """
    from jinja2 import Environment, StrictUndefined  # hard dependency — no fallback

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
        "report_sections": context.report_sections,
        **{f"SECTION_{k.upper()}": v for k, v in _section_constants().items()},
    }

    env = Environment(undefined=StrictUndefined)
    env.filters["tojson"] = lambda v, indent=None: json.dumps(v, indent=indent, default=str)
    tmpl = env.from_string(raw_template)
    rendered = tmpl.render(**ctx_dict)

    _assert_no_unresolved_tokens(rendered, template_name)
    return rendered


def _assert_no_unresolved_tokens(rendered: str, template_name: str) -> None:
    """Raise RuntimeError if any Jinja2 template markers survive rendering."""
    found = [m for m in _UNRESOLVED_MARKERS if m in rendered]
    if found:
        raise RuntimeError(
            f"Prompt template '{template_name}' contains unresolved template tokens after "
            f"rendering: {found}. Aborting inference — broken semantic context must never "
            f"reach the model."
        )


def _extract_sections(text: str) -> dict[str, str]:
    """Parse markdown sections from review text into a keyed dict."""
    sections: dict[str, str] = {}

    lines = text.split("\n")
    current_key: str | None = None
    buffer: list[str] = []

    for line in lines:
        if line.startswith("###"):
            if current_key and buffer:
                sections[current_key] = "\n".join(buffer).strip()
            heading = line.lstrip("#").strip().lower().replace(" ", "_")
            current_key = heading
            buffer = []
        else:
            if current_key:
                buffer.append(line)

    if current_key and buffer:
        sections[current_key] = "\n".join(buffer).strip()

    return sections


def _extract_flags(context: LLMContext) -> list[str]:
    """Derive a list of short flag strings from detected failure modes."""
    return [
        f"{fm['severity'].upper()}: {fm['name']}"
        for fm in context.failure_modes
        if fm.get("severity") in ("critical", "warning")
    ]


def _compute_context_hash(context: LLMContext) -> str:
    """Return SHA256 hex of the deterministically serialized LLMContext."""
    from src.orchestration.context.context_builder import _context_to_dict

    ctx_dict = _context_to_dict(context)
    ctx_bytes = json.dumps(ctx_dict, sort_keys=True, default=str).encode()
    return hashlib.sha256(ctx_bytes).hexdigest()


def _persist_review(
    output: LLMReviewOutput,
    llm_base: Path | str | None,
    context_hash: str | None = None,
) -> None:
    path = llm_review_path(output.experiment_name, llm_base)
    payload = {
        "experiment_name": output.experiment_name,
        "provider": output.provider,
        "model": output.model,
        "prompt_template": output.prompt_template,
        "generated_at": output.generated_at,
        "token_usage": output.token_usage,
        "flags": output.flags,
        "sections": output.sections,
        "review_text": output.review_text,
        "review_version": REVIEW_VERSION,
    }
    if context_hash is not None:
        payload["context_hash"] = context_hash
    dump_json(payload, path)
    logger.info("LLM review persisted to %s", path)


def _section_constants() -> dict[str, str]:
    from src.orchestration.llm.review_schema import (
        SECTION_FAILURE_MODES,
        SECTION_FEATURE_CONTRIBUTION,
        SECTION_PERFORMANCE,
        SECTION_RECOMMENDATIONS,
        SECTION_SIGNAL_QUALITY,
        SECTION_VALIDATION,
    )
    return {
        "performance": SECTION_PERFORMANCE,
        "signal_quality": SECTION_SIGNAL_QUALITY,
        "validation": SECTION_VALIDATION,
        "failure_modes": SECTION_FAILURE_MODES,
        "feature_contribution": SECTION_FEATURE_CONTRIBUTION,
        "recommendations": SECTION_RECOMMENDATIONS,
    }
