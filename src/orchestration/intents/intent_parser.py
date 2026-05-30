"""Natural-language intent parser for the orchestration layer.

Parsing strategy (two-stage):

  Stage 1 — Rule-based:
    Keyword regex patterns classify the text into one of the 8 known intents.
    Experiment names are extracted by matching tokens against known_experiments.
    This path is deterministic, zero-latency, and requires no API key.

  Stage 2 — LLM fallback:
    When rule-based classification is ambiguous or yields UnrecognisedIntent,
    a structured JSON prompt is sent to the configured LLM provider.
    The LLM returns an intent name + extracted parameters as JSON.
    The result is parsed into the appropriate dataclass.

The function ``parse`` is the sole public entry point.  It returns exactly one
``Intent`` instance; callers should dispatch on the concrete type.
"""

from __future__ import annotations

import json
import re
from typing import Sequence

from src.orchestration.intents.intent_schema import (
    BuildContextIntent,
    BuildEvolutionChainIntent,
    CompareExperimentsIntent,
    GenerateDraftIntent,
    GenerateIterationIntent,
    Intent,
    ListExperimentsIntent,
    RankExperimentsIntent,
    RetrieveArtefactIntent,
    ReviewExperimentIntent,
    UnrecognisedIntent,
)
from src.orchestration.llm.llm_interface import call_llm
from src.orchestration.llm.review_schema import PROVIDER_ANTHROPIC, PROVIDER_STUB


# ---------------------------------------------------------------------------
# Keyword pattern tables
# ---------------------------------------------------------------------------

_REVIEW_PATTERNS = re.compile(
    r"\b(review|analyze|analyse|interpret|assess|evaluate|examine)\b",
    re.IGNORECASE,
)
_COMPARE_PATTERNS = re.compile(
    r"\b(compare|versus|against|diff|difference between)\b|\bvs\.?\b",
    re.IGNORECASE,
)
_ITERATE_PATTERNS = re.compile(
    r"\b(iterate|iteration|proposal|improve|improvement|suggest|next experiment|next version)\b",
    re.IGNORECASE,
)
_EVOLUTION_PATTERNS = re.compile(
    r"\b(evolution|chain|lineage|history|trace|ancestry)\b",
    re.IGNORECASE,
)
_LIST_PATTERNS = re.compile(
    r"\b(list|show|what experiments|all experiments|available experiments|find experiments)\b",
    re.IGNORECASE,
)
_RANK_PATTERNS = re.compile(
    r"\b(rank|ranking|best|top|sort|sorted|sharpe|performance order)\b",
    re.IGNORECASE,
)
_ARTEFACT_PATTERNS = re.compile(
    r"\b(retrieve|artefact|artifact|fetch)\b",
    re.IGNORECASE,
)
_CONTEXT_PATTERNS = re.compile(
    r"\b(context|build context|llm context|prepare context|assemble context)\b",
    re.IGNORECASE,
)
_DRAFT_PATTERNS = re.compile(
    r"\b(draft|generate draft|synthesize config|synthesise config|config draft|create config)\b",
    re.IGNORECASE,
)

# Keys that commonly appear as artefact identifiers in retrieve requests
_ARTEFACT_KEY_TOKENS = re.compile(
    r"\b(metrics|diagnostics|metadata|timeseries|time.series|plots|signals|weights|returns)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Experiment name extraction
# ---------------------------------------------------------------------------


def _find_experiments_in_text(
    text: str,
    known_experiments: Sequence[str],
) -> list[str]:
    """Return experiment names from known_experiments that appear in text.

    Results are ordered by first occurrence position in text (left to right).
    Length descending is used as a tiebreaker so more specific names win when
    two experiments start at the same offset.
    """
    text_lower = text.lower()
    found = []
    for e in known_experiments:
        idx = text_lower.find(e.lower())
        if idx >= 0:
            found.append((idx, -len(e), e))
    found.sort()
    return [e for _, _, e in found]


def _extract_artefact_key(text: str) -> str:
    """Heuristically extract an artefact key from the text."""
    m = _ARTEFACT_KEY_TOKENS.search(text)
    if m:
        return m.group(0).lower().replace("-", "_").replace(" ", "_")
    # Last resort: first noun-like token after "artefact" or "artifact"
    m2 = re.search(r"\b(?:artefact|artifact)\s+(\w+)", text, re.IGNORECASE)
    if m2:
        return m2.group(1).lower()
    return "metrics"


# ---------------------------------------------------------------------------
# Rule-based classifier
# ---------------------------------------------------------------------------


def _rule_based_parse(
    text: str,
    known_experiments: Sequence[str],
) -> Intent | None:
    """Return an Intent or None if classification is ambiguous/impossible."""
    exps = _find_experiments_in_text(text, known_experiments)

    has_compare = bool(_COMPARE_PATTERNS.search(text))
    has_review = bool(_REVIEW_PATTERNS.search(text))
    has_iterate = bool(_ITERATE_PATTERNS.search(text))
    has_evolution = bool(_EVOLUTION_PATTERNS.search(text))
    has_list = bool(_LIST_PATTERNS.search(text))
    has_rank = bool(_RANK_PATTERNS.search(text))
    has_artefact = bool(_ARTEFACT_PATTERNS.search(text))
    has_context = bool(_CONTEXT_PATTERNS.search(text))
    has_draft = bool(_DRAFT_PATTERNS.search(text))

    # --- CompareExperimentsIntent: requires two experiment names ---
    if has_compare and len(exps) >= 2:
        return CompareExperimentsIntent(baseline=exps[0], candidate=exps[1])

    # --- BuildEvolutionChainIntent ---
    if has_evolution and exps:
        return BuildEvolutionChainIntent(root_experiment=exps[0])

    # --- GenerateIterationIntent ---
    if has_iterate and exps:
        return GenerateIterationIntent(experiment_name=exps[0])

    # --- BuildContextIntent ---
    if has_context and exps:
        return BuildContextIntent(experiment_name=exps[0])

    # --- GenerateDraftIntent ---
    if has_draft and exps:
        return GenerateDraftIntent(experiment_name=exps[0])

    # --- RetrieveArtefactIntent ---
    if has_artefact and exps:
        key = _extract_artefact_key(text)
        return RetrieveArtefactIntent(experiment_name=exps[0], key=key)

    # --- ReviewExperimentIntent ---
    if has_review and exps:
        return ReviewExperimentIntent(experiment_name=exps[0])

    # --- RankExperimentsIntent ---
    if has_rank:
        asc = bool(re.search(r"\b(ascending|worst|lowest|bottom)\b", text, re.IGNORECASE))
        return RankExperimentsIntent(descending=not asc)

    # --- ListExperimentsIntent ---
    if has_list:
        # Optional: detect tag/strategy filter
        tag_m = re.search(r"\btag[:\s]+([^\s,]+)", text, re.IGNORECASE)
        strat_m = re.search(r"\bstrategy[:\s]+([^\s,]+)", text, re.IGNORECASE)
        return ListExperimentsIntent(
            tag=tag_m.group(1) if tag_m else None,
            strategy_pattern=strat_m.group(1) if strat_m else None,
        )

    # If text names an experiment but nothing else matches, default to review
    if exps and not any([has_compare, has_evolution, has_iterate, has_rank, has_list, has_draft]):
        return ReviewExperimentIntent(experiment_name=exps[0])

    return None  # ambiguous — hand off to LLM


# ---------------------------------------------------------------------------
# LLM-fallback classifier
# ---------------------------------------------------------------------------

_LLM_SYSTEM = """\
You are a research orchestration classifier. Given a natural-language request,
classify it into exactly one of these intents and extract the required parameters.

Intents:
  ReviewExperimentIntent     — LLM review of one experiment
  CompareExperimentsIntent   — compare two experiments (need baseline + candidate)
  GenerateIterationIntent    — generate iteration proposal for one experiment
  BuildEvolutionChainIntent  — build evolution chain rooted at one experiment
  ListExperimentsIntent      — list/find experiments (optional: tag, strategy_pattern)
  RankExperimentsIntent      — rank experiments by Sharpe (optional: descending bool)
  RetrieveArtefactIntent     — retrieve named artefact (need experiment_name + key)
  BuildContextIntent         — build LLM context for one experiment
  GenerateDraftIntent        — generate config draft from iteration proposal for one experiment
  UnrecognisedIntent         — cannot be mapped to any of the above

Respond with valid JSON only — no prose, no markdown fences. Schema:
{
  "intent": "<IntentTypeName>",
  "params": { <key>: <value>, ... }
}

For UnrecognisedIntent use params: {"reason": "<brief reason>"}.
"""


def _llm_fallback_parse(
    text: str,
    known_experiments: Sequence[str],
    provider: str,
    model: str | None,
) -> Intent:
    exp_list = "\n".join(f"  - {e}" for e in known_experiments) or "  (none available)"
    prompt = (
        f"Known experiments:\n{exp_list}\n\n"
        f"User request: {text}"
    )
    try:
        resp = call_llm(prompt, provider=provider, model=model, system=_LLM_SYSTEM)
        raw = resp.text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        return _build_intent_from_llm_response(data, text)
    except Exception as exc:
        return UnrecognisedIntent(raw_text=text, reason=f"LLM fallback error: {exc}")


def _build_intent_from_llm_response(data: dict, raw_text: str) -> Intent:
    intent_name = data.get("intent", "")
    params = data.get("params", {}) or {}

    try:
        if intent_name == "ReviewExperimentIntent":
            return ReviewExperimentIntent(
                experiment_name=params["experiment_name"],
                provider=params.get("provider", PROVIDER_ANTHROPIC),
                model=params.get("model"),
            )
        if intent_name == "CompareExperimentsIntent":
            return CompareExperimentsIntent(
                baseline=params["baseline"],
                candidate=params["candidate"],
                provider=params.get("provider", PROVIDER_ANTHROPIC),
                model=params.get("model"),
            )
        if intent_name == "GenerateIterationIntent":
            return GenerateIterationIntent(
                experiment_name=params["experiment_name"],
                provider=params.get("provider", PROVIDER_ANTHROPIC),
                model=params.get("model"),
            )
        if intent_name == "BuildEvolutionChainIntent":
            return BuildEvolutionChainIntent(root_experiment=params["root_experiment"])
        if intent_name == "ListExperimentsIntent":
            return ListExperimentsIntent(
                tag=params.get("tag"),
                strategy_pattern=params.get("strategy_pattern"),
            )
        if intent_name == "RankExperimentsIntent":
            return RankExperimentsIntent(descending=bool(params.get("descending", True)))
        if intent_name == "RetrieveArtefactIntent":
            return RetrieveArtefactIntent(
                experiment_name=params["experiment_name"],
                key=params["key"],
            )
        if intent_name == "BuildContextIntent":
            return BuildContextIntent(experiment_name=params["experiment_name"])
        if intent_name == "GenerateDraftIntent":
            return GenerateDraftIntent(
                experiment_name=params["experiment_name"],
                provider=params.get("provider", PROVIDER_ANTHROPIC),
                model=params.get("model"),
            )
    except (KeyError, TypeError) as exc:
        return UnrecognisedIntent(
            raw_text=raw_text,
            reason=f"LLM returned {intent_name} but params were incomplete: {exc}",
        )

    reason = params.get("reason", "") if intent_name == "UnrecognisedIntent" else f"unknown intent: {intent_name}"
    return UnrecognisedIntent(raw_text=raw_text, reason=reason)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parse(
    text: str,
    known_experiments: Sequence[str] | None = None,
    provider: str = PROVIDER_STUB,
    model: str | None = None,
) -> Intent:
    """Parse a natural-language research request into a typed Intent.

    Args:
        text: The raw user input.
        known_experiments: Experiment names available on disk.  When provided,
            experiment mentions are matched against this list.  Pass an empty
            list to disable name extraction (useful in unit tests).
        provider: LLM provider for fallback classification.  Defaults to
            ``"stub"`` so tests never make external API calls.
        model: Optional model override for the LLM fallback.

    Returns:
        A frozen Intent dataclass.  The concrete type determines which
        research API function the router will invoke.
    """
    if not text or not text.strip():
        return UnrecognisedIntent(raw_text=text, reason="empty input")

    exps: Sequence[str] = known_experiments if known_experiments is not None else []

    # Stage 1: rule-based
    intent = _rule_based_parse(text, exps)
    if intent is not None:
        return intent

    # Stage 2: LLM fallback (only when caller explicitly requests a live provider)
    if provider == PROVIDER_STUB:
        return UnrecognisedIntent(
            raw_text=text,
            reason="rule-based parse failed; LLM fallback skipped (stub provider)",
        )

    return _llm_fallback_parse(text, exps, provider, model)
