"""Draft generator: IterationProposal + base config → ExperimentDraft via LLM.

The LLM receives a structured prompt containing the base config sections and
the iteration proposal text.  It returns JSON only — never YAML, never
executable instructions.  The generator fills current_value from the base
config (not from the LLM) before assembling the draft.

Mirrors iteration_engine.py in structure and style.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.experiments.config_io import load_config
from src.orchestration.api.schemas import IterationProposal
from src.orchestration.config_generation.draft_schema import (
    DraftChange,
    ExperimentDraft,
    compute_draft_hash,
)
from src.orchestration.llm.llm_interface import call_llm
from src.orchestration.llm.review_schema import PROVIDER_ANTHROPIC, PROVIDER_STUB
from src.orchestration.registry.experiment_registry import list_all
from src.orchestration.utils.filesystem import (
    draft_json_path,
    experiment_config_path,
    experiment_root,
    iteration_proposal_json_path,
    report_markdown_path,
)
from src.orchestration.utils.serialization import dump_json, load_json

logger = logging.getLogger(__name__)

# The vocabulary listed in _LLM_SYSTEM must stay in sync with
# _VALID_CHANGE_PATHS in draft_validator.py.  Update both together.
_LLM_SYSTEM = """\
You are a quantitative research configuration synthesizer.

Given an experiment iteration proposal and the current experiment configuration,
extract specific parameter changes as structured JSON.

Rules:
1. Output JSON ONLY — no prose, no YAML, no executable code, no markdown fences.
2. Only use change paths from the valid vocabulary listed below.
3. Proposed values must be typed correctly: int, float, str, or dict for entries.add.
4. Rationale must be concise and grounded in the proposal's supporting evidence.
5. Keep changes minimal — only propose what the proposal explicitly supports.

Valid change paths  (section → field):
  model                  → type, params.alpha, params.C, params.l1_ratio, params.max_iter
  labels                 → type, params.horizon
  signal                 → type, params.n, params.n_long, params.n_short, params.threshold
  validation             → parameters.train_months, parameters.test_months, parameters.gap_days
  execution              → transaction_cost_bps
  portfolio_construction → weighting.scheme, weighting.prediction_normalization, weighting.temperature
  features               → entries.add (value = complete feature dict), entries.remove (value = feature name str)

Output schema:
{
  "proposed_name": "<base_name>_v2",
  "changes": [
    {
      "section": "<section>",
      "field": "<field>",
      "proposed_value": <typed_value>,
      "rationale": "<concise rationale>"
    }
  ]
}
"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_draft(
    experiment_name: str,
    provider: str = PROVIDER_ANTHROPIC,
    model: str | None = None,
    proposal_hash: str | None = None,
    base: Path | str | None = None,
    llm_base: Path | str | None = None,
    configs_base: Path | str | None = None,
    reports_base: Path | str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    base_url: str | None = None,
) -> ExperimentDraft:
    """Generate a typed ExperimentDraft from the most recent IterationProposal.

    Loads the base experiment YAML config and the persisted IterationProposal.
    Calls the LLM for a structured JSON change list.  Fills current_value from
    the base config.  Persists the draft to results/llm_reviews/{name}/.

    The proposed name never collides with an existing config / result / registry
    / report — if the LLM proposes an already-used name the next free ``_vN``
    suffix is chosen, so a new draft cannot silently overwrite prior artefacts.

    Args:
        experiment_name: Source experiment — must have a v2 YAML config and a
                         persisted IterationProposal.
        provider:        LLM provider for structured extraction.
        model:           Optional model override.
        proposal_hash:   If provided, verifies the loaded proposal matches this hash.
        base:            Override for the experiments results directory.
        llm_base:        Override for results/llm_reviews/.
        configs_base:    Override for configs/experiments/.
        reports_base:    Override for the reports directory (proposed-name collision check).
        max_tokens:      Max tokens for LLM response.
        temperature:     Sampling temperature (low = more deterministic).

    Returns:
        ExperimentDraft with approved=False, persisted to disk.

    Raises:
        FileNotFoundError: Base config or IterationProposal not found.
        ValueError: Config version != 2, or malformed LLM JSON.
    """
    # 1. Load and validate base config
    config_path = experiment_config_path(experiment_name, configs_base)
    try:
        base_config = load_config(config_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No YAML config found for {experiment_name!r} at {config_path}. "
            "Draft generation requires a version-2 ML config file."
        ) from None

    if str(base_config.get("version", "1")) != "2":
        raise ValueError(
            f"Draft generation only supported for ML experiments (config version 2). "
            f"Experiment {experiment_name!r} has version {base_config.get('version')!r}."
        )

    # 2. Load IterationProposal
    proposal = _load_proposal(experiment_name, proposal_hash, llm_base)

    # 3. Build prompt and obtain a structured JSON response.
    #    The stub provider returns a fixed placeholder string that is not valid
    #    JSON, so for provider="stub" only we substitute a deterministic,
    #    schema-valid change set (the same one the terminal demo uses:
    #    model.params.alpha -> 1.0).  Real providers go through call_llm.
    prompt = _build_prompt(experiment_name, base_config, proposal)
    if provider == PROVIDER_STUB:
        resp_text = _stub_draft_response(experiment_name)
    else:
        resp = call_llm(
            prompt,
            provider=provider,
            model=model,
            system=_LLM_SYSTEM,
            max_tokens=max_tokens,
            temperature=temperature,
            base_url=base_url,
        )
        resp_text = resp.text

    # 4. Parse structured JSON response
    raw_changes, proposed_name = _parse_llm_response(resp_text, experiment_name)

    # 4b. Never silently collide with existing artefacts.  If the proposed name
    #     already has a config / result / registry / report, choose the next free
    #     '_vN' suffix instead.  This stops the validate -> regenerate loop seen
    #     when an LLM keeps proposing an already-registered name.
    proposed_name = _next_available_proposed_name(
        proposed_name, base=base, configs_base=configs_base, reports_base=reports_base
    )

    # 5. Build DraftChange list — current_value from base config, not from LLM
    changes = _build_changes(base_config, raw_changes)

    # 6. Assemble draft
    draft_id = str(uuid.uuid4())
    draft_hash = compute_draft_hash(experiment_name, proposed_name, changes)

    draft = ExperimentDraft(
        draft_id=draft_id,
        draft_hash=draft_hash,
        base_experiment=experiment_name,
        source_proposal_hash=proposal.context_hash,
        proposed_name=proposed_name,
        changes=changes,
        generated_at=datetime.now(UTC).isoformat(),
    )

    # 7. Persist
    path = draft_json_path(experiment_name, draft_id, llm_base)
    dump_json(draft.to_dict(), path)
    logger.info("Draft persisted to %s", path)

    return draft


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def save_draft(
    draft: ExperimentDraft,
    llm_base: Path | str | None = None,
) -> Path:
    """Persist an ExperimentDraft to its canonical path, overwriting in place.

    Used to persist state transitions (e.g. approval) so that a later, stateless
    load_draft() reflects the updated draft.  Returns the written path.
    """
    path = draft_json_path(draft.base_experiment, draft.draft_id, llm_base)
    dump_json(draft.to_dict(), path)
    return path


def load_draft(
    experiment_name: str,
    draft_id: str,
    llm_base: Path | str | None = None,
) -> ExperimentDraft | None:
    """Load a persisted ExperimentDraft by experiment name and draft ID.

    Returns None if the draft file does not exist.  Never raises.
    """
    path = draft_json_path(experiment_name, draft_id, llm_base)
    data = load_json(path)
    if data is None:
        return None
    changes = [DraftChange(**c) for c in data.get("changes", [])]
    return ExperimentDraft(
        draft_id=data["draft_id"],
        draft_hash=data["draft_hash"],
        base_experiment=data["base_experiment"],
        source_proposal_hash=data["source_proposal_hash"],
        proposed_name=data["proposed_name"],
        changes=changes,
        generated_at=data["generated_at"],
        approved=data.get("approved", False),
        approved_at=data.get("approved_at"),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Matches a trailing version suffix so it can be incremented (e.g. "..._v2").
_VERSION_SUFFIX_RE = re.compile(r"^(?P<stem>.+)_v(?P<num>\d+)$")


def _proposed_name_is_taken(
    name: str,
    base: Path | str | None = None,
    configs_base: Path | str | None = None,
    reports_base: Path | str | None = None,
) -> bool:
    """True if any existing artefact already uses ``name``.

    Checks the rendered config, the result/registry directory, the registry
    listing, and the report markdown.  Used so draft generation never picks a
    name that would later overwrite an existing config, result, or report.
    """
    if experiment_config_path(name, configs_base).exists():
        return True
    if experiment_root(name, base).exists():
        return True
    if name in set(list_all(base)):
        return True
    if report_markdown_path(name, reports_base).exists():
        return True
    return False


def _next_available_proposed_name(
    proposed_name: str,
    base: Path | str | None = None,
    configs_base: Path | str | None = None,
    reports_base: Path | str | None = None,
) -> str:
    """Return ``proposed_name``, or the next free ``_vN`` variant if it is taken.

    Increments the trailing version suffix (``_v2`` -> ``_v3`` -> ``_v4`` ...)
    until a name with no existing config / result / registry / report artefact
    is found, so a new draft never silently overwrites prior artefacts and the
    model never loops re-proposing the same already-registered name.  When the
    name has no ``_vN`` suffix, one is appended (``foo`` -> ``foo_v2``).
    """
    if not _proposed_name_is_taken(proposed_name, base, configs_base, reports_base):
        return proposed_name

    match = _VERSION_SUFFIX_RE.match(proposed_name)
    if match:
        stem = match.group("stem")
        num = int(match.group("num"))
    else:
        stem, num = proposed_name, 1

    # Bounded scan — avoid an unbounded loop if every candidate is somehow taken.
    for candidate_num in range(num + 1, num + 1001):
        candidate = f"{stem}_v{candidate_num}"
        if not _proposed_name_is_taken(candidate, base, configs_base, reports_base):
            return candidate

    raise ValueError(
        f"Could not find an available proposed name near {proposed_name!r}. "
        "Clean existing local demo artefacts or provide a new proposed name."
    )


def _load_proposal(
    experiment_name: str,
    proposal_hash: str | None,
    llm_base: Path | str | None,
) -> IterationProposal:
    path = iteration_proposal_json_path(experiment_name, llm_base)
    data = load_json(path)
    if data is None:
        raise FileNotFoundError(
            f"No IterationProposal found for {experiment_name!r} at {path}. "
            "Run generate_iteration_proposal() first."
        )
    # Persisted proposals carry provenance keys (e.g. iteration_version) that
    # are not IterationProposal fields; keep only recognised dataclass fields.
    known = {f.name for f in dataclasses.fields(IterationProposal)}
    proposal = IterationProposal(**{k: v for k, v in data.items() if k in known})  # type: ignore[arg-type]
    if proposal_hash is not None and proposal.context_hash != proposal_hash:
        raise ValueError(
            f"Proposal hash mismatch for {experiment_name!r}: "
            f"expected {proposal_hash!r}, found {proposal.context_hash!r}."
        )
    return proposal


def _build_prompt(
    experiment_name: str,
    base_config: dict[str, Any],
    proposal: IterationProposal,
) -> str:
    config_summary = {
        "name": experiment_name,
        "model": base_config.get("model", {}),
        "labels": base_config.get("labels", {}),
        "signal": base_config.get("signal", {}),
        "validation": base_config.get("validation", {}),
        "execution": base_config.get("execution", {}),
        "portfolio_construction": base_config.get("portfolio_construction", {}),
        "features": {
            "ticker": base_config.get("features", {}).get("ticker"),
            "entries": [
                {
                    "name": e.get("name"),
                    "type": e.get("type"),
                    "params": e.get("params", {}),
                }
                for e in base_config.get("features", {}).get("entries", [])
            ],
        },
    }
    return "\n".join([
        f"Base experiment: {experiment_name}",
        "",
        "Current configuration:",
        json.dumps(config_summary, indent=2),
        "",
        "Iteration proposal:",
        f"  Research focus: {proposal.research_focus}",
        f"  Rationale: {proposal.rationale}",
        "  Suggested experiments:",
        *[f"    - {s}" for s in proposal.suggested_experiments],
        "  Validation concerns:",
        *[f"    - {s}" for s in proposal.validation_concerns],
        "  Feature risks:",
        *[f"    - {s}" for s in proposal.feature_risks],
        "",
        "Extract the specific parameter changes as JSON using the schema in the system prompt.",
    ])


def _stub_draft_response(experiment_name: str) -> str:
    """Deterministic, schema-valid draft JSON for the stub provider.

    The stub LLM returns a non-JSON placeholder, so this supplies a minimal
    schema-conforming change set (stronger L2 regularisation) so the stub path
    yields a usable draft for demos and tests.  ``current_value`` is still read
    from the base config by the normal flow, not from here.
    """
    return json.dumps(
        {
            "proposed_name": f"{experiment_name}_v2",
            "changes": [
                {
                    "section": "model",
                    "field": "params.alpha",
                    "proposed_value": 1.0,
                    "rationale": (
                        "Increase L2 regularisation to stabilise coefficients and "
                        "improve out-of-sample validation consistency."
                    ),
                }
            ],
        }
    )


def _parse_llm_response(
    text: str,
    experiment_name: str,
) -> tuple[list[dict[str, Any]], str]:
    """Parse LLM JSON. Returns (raw_changes, proposed_name). Raises ValueError on failure."""
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned invalid JSON: {exc}. "
            f"Response (first 300 chars): {text[:300]!r}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"LLM response must be a JSON object, got {type(data).__name__}."
        )
    proposed_name = str(data.get("proposed_name") or f"{experiment_name}_draft")
    raw_changes = data.get("changes", [])
    if not isinstance(raw_changes, list):
        raise ValueError("LLM response 'changes' field must be a list.")
    return raw_changes, proposed_name


def _get_current_value(
    base_config: dict[str, Any],
    section: str,
    field: str,
) -> Any:
    """Navigate into base_config[section] using the dotted field path."""
    if field in ("entries.add", "entries.remove"):
        return None
    target: Any = base_config.get(section, {})
    for part in field.split("."):
        if not isinstance(target, dict):
            return None
        target = target.get(part)
    return target


def _build_changes(
    base_config: dict[str, Any],
    raw_changes: list[dict[str, Any]],
) -> list[DraftChange]:
    changes = []
    for raw in raw_changes:
        if not isinstance(raw, dict):
            continue
        section = str(raw.get("section", ""))
        field = str(raw.get("field", ""))
        if not section or not field:
            continue
        changes.append(DraftChange(
            section=section,
            field=field,
            current_value=_get_current_value(base_config, section, field),
            proposed_value=raw.get("proposed_value"),
            rationale=str(raw.get("rationale", "")),
        ))
    return changes
