"""Research evolution chain builder.

Resolves experiment lineage, assembles chronological chains, derives evolution
steps from existing diagnostic artefacts, and persists human-readable summaries.

Design constraints:
- Reads only from persisted artefacts (lineage.json, comparative_review.json,
  LLMContext semantic summaries).
- Does not execute experiments, recompute metrics, or call the quant engine.
- Lineage is human-authored and optional — chains stop at any missing link.
- Evolution summaries are deterministically generated from diagnostic deltas.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.orchestration.api.schemas import (
    EvolutionStep,
    ExperimentLineage,
    ResearchEvolutionChain,
)
from src.orchestration.utils.filesystem import (
    comparative_review_json_path,
    evolution_chain_json_path,
    evolution_chain_md_path,
    lineage_path,
    list_experiments,
)
from src.orchestration.utils.serialization import dump_json, load_json

logger = logging.getLogger(__name__)

_MAX_CHAIN_DEPTH = 50  # safety cap against runaway lineage loops


# ---------------------------------------------------------------------------
# Lineage registration
# ---------------------------------------------------------------------------


def register_lineage(
    experiment_name: str,
    parent_experiment: str | None,
    iteration_reason: str | None = None,
    derived_from_iteration: bool = False,
    derived_from_comparison: bool = False,
    context_hash: str = "",
    experiments_base: Path | str | None = None,
) -> ExperimentLineage:
    """Write a lineage.json record for an experiment.

    Human-triggered only — does not start orchestration or modify experiments.
    Overwrites any existing lineage record for this experiment.
    """
    now = datetime.now(UTC).isoformat()

    # Resolve created_at from experiment metadata if available
    created_at = _load_experiment_created_at(experiment_name, experiments_base) or now

    lineage = ExperimentLineage(
        experiment_name=experiment_name,
        parent_experiment=parent_experiment,
        created_at=created_at,
        registered_at=now,
        iteration_reason=iteration_reason,
        derived_from_iteration=derived_from_iteration,
        derived_from_comparison=derived_from_comparison,
        context_hash=context_hash,
    )

    path = lineage_path(experiment_name, experiments_base)
    dump_json(_lineage_to_dict(lineage), path)
    logger.info("Lineage registered for %s → parent=%s", experiment_name, parent_experiment)
    return lineage


def load_lineage(
    experiment_name: str,
    experiments_base: Path | str | None = None,
) -> ExperimentLineage | None:
    """Load lineage.json for an experiment, or None if not registered."""
    raw = load_json(lineage_path(experiment_name, experiments_base))
    if not raw:
        return None
    return ExperimentLineage(
        experiment_name=raw.get("experiment_name", experiment_name),
        parent_experiment=raw.get("parent_experiment"),
        created_at=raw.get("created_at", ""),
        registered_at=raw.get("registered_at", ""),
        iteration_reason=raw.get("iteration_reason"),
        derived_from_iteration=raw.get("derived_from_iteration", False),
        derived_from_comparison=raw.get("derived_from_comparison", False),
        context_hash=raw.get("context_hash", ""),
    )


# ---------------------------------------------------------------------------
# Chain resolution
# ---------------------------------------------------------------------------


def resolve_chain(
    root_experiment: str,
    experiments_base: Path | str | None = None,
) -> list[str]:
    """Resolve the ordered chain of experiments starting from root.

    Follows child→parent links in reverse: scans all experiments for ones
    whose parent_experiment == current node, advancing one step at a time.
    Stops when no child is found or the chain exceeds _MAX_CHAIN_DEPTH.
    """
    all_experiments = list_experiments(experiments_base)
    lineages = _load_all_lineages(all_experiments, experiments_base)

    chain: list[str] = [root_experiment]
    seen: set[str] = {root_experiment}
    current = root_experiment

    for _ in range(_MAX_CHAIN_DEPTH):
        children = [
            exp for exp, lin in lineages.items()
            if lin is not None
            and lin.parent_experiment == current
            and exp not in seen
        ]
        if not children:
            break
        # Deterministic: take lexicographically first child when multiple exist
        next_exp = sorted(children)[0]
        chain.append(next_exp)
        seen.add(next_exp)
        current = next_exp

    return chain


# ---------------------------------------------------------------------------
# Evolution step derivation
# ---------------------------------------------------------------------------


def build_evolution_step(
    curr_name: str,
    prev_name: str | None,
    lineage: ExperimentLineage | None,
    experiments_base: Path | str | None = None,
    comparisons_base: Path | str | None = None,
) -> EvolutionStep:
    """Derive an EvolutionStep for one node in the chain.

    Strategy (in order of preference):
    1. Extract from an existing comparative_review.json for (prev, curr).
    2. Compute diagnostics deltas directly from LLMContext semantic summaries.
    3. Return a minimal step with available metadata only.
    """
    research_direction = (lineage.iteration_reason or "") if lineage else ""

    if prev_name is None:
        # Root node — no prior experiment to compare against
        persistent = _get_failure_names_from_context(curr_name, experiments_base)
        return EvolutionStep(
            experiment_name=curr_name,
            key_improvements=[],
            new_risks=[],
            persistent_failures=persistent,
            validation_changes=[],
            research_direction=research_direction or "Baseline experiment",
        )

    # 1. Try existing comparative review (prev as baseline, curr as candidate)
    comp = _load_comparative_review(prev_name, curr_name, comparisons_base)
    if comp is not None:
        return _step_from_comparative_review(curr_name, comp, research_direction)

    # 2. Fallback: compute deltas from LLMContexts
    return _step_from_contexts(curr_name, prev_name, research_direction, experiments_base)


def _step_from_comparative_review(
    curr_name: str,
    comp: dict[str, Any],
    research_direction: str,
) -> EvolutionStep:
    """Derive EvolutionStep from an existing comparative_review.json payload."""
    fm_changes = comp.get("failure_mode_changes", [])
    improvements = [
        line.lstrip("-").strip()
        for line in fm_changes
        if any(kw in line.lower() for kw in ("resolved", "removed", "lost"))
    ]
    new_risks = [
        line.lstrip("-").strip()
        for line in fm_changes
        if any(kw in line.lower() for kw in ("gained", "new", "introduced"))
    ]
    # Shared failure modes come from robustness_changes that mention "persistent"
    persistent = [
        line.lstrip("-").strip()
        for line in fm_changes
        if "persistent" in line.lower()
    ]

    val_changes = comp.get("validation_changes", [])[:4]
    if not research_direction:
        research_direction = comp.get("research_progression_summary", "")[:200]

    return EvolutionStep(
        experiment_name=curr_name,
        key_improvements=improvements,
        new_risks=new_risks,
        persistent_failures=persistent,
        validation_changes=[v.lstrip("-").strip() for v in val_changes],
        research_direction=research_direction,
    )


def _step_from_contexts(
    curr_name: str,
    prev_name: str,
    research_direction: str,
    experiments_base: Path | str | None,
) -> EvolutionStep:
    """Compute EvolutionStep directly from two LLMContext semantic summaries."""
    from src.orchestration.context.context_builder import build_context
    from src.orchestration.llm.comparison_engine import (
        _compare_failure_modes,
        _compare_features,
        _compare_validation,
    )

    try:
        prev_ctx = build_context(prev_name, experiments_base)
        curr_ctx = build_context(curr_name, experiments_base)
    except Exception as exc:
        logger.warning("Could not load contexts for %s→%s: %s", prev_name, curr_name, exc)
        return EvolutionStep(
            experiment_name=curr_name,
            key_improvements=[],
            new_risks=[],
            persistent_failures=[],
            validation_changes=[],
            research_direction=research_direction,
        )

    fm_diff = _compare_failure_modes(prev_ctx.failure_modes, curr_ctx.failure_modes)
    val_diff = _compare_validation(prev_ctx.validation, curr_ctx.validation)
    feat_diff = _compare_features(prev_ctx.ml_diagnostics, curr_ctx.ml_diagnostics)

    improvements = [
        f"Resolved failure mode: {name}"
        for name in fm_diff.get("baseline_only", [])
    ]
    new_risks = [
        f"New failure mode: {name}"
        for name in fm_diff.get("candidate_only", [])
    ]

    # HHI concentration change — only flag as risk when concentration increases
    hhi_delta = feat_diff.get("mean_hhi", {}).get("delta")
    if hhi_delta is not None and abs(hhi_delta) > 0.05:
        if hhi_delta > 0:
            new_risks.append(
                f"Concentration (mean_hhi) increased by {hhi_delta:.3f}"
            )
        else:
            improvements.append(
                f"Concentration (mean_hhi) decreased by {abs(hhi_delta):.3f}"
            )

    persistent = fm_diff.get("shared", [])

    val_changes: list[str] = []
    oos = val_diff.get("mean_oos_sharpe", {})
    bv, cv, delta = oos.get("baseline"), oos.get("candidate"), oos.get("delta")
    if delta is not None:
        direction = "improved" if delta > 0 else "deteriorated"
        val_changes.append(
            f"mean_oos_sharpe {direction}: {bv} → {cv} (Δ{delta:+.3f})"
        )

    neg = val_diff.get("n_negative_sharpe_splits", {})
    neg_delta = neg.get("delta")
    if neg_delta is not None and neg_delta != 0:
        direction = "decreased" if neg_delta < 0 else "increased"
        val_changes.append(
            f"Negative OOS splits {direction} by {abs(int(neg_delta))}"
            f" ({neg.get('baseline')} → {neg.get('candidate')})"
        )

    tier = val_diff.get("consistency_tier", {})
    if tier.get("baseline") != tier.get("candidate"):
        val_changes.append(
            f"consistency_tier: {tier.get('baseline')} → {tier.get('candidate')}"
        )

    return EvolutionStep(
        experiment_name=curr_name,
        key_improvements=improvements,
        new_risks=new_risks,
        persistent_failures=persistent,
        validation_changes=val_changes,
        research_direction=research_direction,
    )


# ---------------------------------------------------------------------------
# Chain assembly
# ---------------------------------------------------------------------------


def build_evolution_chain(
    root_experiment: str,
    experiments_base: Path | str | None = None,
    comparisons_base: Path | str | None = None,
) -> ResearchEvolutionChain:
    """Build a complete ResearchEvolutionChain from persisted artefacts.

    Resolves the lineage chain, derives an EvolutionStep for each experiment,
    and generates a deterministic evolution summary — no LLM call required.
    """
    chain_names = resolve_chain(root_experiment, experiments_base)
    lineages = {
        name: load_lineage(name, experiments_base)
        for name in chain_names
    }

    steps: list[EvolutionStep] = []
    for i, name in enumerate(chain_names):
        prev = chain_names[i - 1] if i > 0 else None
        step = build_evolution_step(
            curr_name=name,
            prev_name=prev,
            lineage=lineages.get(name),
            experiments_base=experiments_base,
            comparisons_base=comparisons_base,
        )
        steps.append(step)

    summary = _generate_evolution_summary(chain_names, steps)

    return ResearchEvolutionChain(
        root_experiment=root_experiment,
        experiments=chain_names,
        generated_at=datetime.now(UTC).isoformat(),
        evolution_summary=summary,
        steps=steps,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def persist_evolution_chain(
    chain: ResearchEvolutionChain,
    evolution_base: Path | str | None = None,
) -> None:
    """Write evolution_chain.json and evolution_chain.md."""
    json_path = evolution_chain_json_path(chain.root_experiment, evolution_base)
    md_path = evolution_chain_md_path(chain.root_experiment, evolution_base)

    dump_json(_chain_to_dict(chain), json_path)
    logger.info("Evolution chain (JSON) persisted to %s", json_path)

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_chain_to_md(chain), encoding="utf-8")
    logger.info("Evolution chain (MD) persisted to %s", md_path)


# ---------------------------------------------------------------------------
# Summary generation (deterministic — no LLM)
# ---------------------------------------------------------------------------


def _generate_evolution_summary(chain: list[str], steps: list[EvolutionStep]) -> str:
    if not steps:
        return "No experiments in evolution chain."
    if len(steps) == 1:
        failures = ", ".join(steps[0].persistent_failures) or "none identified"
        return (
            f"Single-experiment chain rooted at {steps[0].experiment_name}. "
            f"Persistent failure modes: {failures}."
        )

    parts: list[str] = [
        f"The research chain began with {steps[0].experiment_name} as the baseline."
    ]
    for step in steps[1:]:
        fragments: list[str] = []
        if step.key_improvements:
            fragments.append("; ".join(step.key_improvements))
        if step.new_risks:
            fragments.append("new risks: " + "; ".join(step.new_risks))
        if step.validation_changes:
            fragments.append(step.validation_changes[0])
        body = ", ".join(fragments) if fragments else "no significant diagnostic changes detected"
        direction = f" Research direction: {step.research_direction}." if step.research_direction else ""
        parts.append(f"{step.experiment_name}: {body}.{direction}")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _chain_to_dict(chain: ResearchEvolutionChain) -> dict:
    return {
        "root_experiment": chain.root_experiment,
        "experiments": chain.experiments,
        "generated_at": chain.generated_at,
        "evolution_summary": chain.evolution_summary,
        "steps": [
            {
                "experiment_name": s.experiment_name,
                "key_improvements": s.key_improvements,
                "new_risks": s.new_risks,
                "persistent_failures": s.persistent_failures,
                "validation_changes": s.validation_changes,
                "research_direction": s.research_direction,
            }
            for s in chain.steps
        ],
    }


def _chain_to_md(chain: ResearchEvolutionChain) -> str:
    arrow = " → ".join(chain.experiments)
    lines = [
        f"# Research Evolution Chain: {chain.root_experiment}",
        "",
        f"Generated: {chain.generated_at}",
        f"Chain: {arrow}",
        "",
        "## Evolution Summary",
        "",
        chain.evolution_summary,
        "",
        "## Steps",
        "",
    ]
    for i, step in enumerate(chain.steps):
        label = "(Root)" if i == 0 else f"(Step {i})"
        lines += [
            f"### {step.experiment_name} {label}",
            "",
            f"**Research Direction:** {step.research_direction or 'Not specified'}",
            "",
        ]
        if step.key_improvements:
            lines.append("**Key Improvements:**")
            lines += [f"- {item}" for item in step.key_improvements]
            lines.append("")
        if step.new_risks:
            lines.append("**New Risks:**")
            lines += [f"- {item}" for item in step.new_risks]
            lines.append("")
        if step.persistent_failures:
            lines.append("**Persistent Failures:**")
            lines += [f"- {item}" for item in step.persistent_failures]
            lines.append("")
        if step.validation_changes:
            lines.append("**Validation Changes:**")
            lines += [f"- {item}" for item in step.validation_changes]
            lines.append("")

    return "\n".join(lines)


def _lineage_to_dict(lineage: ExperimentLineage) -> dict:
    return {
        "experiment_name": lineage.experiment_name,
        "parent_experiment": lineage.parent_experiment,
        "created_at": lineage.created_at,
        "registered_at": lineage.registered_at,
        "iteration_reason": lineage.iteration_reason,
        "derived_from_iteration": lineage.derived_from_iteration,
        "derived_from_comparison": lineage.derived_from_comparison,
        "context_hash": lineage.context_hash,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_all_lineages(
    experiments: list[str],
    base: Path | str | None,
) -> dict[str, ExperimentLineage | None]:
    return {name: load_lineage(name, base) for name in experiments}


def _load_comparative_review(
    prev_name: str,
    curr_name: str,
    comparisons_base: Path | str | None,
) -> dict | None:
    path = comparative_review_json_path(prev_name, curr_name, comparisons_base)
    if path.exists():
        return load_json(path)
    return None


def _get_failure_names_from_context(
    experiment_name: str,
    experiments_base: Path | str | None,
) -> list[str]:
    try:
        from src.orchestration.context.context_builder import build_context
        ctx = build_context(experiment_name, experiments_base)
        return [fm["name"] for fm in ctx.failure_modes]
    except Exception:
        return []


def _load_experiment_created_at(
    experiment_name: str,
    base: Path | str | None,
) -> str | None:
    from src.orchestration.utils.filesystem import metadata_path
    raw = load_json(metadata_path(experiment_name, base))
    if raw:
        return raw.get("created_at")
    return None
