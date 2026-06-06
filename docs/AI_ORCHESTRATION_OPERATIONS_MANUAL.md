# AI Orchestration Operations Manual

**Repository:** `quant-research-platform`
**Audit date:** 2026-05-30
**Subject:** How to run the AI orchestration system operationally

---

## 1. Prerequisites

### API Keys

The orchestration system supports three providers. Only one is needed at runtime:

| Provider | Environment variable | Default model |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o` |
| Local (OpenAI-compatible) | `OPENAI_API_KEY` (any value) + `base_url` param | (caller-specified) |
| Stub (testing only) | (none) | `"stub"` |

Set the key in `.env` or export it before running:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

The stub provider makes no network calls and works without any key. It is the default in all tests and is appropriate for verifying prompt rendering without incurring cost.

### Python Dependencies

```bash
pip install anthropic          # for Anthropic provider
pip install openai             # for OpenAI provider
pip install jinja2             # required for all prompt rendering
```

### Artefact Requirements

Most orchestration functions require a pre-existing experiment result tree:

```
results/experiments/<experiment_name>/
    metadata.json
    metrics.json
    diagnostics/
        ml_diagnostics.json
        split_metrics.json
        backtest_diagnostics.json
        universe_coverage.json
    research/
        feature_summary.json
        alignment_diagnostics.json
    plots/
        plot_index.json
```

Run the two canonical experiments to generate these artefacts:

```bash
python scripts/run_from_config.py configs/experiments/canonical_ml_showcase.yaml --report --preset canonical
python scripts/run_from_config.py configs/experiments/canonical_ml_multi_asset.yaml --report --preset canonical
```

---

## 2. Running an LLM Review

### Via research_api (direct)

```python
from src.orchestration.api.research_api import run_llm_review

review = run_llm_review(
    "canonical_ml_showcase",
    provider="anthropic",        # or "openai"
    model=None,                  # None = use provider default
    persist=True,                # write llm_review.json
)

print(review.sections["performance_interpretation"])
print(review.sections["recommendations"])
print(review.flags)              # list of failure mode names
```

Output is written to `results/llm_reviews/canonical_ml_showcase/llm_review.json`.

### Via workflow router (intent-based)

```python
from src.orchestration.intents.intent_parser import parse
from src.orchestration.router.workflow_router import route
from src.orchestration.registry.experiment_registry import list_experiments

known = list_experiments()
intent = parse("review canonical_ml_showcase", known_experiments=known)
result = route(intent)

if result.error:
    print("Error:", result.error)
else:
    print(result.result.sections["validation_assessment"])
```

---

## 3. Running a Comparative Review

```python
from src.orchestration.api.research_api import run_llm_comparative_review

comparison = run_llm_comparative_review(
    baseline="canonical_ml_showcase",
    candidate="canonical_ml_multi_asset",
    provider="anthropic",
    persist=True,
)

print(comparison.overall_assessment)
print(comparison.key_tradeoffs)    # list[str]
print(comparison.failure_mode_changes)
```

Output is written to:
```
results/comparisons/canonical_ml_showcase__vs__canonical_ml_multi_asset/
    comparative_review.json
    comparative_review.md
```

---

## 4. Generating an Iteration Proposal

```python
from src.orchestration.api.research_api import generate_iteration_proposal

proposal = generate_iteration_proposal(
    "canonical_ml_showcase",
    provider="anthropic",
    persist=True,
)

print(proposal.research_focus)
print(proposal.suggested_experiments)   # list[str]
print(proposal.confidence)              # "medium — ..."
```

Output: `results/llm_reviews/canonical_ml_showcase/iteration_proposal.json`

---

## 5. Generating a Config Draft

Config draft generation requires:
1. A persisted `IterationProposal` for the experiment
2. A YAML config file at `configs/experiments/<name>.yaml` (version 2)

```python
from src.orchestration.api.research_api import generate_experiment_draft

draft = generate_experiment_draft(
    "canonical_ml_showcase",
    provider="anthropic",
    proposal_hash=None,    # None = use most recent proposal without hash check
)

# Inspect proposed changes
for change in draft.changes:
    print(f"{change.section}.{change.field}: {change.current_value} → {change.proposed_value}")
    print(f"  Rationale: {change.rationale}")
```

Output: `results/llm_reviews/canonical_ml_showcase/draft_<uuid>.json`

---

## 6. Approving and Rendering a Draft

```python
from src.orchestration.api.research_api import approve_experiment_draft, render_draft_to_yaml

# Step 1: human reads draft.changes (see above)

# Step 2: approve
approved = approve_experiment_draft(
    "canonical_ml_showcase",
    draft_id=draft.draft_id,
)
assert approved.approved is True

# Step 3: render to YAML
config_path = render_draft_to_yaml(
    "canonical_ml_showcase",
    draft_id=draft.draft_id,
    output_path=None,   # None = auto-name from draft.proposed_name
)
print("Config written to:", config_path)
```

The researcher then manually reviews the generated YAML and decides whether to run the experiment:

```bash
python scripts/run_from_config.py configs/experiments/<proposed_name>.yaml --report --preset canonical
```

---

## 7. Evolution Chain

Evolution chains require lineage records to be registered first.

```python
from src.orchestration.api.research_api import register_experiment_lineage, build_research_evolution_chain

# Register lineage (human-triggered)
register_experiment_lineage(
    "canonical_ml_multi_asset",
    parent_experiment="canonical_ml_showcase",
    iteration_reason="Expand from single-asset to 15-ETF universe to test cross-sectional signals",
    derived_from_iteration=True,
    context_hash=proposal.context_hash,   # from the IterationProposal that prompted this experiment
)

# Build chain (no LLM call)
chain = build_research_evolution_chain(
    "canonical_ml_showcase",   # root experiment
    persist=True,
)

print(chain.evolution_summary)
for step in chain.steps:
    print(f"{step.experiment_name}: {step.key_improvements}")
```

Output: `results/evolution/canonical_ml_showcase/evolution_chain.json`

---

## 8. Research Sessions (Optional)

Sessions are an optional audit trail. No core research function requires a session.

```python
from src.orchestration.api.research_api import (
    create_research_session,
    record_session_event,
    update_session_status,
    get_session_summary,
)
from src.orchestration.session.session_schema import SessionEventType, SessionStatus

# Create session
session = create_research_session(
    root_experiment="canonical_ml_showcase",
    research_goal="Investigate whether breakout_63d instability drives catastrophic_split",
)

# After running review
review = run_llm_review("canonical_ml_showcase", provider="anthropic")
record_session_event(
    session,
    event_type=SessionEventType.REVIEW_GENERATED,
    experiment_name="canonical_ml_showcase",
    data={"context_hash": review.context_hash},
)

# After generating proposal
proposal = generate_iteration_proposal("canonical_ml_showcase", provider="anthropic")
record_session_event(
    session,
    event_type=SessionEventType.ITERATION_PROPOSAL_GENERATED,
    experiment_name="canonical_ml_showcase",
    data={"context_hash": proposal.context_hash},
)

# Get summary
summary = get_session_summary(session.session_id)
print(summary["experiments_visited"])
print(summary["latest_proposal"])

# Close session
update_session_status(session, status=SessionStatus.COMPLETED)
```

---

## 9. Intent Parser Usage

The intent parser is useful for natural-language request routing (e.g., in a chat interface or CLI):

```python
from src.orchestration.intents.intent_parser import parse
from src.orchestration.registry.experiment_registry import list_experiments

known = list_experiments()

# Rule-based (no LLM)
intent = parse("compare canonical_ml_showcase vs canonical_ml_multi_asset", known_experiments=known)
# → CompareExperimentsIntent(baseline="canonical_ml_showcase", candidate="canonical_ml_multi_asset")

intent = parse("rank experiments by sharpe", known_experiments=known)
# → RankExperimentsIntent(descending=True)

intent = parse("what experiments are tagged ml?", known_experiments=known)
# → ListExperimentsIntent(tag="ml")

# LLM fallback (only when rule-based fails and provider is not "stub")
intent = parse(
    "what should I investigate next for my momentum strategy?",
    known_experiments=known,
    provider="anthropic",   # enables LLM fallback
)
```

The default provider is `"stub"`, which disables the LLM fallback. Pass `provider="anthropic"` to enable it.

---

## 10. Artefact Retrieval

```python
from src.orchestration.api.research_api import retrieve_artefact, list_all_experiments, rank_experiments_by_sharpe

# List all experiments
names = list_all_experiments()

# Rank by Sharpe
ranked = rank_experiments_by_sharpe(descending=True)
for name, sharpe in ranked:
    print(f"{name}: {sharpe:.3f}")

# Retrieve a specific artefact
data = retrieve_artefact("canonical_ml_showcase", "ml_diagnostics")
print(data["ic"]["mean_ic"])
```

Valid artefact keys: `metrics`, `ml_diagnostics`, `ml_model_diagnostics`, `split_metrics`, `backtest_diagnostics`, `universe_coverage`, `wf_equity_curves`, `alignment_diagnostics`, `data_summary`, `feature_correlations`, `feature_families`, `feature_registry`, `feature_summary`, `signal_transitions`

---

## 11. Building Context Without LLM

If you want to inspect the assembled context without making an LLM call:

```python
from src.orchestration.api.research_api import build_llm_context

ctx = build_llm_context("canonical_ml_showcase", persist=False)

print(ctx.failure_modes)          # deterministic, no LLM
print(ctx.ml_diagnostics)         # assembled from ml_diagnostics.json
print(ctx.feature_summary)        # assembled from feature_summary.json
print(len(ctx.available_plots))   # primary-importance plots only
```

---

## 12. Provider Override for Local Models

To use a local OpenAI-compatible server (e.g., LM Studio):

```python
review = run_llm_review(
    "canonical_ml_showcase",
    provider="openai",
    model="llama-3.2-3b-instruct",
    base_url="http://127.0.0.1:1234/v1",
    persist=True,
)
```

The `OPENAI_API_KEY` environment variable can be set to any non-empty string when `base_url` is provided (local servers do not validate the key).

---

## 13. Output File Reference

| Operation | Output path |
|---|---|
| `build_llm_context(persist=True)` | `results/llm_reviews/<name>/llm_context.json` |
| `run_llm_review` | `results/llm_reviews/<name>/llm_review.json` |
| `generate_iteration_proposal` | `results/llm_reviews/<name>/iteration_proposal.json` + `.md` |
| `generate_experiment_draft` | `results/llm_reviews/<name>/draft_<uuid>.json` |
| `render_draft_to_yaml` | `configs/experiments/<proposed_name>.yaml` |
| `run_llm_comparative_review` | `results/comparisons/<baseline>__vs__<candidate>/comparative_review.json` + `.md` |
| `build_research_evolution_chain(persist=True)` | `results/evolution/<root>/evolution_chain.json` + `.md` |
| `create_research_session` | `results/research_sessions/<session_id>/session.json` |
| `register_experiment_lineage` | `results/experiments/<name>/lineage.json` |

---

## 14. Troubleshooting

**`RuntimeError: Template 'experiment_review' has N unresolved tokens after render`**

The experiment's artefacts are incomplete. Run the experiment with `--report` to generate all diagnostic files before calling the orchestration layer.

**`FileNotFoundError: No IterationProposal found for 'name'`**

`generate_experiment_draft` requires a persisted proposal. Call `generate_iteration_proposal(..., persist=True)` first.

**`ValueError: Draft generation only supported for ML experiments (config version 2)`**

Config draft synthesis is only implemented for ML configs (`version: 2` in YAML). Equal-weight and other non-ML strategies are not supported.

**`ValueError: Draft must be approved before rendering to YAML`**

Call `approve_experiment_draft()` before `render_draft_to_yaml()`.

**`RuntimeError: ANTHROPIC_API_KEY environment variable is not set`**

Set `ANTHROPIC_API_KEY` in environment or `.env`. To test without a key, use `provider="stub"`.

**`DatasetNotFoundError` in tests**

179 tests require locally generated experiment artefacts. See `results/README.md` for resolution steps.
