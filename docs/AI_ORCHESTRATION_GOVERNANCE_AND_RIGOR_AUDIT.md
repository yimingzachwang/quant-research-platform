# AI Orchestration Governance and Rigor Audit

**Repository:** `quant-research-platform`
**Audit date:** 2026-05-30
**Subject:** Controls that keep LLM output grounded, advisory, and non-autonomous

---

## 1. Governance Philosophy

The orchestration layer is built around one invariant: **LLM output is advisory and the researcher is the decision-maker.** Every design choice in the system exists to enforce this invariant at multiple levels — prompt, code, schema, and test.

The system makes no autonomous decisions about research direction, parameter values, or experiment execution. It provides structured analysis of already-computed artefacts for human interpretation.

---

## 2. Failure Mode Detection: Zero LLM Involvement

**File:** `src/orchestration/context/failure_mode_detector.py`

Failure mode detection is fully deterministic. No LLM call is involved at any point in this path.

**Detection functions and their rules:**

| Detector | Failure Mode | Severity | Trigger |
|---|---|---|---|
| `_check_performance` | `negative_sharpe` | critical | `sharpe < 0` |
| `_check_performance` | `weak_sharpe` | warning | `sharpe < 0.5` |
| `_check_performance` | `severe_drawdown` | warning | `max_drawdown < -0.40` |
| `_check_performance` | `negative_return` | warning | `annual_return < 0` |
| `_check_validation` | `poor_oos_consistency` | critical | `mean_oos_sharpe < 0` |
| `_check_validation` | `high_split_sharpe_variance` | warning | `std_oos_sharpe > 1.0` |
| `_check_validation` | `catastrophic_split` | warning | `worst_split_sharpe < -1.0` |
| `_check_ml_signal` | `weak_ic` | warning | `mean_ic < 0.02` |
| `_check_ml_signal` | `coefficient_instability` | warning | `n_sign_reversal_features / n_features > 0.5` |
| `_check_ml_signal` | `poor_directional_accuracy` | warning | directional accuracy < 48% |
| `_check_alignment` | `high_alignment_loss` | warning | `alignment_loss_pct > 30` |
| `_check_backtest` | `high_turnover` | warning | `monthly_avg_turnover > 1.5` |

Each detected failure mode carries: `name`, `severity`, `description` (plain language), `evidence` (specific metric value string).

**The `failure_modes` list in `LLMContext` is constructed before the LLM sees anything.** The LLM is given these pre-labeled failure modes and is expected to interpret them — not to discover them.

---

## 3. Prompt Grounding Constraints

All three prompt templates include explicit CRITICAL CONSTRAINTS sections that appear before the data context.

### Experiment Review (`experiment_review.txt`)

```
CRITICAL CONSTRAINTS:
- You MUST NOT perform any numerical computation or statistical inference.
- You MUST NOT claim any metric values not explicitly stated in the context.
- You MUST NOT invent features, assets, splits, or model behaviour not described below.
- Ground every observation in the context data provided.
```

Recommendations section further requires:
```
STRICT REQUIREMENTS:
- Every recommendation MUST cite a specific diagnostic value ..., a failure mode name ...,
  or a named feature or family from the context.
- FORBIDDEN generic boilerplate: do NOT recommend "add stop-losses", "diversify the universe",
  "monitor continuously", or "reduce leverage" without grounding each in a specific finding.
```

### Iteration Proposal (`iteration_proposal.txt`)

```
CRITICAL CONSTRAINTS:
- You MUST NOT prescribe specific parameter values (e.g. "set alpha=0.1", "use 63-day lookback").
- You MUST NOT suggest deploying, trading, operating, or allocating capital to the strategy.
- You MUST NOT make causal claims not supported by the diagnostic evidence below.
- You MUST NOT recommend generic financial risk management unconnected to specific findings.
- Every observation MUST be grounded in named metrics, named features, named failure modes,
  or named splits from the context.
```

The FORBIDDEN suggestions list includes explicit anti-patterns:
- "Deploy this strategy"
- "Increase the Sharpe ratio by adjusting X"
- "Add stop-losses" (without diagnostic grounding)
- "Monitor continuously" (generic)
- "Buy more momentum exposure" (investment advice)

The template provides GOOD research direction examples to model the expected framing.

### Comparative Review (`comparative_review.txt`)

```
CRITICAL CONSTRAINTS:
- You MUST NOT claim one experiment is "better" without qualification.
- You MUST NOT recommend deploying, trading, or allocating capital to either experiment.
- You MUST NOT use generic ranking language ("outperforms", "superior", "should be preferred").
- Every observation MUST cite specific named metrics, failure modes, feature names, or split results.
```

---

## 4. Pre-Computed Deltas: LLM Cannot Re-Derive Numbers

In comparative reviews, all numeric deltas are computed in Python before the LLM receives the prompt. The prompt template includes this instruction:

```
## PRE-COMPUTED DIAGNOSTIC DELTAS
The following differences are pre-computed from experiment artefacts.
Use them directly — do not re-derive or contradict these values.
```

This means the LLM's role in comparison is interpretation and communication, not arithmetic. The factual numbers are fixed before the LLM is invoked.

**Deltas pre-computed by `_build_comparative_payload()`:**

- `metric_comparison`: Sharpe delta, drawdown delta, return delta
- `validation_comparison`: OOS Sharpe delta, std OOS Sharpe delta, hit rate delta, n_negative_splits delta
- `ml_comparison`: IC delta, IC tier change, DA tier change, sign reversal features delta
- `feature_comparison`: dominant family change, HHI delta, n_family_transitions delta, most_volatile_feature change
- `universe_comparison`: n_assets change, tickers diff, coverage change
- `failure_mode_comparison`: baseline-only modes, candidate-only modes, shared modes

---

## 5. Rendering Guard: No Broken Context Reaches LLM

**File:** `src/orchestration/llm/review_engine.py`

Two-stage guard before any LLM call:

**Stage 1: Jinja2 `StrictUndefined`**

```python
env = Environment(undefined=StrictUndefined)
```

If any template variable references a field not present in the render context dict, Jinja2 raises `UndefinedError` at render time — before `call_llm()` is invoked.

**Stage 2: `_assert_no_unresolved_tokens()`**

```python
def _assert_no_unresolved_tokens(rendered: str, template_name: str) -> None:
    _UNRESOLVED_RE = re.compile(r"\{\{|\}\}|\{%|%\}")
    remaining = _UNRESOLVED_RE.findall(rendered)
    if remaining:
        raise RuntimeError(
            f"Template {template_name!r} has {len(remaining)} unresolved tokens after render: ..."
        )
```

This catches any Jinja2 bypass or template bug that `StrictUndefined` might miss. The test suite verifies this at `tests/orchestration/test_iteration.py:41`.

---

## 6. Context Hashing: Provenance Chain

**File:** `src/orchestration/llm/review_engine.py`

Every LLM output is linked to the exact context that produced it:

```python
def _compute_context_hash(context: LLMContext) -> str:
    raw = json.dumps(_context_to_dict(context), sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()
```

The 64-character SHA256 hex is embedded in every persisted JSON artefact. It is also used in `generate_experiment_draft` to verify that the draft is built from the expected proposal (optional `proposal_hash` parameter).

**Determinism guarantee:** The test `test_context_hash_same_across_two_iteration_calls` verifies that calling `run_iteration_proposal` twice with the same context produces the same hash.

**Provenance chain:** `LLMContext.context_hash` → `IterationProposal.context_hash` → `ExperimentDraft.source_proposal_hash`. A reviewer can trace any config draft back to the specific diagnostic snapshot that produced it.

---

## 7. Human Approval Gate

**File:** `src/orchestration/config_generation/draft_schema.py`

```python
@dataclass
class ExperimentDraft:
    ...
    approved: bool = False
    approved_at: str | None = None
```

`render_draft_to_yaml()` raises `ValueError` if `draft.approved` is False. The `approved_at` timestamp is set only by `approve_experiment_draft()`.

No orchestration function ever sets `approved=True` except `approve_experiment_draft()`. No automated flow calls `approve_experiment_draft()`.

This means:
1. The researcher must explicitly read the draft
2. Call `approve_experiment_draft()`
3. Then call `render_draft_to_yaml()`

The YAML config is not generated without a human decision in the middle.

---

## 8. Draft Validation: Change Path Vocabulary

**File:** `src/orchestration/config_generation/draft_validator.py`

Before YAML rendering, `validate_draft()` checks every `DraftChange` against `_VALID_CHANGE_PATHS` — a hardcoded set of (section, field) pairs that are allowed to change. Any change outside this vocabulary marks the draft as invalid.

The vocabulary is kept in sync between `draft_validator.py` and the LLM system prompt in `draft_generator.py` (documented with a comment: "The vocabulary listed in `_LLM_SYSTEM` must stay in sync with `_VALID_CHANGE_PATHS` in `draft_validator.py`. Update both together.").

Additionally, `current_value` in each `DraftChange` is always read from the base config by Python code — never from the LLM. This prevents the LLM from falsifying what the current parameter values are.

---

## 9. Advisory-Only Labeling in Schemas

**File:** `src/orchestration/api/schemas.py`

Module-level docstrings on every output schema explicitly state advisory status:

- `ComparativeReview`: "Advisory only — characterises research evolution..."
- `IterationProposal`: "Advisory only — the researcher remains the decision-maker"
- `ExperimentDraft`: "Not executable on its own. `render_to_yaml()` requires `approved=True`"

The `ResearchEvolutionChain` has no LLM dependency and no advisory disclaimer needed — it is deterministically assembled from diagnostic artefacts.

---

## 10. Lineage: Human-Registered Only

`register_experiment_lineage()` in `research_api.py` is never called by any other orchestration function. It is only called by:
- The researcher manually (CLI or API)
- Tests that explicitly set up lineage chains

No LLM output triggers lineage registration. No automated workflow registers lineage.

---

## 11. No Autonomous Execution Loops

The workflow router (`router/workflow_router.py`) contains this in its docstring and inline comment:

```
Design constraints (from the implementation mandate):
  - DO NOT call quant engine functions
  - DO NOT synthesise experiment configs
  - DO NOT introduce autonomous execution loops
  - Only call functions already present in research_api.py
```

No function in `src/orchestration/` calls `subprocess`, `os.system`, or any quant engine function. The architectural decoupling test (`test_api_non_coupling.py`) verifies this at import level.

---

## 12. Test Coverage of Governance Properties

| Governance property | Test location | Mechanism |
|---|---|---|
| No unresolved tokens in rendered prompt | `test_iteration.py:41` | `_UNRESOLVED_RE.findall(prompt) == []` |
| Template contains FORBIDDEN keyword | `test_iteration.py:97` | `assert "FORBIDDEN" in tmpl` |
| Template requires grounding | `test_iteration.py:106` | `assert "grounded" in tmpl.lower()` |
| Context hash is 64 chars | `test_iteration.py:333` | `assert len(data["context_hash"]) == 64` |
| Context hash is deterministic | `test_iteration.py:415` | Two calls, same hash |
| Draft approved=False by default | `test_config_generation.py` | `assert not draft.approved` |
| Rendering blocked without approval | `test_config_generation.py` | `pytest.raises(ValueError)` |
| Orchestration doesn't import quant engine | `test_api_non_coupling.py` | Import graph traversal |
| Failure mode detection uses no LLM | Structural | `failure_mode_detector.py` has no `llm_interface` import |
