# Research Initiation Audit

**Repository:** `quant-research-platform`
**Audit date:** 2026-05-30
**Scope:** How research ideas enter the platform and the degree to which intent-driven research initiation is implemented

---

## 1. Executive Summary

The platform is **approximately 65% of the way** to supporting intent-driven research initiation.

The front half of the lifecycle — natural-language parsing, typed intent classification, workflow routing, analysis functions, config synthesis, human approval, and YAML rendering — is fully implemented and connected. A researcher can type a natural-language request, have it classified into a typed intent, routed to the correct analysis function, and end up with a validated, human-approved YAML config file — all through a single API call chain.

The back half — triggering experiment execution from that config — is completely absent from the orchestration layer. There is no subprocess call, no `run_experiment_from_config()` invocation, no `RunExperimentIntent`, and no job queue anywhere in `src/orchestration/` or `src/api/`. The YAML file's comment block reads: `"Review and run with: python scripts/run_from_config.py configs/experiments/{name}.yaml"` — addressed to the researcher, not to any automated system.

**The critical gap** is the connection between stage 1 (AI Research Planning) and stage 2 (Quantitative Research Engine). Every stage within each half is well-built. The bridge between them does not exist.

| Stage | Status |
|---|---|
| Natural-language → intent | Implemented |
| Intent → typed dataclass | Implemented |
| Intent → research analysis | Implemented |
| Intent → config draft | Implemented |
| Config draft → validated YAML | Implemented |
| Validated YAML → experiment execution | **Not implemented** |
| Experiment execution → artefacts | Implemented (quant engine, manual trigger) |
| Artefacts → AI review | Implemented |
| AI review → iteration proposal | Implemented |
| Iteration proposal → next intent | **Not implemented** (no loop automation) |

---

## 2. Existing Components

### `intent_schema.py`

**Responsibilities:** Defines the ten typed intent dataclasses. All are `frozen=True` dataclasses, usable as dict keys.

**Supported intent classes:**

| Class | Parameters |
|---|---|
| `ReviewExperimentIntent` | `experiment_name`, `provider`, `model` |
| `CompareExperimentsIntent` | `baseline`, `candidate`, `provider`, `model` |
| `GenerateIterationIntent` | `experiment_name`, `provider`, `model` |
| `BuildEvolutionChainIntent` | `root_experiment` |
| `ListExperimentsIntent` | `tag`, `strategy_pattern` |
| `RankExperimentsIntent` | `descending` |
| `RetrieveArtefactIntent` | `experiment_name`, `key` |
| `BuildContextIntent` | `experiment_name` |
| `GenerateDraftIntent` | `experiment_name`, `provider`, `model` |
| `UnrecognisedIntent` | `raw_text`, `reason` |

**Limitations:** No `RunExperimentIntent`, no `ScheduleExperimentIntent`, no `ExecuteIntent`. The schema deliberately excludes execution. The `Intent` union type alias enumerates exactly these ten classes — extending it to add execution would require parallel additions to the parser, router, and API.

---

### `intent_examples.py`

**Responsibilities:** Documents 25 canonical natural-language inputs paired with expected intent type and notes. Used directly by tests — not test fixtures, not documentation examples, but actual test inputs.

**Coverage:** All 9 non-`UnrecognisedIntent` types have at least 2 examples each. Examples cover synonyms (review/analyze/assess/interpret), question forms, shorthand, and multi-word trigger phrases.

**Limitations:** No examples for execution-class requests ("run the next experiment", "execute the approved config"). This absence is by design, not oversight — execution intent is out of scope.

---

### `intent_parser.py`

**Responsibilities:** Two-stage parser that converts raw text into a typed `Intent`. Stage 1 is rule-based (deterministic, no API key). Stage 2 is an LLM fallback (only fires when rule-based fails and a live provider is requested).

**Current behaviour:** Rule-based classification uses 9 compiled regex patterns against keyword sets. Experiment name extraction uses substring matching against a caller-provided list of known experiments, ordered by first-occurrence position with length as a tiebreaker.

**Supported workflows:** All 9 non-error intent types via rule-based parsing. Ambiguous or novel phrasings fall through to the LLM fallback. The default provider in `parse()` is `"stub"`, which skips the LLM fallback — callers must explicitly pass a live provider to enable it.

**Limitations:**
- Rule-based classification is single-pass — it does not handle chained or multi-step requests ("review this, then generate a proposal, then draft a config")
- No execution-related keywords (`run`, `execute`, `launch`) are in any pattern table
- Experiment name extraction requires `known_experiments` to be passed in; it does not self-discover available experiments
- The LLM fallback still maps to the same 9 intent types — it cannot invent a new intent class

---

### `workflow_router.py`

**Responsibilities:** Dispatches a typed `Intent` to the corresponding `research_api` function. Measures elapsed time. Catches all exceptions and returns a `WorkflowResult` (never raises).

**Current behaviour:** 9 `isinstance` branches, one per concrete intent type. `UnrecognisedIntent` returns an error `WorkflowResult` without making any API call.

**Supported workflows:** All 9 intents map to exactly one `research_api` function. The mapping is one-to-one — no branching, no multi-step sequences, no chaining.

**Limitations:**
- No `RunExperimentIntent` branch exists and no API function for it is called
- Router is stateless — it has no memory of previous intents in a session
- Cannot execute research plans that require multiple sequential steps (e.g., "review, then propose, then draft")
- The router's docstring explicitly prohibits execution: "DO NOT introduce autonomous execution loops"

---

### `routing_schema.py`

**Responsibilities:** Defines `WorkflowResult` — a frozen dataclass wrapping the API return value, elapsed time, and any error string.

**Current behaviour:** `success` property returns `not bool(self.error)`. No retry logic, no partial-success handling, no streaming.

**Limitations:** Single-call semantics only. Does not support multi-step workflow results.

---

### `research_api.py`

**Responsibilities:** The sole public entry point for all orchestration. 19 functions covering the full post-experiment research lifecycle.

**Current behaviour:** Every function reads from persisted experiment artefacts, calls LLM where needed, persists output, and returns a typed dataclass. No function calls the quant engine. No function calls `subprocess`. No function launches experiments.

**Supported workflows:** Complete advisory lifecycle from context assembly through YAML rendering. Complete session management. Experiment discovery and ranking.

**Limitations:**
- No `run_experiment()`, no `launch_experiment()`, no `schedule_experiment()`
- Draft synthesis only works for ML configs (version 2). Equal-weight, momentum-rotation, and other non-ML strategy configs are not supported
- YAML render is the final step — the system explicitly stops there and asks the researcher to run the script manually

---

## 3. Intent Parsing Audit

### Rule-based parsing — what works

The rule-based stage handles all 9 supported intent types reliably for standard phrasings. Below are tested canonical examples from `intent_examples.py`:

**Review:**
```
"review canonical_ml_showcase"             → ReviewExperimentIntent
"analyze the results for canonical_ml_showcase"  → ReviewExperimentIntent
"assess canonical_ml_multi_asset performance"    → ReviewExperimentIntent
```

**Compare:**
```
"compare canonical_ml_showcase vs canonical_ml_multi_asset"     → CompareExperimentsIntent
"diff canonical_ml_showcase and canonical_ml_multi_asset"       → CompareExperimentsIntent
```

**Generate proposal:**
```
"generate an iteration proposal for canonical_ml_showcase"      → GenerateIterationIntent
"suggest improvements to canonical_ml_multi_asset"              → GenerateIterationIntent
"what should the next experiment after canonical_ml_showcase be" → GenerateIterationIntent
```

**Draft:**
```
"generate draft for canonical_ml_showcase"      → GenerateDraftIntent
"synthesize config for canonical_ml_showcase"   → GenerateDraftIntent
```

**Rank/list:**
```
"rank experiments by sharpe"       → RankExperimentsIntent
"list all experiments"             → ListExperimentsIntent
"which experiments perform best"   → RankExperimentsIntent
```

### Rule-based parsing — what fails

Execution-adjacent phrasings produce `UnrecognisedIntent` because no pattern covers them:

```
"run the approved config for canonical_ml_showcase"     → UnrecognisedIntent
"execute the next experiment"                           → UnrecognisedIntent
"launch canonical_ml_showcase_v2"                       → UnrecognisedIntent
"start the experiment from the draft"                   → UnrecognisedIntent
```

Multi-step chained requests also fail:
```
"review canonical_ml_showcase and then generate a proposal"  → ReviewExperimentIntent
   (only first intent recognized; second is silently dropped)
```

### LLM fallback

When enabled (provider ≠ `"stub"`), the LLM fallback sends the raw text plus the list of known experiments to a structured-JSON classification prompt. The LLM returns a JSON object with `{"intent": "<ClassName>", "params": {...}}`. This is robust to novel phrasings but still constrained to the same 9 intent types — the prompt enumerates exactly those classes. A request for experiment execution would return `UnrecognisedIntent`.

### FastAPI endpoint

```
POST /api/route
Body: {"text": "...", "provider": "anthropic", "model": null}
Response: {"intent_type": "...", "success": true/false, "result": {...}, "error": null, "elapsed_seconds": 0.41}
```

The `/api/route` endpoint is the only HTTP surface that exposes the full parse → route → execute pipeline to external callers.

---

## 4. Research Initiation Capability Matrix

| Stage | Implemented | Partial | Missing | Evidence |
|---|---|---|---|---|
| Natural-language request | ✅ | | | `POST /api/route`, `intent_parser.parse()` |
| Intent classification | ✅ | | | 9 intent types, rule-based + LLM fallback |
| Typed research intent | ✅ | | | `intent_schema.py` — 10 frozen dataclasses |
| Experiment selection | ✅ | | | `ListExperimentsIntent`, `RankExperimentsIntent`, `RetrieveArtefactIntent` all implemented |
| Experiment modification planning | ✅ | | | `GenerateIterationIntent` → `GenerateDraftIntent` pipeline full |
| Config synthesis | | ✅ | | ML experiments only (version 2); equal-weight/other strategies unsupported |
| Experiment launch planning | | ✅ | | YAML is written; a comment tells the researcher the CLI command; no automated handoff |
| Experiment execution triggering | | | ❌ | No subprocess, no `run_experiment_from_config()` call, no `RunExperimentIntent` |
| Validation of execution plans | ✅ | | | `validate_draft()` calls `validate_ml_config()` before rendering |
| Human approval gates | ✅ | | | `ExperimentDraft.approved = False` default; `POST /sessions/{id}/draft/approve` endpoint |
| Experiment scheduling | | | ❌ | No queue, no async execution, no job management |
| Research session creation | ✅ | | | `POST /api/sessions`, full CRUD, event log |
| Lineage registration | | ✅ | | Implemented but human-triggered only; no automatic registration after experiment run |

---

## 5. Deterministic Execution Planner Audit

**No component currently functions as an execution planner.**

The `workflow_router.py` is the closest existing analogue. It accepts a typed intent and dispatches it to one API function. But it is not a planner — it makes no decisions about what to do next, cannot queue multiple actions, has no concept of a research plan as a multi-step sequence, and cannot observe the output of one step to determine the next.

**What already exists that would support a planner:**

| Component | Relevance |
|---|---|
| `WorkflowResult` | Could be used as input to a planner that inspects `result` to decide the next step |
| `ResearchSession` + event log | Could serve as the planner's state — session already tracks which steps have occurred |
| `Intent` dataclasses | A planner could construct and enqueue these instead of the parser |
| `research_api.py` functions | All callable sequentially without state conflict |
| `SessionEventType` constants | Could drive a state machine: `REVIEW_GENERATED` → `ITERATION_PROPOSAL_GENERATED` → `DRAFT_GENERATED` → `DRAFT_APPROVED` → `YAML_RENDERED` |

The session event type sequence implies the intended workflow order but does not enforce it. Nothing checks that `DRAFT_GENERATED` only happens after `ITERATION_PROPOSAL_GENERATED`, or that `YAML_RENDERED` only happens after `DRAFT_APPROVED` (the approval check is the only enforced gate).

**What is missing for a planner:**

- A `ResearchPlan` dataclass or equivalent (ordered sequence of steps with preconditions)
- A plan executor that calls steps in order and checks preconditions
- A bridge from `render_draft_to_yaml()` output to `run_experiment_from_config()` input
- A `RunExperimentIntent` and corresponding API function

---

## 6. Config Generation Path

**Full path:**

```
Intent (GenerateIterationIntent)
  → generate_iteration_proposal()
  → IterationProposal [JSON persisted]
  ↓
Intent (GenerateDraftIntent)
  → generate_experiment_draft()
  → load base YAML from configs/experiments/<name>.yaml
  → LLM extracts DraftChange list (JSON only, bounded vocabulary)
  → current_value filled from base config (not LLM)
  → ExperimentDraft [JSON persisted, approved=False]
  ↓
validate_experiment_draft()
  → whitelist check (_VALID_CHANGE_PATHS)
  → name collision check (registry)
  → apply_changes() to deep copy
  → validate_ml_config() (authoritative quant engine validator)
  → DraftValidationResult
  ↓
approve_experiment_draft()
  → ExperimentDraft.approved = True, approved_at = <ISO timestamp>
  ↓
render_draft_to_yaml()
  → apply_changes() to deep copy of base config
  → normalize_ml_config() — fills all defaults
  → validate_ml_config() — belt-and-suspenders
  → ml_experiment_hash() — provenance
  → writes YAML to configs/experiments/<proposed_name>.yaml
  → returns yaml_str
```

**What already exists:** This entire path is implemented and tested. It is deterministic — given the same base config and the same LLM output, the same YAML is always produced.

**What is missing:**

1. The path from `render_draft_to_yaml()` to experiment execution. The YAML exists on disk. Nothing reads it back and runs it.
2. Support for non-ML config versions. Draft synthesis calls `validate_ml_config()` and loads via `src.experiments.ml_config`. If the base experiment is not a v2 ML config, it raises immediately.

**Is the path deterministic?** Yes, after the LLM call. The LLM introduces non-determinism (temperature 0.1, but not zero). The `normalize_ml_config()` and `validate_ml_config()` steps are fully deterministic.

---

## 7. Experiment Launch Audit

**Can the orchestration layer currently trigger experiments?**

**No.**

Evidence:

1. `grep -rn "subprocess\|os.system\|Popen"` across all of `src/orchestration/` and `src/api/` returns zero results.
2. No import of `src.experiments.orchestrator` or `run_experiment_from_config` exists anywhere in `src/orchestration/`.
3. No `RunExperimentIntent` exists in `intent_schema.py`.
4. No router branch in `workflow_router.py` maps to any execution function.
5. The YAML renderer's header comment is: `"Review and run with: python scripts/run_from_config.py configs/experiments/{proposed_name}.yaml"` — this is addressed to the human researcher.

`run_experiment_from_config()` does exist in `src/experiments/orchestrator.py` (line 680) and is exposed via `src/cli.py` and `scripts/run_from_config.py`. But these are in the quant engine layer — not accessible from the orchestration layer, which explicitly prohibits such imports (`test_api_non_coupling.py`).

**The coupling test that prevents this:**

`tests/integration/test_api_non_coupling.py` verifies at import-graph level that `src/orchestration/` has zero imports from `src/strategies/`, `src/models/`, `src/backtest/`, `src/features/`, `src/data/`, or `src/validation/`. Adding a direct call to `run_experiment_from_config` from orchestration would violate this test.

Bridging experiment execution would require either: (a) relaxing the coupling boundary (not recommended), or (b) implementing an inter-process mechanism (subprocess/message queue/file watch) that keeps the two layers decoupled.

---

## 8. Governance Requirements

If intent-driven execution were added, the following governance controls from the Governance Audit **must remain**:

**Human approval gates (mandatory):**
- The `ExperimentDraft.approved` gate must remain. No execution path should be callable without an explicit approval step.
- Adding a `RunExperimentIntent` must require that the draft has `approved=True` before triggering execution — not just `render_draft_to_yaml()`.

**Validation requirements (mandatory):**
- `validate_draft()` → `validate_ml_config()` must run before any execution trigger, not just before YAML rendering.
- Any execution path that bypasses validation is a regression.

**Provenance requirements (mandatory):**
- The execution trigger must record: `draft_hash`, `source_proposal_hash`, `context_hash`, timestamp, and the experiment name in the session event log.
- `lineage.json` should be auto-registered post-execution linking the new experiment to its parent and draft hash — this is currently human-triggered.

**Audit requirements:**
- Session event log must gain a new `EXPERIMENT_QUEUED` or `EXPERIMENT_LAUNCHED` event type.
- All execution events must be traceable back to the approved draft.

**Trust boundary (critical):**
- The coupling test (`test_api_non_coupling.py`) should be extended, not weakened. The execution bridge should be an inter-process boundary (subprocess, message queue, file lock trigger), not a direct function import from the quant engine.
- The LLM must remain on the analysis side of this boundary. No LLM output should ever directly invoke execution without passing through: validation → human approval → execution trigger.
- The `_LLM_SYSTEM` prompt in `draft_generator.py` explicitly forbids executable instructions. This constraint must extend to any future execution-planning prompt.

---

## 9. Gap Analysis

### Implemented Today

- Natural-language input accepted via `POST /api/route`
- Rule-based intent classification for 9 intent types
- LLM fallback classification (same 9 types)
- Typed frozen intent dataclasses
- One-to-one intent → API function dispatch via `WorkflowResult`
- Experiment listing, ranking, artefact retrieval
- LLM context assembly from experiment artefacts
- Deterministic failure mode detection
- LLM experiment review with structured section extraction
- LLM comparative review with pre-computed deltas
- LLM iteration proposal (advisory)
- Config draft synthesis with bounded vocabulary (ML v2 only)
- Draft whitelist validation and `validate_ml_config()` check
- Human approval gate (draft must be approved before YAML renders)
- YAML config rendering with normalization, validation, provenance hash
- Research session CRUD with event log
- FastAPI HTTP layer exposing all above functions
- Architecture coupling test preventing quant engine contamination

### Partially Implemented

- Config synthesis: **ML experiments only** — version 2 required; non-ML strategies not supported
- Evolution chain: implemented but requires human lineage registration; not automatically populated after experiment runs
- Intent classification of ambiguous requests: LLM fallback handles many cases but cannot classify execution-class requests
- Session event log: exists and is populated, but no state machine enforcement of step ordering

### Missing

```
Natural-Language Request
    ↓ ← implemented
Intent
    ↓ ← implemented
Planning           ← MISSING: no multi-step planner, no ResearchPlan dataclass
    ↓
Execution          ← MISSING: no RunExperimentIntent, no subprocess trigger, no job queue
    ↓              ← the quant engine runs, but only when invoked manually
Artefacts
    ↓ ← implemented (quant engine produces artefacts)
Review             ← implemented
```

**Specific missing pieces:**

| Missing component | Description |
|---|---|
| `RunExperimentIntent` | Typed intent for triggering experiment execution |
| Execution bridge | Inter-process mechanism connecting YAML output to `run_experiment_from_config()` |
| `ResearchPlan` | Multi-step ordered plan (review → propose → draft → approve → run) |
| Plan executor | Component that walks a `ResearchPlan`, checking preconditions at each step |
| Experiment scheduler | Job queue or async execution manager |
| Auto-lineage registration | Automatic `lineage.json` creation after a generated config is executed |
| Non-ML config synthesis | Draft synthesis for equal-weight, momentum-rotation, and other non-ML strategies |
| Execution audit event types | `EXPERIMENT_QUEUED`, `EXPERIMENT_LAUNCHED`, `EXPERIMENT_COMPLETED` session events |

---

## 10. Website Implications

### What the website currently shows

The homepage lifecycle diagram presents 5 stages:

```
Research Goal
  → AI Research Planning     [AI-Assisted]    Intent Parser, Workflow Router, Config Synthesiser, Human Approval
  → Quantitative Research Engine [Deterministic]  Data, Features, ML, Portfolio Construction
  → Validation & Evidence    [Governed]        Walk-Forward, IC Diagnostics, Failure Visibility
  → Research Artefacts       [Reproducible]    Reports, Diagnostics, Registries, Provenance
  → AI Research Review       [AI-Assisted]     Context Builder, LLM Review, Iteration Proposal, Experiment Draft
  ↻ (loop back to Research Goal)
```

A "Human Approval Required" gate is shown inline. The loop carries the label: "research evolves through successive iterative investigations."

### What the website can truthfully claim

**The website can truthfully claim:**

- "AI Research Planning" is implemented: intent parsing, workflow routing, config synthesis, and the human approval gate all exist and function
- "AI Research Review" is implemented: context builder, failure detection, LLM review, iteration proposal, and experiment draft are all implemented
- "Research Artefacts" are implemented: the quant engine produces reproducible artefacts with provenance records
- "Human Approval Required" is truthful: the `approved=False` default and approval gate are real enforcement, not documentation fiction
- "Validation & Evidence" is implemented: walk-forward, IC diagnostics, failure mode detection all exist

**The website cannot truthfully claim:**

- A seamless automated connection between "AI Research Planning" and "Quantitative Research Engine" — the → arrow between these two stages implies a connection that does not exist in code
- "End-to-end automated research lifecycle" — the system requires manual intervention between YAML generation and experiment execution
- Intent-driven experiment launching — there is no `RunExperimentIntent` and no mechanism to trigger `run_experiment_from_config()` from the orchestration layer

### Recommended labeling

**Current Capability** (label these stages as implemented):

- AI Research Planning *(through YAML rendering — the researcher then runs the script)*
- Quantitative Research Engine *(implemented; manually triggered)*
- Validation & Evidence *(fully implemented)*
- Research Artefacts *(fully implemented)*
- AI Research Review *(fully implemented)*

**Planned Research Workflow** (label if added):

- Automated execution trigger *(YAML → run without manual step)*
- Multi-step research planner *(review → propose → draft → run as a single intent)*
- Experiment scheduler *(queued or async execution)*

**Specific recommendation:**

The lifecycle diagram's arrow between "AI Research Planning" and "Quantitative Research Engine" should carry a footnote or annotation: *"Config rendered as YAML — researcher reviews and launches."* The "Human Approval Required" gate already exists in the diagram and is accurate. Extend its scope annotation to make clear that this gate is also where the human triggers execution, not just approves the config.

The loop annotation ("research evolves through successive iterative investigations") is truthful and well-framed — it describes a research practice, not an automated loop. No change needed.
