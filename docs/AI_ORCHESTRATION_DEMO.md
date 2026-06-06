# AI Orchestration Demo

A runnable walk-through of the governed, advisory AI-assisted research cycle.

```
Research Session
  → Structured LLM Context
  → LLM Review
  → Iteration Proposal
  → Experiment Draft
  → Draft Validation
  → Governance Check (render-before-approval is blocked)
  → Human Approval
  → YAML Rendering
  → Session Summary
```

The demo communicates one invariant:

> **AI proposes. The platform validates. The researcher approves. The quant engine remains authoritative.**

Script: [`scripts/demo_ai_orchestration.py`](../scripts/demo_ai_orchestration.py)

---

## Purpose

This demo proves the governed AI-assisted research workflow end-to-end against
**already-persisted experiment artefacts**. It exercises the existing
orchestration API directly (not the FastAPI bridge) and ends at YAML
generation. The rendered config is the final artefact — the researcher inspects
and runs it manually later.

## What It Shows

- structured context construction (`build_llm_context`)
- deterministic failure mode detection (no LLM)
- LLM review (`run_llm_review`)
- iteration proposal (`generate_iteration_proposal`)
- config draft synthesis — the LLM proposes **deltas**, never YAML
- existing draft validation (`validate_experiment_draft`)
- the human approval gate (`approve_experiment_draft`)
- YAML rendering with provenance (`render_draft_to_yaml`)
- research session tracking and summary

## What It Does Not Show

- autonomous experiment execution
- automatic alpha discovery
- trading or portfolio management
- self-improving loops

By default the demo never runs an experiment, never touches the backtest/ML
engine, and never registers lineage automatically — it ends at rendered YAML.
Execution is available only as an explicit, human-authorised opt-in (see
[Optional Human-Controlled Execution](#optional-human-controlled-execution)),
runs exactly one approved config, and is never triggered by the LLM.

---

## How This Maps To The Five-Layer Architecture

The demo console output is grouped under the five architectural layers so it can
be read — or screen-recorded — as a walk down the diagram. Layer 5 (research
memory) bookends the run: the session opens it and the summary closes it.

### Layer 1 — AI Planning Layer

A research request is routed into a supported workflow and translated into a
governed configuration draft. In the demo: the LLM review (stage 3), the
iteration proposal (stage 4), and the experiment draft of typed deltas
(stage 5). The LLM proposes deltas, never YAML.

### Layer 2 — Human Governance Layer

The draft cannot become YAML until explicit approval is recorded. In the demo:
draft validation (stage 6), the proof that rendering before approval is blocked
(stage 7), and the explicit human approval gate (stage 8).

### Layer 3 — Quant Research Engine

Execution is deliberately outside the AI loop. The demo stops at the rendered
YAML and prints the command the researcher *would* run manually later — it never
runs it. The quant engine remains the sole authority for producing results.

### Layer 4 — AI Context Engineering

Research artefacts are assembled into structured context before any LLM review.
In the demo: structured context assembly and deterministic, rule-based failure
mode detection (stage 2) — no LLM is involved in this step.

### Layer 5 — AI Reasoning & Research Memory

Review, proposal, draft, approval, and rendered YAML are recorded as a research
session. In the demo: the session is created (stage 1) and projected into a
summary (stage 10); LLM reasoning produces the review and proposal; all
orchestration artefacts (review, proposal, draft, session) are persisted with
provenance hashes that chain context → proposal → draft → rendered YAML.

| Layer | Demo stages |
|---|---|
| 1 · AI Planning | 3 (review), 4 (proposal), 5 (draft) |
| 2 · Human Governance | 6 (validate), 7 (render-blocked), 8 (approve) |
| 3 · Quant Research Engine | manual boundary after 9 (not executed) |
| 4 · AI Context Engineering | 2 (context + failure modes) |
| 5 · AI Reasoning & Research Memory | 1 (session open), 10 (summary), reasoning across 3–4 |

---

## Prerequisites

The demo reads a pre-existing experiment result tree. By default it uses
`canonical_ml_showcase`. If the required artefacts are missing, the demo prints
a clear message and exits non-zero **without** running anything.

Generate the artefacts first (separate, researcher-initiated step):

```bash
python scripts/run_from_config.py configs/experiments/canonical_ml_showcase.yaml --report --preset canonical
```

Required artefacts checked by the preflight:

- `results/experiments/<name>/metadata.json`
- `results/experiments/<name>/metrics.json`
- `configs/experiments/<name>.yaml` (version-2 ML config — needed for draft synthesis)

Recommended (richer review/proposal, but not fatal if absent):

- `results/experiments/<name>/diagnostics/ml_diagnostics.json`
- `results/experiments/<name>/diagnostics/split_metrics.json`
- `results/experiments/<name>/research/feature_summary.json`
- `results/experiments/<name>/plots/plot_index.json`

---

## Stub Mode

Default mode. Deterministic, no API key, no network call, safe to run locally.
Does not require LM Studio, Anthropic, or OpenAI.

```bash
python scripts/demo_ai_orchestration.py --provider stub
```

Add `--dry-run` to render the final YAML in memory without writing it to disk:

```bash
python scripts/demo_ai_orchestration.py --provider stub --dry-run
```

Add `--transcript` to also save the console output (useful for docs/website):

```bash
python scripts/demo_ai_orchestration.py --provider stub --transcript
# writes results/demo/ai_orchestration_demo_transcript.txt
```

> Note: the stub LLM returns a fixed placeholder, which is not valid JSON for
> the draft-synthesis step. For the stub provider only, the demo supplies a
> deterministic, schema-valid change set (stronger L2 regularisation), mirroring
> how the test suite injects a canned response. Real providers are untouched.

## LM Studio Mode

Uses the existing OpenAI-compatible provider path — there is no separate
`lmstudio` provider. Use `--provider openai` together with `--base-url`.

1. Open LM Studio.
2. Load a local model (e.g. a Qwen instruct model).
3. Start the OpenAI-compatible local server.
4. Confirm it is running:

   ```bash
   curl http://127.0.0.1:1234/v1/models
   ```

5. Run the demo:

   ```bash
   export OPENAI_API_KEY=lm-studio

   python scripts/demo_ai_orchestration.py \
     --provider openai \
     --model <exact-model-id-from-lm-studio> \
     --base-url http://127.0.0.1:1234/v1
   ```

If the server is unavailable, the demo fails clearly and prints the `curl`
check above to help diagnose the connection.

## Anthropic Mode (optional)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/demo_ai_orchestration.py --provider anthropic
```

---

## Optional Human-Controlled Execution

By default the demo stops at the rendered YAML — that is the governance
boundary. Execution is an **opt-in, human-controlled** extension.

> The AI never executes an experiment by itself. Execution only occurs when the
> researcher explicitly authorises the approved YAML to be run. After the quant
> engine generates artefacts, the AI orchestration layer can analyse the result.

How it works:

- `--execute-approved` adds an execution phase **after** YAML rendering.
- Without `--yes`, the demo prints a boundary banner and requires the researcher
  to type exactly `RUN` to proceed. Anything else cancels and exits `0`.
- `--yes` skips the prompt — but only when `--execute-approved` is also passed.
  `--yes` on its own does nothing.
- Exactly **one** approved config is run through the existing quant engine
  (the same path as `scripts/run_from_config.py`). There is no loop, no retry,
  and no metric-driven re-run.
- After execution, the platform builds context from the newly generated
  artefacts and runs an LLM review of the result.
- The lifecycle is recorded as session events: `EXECUTION_REQUESTED`,
  `EXECUTION_COMPLETED`, `POST_RUN_REVIEW_GENERATED`.

This is **not** autonomous execution: the LLM does not decide to run, cannot run
an unapproved config, and never bypasses validation or approval.

### Stub with no execution (default)

```bash
python scripts/demo_ai_orchestration.py --provider stub
```

### Stub with execution, interactive

```bash
python scripts/demo_ai_orchestration.py --provider stub --execute-approved
# prints the boundary, then waits for you to type: RUN
```

### Stub with execution, non-interactive

```bash
python scripts/demo_ai_orchestration.py --provider stub --execute-approved --yes
```

### LM Studio with execution

```bash
export OPENAI_API_KEY=lm-studio

python scripts/demo_ai_orchestration.py \
  --provider openai \
  --model <exact-lm-studio-qwen-model-id> \
  --base-url http://127.0.0.1:1234/v1 \
  --execute-approved
```

Add `--yes` for a non-interactive local run. The same `provider="openai"` +
`base_url` path drives review, proposal, draft, and the post-run review — there
is no separate `lmstudio` provider.

### API endpoint

The same governed bridge is exposed once over HTTP:

```http
POST /api/sessions/{session_id}/execute-approved-config
```

```json
{
  "config_path": "configs/experiments/canonical_ml_showcase_v2.yaml",
  "provider": "stub",
  "report": true,
  "preset": "canonical",
  "dry_run": false
}
```

With `dry_run: true` the endpoint returns the planned command and records no
events. Otherwise it records `EXECUTION_REQUESTED`, runs one approved config,
records `EXECUTION_COMPLETED`, reviews the new artefacts, records
`POST_RUN_REVIEW_GENERATED`, and returns `{execution, review, session, summary}`.
A failed execution returns a clear error and skips the post-run review.

---

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--provider` | `stub` | `stub`, `openai`, or `anthropic` |
| `--model` | provider default | Model ID; for LM Studio the exact server model id |
| `--base-url` | none | OpenAI-compatible base URL (LM Studio: `http://127.0.0.1:1234/v1`) |
| `--experiment` | `canonical_ml_showcase` | Experiment name to analyse |
| `--goal` | "Investigate whether the model can improve validation robustness." | Recorded research goal |
| `--dry-run` | off | Render final YAML in memory without writing to disk |
| `--transcript` | off | Also write console output to `results/demo/ai_orchestration_demo_transcript.txt` |
| `--execute-approved` | off | Optionally run the approved YAML after rendering (requires authorisation) |
| `--yes` | off | Skip the interactive `RUN` confirmation (no effect without `--execute-approved`) |
| `--execution-preset` | `canonical` | Report preset for the execution run |
| `--no-report` | off | Run the approved config without generating a report |

Exit codes: `0` success · `1` assertion/validation/LLM/execution failure · `2` missing required artefacts.

---

## Expected Output (shortened)

```text
======================================================================
AI Orchestration Demo — governed, advisory research cycle
======================================================================
    experiment: canonical_ml_showcase
    provider: stub

Five-layer AI orchestration architecture:
    1. AI Planning Layer
    2. Human Governance Layer
    3. Quant Research Engine
    4. AI Context Engineering
    5. AI Reasoning & Research Memory

Doctrine: AI proposes. The platform validates. The researcher approves. The quant engine remains authoritative.

=== AI REASONING & RESEARCH MEMORY (session opened) ===
    LAYER 5 · AI REASONING & RESEARCH MEMORY — ...
[1/10] Creating research session...
    session id: 9ececd6f-...

=== AI CONTEXT ENGINEERING ===
    LAYER 4 · AI CONTEXT ENGINEERING — ...
[2/10] Building structured LLM context (deterministic, no LLM call)...
    context hash: 0f89469751b5ab4b...
    detected failure modes: ...

=== AI PLANNING LAYER  (grounded by AI Reasoning) ===
    LAYER 1 · AI PLANNING — ...
[3/10] Running LLM review...
[4/10] Generating iteration proposal...
[5/10] Generating experiment draft (deltas only)...
    proposed config name: canonical_ml_showcase_v2
    approved: False
    proposed changes:
      - model.params.alpha: 0.5 -> 1.0
        rationale: Increase L2 regularisation ...

=== HUMAN GOVERNANCE LAYER ===
    LAYER 2 · HUMAN GOVERNANCE — ...
[6/10] Validating draft against the config schema...
    validation: PASS
[7/10] Governance check: rendering YAML before approval must be blocked...
Unapproved YAML render blocked: PASS
[8/10] Approving draft (explicit human gate)...
    approved: True

=== APPROVED CONFIGURATION ===
[9/10] Rendering approved draft to YAML...
    rendered YAML path: configs/experiments/canonical_ml_showcase_v2.yaml
    draft hash (provenance): 0c7bde452430
    (the rendered config is NOT executed by this demo)

=== QUANT RESEARCH ENGINE (manual execution boundary) ===
    LAYER 3 · QUANT RESEARCH ENGINE — execution is outside the AI loop.
    The AI loop ends here. To run the approved config, the researcher
    manually executes (this demo never does):
      python scripts/run_from_config.py configs/experiments/canonical_ml_showcase_v2.yaml --report --preset canonical

=== AI REASONING & RESEARCH MEMORY (session summary) ===
    LAYER 5 · AI REASONING & RESEARCH MEMORY — ...
[10/10] Session summary...
    event count: 6
    active draft: None
    approved config path: configs/experiments/canonical_ml_showcase_v2.yaml

AI orchestration demo complete.
```

When artefacts are missing, the demo instead prints the generation command and
exits with code `2`.

---

## Governance Assertions

The demo fails (non-zero exit) if any of these do not hold:

- context built successfully
- review generated
- proposal generated
- draft generated with `approved=False`
- draft validation succeeds
- **YAML rendering before approval raises an error**
- approval sets `approved=True`
- YAML rendering succeeds only after approval
- rendered YAML contains the draft hash (provenance)
- session summary has `active_draft is None` after `YAML_RENDERED`
- session summary has a non-null approved config path
- the session timeline contains the full set of expected events
- no experiment execution is triggered unless `--execute-approved` is passed

Execution governance (enforced in code and tests):

- `--yes` alone never authorises execution
- with `--execute-approved` and no `--yes`, only typing exactly `RUN` proceeds
- the execution bridge runs exactly one config — no loop, no retry
- a failed execution skips the post-run review
- lineage is never registered automatically

---

## Website Copy

Ready-to-use, non-hype copy for the website and presentation surfaces.

### One-sentence description

> Zeto's AI orchestration layer turns validated research artefacts into grounded
> reviews, iteration proposals, and human-approved configuration drafts without
> allowing the AI to execute experiments.

### Three-bullet summary

- AI reviews structured research evidence, not raw trading data.
- The platform validates every proposed configuration change.
- The researcher must approve before YAML is rendered; execution remains manual.

### Diagram caption

> Governed AI-assisted research orchestration: AI proposes, the platform
> validates, and the researcher approves before execution.

---

## Tests

[`tests/orchestration/test_ai_orchestration_demo.py`](../tests/orchestration/test_ai_orchestration_demo.py)
covers argument parsing, the artefact preflight, the stub draft schema, the
render-before-approval block, render-after-approval provenance, and a guard that
the demo never wires in experiment execution. The tests require no LM Studio, no
API keys, no live LLM, and do not run experiments.

```bash
python -m pytest tests/orchestration/test_ai_orchestration_demo.py -q
```
