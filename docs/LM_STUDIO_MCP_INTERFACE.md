# LM Studio MCP Interface

Use **LM Studio** (with a local Qwen model) as the chat interface for the Zeto
AI-assisted research platform, via a minimal MCP server that exposes the
existing governed Research API as tools.

```text
LM Studio chat → local Qwen → MCP tool calls → Zeto MCP server
              → Zeto Research API
              → review / proposal / draft / validate / approve / render
              → optional confirmed execution → post-run review
              → results shown in LM Studio
```

Server: [`src/mcp/zeto_server.py`](../src/mcp/zeto_server.py)

---

## Purpose

Let a researcher drive the full governed research workflow from an LM Studio
chat — while the platform keeps enforcing validation, explicit approval, and
human-confirmed execution. This is an **interface layer**: it adds no new
orchestration behaviour and wraps the Research API only.

## Architecture

```text
LM Studio  ──MCP (stdio)──▶  src/mcp/zeto_server.py
                                    │  (wraps only)
                                    ▼
                    src/orchestration/api/research_api.py
                                    │  (reads artefacts; one governed exec path)
                                    ▼
                         Zeto quant research engine (authoritative)
```

The model can only call the explicitly exposed Zeto tools. It has **no** shell
access, **no** file/eval access, and cannot import or call quant-engine
internals.

## Compact MCP Responses for LM Studio

LM Studio has a limited chat context (~32k tokens for local Qwen). When the same
local model is **both** the chat operator and the backend review/proposal/draft
provider (`provider=openai`, `base_url=http://127.0.0.1:1234/v1`), large tool
outputs accumulate in the conversation and the next tool call can overflow the
window.

To prevent this, **MCP tools return compact state summaries only**:

- Full artefacts (reviews, proposals, drafts, configs, reports, plots, run
  outputs) stay **persisted on disk** at full fidelity — nothing is reduced
  there.
- `data` carries only essentials: IDs, hashes, paths, booleans, counts, short
  failure-mode names/severities, short flags, a compact config diff, and next
  state indicators.
- Full bodies are referenced by path/hash/ID (`review_path`, `proposal_path`,
  `draft_path`, `config_path`, `artefact_root`, `report_path`, `plots_dir`,
  `context_hash`).
- `context_hash` is kept **in full** in `data` (it is a small provenance link
  between context → review → proposal → draft); the `display` shows only its
  first 8 characters (e.g. `hash 83765162`).

The model should reason from these summaries and references. If a full artefact
is genuinely needed, open it on disk **outside** LM Studio (the paths are in the
responses) rather than asking a tool to inline it. Compact is the only mode in
the recorded demo.

Example compact `build_context_summary` response:

```json
{
  "ok": true,
  "stage": "context_built",
  "display": "Context for canonical_ml_showcase built (hash 83765162). Failure modes: [CRITICAL] poor_oos_consistency; [WARNING] high_split_sharpe_variance. Validation: mean_oos_sharpe=-0.32, std_oos_sharpe=1.274, n_splits=7.",
  "data": {
    "experiment_name": "canonical_ml_showcase",
    "context_hash": "83765162d7d91cadbfae8450cb1da3bdcb4c23e721253eff9587a885ff451b99",
    "failure_modes": [
      {"name": "poor_oos_consistency", "severity": "critical"},
      {"name": "high_split_sharpe_variance", "severity": "warning"}
    ],
    "key_metrics": {
      "sharpe_ratio": 0.695, "max_drawdown_pct": "-34.10%",
      "mean_oos_sharpe": -0.32, "std_oos_sharpe": 1.274,
      "n_splits": 7, "n_negative_sharpe_splits": 4
    }
  },
  "next_suggested_action": "run_experiment_review"
}
```

## Install

The MCP SDK is an optional extra:

```bash
pip install -e ".[mcp]"     # or: pip install "mcp>=1.2"
```

## Setup

1. Open LM Studio.
2. Load a local Qwen instruct model and start the OpenAI-compatible local
   server (Developer ▸ Local Server). Confirm it is up:
   ```bash
   curl http://127.0.0.1:1234/v1/models
   ```
3. Enable MCP in LM Studio and add the `zeto` server to `mcp.json` (below).
4. Start a new chat with tools enabled — the `zeto` tools should appear.
5. Paste the suggested system prompt (below) and drive the workflow.

The same `provider="openai"` + `base_url` path drives review, proposal, draft,
and post-run review — there is **no** separate `lmstudio` provider.

## LM Studio `mcp.json`

```json
{
  "mcpServers": {
    "zeto": {
      "command": "python",
      "args": [
        "/Users/zachmac/Documents/quant-research-platform/src/mcp/zeto_server.py"
      ],
      "cwd": "/Users/zachmac/Documents/quant-research-platform",
      "env": {
        "OPENAI_API_KEY": "lm-studio"
      }
    }
  }
}
```

(Adjust the path if your checkout lives elsewhere. Do not put real secrets here —
local LM Studio accepts any placeholder key.)

## Start each chat by loading the operator manual

Instead of pasting the long rules into every chat, begin a new LM Studio session
with a single message that loads the compact operator manual via the read-only
`get_zeto_operator_manual` tool:

```text
Load the Zeto operator manual and reply READY. Do not call any other tools yet.
```

The model calls `get_zeto_operator_manual` once (it returns the fixed rules — no
LLM, no execution, no file reads, no state change), acknowledges with `READY`,
and then waits. After that, normal prompts can be natural:

- "Start a research cycle for canonical_ml_showcase."
- "Review it and propose one controlled improvement."
- "Generate a draft and show the diff."
- "Validate it."
- "I approve it. Render the YAML."
- "I authorise execution. RUN."
- "Review the new result and summarise the session."

The full manual lives at
[`docs/LM_STUDIO_QWEN_OPERATOR_MANUAL.md`](LM_STUDIO_QWEN_OPERATOR_MANUAL.md);
the tool returns a compact rules summary (under 4000 characters). The system
prompt below remains available if you prefer to paste rules explicitly.

## Suggested LM Studio system prompt

```text
You are operating the Zeto AI-assisted research orchestration tools.

Follow these rules:
1. Do not claim to execute experiments unless the execute_approved_config tool succeeds.
2. Do not call execute_approved_config unless I explicitly provide confirmation RUN.
3. Do not auto-approve drafts. Ask me before approval.
4. Do not invent metrics. Use tool outputs only.
5. Session IDs: create_research_session returns a session_id (a UUID). Store it
   and pass that exact UUID as session_id to every session-aware tool
   (run_experiment_review, generate_iteration_proposal, generate_experiment_draft,
   validate_experiment_draft, approve_experiment_draft, render_draft_to_yaml,
   execute_approved_config, review_post_run_result, get_session_summary). Never
   use an experiment_name, context_hash, or draft_id as a session_id. If you have
   lost the session_id, call get_latest_research_session (or list_research_sessions)
   to recover it before continuing.
6. Never invent or hand-write a draft config. If a tool returns ok=false (for
   example generate_experiment_draft), report the tool's error verbatim and stop.
   Do not substitute a placeholder config (no learning_rate, batch_size, layers, etc.).
7. For LLM-backed tools against local Qwen in LM Studio, always pass
   provider="openai", model="<exact LM Studio model id>", and
   base_url="http://127.0.0.1:1234/v1". Never ask me for a real OpenAI API key.
8. Before every major transition (build context, review, proposal, draft,
   validate, approve, render, execute, post-run review), call
   check_research_workflow_state first and follow its next_suggested_action.
   Do not validate/approve/render before the required artefacts exist, and do
   not confuse identifiers: context_hash is NOT a session_id and NOT a draft_id.
9. Show each tool's `display` field. Use compact tool outputs only. Do not ask
   tools to return full artefacts unless debugging. Reason from paths, hashes,
   IDs and summaries; if more detail is needed, ask me before requesting full
   artefacts.
10. The normal workflow is:
   list experiments → create session (store session_id) → build context → review → proposal → draft → validate → ask approval → approve → render YAML → ask execution authorisation → execute → review post-run result.
11. The quant engine remains authoritative. You interpret and coordinate only.
```

When calling the LLM-backed tools (`run_experiment_review`,
`generate_iteration_proposal`, `generate_experiment_draft`,
`review_post_run_result`) against LM Studio, you **must** pass all three:

```json
{ "provider": "openai", "model": "<exact LM Studio model id>", "base_url": "http://127.0.0.1:1234/v1" }
```

- `model` must be the **exact** model id shown in LM Studio's server (the
  `/v1/models` list); a wrong id fails the call.
- When `base_url` is set, **no real OpenAI API key is required** — LM Studio
  accepts the `lm-studio` placeholder. Do not paste a real OpenAI key.
- `provider="stub"` runs fully offline (no model, no network). In stub mode,
  `generate_experiment_draft` returns a deterministic, schema-valid draft
  (`model.params.alpha -> 1.0`) so the workflow is demoable without a model.

## Tools

| Tool | Purpose | LLM? | Mutates? |
|---|---|---|---|
| `get_zeto_operator_manual` | Load the fixed operator rules (read-only) | no | no |
| `get_research_memory_status` | Whether the memory index exists + item/experiment counts | no | no |
| `index_research_memory` | Build/refresh the memory index from known artefacts only | no | memory index |
| `retrieve_research_memory` | Retrieve prior evidence by keyword/metadata (compact) | no | no |
| `list_experiments` | List experiments with artefacts | no | no |
| `create_research_session` | Start a session | no | session |
| `get_session_summary` | Current session state (needs session_id UUID) | no | no |
| `list_research_sessions` | List session UUIDs (recover a lost session_id) | no | no |
| `get_latest_research_session` | Most recent session UUID + summary (recovery) | no | no |
| `check_research_workflow_state` | Read-only preflight: which artefacts exist + next step | no | no |
| `build_context_summary` | Deterministic structured context + failure modes | no | no |
| `run_experiment_review` | LLM review of diagnostics | yes | session event |
| `generate_iteration_proposal` | Advisory next-step proposal | yes | session event |
| `generate_experiment_draft` | Schema-bounded config draft (deltas, not YAML) | yes | draft + event |
| `validate_experiment_draft` | Validate draft against schema | no | session event |
| `approve_experiment_draft` | **Explicit** approval (only here); needs `approval_confirmation="APPROVE"` | no | draft + event |
| `render_draft_to_yaml` | Render approved draft to YAML (no execution) | no | config + event |
| `execute_approved_config` | Run ONE approved config; needs `confirmation="RUN"` | no | run + events |
| `review_post_run_result` | LLM review of freshly generated artefacts | yes | session event |

Governance built into the tools:

- approval is its own tool — no other tool auto-approves;
- rendering is separate from execution;
- `execute_approved_config` refuses unless `confirmation="RUN"`, runs exactly
  one config, and never loops, retries, trades, or registers lineage;
- LLM-backed tools interpret persisted artefacts — they never compute metrics or
  run experiments.

## Research Memory / Phase 1 RAG

A lightweight, controlled **evidence layer** so a chat can recall prior Zeto
research before reviewing, proposing, drafting, or interpreting a post-run
result.

```text
research artefacts → compact memory records → local JSONL index
                   → retrieval tool → compact evidence snippets to LM Studio
```

Phase 1 scope (deliberately minimal): metadata / keyword retrieval only — **no**
embeddings, **no** vector database, **no** LangChain/LangGraph/agent framework,
**no** semantic retrieval. Full artefacts stay on disk; memory holds only
compact, provenance-aware pointers.

Index file: `results/research_memory/memory_index.jsonl` (one record per line).

Indexed sources (and the assigned `artefact_type`):

| Source | artefact_type |
|---|---|
| `results/experiments/*/metadata.json` | `experiment_metadata` |
| `results/experiments/*/metrics.json` | `experiment_metrics` |
| `results/llm_reviews/*/llm_review.json` | `llm_review` |
| `results/llm_reviews/*/iteration_proposal.json` | `iteration_proposal` |
| `results/llm_reviews/*/draft_*.json` | `draft` |
| `reports/markdown/*.md` | `report` |
| `results/research_sessions/*/session.json` | `session` |

**Not indexed:** raw data, parquet files, plot binaries, secrets/`.env`,
arbitrary repo files, and full report bodies (only a short summary line is kept).

Each record is compact:

```json
{
  "memory_id": "mem_…",
  "experiment_name": "canonical_ml_showcase_v2",
  "session_id": null,
  "artefact_type": "llm_review",
  "context_hash": "…",
  "path": "results/llm_reviews/canonical_ml_showcase_v2/llm_review.json",
  "created_at": "…",
  "failure_modes": ["poor_oos_consistency", "catastrophic_split"],
  "tags": ["validation", "oos_consistency", "regularisation"],
  "short_summary": "LLM review flagged: poor_oos_consistency, catastrophic_split."
}
```

Tools:

- **`get_research_memory_status`** — read-only. Reports whether the index exists
  and how many items / experiments it holds.
- **`index_research_memory`** — controlled write. Builds/refreshes the index from
  the known locations above. It runs no experiment, calls no LLM, approves no
  draft, renders no YAML, never calls RUN, inspects no arbitrary path, and never
  mutates source artefacts.
- **`retrieve_research_memory`** — read-only. Retrieve by `query`,
  `experiment_name`, `failure_modes`, `artefact_type`, and `top_k`. Returns up to
  `top_k` compact items (summaries, paths, hashes, tags, **matched terms**) —
  never full artefact contents.

Retrieval is simple and deterministic: case-insensitive keyword matching,
failure-mode matching, artefact-type matching, experiment-name matching, and a
small additive score over the matched fields. `experiment_name` matches by
containment, so a base name (`canonical_ml_showcase`) also surfaces its versioned
descendants (`canonical_ml_showcase_v2`).

Governance: retrieval is **evidence-only** — retrieved memory does **not**
authorise execution or approval, and is never proof of performance. The quant
metrics remain authoritative.

Example retrieve input:

```json
{
  "query": "poor OOS consistency and catastrophic split",
  "experiment_name": "canonical_ml_showcase",
  "failure_modes": ["poor_oos_consistency", "catastrophic_split"],
  "top_k": 5
}
```

Example LM Studio prompts:

```text
Index the current Zeto research memory.
Have we seen this failure mode before?
Retrieve prior memory related to poor_oos_consistency and catastrophic_split.
Use retrieved memory to inform the next proposal, but do not execute anything.
```

## Preflight: `check_research_workflow_state`

A local Qwen chat can lose track of how far the workflow has progressed and call
tools out of order (e.g. validating a draft before one exists, or confusing a
`context_hash` with a `session_id`). `check_research_workflow_state` is a
read-only preflight that inspects the on-disk state and reports the recommended
next step. It executes nothing.

```json
{ "experiment_name": "canonical_ml_showcase" }
```

Returns the standard envelope; `data` includes `baseline_artefacts_exist`,
`metadata_exists`, `metrics_exists`, `context_ready`, `review_exists`,
`proposal_exists`, `draft_exists`, `latest_draft_id`, `latest_draft_approved`,
`proposed_name`, `rendered_yaml_exists`, `rendered_yaml_path`,
`revised_artefacts_exist`, `report_path`, and `plots_dir`. Example display:

```text
[canonical_ml_showcase] proposal exists; draft missing. Next suggested action: generate_experiment_draft.
```

**Call it before every major transition** and follow `next_suggested_action`.
It does not run, approve, render, or execute anything — it only reports state.

## Tool response contract

Every tool returns the same visible-state envelope so LM Studio can show each
intermediate state clearly:

```json
{
  "ok": true,
  "stage": "draft_validated",
  "display": "Validation PASS — approval still required. Stop and ask the user before approving or rendering.",
  "data": { "experiment_name": "canonical_ml_showcase", "draft_id": "…", "is_valid": true, "rendering_blocked": false, "error_count": 0, "errors": [] },
  "next_suggested_action": "ask_user_for_approval"
}
```

- `ok` — success flag (`false` on errors and on a refused execution);
- `stage` — machine-readable lifecycle stage (e.g. `context_built`, `yaml_rendered`);
- `display` — a concise human-readable line to surface in the chat;
- `data` — the structured payload;
- `next_suggested_action` — the single recommended next step. This is **advisory
  only** — there is no auto-chaining and no all-in-one loop tool; the researcher
  (or model, with your confirmation) chooses each step.

### Governance stops (non-tool sentinels)

At each governance boundary, `next_suggested_action` is a **non-tool sentinel**,
not the next mutating tool — the model must stop and get the user before
proceeding:

| After | `next_suggested_action` | Meaning |
|---|---|---|
| `validate_experiment_draft` (PASS) | `ask_user_for_approval` | validation never authorises approval |
| `approve_experiment_draft` **refused** (`approval_confirmation` ≠ `APPROVE`) | `ask_user_for_approval` | stop — do **not** retry; wait for the user to explicitly approve |
| `approve_experiment_draft` (`APPROVE`) | `ask_user_to_render_yaml` (default) / `render_draft_to_yaml` (only if the user asked to approve **and** render in one message via `render_requested=true`) | approval does not auto-render |
| `render_draft_to_yaml` | `ask_user_for_execution_authorisation` | rendering never authorises execution |
| `execute_approved_config` **refused** (confirmation ≠ `RUN`) | `ask_user_for_execution_authorisation` | stop — do **not** retry; wait for a fresh user message containing `RUN` |
| `execute_approved_config` (RUN) | `review_post_run_result` | post-run review may proceed |
| `review_post_run_result` | `get_session_summary` | summarise the cycle |
| `get_session_summary` | `stop_cycle_complete` | cycle done — do **not** start a new proposal/draft/experiment |

A high-level request such as "start a research cycle" does **not** carry through
these stops: it authorises only the read/advisory steps up to validation.

Approval is tool-enforced the same way: `approve_experiment_draft` requires
`approval_confirmation="APPROVE"` and only a fresh user message that explicitly
approves the draft (e.g. "I approve the draft.") authorises it. Validation
success, "start a research cycle", and `next_suggested_action` do **not** count.
When approval is refused (`approval_refused`), the model must stop and wait for
the user; it must not retry `approve_experiment_draft` or jump to rendering.

Execution still requires `confirmation="RUN"`. Only a fresh user message
containing the literal token `RUN` authorises execution — `execute`, `yes`,
`proceed`, `continue`, a prior approval/render, or a previously refused attempt
do **not** count. When execution is refused, the model must stop and wait for a
new RUN message; it must not immediately retry `execute_approved_config`.

Notable displays: `build_context_summary` shows failure modes and key validation
metrics; `generate_experiment_draft` shows the proposed config diff;
`validate_experiment_draft` shows PASS/FAIL and (on PASS) that approval is still
required; `render_draft_to_yaml` states execution has NOT occurred and to stop
for RUN; `review_post_run_result` shows the post-run context hash, flags, and
sections.

## Example User Workflow (interpret an existing result)

```text
User: List available experiments.

User: Create a research session for canonical_ml_showcase.
      Goal: investigate whether validation robustness can improve.

User: Build context and review the experiment.

User: Generate an iteration proposal.

User: Generate and validate a draft.

User: Show me the proposed changes.

User: I approve the draft.

User: Render the YAML.

User: I authorise execution. RUN.

User: Review the post-run result.
```

The model maps these to: `list_experiments` → `create_research_session` →
`build_context_summary` + `run_experiment_review` → `generate_iteration_proposal`
→ `generate_experiment_draft` + `validate_experiment_draft` → (asks you) →
`approve_experiment_draft` → `render_draft_to_yaml` → (asks you) →
`execute_approved_config` with `confirmation="RUN"` → `review_post_run_result`.

## Baseline-from-scratch Workflow

When there are no artefacts yet, run the baseline config first, then iterate to a
v2. The full demo path is:

```text
create session
→ execute baseline config with RUN          (execute_approved_config, confirmation="RUN")
→ build context                              (build_context_summary)
→ review                                     (run_experiment_review)
→ proposal                                   (generate_iteration_proposal)
→ draft                                      (generate_experiment_draft)
→ validate                                   (validate_experiment_draft)
→ approve                                    (approve_experiment_draft)
→ render                                     (render_draft_to_yaml)
→ execute v2 with RUN                        (execute_approved_config, confirmation="RUN")
→ list figures                              (see note below)
→ post-run review                            (review_post_run_result)
→ session summary                            (get_session_summary)
```

Notes:

- "execute baseline config" runs the existing baseline YAML
  (`configs/experiments/canonical_ml_showcase.yaml`) — it is still gated on
  `confirmation="RUN"`.
- "list figures" is not a separate tool. Each successful execution returns the
  experiment's `artefact_root` in `data.execution`; figures live under
  `<artefact_root>/plots/` (and the rendered report at `report_path`). Inspect
  those paths directly — the MCP layer adds no figure-listing tool.
- The v2 path still passes through validate → approve → render before any
  execution; nothing is auto-approved or auto-run.

## What This Interface Does Not Do

- no arbitrary shell access
- no source-code editing
- no arbitrary Python eval/exec
- no autonomous loops or multi-run optimisation
- no execution without `confirmation="RUN"`
- no live trading, broker, or portfolio execution
- no automatic alpha discovery
- no automatic lineage registration

The default tool flow stops at rendered YAML; execution is a separate,
explicitly confirmed step. The quant engine remains authoritative.

## Tests

[`tests/mcp_server/test_zeto_mcp_server.py`](../tests/mcp_server/test_zeto_mcp_server.py)
covers tool registration, Research-API delegation, the `confirmation="RUN"`
gate, no implicit approval/rendering, and the non-coupling guarantees. It
requires no LM Studio, no API keys, no live LLM, and runs no experiments.

```bash
python -m pytest tests/mcp_server/ -q
```
