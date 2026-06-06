# Zeto Operator Manual (LM Studio / local Qwen)

Compact operating rules for driving the Zeto AI-assisted research orchestration
tools from an LM Studio chat. Load this at the start of each session with the
read-only MCP tool `get_zeto_operator_manual` instead of pasting long rules.

First message in a new chat:

```text
Load the Zeto operator manual and reply READY. Do not call any other tools yet.
```

## Rules

1. Use exact snake_case MCP tool names. Do not camelCase tool names.
2. Show each tool's `display` field after every tool call.
3. Use compact tool outputs only. Reason from paths, hashes, and IDs.
4. Before major transitions, call `check_research_workflow_state` and follow its
   `next_suggested_action`.
5. Store the `session_id` UUID returned by `create_research_session`.
6. Pass `session_id` to all session-aware tools (review, proposal, draft,
   validate, approve, render, execute, post-run review, get_session_summary).
   If a session exists, always pass its `session_id` to approve/render/execute/
   post-run review — never pass `session_id=null` while an active session exists.
7. If `session_id` is lost, call `get_latest_research_session` before continuing.
8. Never use `experiment_name`, `context_hash`, `draft_id`, or `config_path` as
   `session_id`.
9. Never invent metrics. Use tool outputs only.
10. Never invent or hand-write draft configs. Drafts come from
    `generate_experiment_draft` only.
11. If a tool returns `ok=false`, report the failure verbatim and stop.
12. If validation fails, report the failure and stop. Do not repeatedly call
    `generate_experiment_draft`.
13. If validation fails because the proposed experiment name already exists, ask
    the user whether to use the next available suffix or clean existing local
    demo artefacts. Do not regenerate repeatedly.
14. Do not auto-approve drafts. Ask the user before approval. Approval requires
    `approval_confirmation="APPROVE"`. Validation success never authorises
    approval. Only a fresh user message that explicitly approves the draft (prefer
    "I approve the draft.") authorises `approve_experiment_draft` — never infer
    approval from "start a research cycle", validation success, or a
    `next_suggested_action`. After `approval_refused`, stop and wait for the user;
    do not retry automatically.
15. Do not render YAML unless the user approves the draft.
16. Do not execute unless the user explicitly provides `RUN`
    (`confirmation="RUN"`). Do not treat `execute`, `yes`, `proceed`, or
    `continue` as RUN — only a fresh user message containing the literal token
    `RUN` authorises execution. Never infer RUN from a previous approval, a
    previous render, or a prior failed/refused attempt. After `execution_refused`,
    stop and wait for a new user message; do not retry automatically.
17. Do not run extra experiments, optimise automatically, or loop.
18. Before review/proposal/draft, optionally call `retrieve_research_memory` for
    prior related evidence if relevant.
19. Research memory is evidence-only. Retrieved memory does not authorise
    execution or approval.
20. Do not use memory as proof of performance; quant metrics remain
    authoritative.
21. For local Qwen-backed review/proposal/draft calls, use provider `openai`,
    model `qwen2.5-7b-instruct`, base_url `http://127.0.0.1:1234/v1`
    (no real OpenAI API key needed).
22. Follow the governed sequence: create session → check state → execute
    baseline if needed → build context → review → proposal → draft → validate →
    approval → render YAML → RUN execution → post-run review → session summary.
23. The quant engine remains authoritative. Qwen coordinates and interprets only.

## Governance stopping rules

These are hard boundaries. A high-level request never carries through them.

- A high-level user request such as "start a research cycle" does **not**
  authorise approval, rendering, or execution. It only authorises the
  read/advisory steps (context, review, proposal, draft, validate).
- **Stop after validation** and ask the user for approval. Validation passing
  never authorises approval (`next_suggested_action: ask_user_for_approval`).
- **Approval is tool-enforced.** `approve_experiment_draft` requires
  `approval_confirmation="APPROVE"`; only a fresh user message that explicitly
  approves the draft authorises it. On `approval_refused`, stop and wait for the
  user — do **not** retry `approve_experiment_draft` automatically. The refusal's
  `next_suggested_action` is `ask_user_for_approval`, not `render_draft_to_yaml`.
- **Stop after YAML rendering** and ask the user for RUN. Rendering never
  authorises execution (`next_suggested_action: ask_user_for_execution_authorisation`).
- **On `execution_refused`** (confirmation was not the literal `RUN`), stop and
  wait for a fresh user message — do **not** retry `execute_approved_config`
  automatically. The refusal's `next_suggested_action` is
  `ask_user_for_execution_authorisation`, not `execute_approved_config`.
- Approve and render in one step only if the user explicitly asked for both in
  the same message; otherwise stop and ask before rendering.
- **Stop after the final session summary** — the cycle is complete
  (`next_suggested_action: stop_cycle_complete`).
- **Do not start a second iteration** (a new proposal, draft, or experiment)
  unless the user explicitly asks for one.

## Research Memory / Phase 1 RAG

A lightweight, controlled evidence layer. It indexes **compact summaries** of
existing Zeto artefacts (metadata, metrics, reviews, proposals, drafts, reports,
sessions) into a local JSONL index and retrieves prior research evidence by
deterministic keyword / metadata matching. Phase 1 only: no embeddings, no
vector database, no semantic retrieval, no agent framework.

Tools:

- `get_research_memory_status` — read-only; whether the index exists and how many
  items / experiments it holds.
- `index_research_memory` — controlled write; builds/refreshes the index from
  known artefact locations only. Runs no experiment, calls no LLM, approves/
  renders/executes nothing, mutates no source artefacts.
- `retrieve_research_memory` — read-only; retrieve by `query`, `experiment_name`,
  `failure_modes`, `artefact_type`, `top_k`. Returns compact items (summaries,
  paths, hashes, tags, matched terms) — never full artefacts.

Rules of use:

- Retrieval is **evidence-only**. It does not authorise execution or approval.
- Do not treat retrieved memory as proof of performance — quant metrics remain
  authoritative.
- Memory returns pointers; open the referenced artefact paths for full detail.

Example LM Studio prompts:

```text
Index the current Zeto research memory.
Have we seen this failure mode before?
Retrieve prior memory related to poor_oos_consistency and catastrophic_split.
Use retrieved memory to inform the next proposal, but do not execute anything.
```

## How to load

`get_zeto_operator_manual` is read-only: it returns this fixed manual (a short
rules list and this file's path) and nothing else. It executes nothing, calls no
LLM, inspects no experiment artefacts, creates no session, mutates no state, and
reads no arbitrary files.
