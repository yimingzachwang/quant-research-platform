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
7. If `session_id` is lost, call `get_latest_research_session` to recover it.
8. Never use `experiment_name`, `context_hash`, `draft_id`, or `config_path` as
   `session_id`.
9. Never invent metrics. Use tool outputs only.
10. Never invent or hand-write draft configs. Drafts come from
    `generate_experiment_draft` only.
11. If a tool returns `ok=false`, report the failure verbatim and stop.
12. Do not auto-approve drafts. Ask the user before approval.
13. Do not render YAML unless the user approves the draft.
14. Do not execute unless the user explicitly provides `RUN`
    (`confirmation="RUN"`).
15. Do not run extra experiments, optimise automatically, or loop.
16. For local Qwen-backed review/proposal/draft calls, use provider `openai`,
    model `qwen2.5-7b-instruct`, base_url `http://127.0.0.1:1234/v1`
    (no real OpenAI API key needed).
17. Follow the governed sequence: create session → check state → execute
    baseline if needed → build context → review → proposal → draft → validate →
    approval → render YAML → RUN execution → post-run review → session summary.
18. The quant engine remains authoritative. Qwen coordinates and interprets only.

## How to load

`get_zeto_operator_manual` is read-only: it returns this fixed manual (a short
rules list and this file's path) and nothing else. It executes nothing, calls no
LLM, inspects no experiment artefacts, creates no session, mutates no state, and
reads no arbitrary files.
