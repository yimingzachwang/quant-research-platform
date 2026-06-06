# AI Orchestration Demo Doctrine

**Repository:** `quant-research-platform`
**Date:** 2026-05-30
**Scope:** Principles governing any canonical demo of the AI orchestration system

---

## Purpose

This document defines the principles for presenting the AI orchestration system in any demo context — video, live walkthrough, or embedded media. The doctrine must hold across all formats.

The demo must convey one thesis:

> The platform is a governed quantitative research workflow. The LLM assists with planning and interpretation. Execution, validation, approval, and lineage remain human-controlled and auditable.

Every scene, caption, and transition must serve this thesis or be cut.

---

## Principle 1 — Evidence Before Interpretation

The demo must show that experiment artefacts exist before any LLM is involved. Metrics, diagnostics, plots, configs, and provenance records are produced by the quant engine and persisted to disk. The LLM receives structured summaries of those artefacts — it does not produce them, invent them, or recompute them.

**In practice:** Show the artefact tree (`results/experiments/<name>/`) before showing any LLM output. Show context assembly (`build_llm_context`) before showing the review. Make the dependency explicit.

**Forbidden:** Opening the demo with LLM output. Implying the LLM discovered the metrics.

---

## Principle 2 — Intent Before Action

Natural-language input may initiate research planning, but the demo must show the conversion step. A typed intent — a frozen dataclass — is the object that the router acts on, not the raw text string. This conversion is the first safeguard.

**In practice:** Show the parse step and its output (`ReviewExperimentIntent`, `GenerateIterationIntent`, etc.) before showing any downstream result.

**Suggested caption:**

> Natural language is converted into typed research actions before any analysis is performed.

**Forbidden:** Showing natural-language input connected directly to LLM output with no intermediate step visible.

---

## Principle 3 — No Autonomous Execution Claim

The demo must never imply that the LLM runs experiments. The orchestration layer has no subprocess call, no `run_experiment_from_config()` invocation, and no `RunExperimentIntent`. The YAML config is the final output of the orchestration layer. Running it is a human action.

**In practice:** Every demo must include the explicit statement — verbally, in caption, or on screen:

> The generated YAML is reviewed and executed by the researcher.

**Forbidden:** Any transition that shows YAML config → running experiment without a visible human-controlled execution boundary between them.

---

## Principle 4 — Human-Controlled Execution Boundary

The demo must visibly mark the manual transition between the orchestration layer and the quant engine. This is not an error or a gap — it is a governance feature. Mark it explicitly.

**Required transition in every full-lifecycle demo:**

```
Approved YAML Config
        ↓
[Human-Controlled Execution Boundary]
        ↓
Research Engine Run
```

The boundary label must appear on screen or in narration. Do not soft-pedal it. Frame it as intentional governance.

**Suggested caption at this transition:**

> Execution is intentionally human-controlled. The orchestration layer does not run the quant engine automatically.

---

## Principle 5 — Deterministic Core, LLM Interpretation

The demo should make a clear architectural distinction:

| Component | Character |
|---|---|
| Quant engine, context assembly, failure mode detection, delta computation, draft validation, YAML rendering | Deterministic or rule-governed |
| LLM review, comparative review, iteration proposal, intent fallback classification | Advisory and interpretive |

The deterministic components are the foundation. The LLM components are the interpretation layer on top.

**In practice:** When showing failure mode detection, state that it uses rule-based thresholds. When showing comparative review deltas, state they are precomputed in Python before the LLM call. Do not present these as LLM outputs.

---

## Principle 6 — Research Loop, Not Chatbot

The system is a research lifecycle. It is not a conversational chatbot. The demo structure should mirror the research lifecycle, not a Q&A session.

**Required narrative arc:**

```
Research intent
→ planning
→ approved config
→ human-run experiment
→ artefacts
→ diagnostics
→ AI review
→ proposal
→ lineage
```

**Forbidden framing:** "I asked the AI and it said..." / "The AI recommended..." without showing the structured pipeline that produced the output.

---

## Principle 7 — Governance as a Strength

The human execution boundary, the approval gate, the draft vocabulary restriction, and the context hash provenance are not limitations to hide. They are the strongest design claims in the system.

The demo should say or imply:

> Execution is intentionally human-controlled; the LLM cannot run experiments or mutate results. LLM output is advisory. Provenance is preserved throughout.

An interviewer at a quant firm will ask: "How do you prevent the LLM from doing something wrong?" The answer is visible in the demo: approval gates, deterministic diagnostics, bounded change vocabulary, no execution authority.

---

## Principle 8 — Portfolio Objective

The demo exists to prove skills. Every scene should contribute evidence of:

| Skill | What demonstrates it |
|---|---|
| Quantitative research infrastructure | Artefact tree, diagnostics, reports, provenance records |
| ML experiment governance | Walk-forward validation, failure mode detection, split diagnostics |
| LLM system design | Jinja2 templates with StrictUndefined, context hashing, bounded output schema |
| Reproducible research | Config versioning, experiment hash, manifest provenance |
| Human-in-the-loop workflow | Approval gate, review-before-execution, manual lineage registration |
| Artefact-based documentation | Research evolution chain, comparative review, persisted JSON |

The demo should not show any of these in isolation. Show how they connect.

---

## Forbidden Demo Claims

The following must never appear in narration, captions, or on-screen text:

| Forbidden claim | Why |
|---|---|
| "The AI runs experiments automatically." | False. No subprocess call exists in orchestration. |
| "The LLM discovers profitable strategies." | False. The LLM interprets pre-computed diagnostics. |
| "The platform is a fully autonomous quant researcher." | False. Human controls execution, approval, and lineage. |
| "The system executes trades." | Out of scope entirely. |
| "The LLM validates performance." | False. Failure modes are deterministic. Validation is rule-based. |
| "The LLM directly modifies the research engine." | False. The coupling test structurally prevents this. |

---

## Preferred Language

| Use | Avoid |
|---|---|
| intent-governed | AI-powered |
| evidence-grounded | AI-driven |
| human-controlled | automated |
| advisory | autonomous |
| deterministic context assembly | AI analysis |
| research artefact lineage | AI memory |
| LLM-assisted interpretation | AI decisions |
| governed research loop | AI pipeline |
| constrained draft generation | AI config |
| rule-based failure detection | AI diagnostics |

---

## Canonical Demo Length

**Target:** 90–150 seconds for video.

- 0–20s: Research request → intent parsing
- 20–45s: Planning output (proposal → draft → approval → YAML)
- 45–55s: Human-Controlled Execution Boundary + CLI run (visible)
- 55–90s: Artefacts → context assembly → failure modes → LLM review
- 90–120s: Comparative review or evolution chain
- 120–150s: Closing loop diagram

Scenes 7–10 (artefacts through review) may be condensed for a shorter cut. Scenes 4–6 (draft, approval, YAML, boundary) must never be cut — they carry the governance narrative.
