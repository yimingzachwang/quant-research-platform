# AI Orchestration Website Doctrine

**Repository:** `quant-research-platform`
**Date:** 2026-05-30
**Scope:** Principles governing how the AI orchestration system is presented on the personal website

---

## Purpose

This document defines the positioning, tone, and boundaries for the AI orchestration section of the website. It should be treated as a constraint document for any copywriter or frontend implementation. No website copy or diagram should conflict with these principles.

---

## Core Thesis

The website must communicate one proposition clearly:

> The platform converts research intent into governed research actions, preserves human control over execution, then uses structured diagnostics and LLM-assisted interpretation to support the next research iteration.

This is a systems engineering and research infrastructure claim. It is not a claim about AI capability, autonomous research, or trading performance.

---

## Recommended Section Title

**Primary:**

```
Intent-Governed Research Orchestration
```

**Acceptable alternatives:**

```
Governed AI Research Orchestration
AI-Assisted Quant Research Workflow
Human-in-the-Loop Research Orchestration
```

**Forbidden alternatives:**

```
Autonomous AI Research
AI-Powered Strategy Discovery
The AI Runs the Research
```

---

## Positioning Rule

The AI orchestration section must appear **after**:

1. The platform overview
2. The canonical research showcase (ML results, diagnostics, reports)
3. The research architecture section

**The visitor must first understand that real quantitative experimentation is happening.** The ML showcase, walk-forward validation, feature engineering, and portfolio construction are the foundation. The AI orchestration layer sits on top of them — it is an interpretation and planning layer, not the engine.

If a visitor sees the AI orchestration section first, they will not have the context to evaluate it. They need to know what the quant engine produces before they can understand what the orchestration layer does with those outputs.

---

## What the AI Layer Is

The website should introduce the orchestration layer as:

| Function | Description |
|---|---|
| Planning support | Converts natural-language research requests into typed, routable research actions |
| Evidence interpretation | Assembles structured context from artefacts and generates governed LLM reviews |
| Proposal generation | Produces advisory iteration proposals grounded in named diagnostics |
| Research lineage support | Records how experiments relate to each other across iterations |

None of these functions involve the LLM running experiments, modifying results, or bypassing human review.

---

## Visual Doctrine

### One main lifecycle diagram

The website must use one lifecycle diagram that shows the complete research loop. The diagram must:

1. Show both the planning flow and the post-experiment review flow
2. Make the Human-Controlled Execution Boundary visually distinct — a separator, gate element, or horizontal band
3. Distinguish deterministic components from LLM-assisted components (colour, label, or border style)
4. Not look like a chatbot or conversational interface

**Required diagram flow:**

```
Natural-Language Research Request
        ↓
LLM / Rule-Based Intent Parser
        ↓
Typed Research Intent
        ↓
Research Planning / Routing
        ↓
Iteration Proposal
        ↓
Draft Generation
        ↓
Human Approval Gate
        ↓
Approved YAML Config
        ↓
[Human-Controlled Execution Boundary]
        ↓
Research Engine Run
        ↓
Research Artefacts
(metrics · diagnostics · plots · configs · metadata)
        ↓
Deterministic Context Assembly
        ↓
Failure Mode Detection
        ↓
LLM Review Engine
        ↓
Comparative Review / Iteration Proposal
        ↓
Research Evolution Chain
        ↻ (returns to Research Request)
```

**The diagram must not imply:**

```
LLM → direct experiment execution
```

### Three visual groups

The diagram should be readable as three conceptual groups:

| Group | Stages | Visual treatment |
|---|---|---|
| Research Intent & Planning | Request → Intent → Routing → Proposal → Draft → Approval → YAML | AI-Assisted badge (or equivalent) |
| Human-Controlled Execution | Boundary → CLI → Research Engine → Artefacts | Neutral / Deterministic |
| Evidence-Grounded Review | Context Assembly → Failure Detection → LLM Review → Comparative → Evolution | AI-Assisted badge + Governed label |

---

## Tone

| Use | Avoid |
|---|---|
| Restrained | Breathless |
| Technical | Vague |
| Institutional | Hype-driven |
| Specific | Aspirational |
| Evidence-referenced | Claim-only |

The strongest message on the website is not:

> "I connected an LLM to a quant system."

The strongest message is:

> I designed a governed research workflow where LLMs operate only on structured evidence, while execution, validation, approval, and lineage remain controlled and auditable.

Every sentence of copy should serve that message.

---

## Current Capability

The website may truthfully present the following as implemented and working:

- Natural-language request parsing (rule-based + LLM fallback, 9 intent types)
- Typed intent dispatch via `POST /api/route`
- Iteration proposal generation (advisory, evidence-grounded)
- Config draft synthesis (bounded vocabulary, approval-gated)
- Human approval gate (`approved=False` default, raises if bypassed)
- YAML config rendering with normalization and provenance hash
- Manual CLI execution by the researcher
- Experiment artefact generation (metrics, diagnostics, plots, metadata)
- Deterministic context assembly from persisted artefacts
- Rule-based failure mode detection (no LLM)
- LLM review with structured section extraction and provenance
- Comparative review with pre-computed deltas
- Research evolution chain (deterministic, from lineage records)
- Research sessions with event log

---

## Not Current Capability

The website must not present the following as implemented:

- Automatic experiment execution from generated YAML
- Autonomous research engine launch triggered by the orchestration layer
- Direct LLM control over backtests
- LLM mutation of results or artefacts
- Autonomous strategy discovery
- Multi-step research plan execution without human intervention

---

## Framing the Execution Boundary

The gap between YAML rendering and experiment execution is **not** a missing feature to apologise for. It is a governance boundary to explain with confidence.

**Use:**

> Human-controlled execution boundary

**Not:**

> Missing automation  
> Not yet implemented  
> Coming soon

The framing should position intentional human control as a design choice — one that makes the system safer, more auditable, and more credible in a quant research context.

Compare:

- Weak: "The system doesn't yet automate experiment runs."
- Strong: "Execution is intentionally human-controlled; the LLM cannot run experiments or mutate results."

The strong framing is factually identical to the weak framing, but it reads as a design claim, not an excuse.

---

## Safe Microcopy

The following phrases may appear anywhere on the website near the AI orchestration section:

> The LLM assists with interpretation and planning. It does not run experiments, modify results, or bypass approval.

> Execution is intentionally human-controlled; the AI layer works on structured research evidence.

> Every LLM output is linked to the experiment context that produced it via SHA256 provenance hash.

> Failure modes are detected deterministically before any LLM receives the context.

> Config changes proposed by the LLM are validated against a bounded vocabulary before the researcher reviews them.

---

## Acceptance Criteria

The AI orchestration section of the website is acceptable only when all of the following are true:

| Criterion | Required |
|---|---|
| The full lifecycle is visible in the diagram | Yes |
| The Human-Controlled Execution Boundary is explicit and labelled | Yes |
| The AI layer is presented as governed and advisory | Yes |
| The quant engine is presented as the source of truth | Yes |
| The section is understandable within 30 seconds | Yes |
| The demo and the diagram tell the same story | Yes |
| No unsupported automation claim appears anywhere | Yes |
| The section strengthens — not weakens — credibility of the quant platform | Yes |
| The section appears after the quant research showcase in page order | Yes |
