# AI Orchestration Website Specifications

**Repository:** `quant-research-platform`
**Date:** 2026-05-30
**Scope:** Section structure, copy, diagram requirements, and demo integration for the AI orchestration website section

---

## Section Identity

**Recommended section title:**

```
Intent-Governed Research Orchestration
```

**Recommended subtitle:**

```
A governed research workflow that converts natural-language research intent into structured
analysis, approved experiment configs, deterministic artefacts, and evidence-grounded LLM review.
```

**Section slug:** `/orchestration`

**Page position:** After the canonical research showcase. After the platform architecture section. This section must not be the entry point — visitors need to understand the quant engine first.

---

## 1. Intro Copy

```
The platform includes an intent-governed orchestration layer for quantitative research.
Natural-language requests are parsed into typed research actions, routed through constrained
planning and config-drafting stages, and converted into approved YAML experiment configs.
Execution remains human-controlled: the researcher explicitly runs the quant engine, which
produces metrics, diagnostics, plots, and metadata. Those artefacts are then assembled into
structured context for deterministic failure-mode detection, LLM-assisted review, comparative
analysis, iteration proposals, and research lineage tracking.
```

**Length:** ~80 words. Do not expand. The diagram carries the detail.

**Tone check:** This reads as a system description, not a product pitch. That is correct.

---

## 2. Main Lifecycle Diagram

### Required content

The diagram must render the following 15-stage flow. Every stage must be present. The Human-Controlled Execution Boundary must be visually distinct from all adjacent stages.

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
Human Approval Gate           ← gate element, visually distinct
        ↓
Approved YAML Config
        ↓
━━━━━━ Human-Controlled Execution Boundary ━━━━━━
        ↓
Research Engine Run
        ↓
Research Artefacts
(metrics · diagnostics · plots · configs · metadata)
        ↓
Deterministic Context Assembly
        ↓
Failure Mode Detection         ← deterministic, no LLM
        ↓
LLM Review Engine
        ↓
Comparative Review / Iteration Proposal
        ↓
Research Evolution Chain
        ↻
```

### Three visual groups

Implement three visually distinct groups using colour, border, or background treatment:

**Group A — Research Intent & Planning**

Stages: Natural-Language Request · Intent Parser · Typed Intent · Research Planning · Iteration Proposal · Draft Generation · Human Approval · YAML Config

Badge: `AI-Assisted`

**Group B — Human-Controlled Execution**

Stages: Human-Controlled Execution Boundary · Research Engine Run · Artefact Generation

Badge: `Human-Controlled` or no badge (neutral)

The boundary itself should be a horizontal separator element — a band, line, or gate icon — clearly labelled. Not a box node. A barrier.

**Group C — Evidence-Grounded Review**

Stages: Context Assembly · Failure Mode Detection · LLM Review Engine · Comparative Review · Evolution Chain

Badge: `AI-Assisted` + `Governed`

Failure Mode Detection should carry an additional sub-label: `Deterministic — no LLM`

### Desktop layout

Prefer a vertical single-column pipeline or a two-column segmented layout.

- Column A: Group A + boundary + Group B header
- Column B: Group C

Or: single vertical column with visual group separators between A, B, C.

Do not use a horizontal scroll pipeline — readability collapses on 1280px screens.

### Mobile layout

Vertical stacked timeline. Each stage is a card. The boundary is a full-width horizontal bar between Approved YAML Config and Research Engine Run. Group labels appear as section headers.

### Accessibility

- All stage nodes must have `aria-label`
- The boundary element must have `role="separator"` and `aria-label="Human-Controlled Execution Boundary"`
- The diagram must be readable without colour (use labels, not only colour, to distinguish groups)

---

## 3. Demo Video Embed

**Placement:** Directly below the lifecycle diagram, or in a tab/toggle alongside it.

**Aspect ratio:** 16:9

**Autoplay:** No. Poster frame required (show the lifecycle diagram or the terminal).

**Caption below the embed:**

```
Demo: from research request to approved config, human-controlled execution,
structured diagnostics, LLM review, and research evolution.
```

**The video must show the Human-Controlled Execution Boundary on screen.** This is the acceptance criterion for the embed. If a shorter cut cannot include it, the cut is too short.

**Fallback if video is not ready:** A static annotated screenshot sequence (carousel or scroll) following the same 14-scene structure defined in `AI_ORCHESTRATION_DEMO_SPECS.md`.

---

## 4. Capability Cards

Six cards arranged in a 2×3 or 3×2 grid on desktop; single column on mobile.

Each card: title + 1–2 sentence description. No bullets. No icons required.

---

### Card 1 — Intent Routing

**Title:** Intent Routing

**Copy:**

```
Natural-language research requests are parsed into typed intents and routed through
the orchestration API, not executed directly. Rule-based classification handles standard
phrasings without an LLM call.
```

---

### Card 2 — Constrained Draft Generation

**Title:** Constrained Draft Generation

**Copy:**

```
Iteration proposals can be converted into config drafts using a restricted change vocabulary.
Current parameter values are read from the base config by Python; the LLM only proposes
changes and provides rationale.
```

---

### Card 3 — Human Approval Gate

**Title:** Human Approval Gate

**Copy:**

```
Drafts start unapproved and cannot be rendered to YAML until the researcher explicitly
approves them. The gate cannot be bypassed — rendering raises an exception if approval
is missing.
```

---

### Card 4 — Deterministic Diagnostics

**Title:** Deterministic Diagnostics

**Copy:**

```
Metrics, validation outputs, feature summaries, plot metadata, and rule-based failure modes
are assembled from persisted artefacts before any LLM receives the context. Failure modes
are detected by threshold rules, not by the LLM.
```

---

### Card 5 — Evidence-Grounded Review

**Title:** Evidence-Grounded Review

**Copy:**

```
LLM reviews operate on structured experiment context assembled from real artefacts.
The template enforces citation of named diagnostics, failure modes, and feature names;
generic advice is explicitly forbidden by the prompt constraints.
```

---

### Card 6 — Research Lineage

**Title:** Research Lineage

**Copy:**

```
Comparison reviews and lineage records capture how experiments evolve across hypotheses,
configs, and diagnostics. Evolution chains are assembled deterministically from lineage
artefacts — no LLM call is required.
```

---

## 5. Governance Controls Panel

**Panel title:** Governance Controls

**Panel style:** Small panel below the capability cards. Monospace or table treatment. Not a feature list — a controls list.

**Content:**

```
LLM output is advisory only
Quant engine remains the source of truth
No autonomous experiment execution
Context hashes link every review to its evidence snapshot
Config drafts require explicit researcher approval
Failure modes are detected deterministically before LLM review
Comparative deltas are precomputed in Python before LLM receives the prompt
Research lineage is human-registered, not automated
Prompt templates use StrictUndefined — broken context raises before LLM call
```

**Do not style this as a feature list.** It is a constraint list. Present it as such. A quant hiring manager or ML systems engineer reading this should feel confident that the LLM is contained.

---

## 6. Safe Microcopy

The following phrases may appear in callouts, captions, tooltips, or transition text anywhere in the section:

**Near the execution boundary:**

```
The LLM assists with interpretation and planning.
It does not run experiments, modify results, or bypass approval.
```

**Near the review section:**

```
Execution is intentionally human-controlled;
the AI layer works on structured research evidence.
```

**Near the provenance badge (if shown):**

```
Every LLM output is linked to the experiment context that produced it
via SHA256 provenance hash.
```

**Near the failure mode detection node:**

```
Failure modes are detected by deterministic rules —
no LLM involvement at this stage.
```

**Near the draft generation card:**

```
Config changes proposed by the LLM are validated against a bounded vocabulary
before the researcher reviews them.
```

---

## 7. Claims Allowed

The website section may say:

- "Natural-language requests can be parsed into typed research actions."
- "The system supports governed research planning and config drafting."
- "Experiment execution remains human-controlled."
- "Generated YAML configs are reviewed and run by the researcher."
- "Research artefacts are assembled into structured LLM context."
- "Failure modes are detected deterministically before LLM review."
- "LLM reviews are persisted with SHA256 provenance hashes."
- "Research evolution chains track experiment lineage across iterations."
- "Config drafts require explicit human approval before rendering."
- "The LLM receives pre-computed numeric deltas in comparative reviews."

---

## 8. Claims Forbidden

The website must not say:

- "The AI runs experiments automatically."
- "The LLM executes the quant engine."
- "The system autonomously discovers strategies."
- "The model validates performance."
- "The platform is a fully autonomous quant researcher."
- "The AI controls the research pipeline end-to-end."
- "Results are AI-verified."
- "The system learns from results." (without specifying what this means technically)

---

## 9. Acceptance Criteria

The section is acceptable for publication only when all of the following are verified:

| Criterion | Pass condition |
|---|---|
| Full lifecycle visible in diagram | All 15 stages present |
| Human-Controlled Execution Boundary | Visually labelled, between YAML and Engine Run |
| AI layer framed as governed and advisory | No autonomous claims anywhere |
| Quant engine as source of truth | Stated in intro copy or diagram label |
| Readable within 30 seconds | Intro copy ≤ 100 words; diagram scannable |
| Demo and diagram consistent | Same lifecycle, same boundary position |
| No unsupported automation claim | Manual review of all copy |
| Strengthens quant platform credibility | Section appears after the ML showcase |
| Mobile layout functions | Vertical timeline, boundary bar full-width |
| Accessibility | Aria labels, non-colour group differentiation |

---

## 10. Frontend Implementation Notes

These notes are for the eventual frontend implementer. They do not represent current implementation.

**Diagram component recommendations:**

- Build the lifecycle diagram as a custom React component, not an embedded SVG image, so that it can be maintained without a design tool.
- Use a data-driven approach: define stages as an array of objects `{id, label, group, badge, isDeterministic}` and render them with a map. This makes updating the diagram straightforward if stages are added.
- The Human-Controlled Execution Boundary should be a separate component rendered between the Group A and Group B stages — not a node in the stage array.

**Capability cards:**

- 2×3 grid, CSS Grid. No carousel. Cards should be statically rendered — no hover-reveal copy.

**Governance controls panel:**

- Consider a `<dl>` (definition list) or a simple `<ul>` with no bullets. Present as a specification list, not a marketing list. Font should be slightly smaller than body text.

**Demo embed:**

- Use a native `<video>` element with `controls`, `preload="metadata"`, and a `poster` attribute. No autoplay, no looping.

**Performance:**

- The lifecycle diagram should not require JavaScript to render its structure. Interactivity (hover states, group highlighting) is optional.
- The diagram must be server-side renderable for LCP.
