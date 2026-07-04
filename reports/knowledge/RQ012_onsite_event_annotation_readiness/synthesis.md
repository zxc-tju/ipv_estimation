# RQ012 Knowledge Synthesis

Status: consolidated from the frozen `decision.md` (ACCEPTED — scope revised by PI 2026-06-24: automatic-event readiness accepted, two-human blind annotation deprecated). `decision.md` is the canonical claim ledger.

## What was accepted (readiness)

- `RQ012-KC-READINESS`: Gates 012-0/012-1 pass; 012-2 surface-cleared; 012-3 ready — a blinded, outcome-free event design and extractor-readiness checks exist.
- `RQ012-KC-AUTOEVENTS`: the automatic event extractor (9 events; precedence/identity guards) is computable without humans, usable for event-aligned analysis (extractor health only, not outcomes).
- `RQ012-KC-CODEBOOK`: the codebook separates automatic, human-only, and removed events; construct-proximal labels are secondary.

## What was deprecated

Two real human blind labels + agreement (κ/AC1) are **deprecated** (Gate 012B closed as not-pursued). Human-only events as primary endpoints are dropped. Rationale: the program retains two stronger signals — WOD-E2E released human-preference scores and OnSite official rankings/scores/collisions/deductions — so a slow 2-annotator study was not worth its weight.

## Downstream effect

RQ012B (event-aligned harm) is reframed to **automatic events + OnSite official outcomes** (no human labels), removing the human-label dependency from RQ012B and RQ013. OnSite consequence evidence = automatic events + official collisions/scores/deductions (objective); do **not** claim an OnSite human-judgment convergent leg — human alignment is carried by WOD-E2E preference + InterHub. The RQ012B extractor-health execution result is recorded in the `## RQ012B …` section of `decision.md`; RQ012B is not separately registered and remains in this folder.

Sources: `decision.md`; `reviews/claude_review.md`; `reviews/codex_review.md`.
