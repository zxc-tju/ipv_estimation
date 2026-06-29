# RQ009 Decision: Estimability-Aware Context-Conditioned Conformal Envelope (M3)

Status: ACCEPTED — R3 context-conditioned envelope (knowledge-layer freeze, PI-directed 2026-06-29). The IPV-conditioning channels (counterpart-IPV, ego self-IPV) are internal ablations (null) and are NOT manuscript claims.

Run ID: `RQ009_1_dynamic_envelope_20260625T121905Z_98c433de`
Basis: study COMPLETE, all gates PASS; formal M3-vs-M4 gate `ESCALATE_TO_PI=false`; `reviews/claude_review.md`; PI acceptance 2026-06-29.

## Accepted Claim

| ID | Claim |
|---|---|
| RQ009-KC-R3 | An estimability-aware, **context-conditioned**, split-conformal dynamic envelope is sharp and near-nominal: context (M2) vs global floor (M0) at 90% gives width −42.3% and Winkler −35.6% at coverage ≈0.899; abstention 4.78%; directional tails pass. It is an empirical runtime monitor that excludes observed risk/PET from online inputs and abstains out of support. |

## Internal Ablations (not manuscript claims — PI: internal trial-and-error)

The IPV-conditioning channels add no measurable value over the context-only envelope: M3 (context +
counterpart IPV) ≈ M2 (context-only) ≈ ipv_removed (paired 90% Winkler diff −0.0002, case-cluster
p=0.863); M4 (context + ego self-IPV) ≈ −2% width. Recorded for traceability only; do **not** argue or
feature this null externally; the envelope is reported simply as context-conditioned.

## Boundaries

InterHub σ=0.1; empirical/probabilistic monitor, not a formal proof. Subgroup conditional validity is
uneven (126/264 supported rows outside ±3 pp). LODO 90% coverage ranges 0.749–0.991 → transfer is not
unconditional. Target exact-zero atom ~21.6% qualifies endpoint/tie interpretation.

## Paper Handoff

Supports **R3** as a context-conditioned, estimability-aware conformal runtime envelope. Phrase the
envelope as context-conditioned; do not present IPV-conditioning as a mechanism. Empirical monitor only —
no formal-proof or planner-benefit language.

## Downstream Clearance

M3-vs-M4 pivot cleared (no escalation). The frozen M3 scorer is cleared for downstream consumption
(RQ010B / RQ011B / RQ012B) **as the context-conditioned envelope**; downstream lines must not expect a
distinct counterpart channel.
