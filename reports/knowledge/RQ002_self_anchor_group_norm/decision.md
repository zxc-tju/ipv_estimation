# RQ002 Decision: Self-Anchor as Group Norm

Status: ACCEPTED as a falsification / boundary (MUST REVISE → guarded hybrid is the design direction; not end-to-end validated) — knowledge-layer freeze, human-directed 2026-06-24.

Runs: `RQ002_1_self_anchor_validation_main_20260619` + `RQ002_2_self_anchor_validation_codex_20260619` (both `MUST_REVISE`).
Basis: ChatGPT review "MUST REVISE — REJECT SELF-ANCHOR-ONLY NORMATIVE CLAIM" (`reviews/chatgpt_review.md`); frozen at PI direction.

## Accepted Claims

| ID | Claim |
|---|---|
| RQ002-KC-NECESSARY | Self-anchor carries substantial individual-disposition information and is necessary for sharpness (E4: situation-only R² ≈0.044; disposition-residual incremental R² ≈0.45; situation/self width ratio ≈1.34) but is INSUFFICIENT for normative authority. |
| RQ002-KC-LAUNDERING | Norm-laundering is real: a self-consistent aggressive agent can shift its own reference (E5 high-risk self-anchor flag-lift ≈0.850 < situational ≈1.129; "self-anchor passes / situation flags" subset enriched ≈1.507× for bad outcomes; E3 residual washout at moderate shift Δ≈0.4–0.6). |
| RQ002-KC-HYBRID | Evidence-driven design direction: self-anchor + split-conformal + one-sided situational floor (q05←max(q05, s05−τ_flr), online risk proxy only, no observed PET) + moderate-Δ / out-of-support abstention. |

## Rejected

Self-anchor alone = valid population group norm (REJECTED/blocked); situation-only can replace self-anchor (rejected); pure self-anchor immune to laundering (rejected); current full-window target strictly non-overlapping (rejected — needs post-anchor rebuild); guarded verifier already validated end-to-end (not yet — thresholds τ_flr/τ_abs need frozen calibration + integrated re-eval); observed PET admissible in the deployed floor (rejected).

## Paper Handoff

Use as the falsification of self-anchor normative authority; self-anchor → **M4 ablation**. The norm is the human population conditional distribution, not an agent-owned interval. The guarded hybrid is a design recommendation, NOT a validated result. (Superseded as headline by the v4.1 counterpart-conditioned dynamic envelope / RQ009.)
