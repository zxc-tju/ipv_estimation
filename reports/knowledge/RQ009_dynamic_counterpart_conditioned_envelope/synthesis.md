# RQ009 Knowledge Synthesis

Status: consolidated from the frozen `decision.md` (ACCEPTED — R3 context-conditioned envelope, PI-directed 2026-06-29). This file summarizes the accepted record; `decision.md` is the canonical claim ledger.

## What was accepted

An estimability-aware, **context-conditioned**, split-conformal dynamic envelope is sharp and near-nominal (claim `RQ009-KC-R3`): at 90%, context (M2) vs global floor (M0) gives width −42.3% and Winkler −35.6% at coverage ≈0.899, with 4.78% abstention and passing directional tails. It is an empirical runtime monitor that excludes observed risk/PET from online inputs and abstains out of support.

## What was not supported (internal ablations only)

The IPV-conditioning channels add no measurable value over the context-only envelope: M3 (context + counterpart IPV) ≈ M2 (paired 90% Winkler diff −0.0002, case-cluster p=0.863); M4 (context + ego self-IPV) ≈ −2% width. These are recorded for traceability only and are **not** manuscript claims — the envelope is reported simply as context-conditioned.

## Boundaries

InterHub σ=0.1; empirical/probabilistic monitor, not a formal proof. Subgroup conditional validity is uneven (126/264 supported rows outside ±3 pp); LODO 90% coverage ranges 0.749–0.991, so transfer is not unconditional; exact-zero atom ~21.6% qualifies endpoint/tie interpretation.

## Downstream

The frozen M3 scorer is cleared for downstream consumption (RQ010B / RQ011B / RQ012B) **as the context-conditioned envelope**; downstream lines must not expect a distinct counterpart channel.

Sources: `decision.md`; `reviews/claude_review.md`.
