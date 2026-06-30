# RQ011B Knowledge Synthesis

Status: consolidated from the parent RQ011 `decision.md` (§ RQ011B close-out) and the RQ011B reviews. Canonical claim record is `../RQ011_onsite_full_universe_readiness/decision.md`.

## What was tested

A directional, interpretable runtime monitor: per-frame signed M3 deviation from the human cooperative-norm envelope, built outcome-blind (19,044 supported frames / 245 units; 4,243 frames outside the 90% norm, split 2,300 passive / 1,943 aggressive; build hash `417b0078…`). Failure ledger: 23,521 occurrences, 23,004 timestamped, 780 timestamped and pre-window IPV-eligible.

## Result — under-identified null

The primary contrast (any-failure-moment vs C1 within-interaction matched controls, onset-safe pre-window) is `UNDER_IDENTIFIED`: **C1 has 0 controls**, so effect, efficiency, and CI are not estimable; the primary gate did not pass. Robustness controls cannot rescue it (C2 ROC AUC = 0.493, eff = 0.0084; C3 eff ≈ 0; C4 eff = 0.0084; fixed alarm 54.2 false alarms per interaction-minute at recall 0.20; label `NON_DIRECTIONAL`). Specificity, LOSO, and per-category BH-FDR (m=18) are all under-identified; no category is BH-significant.

## Headline limitation (PI-mandated)

The binding bottleneck is **interaction-failure segment retrieval/segmentation**, not the IPV monitor contrast. The same measurement problem recurs across the audit chain: collision-only criteria too sparse (19/285), broad any-failure saturated (285/285), moment-level within-interaction controls vanish (C1 = 0). The null is therefore **provisional and measurement-limited** — not a clean refutation of IPV monitoring.

## Manuscript role

Bounded statement only: moment-level IPV monitoring was not demonstrated on OnSite under the current failure-segment retrieval, pending an adequate method. Convergent with RQ009's counterpart-IPV practical null and RQ003's NSFC null (carry the measurement caveat). Revisit only after a future RQ solves failure-segment retrieval/segmentation.

Sources: `../RQ011_onsite_full_universe_readiness/decision.md` (§ RQ011B); `reviews/chatgpt_review.md`; `reviews/claude_review.md`.
