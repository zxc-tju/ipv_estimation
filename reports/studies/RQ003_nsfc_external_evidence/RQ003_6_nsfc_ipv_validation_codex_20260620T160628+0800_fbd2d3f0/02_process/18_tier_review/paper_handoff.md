# Paper Handoff: RQ003 Tier B

Worker: `RQ003_phase9_tier_001`  
Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`  
Tier: `B`

This is a label-based handoff only. It does not edit the manuscript repository or any analysis artifact.

## Recommended Label

Use this result as Tier B diagnostic external evidence: an interpretable, power-limited top-five NSFC check with no robust independent held-out increment.

Preferred wording:

> No robust incremental predictive utility relative to the prespecified kinematic+safety baseline was demonstrated (power-limited, top-five cohort, N=53; apparent favorable direction not IPV-specific).

## Results Wording

- Primary label: weak favorable numerical direction, not a validation result.
- Scope label: approved top-five cohort only; do not generalize to the full 20-team universe.
- Outcome label: official/generated coordination score.
- Replication label: implementation/data path reproduced; robustness and specificity unresolved.
- H3 label: blocked pending real two-human labels.
- NPC label: boundary-only; no effect analysis.

Suggested sentence:

The corrected top-five NSFC analysis showed a weak favorable LOTO direction after adding directional IPV summaries to the prespecified kinematic+safety baseline, but the increment was nonsignificant (delta Spearman +0.137, p=0.30; delta CV-R2 +0.086, p=0.13), did not generalize under LOSO (delta Spearman about +0.017), and was matched or exceeded by non-primary controls.

## D_comp / D_yield Sign Narrative

Use the Gate 0 sign convention:

- `D_comp = max(0, (Q_low - theta_ego) / w)` marks below-norm competitive shortfall relative to the InterHub human conditional norm.
- `D_yield = max(0, (theta_ego - Q_high) / w)` marks above-norm yielding excess relative to the same norm.
- Positive theta remains prosocial under the estimator sign contract.

These are directional diagnostic deviations, not standalone social ground-truth labels.

## Old Claims Not To Reuse

- Do not state or imply stable validation of IPV on NSFC.
- Do not describe the coordination score as independently human-rated ground truth.
- Do not present the mechanical safe-subset flag as independent robustness support.
- Do not present LOSO, state-dependence, H3, NPC, or negative controls as rescue evidence.
- Do not state that the effect is exactly zero; the correct conclusion is power-limited and nonspecific.
- Do not make NSFC coverage guarantees from the InterHub-calibrated support boundary.

## Figure-Role Suggestions

- Main text: use at most one compact evidence-limit figure if RQ003 must appear in the paper. Its role should be "external diagnostic boundary", not validation.
- Panel A: corrected primary LOTO delta with CI and p-value, explicitly labeled nonsignificant.
- Panel B: LOSO and safe-subset boundary summary showing LOSO near zero, S1/S2 duplicate primary, and S3 n=6.
- Panel C: negative-control comparison showing controls that match or exceed the primary delta.
- Supplement only: provenance/gate flow, red-team closure, and replication2 implementation agreement.

## Limitations To Carry Forward

- Top-five cohort only; full universe not analysis-ready.
- N=53 primary cells after corrected labels and exclusions.
- Official/generated coordination outcome, not an independent behavioral annotation endpoint.
- Safe subsets do not provide independent support.
- Apparent favorable direction is not IPV-specific under controls.
- No FDR-stable state-dependent favorable stratum.
- H3 blind annotation is blocked until real two-human labels exist.
- NPC pre-onset matching is not identifiable from available fields.

## Do-Not-Write List

- Do not write a validation headline from this package.
- Do not claim that directional IPV supplied stable independent information beyond the baseline.
- Do not call the official coordination outcome an independent expert criterion.
- Do not state a proven-null conclusion.
- Do not claim NSFC conformal coverage from the InterHub boundary.
- Do not use NPC cause-effect language.
- Do not claim that manuscript files were changed in this phase.

## Paper Placement

Recommended placement is a limitation-aware supplementary diagnostic or a short main-text cautionary sentence. If a future paper draft needs a positive claim, it should require a new evidence package with real H3 labels, genuinely distinct safe subsets, a control pattern that does not match the primary increment, and scenario-level generalization support.
