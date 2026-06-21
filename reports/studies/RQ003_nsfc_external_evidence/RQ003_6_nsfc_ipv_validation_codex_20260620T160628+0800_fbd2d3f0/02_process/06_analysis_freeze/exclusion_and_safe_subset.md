# RQ003 Frozen Exclusion and Safe-Subset Contract

Worker: `RQ003_phase3_freeze_002`

## Exclusion Order

Apply exclusions in this order before the primary model:

1. Approved top-five cohort only.
2. Mapped `team x scenario` cell only.
3. High-support IPV summary available.
4. Non-A1 scenario only.
5. Collision-free primary sample.

No exclusion threshold may be tuned on official coordination, efficiency, comprehensive score, ranks, or any predictor-outcome association.

## Primary Exclusions

- A1 is excluded from continuous primary modeling and handled as a catastrophic-safety boundary case.
- Collision cells are excluded from the primary continuous coordination-residual model.
- Cells without high-support conflict-window IPV summaries are abstained.
- Cells whose scenario membership can only be recovered from score-joined tables are blocked until an outcome-free scenario map is available.

## Safe Subsets

The primary conclusion requires at least two outcome-independent safe subsets to agree in IPV direction.

| subset_id | frozen definition | source | role |
|---|---|---|---|
| S1 | `collision == 0` | Frozen plan Section 7 safety subset definition; safety primitive schema only. | Primary collision-free subset. |
| S2 | `safety_score == 100 AND collision == 0` | Frozen plan Section 7 safety subset definition; safety-score schema only, not values. | Strict official-safety-top subset. |
| S3 | `collision == 0 AND takeover == 0 AND line_crossing == 0 AND TTC >= 1.5 s AND lateral_gap >= 2.0 m` | Gate 0 `operational_parameters.yaml` safety guard thresholds; primitive definitions only. | Strong primitive-clean subset. |

## Frozen Thresholds

- TTC threshold: `1.5 s`.
- Lateral-gap threshold: `2.0 m`.
- Source: `safety_guard.safe_guard_ttc_seconds` and `safety_guard.safe_guard_lateral_gap_m` in Gate 0 operational parameters.
- Rationale: engineering guard thresholds already frozen during outcome-free Gate 0 and separated from empirical `D_comp`/`D_yield` verifier outputs.

## Safe-Subset Decision Rule

The primary conclusion can be stated only when all are true:

1. The confirmatory leave-one-team-out comparison is directionally favorable for the IPV-added model.
2. At least two of S1, S2, and S3 agree in the direction of the IPV effect.
3. No safe subset shows a sign reversal large enough for the freeze reviewer or red team to treat the primary claim as unstable.

If only one subset agrees, report Tier B/C diagnostic evidence, not confirmatory validation.

## Exploratory Safe-Subset Analyses

All other safe-subset combinations, threshold variants, and post-hoc safety filters are exploratory discovery-family analyses and require FDR control.
