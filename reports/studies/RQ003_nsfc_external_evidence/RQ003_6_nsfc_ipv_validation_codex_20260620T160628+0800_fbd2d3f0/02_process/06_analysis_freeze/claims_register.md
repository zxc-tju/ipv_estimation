# RQ003 Phase 3 Analysis Freeze Claims Register

Worker: `RQ003_phase3_freeze_002`

Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`

Freeze date: 2026-06-20

This register freezes the claim family before any Phase 4 predictor-outcome result, official coordination value, rank value, or score-joined table is read by this rerun worker. The prior Phase 3 v1 artifacts were quarantined as tainted because that worker opened a score-joined table during a structural lookup.

## Confirmatory Family

Only one claim is confirmatory.

| claim_id | tag | frozen claim | required evidence | pass condition | reviewer gate |
|---|---|---|---|---|---|
| H4-C1 | confirmatory | In the approved top-five cohort, among mapped, high-support, non-A1, collision-free team-by-scenario cells, adding conflict-window time-normalized `D_comp` AUC and `D_yield` AUC improves held-out prediction of official coordination residuals over a capacity-matched `state + causal kinematics + safety` baseline. | Leave-one-team-out primary CV comparing baseline vs baseline plus `D_comp`/`D_yield`; residualization by scenario and area within training folds; safe-subset agreement check. | Full model directionally improves the prespecified validation metric over baseline in leave-one-team-out, and the same IPV direction agrees in at least two outcome-independent safe subsets. | Phase 4 real optimizer/sign reconfirmation under `G0R-COND-001`; freeze reviewer; red-team review; independent reproduction before manuscript Tier A. |

## Sensitivity Claims

These claims may qualify or support the confirmatory interpretation, but they do not create additional confirmatory comparisons.

| claim_id | tag | frozen claim | required evidence | pass condition | reviewer gate |
|---|---|---|---|---|---|
| H1a-S1 | sensitivity | The transferred InterHub human conditional envelope has enough high-support NSFC coverage to support primary analysis for the approved top-five cohort. | Outcome-free support/OOD/abstention map from Gate 0 parameters. | Primary sample contains mapped, high-support, non-A1, collision-free cells; low-support frames remain monitor-only. | Measurement reviewer. |
| H1b-S1 | sensitivity | State-conditioned directional deviations are more decision-relevant than scalar or marginal deviations under the frozen model family. | Training-fold-only comparison of conditional vs scalar/marginal summaries after primary analysis is complete. | Direction and magnitude are consistent with the primary comparison; no new significance headline. | Freeze reviewer and red-team review. |
| H2-S1 | sensitivity | Directional `D_comp`/`D_yield` summaries are more interpretable than absolute deviation summaries. | Frozen sensitivity family comparing signed tails vs absolute, p90, max, onset, persistence, and alternative windows. | FDR-controlled exploratory/sensitivity results align with primary direction; failures are recorded in `tried.md`. | Red-team review. |
| H4-SAFE | sensitivity | The primary result is not an artifact of a single safety definition. | S1, S2, and S3 safe-subset analyses defined without outcomes. | At least two outcome-independent safe subsets agree in IPV direction before any primary conclusion is allowed. | Freeze reviewer. |
| H4-GEN-SCENE | sensitivity | Leave-one-scenario-out generalization is directionally compatible with leave-one-team-out. | Secondary leave-one-scenario-out fold results. | Directional agreement with primary; no replacement of primary LOT import. | Red-team review. |

## Boundary and Blocked Claims

| claim_id | tag | frozen claim | required evidence | pass condition | reviewer gate |
|---|---|---|---|---|---|
| H4-GEN-FAMILY | exploratory | Leave-one-family-out describes transfer boundaries across A/B/C scenario families. | Boundary leave-one-family-out folds for families A, B, and C. | Reported only as boundary evidence; no significance headline because there are only three folds. | Freeze reviewer. |
| H3-B1 | blocked | Blind behavior labels can test whether `D_comp` aligns with aggressive intrusion/forcing modes and `D_yield` aligns with over-yielding/freezing modes. | New blinded annotation sample with at least two annotators, no team names, no official score, no IPV output. | Blocked until blind labels exist. | Blind-annotation protocol review. |

## Exploratory Claims

| claim_id | tag | frozen claim | required evidence | pass condition | reviewer gate |
|---|---|---|---|---|---|
| EXP-RANK | exploratory | Comprehensive score, area rank, overall rank, and score-family analyses may contextualize results but cannot validate the IPV verifier. | Exploratory discovery family after confirmatory analysis. | FDR-controlled and labelled exploratory. | Red-team review. |
| EXP-WINDOW | exploratory | p90, max, onset, latency, gain, phase, other conflict windows, and all safe-subset combinations may reveal diagnostic mechanisms. | Discovery-family analysis after primary comparison. | FDR-controlled and labelled exploratory. | Red-team review. |
| EXP-PRIOR | exploratory | Prior exploratory patterns, including signed association, near-null absolute association, and safe-but-low-coordination patterns, may motivate interpretation only. | Logged as prior context in `tried.md`; never used to choose Phase 4 thresholds or models. | No confirmatory inference based on prior values. | Freeze reviewer. |

## Rejected Confirmatory Choices

- Reject multiple confirmatory outcomes. The sole confirmatory outcome is the within-fold scenario+area residual of official coordination.
- Reject a broad model ladder as confirmatory. The sole confirmatory comparison is baseline vs baseline plus `D_comp`/`D_yield`.
- Reject comprehensive score, area rank, and overall rank as confirmatory endpoints.
- Reject full-window, observed-PET, realized-order, post-hoc-phase, and outcome-tuned features from confirmatory models.
