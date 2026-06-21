# Phase 6 State-Dependence Boundary Report

Worker: `RQ003_phase6_state_npc_001`
Run ID: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`
Generated: `2026-06-20T21:51:52+08:00`

## Scope and Interpretation Guardrails

This is exploratory boundary mapping. The frozen Phase 4 primary result is null/reverse, so no row below validates a global IPV verifier. The expected local alignment direction is negative Spearman rho: larger directional IPV deviation (`D_comp + D_yield`) should correspond to lower official coordination. Positive rho is treated as reverse/context-mismatch evidence, not as support.

Multiple-stratum testing is handled with BH-FDR q-values on interpretable rows only (`n >= 10`, non-constant feature and outcome). Low-n and constant rows remain in `state_dependence_results.csv` but are flagged as abstention boundaries and are not interpreted, even when the point estimate is numerically large.

## Primary Context Reused

- Primary leave-one-team-out sample: n=48, delta Spearman=-0.110 [-0.262, 0.075], direction=null_or_reverse; safe-subset agreement=0, requirement met=False.
- Fallback sensitivity sample: n=68, delta Spearman=0.050 [-0.004, 0.132], direction=favorable; non-confirmatory.

## Where Local Alignment Appears

These are local exploratory signals only; none should be promoted to global validity. The strongest favorable rows are:

| stratum_type | stratum | variant | n | effect | ci | q_value_fdr | usage_class |
| --- | --- | --- | --- | --- | --- | --- | --- |
| geometry_lateral_gap_auc_tertile | low | high_support_only | 19 | -0.476 | [-0.777, -0.000] | 0.508 | local_alignment_signal_not_fdr_stable |
| geometry_closing_fraction_tertile | low | high_support_only | 12 | -0.282 | [-0.798, 0.376] | 0.922 | weak_or_uncertain_alignment_abstain |
| risk_s3_primitive | s3_clean | fallback_inclusive | 14 | -0.271 | [-0.751, 0.357] | 0.922 | weak_or_uncertain_alignment_abstain |
| geometry_closing_fraction_tertile | middle | fallback_inclusive | 26 | -0.270 | [-0.640, 0.207] | 0.738 | weak_or_uncertain_alignment_abstain |
| area_beijing_vs_shanghai | beijing | high_support_only | 23 | -0.242 | [-0.544, 0.172] | 0.883 | weak_or_uncertain_alignment_abstain |
| geometry_inverse_distance_tertile | middle | high_support_only | 18 | -0.233 | [-0.717, 0.307] | 0.922 | weak_or_uncertain_alignment_abstain |
| team | T11 | high_support_only | 12 | -0.231 | [-0.718, 0.480] | 0.964 | weak_or_uncertain_alignment_abstain |
| risk_lateral_guard | lat_gap_lt_2m | high_support_only | 41 | -0.226 | [-0.515, 0.127] | 0.738 | weak_or_uncertain_alignment_abstain |
| family | C | high_support_only | 18 | -0.220 | [-0.666, 0.257] | 0.922 | weak_or_uncertain_alignment_abstain |

Practical reading: the only notable favorable cell-level signal is the high-support, low lateral-gap-geometry slice. It is narrow, exploratory, and does not overturn the null/reverse primary result. Beijing high-support and family C high-support are directionally negative but weak/uncertain.

## Where the Verifier Must Abstain

Abstention is required for three reasons: missing/monitor-only support, low-n local slices, and reverse/context-mismatch direction. The strongest reverse/context-mismatch rows among interpretable strata are:

| stratum_type | stratum | variant | n | effect | ci | q_value_fdr | usage_class |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scenario | A1 | fallback_inclusive | 10 | 0.936 | [0.660, 1.000] | 0.007 | reverse_or_context_mismatch_abstain |
| team | T6 | high_support_only | 10 | 0.472 | [-0.359, 0.946] | 0.738 | reverse_or_context_mismatch_abstain |
| family | A | fallback_inclusive | 25 | 0.418 | [-0.003, 0.741] | 0.508 | reverse_or_context_mismatch_abstain |
| geometry_distance_auc_tertile | middle | fallback_inclusive | 26 | 0.416 | [-0.009, 0.754] | 0.508 | reverse_or_context_mismatch_abstain |
| geometry_closing_fraction_tertile | low | fallback_inclusive | 27 | 0.396 | [-0.021, 0.702] | 0.508 | reverse_or_context_mismatch_abstain |
| geometry_inverse_distance_tertile | middle | fallback_inclusive | 26 | 0.390 | [-0.045, 0.719] | 0.508 | reverse_or_context_mismatch_abstain |
| geometry_closing_fraction_tertile | high | fallback_inclusive | 26 | 0.380 | [-0.024, 0.698] | 0.508 | reverse_or_context_mismatch_abstain |
| geometry_lateral_gap_auc_tertile | middle | fallback_inclusive | 26 | 0.370 | [-0.059, 0.683] | 0.508 | reverse_or_context_mismatch_abstain |
| risk_ttc_guard | ttc_missing | fallback_inclusive | 12 | 0.368 | [-0.243, 0.792] | 0.830 | reverse_or_context_mismatch_abstain |
| geometry_lateral_gap_auc_tertile | high | high_support_only | 18 | 0.332 | [-0.183, 0.767] | 0.738 | reverse_or_context_mismatch_abstain |
| abstention_coverage_boundary | low_coverage_le_0p1 | fallback_inclusive | 40 | 0.323 | [-0.019, 0.601] | 0.508 | reverse_or_context_mismatch_abstain |
| role_dominant_distance_bin | mid | fallback_inclusive | 55 | 0.315 | [0.059, 0.546] | 0.508 | reverse_or_context_mismatch_abstain |

Rows with exploratory q <= 0.10 are shown below. In this run these are reverse/context-mismatch or otherwise not global validation signals:

| stratum_type | stratum | variant | n | effect | ci | q_value_fdr | usage_class |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scenario | A1 | fallback_inclusive | 10 | 0.936 | [0.660, 1.000] | 0.007 | reverse_or_context_mismatch_abstain |

Low-n/constant accounting across all rows: `{"constant_not_interpretable": 1, "interpretable": 97, "low_n_not_interpretable": 50, "not_estimable_no_feature": 22}`.

## State-Specific Notes

- Risk: high-support lateral-gap-risk cells (`lat_gap_lt_2m`) are directionally favorable but uncertain; fallback-inclusive `not_s3_clean` and other broad risk slices trend reverse. Strong primitive-clean fallback rows are favorable but n=14 and uncertain.
- Geometry: high-support low lateral-gap-AUC cells show the clearest local favorable signal, but fallback-inclusive middle-distance and low/high closing-fraction rows reverse. This makes geometry a boundary condition, not a validation domain.
- Role: high-support role and motion strata are near zero or reverse. Fallback frame-level `npc_ahead` trends reverse, so role-conditioned use should abstain unless future evidence isolates a stable negative direction.
- Phase: high-support rows occur only in the late tau bin in the cached frame data. Early and middle phase estimates are monitor/fallback-only; the fallback phase rows are near-zero/reverse or constant, so early/middle phase should abstain.
- Team/scenario: most high-support team and scenario rows are low-n. Fallback A1 is strongly reverse and exploratory-FDR stable, which argues against using a favorable local scenario as a global-validity shortcut.
- Area/family: Beijing high-support is weakly favorable and Shanghai fallback trends reverse; family A fallback is reverse. Area/family heterogeneity supports boundary mapping only.
- Support boundary: rows with no valid directional IPV feature have no verifier value. Low-coverage fallback rows reverse, so fallback-inclusive estimates should be treated as opportunity-screening diagnostics, not as verifier evidence.

## High-Efficiency, Large-Deviation Counterexamples

These cases have efficiency >= 90 and directional deviation at or above the within-variant 75th percentile. They show why large IPV deviation alone cannot be treated as a sufficient marker of poor operational outcome.

| variant | cell_id | efficiency | coordination | deviation_value | dominant_deviation_direction | coverage_rate_both |
| --- | --- | --- | --- | --- | --- | --- |
| fallback_inclusive | T17_B4_task6931_case2348 | 100.000 | 87.220 | 0.128 | D_comp | 0.137 |
| fallback_inclusive | T14_A1_task6922_case2344 | 100.000 | 100.000 | 0.109 | D_comp | 0.339 |
| fallback_inclusive | T17_A1_task6931_case2344 | 100.000 | 100.000 | 0.108 | D_comp | 0.301 |
| fallback_inclusive | T11_A2_task6923_case2328 | 100.000 | 89.870 | 0.107 | D_comp | 0.492 |
| fallback_inclusive | T16_A1_task6926_case2344 | 100.000 | 99.160 | 0.083 | D_comp | 0.349 |
| fallback_inclusive | T11_C2_task6923_case2314 | 100.000 | 81.000 | 0.059 | D_comp | 0.388 |
| fallback_inclusive | T11_B4_task6923_case2317 | 100.000 | 85.420 | 0.054 | D_comp | 0.033 |
| fallback_inclusive | T15_A1_task6924_case2344 | 100.000 | 98.960 | 0.046 | D_yield | 0.007 |
| fallback_inclusive | T15_B4_task6924_case2348 | 94.800 | 82.630 | 0.107 | D_comp | 0.201 |
| fallback_inclusive | T7_B5_task6938_case2316 | 92.890 | 74.920 | 0.046 | D_comp | 0.147 |
| fallback_inclusive | T14_C5_task6922_case2330 | 92.080 | 99.550 | 0.078 | D_comp | 0.541 |
| high_support_only | T15_B5_task6924_case2335 | 100.000 | 91.990 | 0.222 | D_comp | 0.309 |

## Bottom Line

The directional IPV verifier has, at most, narrow exploratory local alignment in high-support geometry slices. It must abstain in low-support/fallback-heavy, low-n team/scenario, early/middle phase, and reverse-direction contexts. The Phase 4 null/reverse primary result remains the governing conclusion.
