# Phase 7 Corrected State-Dependence Boundary Report

Worker: `RQ003_phase7_scenario_fix_001`
Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`
Generated UTC: `2026-06-20T14:52:11.151725+00:00`

## Scope

Exploratory state-dependence strata were rerun under corrected official scenario/family labels. Expected local alignment direction remains negative Spearman: larger directional IPV deviation should align with lower coordination.
Primary corrected LOTO sample: n=53, delta Spearman=0.136833, MAE reduction=0.479264, delta CV-R2=0.086249.
State bootstrap budget: 500; FDR: BH over interpretable rows only; low-n threshold: n<10.

## Strongest Favorable Rows

| stratum_type | stratum | variant | n | effect | ci | q | usage |
|---|---|---|---:|---:|---|---:|---|
| geometry_lateral_gap_auc_tertile | low | high_support_only | 19 | -0.476 | [-0.778, 0.00157] | 0.628 | weak_or_uncertain_alignment_abstain |
| geometry_closing_fraction_tertile | middle | fallback_inclusive | 26 | -0.27 | [-0.625, 0.147] | 0.856 | weak_or_uncertain_alignment_abstain |
| risk_lateral_guard | lat_gap_lt_2m | high_support_only | 41 | -0.226 | [-0.523, 0.0978] | 0.856 | weak_or_uncertain_alignment_abstain |
| geometry_closing_fraction_tertile | low | high_support_only | 12 | -0.282 | [-0.787, 0.356] | 0.979 | weak_or_uncertain_alignment_abstain |
| risk_s3_primitive | s3_clean | fallback_inclusive | 14 | -0.271 | [-0.804, 0.313] | 0.979 | weak_or_uncertain_alignment_abstain |
| area_beijing_vs_shanghai | beijing | high_support_only | 23 | -0.242 | [-0.539, 0.145] | 0.979 | weak_or_uncertain_alignment_abstain |
| geometry_inverse_distance_tertile | middle | high_support_only | 18 | -0.233 | [-0.71, 0.297] | 0.979 | weak_or_uncertain_alignment_abstain |
| team | T11 | high_support_only | 12 | -0.231 | [-0.748, 0.568] | 0.979 | weak_or_uncertain_alignment_abstain |
| family | C | high_support_only | 12 | -0.161 | [-0.712, 0.559] | 0.979 | weak_or_uncertain_alignment_abstain |
| role_dominant_distance_bin | far | high_support_only | 15 | -0.16 | [-0.658, 0.415] | 0.979 | weak_or_uncertain_alignment_abstain |

## Strongest Reverse / Abstention Rows

| stratum_type | stratum | variant | n | effect | ci | q | usage |
|---|---|---|---:|---:|---|---:|---|
| geometry_distance_auc_tertile | middle | fallback_inclusive | 26 | 0.416 | [-0.019, 0.731] | 0.628 | reverse_or_context_mismatch_abstain |
| geometry_closing_fraction_tertile | low | fallback_inclusive | 27 | 0.396 | [0.0124, 0.74] | 0.628 | reverse_or_context_mismatch_abstain |
| geometry_inverse_distance_tertile | middle | fallback_inclusive | 26 | 0.39 | [-0.0313, 0.726] | 0.628 | reverse_or_context_mismatch_abstain |
| geometry_closing_fraction_tertile | high | fallback_inclusive | 26 | 0.38 | [0.00343, 0.694] | 0.628 | reverse_or_context_mismatch_abstain |
| geometry_lateral_gap_auc_tertile | middle | fallback_inclusive | 26 | 0.37 | [-0.0515, 0.717] | 0.628 | reverse_or_context_mismatch_abstain |
| role_dominant_distance_bin | mid | fallback_inclusive | 55 | 0.315 | [0.0419, 0.543] | 0.628 | reverse_or_context_mismatch_abstain |
| frame_role_relative_position | npc_ahead | fallback_inclusive | 53 | 0.258 | [-0.0328, 0.512] | 0.628 | reverse_or_context_mismatch_abstain |
| risk_s3_primitive | not_s3_clean | fallback_inclusive | 65 | 0.257 | [0.0376, 0.476] | 0.628 | reverse_or_context_mismatch_abstain |
| area_beijing_vs_shanghai | shanghai | fallback_inclusive | 56 | 0.235 | [-0.0258, 0.467] | 0.67 | reverse_or_context_mismatch_abstain |
| frame_role_motion_state | opening | fallback_inclusive | 77 | 0.204 | [-0.036, 0.436] | 0.67 | reverse_or_context_mismatch_abstain |

## FDR Rows q <= 0.10

No interpretable exploratory stratum reached q <= 0.10.

## Counterexamples

High-efficiency cells with large deviation remain counterexamples to treating IPV deviation alone as a sufficient operational-failure marker.
Counterexample rows written: 18.

## Bottom Line

This exploratory rerun is boundary mapping only. Low-n, reverse-direction, and non-FDR-stable rows require abstention and do not override the corrected confirmatory result.
