# Phase 4 Statistical Independent Review

Worker: `RQ003_phase4_stats_review_001`
Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`
Generated UTC: `2026-06-20T13:03:46.183855+00:00`

## Status

Status: **FAIL** for the Phase 4 negative-control package, because the requested future-leaky upper-bound control is a degenerate no-op and is described too strongly in the negative-control report.

The primary confirmatory LOTO result itself is statistically reproducible and valid as a power-limited NULL/REVERSE result: no detected incremental utility of the online directional IPV block over the frozen kinematic/safety baseline. This is not evidence of proven absence of effect.

## Identity Verification

PASS. `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`, `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/10_confirmatory_analysis`, and `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/11_negative_controls` exist. The run manifest RUN_ID matches `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`, and `plan_sha256.txt` matches `98900786a4819e58e91f16eb8da84724c928ca5a6d2c40b2252cf106b56625e1`. Required result tables and scripts were present. `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/10_confirmatory_analysis/statistical_independent_review` was created for this review.

## Check Evidence

1. Nesting and unit: PASS. The frozen primary unit is `team x scenario` cell. The assembled analysis frame has 150 cells, and the primary sample has 48 unique cells across 9 teams and 14 non-A1 scenarios. LOTO, LOSO, and LOFO folds are built by team, scenario, and family respectively.

2. Pseudoreplication: PASS. The frame-level source has 16,771 rows, but feature tables aggregate to 150 unique `cell_id` rows and the primary model produces 48 prediction rows for 48 unique primary cells. No frame-level row count is used as model N or significance N.

3. Residualization leakage: PASS. `residualize(train, test)` fits scenario+area fixed effects on training rows only and applies train levels to test rows. A held-out T17 perturbation of +1,000,000 changed held-out residuals by +1,000,000 but changed fixed-effect predictions, baseline predictions, and full predictions by 0.

4. Preprocessing leakage: PASS. `fit_transform(train, test, features)` fits medians, means, and standard deviations on training rows only. `choose_alpha()` uses only training folds with inner group validation. No feature selection step exists.

5. Permutation: PASS. The permutation routine shuffles only the D columns within scenario groups and preserves scenario structure. A 19-permutation independent spot check gave `p_delta_spearman_greater=0.60`, consistent with the reported 99-permutation `p=0.58` for a negative observed delta.

6. Bootstrap: PASS. Bootstrap resampling unit is scenario cluster. A 200-resample spot CI for delta Spearman was `[-0.2656, 0.0549]`, consistent with the reported 500-resample CI `[-0.2624, 0.0747]`.

7. Generalization tiers: PASS. Leave-one-team-out is labelled primary, leave-one-scenario-out secondary, and leave-one-family-out boundary/no significance headline in freeze, result table, and report.

8. Capacity match: PASS. The baseline feature list has the same 18 state/kinematic/safety columns in both arms; the full model adds only `D_comp_auc` and `D_yield_auc` (or named control substitutes). Folds, residualization, preprocessing, ridge loss, and alpha grid are shared.

9. Negative controls: MOSTLY PASS. Spot recomputation matched the table: kinematics-only has zero increment and retains baseline Spearman 0.3541; state shuffle degrades baseline Spearman to -0.1914 with worse MAE/CV-R2; IPV shuffles/wrong envelope/counterpart/role/sign controls are null or reverse. The future-leaky control fails separately below.

10. Null disclosure: PASS for confirmatory report. It explicitly reports delta Spearman=-0.1103, p=0.58, worse MAE, worse CV-R2, safe-subset agreement 0/3, and describes the result as null/reverse with a power limitation.

11. Feature-table outcome cleanliness: PASS. `cell_level_directional_ipv.csv` and `baseline_features_cells.csv` have 150 unique cells each and contain no column names with `coordination`, `score`, `rank`, or `residual`. Object/string columns are identifiers/provenance fields, not outcome values. The feature worker's PARTIAL status is due to a metadata/header touch, not feature-value contamination.

12. Solver preset: PASS with caveat. Feature generation used real SciPy SLSQP through `solver_preset=balanced` and `candidate_grid=legacy7` after sandbox semaphore limits blocked `parallel_accurate`. Gate 0 condition G0R-COND-001 was satisfied by restored scipy/matplotlib and real-optimizer sign reconfirmation. Balanced mode may attenuate noisy IPV measurement, so it remains a measurement-power caveat, but I found no directional/capacity bias that invalidates the null/reverse primary result.

13. Future-leaky control: FAIL. The cache has 48 rows all labelled `computed_full_conflict_window_non_deployable_bounded_30_frames`, but `future_leaky_theta_ego`, `future_leaky_theta_npc`, `future_leaky_D_comp`, and `future_leaky_D_yield` are exactly zero for all 48 cells. Recomputed cross-validation is numerically identical to kinematics-only: delta Spearman=0, delta MAE=0, delta CV-R2=0, p=1.0. This is a computed no-op, not a meaningful optimistic upper bound. The negative-control report's claim that this strengthens the no-criterion-validity conclusion is therefore not statistically valid.

14. Power: PASS caveat. The primary sample is only 48 high-support cells across 9 teams, with sparse per-team counts and one team contributing no primary LOTO prediction. The correct interpretation is power-limited no detected incremental utility, not proven absence of IPV effect.

## Required Fixes

- Reclassify current future_leaky_full_window_ipv as invalid/no-op: remove language claiming it is a meaningful bounded upper bound or that it strengthens the no-criterion-validity conclusion.
- Either recompute a genuine future-leaky control with nonconstant leaky theta/D features and explicit feature-health checks, or mark the leaky upper-bound control unavailable and exclude it from NULL robustness claims.
- Rerun/update negative_controls.csv, negative_control_report.md, worker_report.json, and downstream status after the leaky-control fix; keep primary LOTO confirmatory results unchanged unless the fixed control exposes a broader pipeline issue.

## Conclusion

Primary confirmatory analysis: statistically valid and reproducible as a NULL/REVERSE finding.

Negative controls: valid except for the future-leaky upper-bound control. Because that control is degenerate and currently over-interpreted, the review status is FAIL pending a focused fixer loop.
