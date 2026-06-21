# Future-Leaky Negative-Control Fix Report

Worker: `RQ003_phase4_negcontrol_fix_001`
Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`
Generated UTC: `2026-06-20T13:25:16.093032+00:00`

## Decision

A genuine future-leaky control was computed and used to replace the invalid before-fix no-op row.
The old full-window cached control is reclassified as invalid/no-op evidence because its leaky theta and D features were constant and produced exactly zero deltas.

## Method

- Optimizer: existing feature-worker parser plus `estimate_ipv_current`, solver preset `balanced`, candidate grid `legacy7`.
- Sharding: 10 shard files under `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/onsite_competition/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/intermediate/negative_controls/future_leaky_fix`.
- Future leak: for each sampled primary-cell start, the actual next 30 conflict-window frames were supplied to the optimizer; an online deployable estimator would not have those future frames.
- Aggregation: time-normalized AUC over up to 30 evenly spaced starts per primary cell.

## Feature-Health Checks

- Status: `PASS`
- D_comp nonconstant: `True`; stats={'n_finite': 48, 'min': 0.0, 'max': 0.8944177270429957, 'mean': 0.17063938690314776, 'std': 0.17504706655022065, 'n_unique_rounded_1e12': 44}
- D_yield nonconstant: `True`; stats={'n_finite': 48, 'min': 0.0, 'max': 1.2050591350092403, 'mean': 0.22104829483397156, 'std': 0.23792002779572471, 'n_unique_rounded_1e12': 44}
- theta nonconstant: `True`; stats={'n_finite': 48, 'min': -0.17999362644697608, 'max': 0.33167917095672544, 'mean': 0.04679606091904807, 'std': 0.09269982601944551, 'n_unique_rounded_1e12': 48}
- Differs from rolling: `True`; {'mean_abs_delta_D_comp': 0.1414955408465355, 'mean_abs_delta_D_yield': 0.20927295683384806, 'changed_cells_D_comp': 47, 'changed_cells_D_yield': 46, 'spearman_corr_D_comp_vs_rolling': 0.16423718204727303, 'spearman_corr_D_yield_vs_rolling': -0.1007029180609342}
- Sane ranges: `True`

## Frozen LOTO Result For Repaired Leaky Control

- delta Spearman: `-0.08206686930091184`
- 95% CI: `[-0.188652, 0.0524835]`
- p(delta Spearman > 0): `0.2`
- delta MAE reduction: `-0.4094253239715062`
- delta CV-R2: `-0.056862625685136337`

## Artifacts

- cell_features: `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/onsite_competition/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/intermediate/negative_controls/future_leaky_full_window_ipv_features.csv`
- frame_detail: `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/onsite_competition/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/intermediate/negative_controls/future_leaky_fix/future_leaky_full_window_ipv_frame_detail.csv`
- health_json: `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/onsite_competition/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/intermediate/negative_controls/future_leaky_fix/future_leaky_feature_health.json`
- cv_predictions: `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/onsite_competition/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/intermediate/negative_controls/future_leaky_fix/future_leaky_control_cv_predictions.csv`
- old_invalid_noop_features: `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/onsite_competition/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/intermediate/negative_controls/future_leaky_full_window_ipv_features__before_fix_invalid_noop.csv`
