# Phase 4 Confirmatory Analysis Report

Worker: `RQ003_phase4_confirmatory_001`
Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`
Generated UTC: `2026-06-20T12:07:12.456466+00:00`

## Identity

- PASS: RUN_ROOT exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0
- PASS: CONF exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/10_confirmatory_analysis
- PASS: run_manifest RUN_ID matches 'RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0'
- PASS: plan_sha256 matches 
- PASS: analysis_freeze.yaml exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/06_analysis_freeze/analysis_freeze.yaml
- PASS: fold_contract.csv exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/06_analysis_freeze/fold_contract.csv
- PASS: model_capacity_contract.md exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/06_analysis_freeze/model_capacity_contract.md
- PASS: exclusion_and_safe_subset.md exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/06_analysis_freeze/exclusion_and_safe_subset.md
- PASS: primary_endpoints.md exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/06_analysis_freeze/primary_endpoints.md
- PASS: acceptance_matrix.csv exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/06_analysis_freeze/acceptance_matrix.csv
- PASS: cell_level_directional_ipv.csv exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/01_results/tables/cell_level_directional_ipv.csv
- PASS: baseline_features_cells.csv exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/01_results/tables/baseline_features_cells.csv
- PASS: scenario_map_outcome_free.csv exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/01_results/tables/scenario_map_outcome_free.csv
- PASS: replay_score_mapping.csv exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/01_results/tables/replay_score_mapping.csv
- PASS: support_coverage.csv exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/01_results/tables/support_coverage.csv
- PASS: python exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/onsite_competition/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/model_cache/venv/bin/python
- PASS: scipy import in active interpreter scipy.stats

## Frozen Scope

Only the frozen comparison was run: `state + causal kinematics + safety` versus the same baseline plus `D_comp_auc` and `D_yield_auc`.
The primary outcome is official coordination residualized by scenario and area fixed effects within each training fold. The score source is the Gate -1 `replay_score_mapping.csv` joined to analysis cells by team, area, and case_id; canonical scenario labels come from the outcome-free scenario map.

## Primary Sample

- Primary cells: 48 cells, 9 teams, 14 scenarios.
- Per-team counts: `{"T11": 11, "T14": 3, "T15": 5, "T16": 1, "T17": 4, "T20": 4, "T5": 7, "T6": 9, "T7": 4}`.
- Per-scenario counts: `{"A2": 3, "A3": 3, "A4": 2, "A5": 2, "B1": 3, "B2": 3, "B3": 5, "B4": 5, "B5": 5, "C1": 3, "C2": 3, "C3": 5, "C4": 2, "C5": 4}`.
- T8 has no eligible held-out primary cell after the frozen high-support and non-A1 exclusions; its LOTO fold is materialized but contributes no prediction row.
- Collision-free was operationalized from the frozen score/mapping evidence: the only non-100 safety rows in the top-five cohort are two zero-safety A1 rows, both excluded by the non-A1 primary rule.

## Confirmatory Result

- Leave-one-team-out delta Spearman (full - baseline): -0.110291, 95% scenario-cluster bootstrap CI [-0.262366, 0.0747422], scenario-stratified permutation p=0.58.
- Baseline Spearman=0.354103; full Spearman=0.243812.
- Baseline MAE=7.04413; full MAE=7.78304; MAE reduction=-0.738907, CI [-1.62001, 0.118369], p=0.82.
- Baseline CV-R2=0.026223; full CV-R2=-0.0218113; delta CV-R2=-0.0480344, CI [-0.331561, 0.174223], p=0.48.

Null/reverse disclosure: the primary Spearman, MAE, and CV-R2 increments are all null/reverse for the IPV-added model; full-model MAE is higher than baseline MAE.
The small high-support primary sample (48 cells, with sparse team/scenario coverage) is a material power limitation.

## Scenario-Wise IPV Association

- Median scenario-wise Spearman for `D_comp + D_yield` versus held-out coordination residual: 0.3.
- Direction-consistent scenarios: 3 / 11 with computable scenario-wise rank correlations.
- Expected direction was negative: larger directional IPV deviation should align with lower coordination residual.

## Generalization

- Secondary leave-one-scenario-out: delta Spearman=-0.0679548, CI [-0.325787, 0.119551], p=0.76; baseline/full MAE=7.73874/7.91997; baseline/full CV-R2=0.0220139/-0.0628162.
- Boundary leave-one-family-out: delta Spearman=-0.124077, CI [-0.272933, 0.00776914], p=0.74; baseline/full MAE=10.3082/10.4955; baseline/full CV-R2=-0.0134823/-0.0818286. This is boundary evidence only, not a significance headline.

## Safe Subsets

- safe_s1_loto: N=48, delta Spearman=-0.110291, MAE reduction=-0.738907, delta CV-R2=-0.0480344, direction=null_or_reverse.
- safe_s2_loto: N=48, delta Spearman=-0.110291, MAE reduction=-0.738907, delta CV-R2=-0.0480344, direction=null_or_reverse.
- safe_s3_loto: N=6, delta Spearman=0, MAE reduction=-0.0918554, delta CV-R2=-0.0124901, direction=null_or_reverse.
- Frozen safe-subset agreement count: 0 / 3 by the primary direction metric.
- Agreement requirement met: False.
- S1 and S2 are empirically identical after the primary exclusions because every non-A1 eligible primary cell has safety_score=100. S3 is very small (6 cells) and should be treated as underpowered.
- No explicit takeover or line-crossing primitive column exists in the authorized Phase 4 tables; S3 therefore uses the available collision/safety plus frozen TTC and lateral-gap guards and discloses this source limitation.

## Sensitivity

- Fallback-inclusive predictor sensitivity (not confirmatory): N=68, delta Spearman=0.0497003, CI [-0.00441843, 0.131812], p=0.22; baseline/full MAE=8.05689/7.88241; baseline/full CV-R2=-0.476696/-0.429928.

## Spec Deviations And Limitations

- No explicit takeover or line_crossing primitive field was present in the authorized Phase 4 tables or mapped log manifests; S3 uses the available collision/safety, TTC, and lateral-gap guards and discloses this limitation.
- The collision-free flag is operationalized from the frozen mapping/safety evidence: only two top-five rows have safety below 100, both zero-safety A1 cells already excluded by the non-A1 primary rule.
- S1 and S2 are empirically identical after primary exclusions because all non-A1 primary cells have safety_score=100.
- No negative controls, state-dependence/NPC analyses, plotting, or tier decision were run.
- The model is a simple ridge-linear fallback with the same fold-only preprocessing and in-train alpha grid for baseline and full arms because the high-support sample is small.
- For leave-one-scenario-out and leave-one-family-out, held-out scenario levels can be unseen by the training fixed-effect fit; unseen fixed-effect levels are applied as the training reference level to avoid leakage.
