# Phase 7 Corrected Confirmatory Analysis Report

Worker: `RQ003_phase7_scenario_fix_001`
Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`
Generated UTC: `2026-06-20T14:44:03.464651+00:00`

## Identity

- PASS: RUN_ROOT exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0
- PASS: FIX exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/16_red_team_fixes
- PASS: run_manifest RUN_ID matches 'RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0'
- PASS: plan_sha256 matches 
- PASS: red_team_status BLOCKERS_FOUND 'BLOCKERS_FOUND'
- PASS: cell_level_directional_ipv.csv exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/01_results/tables/cell_level_directional_ipv.csv
- PASS: baseline_features_cells.csv exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/01_results/tables/baseline_features_cells.csv
- PASS: support_coverage.csv exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/01_results/tables/support_coverage.csv
- PASS: replay_score_mapping.csv exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/01_results/tables/replay_score_mapping.csv
- PASS: scenario_map_outcome_free.csv exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/01_results/tables/scenario_map_outcome_free.csv
- PASS: analysis_freeze.yaml exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/06_analysis_freeze/analysis_freeze.yaml
- PASS: fold_contract.csv exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/06_analysis_freeze/fold_contract.csv
- PASS: run_confirmatory_analysis.py exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/10_confirmatory_analysis/run_confirmatory_analysis.py
- PASS: run_negative_controls.py exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/11_negative_controls/run_negative_controls.py
- PASS: frame_level_directional_ipv.csv exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/onsite_competition/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/frame_level/frame_level_directional_ipv.csv
- PASS: python exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/onsite_competition/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/model_cache/venv/bin/python
- PASS: tjjhs_db.sql exists /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/onsite_competition/raw/beijing/tjjhs_db.sql
- PASS: running expected project Python /Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/onsite_competition/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/model_cache/venv/bin/python

## Correction Applied

Scenario, family, A1, and LOSO/LOFO labels were replaced mechanically with the official structural scenario code from `replay_score_mapping.csv`, cross-checked by raw `tjjhs_referee_scoring` case-name lines. The cell-level IPV and baseline feature tables were not recomputed.
The official top-five code space in the authoritative files is `A1-A7`, `B1-B4`, and `C1-C4`; the old outcome-free map imposed a positional `A1-C5` grid and mislabeled 120/150 cells.

## Interpretation Correction

This report was corrected by `RQ003_phase7_interp_fix_001` after red-team-v2 review. The corrected primary LOTO result has a favorable numerical direction, but it is suggestive, nonsignificant (p=0.30, CI includes 0), non-generalizing (LOSO delta Spearman +0.017), and non-IPV-specific. No robust incremental predictive utility relative to the prespecified kinematic+safety baseline was demonstrated.

The table fields `confirmatory_status=confirmatory`, `effect_direction_*=favorable`, and the safe-subset requirement flag are row labels or mechanical flags only. They must not be used as reader-facing support for a robust claim.

Independent replication2 has now closed RT2-BLOCK-002: `02_process/17_independent_replication/replication2/replication2_status.json` reports corrected N=53, exact reproduction under the reported-alpha refit (delta Spearman +0.136833), and the same favorable direction under independent training-tuned analysis (delta Spearman +0.054669). This verifies the corrected data/model path direction, but it does not establish significance, generalization, or IPV-specificity.

## Primary Sample

- Primary cells: 53; predictions: 53; teams: 9; official scenarios: 14.
- Per-team counts: `{"T11": 11, "T14": 4, "T15": 6, "T16": 2, "T17": 5, "T20": 5, "T5": 7, "T6": 9, "T7": 4}`.
- Per-official-scenario counts: `{"A2": 3, "A3": 2, "A4": 5, "A5": 4, "A6": 5, "A7": 7, "B1": 3, "B2": 3, "B3": 4, "B4": 5, "C1": 3, "C2": 3, "C3": 2, "C4": 4}`.
- Inclusion: mapped, high-support, official non-A1, collision-free.

## Corrected A1 / Safety Identity

- Official A1 rows: 10.
- Safety=0 or coordination=0 official A1 rows: 2.
  - `T15_C2_task6924_case2333`: old_label=C2, official_scenario=A1, safety=0, coordination=0, comprehensive=0.
  - `T20_C2_task6941_case2333`: old_label=C2, official_scenario=A1, safety=0, coordination=0, comprehensive=0.
- Collision-free membership is `safety >= 100`; the only safety<100 rows are the two official A1 zero-score rows listed above. The previous statement using old structural C2/A1 identity was false and is removed.

## Corrected Primary Result

- LOTO delta Spearman (full - baseline): 0.136833, 95% scenario-cluster bootstrap CI [-0.0387811, 0.305797], scenario-stratified permutation p=0.3.
- Baseline Spearman=-0.190937; full Spearman=-0.0541042; numerical direction=favorable but nonsignificant.
- Baseline MAE=7.41645; full MAE=6.93719; MAE reduction=0.479264; numerical direction=favorable.
- Baseline CV-R2=-0.141026; full CV-R2=-0.0547772; delta CV-R2=0.086249; numerical direction=favorable.
- Honest status: suggestive, nonsignificant favorable direction only. This is not a reader-facing claim and is underpowered; it is not evidence that the effect is exactly zero.

## Generalization And Safe Subsets

- Secondary LOSO: N=53, delta Spearman=0.0167315, MAE reduction=-0.0600001, delta CV-R2=-0.0045793. This is approximately zero and does not generalize.
- Boundary LOFO: N=53, delta Spearman=0.0895438, MAE reduction=0.356985, delta CV-R2=0.034094.
- safe_s1_loto: N=53, exactly the same corrected cells as `primary_inclusion`; delta Spearman=0.136833, MAE reduction=0.479264, delta CV-R2=0.086249. This is not an independent robustness check.
- safe_s2_loto: N=53, exactly the same corrected cells as `primary_inclusion`; delta Spearman=0.136833, MAE reduction=0.479264, delta CV-R2=0.086249. This is not an independent robustness check.
- safe_s3_loto: N=6, delta Spearman=0, MAE reduction=-0.0918554, delta CV-R2=0.00981387, direction=null_or_reverse. This low-n subset provides no robustness support.
- The previous safe_subset_agreement_count=2 and safe-subset requirement flag are vacuous because S1 and S2 duplicate the primary 53 cells and S3 is small/null. Remove any safe-subset robustness claim.

## IPV-Specificity Check

Negative controls match or exceed the primary delta Spearman (+0.136833): future_leaky_full_window_ipv +0.231817, ipv_time_shuffle +0.196823, counterpart_swap +0.168441, role_flip +0.136833, and sign_flip +0.136833. State/baseline degradation controls also failed: state_shuffle and wrong_state did not degrade as expected, and state_shuffle improved baseline Spearman from -0.190937 to +0.105547.

These controls block any interpretation that the favorable primary direction is robust or IPV-specific. No IPV-specific signal was demonstrated.

## Budgets And Guardrails

- Bootstrap budget: 500; primary/safe permutation budget: 99; secondary/boundary/fallback permutation budget: 49.
- No optimizer, frame-level IPV recomputation, baseline feature recomputation, freeze edit, plotting, or tier decision was performed.

## Settled Conclusion

The corrected scenario-fixed results show a suggestive, nonsignificant (p=0.30, CI includes 0), non-generalizing (LOSO delta Spearman +0.017), non-IPV-specific favorable direction; robustness is not established. No robust incremental predictive utility relative to the prespecified kinematic+safety baseline was demonstrated.
