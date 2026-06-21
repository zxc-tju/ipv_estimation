# Stats Re-review v2 Report

Worker: `RQ003_phase7_stats_rereview2_001`  
Role: `STATS RE-REVIEW v2 on CORRECTED scenario-fixed results`  
Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`  
Generated UTC: `2026-06-20T15:08:33.962603+00:00`

## Verdict

**PASS**, with required interpretive caveats and no blocking statistical defect found in the corrected scenario-fixed confirmatory/control/state-dependence artifacts.

The corrected primary LOTO result is directionally favorable but statistically underpowered/nonsignificant: delta Spearman = `+0.136833`, scenario-cluster bootstrap CI `[-0.038781, 0.305797]`, scenario-stratified permutation `p=0.30`; MAE and CV-R2 deltas are also favorable but not confirmatory (`p=0.07` and `p=0.13`). Secondary LOSO generalization is approximately null: delta Spearman = `+0.016732`, MAE reduction = `-0.060000`, delta CV-R2 = `-0.004579`.

## Pre-write Identity

All requested identity gates passed before review artifacts were written.

| gate | result | evidence |
|---|---|---|
| RUN_ROOT exists | PASS | `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0` |
| REV created | PASS | `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/16_red_team_fixes/stats_rereview2` |
| run_manifest RUN_ID | PASS | `RUN_ID=RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0` |
| plan_sha256 | PASS | `98900786a4819e58e91f16eb8da84724c928ca5a6d2c40b2252cf106b56625e1` |
| corrected confirmatory table | PASS | `01_results/tables/confirmatory_results.csv` |
| corrected negative-control table | PASS | `01_results/tables/negative_controls.csv` |
| corrected scenario crosswalk | PASS | `01_results/tables/scenario_crosswalk_corrected.csv` |
| project Python | PASS | `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/onsite_competition/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/model_cache/venv/bin/python` |

## Crosswalk And Fold Consistency

The corrected crosswalk is authoritative and is applied consistently to the corrected outputs.

- `scenario_crosswalk_corrected.csv` has 150 unique cells and joins back to `replay_score_mapping.csv` by team/area/case_id with 0 missing joins, 0 scenario mismatches, and 0 family mismatches.
- The fix correctly records 120/150 relabelled cells versus the old positional `scenario_map_outcome_free.csv`.
- Official corrected labels are `A1-A7`, `B1-B4`, and `C1-C4`, each with 10 cells.
- Corrected `fold_assignments.csv` and corrected `cv_predictions.csv` both have 0 cell-label mismatches against `scenario_crosswalk_corrected.csv`.
- Corrected primary predictions use the 14 non-A1 official scenarios: `A2, A3, A4, A5, A6, A7, B1, B2, B3, B4, C1, C2, C3, C4`.
- In-memory LOSO/LOFO regeneration from the official labels is a legitimate error-fix because the frozen scenario universe was proven wrong by the red-team crosswalk. The correction is mechanical and documented; it is not outcome-selection if the official structural labels remain the only source of scenario identity.

## Leakage And Preprocessing

No leakage or train/test contamination was found in the corrected statistical pipeline.

- Residualization is fold-local: `residualize(train, test)` fits scenario+area fixed effects on training outcomes only and applies training design levels to held-out cells.
- Imputation and scaling are fold-local: feature medians, means, and standard deviations are fitted on training rows only.
- Alpha selection is nested inside training data only via leave-group validation inside the training fold.
- Baseline feature columns contain no official score, rank, residual, coordination, or comprehensive-score fields.
- A spot recompute of corrected primary LOTO predictions matched saved `cv_predictions.csv` to floating-point tolerance: residual target max diff `1.78e-15`, baseline prediction max diff `1.11e-16`, full prediction max diff `4.44e-16`, fixed-effect prediction max diff `1.42e-14`.

## Capacity Match

The capacity contract is preserved.

- Baseline features are the frozen kinematic/safety block.
- Full features differ only by `D_comp_auc` and `D_yield_auc` for the primary comparison.
- `kinematics_only` and `ipv_removed` recompute exactly as the baseline reference: delta Spearman = `0`, delta MAE reduction = `0`, delta CV-R2 = `0`.
- Recomputed `counterpart_swap`, `state_shuffle`, `wrong_state`, and `kinematics_only` rows matched the saved negative-control table to numerical tolerance.

## Primary Framing

The corrected favorable direction is honestly nonsignificant/underpowered, provided it is framed as suggestive only.

- Primary LOTO: base Spearman `-0.190937`, full Spearman `-0.054104`, delta `+0.136833`, CI crosses 0, `p=0.30`.
- MAE: base `7.416451`, full `6.937187`, reduction `+0.479264`, `p=0.07`.
- CV-R2: base `-0.141026`, full `-0.054777`, delta `+0.086249`, `p=0.13`.
- LOSO generalization is effectively null and must remain disclosed.
- No reviewed corrected report text claims statistical significance or validated generalization, but any reader-facing summary must avoid converting the sign flip into a confirmation claim.

## Negative-Control Degradation Anomaly

**Verdict: non-blocking anomaly, not a detected pipeline defect.** It is consistent with a fragile/underpowered and partly anti-predictive baseline.

The two baseline-degradation controls failed as expected:

| control | expected | pass | base Spearman vs reference | base MAE vs reference | base CV-R2 vs reference |
|---|---|---:|---:|---:|---:|
| state_shuffle | degradation | False | `+0.105547` vs `-0.190937` | `6.851252` vs `7.416451` | `+0.013881` vs `-0.141026` |
| wrong_state | degradation | False | `+0.033624` vs `-0.190937` | `7.356492` vs `7.416451` | `-0.069878` vs `-0.141026` |

The anomaly is not evidence of leakage by itself because the uncorrupted reference baseline already has negative held-out rank correlation and negative CV-R2. Under that condition, corrupting baseline state features can easily remove harmful/noisy fitted structure and improve held-out metrics.

Additional stress check: across 40 alternate deterministic state shuffles, only 2/40 degraded on at least two of the three baseline metrics. The shuffle distribution had median base Spearman `-0.116433` (better than `-0.190937`), median MAE `6.949435` (better than `7.416451`), and median CV-R2 `-0.011786` (better than `-0.141026`). This strongly supports the interpretation that the degradation expectation is invalid for this weak baseline, not that the corrected pipeline is broken.

Caveat: degradation-control failure weakens robustness language. These controls should be reported as failed diagnostics and should not be counted as positive validation of the baseline or of the primary result.

## Other Controls

The IPV-null controls are mostly directionally favorable but not statistically supported under the scenario-stratified permutation criterion.

- `ipv_time_shuffle`: delta Spearman `+0.196823`, CI `[0.027138, 0.381227]`, but `p=0.29`.
- `counterpart_swap`: delta Spearman `+0.168441`, CI includes 0, `p=0.10`.
- `shuffled_ipv`: delta Spearman `+0.090953`, CI includes 0, `p=0.50`.
- `wrong_envelope_cell`: no gain and worse MAE/CV-R2.
- `role_flip` and `sign_flip` are not strong mechanistic direction controls in an unconstrained standardized ridge model; sign changes and column swaps can be absorbed by coefficients.

These results are compatible with an unstable, low-powered signal. They do not reveal a capacity or leakage defect, but they should not be overstated as strong null-robustness proof.

## Safe-Subset Agreement

The reported safe-subset agreement count is formally true but weak.

- S1 and S2 are exact duplicates of the primary sample after corrected official A1 exclusion: all 53 primary cells are collision-free and have `safety=100`.
- S1/S2 therefore provide one effective agreement, not two independent safety checks.
- S3 has only 6 cells across 3 scenarios and is null/reverse on Spearman/MAE.

Safe-subset evidence should be described as weak/duplicated. It should not be used as independent confirmation despite `safe_subset_requirement_met=True` in the table.

## Future-Leaky Diagnostic

The future-leaky row is correctly excluded from null-robustness and deployable claims.

- Corrected future-leaky diagnostic: delta Spearman `+0.231817`, CI `[0.007696, 0.482317]`, permutation `p=0.09`.
- Cache coverage is incomplete under corrected primary membership: 48/53 primary cells covered, 5 cells missing and fold-locally median-imputed.
- Missing cells are official corrected primary cells whose old-label identity excluded them from the cached future-leaky feature generation.
- Because the feature is non-deployable and cache-incomplete, exclusion from null-robustness is appropriate.

## Status Decision

PASS. Corrected stats are internally valid under the official scenario-crosswalk fix and are framed as favorable-but-nonsignificant/underpowered. The degradation-control anomaly is not blocking, but it is an important caveat and should remain visible in any downstream synthesis.
