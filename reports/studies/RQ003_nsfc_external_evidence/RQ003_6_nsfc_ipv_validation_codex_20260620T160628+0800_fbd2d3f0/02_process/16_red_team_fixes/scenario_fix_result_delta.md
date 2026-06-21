# Scenario Fix Result Delta

Worker: `RQ003_phase7_scenario_fix_001`
Generated UTC: `2026-06-20T14:52:11.152984+00:00`

## Primary LOTO Before vs After

| metric | before | after | after-before |
|---|---:|---:|---:|
| base_spearman | 0.354103 | -0.190937 | -0.54504 |
| full_spearman | 0.243812 | -0.0541042 | -0.297917 |
| delta_spearman | -0.110291 | 0.136833 | 0.247124 |
| base_mae | 7.04413 | 7.41645 | 0.372318 |
| full_mae | 7.78304 | 6.93719 | -0.845854 |
| delta_mae_reduction | -0.738907 | 0.479264 | 1.21817 |
| base_cv_r2 | 0.026223 | -0.141026 | -0.167249 |
| full_cv_r2 | -0.0218113 | -0.0547772 | -0.0329659 |
| delta_cv_r2 | -0.0480344 | 0.086249 | 0.134283 |

## Direction

- Before: Spearman direction `null_or_reverse`, MAE direction `null_or_reverse`, CV-R2 direction `null_or_reverse`.
- After: Spearman direction `favorable`, MAE direction `favorable`, CV-R2 direction `favorable`.
- This is an error-fix rerun from authoritative official labels, not outcome-chasing. The result is reported without changing the frozen predictor or baseline features.
