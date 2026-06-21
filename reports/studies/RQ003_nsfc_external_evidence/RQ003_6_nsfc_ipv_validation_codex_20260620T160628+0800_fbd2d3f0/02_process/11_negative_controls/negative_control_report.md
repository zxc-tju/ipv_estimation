# Phase 7 Corrected Negative-Control Report

Worker: `RQ003_phase7_scenario_fix_001`
Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`
Generated UTC: `2026-06-20T14:52:04.464799+00:00`

## Scope

Controls reuse the frozen primary LOTO pipeline after replacing only scenario/family/A1 labels with the corrected official crosswalk.
Reference corrected primary baseline: Spearman=-0.190937, MAE=7.41645, CV-R2=-0.141026.

## Future-Leaky Diagnostic Guardrail

- Cached future-leaky feature path: `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/onsite_competition/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/intermediate/negative_controls/future_leaky_full_window_ipv_features.csv`.
- Corrected primary cells covered by cache: 48/53; missing=5.
- The optimizer was not rerun. Missing cached future-leaky values were left NaN and handled by the same fold-local median imputer as all model features.
- Future-leaky remains a non-deployable diagnostic and is excluded from null-robustness claims.

## Control Results

| control | delta Spearman | CI | p | delta MAE reduction | delta CV-R2 | pass |
|---|---:|---|---:|---:|---:|---|
| state_shuffle | 0.0501532 | [-0.124729, 0.219648] | 0.29 | 0.0480326 | 0.0169706 | False |
| ipv_time_shuffle | 0.196823 | [0.0271377, 0.381227] | 0.29 | 0.103538 | 0.0558107 | True |
| counterpart_swap | 0.168441 | [-0.00150399, 0.347969] | 0.1 | 0.12608 | 0.0625445 | True |
| role_flip | 0.136833 | [-0.0390843, 0.32971] | 0.27 | 0.479264 | 0.086249 | True |
| sign_flip | 0.136833 | [-0.0470638, 0.314614] | 0.33 | 0.479264 | 0.086249 | True |
| wrong_envelope_cell | 0.0257217 | [-0.095683, 0.153071] | 0.69 | -0.837292 | -0.788148 | True |
| kinematics_only | 0 | [0, 0] | 1 | 0 | 0 | True |
| ipv_removed | 0 | [0, 0] | 1 | 0 | 0 | True |
| shuffled_ipv | 0.0909531 | [-0.0178418, 0.246332] | 0.5 | 0.0845946 | 0.0245986 | True |
| wrong_state | 0.0261248 | [-0.111381, 0.0971297] | 0.6 | 0.222908 | 0.0570254 | False |
| future_leaky_full_window_ipv | 0.231817 | [0.00769582, 0.482317] | 0.09 | 0.12975 | 0.0417472 | True |

## Interpretation

- Null-control pass/fail flags are diagnostic, not a new confirmatory specification, and they do not support a robust claim.
- Primary corrected LOTO delta Spearman is +0.136833 with p=0.30 and CI crossing zero. The favorable direction is suggestive and nonsignificant only.
- Controls that match or exceed the primary delta Spearman are future_leaky_full_window_ipv +0.231817, ipv_time_shuffle +0.196823, counterpart_swap +0.168441, role_flip +0.136833, and sign_flip +0.136833.
- The future-leaky row remains a non-deployable diagnostic with incomplete corrected-primary cache coverage, but its larger delta further undermines specificity of the primary favorable direction.
- State/baseline corruption controls were expected to degrade baseline signal, but state_shuffle and wrong_state failed this degradation check. In particular, state_shuffle improved baseline Spearman from -0.190937 to +0.105547.
- Therefore, the corrected results do not demonstrate IPV-specific signal. No robust incremental predictive utility relative to the prespecified kinematic+safety baseline was demonstrated.
