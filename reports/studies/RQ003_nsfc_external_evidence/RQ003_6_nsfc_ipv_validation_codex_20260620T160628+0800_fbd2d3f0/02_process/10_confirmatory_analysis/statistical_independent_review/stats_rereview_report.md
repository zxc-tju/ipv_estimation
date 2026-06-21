# Phase 4 Statistical Re-Review

Worker: `RQ003_phase4_stats_rereview_001`
Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`
Generated UTC: `2026-06-20T13:34:37Z`

## Final Status

PASS.

The prior Phase 4 statistical FAIL was limited to the degenerate future-leaky negative control. The repaired artifacts resolve that issue: the future-leaky control is now a genuine nonconstant future-inclusive computation, its language is correctly scoped as a separate non-deployable diagnostic, and the primary confirmatory LOTO result is unchanged.

## Identity Verification

PASS.

- `RUN_ROOT`, `NEG`, and `REV` exist.
- `run_manifest.json` has `RUN_ID=RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`.
- `plan_sha256.txt` equals `98900786a4819e58e91f16eb8da84724c928ca5a6d2c40b2252cf106b56625e1`.
- Required tables exist: `negative_controls.csv` and `confirmatory_results.csv`.
- Before-fix quarantine artifacts exist under `02_process/11_negative_controls/`.
- Project Python exists at the requested derived-cache path.

## Leaky Genuineness

PASS.

The fixed leaky feature-health JSON reports `status=PASS` with all checks true:

- `D_comp_nonconstant=True`, 44 unique rounded finite values across 48 primary cells.
- `D_yield_nonconstant=True`, 44 unique rounded finite values across 48 primary cells.
- `theta_nonconstant=True`, 48 unique rounded finite values.
- `differs_from_rolling=True`; 47 cells changed for `D_comp`, 46 for `D_yield`.
- Mean absolute difference from rolling features: `D_comp=0.1414955408465355`, `D_yield=0.20927295683384806`.

The old quarantined leaky row was the degenerate no-op: `delta_spearman=0.0`, CI `[0, 0]`, `p=1.0`, and zero MAE/CV-R2 deltas. The current row is nonzero and matches the fixed computation:

- `delta_spearman=-0.08206686930091184`
- 95% CI `[-0.1886515914278722, 0.05248347449132109]`
- `p_delta_spearman_greater=0.2`
- `delta_mae_reduction=-0.4094253239715062`
- `delta_cv_r2=-0.056862625685136337`

The fix script computes future-inclusive framewise horizons and feeds the repaired columns through the frozen confirmatory cross-validation functions. I found no evidence that the current row is a placeholder.

## Language And Scope

PASS.

`negative_control_report.md` no longer claims that the leaky control strengthens the no-criterion-validity conclusion. It states that NULL-robustness rests on the frozen primary LOTO result plus the 10 valid negative controls, and that `future_leaky_full_window_ipv` is reported separately as a non-deployable optimistic diagnostic excluded from NULL-robustness claims. Because the repaired leaky diagnostic is slightly negative and statistically unsupported for positive increment, it does not contradict the primary null/reverse interpretation.

## Primary And Control Invariance

PASS.

Primary confirmatory LOTO values in `confirmatory_results.csv` match the frozen reference:

- `base_spearman=0.3541033434650456`
- `delta_spearman=-0.11029092488059053`
- Current `confirmatory_results.csv` SHA-256: `9b3b79f0704a09e5509780d9b096570b18d9275fd3a52d1381b2f74ea3be90d8`
- The fix script and negative-control worker report both record the same pre-fix confirmatory SHA-256 and mark the confirmatory output unchanged.

The other 10 negative controls are unchanged versus `negative_controls__before_fix.csv` by row-dictionary comparison. Only the `future_leaky_full_window_ipv` row changed, from the invalid no-op to the genuine future-inclusive diagnostic.

## Phase 4 Overall

PASS.

The prior statistical review already passed identity, primary confirmatory statistics, pseudoreplication, permutation/bootstrap units, capacity match, feature/outcome cleanliness, residualization/preprocessing leakage, power interpretation with caveat, and the 10 non-leaky negative controls. This re-review verifies the sole FAIL component is fixed without changing primary results or the other controls. No new statistical issue was found.
