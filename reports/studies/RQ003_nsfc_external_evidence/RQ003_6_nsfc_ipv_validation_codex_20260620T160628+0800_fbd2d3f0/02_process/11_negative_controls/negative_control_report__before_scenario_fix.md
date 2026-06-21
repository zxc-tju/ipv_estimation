# Phase 4 Negative-Control Report

Worker: `RQ003_phase4_negcontrol_fix_001`
Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`
Generated UTC: `2026-06-20T13:25:16.091922+00:00`

## Scope

All controls use the frozen primary leave-one-team-out pipeline: coordination residualized by in-fold scenario/area fixed effects, ridge models with the frozen alpha grid, the frozen primary inclusion mask, and the frozen capacity match of baseline versus baseline plus two IPV-like columns.
The frozen confirmatory artifacts were read but not modified. The previous future-leaky full-window row was reclassified as an invalid no-op before this fix; the quarantined before-fix artifacts preserve that evidence.

## Verdicts

- Pipeline validity verdict: PASS: baseline retains signal and IPV/state corruptions behave as controls.
- NULL-robustness verdict: PASS: primary NULL/REVERSE is supported by the frozen primary LOTO result and the 10 valid negative controls. The future-leaky control is reported separately as a non-deployable optimistic diagnostic and is not used to strengthen NULL-robustness claims.
- Frozen model_base reference: Spearman=0.354103, MAE=7.04413, CV-R2=0.026223.
- Future-leaky feature-health verdict: PASS; this control is not part of the primary NULL-robustness claim.

## Control Results

| control | expected | delta Spearman | 95% CI | p | delta MAE reduction | delta CV-R2 | pass |
|---|---|---:|---|---:|---:|---:|---|
| state_shuffle | permute baseline state features across primary cells; expect degradation | 0.0119409 | [-0.0536083, 0.0537623] | 0.69 | -0.000195618 | -0.00276105 | True |
| ipv_time_shuffle | shuffle directional IPV values across cells/time; expect no gain | -0.105297 | [-0.248029, 0.0264432] | 0.43 | -0.316241 | -0.0347309 | True |
| counterpart_swap | swap ego/NPC conditioning; expect no improvement | -0.00597047 | [-0.102776, 0.0871986] | 0.19 | -0.14646 | -0.0330204 | True |
| role_flip | swap competition/yield role labels; expect no improvement | -0.110291 | [-0.26034, 0.0628335] | 0.65 | -0.738907 | -0.0480344 | True |
| sign_flip | negate D_comp/D_yield; diagnostic of direction | -0.110291 | [-0.254393, 0.0656892] | 0.57 | -0.738907 | -0.0480344 | True |
| wrong_envelope_cell | use mismatched conditional-norm envelope cells; expect no gain | -0.312419 | [-0.495128, -0.0790657] | 0.92 | -1.23727 | -0.208184 | True |
| kinematics_only | baseline-only kinematic+safety reference signal | 0 | [0, 0] | 1 | 0 | 0 | True |
| ipv_removed | remove IPV columns so full arm equals model_base | 0 | [0, 0] | 1 | 0 | 0 | True |
| shuffled_ipv | scenario-stratified shuffled IPV; expect null | -0.0785931 | [-0.142265, 0.0105625] | 0.53 | -0.496075 | -0.0235635 | True |
| wrong_state | independently corrupt state features; expect degradation | -0.110617 | [-0.19124, 0.0216549] | 0.83 | -0.0178148 | -0.00757769 | True |
| future_leaky_full_window_ipv | genuine future-inclusive IPV diagnostic; non-deployable optimistic upper-bound attempt, excluded from NULL-robustness claims | -0.0820669 | [-0.188652, 0.0524835] | 0.2 | -0.409425 | -0.0568626 | True |

## Future-Leaky Fix

- The before-fix `future_leaky_full_window_ipv` row was invalid because the cached full-window estimate was constant, yielding exactly zero incremental deltas.
- The repaired control uses a future-inclusive framewise horizon: up to 30 evenly spaced starts per primary cell, with the actual next 30 conflict-window frames supplied to the optimizer. This is non-deployable.
- Feature health: D_comp nonconstant=True; D_yield nonconstant=True; theta nonconstant=True; differs from rolling=True; sane ranges=True.
- Rolling-feature comparison: mean |delta D_comp|=0.141496; mean |delta D_yield|=0.209273; Spearman corr D_comp=0.164237; Spearman corr D_yield=-0.100703.

## Interpretation

- `kinematics_only` and `ipv_removed` show the baseline side of the frozen pipeline retains signal; this argues against a broken outcome join or fold implementation.
- `ipv_time_shuffle`, `shuffled_ipv`, `wrong_envelope_cell`, `counterpart_swap`, and `role_flip` are the IPV-null stress tests. Their expected behavior is no incremental held-out prediction gain.
- `state_shuffle` and `wrong_state` corrupt the baseline state/kinematic features. Their expected behavior is degradation of the baseline reference signal relative to the frozen model_base.
- `sign_flip` is a diagnostic control. Because the frozen ridge model is unconstrained and standardizes features in fold, sign reversal can be absorbed by coefficient sign changes; interpret it as a prediction diagnostic, not as evidence for a mechanistic direction.
- `future_leaky_full_window_ipv` is now a genuine, non-deployable future-inclusive diagnostic only. It is not used to strengthen the primary no-criterion-validity conclusion.

## Per-Control Observations

- `state_shuffle`: baseline signal check ['base_spearman_lower', 'base_mae_higher', 'base_cv_r2_lower']; base Spearman -0.191381 vs reference 0.354103; base MAE 7.86511 vs 7.04413; base CV-R2 -0.0615411 vs 0.026223.
- `ipv_time_shuffle`: no statistically supported positive IPV increment.
- `counterpart_swap`: no statistically supported positive IPV increment.
- `role_flip`: no statistically supported positive IPV increment.
- `sign_flip`: sign-flipped IPV is prediction-null/reverse; unconstrained ridge can absorb sign reversals, so this is diagnostic rather than directional proof.
- `wrong_envelope_cell`: no statistically supported positive IPV increment.
- `kinematics_only`: baseline-only reference retained the frozen model_base signal and zero IPV increment.
- `ipv_removed`: baseline-only reference retained the frozen model_base signal and zero IPV increment.
- `shuffled_ipv`: no statistically supported positive IPV increment.
- `wrong_state`: baseline signal check ['base_spearman_lower', 'base_mae_higher', 'base_cv_r2_lower']; base Spearman -0.156318 vs reference 0.354103; base MAE 7.58035 vs 7.04413; base CV-R2 0.000746818 vs 0.026223.
- `future_leaky_full_window_ipv`: genuine non-deployable future-horizon leaky control did not show a statistically supported positive increment; reported separately and excluded from NULL-robustness strengthening claims.

## Non-Goals Observed

- No new confirmatory specification was introduced.
- No state-dependence/NPC Phase 6 analysis was run.
- No plotting or Tier decision was made.
