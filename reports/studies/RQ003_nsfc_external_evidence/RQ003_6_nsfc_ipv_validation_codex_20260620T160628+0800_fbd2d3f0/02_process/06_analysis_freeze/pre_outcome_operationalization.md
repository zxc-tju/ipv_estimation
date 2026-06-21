# RQ003 Pre-Outcome Operationalization

Worker: `RQ003_phase3_freeze_002`

This document consolidates values frozen before Phase 4 predictor-outcome analysis. No value below was tuned on NSFC coordination, efficiency, comprehensive score, rank, or any predictor-outcome association.

## Identity

- Run id: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`.
- Plan SHA-256: `98900786a4819e58e91f16eb8da84724c928ca5a6d2c40b2252cf106b56625e1`.
- Gate 0 review: `PASS`.
- Required condition: `G0R-COND-001` must pass before confirmatory interpretation.

## Gate 0 Measurement Values

| value | frozen setting | source | rationale |
|---|---|---|---|
| estimator | `sociality_estimation.core.ipv_estimation.estimate_ipv_pair` | Gate 0 operational parameters | Same estimator for InterHub and NSFC. |
| primary history window | 10 frames | Gate 0 operational parameters | Existing estimator/InterHub default. |
| minimum observation | 4 frames | Gate 0 operational parameters | Existing estimator/InterHub default. |
| sensitivity windows | 5, 10, 20 frames | Gate 0 operational parameters | 0.5x/1x/2x of primary window. |
| sampling contract | 10 Hz rolling prefix | Gate 0 operational parameters | Online same-mouth rolling-to-rolling contract. |
| calibration split seed | `gate0_interhub_calibration_v1|98900786a4819e58e91f16eb8da84724c928ca5a6d2c40b2252cf106b56625e1` | Gate 0 operational parameters | InterHub-only calibration split. |
| calibration fraction | 0.80 | Gate 0 operational parameters | Prespecified InterHub split. |
| `Q_low` | 0.25 | Gate 0 operational parameters | Lower human conditional envelope. |
| median `m` | 0.50 | Gate 0 operational parameters | Human conditional median. |
| `Q_high` | 0.75 | Gate 0 operational parameters | Upper human conditional envelope. |
| `w_min` | 0.19634954084936207 rad | Gate 0 operational parameters | Max of candidate-grid half-step and InterHub high-support p10 half-IQR. |
| theta NPC bin width | 0.39269908169872414 rad | Gate 0 operational parameters | Conditional norm binning. |
| state condition | `role_rel|distance_bin|closing_bin` | Gate 0 operational parameters | Current-frame causal state only. |
| tau | `min(1,(frame_index-min_observation_frames)/primary_history_window_frames)` | Gate 0 operational parameters | No full-window endpoint. |
| high-support exact cell count | `>= 30` | Gate 0 operational parameters | InterHub support threshold. |
| monitor-only exact cell count | `>= 10` | Gate 0 operational parameters | Monitor-only threshold. |
| estimator error max | `<= 0.6216308869824523` | Gate 0 operational parameters | InterHub calibration q0.75. |
| theta NPC OOD range | `[-1.178096549046999, 1.178097245096148] rad` | Gate 0 operational parameters | InterHub calibration q01/q99. |
| conflict distance cap | `<= 41.31042720375975 m` | Gate 0 operational parameters | Min of 50 m engineering cap and InterHub calibration p95. |
| conformal alpha | 0.1 | Gate 0 operational parameters | InterHub-only conformal boundary. |
| conformal threshold | 2.0677436745727538 | Gate 0 operational parameters | InterHub high-support one-sided deviation threshold. |
| NSFC nominal coverage claim | forbidden | Gate 0 operational parameters | NSFC algorithms are not exchangeable with InterHub humans. |

## Primary Endpoint Values

| value | frozen setting | source | rationale |
|---|---|---|---|
| unit | `team x scenario` cell | Frozen plan Section 7 | Avoid frame-level pseudoreplication. |
| sample | mapped + high-support + non-A1 + collision-free | Frozen plan Section 6/7 | Primary continuous sample. |
| scope | approved top-five cohort | User authority and top-five session manifest | Gate -1 approved top-five cohort only. |
| primary outcome | official coordination residual after scenario+area fixed effects | Frozen plan Section 6 | One confirmatory outcome. |
| primary predictor | conflict-window time-normalized `D_comp` and `D_yield` AUC | Frozen plan Section 6 and Gate 0 parameters | Directional conditional IPV tails. |
| confirmatory comparison | baseline vs baseline plus `D_comp`/`D_yield` | Frozen plan Section 6 | Single confirmatory comparison. |
| primary generalization | leave-one-team-out | Frozen plan Section 6 | Team-level held-out generalization. |
| secondary generalization | leave-one-scenario-out | Frozen plan Section 6 | Scenario held-out sensitivity. |
| boundary generalization | leave-one-family-out | Frozen plan Section 6 | Boundary only; three folds. |
| fold seed | 2266481337 | SHA-256 of `analysis_freeze_v2|RUN_ID|PLAN_SHA256` | Outcome-free deterministic seed. |

## Safe-Subset Values

| subset | frozen setting | source | rationale |
|---|---|---|---|
| S1 | `collision == 0` | Frozen plan Section 7 | Primary collision-free subset. |
| S2 | `safety_score == 100 AND collision == 0` | Frozen plan Section 7 | Strict official safety-top subset; values not read here. |
| S3 | `collision == 0 AND takeover == 0 AND line_crossing == 0 AND TTC >= 1.5 s AND lateral_gap >= 2.0 m` | Gate 0 operational parameters safety guard | Strong primitive-clean subset. |

## Source and Firewall Assertions

- Tainted v1 artifacts were quarantined before this v2 freeze was written.
- No file under `01_results/tables`, `02_process/02_gate_minus1`, or `02_process/03_gate_minus1_review` was opened by this v2 freeze worker.
- No file listed as outcome-denylisted in the inventory was opened by this v2 freeze worker.
- The full plan was read as allowed context; prior exploratory values in that plan are logged only as non-confirmatory prior context.
- Scenario/family cell membership must be populated from outcome-free scenario/session metadata in Phase 4. If it exists only in a score-joined table, Phase 4 must stop.
