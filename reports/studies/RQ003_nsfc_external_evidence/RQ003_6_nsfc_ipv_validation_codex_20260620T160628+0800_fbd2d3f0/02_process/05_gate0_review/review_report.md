# Gate 0 Independent Review

Status: PASS
Worker: RQ003_phase2_gate0_review_001
Role: Gate 0 independent reviewer
Run: RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0
Review date: 2026-06-20

## Identity And Scope

Identity checks passed before writing review artifacts:

- Run root, Gate 0 measurement directory, and Gate 0 review directory exist.
- `02_process/00_meta/run_manifest.json` has the expected run id.
- `02_process/00_meta/plan_sha256.txt` equals `98900786a4819e58e91f16eb8da84724c928ca5a6d2c40b2252cf106b56625e1`.
- `02_process/04_gate0_measurement/gate0_status.json` exists.

This review stayed outcome-clean. I did not read Gate -1 folders, official scores, outcome-joined tables, prior predictor-outcome results, or the raw plan.

## Firewall Review

Independent manifest check: PASS.

I parsed `gate0_access_manifest.txt` and cross-checked all 40 manifest paths against `gate0_outcome_denylist.txt` using exact paths, the listed denylisted directory globs, and NSFC/RQ003 outcome-name terms. Independent denylisted-read count: 0.

The frozen `operational_parameters.yaml` and `support_definition.md` do not encode NSFC outcome-derived values. Keyword review found only outcome-firewall statements and the InterHub conformal phrase "standardized one-sided deviation scores"; no official-score, rank, coordination, comprehensive, outcome, correlation, association, regression, PCA, PET, realized-order, or phase parameter was used.

## Measurement Logic

D_comp / D_yield orientation: PASS.

The sanitized spec and frozen parameters define:

- `D_comp = max(0, (Q_low - theta_ego) / w)`, so it fires only when ego theta falls below the human conditional low quantile.
- `D_yield = max(0, (theta_ego - Q_high) / w)`, so it fires only when ego theta exceeds the human conditional high quantile.

I reran one-sided guards without project imports:

- theta below `Q_low`: `D_comp=1.0`, `D_yield=0.0`.
- theta above `Q_high`: `D_comp=0.0`, `D_yield=1.0`.

I also recomputed the formulas for all 80 trace-sample rows; mismatch count was 0. The orientation is not flipped.

Rolling-to-rolling and future-leakage checks: PASS.

Source inspection confirmed `estimate_ipv_pair` loops over `t`, computes `start = max(0, t - history_window)`, and passes only `start:t+1` slices for both agents. `estimate_ipv_current` delegates with `min_observation=steps-1`, so current estimation sees only the caller-supplied prefix. The InterHub pipeline applies dataset downsampling before estimation and passes the same `history_window`/`min_observation` into `estimate_ipv_pair`.

The trace sample has only `rolling_primary_10_frames` and `source_window` states the same `history_window=10/min_observation=4` rolling contract. There is no evidence that rolling NSFC values are compared to a full-window envelope. Pipeline-level `ipv_*_mean` summary columns exist in source code for case summaries, but they are not the deployed Gate 0 trace metric.

The trace sample contains no PET, realized-order, post-hoc phase, score, rank, outcome, coordination, comprehensive, correlation, association, regression, or PCA columns. The InterHub calibration CSV header contains retrospective metadata such as `PET` and `actual_order`, but the Gate 0 frozen parameters and output trace do not use those columns.

## Support, Views, And Conformal Boundary

Support/OOD/abstention: PASS.

I checked the trace sample against the frozen support thresholds:

- high support: exact cell `n >= 30`, both estimator errors `<= 0.6216308869824523`, `theta_npc` inside `[-1.178096549046999, 1.178097245096148]`, distance `<= 41.31042720375975 m`, and no OOD flag.
- monitor-only: at least 10 exact-cell records but not high-support eligible; abstained from primary scalar summaries.

Observed support counts were 78 high and 2 monitor-only; support-rule violations were 0.

Three views are recomputable from `ipv_trace_sample.csv`:

- Marginal: 78 high-support non-abstained rows; mean `D_comp=0.467662764104`, mean `D_yield=0.232693362889`.
- Conditional: 46 `(theta_npc_bin, state_condition, tau_bin, perspective)` cells.
- Scalar: mean `max(D_comp,D_yield)=0.700356126993`; onset and persistence can be derived from ordered frame sequences.

Conformal boundary: PASS.

The conformal threshold is recorded as InterHub calibration split only, with `nsfc_nominal_coverage_claim_allowed: false`. I found no NSFC outcome threshold tuning and no competitor-distribution renormalization of normal behavior. The empirical verifier is recorded as human conditional `D_comp`/`D_yield`; the safety guard is separated and not added to Gate 0 empirical deviations.

## Optimizer Stub Deviation

Adjudication: PASS acceptable on measurement-logic grounds, with a required Phase 4 condition.

The real optimizer was not invoked by the auditor's import-based unit tests because scipy/matplotlib were unavailable and local stubs were used. That limitation does not invalidate the formula orientation, static prefix-slicing, trace-schema, support-rule, conformal-boundary, or outcome-firewall conclusions, because those checks do not depend on solving the optimizer.

It does leave one condition: before any confirmatory NSFC result is trusted, Phase 4 must install or restore scipy/matplotlib, run the real optimizer path, and reconfirm the sign contract on real theta outputs from real replay estimation. If that re-confirmation fails, downstream confirmatory claims must stop even though Gate 0 measurement-logic review passes here.

## Verdict

Gate 0 review status: PASS, subject to the required optimizer re-confirmation condition above.

No FAIL rule was triggered: I found zero denylisted reads, no flipped D_comp/D_yield orientation, no rolling/full-window mixing in the deployed trace metric, no future leakage in the inspected code path or trace schema, no outcome-tuned threshold, and no competitor-distribution calibration.
