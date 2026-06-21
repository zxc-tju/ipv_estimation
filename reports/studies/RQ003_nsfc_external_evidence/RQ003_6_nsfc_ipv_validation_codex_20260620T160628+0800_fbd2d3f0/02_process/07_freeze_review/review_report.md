# RQ003 Phase 3 Freeze Review Rerun v2

Worker: `RQ003_phase3_freeze_review_002`  
Role: freeze reviewer (rerun v2)  
Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`  
Review time: `2026-06-20T18:25:04+08:00`  
Status: `PASS` with Phase 4 conditions

## Identity and Quarantine

Identity verification passed before any write:

- `RUN_ROOT`, `06_analysis_freeze/`, and `07_freeze_review/` existed.
- `run_manifest.json` matched the requested run id.
- `plan_sha256.txt` matched `98900786a4819e58e91f16eb8da84724c928ca5a6d2c40b2252cf106b56625e1`.

The prior contaminated freeze-review artifacts in `07_freeze_review/` were renamed to `*__contaminated_v001.*` before this review was written. A dated note was appended to `02_process/00_meta/spec_deviation_log.md`.

## Outcome-Clean Review Claim

Reviewer outcome-clean verdict: `true`.

No hard-denylisted file, official score/rank table, `01_results/tables/**`, Gate -1 process/review directory, or predictor-outcome result file was opened. Two non-outcome process deviations are recorded in `file_access_manifest.txt`: a line-count glob touched two tainted freeze artifacts without exposing contents, and bounded raw-log/SQL searches exposed score-free `caseId`/trajectory metadata while checking the scenario-source question. Neither exposed official score, rank, coordination, efficiency, comprehensive score, or predictor-outcome results.

## Findings

### 1. Confirmatory Family

PASS. The frozen confirmatory family contains exactly one comparison:

`state + causal kinematics + safety`

versus

`state + causal kinematics + safety + D_comp_auc_conflict_time_norm + D_yield_auc_conflict_time_norm`.

Evidence: `claims_register.md`, `primary_endpoints.md`, `analysis_freeze.yaml`, and `acceptance_matrix.csv` all make `H4-C1` the only confirmatory claim. Other endpoints, windows, ranks, family folds, and negative controls are sensitivity, exploratory, blocked, or boundary only with FDR/qualifier requirements.

### 2. Freeze Outcome-Cleanliness

PASS. The clean freeze worker's manifest reports hard-denylist content reads, outcome table reads, predictor-outcome result reads, official score/rank value-table reads, rank-table reads, Gate -1 directory reads, and prior-tainted-artifact opens all as zero. Clean freeze artifacts are present beside quarantined `__tainted_phase3_v001` artifacts.

Evidence: `06_analysis_freeze/file_access_manifest.txt`; clean `analysis_freeze.yaml`; clean artifact manifest.

### 3. Frozen Thresholds

PASS. Measurement and safety thresholds are frozen with outcome-free sources:

- primary history window: 10 frames; minimum observation: 4 frames;
- sensitivity windows: 5, 10, 20 frames;
- 10 Hz rolling prefix contract;
- `Q_low=0.25`, median `0.5`, `Q_high=0.75`;
- `w_min=0.19634954084936207 rad`;
- high-support thresholds from Gate 0;
- conformal threshold `2.0677436745727538`, InterHub calibration only;
- `TTC >= 1.5 s` and `lateral_gap >= 2.0 m` for S3.

Evidence: `04_gate0_measurement/operational_parameters.yaml`, `pre_outcome_operationalization.md`, and `exclusion_and_safe_subset.md`.

### 4. Capacity Match

PASS. The baseline and full models differ only by `D_comp` and `D_yield`. Both share the same online causal information budget, folds, preprocessing, tuning budget, and train-only fitting requirements. Forbidden features include full-window IPV, observed PET, realized order, post-hoc phase labels, future frames, official score/rank predictors, outcome-tuned thresholds, and scenario membership available only from score-joined tables.

Evidence: `model_capacity_contract.md`, `primary_endpoints.md`, and `analysis_freeze.yaml`.

### 5. Fold No Leakage

PASS. The fold contract is frozen with deterministic seed `2266481337`. It contains 10 leave-one-team-out folds, 15 leave-one-scenario-out folds, and 3 leave-one-family-out folds. Leave-one-team-out is the primary generalization. The freeze requires all residualization, imputers, scalers, encoders, fixed effects, and tuning to be fit inside training folds only, with saved train/test row identifiers before model fitting.

Evidence: `fold_contract.csv`, `primary_endpoints.md`, `model_capacity_contract.md`, and structural validation showing 28 fold rows, one seed, 10 unique team holdouts, 15 scenario holdouts, and A/B/C family holdouts.

### 6. Plan Coverage

PASS. The frozen package covers the plan's required generalization tiers, safe-subset agreement rule, carried `G0R-COND-001`, and sensitivity/exploratory separation. Blind behavior labels remain correctly blocked until new blinded annotations exist.

Evidence: `claims_register.md`, `acceptance_matrix.csv`, `exclusion_and_safe_subset.md`, and the frozen plan.

### 7. Scenario/Family Membership Source

PASS with Phase 4 condition. The clean freeze did not open score-joined scenario-score files. For team folds it used a session manifest source. For scenario and family folds it froze canonical labels from the plan and explicitly added a Phase 4 blocker requiring cell scenario labels to be derived from outcome-free scenario/session metadata; `fold_contract.csv` marks 18 scenario/family rows with that blocker.

Independent outcome-clean checks found:

- `00_manifest/session_manifest.csv` is score/rank-free but contains only area/team/session/log structural fields, not per-cell scenario labels.
- Raw replay logs expose score/rank-free `caseId`, `taskId`, `recordId`, and `caseName` fields; `caseName` is null in sampled records.
- Approved subset directory names expose team/session structure only.
- A complete A/B/C scenario-cell map was not materialized in the clean freeze package or `00_manifest/session_manifest.csv`.

Therefore Phase 4 must derive and freeze a score-free `team x scenario x family` map from raw `caseId`/session metadata or another outcome-free source before any outcome read, residualization, or model fitting. If the A/B/C membership map exists only in top-five scenario-score, validation, score-team-coverage, or other score-joined tables, Phase 4 must stop before model fitting. This condition does not expand the confirmatory family; it gates execution of the already frozen team-level primary analysis.

## Conditions Carried Forward

- `G0R-COND-001`: Phase 4 must install/restore scipy/matplotlib, run the real optimizer path, and reconfirm the sign contract on real theta outputs before any confirmatory NSFC result is trusted.
- `FRZR-COND-004`: Phase 4 must derive and confirm the `team x scenario x family` membership map from outcome-free raw replay/session metadata before any outcome read, residualization, or model fitting; stop if the map requires score/rank or score-joined files.
- `FRZR-COND-005`: Phase 4 must keep the denylist active and must not open `top5_scenario_scores.csv`, `top5_selection_summary.csv`, `validation_summary.csv`, `score_team_coverage.csv`, top-five/team README files, Gate -1 process/review directories, or any official score/rank table for scenario-map derivation before the outcome firewall intentionally opens Phase 4 outcomes.

## Decision

`PASS`. Checks 1-6 pass. Check 7 passes as a Phase 4 gating condition because the clean freeze already blocks model execution unless scenario/family membership is confirmed from an outcome-free map, and the frozen primary comparison itself remains single, capacity-matched, and team-level.
