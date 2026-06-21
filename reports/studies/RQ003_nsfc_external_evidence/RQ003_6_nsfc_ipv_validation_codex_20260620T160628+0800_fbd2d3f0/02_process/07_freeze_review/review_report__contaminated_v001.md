# RQ003 Phase 3 Freeze Review

Worker: `RQ003_phase3_freeze_review_001`

Role: independent freeze reviewer

Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`

Status: `BLOCKED`

## Reviewer Status

This review cannot issue `PASS`. During the scenario-membership check, this reviewer opened structural-looking top-five subset metadata that contains score/rank values:

- `data/onsite_competition/top5_research_subset/README.md`
- `data/onsite_competition/top5_research_subset/tables/materialized_analysis_files.csv`
- `data/onsite_competition/top5_research_subset/teams/beijing/01_T17_panda/README.md`

Because the brief denies reads of official scores/ranks, this reviewer session is no longer outcome-clean. I stopped the audit and wrote a `BLOCKED` review package. A fresh freeze reviewer must rerun from a clean session, treating the top-five README files and rank-bearing materialization metadata as score/rank contaminated unless a separate source inventory marks them safe.

## Identity Verification

Identity verification passed before the blocked source read:

- `RUN_ROOT` exists.
- `06_analysis_freeze/` exists.
- `07_freeze_review/` exists.
- `02_process/00_meta/run_manifest.json` reports the expected run id.
- `02_process/00_meta/plan_sha256.txt` equals `98900786a4819e58e91f16eb8da84724c928ca5a6d2c40b2252cf106b56625e1`.

## Independent Checks

1. Confirmatory family: preliminary verdict `PASS`.
   The plan freezes exactly one confirmatory comparison, `state + causal kinematics + safety` versus the same baseline plus `D_comp` and `D_yield` (plan lines 125-137). Clean freeze artifacts preserve this in `analysis_freeze.yaml` lines 75-78, `primary_endpoints.md` lines 61-71, `claims_register.md` lines 11-18, and `acceptance_matrix.csv` line 2. Other endpoints/windows/ranks are labelled sensitivity or exploratory with FDR, not confirmatory.

2. No outcome post-selection by the freeze: preliminary verdict `PASS` for the freeze, `BLOCKED` for this reviewer.
   The clean freeze manifest reports zero hard-denylist, outcome table, predictor-outcome, score-value, rank-table, and Gate -1 process reads (`file_access_manifest.txt` lines 8-17). It also records the v1 contamination and quarantine (`file_access_manifest.txt` lines 87-91; `spec_deviation_log.md` line 6). This reviewer then read score/rank-bearing top-five metadata during check 7, so this review cannot validate the freeze outcome-cleanly.

3. Thresholds frozen: preliminary verdict `PASS`.
   Gate 0 operational parameters set `Q_low=0.25`, `Q_median=0.5`, `Q_high=0.75`, `w_min=0.19634954084936207`, high-support thresholds, conflict distance, and guard thresholds from InterHub calibration or engineering rules (`operational_parameters.yaml` lines 17-63). `pre_outcome_operationalization.md` lines 14-39 and 56-63 carry those values forward without NSFC outcome tuning.

4. Capacity match: preliminary verdict `PASS`.
   `model_capacity_contract.md` lines 9-23 states the two models differ only by `D_comp_auc_conflict_time_norm` and `D_yield_auc_conflict_time_norm`. Lines 25-35 freeze the common online information budget; lines 49-64 require identical within-fold preprocessing and tuning. Forbidden feature lists in `analysis_freeze.yaml` lines 192-203 and `model_capacity_contract.md` lines 36-47 exclude full-window IPV, observed PET, realized order, post-hoc phase, future frames, official scores/ranks, and outcome-tuned thresholds.

5. Fold no leakage: preliminary verdict `PASS` by frozen rule construction.
   `fold_contract.csv` freezes seed `2266481337` and group-disjoint train rules: leave-one-team-out lines 2-11, leave-one-scenario-out lines 12-26, and leave-one-family-out lines 27-29. `primary_endpoints.md` lines 29-34 and `model_capacity_contract.md` lines 49-57 require residualization, preprocessing, and row identifiers to be handled within training folds before fitting. A fresh reviewer should still verify Phase 4 saves actual train/test cell IDs before model fitting.

6. Plan coverage: preliminary verdict `PASS`.
   The clean freeze covers the plan-required primary endpoint, team primary generalization, scenario secondary generalization, family boundary-only generalization, and the safe-subset rule that at least two safe subsets must agree (`analysis_freeze.yaml` lines 46-82 and 110-123; `exclusion_and_safe_subset.md` lines 24-53). `G0R-COND-001` is carried forward in `analysis_freeze.yaml` lines 12-16, `pre_outcome_operationalization.md` lines 7-12, and `ipv_sign_contract.md` lines 37-43.

7. Scenario-membership concern: `BLOCKED` for this reviewer session.
   Outcome-free session manifests expose team, area, session id, and replay-log availability, but no A1-C5 per-cell scenario membership (`top5_session_manifest.csv` and `session_manifest.csv`). The top-five directory tree exposes team/session/log layout but not per-cell scenario membership. While investigating adjacent structural-looking files, this reviewer opened score/rank-bearing top-five README/materialization metadata. Therefore this check was not completed outcome-cleanly. The clean freeze already includes a Phase 4 stop gate requiring scenario/family membership from outcome-free scenario/session metadata and blocking if labels exist only in score-joined tables (`analysis_freeze.yaml` line 191; `primary_endpoints.md` lines 9-12; `exclusion_and_safe_subset.md` line 22).

## Decision

`BLOCKED`, not `PASS` and not a freeze-spec `FAIL`.

Rationale: preliminary checks did not identify confirmatory-family expansion, threshold tuning, capacity mismatch, or fold-leakage defects in the clean freeze. However, this reviewer session read score/rank-bearing top-five metadata, so it cannot provide the required outcome-clean PASS that unlocks Phase 4.

## Required Follow-Up

Run a fresh Phase 3 freeze review from a clean session. For that rerun, denylist the top-five README files, team README files, `top5_selection_summary.csv`, `top5_scenario_scores.csv`, `validation_summary.csv`, `score_team_coverage.csv`, and rank/score-bearing materialization metadata unless a source inventory explicitly proves the file is score/rank-free.

Phase 4 must not open score-joined tables to recover scenario/family membership. It must use raw replay/session metadata or a newly generated outcome-free scenario map, and it must satisfy `G0R-COND-001` before confirmatory interpretation.
