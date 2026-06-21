# Spec Deviation Log

- `2026-06-20T16:06:28.748822+08:00`: `GIT_SYNC=BLOCKED`. `git fetch origin main` failed with `error: cannot open '.git/FETCH_HEAD': Operation not permitted`; no pull was attempted. Initialization continued against local HEAD `c23074a091f9ff57b1034144571f68f771db9d8d` as allowed by the Phase 0A instructions.
- `2026-06-20T16:06:28.748822+08:00`: Repository-level instructions normally require updating `START_HERE.md` and `main_workflow.log` at workflow finish, but Phase 0A write scope allowed only the new run root, derived root, and RQ003 execution-lock directory. Those root tracked files were not modified by this worker.
- `2026-06-20T16:16:59+08:00`: User-authorized elevated-sandbox deviation for `RQ003_phase0A_sync_001`. The normal orchestration protocol forbids `danger-full-access` for workers, but the user explicitly authorized elevated permissions for the narrow sync completion only: `git fetch origin main`, `git pull --ff-only origin main` if safe, and writing after-sync records under this run meta. Rationale: the bootstrap worker was blocked by `.git/FETCH_HEAD` write denial under workspace-write. Actual effect: `git fetch origin main` completed; `git pull --ff-only origin main` was not run because fetched `origin/main` already equaled local HEAD.
- `2026-06-20T17:55:00+08:00`: Phase 3 v1 freeze worker opened `replay_score_mapping.csv` with coordination columns during a structural lookup, self-aborted `FAIL`, and its freeze artifacts were discarded for outcome contamination. Phase 3 rerun v2 (`RQ003_phase3_freeze_002`) quarantined the v1 artifacts as `*__tainted_phase3_v001.*` and proceeds using only outcome-free structural sources.
- `2026-06-20T18:20:00+08:00`: Repository-level workflow instructions normally require appending `main_workflow.log` at completion, but Phase 3 v2 write scope allowed only `06_analysis_freeze/` plus append-only files under `02_process/00_meta/`. `main_workflow.log` was not modified by this worker; this deviation is recorded here instead.
- `2026-06-20T18:19:49+08:00`: Phase 3 freeze reviewer v1 self-contaminated via score/rank top-five metadata, returned `BLOCKED`, and its review artifacts were quarantined as `*__contaminated_v001.*`; Phase 3 freeze review v2 (`RQ003_phase3_freeze_review_002`) reruns outcome-clean.
- `2026-06-20T18:25:04+08:00`: Phase 3 freeze review v2 had non-outcome read-scope deviations while checking scenario membership source: a line-count glob touched two tainted Phase 3 freeze files without content exposure, and bounded raw-log/SQL searches exposed score-free `caseId`/trajectory metadata. No official score, rank, outcome table, Gate -1 process/review artifact, or predictor-outcome result was opened; details are recorded in `07_freeze_review/file_access_manifest.txt`.

- `2026-06-20T18:52:54+08:00`: Phase 4 prep worker `RQ003_phase4_prep_001` used the allowed local venv fallback under `data/derived/onsite_competition/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/model_cache/venv` because the active global Python lacked scipy/matplotlib and other existing project runtime dependencies needed by the real optimizer path.
- `2026-06-20T18:52:54+08:00`: Repository-level workflow instructions normally require appending `main_workflow.log` and potentially updating `START_HERE.md`, but this worker's explicit WRITE_SCOPE allowed only `08_directional_ipv/`, `01_results/tables/scenario_map_outcome_free.csv`, derived artifacts, and append-only `02_process/00_meta` files. Root tracked files were not modified by this worker; this deviation is recorded here instead.

- `2026-06-20T21:25:16+08:00`: Phase 4 negative-control fixer `RQ003_phase4_negcontrol_fix_001` reclassified the before-fix `future_leaky_full_window_ipv` row as invalid/no-op and replaced it with a genuine non-deployable future-inclusive diagnostic after feature-health status `PASS`. The primary confirmatory results and the other 10 controls were not modified. Repository-level `START_HERE.md` and `main_workflow.log` maintenance was not written because this worker's explicit WRITE_SCOPE was limited to `11_negative_controls/`, `negative_controls.csv`, derived intermediates, and append-only `00_meta` records.

- `2026-06-20T21:27:55+08:00`: `RQ003_phase4_negcontrol_fix_001` patched `11_negative_controls/run_negative_controls.py` after the leaky-control merge to remove legacy future-leaky upper-bound over-claim language from future generated reports; active results and primary confirmatory outputs were not recomputed or changed by this wording-only script patch.

## 2026-06-20T14:52:11.166914+00:00 - Phase 7 scenario-crosswalk correction

- Worker: `RQ003_phase7_scenario_fix_001`.
- Reason: red-team blockers RT-BLOCK-001 and RT-BLOCK-002 identified that old structural scenario labels disagreed with authoritative official scenario codes and caused false A1/safety reporting.
- Correction: built `scenario_crosswalk_corrected.csv` from `replay_score_mapping.csv` structural scenario fields and raw SQL case-name line pointers; relabeled 120/150 cells mechanically.
- This is an error fix, not outcome chasing. The corrected labels are authoritative official scenario codes; existing IPV and baseline feature tables were unchanged.
- The freeze file was not edited. Corrected LOSO folds were generated in memory from the corrected official code space for the rerun.
- Future-leaky diagnostic reused existing cache without optimizer recomputation; cache coverage 48/53 corrected primary cells.

## 2026-06-20T15:28:50.011185+00:00 - Phase 7 interpretation correction

- Worker: `RQ003_phase7_interp_fix_001`.
- No heavy recompute was performed; no numeric result tables, freeze files, feature tables, tier decision, or plots were changed.
- Corrected wording cites independent replication2 for RT2-BLOCK-002 and settles the result as no robust incremental predictive utility relative to the prespecified kinematic+safety baseline.
- `START_HERE.md` was not changed because this run-local interpretation correction did not change canonical operating paths, entrypoints, or test commands. `main_workflow.log` was updated per repository workflow logging policy.
