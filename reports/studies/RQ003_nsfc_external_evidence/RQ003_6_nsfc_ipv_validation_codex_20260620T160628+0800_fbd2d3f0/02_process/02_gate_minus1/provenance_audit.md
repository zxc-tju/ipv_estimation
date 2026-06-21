# Gate -1 Provenance Audit

Worker: `RQ003_phase1_gate_minus1_audit_001`  
Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`  
Generated: 2026-06-20T16:58:14+08:00  
Status scope: approved RQ003 plan cohort (`top5` Beijing + `top5` Shanghai; 10 teams x 15 scenarios = 150 cells). Full 20-team scored universe is reported as a diagnostic layer and is not cleared for confirmatory analysis.

## Identity Verification

PASS.

- Run root exists: `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`.
- Gate -1 and tables output directories exist: `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/02_gate_minus1`, `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/01_results/tables`.
- `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/00_meta/run_manifest.json` has RUN_ID `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`.
- `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/00_meta/plan_sha256.txt` equals `98900786a4819e58e91f16eb8da84724c928ca5a6d2c40b2252cf106b56625e1`.
- Git HEAD checked as `c23074a091f9ff57b1034144571f68f771db9d8d`.

No predictor-outcome association, IPV computation, model fitting, or criterion-validity analysis was run.

## Evidence Base

- Plan unit and Gate -1 requirements: `reports/studies/RQ003_nsfc_external_evidence/plans/RQ003_plan_v2_nsfc_ipv_validation_20260620.md` section 3 states the main statistical unit as 150 `team x scenario` cells and requires replay-score mapping, computable-IPV cell count, coverage matrix, missingness bias, coordination source, and Beijing/Shanghai rubric checks.
- Official score CSVs: `archived/onsite_competition_results_legacy/score_beijing_abilities.csv` and `archived/onsite_competition_results_legacy/score_shanghai_abilities.csv` with fields `scenario,safety,efficiency,comfort,compliance,coordination,comprehensive,team`.
- Plan cohort score copy: `data/onsite_competition/top5_research_subset/tables/top5_scenario_scores.csv`. Audit result: 150 / 150 rows match the archived score CSV values exactly; mismatches=0.
- Raw SQL platform evidence: `data/onsite_competition/raw/beijing/tjjhs_db.sql` lines 219019-219033 define `tjjhs_referee_scoring`; lines 219673-219702 define `tjjhs_team_info`; example insert lines 219038-219063 and 219707-219730 show task/scenario/judge/team score records.
- Diagnosis report evidence: e.g. `data/onsite_competition/top5_research_subset/teams/beijing/01_T17_panda/support_materials/诊断报告.pdf` page 1 lists the six dimensions including `交通协调性`; section `交通协调性得分` breaks that score into background-traffic safety/efficiency/comfort components. `data/onsite_competition/raw/beijing/6-223/诊断报告.pdf` and `data/onsite_competition/raw/beijing/5-BIT_Site/诊断报告.pdf` page 1 independently confirm the T19 and T18 score tables.
- Replay schema: `archived/onsite_competition_results_legacy/SCHEMA.md` describes `vehicle_perception_simulation_trajectory.log` roles `av` and `mvSimulation` and `vehicle_trajectory.log` ego stream. Streaming checks used presence/format only.

## 1. Official Score Cell to Replay Mapping

Plan top-five cohort: PASS.

- Official cells: 150.
- Clean unique replay+case mappings with usable trajectory presence: 150 / 150 (100.0%).
- Naive manifest mapping for plan cohort: one-to-one for all 150 cells via `data/onsite_competition/top5_research_subset/tables/top5_session_manifest.csv`; every selected team has one session and `required_logs_present=True`.

Full scored universe diagnostic, not cleared as a confirmatory cohort:

- Official cells: 300.
- Clean or SQL-disambiguated usable mappings: 267 / 300 (89.0%).
- Candidate usable mappings if a wrong-folder replay is accepted: 282 / 300 (94.0%).
- Mapping status counts: {'unique_clean': 225, 'candidate_wrong_folder_replay': 15, 'unmatched_no_sql_task_for_score_vector': 15, 'unique_sql_disambiguated_from_manifest_multi_session': 45}.
- Manifest-naive classes before SQL correction: {'one_to_one': 240, 'unmatched': 15, 'one_to_many': 45}.

Specific non-clean findings:

- T18 / `bitsite`: archived score mean 69.0047 matches SQL task `6937` exactly (`tjjhs_referee_scoring` unique-scene mean 69.0047), and `data/onsite_competition/raw/beijing/5-BIT_Site/诊断报告.pdf` page 1 reports total score 69.0. However session `6937-1766209673` is stored under raw folder `北京赛区/6-223` and manifest team T19, while SQL `tjjhs_team_info` line 219743 identifies task `6937` as `BIT_OnSite`. Marked `candidate_wrong_folder_replay`, not clean.
- T19 / `223`: `data/onsite_competition/raw/beijing/6-223/诊断报告.pdf` page 1 reports total score 29.6 and matches the archived CSV, but no `6949-*` replay folder was found and the 29.6013 score vector does not match any 15-scene SQL `tjjhs_referee_scoring` task vector. Marked unmatched.
- Multi-session teams T4, T10, and T12 are disambiguated by SQL official task IDs (`6944`, `6948`, `6953` respectively), not by naive folder join alone.

See `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/01_results/tables/replay_score_mapping.csv` for row-level evidence.

## 2. Actually Computable IPV Feasibility

No IPV was computed. Feasibility was assessed only from replay presence and trajectory format: a cell is feasible when the mapped session has `vehicle_trajectory.log`, the mapped `caseId` appears in `vehicle_perception_simulation_trajectory.log`, and at least one ego frame plus at least one counterpart/non-ego frame with coordinates is present.

- Plan top-five cohort: 150 / 150 cells feasible by presence/format.
- Full scored universe clean mapping: 267 / 300 cells feasible by presence/format.
- Full scored universe if T18 wrong-folder replay were accepted as a candidate: 282 / 300 cells feasible by presence/format.
- Known case-level gaps in the full universe include T12 official task `6953` with only 13 / 15 case IDs present and T21 task `6934` with 14 / 15 case IDs present.

## 3. Team x Scenario x Area Coverage Matrix

Wrote `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/01_results/tables/coverage_matrix.csv` with one row per official score cell. Columns include `area`, `team_code`, `scenario`, `score_present`, `plan_top5_coverage_status`, `full_universe_clean_coverage_status`, `mapped_session_id`, `case_id`, and issue flags.

Plan cohort summary: all 150 cells are `covered_for_plan`. Full universe summary: 267 clean covered cells, 33 missing/not-clean cells under the conservative full-universe definition.

## 4. Duplicates, Ambiguous Joins, Missing Cells, Wrong Joins, Media-only Folders

- Duplicate official score keys: none for `(area, team_code, scenario)` in the full 300-cell score universe.
- Manifest one-to-many joins: T4/1216, T10/tievplus, T12/mingde have multiple replay sessions in `data/onsite_competition/00_manifest/score_team_coverage.csv`; SQL task IDs disambiguate them for official-score mapping.
- Wrong join: T18/T19 session assignment around task `6937` is inconsistent across manifest folder, SQL task owner, and score/PDF owner.
- Missing official replay: T19/223 has no verified replay for its score vector/task.
- Partial case coverage: T12 official task has 13/15 case IDs; T21 has 14/15 case IDs.
- Media-only/non-scored folders from `data/onsite_competition/00_manifest/team_manifest.csv`: Shanghai `长沙理工大学_csust` (videos, no score/replay), Beijing `8-高中部` (video/PDF/text but no official score mapping), Shanghai `UE` metadata folder. T18/`5-BIT_Site` is scored and has media/PDF but no clean replay session under its own folder.

## 5. Missingness Analysis (Selection-bias Diagnostic Only)

Wrote `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/01_results/tables/missingness_audit.csv`. This table compares official outcome distributions between cells with clean usable replay mapping and cells missing/not clean. It is explicitly not criterion validity and uses no IPV/predictor variables.

High-level direction:

- Plan top-five cohort: no missing cells, so no within-plan selection gradient can be estimated.
- Full scored universe: missing/not-clean cells are concentrated in Beijing and in specific teams (T18/T19/T21 plus partial T12). The missing/not-clean set is not random by team or area.
- Because T19 includes many zero-score cells and T18/T21 include task/file issues, missing/not-clean cells differ materially in official coordination/efficiency distributions from clean cells. Treat this as selection-bias risk for any full-universe analysis.

## 6. Source of Coordination

Verdict: `交通协调性` / coordination is best characterized as an official generated/report metric with rule/kinematic components, not as proven human expert rating.

Evidence:

- Diagnosis reports page 1 list `交通协调性` beside safety, efficiency, comfort, and compliance. Later report sections titled `交通协调性得分` break the metric into `背景交通流的安全得分`, `背景交通流的效率得分`, and `背景交通流的舒适性得分`, with affected-background-traffic counts. This points to a system/rule/kinematic report metric.
- SQL `tjjhs_referee_scoring` lines 219019-219033 define `user_id`, `user_name`, `score_point1` (`舒适性得分` comment), and `score_point2` (`系统得分` comment). The SQL rows prove referee records exist for score fields, but they do not prove the CSV `coordination` field is a human expert rating.
- No standalone scoring rubric, rater instruction document, or direct rater-by-scenario coordination records were found in the inventory or raw files.

Required wording: use `official coordination score` or `official generated coordination score`; do not write `expert-rated coordination`.

## 7. Beijing vs Shanghai Comparability

- Same numeric scale: supported. Both area score CSVs and diagnosis PDFs use six 0-100 dimensions, including `交通协调性` / coordination.
- Same report schema/rubric surface: partially supported. Beijing and Shanghai diagnosis PDFs share the same score table labels and coordination subsection labels. However, no formal rubric/rules document was found, so identical rubric cannot be fully established beyond the generated report schema.
- Same judges: not supported. SQL referee rows show different area-specific judge names: Shanghai examples include `同济跟车裁判1/2` for task `6923`; Beijing examples include `通州跟车裁判1/2` for task `6931`.

## 8. Data Provenance Manifest

Wrote `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/01_results/tables/data_provenance_manifest.csv` with artifact-level source/generator/evidence/status rows. The raw SQL, top-five session manifest, top-five materialized file map, diagnosis report score fields, and replay logs are `PROVEN` for their observed fields. The archived score CSV exports and `anno_trans.csv` are `CANDIDATE` because the generating script/export process is absent.

## 9. Unresolved Source Questions

Wrote `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/02_gate_minus1/unresolved_source_questions.md`. The main unresolved questions are: missing score-export generator, absent formal scoring rubric, T19/223 score vector not matching SQL task vectors, T18/T19 replay-folder inconsistency, and no proof that Beijing/Shanghai used the same judges or a formally identical rubric.

## Gate -1 Status

`PASS` for the approved RQ003 plan top-five cohort only.

This pass authorizes later Gate 0 measurement work on the 150 top-five cells after applying the planned exclusions and support checks. It does not authorize full 20-team criterion-validity analysis. The full universe remains not analysis-ready until the T18/T19 replay/source issues and partial-session case gaps are resolved or explicitly excluded in a frozen analysis spec.
