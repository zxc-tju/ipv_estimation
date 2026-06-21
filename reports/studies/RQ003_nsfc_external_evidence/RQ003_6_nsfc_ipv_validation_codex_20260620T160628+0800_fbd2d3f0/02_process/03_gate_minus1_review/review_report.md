# Gate -1 Independent Review Report

Worker: `RQ003_phase1_gate_minus1_review_001`  
Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`  
Generated: 2026-06-20T17:16:47+08:00  
Review status: **PASS**, scoped to the approved RQ003 top-five cohort only.

This review independently re-derived the Gate -1 load-bearing checks from the audit tables and cited raw evidence. I did not run predictor-outcome association, criterion-validity analysis, model fitting, or IPV computation. The PASS below authorizes the next outcome-denylisted Gate 0 measurement-audit path for the top-five cohort; it does not authorize full 20-team criterion-validity analysis.

## Identity Verification

PASS. `RUN_ROOT`, `G1`, `G1R`, and `TABLES` exist. `02_process/00_meta/run_manifest.json` has the expected run id, `02_process/00_meta/plan_sha256.txt` equals `98900786a4819e58e91f16eb8da84724c928ca5a6d2c40b2252cf106b56625e1`, and `02_process/02_gate_minus1/gate_minus1_status.json` exists.

Evidence: `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/00_meta/run_manifest.json`, `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/00_meta/plan_sha256.txt`, `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/02_gate_minus1/gate_minus1_status.json`.

## 1. Join Uniqueness

Independent recomputation from `replay_score_mapping.csv` confirms the auditor's 150/150 top-five result.

- Official archived score exports: 300 rows, 0 duplicate `(area, team, scenario)` keys.
- Mapping table: 300 rows, 0 duplicate `(area, team_code, scenario)` keys.
- Approved top-five cohort: 150 rows, 150 `unique_clean`, 150 unique `(mapped_session_id, case_id)` pairs.
- Top-five team/session structure: 10 teams, each with exactly 15 mapped scenarios and exactly one mapped session.
- Full universe mapping statuses independently recomputed: `unique_clean=225`, `unique_sql_disambiguated_from_manifest_multi_session=45`, `candidate_wrong_folder_replay=15`, `unmatched_no_sql_task_for_score_vector=15`.

Raw spot checks against manifests, SQL, and replay logs all supported the mapped chain. Six sampled cells across Beijing and Shanghai matched top-five session manifests, had required replay logs, had two SQL referee rows for the mapped task/case, had SQL `score_point2` equal to the mapped comprehensive score, and had `av` plus `mvSimulation` frames with coordinates in the mapped case replay.

Sample evidence: `beijing|T14|A7` had task `6922`, case `2344`, 251 replay case frames, and SQL lines 219068/219083. `shanghai|T6|A2` had task `6921`, case `2328`, 152 replay case frames, and SQL lines 219039/219054.

Primary evidence: `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/01_results/tables/replay_score_mapping.csv`, `data/onsite_competition/top5_research_subset/tables/top5_session_manifest.csv`, `data/onsite_competition/raw/beijing/tjjhs_db.sql`, mapped replay logs under `data/onsite_competition/top5_research_subset/teams/`.

## 2. Duplicates And Silent Loss

PASS. I found no silent drop or double-counting in the Gate -1 tables.

- `replay_score_mapping.csv` rows = 300; `coverage_matrix.csv` rows = 300.
- Mapping-vs-coverage key delta = 0 in both directions.
- Top-five score copy rows = 150; top-five-vs-plan key delta = 0 in both directions.
- Top-five score copy mismatches against mapping numeric values = 0.
- Archived score export mismatches against mapping `official_name`, scenario, and six score dimensions = 0 after accounting for the expected `team_code` vs team-slug naming layer.
- Coverage table status/identity columns matched the mapping table for every score key.

Evidence: `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/01_results/tables/replay_score_mapping.csv`, `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/01_results/tables/coverage_matrix.csv`, `data/onsite_competition/top5_research_subset/tables/top5_scenario_scores.csv`, `archived/onsite_competition_results_legacy/score_beijing_abilities.csv`, `archived/onsite_competition_results_legacy/score_shanghai_abilities.csv`.

## 3. Missingness Adequacy

PASS, with the same narrow scope as the auditor. The missingness analysis is a selection-bias diagnostic only, not a criterion-validity result.

Every row in `missingness_audit.csv` has `diagnostic_only_not_criterion_validity=True`, and the basis text states that it compares official outcome distributions between clean and missing/not-clean cells with no predictor or IPV association.

Independent recomputation of the full-universe overall row matched the table exactly: 267 observed clean cells and 33 missing/not-clean cells; observed coordination mean 77.912 vs missing/not-clean coordination mean 47.103, difference -30.809; observed efficiency mean 70.288 vs missing/not-clean efficiency mean 37.322, difference -32.966.

The full-universe bias is disclosed rather than hidden. Coverage status counts are 267 clean/sql-disambiguated cells, 15 wrong-folder candidates, 15 missing/unmatched score-source cells, and 3 case/format gaps. The non-clean cells concentrate in T18, T19, T21, and T12; the full universe is therefore not analysis-ready.

Evidence: `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/01_results/tables/missingness_audit.csv`, `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/01_results/tables/coverage_matrix.csv`, `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/02_gate_minus1/provenance_audit.md`.

## 4. Outcome Source

PASS. The auditor did not over-claim expert-rated coordination, and the conservative source verdict is supported by raw evidence.

The diagnosis PDFs expose `交通协调性` as one of six official 0-100 dimensions and show later `交通协调性得分` sections decomposed into background-traffic safety, efficiency, and comfort components with affected-background-traffic counts. The SQL dump proves referee records and judge names exist for task/scenario rows, but its fields are `score_point1` (comfort) and `score_point2` (system score); it does not provide a rater-level coordination rubric, scale, or direct per-rater coordination field.

A local filename search under `data/onsite_competition` found no standalone rubric, rating instruction, manual, guideline, standard, judge-instruction, or rules document beyond score/coverage CSVs. The allowed wording should remain `official coordination score` or `official generated coordination score`, not `expert-rated coordination`.

Evidence: `data/onsite_competition/top5_research_subset/teams/beijing/01_T17_panda/support_materials/诊断报告.pdf`, `data/onsite_competition/top5_research_subset/teams/shanghai/01_T11_wsd/support_materials/武汉理工大学_WUT_FSD.pdf`, `data/onsite_competition/raw/beijing/tjjhs_db.sql`, `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/01_results/tables/data_provenance_manifest.csv`.

## 5. Beijing/Shanghai Comparability

PASS, because the auditor's comparability statement stays within evidence.

Same numeric scale is supported: Beijing and Shanghai score CSVs/PDFs use six 0-100 dimensions including coordination. Same generated report schema is partially supported: sampled Beijing and Shanghai diagnosis PDFs share page-1 score-table labels and background-traffic coordination subsection labels. Same judges are not supported: SQL rows show area-specific judge names, including `同济跟车裁判1/2` for Shanghai examples and `通州跟车裁判1/2` for Beijing examples. Same formal rubric is not established because no rubric or rules document was found.

Evidence: `archived/onsite_competition_results_legacy/score_beijing_abilities.csv`, `archived/onsite_competition_results_legacy/score_shanghai_abilities.csv`, `data/onsite_competition/raw/beijing/tjjhs_db.sql`, sampled diagnosis PDFs above.

## 6. Gate Status Reasonableness

Independent status: **PASS** for `approved_RQ003_plan_top5_cohort_only`.

This PASS is justified because the approved top-five cohort has 150/150 clean unique mappings, no detected score/coverage reconciliation gaps, and no missing cells inside the scoped cohort. It should be read as a provenance gate for the next outcome-denylisted Gate 0 measurement audit. It should not be read as permission to run full-universe or criterion-validity analysis: the 20-team universe remains not analysis-ready until T18/T19 source/replay issues and T12/T21 case gaps are resolved or frozen as explicit exclusions.

## Spec Deviations

Project guidelines normally request appending workflow summaries to `main_workflow.log`, but the user-specified write scope allowed only files under `03_gate_minus1_review` plus append-only updates to `02_process/00_meta/artifact_index.csv`. I did not modify `main_workflow.log` or `START_HERE.md`.
