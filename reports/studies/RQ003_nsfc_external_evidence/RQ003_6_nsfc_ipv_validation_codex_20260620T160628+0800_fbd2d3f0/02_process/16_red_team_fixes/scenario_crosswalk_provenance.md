# Scenario Crosswalk Provenance

Worker: `RQ003_phase7_scenario_fix_001`
Generated UTC: `2026-06-20T14:41:56.150129+00:00`

## Authoritative Source

The corrected scenario labels use `replay_score_mapping.csv` structural fields (`team_code`, `area`, `case_id`, `scenario`, `scenario_family`, `scenario_name`) joined to each analysis cell by `team`, `area`, and `case_id`. Each row also carries the raw SQL line pointer (`scenario_map_sql_line`) into `data/onsite_competition/raw/beijing/tjjhs_db.sql`, where `tjjhs_referee_scoring` stores the case name. Score values were not used to choose labels.

## Result

- Cells reconciled: 150.
- Cells relabeled versus old `scenario_map_outcome_free.csv`: 120 / 150.
- Unchanged cells: 30 / 150.
- Official scenario codes found: `A1, A2, A3, A4, A5, A6, A7, B1, B2, B3, B4, C1, C2, C3, C4`.
- The old map imposed 15 positional labels as A1-A5/B1-B5/C1-C5 per area. The official competition labels use A1-A7, B1-B4, and C1-C4 for this top-five cohort.

## Old Label -> Official Label Counts

| old_label | official_scenario | n |
|---|---|---:|
| A1 | A1 | 5 |
| A1 | A7 | 5 |
| A2 | A2 | 5 |
| A2 | C3 | 5 |
| A3 | A3 | 5 |
| A3 | C1 | 5 |
| A4 | B1 | 5 |
| A4 | B4 | 5 |
| A5 | C1 | 5 |
| A5 | C2 | 5 |
| B1 | A3 | 5 |
| B1 | B1 | 5 |
| B2 | B2 | 5 |
| B2 | C2 | 5 |
| B3 | A4 | 5 |
| B3 | B3 | 5 |
| B4 | A5 | 5 |
| B4 | B2 | 5 |
| B5 | A5 | 5 |
| B5 | A6 | 5 |
| C1 | A4 | 5 |
| C1 | A7 | 5 |
| C2 | A1 | 5 |
| C2 | B3 | 5 |
| C3 | A6 | 5 |
| C3 | B4 | 5 |
| C4 | A2 | 5 |
| C4 | C3 | 5 |
| C5 | C4 | 10 |

## Files

- Corrected crosswalk: `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/01_results/tables/scenario_crosswalk_corrected.csv`.
- Raw SQL evidence: `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/onsite_competition/raw/beijing/tjjhs_db.sql`.
- Original disputed map: `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/01_results/tables/scenario_map_outcome_free.csv`.
