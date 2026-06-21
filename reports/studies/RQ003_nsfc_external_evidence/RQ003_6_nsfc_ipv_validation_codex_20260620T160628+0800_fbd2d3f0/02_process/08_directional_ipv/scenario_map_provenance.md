# Outcome-Free Scenario Map Provenance

Worker: `RQ003_phase4_prep_001`
Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`
Status: `PASS`
Generated: `2026-06-20T18:52:54+08:00`

## Sources Used

- `data/onsite_competition/top5_research_subset/tables/top5_session_manifest.csv`: structural top-five session fields (`area`, `team_code`, `official_name`, `session_id`, `session_relative_path`).
- `data/onsite_competition/raw/beijing/tjjhs_db.sql`: structural `tj_competition_case.task_id`, `case_id`, and `sort` fields only. Score tables and score/rank fields were not opened or joined.
- `data/onsite_competition/top5_research_subset/teams/*/*/sessions/*/vehicle_perception_simulation_trajectory.log`: structural replay `taskId` and `caseId` fields only, used to verify each task-case membership appears in raw replay logs.

## Scenario/Family Rule

`tj_competition_case.sort` defines the structural case order within each approved task. The frozen 15-scenario A/B/C scheme is applied by order:

- Sort 1-5 -> family A, scenarios A1-A5.
- Sort 6-10 -> family B, scenarios B1-B5.
- Sort 11-15 -> family C, scenarios C1-C5.

This rule uses task/session/case structure only and does not use scores, ranks, validation summaries, or any predictor-outcome result.

## Validation

```json
{
  "cell_count": 150,
  "unique_cell_id_count": 150,
  "team_count": 10,
  "scenario_count": 15,
  "families": [
    "A",
    "B",
    "C"
  ],
  "team_scenario_unique": true,
  "each_team_has_15_scenarios": true,
  "each_scenario_has_10_teams": true,
  "missing_required_fields": 0,
  "raw_log_case_membership_checked": 150,
  "denylisted_content_reads": 0,
  "outcome_value_reads": 0
}
```

Output: `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/01_results/tables/scenario_map_outcome_free.csv`.
