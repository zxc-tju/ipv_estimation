# Phase 0B Inventory Report

Worker: `RQ003_phase0B_inventory_001`  
Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`  
Generated: 2026-06-20T16:37:49+08:00  
Mode: read-only reconnaissance with inventory-only writes.

## Identity Verification

- Run root exists: `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`
- Inventory folder exists: `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/01_inventory`
- `run_manifest.json` RUN_ID matched this worker's RUN_ID.
- `plan_sha256.txt` matched `98900786a4819e58e91f16eb8da84724c928ca5a6d2c40b2252cf106b56625e1`.

## Scope and Hygiene

This inventory records locations, roles, file sizes, tracking tier, and lightweight hashes. It does not run IPV estimation, compute any IPV-outcome association, or transcribe official score values/ranks/prior correlations into deliverables. Outcome-bearing files are location-catalogued and denylisted.

## Inventory Counts

- Total inventory rows: 1278
- Tracked lightweight metadata rows: 40
- Ignored raw payload rows: 311
- Large derived data rows: 11
- Historical exploratory output rows: 899
- Outcome-denylisted rows: 202

## NSFC / Onsite Data

- Raw onsite files under `data/onsite_competition/raw/`: 127
- Raw replay session directories matching session-id pattern: 24
- Top-five subset files under `data/onsite_competition/top5_research_subset/teams/`: 80
- Top-five subset replay session directories: 10
- Top-five trajectory logs: 37
- Top-five monitor logs: 10

Official score/outcome locations are present and denylisted. Key files include:

- `data/onsite_competition/top5_research_subset/tables/top5_scenario_scores.csv`
- `data/onsite_competition/00_manifest/score_team_coverage.csv`
- `archived/onsite_competition_results_legacy/score_beijing_abilities.csv`
- `archived/onsite_competition_results_legacy/score_shanghai_abilities.csv`

Headers were inspected only to identify fields; no outcome values were copied into this report.

## InterHub Data and Estimator Inputs

- InterHub raw pkl files found: 48
- InterHub raw/metadata files inventoried: 77
- InterHub derived files inventoried: 10
- Canonical raw subset CSV: `data/interhub/raw/subsets_for_yiru/selected_interactive_segments_equalized.csv`
- Canonical raw subset pkl root: `data/interhub/raw/subsets_for_yiru/pkl/`
- Canonical full-dataset pkl root: `data/interhub/raw/full_datasets/pkl/`
- Current stable derived full rerun: `data/derived/interhub/20260612_sigma_0_1_full_rerun/00_hpc_outputs/`

## Implementation Map

- IPV estimator: `src/sociality_estimation/core/ipv_estimation.py`
- Agent optimizer / candidate grid: `src/sociality_estimation/core/agent.py`
- Planning and geometry helpers: `src/sociality_estimation/planning/`
- Active InterHub pipeline: `pipelines/interhub/process_interhub.py`
- Simulation entrypoint: `pipelines/simulation/simulator.py`
- Available tests: `tests/test_shortcut_scripts.py`
- Test command from operating brief: `python3 -m unittest tests.test_shortcut_scripts -q`

## Locked Parameters and Gaps

- Existing parameter/lock rows recorded: 40
- Found existing locked/source constants: 28
- Missing values requiring pre-outcome freeze: 12

Most estimator and pipeline constants are present in active code. The current run does not yet contain a frozen NSFC human conditional norm/envelope artifact with numeric `Q_low`, `Q_high`, or `w_min`; those are recorded as missing in `existing_locked_parameters.csv`.

## Safety Primitives / Guards

Located source support for collision/TTC/gap concepts:

- `src/sociality_estimation/planning/Lattice.py` — collision checking and longitudinal collision cost.
- `src/sociality_estimation/planning/lattice_planner.py` — static TTC sample grid for local planning.
- `src/sociality_estimation/core/agent.py` — collision avoidance and IDM gap helper.
- `data/onsite_competition/*/session_manifest.csv` — replay logs needed for collision/takeover/line-crossing/TTC/lateral-gap extraction candidates.

Thresholds for Gate 0 safe subset S3 are not frozen in the current run.

## Exploratory Prior Locations

Existing RQ003 prior execution directories found: 6

- `reports/studies/RQ003_nsfc_external_evidence/RQ003_1_nsfc_core_evidence_20260618`
- `reports/studies/RQ003_nsfc_external_evidence/RQ003_2_nsfc_detailed_synthesis_20260619`
- `reports/studies/RQ003_nsfc_external_evidence/RQ003_3_nsfc_open_explore_fleet_20260619`
- `reports/studies/RQ003_nsfc_external_evidence/RQ003_4_nsfc_open_explore_stdlib_20260619`
- `reports/studies/RQ003_nsfc_external_evidence/RQ003_5_nsfc_open_explore_codex_20260619`
- `reports/studies/RQ003_nsfc_external_evidence/RQ003_7_nsfc_ipv_validation_codex_20260620T161246+0800_06bbd516`

These are marked as `historical exploratory output` or outcome-denylisted in the inventory. They are not confirmatory evidence for this run.

## Nature Skill

`nature-figure` was found and marked usable. See `nature_skill_capability_check.md`.

## Deliverables

- `input_inventory.csv`
- `requirement_gap_matrix.csv`
- `existing_locked_parameters.csv`
- `protected_paths.md`
- `gate0_sanitized_spec.md`
- `gate0_outcome_denylist.txt`
- `nature_skill_capability_check.md`
- `worker_report.json`
- `file_access_manifest.txt`
- `artifact_manifest.csv`

## Known Risks for Next Phase

- Gate -1 must still verify score provenance, replay-score uniqueness, missing-cell bias, and Beijing/Shanghai rubric consistency.
- Gate 0 must implement or audit the measurement trace without reading denylisted outcome files.
- Historical envelope/conformal candidates exist, but none is accepted as the frozen RQ003_6 measurement norm until provenance and parameter-lock checks pass.
