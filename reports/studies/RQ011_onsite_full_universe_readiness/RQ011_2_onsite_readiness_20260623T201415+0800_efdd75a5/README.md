# RQ011_2_onsite_readiness_20260623T201415+0800_efdd75a5

Purpose: RQ011A re-run on COMPLETE OnSite competition data. This is a readiness-only execution package. It supersedes the suspended `RQ011_1` run, which used incomplete local OneDrive data, but does not modify or overwrite that run.

## Fixed Paths

- Repo root: `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation`
- Spec path: `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/plans/RQ011_plan_v0_onsite_full_universe_readiness_20260622.md`
- Run root: `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ011_onsite_full_universe_readiness/RQ011_2_onsite_readiness_20260623T201415+0800_efdd75a5`
- Derived root: `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/onsite_competition/RQ011_onsite_full_universe_readiness/RQ011_2_onsite_readiness_20260623T201415+0800_efdd75a5`
- Prior run reused read-only: `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ011_onsite_full_universe_readiness/RQ011_1_onsite_readiness_20260623T104838+0800_20aaee57`
- Canonical data root: `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/onsite_competition/all_teams_dataset`

## Canonical Data Semantics

- Per-team `诊断报告.pdf` is the authoritative source for per-scenario scores, sub-item score breakdown, and deduction reasons.
- A scenario score of `0` means a collision in that scenario.
- The competition has 15 scenarios, identical across all teams.
- Each team has one run; there are no repeated runs.
- `onsite_*.log` records the `avalgorithm` flag (`1` = autonomous, otherwise manual).
- The competition is autonomous-only, so all units default to autonomous.
- Logs exist for only some teams and are optional; PDF authority supersedes logs.

## Recorded Data Root Structure

- Table files: `all_scenario_scores.csv, all_session_manifest.csv, all_team_summary.csv, materialized_analysis_files.csv, validation_summary.csv`
- Area directories: `beijing=8, shanghai=12`
- Team directories: `20`
- Support-material PDF files under teams: `20`
- Exact `诊断报告.pdf` filename matches observed at structure level: `8` (20 support-material PDFs exist total; Phase 2 must normalize Shanghai filename variants without changing the authority rule)
- `onsite_*.log` files: `7`

Phase 0 records structure only; it did not parse CSVs, PDFs, logs, replay payloads, IPV predictor tables, or IPV-outcome results.

## Allowed Final Readiness States

- `READY_FULL_UNIVERSE`
- `READY_WITH_FROZEN_EXCLUSIONS`
- `TOP5_ONLY`
- `RUN_LEVEL_NOT_IDENTIFIABLE`
- `BLOCKED_MAPPING`

## Phase-1 Import

Phase 1 is complete by reuse: verified addendum artifacts were copied from `reports/studies/RQ011_onsite_full_universe_readiness/RQ011_1_onsite_readiness_20260623T104838+0800_20aaee57/02_process/01_plan_review/` into `02_process/01_plan_review/`. See `02_process/01_plan_review/IMPORTED_FROM.md`.
