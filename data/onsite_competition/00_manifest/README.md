# Onsite Competition Data Organization Manifest

Generated: 2026-06-18T17:22:55 local time.

This folder is a lightweight index for the onsite competition payload. It does
not replace the raw data. The raw and former top-five subset payloads are now
archived under `archived/onsite_competition_raw_and_top5_subset_20260623/`.

## Files

- `team_manifest.csv`: one row per top-level team folder in the active area folders.
- `session_manifest.csv`: one row per materialized replay session directory named like `6923-1766197775`.
- `file_inventory.csv`: one row per file under the active area folders, with inferred file role and team/session columns.
- `score_team_coverage.csv`: official scored-team coverage from `archived/anno_trans.csv` plus the archived score CSVs, joined to active folders.

## Canonical Reading Pattern

Use `session_manifest.csv` for replay processing. The stable grain is one row per `(area, team_dir, session_id)` and the `session_relative_path` points to the folder containing replay logs.

Required replay logs are:

- `monitor.log`
- `simulation_trajectory.log`
- `vehicle_perception_simulation_trajectory.log`
- `vehicle_trajectory.log`

`vehicle_perception_trajectory.log` is optional: it appears in some replay exports and is tracked in `optional_logs_present`.

## Organization Actions Performed

Shanghai replay zip files were extracted into direct session folders under their team folder when the session folder was missing. Existing same-size extracted logs were skipped. No raw zips, videos, PDFs, SQL dumps, or archived materials were deleted or moved.

## Current Structure Notes

- Shanghai has replay zips for the scored teams; after extraction, scored Shanghai teams have materialized replay session folders. Some Shanghai teams have multiple sessions/reruns.
- Beijing mostly arrived already materialized as session folders. `5-BIT_Site` has scores/media but no materialized replay session directory. `8-???` is present as media/PDF/text but is not matched to the archived official score table.
- `????/app.zip` and `????/tjjhs_db.sql` are area-level operational/platform artifacts, not per-team replay sessions.
- `archived/` preserves the prior lightweight/pointer-era package and the archived score/mapping files used for team-code joins.

## Recommended Downstream Use

1. Use `data/onsite_competition/all_teams_dataset/` as the current all-team
   analysis package.
2. Filter `score_team_coverage.csv` to `has_materialized_replay = True` for replay-based analysis.
3. Join scores by `team_code` from `team_manifest.csv` or `session_manifest.csv`.
4. For multi-run teams, keep `session_id` as part of the key instead of assuming one run per team.
5. Ignore `hidden_metadata` rows in `file_inventory.csv` for analysis.
