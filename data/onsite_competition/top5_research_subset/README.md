# Top-5 Research Subset

Generated: 2026-06-18T18:50:36 local time.

This package contains the useful analysis data for the top five teams in each onsite competition area, selected by `score_mean_comprehensive` from `00_manifest/score_team_coverage.csv`.

## Included Teams

Shanghai top five:

1. T11 / wsd / 89.44
2. T6 / thicv2025 / 85.61
3. T5 / limitless / 81.79
4. T7 / SMU / 80.39
5. T8 / ADVRC / 79.37

Beijing top five:

1. T17 / panda / 89.42
2. T14 / bps / 89.29
3. T16 / autoailab / 88.38
4. T15 / wulala / 83.98
5. T20 / mm / 82.17

## Layout

- `teams/<area>/<rank>_<team_code>_<official_name>/sessions/<session_id>/`: replay logs for trajectory/world-state analysis.
- `teams/<area>/<rank>_<team_code>_<official_name>/support_materials/`: diagnosis PDFs, team info, onsite runtime logs, and appeal/evidence files when present.
- `tables/top5_selection_summary.csv`: selected teams and total score summary.
- `tables/top5_scenario_scores.csv`: scenario-level score rows for the selected teams.
- `tables/top5_session_manifest.csv`: session-level replay manifest for this subset.
- `tables/materialized_analysis_files.csv`: source-to-subset file map plus storage mode.
- `media_index/video_paths.csv`: source paths for videos; videos are intentionally not duplicated in this subset.

## Storage Note

Most analysis files are hardlinked from the raw onsite payload to avoid duplicating large replay logs. Treat files here as analysis inputs and avoid editing them in place. If a downstream workflow needs writable files, copy them into a separate working directory first.

## Replay Logs

Each selected session contains the required logs when available:

- `monitor.log`
- `simulation_trajectory.log`
- `vehicle_perception_simulation_trajectory.log`
- `vehicle_trajectory.log`

Optional `vehicle_perception_trajectory.log` is included for teams whose source session has it.
