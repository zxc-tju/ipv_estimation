# subsets_for_yiru Key-Agent IPV Estimation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Estimate IPV for the two CSV-designated key agents in every `subsets_for_yiru` interaction case, save per-case artifacts, and write a CSV copy with mean IPV values.

**Architecture:** Use a dedicated batch script, `process_subsets_for_yiru_ipv.py`, that indexes pkl events by CSV metadata, extracts only the two `key_agents`, derives lane references from pkl road data, reuses `estimate_ipv_pair()`, supports worker benchmarking, and writes per-case plus CSV-level outputs.

**Tech Stack:** Python 3.9, pandas, numpy, matplotlib, pickle, ProcessPoolExecutor, existing `ipv_estimation.py`.

---

## Summary

- Input CSV: `interhub_traj_lane/0_raw_data/subsets_for_yiru/selected_interactive_segments_equalized.csv`.
- Input pkl root: `interhub_traj_lane/0_raw_data/subsets_for_yiru/pkl`.
- Output root: `interhub_traj_lane/1_ipv_estimation_results/subsets_for_yiru`.
- Current data facts:
  - CSV rows: 7500.
  - Dataset counts: `nuplan_train=2500`, `waymo_train=2500`, `av2_motion_forecasting=2500`.
  - pkl files: 8 files, including `waymo_0-299.pkl`.
  - pkl events: 7500.
  - CSV-to-pkl matching: 7500/7500 by `folder + scenario_idx + key_agents + track_id`.
  - Event types: 6369 `two`, 1131 `multi`.
  - Key-agent references: 15000/15000 from `road_info['all_lane_centerlines'] + lane_ids`; fallback is only for future malformed rows.
- Local worker cap: 24 logical processors * 70% = 16 max workers.

## Implementation Changes

- Add `process_subsets_for_yiru_ipv.py` with:
  - `--preflight-only` to report CSV rows, pkl files/events, match counts, event type counts, and reference coverage.
  - `--limit N --workers K` for smoke runs.
  - `--benchmark-workers` to test `[1, 2, 4, 8, 12, 16]`.
  - `--workers auto` for full processing using benchmark recommendation.
- CSV copy behavior:
  - Preserve the original `key_agents` column as the only key-agent id source.
  - Do not add `ipv_key_agent_1_id` or `ipv_key_agent_2_id`.
  - Add only IPV/status columns: `ipv_key_agent_1_mean`, `ipv_key_agent_1_error_mean`, `ipv_key_agent_2_mean`, `ipv_key_agent_2_error_mean`, `ipv_result_status`, `ipv_result_case_dir`, `ipv_result_error`, `ipv_pkl_file`, `ipv_segment_id`, `ipv_reference_source_1`, `ipv_reference_source_2`.
  - Interpret `key_agents` order as output order: first id maps to key agent 1, second id maps to key agent 2.
- Case behavior:
  - For multi-agent pkl events, ignore all non-key agents.
  - Align the two key agents by common timestamps before estimation.
  - Build references by concatenating lane centerlines from `lane_ids`; use unique `frame_lane_ids` only if `lane_ids` cannot produce a reference.
  - Save each successful case as `data/ipv_results.xlsx`, `data/metadata.json`, and `fig/ipv_curve.png`.
  - Use short hashed case directories to avoid Windows path-length failures.

## Test Plan

- Unit tests cover:
  - CSV+pkl event indexing.
  - Lane-reference construction from `lane_ids` and `frame_lane_ids`.
  - Observed-trajectory reference fallback.
  - Common-timestamp alignment.
  - Mean IPV calculation using only valid rows after `min_observation`.
  - CSV copy columns without duplicate key-agent id columns.
  - Worker recommendation.
  - Short case directory naming.
- Verification commands:
  - `python -m pytest tests/test_process_subsets_for_yiru_ipv.py -q`
  - `python process_subsets_for_yiru_ipv.py --preflight-only`
  - `conda run -n ipv_estimate python process_subsets_for_yiru_ipv.py --limit 1 --workers 1`
- Full run command:
  - `conda run -n ipv_estimate python process_subsets_for_yiru_ipv.py --workers auto --log-workflow`

## Assumptions

- Run real IPV estimation in the `ipv_estimate` conda environment, because base Python lacks `shapely`.
- CSV writes mean IPV, not final IPV.
- Default output excludes virtual-track diagnostics.
- Single-case failures do not interrupt batch processing; status and error are written to the CSV copy and case metadata.
