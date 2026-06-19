# Legacy Script Archive

These scripts are preserved for reference, but they are no longer active root-level entrypoints for the current project layout.

## Archived Scripts

- `process_interhub_json_legacy.py`: Legacy InterHub JSON pipeline. It expects `trajectory_data_*.json` directly under the old `interhub_traj_lane/` layout and writes to `interhub_traj_lane/ipv_estimation/`. Current InterHub work uses CSV/pkl inputs under `data/interhub/raw/` and `pipelines/interhub/process_interhub.py`.
- `process_subsets_for_yiru_ipv.py`: Backward-compatible wrapper around the old root `process_interhub.py`. Current subset and full-dataset runs should call `pipelines/interhub/process_interhub.py` directly.
- `process_argoverse.py`: Legacy Argoverse CSV pipeline. The current repository keeps Argoverse data under `archived/argoverse/0_souce_data/`, while this script expects `0_souce_data/` beside the script.
- `batch_process_ipv.py`: Old post-processing script for writing `mean_ipv` into metadata under `interhub_traj_lane/ipv_estimation_results/`, a path not used by the current CSV/pkl pipeline.

## Restore Notes

If one of these workflows is needed again, copy the script back to the repository root, review its hard-coded input/output paths, and run a small representative case before using it for full processing.
