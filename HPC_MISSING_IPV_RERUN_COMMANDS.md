# HPC missing-IPV rerun commands

This plan assumes Codex can SSH to the cluster with the local alias:

```bash
ssh tongji-hpc
```

Remote repository:

```text
/share/home/u25310231/ZXC/ipv_estimation
```

## What is different in this batch

- The source list is the single curated all-case manifest:
  `interhub_traj_lane/1_ipv_estimation_results/full_datasets/curated_valid_ipv_cases/valid_cases_manifest.csv`.
- The rerun input is generated on HPC by filtering:
  `curation_status == missing_ipv_from_index`.
- The generated rerun CSV is:
  `interhub_traj_lane/0_raw_data/full_datasets/missing_ipv_rerun_input.csv`.
- The raw pkl schema is `scene_unique_id -> {timestamps, trajectories, lane_centerlines}`, not the earlier `metadata/vehicles/road_info` schema. `process_subsets_for_yiru_ipv.py` handles both schemas.
- All 7,226 missing-IPV rows were audited locally as raw-usable.
- nuPlan still runs at 10Hz through the built-in downsample factor 2.
- This batch has long references and heavier cases. Defaults are conservative:
  `REFERENCE_CLIP_MARGIN_M=60`, `REFERENCE_MAX_POINTS=40`, `REFERENCE_SMOOTH_POINTS=40`, `CASE_TIMEOUT_SECONDS=1800`.
- Individual failed or timed-out cases are recorded and do not stop the whole shard.

## 1. Update code on HPC

Run from local PowerShell or any terminal that can use the `tongji-hpc` SSH alias:

```bash
ssh tongji-hpc "cd /share/home/u25310231/ZXC/ipv_estimation && git pull origin main"
```

Confirm the scripts exist:

```bash
ssh tongji-hpc "cd /share/home/u25310231/ZXC/ipv_estimation && \
  ls submit_full_datasets_missing_ipv_array.sh \
     submit_full_datasets_missing_ipv_merge.sh \
     tools/build_missing_ipv_rerun_input.py"
```

## 2. Sync the minimum data needed

The process files under `curated_valid_ipv_cases` are archived locally, so only the all-case manifest is needed.

From local PowerShell:

```powershell
$LOCAL_REPO = "C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation"
$REMOTE_REPO = "/share/home/u25310231/ZXC/ipv_estimation"

ssh tongji-hpc "mkdir -p $REMOTE_REPO/interhub_traj_lane/1_ipv_estimation_results/full_datasets/curated_valid_ipv_cases $REMOTE_REPO/interhub_traj_lane/0_raw_data/full_datasets/pkl"

scp `
  "$LOCAL_REPO\interhub_traj_lane\1_ipv_estimation_results\full_datasets\curated_valid_ipv_cases\valid_cases_manifest.csv" `
  "tongji-hpc:$REMOTE_REPO/interhub_traj_lane/1_ipv_estimation_results/full_datasets/curated_valid_ipv_cases/"
```

HPC already has the earlier nuPlan/AV2 pkl files under:

```text
interhub_traj_lane/0_raw_data/full_datasets/nuplan_agv_all/pkl
```

Link them into the new unified pkl root:

```bash
ssh tongji-hpc 'cd /share/home/u25310231/ZXC/ipv_estimation && \
  mkdir -p interhub_traj_lane/0_raw_data/full_datasets/pkl && \
  cd interhub_traj_lane/0_raw_data/full_datasets/pkl && \
  for f in ../nuplan_agv_all/pkl/*.pkl; do ln -sfn "$f" "$(basename "$f")"; done'
```

Then sync the pkl files that are not already on HPC: Lyft and Waymo.

```powershell
scp `
  "$LOCAL_REPO\interhub_traj_lane\0_raw_data\full_datasets\pkl\lyft_train_full.pkl" `
  "$LOCAL_REPO\interhub_traj_lane\0_raw_data\full_datasets\pkl\waymo_0-299.pkl" `
  "$LOCAL_REPO\interhub_traj_lane\0_raw_data\full_datasets\pkl\waymo_300-499.pkl" `
  "$LOCAL_REPO\interhub_traj_lane\0_raw_data\full_datasets\pkl\waymo_500-799.pkl" `
  "$LOCAL_REPO\interhub_traj_lane\0_raw_data\full_datasets\pkl\waymo_800-999.pkl" `
  "tongji-hpc:$REMOTE_REPO/interhub_traj_lane/0_raw_data/full_datasets/pkl/"
```

Check the pkl root. Expected count is 15 files or symlinks:

```bash
ssh tongji-hpc "cd /share/home/u25310231/ZXC/ipv_estimation && \
  find interhub_traj_lane/0_raw_data/full_datasets/pkl -maxdepth 1 \( -type f -o -type l \) -name '*.pkl' | wc -l"
```

## 3. Generate rerun input and preflight

```bash
ssh tongji-hpc
cd /share/home/u25310231/ZXC/ipv_estimation
source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate ipv

python tools/build_missing_ipv_rerun_input.py

python process_subsets_for_yiru_ipv.py \
  --csv interhub_traj_lane/0_raw_data/full_datasets/missing_ipv_rerun_input.csv \
  --pkl-root interhub_traj_lane/0_raw_data/full_datasets/pkl \
  --output-root interhub_traj_lane/1_ipv_estimation_results/full_datasets/missing_ipv_rerun \
  --reference-clip-margin-m 60 \
  --reference-max-points 40 \
  --reference-smooth-points 40 \
  --preflight-only
```

Expected core preflight result:

```text
csv_rows=7226
matched_rows=7226
unmatched_rows=0
```

## 4. Submit 4 nodes x 96 CPUs

```bash
SHARD_COUNT=4 \
WORKERS=96 \
CASE_TIMEOUT_SECONDS=1800 \
REFERENCE_CLIP_MARGIN_M=60 \
REFERENCE_MAX_POINTS=40 \
REFERENCE_SMOOTH_POINTS=40 \
sbatch submit_full_datasets_missing_ipv_array.sh
```

The Slurm array is `0-3`:

```text
4 array tasks x 1 node/task x 96 CPUs/task = 4 nodes, 384 CPUs total
```

## 5. Monitor and resume

```bash
squeue -u u25310231
```

Check completion without computing:

```bash
python process_subsets_for_yiru_ipv.py \
  --skip-preflight \
  --scan-incomplete-only \
  --csv interhub_traj_lane/0_raw_data/full_datasets/missing_ipv_rerun_input.csv \
  --pkl-root interhub_traj_lane/0_raw_data/full_datasets/pkl \
  --output-root interhub_traj_lane/1_ipv_estimation_results/full_datasets/missing_ipv_rerun
```

If a job is interrupted, submit the same array command again. `--only-incomplete` is enabled by default in the Slurm script, so completed `metadata.json + ipv_results.xlsx` cases are skipped.

## 6. Merge shard CSVs

After all four shard jobs finish:

```bash
SHARD_COUNT=4 sbatch submit_full_datasets_missing_ipv_merge.sh
```

Or merge directly on the login node:

```bash
python process_subsets_for_yiru_ipv.py \
  --skip-preflight \
  --merge-shards \
  --only-incomplete \
  --csv interhub_traj_lane/0_raw_data/full_datasets/missing_ipv_rerun_input.csv \
  --pkl-root interhub_traj_lane/0_raw_data/full_datasets/pkl \
  --output-root interhub_traj_lane/1_ipv_estimation_results/full_datasets/missing_ipv_rerun \
  --shard-count 4
```

Merged output:

```text
interhub_traj_lane/1_ipv_estimation_results/full_datasets/missing_ipv_rerun/selected_interactive_segments_equalized_with_ipv.csv
```

## 7. Failure policy and knobs

Individual failures and timeouts are written as failed case artifacts and shard CSV rows. They do not stop other cases.

Useful knobs:

```bash
# More aggressive timeout protection
CASE_TIMEOUT_SECONDS=900 sbatch submit_full_datasets_missing_ipv_array.sh

# More conservative references if long references still appear
REFERENCE_CLIP_MARGIN_M=50 REFERENCE_MAX_POINTS=30 REFERENCE_SMOOTH_POINTS=30 sbatch submit_full_datasets_missing_ipv_array.sh
```
