# HPC missing-IPV rerun commands

This batch is different from the earlier `nuplan_agv_all` run:

- Input cases come from `curated_valid_ipv_cases/index_missing_ipv_cases.csv`.
- A clean HPC input CSV is generated at `interhub_traj_lane/0_raw_data/full_datasets/missing_ipv_rerun_input.csv`.
- Raw pkl files are under `interhub_traj_lane/0_raw_data/full_datasets/pkl`.
- The pkl schema is `scene_unique_id -> {timestamps, trajectories, lane_centerlines}`, not the older `metadata/vehicles/road_info` schema. `process_subsets_for_yiru_ipv.py` now handles both.
- All 7,226 missing-IPV cases have raw pkl data, scene ids, key agents, covered frame windows, and usable xy positions.
- nuPlan still runs at 10Hz through the built-in downsample factor 2.
- Reference handling is conservative by default: clip to the observed motion neighborhood, cap reference length, and smooth to a bounded point count.

## 1. Upload code and input files

Run locally from Git Bash or WSL. Adjust `LOCAL_REPO` if needed.

```bash
LOCAL_REPO="/c/Users/xiaocongzhao/OneDrive/Desktop/Projects/1_Codes/2_sociality_estimation"
REMOTE_HOST="u25310231@logini.tongji.edu.cn"
REMOTE_REPO="/share/home/u25310231/ZXC/ipv_estimation"

rsync -avz --progress -e "ssh -p 10022" \
  "$LOCAL_REPO/process_subsets_for_yiru_ipv.py" \
  "$LOCAL_REPO/submit_full_datasets_missing_ipv_array.sh" \
  "$LOCAL_REPO/submit_full_datasets_missing_ipv_merge.sh" \
  "$REMOTE_HOST:$REMOTE_REPO/"

ssh -p 10022 "$REMOTE_HOST" "mkdir -p $REMOTE_REPO/tools $REMOTE_REPO/interhub_traj_lane/1_ipv_estimation_results/full_datasets/curated_valid_ipv_cases $REMOTE_REPO/interhub_traj_lane/0_raw_data/full_datasets/pkl"

rsync -avz --progress -e "ssh -p 10022" \
  "$LOCAL_REPO/tools/build_missing_ipv_rerun_input.py" \
  "$REMOTE_HOST:$REMOTE_REPO/tools/"

rsync -avz --progress -e "ssh -p 10022" \
  "$LOCAL_REPO/interhub_traj_lane/1_ipv_estimation_results/full_datasets/curated_valid_ipv_cases/index_missing_ipv_cases.csv" \
  "$REMOTE_HOST:$REMOTE_REPO/interhub_traj_lane/1_ipv_estimation_results/full_datasets/curated_valid_ipv_cases/"

rsync -avz --progress -e "ssh -p 10022" \
  "$LOCAL_REPO/interhub_traj_lane/0_raw_data/full_datasets/pkl/" \
  "$REMOTE_HOST:$REMOTE_REPO/interhub_traj_lane/0_raw_data/full_datasets/pkl/"
```

If the pkl files are already on HPC, skip the last `rsync`.

## 2. Prepare and preflight on HPC

```bash
ssh -p 10022 u25310231@logini.tongji.edu.cn
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

## 3. Submit 4 nodes x 96 CPUs

```bash
SHARD_COUNT=4 \
WORKERS=96 \
CASE_TIMEOUT_SECONDS=1800 \
REFERENCE_CLIP_MARGIN_M=60 \
REFERENCE_MAX_POINTS=40 \
REFERENCE_SMOOTH_POINTS=40 \
sbatch submit_full_datasets_missing_ipv_array.sh
```

The Slurm array is `0-3`. Each array task requests one node and 96 CPUs:

```text
4 array tasks x 1 node/task x 96 CPUs/task = 4 nodes, 384 CPUs total
```

## 4. Monitor

```bash
squeue -u u25310231

tail -f miss_ipv_full_<JOBID>_0.out
tail -f miss_ipv_full_<JOBID>_0.err
```

Check current completion without computing:

```bash
python process_subsets_for_yiru_ipv.py \
  --skip-preflight \
  --scan-incomplete-only \
  --csv interhub_traj_lane/0_raw_data/full_datasets/missing_ipv_rerun_input.csv \
  --pkl-root interhub_traj_lane/0_raw_data/full_datasets/pkl \
  --output-root interhub_traj_lane/1_ipv_estimation_results/full_datasets/missing_ipv_rerun
```

If a job is interrupted, submit the same array command again. `--only-incomplete` is enabled by default in the Slurm script, so completed `metadata.json + ipv_results.xlsx` cases are skipped.

## 5. Merge shard CSVs

After all four shard jobs finish:

```bash
SHARD_COUNT=4 sbatch submit_full_datasets_missing_ipv_merge.sh
```

Or run merge directly in the login shell:

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

## 6. Failure policy

Individual failures and timeouts are written as failed case artifacts and shard CSV rows. They do not stop the other cases.

Useful knobs:

```bash
# More aggressive timeout protection
CASE_TIMEOUT_SECONDS=900 sbatch submit_full_datasets_missing_ipv_array.sh

# More conservative references if long references still appear
REFERENCE_CLIP_MARGIN_M=50 REFERENCE_MAX_POINTS=30 REFERENCE_SMOOTH_POINTS=30 sbatch submit_full_datasets_missing_ipv_array.sh
```
