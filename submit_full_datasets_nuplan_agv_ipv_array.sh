#!/bin/bash
#SBATCH --job-name=full_ipv_agv
#SBATCH --comment="Full nuPlan + Argoverse/AV2 key-agent IPV shards; nuPlan downsampled to 10Hz"
#SBATCH --partition=intel
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=5G
#SBATCH --array=0-11

set -euo pipefail

source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate ipv

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

SCRIPT="process_subsets_for_yiru_ipv.py"
CSV_PATH="interhub_traj_lane/0_raw_data/full_datasets/nuplan_agv_all/pkl/selected_interactive_segments_nuplan_agv_full.csv"
PKL_ROOT="interhub_traj_lane/0_raw_data/full_datasets/nuplan_agv_all/pkl"
OUTPUT_ROOT="interhub_traj_lane/1_ipv_estimation_results/full_datasets/nuplan_agv_all"

SHARD_COUNT=${SHARD_COUNT:-12}
SHARD_INDEX=${SLURM_ARRAY_TASK_ID}
WORKERS=${WORKERS:-96}
ONLY_INCOMPLETE=${ONLY_INCOMPLETE:-0}

if [ "$SHARD_INDEX" -ge "$SHARD_COUNT" ]; then
    echo "SLURM_ARRAY_TASK_ID=$SHARD_INDEX is outside SHARD_COUNT=$SHARD_COUNT" >&2
    exit 1
fi

EXTRA_ARGS=()
MODE="all rows"
if [ "$ONLY_INCOMPLETE" = "1" ]; then
    EXTRA_ARGS+=(--only-incomplete)
    MODE="incomplete rows only"
fi

echo "=========================================="
echo "SLURM Job: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Script: $SCRIPT"
echo "CSV: $CSV_PATH"
echo "PKL root: $PKL_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Node: $SLURM_NODELIST"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Shard: $SHARD_INDEX / $SHARD_COUNT"
echo "Workers: $WORKERS"
echo "Mode: $MODE"
echo "Plots: disabled"
echo "nuPlan sampling: 20Hz -> 10Hz via dataset downsample factor 2"
echo "Argoverse/AV2 sampling: unchanged"
echo "Time limit: 3 days"
echo "=========================================="

if [ ! -f "$SCRIPT" ]; then
    echo "Missing script: $SCRIPT" >&2
    exit 1
fi

if [ ! -f "$CSV_PATH" ]; then
    echo "Missing CSV: $CSV_PATH" >&2
    exit 1
fi

if [ ! -d "$PKL_ROOT" ]; then
    echo "Missing pkl root: $PKL_ROOT" >&2
    exit 1
fi

python "$SCRIPT" \
    --skip-preflight \
    --csv "$CSV_PATH" \
    --pkl-root "$PKL_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --shard-index "$SHARD_INDEX" \
    --shard-count "$SHARD_COUNT" \
    --workers "$WORKERS" \
    --max-workers "$WORKERS" \
    --no-plots \
    "${EXTRA_ARGS[@]}"
