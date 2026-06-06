#!/bin/bash
#SBATCH --job-name=full_ipv_agv_merge
#SBATCH --comment="Merge full nuPlan + Argoverse/AV2 IPV shard outputs"
#SBATCH --partition=intel
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate ipv

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

SCRIPT="process_subsets_for_yiru_ipv.py"
CSV_PATH="interhub_traj_lane/0_raw_data/full_datasets/nuplan_agv_all/pkl/selected_interactive_segments_nuplan_agv_full.csv"
PKL_ROOT="interhub_traj_lane/0_raw_data/full_datasets/nuplan_agv_all/pkl"
OUTPUT_ROOT="interhub_traj_lane/1_ipv_estimation_results/full_datasets/nuplan_agv_all"
EXCLUDE_CSV="interhub_traj_lane/0_raw_data/subsets_for_yiru/selected_interactive_segments_equalized.csv"
FINAL_CSV="${OUTPUT_ROOT}/selected_interactive_segments_nuplan_agv_full_with_ipv.csv"

SHARD_COUNT=${SHARD_COUNT:-12}
ONLY_INCOMPLETE=${ONLY_INCOMPLETE:-0}
BASE_CSV=${BASE_CSV:-}

MERGE_ARGS=()
MODE="all shard rows"
if [ "$ONLY_INCOMPLETE" = "1" ]; then
    MODE="incomplete shard rows"
    BASE_CSV=${BASE_CSV:-"${OUTPUT_ROOT}/selected_interactive_segments_equalized_with_ipv.csv"}
    MERGE_ARGS+=(--only-incomplete)
fi

if [ -n "$BASE_CSV" ]; then
    if [ ! -f "$BASE_CSV" ]; then
        echo "Missing base CSV for patch merge: $BASE_CSV" >&2
        exit 1
    fi
    MERGE_ARGS+=(--merge-base-csv "$BASE_CSV")
fi

echo "=========================================="
echo "SLURM Job: ${SLURM_JOB_ID}"
echo "Script: $SCRIPT"
echo "CSV: $CSV_PATH"
echo "PKL root: $PKL_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Exclude CSV: $EXCLUDE_CSV"
echo "Shard count: $SHARD_COUNT"
echo "Mode: $MODE"
echo "Base CSV: ${BASE_CSV:-none}"
echo "Friendly final CSV: $FINAL_CSV"
echo "=========================================="

if [ ! -f "$SCRIPT" ]; then
    echo "Missing script: $SCRIPT" >&2
    exit 1
fi

if [ ! -f "$EXCLUDE_CSV" ]; then
    echo "Missing exclude CSV: $EXCLUDE_CSV" >&2
    exit 1
fi

python "$SCRIPT" \
    --skip-preflight \
    --csv "$CSV_PATH" \
    --pkl-root "$PKL_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --exclude-csv "$EXCLUDE_CSV" \
    --merge-shards \
    --shard-count "$SHARD_COUNT" \
    --log-workflow \
    "${MERGE_ARGS[@]}"

cp "${OUTPUT_ROOT}/selected_interactive_segments_equalized_with_ipv.csv" "$FINAL_CSV"
echo "Wrote $FINAL_CSV"
