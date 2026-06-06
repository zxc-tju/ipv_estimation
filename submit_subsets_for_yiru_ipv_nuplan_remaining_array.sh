#!/bin/bash
#SBATCH --job-name=yiru_ipv_np_rem
#SBATCH --comment="Continue remaining incomplete nuplan_train IPV cases at 10Hz (4 nodes)"
#SBATCH --partition=intel
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=5G
#SBATCH --array=0-3

set -euo pipefail

source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate ipv

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-1}
export SCIPY_NUM_THREADS=${SCIPY_NUM_THREADS:-1}
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

SCRIPT="process_subsets_for_yiru_ipv.py"
CSV_PATH="interhub_traj_lane/0_raw_data/subsets_for_yiru/selected_interactive_segments_equalized.csv"
PKL_ROOT="interhub_traj_lane/0_raw_data/subsets_for_yiru/pkl"
OUTPUT_ROOT="interhub_traj_lane/1_ipv_estimation_results/subsets_for_yiru"

SHARD_COUNT=${SHARD_COUNT:-4}
SHARD_INDEX=${SLURM_ARRAY_TASK_ID}
WORKERS=${WORKERS:-96}
ONLY_INCOMPLETE=${ONLY_INCOMPLETE:-1}

if [ "$SHARD_INDEX" -ge "$SHARD_COUNT" ]; then
    echo "SLURM_ARRAY_TASK_ID=$SHARD_INDEX is outside SHARD_COUNT=$SHARD_COUNT" >&2
    exit 1
fi

EXTRA_ARGS=()
MODE="all nuplan_train rows"
if [ "$ONLY_INCOMPLETE" = "1" ]; then
    EXTRA_ARGS+=(--only-incomplete)
    MODE="incomplete nuplan_train rows only"
fi

echo "=========================================="
echo "SLURM Job: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Script: $SCRIPT"
echo "CSV: $CSV_PATH"
echo "PKL root: $PKL_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Node: $SLURM_NODELIST"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-unknown}"
echo "Working dir: $(pwd)"
echo "Python: $(command -v python)"
python --version
echo "Mode: $MODE"
echo "Shard: $SHARD_INDEX / $SHARD_COUNT"
echo "Workers: $WORKERS"
echo "Completed-case skip: ${ONLY_INCOMPLETE}"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "OPENBLAS_NUM_THREADS: $OPENBLAS_NUM_THREADS"
echo "MKL_NUM_THREADS: $MKL_NUM_THREADS"
echo "NUMEXPR_NUM_THREADS: $NUMEXPR_NUM_THREADS"
echo "Multiprocessing start method: fork"
echo "Plots: disabled"
echo "NuPlan downsampling: 20Hz -> 10Hz"
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

mkdir -p "$OUTPUT_ROOT"

python "$SCRIPT" \
    --skip-preflight \
    --csv "$CSV_PATH" \
    --pkl-root "$PKL_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --dataset-filter nuplan_train \
    --shard-index "$SHARD_INDEX" \
    --shard-count "$SHARD_COUNT" \
    --workers "$WORKERS" \
    --max-workers "$WORKERS" \
    --mp-start-method fork \
    --no-plots \
    "${EXTRA_ARGS[@]}"
