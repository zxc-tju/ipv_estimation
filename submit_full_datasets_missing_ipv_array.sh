#!/bin/bash
#SBATCH --job-name=miss_ipv_full
#SBATCH --comment="Rerun missing full-dataset IPV cases, 4 nodes x 96 CPUs"
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

SCRIPT="process_interhub.py"
INPUT_BUILDER="tools/build_missing_ipv_rerun_input.py"
CSV_PATH="interhub_traj_lane/0_raw_data/full_datasets/missing_ipv_rerun_input.csv"
PKL_ROOT="interhub_traj_lane/0_raw_data/full_datasets/pkl"
OUTPUT_ROOT="interhub_traj_lane/1_ipv_estimation_results/full_datasets/missing_ipv_rerun"

SHARD_COUNT=${SHARD_COUNT:-4}
SHARD_INDEX=${SLURM_ARRAY_TASK_ID}
WORKERS=${WORKERS:-96}
ONLY_INCOMPLETE=${ONLY_INCOMPLETE:-1}
CASE_TIMEOUT_SECONDS=${CASE_TIMEOUT_SECONDS:-1800}
REFERENCE_CLIP_MARGIN_M=${REFERENCE_CLIP_MARGIN_M:-60}
REFERENCE_MAX_POINTS=${REFERENCE_MAX_POINTS:-40}
REFERENCE_SMOOTH_POINTS=${REFERENCE_SMOOTH_POINTS:-40}

if [ "$SHARD_INDEX" -ge "$SHARD_COUNT" ]; then
    echo "SLURM_ARRAY_TASK_ID=$SHARD_INDEX is outside SHARD_COUNT=$SHARD_COUNT" >&2
    exit 1
fi

EXTRA_ARGS=()
MODE="all missing-IPV rows"
if [ "$ONLY_INCOMPLETE" = "1" ]; then
    EXTRA_ARGS+=(--only-incomplete)
    MODE="incomplete missing-IPV rows only"
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
echo "Case timeout seconds: ${CASE_TIMEOUT_SECONDS}"
echo "Reference clip margin m: ${REFERENCE_CLIP_MARGIN_M}"
echo "Reference max points: ${REFERENCE_MAX_POINTS}"
echo "Reference smooth points: ${REFERENCE_SMOOTH_POINTS}"
echo "Multiprocessing start method: fork"
echo "Plots: disabled"
echo "nuPlan sampling: 20Hz -> 10Hz"
echo "Failure policy: failed/timeout cases are recorded and do not stop other cases"
echo "=========================================="

if [ ! -f "$SCRIPT" ]; then
    echo "Missing script: $SCRIPT" >&2
    exit 1
fi

if [ ! -f "$INPUT_BUILDER" ]; then
    echo "Missing input builder: $INPUT_BUILDER" >&2
    exit 1
fi

if [ ! -d "$PKL_ROOT" ]; then
    echo "Missing pkl root: $PKL_ROOT" >&2
    exit 1
fi

if [ ! -f "$CSV_PATH" ]; then
    echo "Missing rerun CSV; building $CSV_PATH"
    python "$INPUT_BUILDER"
fi

mkdir -p "$OUTPUT_ROOT"

python "$SCRIPT" \
    --skip-preflight \
    --csv "$CSV_PATH" \
    --pkl-root "$PKL_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --shard-index "$SHARD_INDEX" \
    --shard-count "$SHARD_COUNT" \
    --workers "$WORKERS" \
    --max-workers "$WORKERS" \
    --mp-start-method fork \
    --case-timeout-seconds "$CASE_TIMEOUT_SECONDS" \
    --reference-clip-margin-m "$REFERENCE_CLIP_MARGIN_M" \
    --reference-max-points "$REFERENCE_MAX_POINTS" \
    --reference-smooth-points "$REFERENCE_SMOOTH_POINTS" \
    --no-plots \
    "${EXTRA_ARGS[@]}"
