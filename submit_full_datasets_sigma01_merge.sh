#!/bin/bash
#SBATCH --job-name=ipv_s01_merge
#SBATCH --comment="Merge full InterHub IPV sigma=0.1 rerun shards"
#SBATCH --partition=intel
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate ipv

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

SCRIPT="process_interhub.py"
CSV_PATH="interhub_traj_lane/0_raw_data/full_datasets/index.csv"
PKL_ROOT="interhub_traj_lane/0_raw_data/full_datasets/pkl"
OUTPUT_ROOT="interhub_traj_lane/1_ipv_estimation_results/full_datasets/batches/20260612_sigma_0_1_full_rerun"
SHARD_COUNT=${SHARD_COUNT:-4}
ONLY_INCOMPLETE=${ONLY_INCOMPLETE:-1}

MERGE_ARGS=()
MODE="all shard rows"
if [ "$ONLY_INCOMPLETE" = "1" ]; then
    MERGE_ARGS+=(--only-incomplete)
    MODE="incomplete shard rows"
fi

echo "=========================================="
echo "SLURM Job: ${SLURM_JOB_ID}"
echo "Script: $SCRIPT"
echo "CSV: $CSV_PATH"
echo "PKL root: $PKL_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Shard count: $SHARD_COUNT"
echo "Mode: $MODE"
python - <<'PY'
import agent
print(f"agent.sigma={agent.sigma}")
PY
echo "=========================================="

if [ ! -f "$SCRIPT" ]; then
    echo "Missing script: $SCRIPT" >&2
    exit 1
fi

python "$SCRIPT" \
    --skip-preflight \
    --merge-shards \
    --csv "$CSV_PATH" \
    --pkl-root "$PKL_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --shard-count "$SHARD_COUNT" \
    --log-workflow \
    "${MERGE_ARGS[@]}"

echo "Merged CSV: ${OUTPUT_ROOT}/index_with_ipv.csv"
