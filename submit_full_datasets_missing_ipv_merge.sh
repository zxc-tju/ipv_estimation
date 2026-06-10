#!/bin/bash
#SBATCH --job-name=miss_ipv_merge
#SBATCH --comment="Merge missing full-dataset IPV rerun shards"
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

SCRIPT="process_subsets_for_yiru_ipv.py"
CSV_PATH="interhub_traj_lane/0_raw_data/full_datasets/missing_ipv_rerun_input.csv"
PKL_ROOT="interhub_traj_lane/0_raw_data/full_datasets/pkl"
OUTPUT_ROOT="interhub_traj_lane/1_ipv_estimation_results/full_datasets/missing_ipv_rerun"
SHARD_COUNT=${SHARD_COUNT:-4}
ONLY_INCOMPLETE=${ONLY_INCOMPLETE:-1}

EXTRA_ARGS=()
if [ "$ONLY_INCOMPLETE" = "1" ]; then
    EXTRA_ARGS+=(--only-incomplete)
fi

echo "=========================================="
echo "SLURM Job: ${SLURM_JOB_ID}"
echo "CSV: $CSV_PATH"
echo "Output root: $OUTPUT_ROOT"
echo "Shard count: $SHARD_COUNT"
echo "Only incomplete suffix: $ONLY_INCOMPLETE"
echo "=========================================="

python "$SCRIPT" \
    --skip-preflight \
    --merge-shards \
    --csv "$CSV_PATH" \
    --pkl-root "$PKL_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --shard-count "$SHARD_COUNT" \
    "${EXTRA_ARGS[@]}"
