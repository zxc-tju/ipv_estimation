#!/bin/bash
#SBATCH --job-name=yiru_ipv_fix_merge
#SBATCH --comment="Merge incomplete-case IPV rerun outputs"
#SBATCH --partition=intel
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate ipv

SCRIPT="process_subsets_for_yiru_ipv.py"
SHARD_COUNT=${SHARD_COUNT:-4}
RESULT_ROOT="interhub_traj_lane/1_ipv_estimation_results/subsets_for_yiru"
BASE_CSV="${RESULT_ROOT}/selected_interactive_segments_equalized_with_ipv.csv"

echo "=========================================="
echo "SLURM Job: ${SLURM_JOB_ID}"
echo "Script: $SCRIPT"
echo "Mode: merge incomplete-case rerun"
echo "Shard count: $SHARD_COUNT"
echo "Base CSV: $BASE_CSV"
echo "=========================================="

if [ ! -f "$SCRIPT" ]; then
    echo "Missing script: $SCRIPT" >&2
    exit 1
fi

if [ ! -f "$BASE_CSV" ]; then
    echo "Missing base CSV for patch merge: $BASE_CSV" >&2
    exit 1
fi

python "$SCRIPT" \
    --skip-preflight \
    --merge-shards \
    --only-incomplete \
    --shard-count "$SHARD_COUNT" \
    --merge-base-csv "$BASE_CSV" \
    --log-workflow
