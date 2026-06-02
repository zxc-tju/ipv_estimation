#!/bin/bash
#SBATCH --job-name=yiru_ipv_merge
#SBATCH --comment="Merge subsets_for_yiru IPV shard outputs"
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
SHARD_COUNT=${SHARD_COUNT:-6}

echo "=========================================="
echo "SLURM Job: ${SLURM_JOB_ID}"
echo "Script: $SCRIPT"
echo "Shard count: $SHARD_COUNT"
echo "=========================================="

if [ ! -f "$SCRIPT" ]; then
    echo "Missing script: $SCRIPT" >&2
    exit 1
fi

python "$SCRIPT" \
    --skip-preflight \
    --merge-shards \
    --shard-count "$SHARD_COUNT" \
    --log-workflow
