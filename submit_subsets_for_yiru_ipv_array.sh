#!/bin/bash
#SBATCH --job-name=yiru_ipv_array
#SBATCH --comment="subsets_for_yiru key-agent IPV estimation shards (6 nodes, 96 workers each)"
#SBATCH --partition=intel
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=5G
#SBATCH --array=0-5

# Optional: load modules or activate environments here
# module load python/3.x
# source /path/to/venv/bin/activate

source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate ipv

SCRIPT="process_subsets_for_yiru_ipv.py"
SHARD_COUNT=${SHARD_COUNT:-6}
SHARD_INDEX=$SLURM_ARRAY_TASK_ID
WORKERS=${WORKERS:-96}

echo "=========================================="
echo "SLURM Job: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Script: $SCRIPT"
echo "Node: $SLURM_NODELIST"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Shard: $SHARD_INDEX / $SHARD_COUNT"
echo "Workers: $WORKERS"
echo "=========================================="

if [ ! -f "$SCRIPT" ]; then
    echo "Missing script: $SCRIPT" >&2
    exit 1
fi

python "$SCRIPT" \
    --skip-preflight \
    --shard-index "$SHARD_INDEX" \
    --shard-count "$SHARD_COUNT" \
    --workers "$WORKERS" \
    --max-workers "$WORKERS"
