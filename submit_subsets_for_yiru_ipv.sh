#!/bin/bash
#SBATCH --job-name=yiru_ipv
#SBATCH --comment="subsets_for_yiru key-agent IPV estimation (1 node, 64 workers)"
#SBATCH --partition=intel
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=5G

# Optional: load modules or activate environments here
# module load python/3.x
# source /path/to/venv/bin/activate

source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate ipv

SCRIPT="process_subsets_for_yiru_ipv.py"
WORKERS=${WORKERS:-64}

echo "=========================================="
echo "SLURM Job: ${SLURM_JOB_ID}"
echo "Script: $SCRIPT"
echo "Node: $SLURM_NODELIST"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Workers: $WORKERS"
echo "=========================================="

if [ ! -f "$SCRIPT" ]; then
    echo "Missing script: $SCRIPT" >&2
    exit 1
fi

python "$SCRIPT" --preflight-only

python "$SCRIPT" \
    --workers "$WORKERS" \
    --max-workers "$WORKERS" \
    --log-workflow
