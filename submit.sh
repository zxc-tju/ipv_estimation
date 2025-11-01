#!/bin/bash
#SBATCH --job-name=interhub_ipv
#SBATCH --comment="Interhub IPV estimation (1 node, 96 cores per JSON)"
#SBATCH --partition=intel
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=5G
#SBATCH --array=0-6

# Optional: load modules or activate environments here
# module load python/3.x
# source /path/to/venv/bin/activate

source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate ipv

JSON_FILES=(
    "trajectory_data_interaction_single.json"
    "trajectory_data_interaction_multi.json"
    "trajectory_data_waymo_0-299.json"
    "trajectory_data_waymo_300-499.json"
    "trajectory_data_waymo_500-799.json"
    "trajectory_data_waymo_800-999.json"
    "trajectory_data_lyft_train_full.json"
)

SCRIPT="process_interhub.py"
TARGET=${JSON_FILES[$SLURM_ARRAY_TASK_ID]}
WORKERS=$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))   # 1 node × 96 cores = 96

if [ -z "$TARGET" ]; then
    echo "No JSON mapped for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

echo "=========================================="
echo "SLURM Job: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Processing: $TARGET"
echo "Node: $SLURM_NODELIST"
echo "Workers: $WORKERS"
echo "=========================================="

# Launch the Python job with all available CPUs
python "$SCRIPT" --workers "$WORKERS" "interhub_traj_lane/$TARGET"
