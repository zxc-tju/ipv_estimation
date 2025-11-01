#!/bin/bash
#SBATCH --job-name=interhub_ipv
#SBATCH --comment="Interhub IPV estimation (3 nodes per JSON)"
#SBATCH --partition=intel
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=5G
#SBATCH --array=0-6

# Optional: load modules or activate environments here
# module load python/3.x
# source /path/to/venv/bin/activate

JSON_FILES=(
    "trajectory_data_interaction_single.json"
    "trajectory_data_interaction_multi.json"
    "trajectory_data_waymo_0-299.json"
    "trajectory_data_waymo_300-499.json"
    "trajectory_data_waymo_500-799.json"
    "trajectory_data_waymo_800-999.json"
    "trajectory_data_lyft_train_full.json"
)

SCRIPT="interhub_traj_lane/process_interhub.py"
TARGET=${JSON_FILES[$SLURM_ARRAY_TASK_ID]}
WORKERS=$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))   # 3 nodes ¡Á 96 cores = 288

if [ -z "$TARGET" ]; then
    echo "No JSON mapped for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

echo "Processing $TARGET using $SLURM_NTASKS tasks and $WORKERS workers"

# Launch the Python job once; other ranks exit immediately.
srun --ntasks=1 --cpus-per-task=$WORKERS python "$SCRIPT" --workers "$WORKERS" "interhub_traj_lane/$TARGET"
