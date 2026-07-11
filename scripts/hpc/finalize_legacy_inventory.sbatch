#!/usr/bin/env bash
#SBATCH --job-name=zxc-sociality-inv-final
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/share/home/u25310231/ZXC/sociality_estimation/logs/zxc-sociality-inv-final_%j.out
#SBATCH --error=/share/home/u25310231/ZXC/sociality_estimation/logs/zxc-sociality-inv-final_%j.err

set -euo pipefail

BASE="${HPC_SOCIALITY_ROOT:-/share/home/u25310231/ZXC/sociality_estimation}"
REPO="$BASE/code/repo"
RUN_ID="${INVENTORY_ID:-pre_migration_20260711}"

test -d "$REPO/.git"
cd "$REPO"
HASH_WORKERS="${SLURM_CPUS_PER_TASK:-8}" \
  bash scripts/hpc/finalize_legacy_inventory.sh "$RUN_ID"
