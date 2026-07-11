#!/usr/bin/env bash
#SBATCH --job-name=zxc-sociality-migrate
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/share/home/u25310231/ZXC/sociality_estimation/logs/zxc-sociality-migrate_%j.out
#SBATCH --error=/share/home/u25310231/ZXC/sociality_estimation/logs/zxc-sociality-migrate_%j.err

set -euo pipefail

BASE="${HPC_SOCIALITY_ROOT:-/share/home/u25310231/ZXC/sociality_estimation}"
REPO="$BASE/code/repo"
test -d "$REPO/.git"
cd "$REPO"
VERIFY_WORKERS="${SLURM_CPUS_PER_TASK:-8}" \
  bash scripts/hpc/migrate_legacy_payloads.sh
