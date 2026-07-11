#!/usr/bin/env bash
#SBATCH --job-name=zxc-sociality-inventory
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=/share/home/u25310231/ZXC/sociality_estimation/logs/zxc-sociality-inventory_%j.out
#SBATCH --error=/share/home/u25310231/ZXC/sociality_estimation/logs/zxc-sociality-inventory_%j.err

set -euo pipefail

BASE="${HPC_SOCIALITY_ROOT:-/share/home/u25310231/ZXC/sociality_estimation}"
REPO="$BASE/code/repo"
RUN_ID="${INVENTORY_ID:-pre_migration_20260711}"

test -d "$REPO/.git"
cd "$REPO"
bash scripts/hpc/inventory_legacy_layout.sh "$RUN_ID"
