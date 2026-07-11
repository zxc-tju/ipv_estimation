#!/usr/bin/env bash
#SBATCH --job-name=zxc-sociality-retire-legacy
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=/share/home/u25310231/ZXC/sociality_estimation/logs/zxc-sociality-retire-legacy_%j.out
#SBATCH --error=/share/home/u25310231/ZXC/sociality_estimation/logs/zxc-sociality-retire-legacy_%j.err

set -euo pipefail

BASE="${HPC_SOCIALITY_ROOT:-/share/home/u25310231/ZXC/sociality_estimation}"
REPO="$BASE/code/repo"
test -d "$REPO/.git"
cd "$REPO"
bash scripts/hpc/archive_legacy_checkout.sh
