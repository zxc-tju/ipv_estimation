#!/usr/bin/env bash
#SBATCH --job-name=zxc-sociality-attest-snapshots
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/share/home/u25310231/ZXC/sociality_estimation/logs/zxc-sociality-attest-snapshots_%j.out
#SBATCH --error=/share/home/u25310231/ZXC/sociality_estimation/logs/zxc-sociality-attest-snapshots_%j.err

set -euo pipefail
cd /share/home/u25310231/ZXC/sociality_estimation/code/repo
bash scripts/hpc/attest_migrated_snapshots.sh
