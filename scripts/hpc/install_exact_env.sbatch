#!/usr/bin/env bash
#SBATCH --job-name=zxc-sociality-exact-env
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/share/home/u25310231/ZXC/sociality_estimation/logs/zxc-sociality-exact-env_%j.out
#SBATCH --error=/share/home/u25310231/ZXC/sociality_estimation/logs/zxc-sociality-exact-env_%j.err

set -euo pipefail

BASE="/share/home/u25310231/ZXC/sociality_estimation"
REPO="$BASE/code/repo"
SOURCE_ENV="/share/home/u25310231/.conda/envs/ipv"
ENV="$BASE/envs/ipv-exact-sigma01"

test -d "$REPO/.git"
test -x "$SOURCE_ENV/bin/python"
cd "$REPO"
source /share/apps/miniconda3/etc/profile.d/conda.sh
conda list --prefix "$SOURCE_ENV" --explicit > \
  "$BASE/manifests/ipv-exact-conda-explicit-linux-64.txt"
if [[ ! -x "$ENV/bin/python" ]]; then
  conda create --yes --prefix "$ENV" --clone "$SOURCE_ENV"
fi
"$ENV/bin/python" -m pip freeze > "$BASE/manifests/ipv-exact-freeze.txt"
PYTHONPATH="$REPO/src" "$ENV/bin/python" \
  "$REPO/scripts/verify_ipv_estimator_fixture.py"
