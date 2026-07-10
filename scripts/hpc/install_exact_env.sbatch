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
ENV="$BASE/envs/ipv-exact"

test -d "$REPO/.git"
cd "$REPO"
source /share/apps/miniconda3/etc/profile.d/conda.sh
conda create --yes --prefix "$ENV" python=3.9.6 pip
"$ENV/bin/python" -m pip install --upgrade pip
"$ENV/bin/python" -m pip install -r \
  "$REPO/environments/requirements-ipv-exact-linux-64.txt"
"$ENV/bin/python" -m pip freeze > "$BASE/manifests/ipv-exact-freeze.txt"
RQ_IPV_PARITY_STRICT=1 PYTHONPATH="$REPO/src" "$ENV/bin/python" -m pytest -q \
  "$REPO/tests/test_ipv_estimator_parity.py"
