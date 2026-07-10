#!/usr/bin/env bash
#SBATCH --job-name=zxc-sociality-verifier-env
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/share/home/u25310231/ZXC/sociality_estimation/logs/zxc-sociality-verifier-env_%j.out
#SBATCH --error=/share/home/u25310231/ZXC/sociality_estimation/logs/zxc-sociality-verifier-env_%j.err

set -euo pipefail

BASE="/share/home/u25310231/ZXC/sociality_estimation"
REPO="$BASE/code/repo"
ENV="$BASE/envs/ipv-verifier"
MODEL="$BASE/checkpoints/rq009_m3/m3_scorer.joblib"

test -d "$REPO/.git"
test -f "$MODEL"
cd "$REPO"
source /share/apps/miniconda3/etc/profile.d/conda.sh
conda create --yes --prefix "$ENV" python=3.9.6 pip
"$ENV/bin/python" -m pip install --upgrade pip
"$ENV/bin/python" -m pip install -r "$REPO/environments/requirements-ipv-verifier.txt"
SOCIALITY_M3_SCORER="$MODEL" PYTHONPATH="$REPO/src" "$ENV/bin/python" -m pytest -q \
  "$REPO/tests/test_verifier_runtime.py"
