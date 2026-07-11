#!/usr/bin/env bash
set -euo pipefail

BASE="${HPC_SOCIALITY_ROOT:-/share/home/u25310231/ZXC/sociality_estimation}"
REPO="$BASE/code/repo"

exec python3 "$REPO/scripts/hpc/prepare_research_run.py" "$@"
