#!/usr/bin/env bash
#SBATCH --job-name=zxc-rq017-m1
#SBATCH --partition=intel,fata
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/zxc-rq017-m1_%A_%a.out
#SBATCH --error=logs/zxc-rq017-m1_%A_%a.err

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export RQ017_PROCESS_CACHE_BASE="${SCRIPT_DIR}/process_cache"
export MPLCONFIGDIR="${SCRIPT_DIR}/process_cache/mpl/${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
export XDG_CACHE_HOME="${SCRIPT_DIR}/process_cache/xdg/${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

MODE="${RQ017_MODE:?RQ017_MODE must be canary or full}"
MANIFEST_LIST="${SCRIPT_DIR}/inputs/${MODE}_manifest_list.txt"
MANIFEST_REL="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${MANIFEST_LIST}")"
if [[ -z "${MANIFEST_REL}" ]]; then
  echo "No manifest for task ${SLURM_ARRAY_TASK_ID} in ${MANIFEST_LIST}" >&2
  exit 2
fi

cd "${SCRIPT_DIR}/repo_stage"
PYTHON="/share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/bin/python"
export PYTHONPATH="${SCRIPT_DIR}/pydeps:${PWD}/src:${PWD}/.codex-fleet/rq015b-repair/work:${PYTHONPATH:-}"
export RQ017_OUTPUT_ROOT="${SCRIPT_DIR}/outputs"

"${PYTHON}" .codex-fleet/rq017-onsite-materializer/work/M1/rq017_onsite_materializer.py run-shard \
  --manifest "${MANIFEST_REL}" \
  --workers "${RQ017_WORKERS:-6}"
