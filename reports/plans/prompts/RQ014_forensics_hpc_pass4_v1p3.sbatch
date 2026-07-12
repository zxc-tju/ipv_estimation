#!/usr/bin/env bash
#SBATCH --job-name=zxc-rq014-g0-f05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/share/home/u25310231/ZXC/RQ014_recovery/zxc-rq014-g0-f05-%j.out
#SBATCH --error=/share/home/u25310231/ZXC/RQ014_recovery/zxc-rq014-g0-f05-%j.err

# Checksum-bound compute-node fallback for an F05 closure inventory above 200 MiB.
# Usage: sbatch RQ014_forensics_hpc_pass4_v1p3.sbatch RUN_ID
set -euo pipefail
umask 077

if [[ $# -ne 1 ]]; then
  echo "Usage: sbatch $0 RUN_ID" >&2
  exit 64
fi
run_id="$1"
if [[ ! "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ || "${run_id}" == "code" ]]; then
  echo "G0_F05_FATAL: unsafe or reserved RUN_ID" >&2
  exit 64
fi
if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_JOB_NAME:-}" != "zxc-rq014-g0-f05" ]]; then
  echo "G0_F05_FATAL: expected the registered zxc-rq014-g0-f05 Slurm allocation" >&2
  exit 64
fi

durable_root="/share/home/u25310231/ZXC/RQ014_recovery"
repo_root="${durable_root}/code/repo"
run_root="${durable_root}/${run_id}"
output_root="${run_root}/g0_pass4"
checksum_manifest="${repo_root}/reports/plans/RQ014_plan_v1p3_checksums_20260711.sha256"
script_rel="reports/plans/prompts/RQ014_forensics_hpc_pass4_v1p3.sh"
sbatch_rel="reports/plans/prompts/RQ014_forensics_hpc_pass4_v1p3.sbatch"
python_bin="/share/home/u25310231/.conda/envs/ipv/bin/python"
results_root="/share/home/u25310231/ZXC/RQ010B_wod_e2e/results"
phase3_root="/share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/phase3_preference_test"
job_log="/share/home/u25310231/ZXC/RQ010B_wod_e2e/logs/zxc-rq010b-ipv-rating_1746009.out"

for directory in "${durable_root}" "${repo_root}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "G0_F05_FATAL: fixed directory missing or symlinked: ${directory}" >&2
    exit 2
  fi
done
if [[ ! -f "${checksum_manifest}" || -L "${checksum_manifest}" ]]; then
  echo "G0_F05_FATAL: checksum manifest missing or symlinked" >&2
  exit 2
fi
if [[ ! -f "${python_bin}" || ! -x "${python_bin}" ]]; then
  echo "G0_F05_FATAL: fixed Python entrypoint unavailable: ${python_bin}" >&2
  exit 2
fi
if ! python_realpath="$(readlink -f -- "${python_bin}")"; then
  echo "G0_F05_FATAL: cannot resolve fixed Python entrypoint: ${python_bin}" >&2
  exit 2
fi
if [[ -z "${python_realpath}" || ! -f "${python_realpath}" || -L "${python_realpath}" || ! -x "${python_realpath}" ]]; then
  echo "G0_F05_FATAL: fixed Python entrypoint does not resolve to a regular executable: ${python_bin}" >&2
  exit 2
fi
python_version="$("${python_realpath}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
if ! "${python_realpath}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)'; then
  echo "G0_F05_FATAL: fixed Python interpreter must be >=3.9" >&2
  exit 2
fi
python_sha256="$(sha256sum "${python_realpath}" | awk '{print $1}')"

cd "${repo_root}"
sha256sum --check --strict "${checksum_manifest}"
running_sbatch="$(readlink -f -- "${BASH_SOURCE[0]}")"
registered_sbatch="$(readlink -f -- "${repo_root}/${sbatch_rel}")"
if [[ "${running_sbatch}" != "${registered_sbatch}" ]]; then
  echo "G0_F05_FATAL: submit the checksum-bound registered sbatch file" >&2
  exit 2
fi
for required_rel in "${script_rel}" "${sbatch_rel}"; do
  match_count="$(awk -v target="${required_rel}" '$2 == target {count++} END {print count+0}' "${checksum_manifest}")"
  if [[ "${match_count}" != "1" ]]; then
    echo "G0_F05_FATAL: checksum manifest must contain exactly one ${required_rel} entry" >&2
    exit 2
  fi
  expected_sha="$(awk -v target="${required_rel}" '$2 == target {print $1}' "${checksum_manifest}")"
  actual_sha="$(sha256sum "${required_rel}" | awk '{print $1}')"
  if [[ "${actual_sha}" != "${expected_sha}" ]]; then
    echo "G0_F05_FATAL: checksum mismatch: ${required_rel}" >&2
    exit 2
  fi
done

if [[ -e "${run_root}" && ( ! -d "${run_root}" || -L "${run_root}" ) ]]; then
  echo "G0_F05_FATAL: run root exists but is unsafe: ${run_root}" >&2
  exit 2
fi
mkdir -p -- "${run_root}/logs" "${output_root}"
if [[ -L "${run_root}" || -L "${run_root}/logs" || -L "${output_root}" ]]; then
  echo "G0_F05_FATAL: symlinked output path is forbidden" >&2
  exit 2
fi

exec >"${run_root}/logs/zxc-rq014-g0-f05-${SLURM_JOB_ID}.out" \
     2>"${run_root}/logs/zxc-rq014-g0-f05-${SLURM_JOB_ID}.err"
export RQ014_G0_F05_VERIFIED_SLURM_WRAPPER="v1p3:${SLURM_JOB_ID}:zxc-rq014-g0-f05"
export RQ014_G0_PYTHON_REALPATH="${python_realpath}"
export RQ014_G0_PYTHON_SHA256="${python_sha256}"
export RQ014_G0_PYTHON_VERSION="${python_version}"
export PYTHON_BIN="${python_bin}"
exec bash "${repo_root}/${script_rel}" \
  --output-dir "${output_root}" \
  --run-id "${run_id}" \
  --results-root "${results_root}" \
  --phase3-root "${phase3_root}" \
  --job-log "${job_log}"
