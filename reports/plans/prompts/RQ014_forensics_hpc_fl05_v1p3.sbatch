#!/usr/bin/env bash
#SBATCH --job-name=zxc-rq014-fl05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/share/home/u25310231/ZXC/RQ014_recovery/zxc-rq014-fl05-%j.out
#SBATCH --error=/share/home/u25310231/ZXC/RQ014_recovery/zxc-rq014-fl05-%j.err

# CPU-only fallback for an FL05 tree larger than the frozen 200 MiB login limit.
# Submit only after placing the checksum-bound repository at the fixed code root:
#   sbatch RQ014_forensics_hpc_fl05_v1p3.sbatch RUN_ID
set -euo pipefail
umask 077

if [[ $# -ne 1 ]]; then
  echo "Usage: sbatch $0 RUN_ID" >&2
  exit 64
fi
run_id="$1"
if [[ ! "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ ]]; then
  echo "FL05_FATAL: RUN_ID must match ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$" >&2
  exit 64
fi
if [[ "${run_id}" == "code" ]]; then
  echo "FL05_FATAL: RUN_ID=code is reserved for the checksum-bound repository" >&2
  exit 64
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "FL05_FATAL: this wrapper must run inside a Slurm allocation" >&2
  exit 64
fi
if [[ "${SLURM_JOB_NAME:-}" != "zxc-rq014-fl05" ]]; then
  echo "FL05_FATAL: expected SLURM_JOB_NAME=zxc-rq014-fl05" >&2
  exit 64
fi

durable_root="/share/home/u25310231/ZXC/RQ014_recovery"
repo_root="${durable_root}/code/repo"
run_root="${durable_root}/${run_id}"
bundle_root="${run_root}/fl05_bundle"
checksum_manifest="${repo_root}/reports/plans/RQ014_plan_v1p3_checksums_20260711.sha256"
parser_rel="reports/plans/prompts/RQ014_forensics_hpc_fl05_indexer_v1p3.py"
shell_rel="reports/plans/prompts/RQ014_forensics_hpc_fl05_indexer_v1p3.sh"
sbatch_rel="reports/plans/prompts/RQ014_forensics_hpc_fl05_v1p3.sbatch"
python_bin="/share/home/u25310231/.conda/envs/ipv/bin/python"

for required_directory in "${durable_root}" "${repo_root}"; do
  if [[ ! -d "${required_directory}" || -L "${required_directory}" ]]; then
    echo "FL05_FATAL: required fixed directory missing or symlinked: ${required_directory}" >&2
    exit 2
  fi
done
if [[ ! -f "${python_bin}" || ! -x "${python_bin}" ]]; then
  echo "FL05_FATAL: fixed Python entrypoint is missing or not executable: ${python_bin}" >&2
  exit 2
fi
if ! python_realpath="$(readlink -f -- "${python_bin}")"; then
  echo "FL05_FATAL: cannot resolve fixed Python entrypoint: ${python_bin}" >&2
  exit 2
fi
if [[ -z "${python_realpath}" || ! -f "${python_realpath}" || -L "${python_realpath}" || ! -x "${python_realpath}" ]]; then
  echo "FL05_FATAL: fixed Python entrypoint does not resolve to a regular executable: ${python_bin}" >&2
  exit 2
fi
python_version="$("${python_realpath}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
if ! "${python_realpath}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)'; then
  echo "FL05_FATAL: fixed Python interpreter must be >=3.9, observed ${python_version}" >&2
  exit 2
fi
python_sha256="$(sha256sum "${python_realpath}" | awk '{print $1}')"
if [[ ! -f "${checksum_manifest}" || -L "${checksum_manifest}" ]]; then
  echo "FL05_FATAL: v1.3 checksum manifest missing or symlinked: ${checksum_manifest}" >&2
  exit 2
fi

cd "${repo_root}"
sha256sum --check --strict "${checksum_manifest}"
running_sbatch="$(readlink -f -- "${BASH_SOURCE[0]}")"
registered_sbatch="$(readlink -f -- "${repo_root}/${sbatch_rel}")"
if [[ "${running_sbatch}" != "${registered_sbatch}" ]]; then
  echo "FL05_FATAL: submit the checksum-bound sbatch file from ${repo_root}/${sbatch_rel}" >&2
  exit 2
fi
for required_rel in "${parser_rel}" "${shell_rel}" "${sbatch_rel}"; do
  if [[ ! -f "${required_rel}" || -L "${required_rel}" ]]; then
    echo "FL05_FATAL: checksum-bound executable missing or symlinked: ${required_rel}" >&2
    exit 2
  fi
  match_count="$(awk -v target="${required_rel}" '$2 == target {count++} END {print count+0}' "${checksum_manifest}")"
  if [[ "${match_count}" != "1" ]]; then
    echo "FL05_FATAL: checksum manifest must contain exactly one entry for ${required_rel}" >&2
    exit 2
  fi
  expected_sha="$(awk -v target="${required_rel}" '$2 == target {print $1}' "${checksum_manifest}")"
  actual_sha="$(sha256sum "${required_rel}" | awk '{print $1}')"
  if [[ "${actual_sha}" != "${expected_sha}" ]]; then
    echo "FL05_FATAL: executable checksum mismatch: ${required_rel}" >&2
    exit 2
  fi
done

if [[ -e "${run_root}" && ( ! -d "${run_root}" || -L "${run_root}" ) ]]; then
  echo "FL05_FATAL: fixed run root exists but is not a safe directory: ${run_root}" >&2
  exit 2
fi
mkdir -p -- "${run_root}/logs" "${bundle_root}"
if [[ -L "${run_root}" || -L "${run_root}/logs" || -L "${bundle_root}" ]]; then
  echo "FL05_FATAL: symlinked run output path is forbidden" >&2
  exit 2
fi
cd "${run_root}"

# Preserve independent durable stdout/stderr in addition to Slurm's submission logs.
exec >"${run_root}/logs/zxc-rq014-fl05-${SLURM_JOB_ID}.out" \
     2>"${run_root}/logs/zxc-rq014-fl05-${SLURM_JOB_ID}.err"

export RQ014_FL05_VERIFIED_SLURM_WRAPPER="v1p3:${SLURM_JOB_ID}:zxc-rq014-fl05"
export RQ014_FL05_PYTHON_REALPATH="${python_realpath}"
export RQ014_FL05_PYTHON_SHA256="${python_sha256}"
export PYTHON_BIN="${python_bin}"
exec bash "${repo_root}/${shell_rel}" \
  --root /share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis \
  --root /share/home/u25310231/ZXC/RQ010B_wod_e2e/results \
  --bundle-root "${bundle_root}" \
  --audit-format json \
  --compute-byte-budget 4294967296 \
  --login-byte-budget 209715200
