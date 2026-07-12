#!/usr/bin/env bash
# RQ014 FL05 v1.3: invoke the atomic stdlib indexer with registered HPC roots.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FIXED_SLURM_PYTHON="/share/home/u25310231/.conda/envs/ipv/bin/python"
if [[ -n "${RQ014_FL05_VERIFIED_SLURM_WRAPPER:-}" ]]; then
  if [[ "${PYTHON_BIN:-}" != "${FIXED_SLURM_PYTHON}" ]]; then
    echo "FL05_FATAL: verified Slurm execution requires fixed PYTHON_BIN=${FIXED_SLURM_PYTHON}" >&2
    exit 2
  fi
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

if [[ $# -eq 0 ]]; then
  cat >&2 <<'EOF'
Usage: RQ014_forensics_hpc_fl05_indexer_v1p3.sh \
  --bundle-root /absolute/fl05_bundle [--audit-format json|csv] \
  [--root /absolute/input ...]

When no --root is supplied, the two frozen RQ010B HPC roots are used. The
indexer is read-only on those roots, writes an immutable generation, and then
atomically replaces only the bundle's CURRENT pointer.
EOF
  exit 64
fi

has_root=false
for argument in "$@"; do
  if [[ "${argument}" == "--root" || "${argument}" == --root=* ]]; then
    has_root=true
    break
  fi
done

root_arguments=()
if [[ "${has_root}" == false ]]; then
  root_arguments=(
    --root /share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis
    --root /share/home/u25310231/ZXC/RQ010B_wod_e2e/results
  )
fi

exec "${PYTHON_BIN}" \
  "${SCRIPT_DIR}/RQ014_forensics_hpc_fl05_indexer_v1p3.py" \
  "${root_arguments[@]}" \
  "$@"
