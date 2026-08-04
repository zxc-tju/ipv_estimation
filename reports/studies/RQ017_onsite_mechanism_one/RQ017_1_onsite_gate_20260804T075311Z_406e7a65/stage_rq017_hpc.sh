#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
REMOTE="tongji-hpc"
SSH_OPTS=(-o BatchMode=yes -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o ConnectTimeout=12)
SSH=(ssh "${SSH_OPTS[@]}")
RSYNC_RSH="ssh -o BatchMode=yes -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o ConnectTimeout=12"
REMOTE_PROJECT="/share/home/u25310231/ZXC/sociality_estimation"
RQ017_BASE="${REMOTE_PROJECT}/work_dirs/RQ017"
PYTHON="/share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/bin/python"
HPC_PKL_ROOT="${REMOTE_PROJECT}/data/interhub/snapshots/interhub_legacy_20260711_v1/full_datasets/pkl"
REMOTE_ANCHOR_CSV="/share/home/u25310231/ZXC/rq012b_onsite_ipv_20260627T202508/outputs/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.csv"
REMOTE_TIMESERIES_CSV="/share/home/u25310231/ZXC/rq012b_onsite_ipv_20260627T202508/outputs/onsite_anchors_multi/onsite_ipv_timeseries_multi_allvalid.csv"
LATEST_WORKDIR_FILE="${SCRIPT_DIR}/latest_hpc_workdir.txt"

remote_quote() {
  printf "%q" "$1"
}

hpc_workdir() {
  if [[ -n "${HPC_WORKDIR:-}" ]]; then
    printf '%s\n' "${HPC_WORKDIR}"
  elif [[ -s "${LATEST_WORKDIR_FILE}" ]]; then
    cat "${LATEST_WORKDIR_FILE}"
  else
    echo "No HPC_WORKDIR set and ${LATEST_WORKDIR_FILE} is missing" >&2
    exit 2
  fi
}

copy_file() {
  local src="$1"
  local dst="$2"
  rsync -a -e "${RSYNC_RSH}" "${src}" "${REMOTE}:${dst}"
}

copy_dir() {
  local src="$1"
  local dst="$2"
  rsync -a -e "${RSYNC_RSH}" "${src%/}/" "${REMOTE}:${dst%/}/"
}

stage() {
  cd "${LOCAL_ROOT}"
  python3 .codex-fleet/rq017-onsite-materializer/work/M1/rq017_onsite_materializer.py preflight \
    --output .codex-fleet/rq017-onsite-materializer/work/M1/measurement_contract.json
  python3 .codex-fleet/rq017-onsite-materializer/work/M1/rq017_onsite_materializer.py protected-sha \
    --output .codex-fleet/rq017-onsite-materializer/work/M1/local_protected_sha256.json

  local run_id="rq017_onsite_materializer_$(date -u +%Y%m%dT%H%M%SZ)"
  local workdir="${RQ017_BASE}/${run_id}"
  printf '%s\n' "${workdir}" > "${LATEST_WORKDIR_FILE}"

  "${SSH[@]}" "${REMOTE}" "test ! -e $(remote_quote "${workdir}") && mkdir -p $(remote_quote "${workdir}")/{repo_stage,pydeps,inputs,logs,outputs,process_cache} $(remote_quote "${workdir}")/repo_stage/{src,pipelines/interhub,configs,.codex-fleet/rq017-onsite-materializer/work,.codex-fleet/rq015b-repair/work,.codex-fleet/rq015g-hpc-resolve/work,.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/validation,.codex-fleet/rq016c-human-only-envelope/work/H2,data/interhub/raw/full_datasets}"

  copy_dir "${LOCAL_ROOT}/src/sociality_estimation" "${workdir}/repo_stage/src/sociality_estimation"
  copy_file "${LOCAL_ROOT}/pipelines/interhub/process_interhub.py" "${workdir}/repo_stage/pipelines/interhub/process_interhub.py"
  copy_dir "${LOCAL_ROOT}/configs" "${workdir}/repo_stage/configs"
  copy_dir "${LOCAL_ROOT}/.codex-fleet/rq017-onsite-materializer/work/M1" "${workdir}/repo_stage/.codex-fleet/rq017-onsite-materializer/work/M1"
  copy_file "${LOCAL_ROOT}/.codex-fleet/rq015b-repair/work/run_b1_rq015b.py" "${workdir}/repo_stage/.codex-fleet/rq015b-repair/work/run_b1_rq015b.py"
  copy_file "${LOCAL_ROOT}/.codex-fleet/rq015b-repair/work/run_b2_rq015b.py" "${workdir}/repo_stage/.codex-fleet/rq015b-repair/work/run_b2_rq015b.py"
  copy_file "${LOCAL_ROOT}/.codex-fleet/rq015b-repair/work/sample_v1.csv" "${workdir}/repo_stage/.codex-fleet/rq015b-repair/work/sample_v1.csv"
  copy_file "${LOCAL_ROOT}/.codex-fleet/rq015g-hpc-resolve/work/anchor_mse_hpc.csv" "${workdir}/repo_stage/.codex-fleet/rq015g-hpc-resolve/work/anchor_mse_hpc.csv"
  copy_file "${LOCAL_ROOT}/.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py" "${workdir}/repo_stage/.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py"
  copy_file "${LOCAL_ROOT}/.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/validation/g_anchor_hpc_baseline.json" "${workdir}/repo_stage/.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/validation/g_anchor_hpc_baseline.json"
  copy_file "${LOCAL_ROOT}/.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet" "${workdir}/repo_stage/.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet"
  copy_file "${SCRIPT_DIR}/measurement_contract.json" "${workdir}/inputs/measurement_contract.json"
  copy_file "${SCRIPT_DIR}/local_protected_sha256.json" "${workdir}/inputs/local_protected_sha256.json"
  copy_file "${SCRIPT_DIR}/submit_rq017_env_parity.sbatch" "${workdir}/submit_rq017_env_parity.sbatch"
  copy_file "${SCRIPT_DIR}/submit_rq017_array.sbatch" "${workdir}/submit_rq017_array.sbatch"

  "${SSH[@]}" "${REMOTE}" "rm -rf $(remote_quote "${workdir}/repo_stage/data/interhub/raw/full_datasets/pkl") && ln -s $(remote_quote "${HPC_PKL_ROOT}") $(remote_quote "${workdir}/repo_stage/data/interhub/raw/full_datasets/pkl")"

  "${SSH[@]}" "${REMOTE}" "PYTHONPATH=$(remote_quote "${workdir}/pydeps") $(remote_quote "${PYTHON}") -c 'import pyarrow' >/dev/null 2>&1 || $(remote_quote "${PYTHON}") -m pip install --no-input --no-deps --target $(remote_quote "${workdir}/pydeps") 'pyarrow==12.0.1'"

  "${SSH[@]}" "${REMOTE}" "cd $(remote_quote "${workdir}/repo_stage") && PYTHONPATH=$(remote_quote "${workdir}/pydeps:${workdir}/repo_stage/src:${workdir}/repo_stage/.codex-fleet/rq015b-repair/work") $(remote_quote "${PYTHON}") .codex-fleet/rq017-onsite-materializer/work/M1/rq017_onsite_materializer.py prepare-inputs --anchor-csv $(remote_quote "${REMOTE_ANCHOR_CSV}") --timeseries-csv $(remote_quote "${REMOTE_TIMESERIES_CSV}") --inputs-dir $(remote_quote "${workdir}/inputs") --shard-size ${RQ017_SHARD_SIZE:-500}"

  "${SSH[@]}" "${REMOTE}" "cd $(remote_quote "${workdir}/repo_stage") && PYTHONPATH=$(remote_quote "${workdir}/pydeps:${workdir}/repo_stage/src:${workdir}/repo_stage/.codex-fleet/rq015b-repair/work") $(remote_quote "${PYTHON}") .codex-fleet/rq017-onsite-materializer/work/M1/rq017_onsite_materializer.py cluster-capacity --workers 6 --mem-mb 49152 --output $(remote_quote "${workdir}/cluster_capacity_pre_full.json")"
  "${SSH[@]}" "${REMOTE}" "sinfo -h -N -p intel,fata -o '%N|%P|%T|%c|%m|%e' > $(remote_quote "${workdir}/cluster_snapshot_pre_submit.txt") || true"

  printf 'HPC_WORKDIR=%s\n' "${workdir}"
}

submit_env() {
  local workdir
  workdir="$(hpc_workdir)"
  "${SSH[@]}" "${REMOTE}" "cd $(remote_quote "${workdir}") && sbatch --wait submit_rq017_env_parity.sbatch"
  rsync -a -e "${RSYNC_RSH}" "${REMOTE}:${workdir}/env_parity.json" "${SCRIPT_DIR}/env_parity.json"
}

submit_canary() {
  local workdir
  workdir="$(hpc_workdir)"
  "${SSH[@]}" "${REMOTE}" "test ! -e $(remote_quote "${workdir}/outputs/canary")"
  local out
  out="$("${SSH[@]}" "${REMOTE}" "cd $(remote_quote "${workdir}") && sbatch --wait --array=1-2%2 --export=ALL,RQ017_MODE=canary,RQ017_WORKERS=6 submit_rq017_array.sbatch")"
  printf '%s\n' "${out}" | tee "${SCRIPT_DIR}/latest_canary_sbatch_output.txt"
  local job_id
  job_id="$(printf '%s\n' "${out}" | awk '/Submitted batch job/ {print $NF}' | tail -n 1)"
  printf '%s\n' "${job_id}" > "${SCRIPT_DIR}/latest_canary_job_id.txt"
  "${SSH[@]}" "${REMOTE}" "cd $(remote_quote "${workdir}/repo_stage") && PYTHONPATH=$(remote_quote "${workdir}/pydeps:${workdir}/repo_stage/src:${workdir}/repo_stage/.codex-fleet/rq015b-repair/work") RQ017_OUTPUT_ROOT=$(remote_quote "${workdir}/outputs") $(remote_quote "${PYTHON}") .codex-fleet/rq017-onsite-materializer/work/M1/rq017_onsite_materializer.py validate-outputs --l1-root $(remote_quote "${workdir}/outputs/canary/l1_v1") --dryrun .codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet --output $(remote_quote "${workdir}/outputs/canary_validation.json") --mode canary --include-sentinel"
  rsync -a -e "${RSYNC_RSH}" "${REMOTE}:${workdir}/outputs/canary_validation.json" "${SCRIPT_DIR}/canary_validation.json"
}

submit_full() {
  local workdir
  workdir="$(hpc_workdir)"
  "${SSH[@]}" "${REMOTE}" "test ! -e $(remote_quote "${workdir}/outputs/full")"
  local count
  count="$("${SSH[@]}" "${REMOTE}" "wc -l < $(remote_quote "${workdir}/inputs/full_manifest_list.txt")" | tr -d '[:space:]')"
  local concurrency
  concurrency="$("${SSH[@]}" "${REMOTE}" "$(remote_quote "${PYTHON}") - <<'PY' $(remote_quote "${workdir}/cluster_capacity_pre_full.json")
import json, sys
obj=json.load(open(sys.argv[1]))
print(max(1, int(obj['recommended_array_concurrency'])))
PY
")"
  local out
  out="$("${SSH[@]}" "${REMOTE}" "cd $(remote_quote "${workdir}") && sbatch --wait --array=1-${count}%${concurrency} --export=ALL,RQ017_MODE=full,RQ017_WORKERS=6 submit_rq017_array.sbatch")"
  printf '%s\n' "${out}" | tee "${SCRIPT_DIR}/latest_full_sbatch_output.txt"
  local job_id
  job_id="$(printf '%s\n' "${out}" | awk '/Submitted batch job/ {print $NF}' | tail -n 1)"
  printf '%s\n' "${job_id}" > "${SCRIPT_DIR}/latest_full_job_id.txt"
  printf '1-%s%%%s\n' "${count}" "${concurrency}" > "${SCRIPT_DIR}/latest_full_array_shape.txt"
  "${SSH[@]}" "${REMOTE}" "sacct -j $(remote_quote "${job_id}") --format=JobID,JobName%30,Partition,NodeList,State,Elapsed,AllocCPUS,MaxRSS -P > $(remote_quote "${workdir}/outputs/sacct_full.txt")"
  "${SSH[@]}" "${REMOTE}" "! grep -E '(^|\\|)amd(\\||$)' $(remote_quote "${workdir}/outputs/sacct_full.txt")"
  "${SSH[@]}" "${REMOTE}" "cd $(remote_quote "${workdir}/repo_stage") && PYTHONPATH=$(remote_quote "${workdir}/pydeps:${workdir}/repo_stage/src:${workdir}/repo_stage/.codex-fleet/rq015b-repair/work") RQ017_OUTPUT_ROOT=$(remote_quote "${workdir}/outputs") $(remote_quote "${PYTHON}") .codex-fleet/rq017-onsite-materializer/work/M1/rq017_onsite_materializer.py validate-outputs --l1-root $(remote_quote "${workdir}/outputs/full/l1_v1") --dryrun .codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet --output $(remote_quote "${workdir}/outputs/full_validation.json") --expected-rows 67861 --mode full"
  rsync -a -e "${RSYNC_RSH}" "${REMOTE}:${workdir}/outputs/full_validation.json" "${SCRIPT_DIR}/full_validation.remote.json"
  rsync -a -e "${RSYNC_RSH}" "${REMOTE}:${workdir}/outputs/sacct_full.txt" "${SCRIPT_DIR}/sacct_full.txt"
}

retrieve() {
  local workdir
  workdir="$(hpc_workdir)"
  local local_out="${LOCAL_ROOT}/data/derived/rq017_onsite_gate"
  if [[ -e "${local_out}" ]] && find "${local_out}" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite non-empty local output ${local_out}" >&2
    exit 2
  fi
  mkdir -p "${local_out}"
  rsync -a -e "${RSYNC_RSH}" "${REMOTE}:${workdir}/outputs/full/l1_v1/" "${local_out}/l1_v1/"
  rsync -a -e "${RSYNC_RSH}" "${REMOTE}:${workdir}/outputs/full/manifests/" "${SCRIPT_DIR}/full_manifests/"
  rsync -a -e "${RSYNC_RSH}" "${REMOTE}:${workdir}/inputs/prepare_inputs_summary.json" "${SCRIPT_DIR}/prepare_inputs_summary.json"
  rsync -a -e "${RSYNC_RSH}" "${REMOTE}:${workdir}/cluster_capacity_pre_full.json" "${SCRIPT_DIR}/cluster_capacity_pre_full.json"
  rsync -a -e "${RSYNC_RSH}" "${REMOTE}:${workdir}/env_parity.json" "${SCRIPT_DIR}/env_parity.json"
  cd "${LOCAL_ROOT}"
  python3 .codex-fleet/rq017-onsite-materializer/work/M1/rq017_onsite_materializer.py validate-outputs \
    --l1-root data/derived/rq017_onsite_gate/l1_v1 \
    --dryrun .codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet \
    --output .codex-fleet/rq017-onsite-materializer/work/M1/key_numbers.json \
    --expected-rows 67861 \
    --mode full
  python3 .codex-fleet/rq017-onsite-materializer/work/M1/rq017_onsite_materializer.py collect-receipt \
    --output .codex-fleet/rq017-onsite-materializer/work/M1/run_receipt.json \
    --manifest-dir .codex-fleet/rq017-onsite-materializer/work/M1/full_manifests \
    --sacct .codex-fleet/rq017-onsite-materializer/work/M1/sacct_full.txt \
    --hpc-workdir "${workdir}" \
    --job-id "$(cat "${SCRIPT_DIR}/latest_full_job_id.txt")" \
    --array-shape "$(cat "${SCRIPT_DIR}/latest_full_array_shape.txt")" \
    --cluster-capacity .codex-fleet/rq017-onsite-materializer/work/M1/cluster_capacity_pre_full.json \
    --prepare-summary .codex-fleet/rq017-onsite-materializer/work/M1/prepare_inputs_summary.json \
    --canary-validation .codex-fleet/rq017-onsite-materializer/work/M1/canary_validation.json \
    --full-validation .codex-fleet/rq017-onsite-materializer/work/M1/full_validation.remote.json
  python3 .codex-fleet/rq017-onsite-materializer/work/M1/rq017_onsite_materializer.py write-report \
    --output .codex-fleet/rq017-onsite-materializer/board/reports/RQ017_1_onsite_materializer.md \
    --measurement-contract .codex-fleet/rq017-onsite-materializer/work/M1/measurement_contract.json \
    --env-parity .codex-fleet/rq017-onsite-materializer/work/M1/env_parity.json \
    --key-numbers .codex-fleet/rq017-onsite-materializer/work/M1/key_numbers.json \
    --run-receipt .codex-fleet/rq017-onsite-materializer/work/M1/run_receipt.json \
    --local-output-root data/derived/rq017_onsite_gate/l1_v1
}

case "${1:-}" in
  stage) stage ;;
  submit-env) submit_env ;;
  submit-canary) submit_canary ;;
  submit-full) submit_full ;;
  retrieve) retrieve ;;
  *) echo "usage: $0 {stage|submit-env|submit-canary|submit-full|retrieve}" >&2; exit 2 ;;
esac
