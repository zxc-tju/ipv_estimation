#!/bin/bash
#SBATCH --job-name=zxc-rq014-pre-72dd4362f954
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_1_wod_rating_recovery_20260713T161542Z_41ac5280/logs/slurm_%j.out
#SBATCH --error=/share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_1_wod_rating_recovery_20260713T161542Z_41ac5280/logs/slurm_%j.err
#SBATCH --export=NIL
#SBATCH --chdir=/

set -euo pipefail
umask 077
test -z "${BASH_ENV-}${ENV-}${LD_PRELOAD-}${PYTHONHOME-}${PYTHONPATH-}"
test ! -L /share/home/u25310231/ZXC/sociality_estimation/manifests/runtime_maintenance.lock
exec 8>/share/home/u25310231/ZXC/sociality_estimation/manifests/runtime_maintenance.lock
/usr/bin/flock -s 8
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_1_wod_rating_recovery_20260713T161542Z_41ac5280/manifests/run_spec.json | /usr/bin/awk '{print $1}')" = 72dd4362f9546dcf24b792bd3c0f1c6413f1232f069e57891bcbb293fd316c14
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_1_wod_rating_recovery_20260713T161542Z_41ac5280/manifests/code_snapshot.json | /usr/bin/awk '{print $1}')" = 0926ec48fdf38052eceef0741b63b470321e2c72032f5e2e5debbe1ff7b20f14
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_1_wod_rating_recovery_20260713T161542Z_41ac5280/code/configs/research_authorization.json | /usr/bin/awk '{print $1}')" = 4991d1902a034959ab325d924a94b9b25616987169a72699a802045616254771
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_1_wod_rating_recovery_20260713T161542Z_41ac5280/code/reports/plans/RQ014_execution_contract_v1p5.json | /usr/bin/awk '{print $1}')" = c5f5ec35dc30af2be714892b095ecd747444249d33ba70021c330f01113a6f58
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_1_wod_rating_recovery_20260713T161542Z_41ac5280/code/reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/RQ014_formal_G1_v1p6_preflight_20260713.yaml | /usr/bin/awk '{print $1}')" = 755e6a34a8114918890bd57a8cbf41455e3a28d7a7cbeb552868546a97424990
test ! -L /share/home/u25310231/ZXC/sociality_estimation/checkpoints/rq009_m3/m3_scorer.joblib
test -f /share/home/u25310231/ZXC/sociality_estimation/checkpoints/rq009_m3/m3_scorer.joblib
test "$(/usr/bin/readlink -f /share/home/u25310231/ZXC/sociality_estimation/checkpoints/rq009_m3/m3_scorer.joblib)" = /share/home/u25310231/ZXC/sociality_estimation/checkpoints/rq009_m3/m3_scorer.joblib
test "$(/usr/bin/stat -c %s /share/home/u25310231/ZXC/sociality_estimation/checkpoints/rq009_m3/m3_scorer.joblib)" = 88306301
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/sociality_estimation/checkpoints/rq009_m3/m3_scorer.joblib | /usr/bin/awk '{print $1}')" = b04999aba29a82fb71a97ac22c728479a7734e24a0b32189d08f95184d74f253
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/sociality_estimation/manifests/RQ014/RQ014_1_wod_rating_recovery_20260713T161542Z_41ac5280/input_manifest.g2.json | /usr/bin/awk '{print $1}')" = 1be6248e7651ad9e90701db232e6b47caa62b6579b8294bb2bb74683063dcab1
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/sociality_estimation/inputs/RQ014/wod_rated479_score_stripped/v1/sanitization_receipt.json | /usr/bin/awk '{print $1}')" = 4dfc81056fe97db66d0ba0df04955a9c7c3a5464237010f298f3fa64363911a3
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/sociality_estimation/manifests/RQ014/RQ014_1_wod_rating_recovery_20260713T161542Z_41ac5280/materialization_ledger.json | /usr/bin/awk '{print $1}')" = 2413a457cf790090f08551e4342693ed73ab5976bdeb0bc9e0b34890a54b52a4
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_0_score_stripped_export_20260712T154921Z_1ee1e1d1/outputs/rq014_g2_declassification_export_receipt.json | /usr/bin/awk '{print $1}')" = a3839a7a253c869de7a67aee410ca92ea3e238b73ae5346fd2efc86980a67c63
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_0_score_stripped_export_20260712T154921Z_1ee1e1d1/outputs/DONE.json | /usr/bin/awk '{print $1}')" = f3cc34c34082d1c7d4d5ceedce7505a3a9ed35d1f17d6765af5e50443a92bdda
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/sociality_estimation/manifests/RQ014/managed_python_environment_v3.json | /usr/bin/awk '{print $1}')" = 30de86f702101fbfc8065f6a0d7fd4378daf526d0e55c1197a6a0a147752877a
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/bin/python3.9 | /usr/bin/awk '{print $1}')" = 616aea77938978c3578ed480eac4025ef7099f4c944ab8684722bc9455f99f32
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/sociality_estimation/manifests/RQ014/managed_python_stdlib_v1.sha256 | /usr/bin/awk '{print $1}')" = 0a9944e1de0cf2b4168097b3afe82132333189127976c0a60c0891933853f0d5
test -d /share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/lib/python3.9
test -d /share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/lib/python3.9/lib-dynload
test ! -e /share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/lib/python39.zip
test -z "$(/usr/bin/find /share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/lib/python3.9 -type l -print -quit)"
test -z "$(/usr/bin/find /share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/lib/python3.9 ! -type d ! -type f -print -quit)"
test "$(/usr/bin/find /share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/lib/python3.9 -type f -printf 'x\n' | /usr/bin/awk 'END { print NR + 0 }')" = 14326
test "$(/usr/bin/find /share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/lib/python3.9 -type f -printf '%s\n' | /usr/bin/awk '{ total += $1 } END { printf "%.0f", total + 0 }')" = 307357072
(cd / && /usr/bin/sha256sum --check --strict /share/home/u25310231/ZXC/sociality_estimation/manifests/RQ014/managed_python_stdlib_v1.sha256 | /usr/bin/awk '/FAILED/ { print; fflush() } NR % 500 == 0 { print "stdlib-check", NR; fflush() }')
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/sociality_estimation/manifests/RQ014/managed_python_native_libs_v1.tsv | /usr/bin/awk '{print $1}')" = 46004531edef17588151114e6e024a85cac7aad063a887ec5d600903b1dfaa9d
native_rows=0
native_symlinks=0
native_total=0
native_tab=$(/usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/awk 'BEGIN { printf "\t" }')
{
  IFS= read -r native_header
  test "$native_header" = '# rq014-managed-python-native-libs-v1'
  IFS= read -r native_header
  test "$native_header" = '# columns=soname<TAB>loader_path<TAB>link_target_or_dash<TAB>resolved_path<TAB>size_bytes<TAB>sha256'
  IFS= read -r native_header
  test "$native_header" = '# discovery=ldd_python3.9_plus_every_regular_lib-dynload_so'
  IFS= read -r native_header
  test "$native_header" = '# managed_environment_root=/share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01'
  IFS= read -r native_header
  test "$native_header" = '# row_count=20'
  while IFS="$native_tab" read -r soname loader_path link_target resolved_path size_bytes digest extra; do
  test -n "$soname"
  test -z "${extra-}"
  if test "$link_target" = -; then
    test ! -L "$loader_path"
    test -f "$loader_path"
  else
    test -L "$loader_path"
    test "$(/usr/bin/readlink "$loader_path")" = "$link_target"
    case "$link_target" in /*) lexical_target=$link_target ;; *) lexical_target=${loader_path%/*}/$link_target ;; esac
    test ! -L "$lexical_target"
    test -f "$lexical_target"
    test "$(/usr/bin/readlink -f "$lexical_target")" = "$(/usr/bin/readlink -f "$resolved_path")"
    native_symlinks=$((native_symlinks + 1))
  fi
  test "$(/usr/bin/readlink -f "$loader_path")" = "$resolved_path"
  test ! -L "$resolved_path"
  test -f "$resolved_path"
  test "$(/usr/bin/stat -c %s "$resolved_path")" = "$size_bytes"
  test "$(/usr/bin/sha256sum "$resolved_path" | /usr/bin/awk '{print $1}')" = "$digest"
  native_rows=$((native_rows + 1))
  native_total=$((native_total + size_bytes))
  done
} < /share/home/u25310231/ZXC/sociality_estimation/manifests/RQ014/managed_python_native_libs_v1.tsv
test "$native_rows" = 20
test "$native_symlinks" = 16
test "$native_total" = 14656296
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_1_wod_rating_recovery_20260713T161542Z_41ac5280/code/reports/plans/RQ014_plan_v1p6_checksums_20260713.sha256 | /usr/bin/awk '{print $1}')" = 41ac52808cba5eb729829bc031053c49fb49583691ff24f7e2662c38b5ee2f19
(cd /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_1_wod_rating_recovery_20260713T161542Z_41ac5280/code && /usr/bin/sha256sum -c reports/plans/RQ014_plan_v1p6_checksums_20260713.sha256)
exec /usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C SOCIALITY_PRODUCTION_RUN_ROOT=/share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_1_wod_rating_recovery_20260713T161542Z_41ac5280 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 /share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/bin/python3.9 -I -S -B -X utf8 -c 'import importlib.util
import json
import sys
import types

expected_sys_path_json, materializer_path, preflight_path, entrypoint, *arguments = sys.argv[1:]
if sys.path != json.loads(expected_sys_path_json):
    raise RuntimeError("Managed isolated sys.path differs from the reviewed runtime closure")
scripts_package = types.ModuleType("scripts")
scripts_package.__path__ = ()
scripts_package.__package__ = "scripts"
rq014_package = types.ModuleType("scripts.rq014")
rq014_package.__path__ = ()
rq014_package.__package__ = "scripts.rq014"
scripts_package.rq014 = rq014_package
sys.modules["scripts"] = scripts_package
sys.modules["scripts.rq014"] = rq014_package

materializer_spec = importlib.util.spec_from_file_location(
    "scripts.rq014.materialize_registry", materializer_path
)
if materializer_spec is None or materializer_spec.loader is None:
    raise RuntimeError("Cannot construct exact-path registry materializer loader")
materializer_module = importlib.util.module_from_spec(materializer_spec)
rq014_package.materialize_registry = materializer_module
sys.modules["scripts.rq014.materialize_registry"] = materializer_module
materializer_spec.loader.exec_module(materializer_module)

preflight_spec = importlib.util.spec_from_file_location(
    "scripts.rq014.preflight", preflight_path
)
if preflight_spec is None or preflight_spec.loader is None:
    raise RuntimeError("Cannot construct exact-path preflight loader")
preflight_module = importlib.util.module_from_spec(preflight_spec)
rq014_package.preflight = preflight_module
sys.modules["scripts.rq014.preflight"] = preflight_module
preflight_spec.loader.exec_module(preflight_module)

sys.argv = [entrypoint, *arguments]
entrypoint_spec = importlib.util.spec_from_file_location("__main__", entrypoint)
if entrypoint_spec is None or entrypoint_spec.loader is None:
    raise RuntimeError("Cannot construct exact-path entrypoint loader")
entrypoint_module = importlib.util.module_from_spec(entrypoint_spec)
sys.modules["__main__"] = entrypoint_module
entrypoint_spec.loader.exec_module(entrypoint_module)
' '["/share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/lib/python39.zip","/share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/lib/python3.9","/share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/lib/python3.9/lib-dynload"]' /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_1_wod_rating_recovery_20260713T161542Z_41ac5280/code/scripts/rq014/materialize_registry.py /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_1_wod_rating_recovery_20260713T161542Z_41ac5280/code/scripts/rq014/preflight.py /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_1_wod_rating_recovery_20260713T161542Z_41ac5280/code/scripts/rq014/run_managed_g2.py contract-preflight --base /share/home/u25310231/ZXC/sociality_estimation --repo-root /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_1_wod_rating_recovery_20260713T161542Z_41ac5280/code --execution-contract /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_1_wod_rating_recovery_20260713T161542Z_41ac5280/code/reports/plans/RQ014_execution_contract_v1p5.json --m3-artifact /share/home/u25310231/ZXC/sociality_estimation/checkpoints/rq009_m3/m3_scorer.joblib --m3-artifact-size-bytes 88306301 --m3-artifact-sha256 b04999aba29a82fb71a97ac22c728479a7734e24a0b32189d08f95184d74f253 --input-manifest /share/home/u25310231/ZXC/sociality_estimation/manifests/RQ014/RQ014_1_wod_rating_recovery_20260713T161542Z_41ac5280/input_manifest.g2.json --sanitization-receipt /share/home/u25310231/ZXC/sociality_estimation/inputs/RQ014/wod_rated479_score_stripped/v1/sanitization_receipt.json --materialization-ledger /share/home/u25310231/ZXC/sociality_estimation/manifests/RQ014/RQ014_1_wod_rating_recovery_20260713T161542Z_41ac5280/materialization_ledger.json --declassification-export-receipt /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_0_score_stripped_export_20260712T154921Z_1ee1e1d1/outputs/rq014_g2_declassification_export_receipt.json --declassification-export-done /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_0_score_stripped_export_20260712T154921Z_1ee1e1d1/outputs/DONE.json --expected-exporter-git-commit 24be08278adf43371fda14e7ec23a95b986b2fb1 --expected-exporter-environment-sha256 30de86f702101fbfc8065f6a0d7fd4378daf526d0e55c1197a6a0a147752877a --output-root /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_1_wod_rating_recovery_20260713T161542Z_41ac5280/outputs
