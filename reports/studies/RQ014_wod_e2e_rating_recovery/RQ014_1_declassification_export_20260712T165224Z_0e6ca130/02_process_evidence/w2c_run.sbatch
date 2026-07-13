#!/bin/bash
#SBATCH --job-name=zxc-rq014-export-0e6ca13094ad
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_0_score_stripped_export_20260712T154921Z_1ee1e1d1/logs/slurm_%j.out
#SBATCH --error=/share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_0_score_stripped_export_20260712T154921Z_1ee1e1d1/logs/slurm_%j.err
#SBATCH --export=NIL
#SBATCH --chdir=/

set -euo pipefail
umask 077
test -z "${BASH_ENV-}${ENV-}${LD_PRELOAD-}${PYTHONHOME-}${PYTHONPATH-}"
test ! -L /share/home/u25310231/ZXC/sociality_estimation/manifests/runtime_maintenance.lock
exec 8>/share/home/u25310231/ZXC/sociality_estimation/manifests/runtime_maintenance.lock
/usr/bin/flock -s 8
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_0_score_stripped_export_20260712T154921Z_1ee1e1d1/manifests/run_spec.json | /usr/bin/awk '{print $1}')" = 0e6ca13094adc14286f8a1f39d57013322fa2e0e2fb9edc12494c470bb31f62b
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_0_score_stripped_export_20260712T154921Z_1ee1e1d1/manifests/code_snapshot.json | /usr/bin/awk '{print $1}')" = 755fec46b908f71d0b62be1dbe2eb0d1c096bbef38a7647e22481f59da20218a
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_0_score_stripped_export_20260712T154921Z_1ee1e1d1/code/configs/research_authorization.json | /usr/bin/awk '{print $1}')" = bf8adab8793e7007f905571ded3f46cb549fcb620649d8858b8ef0c494c0a40b
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_0_score_stripped_export_20260712T154921Z_1ee1e1d1/code/reports/plans/RQ014_execution_contract_v1p5.json | /usr/bin/awk '{print $1}')" = 02df9042e696bce547278ba8b55be1a61b92b4cda090ab64d7923528606a0b2b
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_0_score_stripped_export_20260712T154921Z_1ee1e1d1/code/reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/RQ014_formal_G1_v1p5_20260712.yaml | /usr/bin/awk '{print $1}')" = a24e69c240700840d908514580ba6f43cb6555d1de02d825121475bdeda88713
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/phase1_ipv_build/shards/shard_0/phase1_post_scene_bundle.pkl | /usr/bin/awk '{print $1}')" = 57b6fb62bf64bfe786245d18063eacf55c09a16bde175a415c6aad29718cd975
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/phase1_ipv_build/shards/shard_1/phase1_post_scene_bundle.pkl | /usr/bin/awk '{print $1}')" = c710c588f0e86a587999f57af487082514483b6611e9ae14141ff53b3b62b507
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/phase1_ipv_build/shards/shard_2/phase1_post_scene_bundle.pkl | /usr/bin/awk '{print $1}')" = 1cab3cd6d59b63fb8beecb3dcb6d0c7251841db16fdfdf64ef28953daa373b58
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/phase1_ipv_build/shards/shard_3/phase1_post_scene_bundle.pkl | /usr/bin/awk '{print $1}')" = 99dffe4e65fa64d4458802868e599f2b268dfe93d1b1a72022a7458d9044c937
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/phase1_ipv_build/shards/shard_4/phase1_post_scene_bundle.pkl | /usr/bin/awk '{print $1}')" = dcf63903ae6c18445d64de633e616a3690596fbd12edcce51e887f01d7b2688c
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/phase1_ipv_build/shards/shard_5/phase1_post_scene_bundle.pkl | /usr/bin/awk '{print $1}')" = 05e1ef602c0d9f426ded36c614d0a0eb8602491dea3b39bf6a465d38582f9734
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/phase1_ipv_build/shards/shard_6/phase1_post_scene_bundle.pkl | /usr/bin/awk '{print $1}')" = 3db1323ebbe13ded066f777128688cfd0521dc328fd438f0f7404703837d4e07
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/phase1_ipv_build/shards/shard_7/phase1_post_scene_bundle.pkl | /usr/bin/awk '{print $1}')" = 510f68f5922feff66866685604c0d99a521b168f9be4ceacbdd3128aaf19cb09
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/RQ010B_wod_e2e/manifests/rated479_segment_readiness.tsv | /usr/bin/awk '{print $1}')" = 425410ac48480513b566509321cd7f4cbac2a65c7db21e4174b39737f953f8a5
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/phase1_ipv_build/selected_counterpart_tracks.csv | /usr/bin/awk '{print $1}')" = 609979ae31cf9b926643b93f282c21eddb3a408ef968cc871c3d48538700c7c9
test ! -e /share/home/u25310231/ZXC/sociality_estimation/inputs/RQ014/wod_rated479_score_stripped/v1
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
test "$(/usr/bin/sha256sum /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_0_score_stripped_export_20260712T154921Z_1ee1e1d1/code/reports/plans/RQ014_plan_v1p5_checksums_20260712.sha256 | /usr/bin/awk '{print $1}')" = 1ee1e1d121b8d24cef7fdca93f05ddcccfcb3282b70727c606ce03c36984c933
(cd /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_0_score_stripped_export_20260712T154921Z_1ee1e1d1/code && /usr/bin/sha256sum -c reports/plans/RQ014_plan_v1p5_checksums_20260712.sha256)
exec /usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C SOCIALITY_PRODUCTION_RUN_ROOT=/share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_0_score_stripped_export_20260712T154921Z_1ee1e1d1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 /share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/bin/python3.9 -I -S -B -X utf8 -c 'import importlib.util
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
' '["/share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/lib/python39.zip","/share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/lib/python3.9","/share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/lib/python3.9/lib-dynload"]' /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_0_score_stripped_export_20260712T154921Z_1ee1e1d1/code/scripts/rq014/materialize_registry.py /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_0_score_stripped_export_20260712T154921Z_1ee1e1d1/code/scripts/rq014/preflight.py /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_0_score_stripped_export_20260712T154921Z_1ee1e1d1/code/scripts/rq014/export_score_stripped_bundle.py --scene-bundle /share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/phase1_ipv_build/shards/shard_0/phase1_post_scene_bundle.pkl --scene-bundle /share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/phase1_ipv_build/shards/shard_1/phase1_post_scene_bundle.pkl --scene-bundle /share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/phase1_ipv_build/shards/shard_2/phase1_post_scene_bundle.pkl --scene-bundle /share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/phase1_ipv_build/shards/shard_3/phase1_post_scene_bundle.pkl --scene-bundle /share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/phase1_ipv_build/shards/shard_4/phase1_post_scene_bundle.pkl --scene-bundle /share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/phase1_ipv_build/shards/shard_5/phase1_post_scene_bundle.pkl --scene-bundle /share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/phase1_ipv_build/shards/shard_6/phase1_post_scene_bundle.pkl --scene-bundle /share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/phase1_ipv_build/shards/shard_7/phase1_post_scene_bundle.pkl --source-expectation phase1_scene_bundle_00 562364 57b6fb62bf64bfe786245d18063eacf55c09a16bde175a415c6aad29718cd975 --source-expectation phase1_scene_bundle_01 568813 c710c588f0e86a587999f57af487082514483b6611e9ae14141ff53b3b62b507 --source-expectation phase1_scene_bundle_02 561347 1cab3cd6d59b63fb8beecb3dcb6d0c7251841db16fdfdf64ef28953daa373b58 --source-expectation phase1_scene_bundle_03 570948 99dffe4e65fa64d4458802868e599f2b268dfe93d1b1a72022a7458d9044c937 --source-expectation phase1_scene_bundle_04 561633 dcf63903ae6c18445d64de633e616a3690596fbd12edcce51e887f01d7b2688c --source-expectation phase1_scene_bundle_05 570955 05e1ef602c0d9f426ded36c614d0a0eb8602491dea3b39bf6a465d38582f9734 --source-expectation phase1_scene_bundle_06 561508 3db1323ebbe13ded066f777128688cfd0521dc328fd438f0f7404703837d4e07 --source-expectation phase1_scene_bundle_07 571301 510f68f5922feff66866685604c0d99a521b168f9be4ceacbdd3128aaf19cb09 --source-expectation rated479_structural_readiness 138666 425410ac48480513b566509321cd7f4cbac2a65c7db21e4174b39737f953f8a5 --source-expectation selected_counterpart_tracks 2449448 609979ae31cf9b926643b93f282c21eddb3a408ef968cc871c3d48538700c7c9 --readiness-tsv /share/home/u25310231/ZXC/RQ010B_wod_e2e/manifests/rated479_segment_readiness.tsv --counterpart-tracks /share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/phase1_ipv_build/selected_counterpart_tracks.csv --schema /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_0_score_stripped_export_20260712T154921Z_1ee1e1d1/code/reports/plans/RQ014_score_stripped_schema_v1.json --output-root /share/home/u25310231/ZXC/sociality_estimation/inputs/RQ014/wod_rated479_score_stripped/v1 --run-receipt-root /share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014/RQ014_0_score_stripped_export_20260712T154921Z_1ee1e1d1/outputs --exporter-git-commit 24be08278adf43371fda14e7ec23a95b986b2fb1 --exporter-environment-sha256 30de86f702101fbfc8065f6a0d7fd4378daf526d0e55c1197a6a0a147752877a --created-at-utc 2026-07-12T15:49:21Z
