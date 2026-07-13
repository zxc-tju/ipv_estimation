#!/bin/sh
set -eu
set -o pipefail

if [ "${HPC_SOCIALITY_ROOT+x}" = x ] || \
    [ "${BASH_ENV+x}" = x ] || \
    [ "${ENV+x}" = x ] || \
    [ "${LD_PRELOAD+x}" = x ] || \
    [ "${PYTHONHOME+x}" = x ] || \
    [ "${PYTHONPATH+x}" = x ]; then
    printf '%s\n' 'Unsafe launcher environment; use the documented env -i operator command.' >&2
    exit 64
fi
if /usr/bin/env | /usr/bin/awk -F= '$1 ~ /^SBATCH_/ { found=1 } END { exit(found ? 0 : 1) }'; then
    printf '%s\n' 'Inherited SBATCH_* variables are forbidden; use the documented env -i operator command.' >&2
    exit 64
fi

BASE=/share/home/u25310231/ZXC/sociality_estimation
REPO=$BASE/code/repo
WRAPPER=$REPO/scripts/hpc/submit_research_run.sh
PYTHON=$BASE/envs/ipv-exact-sigma01/bin/python3.9
LAUNCHER=$REPO/scripts/hpc/prepare_research_run.py
PREFLIGHT=$REPO/scripts/rq014/preflight.py
MATERIALIZER=$REPO/scripts/rq014/materialize_registry.py
ENVIRONMENT_MANIFEST=$BASE/manifests/RQ014/managed_python_environment_v3.json
STDLIB_ROOT=$BASE/envs/ipv-exact-sigma01/lib/python3.9
LIB_DYNLOAD=$STDLIB_ROOT/lib-dynload
PYTHON_ZIP=$BASE/envs/ipv-exact-sigma01/lib/python39.zip
STDLIB_MANIFEST=$BASE/manifests/RQ014/managed_python_stdlib_v1.sha256
NATIVE_MANIFEST=$BASE/manifests/RQ014/managed_python_native_libs_v1.tsv
RUNTIME_LOCK=$BASE/manifests/runtime_maintenance.lock
PYTHON_SHA256=616aea77938978c3578ed480eac4025ef7099f4c944ab8684722bc9455f99f32
LAUNCHER_SHA256=71fcdb74ab25983a583277a3afa630303a42007231c10fbada386f46c59cccb7
PREFLIGHT_SHA256=f91bbd2aef8ab6678109c1391d46672c41a41a03c5b1a9d67c35d01ce3de4102
MATERIALIZER_SHA256=d8cac79f19e07296fef03415cca76474b48ca7b122d733ab2c3ec7146bbdecc4
ENVIRONMENT_MANIFEST_SHA256=30de86f702101fbfc8065f6a0d7fd4378daf526d0e55c1197a6a0a147752877a
STDLIB_MANIFEST_SHA256=0a9944e1de0cf2b4168097b3afe82132333189127976c0a60c0891933853f0d5
NATIVE_MANIFEST_SHA256=46004531edef17588151114e6e024a85cac7aad063a887ec5d600903b1dfaa9d
readonly BASE REPO WRAPPER PYTHON LAUNCHER PREFLIGHT MATERIALIZER ENVIRONMENT_MANIFEST STDLIB_ROOT
readonly LIB_DYNLOAD PYTHON_ZIP STDLIB_MANIFEST NATIVE_MANIFEST PYTHON_SHA256
readonly LAUNCHER_SHA256 PREFLIGHT_SHA256 MATERIALIZER_SHA256 ENVIRONMENT_MANIFEST_SHA256
readonly STDLIB_MANIFEST_SHA256 NATIVE_MANIFEST_SHA256 RUNTIME_LOCK

if test -L "$RUNTIME_LOCK" || test ! -f "$RUNTIME_LOCK" || \
    test -L "$WRAPPER" || test ! -f "$WRAPPER" || \
    test ! -e "/proc/$$/fd/8" || \
    test "$(/usr/bin/readlink "/proc/$$/fd/8")" != "$RUNTIME_LOCK" || \
    test ! -f "/proc/$$/fd/8" || \
    test "$(/usr/bin/stat -Lc '%d:%i:%f' "/proc/$$/fd/8")" != \
        "$(/usr/bin/stat -c '%d:%i:%f' "$RUNTIME_LOCK")" || \
    test ! -e "/proc/$$/fd/9" || \
    test "$(/usr/bin/readlink "/proc/$$/fd/9")" != "$WRAPPER" || \
    test ! -f "/proc/$$/fd/9" || \
    test "$(/usr/bin/stat -Lc '%d:%i:%f' "/proc/$$/fd/9")" != \
        "$(/usr/bin/stat -c '%d:%i:%f' "$WRAPPER")"; then
    printf '%s\n' 'Missing or invalid RQ014 wrapper capability descriptors.' >&2
    exit 64
fi
/usr/bin/flock -s 8

digest_of() {
    /usr/bin/sha256sum "$1" | /usr/bin/awk '{print $1}'
}

test -f "$PYTHON"
test ! -L "$PYTHON"
test "$(/usr/bin/stat -c %s "$PYTHON")" = 16285544
test "$(digest_of "$PYTHON")" = "$PYTHON_SHA256"
test -f "$LAUNCHER"
test ! -L "$LAUNCHER"
test "$(/usr/bin/stat -c %s "$LAUNCHER")" = 116728
test "$(digest_of "$LAUNCHER")" = "$LAUNCHER_SHA256"
test -f "$PREFLIGHT"
test ! -L "$PREFLIGHT"
test "$(/usr/bin/stat -c %s "$PREFLIGHT")" = 69166
test "$(digest_of "$PREFLIGHT")" = "$PREFLIGHT_SHA256"
test -f "$MATERIALIZER"
test ! -L "$MATERIALIZER"
test "$(/usr/bin/stat -c %s "$MATERIALIZER")" = 11502
test "$(digest_of "$MATERIALIZER")" = "$MATERIALIZER_SHA256"
test -f "$ENVIRONMENT_MANIFEST"
test ! -L "$ENVIRONMENT_MANIFEST"
test "$(/usr/bin/stat -c %s "$ENVIRONMENT_MANIFEST")" = 2229
test "$(digest_of "$ENVIRONMENT_MANIFEST")" = "$ENVIRONMENT_MANIFEST_SHA256"
test -f "$STDLIB_MANIFEST"
test ! -L "$STDLIB_MANIFEST"
test "$(/usr/bin/stat -c %s "$STDLIB_MANIFEST")" = 3204661
test "$(digest_of "$STDLIB_MANIFEST")" = "$STDLIB_MANIFEST_SHA256"
test -f "$NATIVE_MANIFEST"
test ! -L "$NATIVE_MANIFEST"
test "$(/usr/bin/stat -c %s "$NATIVE_MANIFEST")" = 5858
test "$(digest_of "$NATIVE_MANIFEST")" = "$NATIVE_MANIFEST_SHA256"
test -d "$STDLIB_ROOT"
test -d "$LIB_DYNLOAD"
test ! -e "$PYTHON_ZIP"
test -z "$(/usr/bin/find "$STDLIB_ROOT" -type l -print -quit)"
test -z "$(/usr/bin/find "$STDLIB_ROOT" ! -type d ! -type f -print -quit)"

registered_count=$(/usr/bin/awk '/^# size_bytes=/ { count += 1 } END { print count + 0 }' "$STDLIB_MANIFEST")
registered_total=$(/usr/bin/awk -F= '/^# size_bytes=/ { total += $2 } END { printf "%.0f", total + 0 }' "$STDLIB_MANIFEST")
actual_count=$(/usr/bin/find "$STDLIB_ROOT" -type f -printf 'x\n' | /usr/bin/awk 'END { print NR + 0 }')
actual_total=$(/usr/bin/find "$STDLIB_ROOT" -type f -printf '%s\n' | /usr/bin/awk '{ total += $1 } END { printf "%.0f", total + 0 }')
test "$registered_count" = 14326
test "$registered_total" = 307357072
test "$actual_count" = "$registered_count"
test "$actual_total" = "$registered_total"

(cd / && /usr/bin/sha256sum --check --strict "$STDLIB_MANIFEST" | /usr/bin/awk '/FAILED/ { print; fflush() } NR % 500 == 0 { print "stdlib-check", NR; fflush() }')

native_rows=0
native_symlinks=0
native_total=0
native_previous=
native_tab=$(/usr/bin/awk 'BEGIN { printf "\t" }')
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
        if test -n "$native_previous"; then
            test "$native_previous" \< "$soname"
        fi
        native_previous=$soname
        case "$loader_path" in
            "$BASE/envs/ipv-exact-sigma01"/*|/lib64/*) ;;
            *) exit 65 ;;
        esac
        case "$resolved_path" in
            "$BASE/envs/ipv-exact-sigma01"/*|/lib64/*) ;;
            *) exit 65 ;;
        esac
        if test "$link_target" = -; then
            test ! -L "$loader_path"
            test -f "$loader_path"
        else
            test -L "$loader_path"
            test "$(/usr/bin/readlink "$loader_path")" = "$link_target"
            case "$link_target" in
                /*) lexical_target=$link_target ;;
                *) lexical_target=${loader_path%/*}/$link_target ;;
            esac
            test ! -L "$lexical_target"
            test -f "$lexical_target"
            test "$(/usr/bin/readlink -f "$lexical_target")" = "$(/usr/bin/readlink -f "$resolved_path")"
            native_symlinks=$((native_symlinks + 1))
        fi
        test "$(/usr/bin/readlink -f "$loader_path")" = "$resolved_path"
        test ! -L "$resolved_path"
        test -f "$resolved_path"
        test "$(/usr/bin/stat -c %s "$resolved_path")" = "$size_bytes"
        test "$(digest_of "$resolved_path")" = "$digest"
        native_rows=$((native_rows + 1))
        native_total=$((native_total + size_bytes))
    done
} < "$NATIVE_MANIFEST"
test "$native_rows" = 20
test "$native_symlinks" = 16
test "$native_total" = 14656296

expected_sys_path=$(/usr/bin/printf '%s\n%s\n%s' "$PYTHON_ZIP" "$STDLIB_ROOT" "$LIB_DYNLOAD")
actual_sys_path=$(/usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
    "$PYTHON" -I -S -B -X utf8 -c 'import sys; print("\n".join(sys.path))')
test "$actual_sys_path" = "$expected_sys_path"

exec /usr/bin/env -i \
    PATH=/usr/bin:/bin \
    LANG=C \
    LC_ALL=C \
    "$PYTHON" -I -S -B -X utf8 "$LAUNCHER" --rq014-only "$@"
