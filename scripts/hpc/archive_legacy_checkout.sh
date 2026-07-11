#!/usr/bin/env bash
set -euo pipefail

BASE="/share/home/u25310231/ZXC/sociality_estimation"
LEGACY="/share/home/u25310231/ZXC/ipv_estimation"
INVENTORY_ID="pre_migration_20260711_v1"
SNAPSHOT_ID="interhub_legacy_20260711_v1"
EXPECTED_LEGACY_HEAD="5edd28104bf5989e2dc258c9405ce897d7523cc4"

OUT="$BASE/manifests/legacy_migration/$INVENTORY_ID"
ARCHIVE_PARENT="$BASE/archives/legacy-code"
ARCHIVE="$ARCHIVE_PARENT/ipv_estimation-$EXPECTED_LEGACY_HEAD"
ARCHIVE_INCOMING="$ARCHIVE.incoming"
ROOT_QUARANTINE="$BASE/quarantine/legacy-code-root-ipv_estimation-$SNAPSHOT_ID"
RAW_SNAPSHOT="$BASE/data/interhub/snapshots/$SNAPSHOT_ID"
RESULTS_SNAPSHOT="$BASE/archives/historical-results/$SNAPSHOT_ID"
RAW_LINK="$LEGACY/interhub_traj_lane/0_raw_data"
RESULTS_LINK="$LEGACY/interhub_traj_lane/1_ipv_estimation_results"
POST_MARKER="$OUT/POST_SWITCH_PREFLIGHT_COMPLETE.json"
RETIREMENT_MARKER="$OUT/CODE_RETIREMENT_COMPLETE"

[[ "$INVENTORY_ID" =~ ^[A-Za-z0-9._-]+$ ]]
[[ "$SNAPSHOT_ID" =~ ^[A-Za-z0-9._-]+$ ]]
test ! -L "$BASE"
test ! -L "$LEGACY"
test "$(readlink -f "$(dirname "$BASE")")" = "/share/home/u25310231/ZXC"
test "$(readlink -f "$(dirname "$LEGACY")")" = "/share/home/u25310231/ZXC"
test -f "$OUT/SWITCH_COMPLETE"
test -f "$POST_MARKER"

command -v flock >/dev/null
exec 9>"$OUT/legacy_checkout_retirement.lock"
flock -n 9 || {
  printf 'Another legacy checkout retirement is running: %s\n' \
    "$OUT/legacy_checkout_retirement.lock" >&2
  exit 4
}
exec 8>"$BASE/manifests/runtime_maintenance.lock"
flock -x -n 8 || {
  printf 'A research run or maintenance operation is active.\n' >&2
  exit 5
}
exec 7>"$OUT/migration.lock"
flock -n 7 || {
  printf 'The payload migration is still active.\n' >&2
  exit 6
}

grep -Fx "snapshot_id=$SNAPSHOT_ID" "$OUT/SWITCH_COMPLETE" >/dev/null
grep -Fx "raw_snapshot=$RAW_SNAPSHOT" "$OUT/SWITCH_COMPLETE" >/dev/null
grep -Fx "results_snapshot=$RESULTS_SNAPSHOT" "$OUT/SWITCH_COMPLETE" >/dev/null
python3 - "$POST_MARKER" "$SNAPSHOT_ID" "$RAW_SNAPSHOT" "$RESULTS_SNAPSHOT" \
  "$OUT/SWITCH_COMPLETE" \
  "$OUT/snapshot_reverification_$SNAPSHOT_ID/attestation.json" \
  "$OUT/raw_sha256.txt" "$OUT/results_sha256.txt" \
  "$OUT/snapshot_reverification_$SNAPSHOT_ID/raw_snapshot_verify.txt" \
  "$OUT/snapshot_reverification_$SNAPSHOT_ID/results_snapshot_verify.txt" \
  "$BASE/code/repo" <<'PY'
import hashlib
import json
import re
import subprocess
import sys

marker = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "schema_version": 1,
    "snapshot_id": sys.argv[2],
    "job_state": "COMPLETED",
    "job_exit_code": "0:0",
    "raw_snapshot": sys.argv[3],
    "results_snapshot": sys.argv[4],
}
for key, value in expected.items():
    if marker.get(key) != value:
        raise SystemExit(f"invalid post-switch marker field: {key}")
if marker.get("verification_counts") != {"raw": 51, "results": 173034}:
    raise SystemExit("invalid post-switch verification counts")
def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
attestation = json.load(open(sys.argv[6], encoding="utf-8"))
if marker.get("data_manifest_path") != sys.argv[7]:
    raise SystemExit("invalid post-switch data manifest path")
if marker.get("data_manifest_sha256") != sha256(sys.argv[7]):
    raise SystemExit("invalid post-switch data manifest hash")
if attestation.get("switch_marker_sha256") != sha256(sys.argv[5]):
    raise SystemExit("migration attestation does not bind switch marker")
if attestation.get("source_manifest_sha256") != {
    "raw": sha256(sys.argv[7]), "results": sha256(sys.argv[8])
}:
    raise SystemExit("migration attestation does not bind source manifests")
if attestation.get("snapshot_verification_sha256") != {
    "raw": sha256(sys.argv[9]), "results": sha256(sys.argv[10])
}:
    raise SystemExit("migration attestation does not bind snapshot verification logs")
if marker.get("migration_attestation_path") != sys.argv[6]:
    raise SystemExit("post-switch marker has wrong migration attestation path")
if marker.get("migration_attestation_sha256") != sha256(sys.argv[6]):
    raise SystemExit("post-switch marker has wrong migration attestation hash")
if marker.get("snapshot_verification_sha256") != attestation.get("snapshot_verification_sha256"):
    raise SystemExit("post-switch marker does not preserve attested verification hashes")
if not re.fullmatch(r"[0-9]+", str(marker.get("job_id", ""))):
    raise SystemExit("invalid post-switch job id")
if not re.fullmatch(r"[A-Za-z0-9._-]+", str(marker.get("run_id", ""))):
    raise SystemExit("invalid post-switch run id")
repo = sys.argv[11]
head = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
origin_main = subprocess.check_output(
    ["git", "-C", repo, "rev-parse", "refs/remotes/origin/main"], text=True
).strip()
if marker.get("git_commit") != head or head != origin_main:
    raise SystemExit("post-switch commit is not the current published canonical checkout")
PY

(cd "$OUT" && sha256sum -c inventory_checksums.sha256 >/dev/null)
test -d "$RAW_SNAPSHOT"
test ! -L "$RAW_SNAPSHOT"
test -d "$RESULTS_SNAPSHOT"
test ! -L "$RESULTS_SNAPSHOT"
test -z "$(find "$RAW_SNAPSHOT" "$RESULTS_SNAPSHOT" ! -type l -perm /222 -print -quit)"

while IFS='|' read -r job_id job_name work_dir state; do
  [[ -n "$job_id" ]] || continue
  [[ "$job_id" = "${SLURM_JOB_ID:-}" ]] && continue
  if [[ "$work_dir" == "$LEGACY"* ]]; then
    printf 'Active Slurm job still uses legacy checkout: %s %s %s %s\n' \
      "$job_id" "$job_name" "$state" "$work_dir" >&2
    exit 7
  fi
  if scontrol show job -o "$job_id" 2>/dev/null | grep -Fq "$LEGACY"; then
    printf 'Active Slurm job command references legacy checkout: %s %s\n' \
      "$job_id" "$job_name" >&2
    exit 7
  fi
done < <(squeue -h -u "$(id -un)" -o '%A|%j|%Z|%T')

safe_remove_stub() {
  test "$LEGACY" = "/share/home/u25310231/ZXC/ipv_estimation"
  test -d "$LEGACY"
  test ! -L "$LEGACY"
  test -z "$(find "$LEGACY" -mindepth 1 -maxdepth 1 \
    ! -name interhub_traj_lane ! -name TOMBSTONE.md -print -quit)"
  if [[ -d "$LEGACY/interhub_traj_lane" ]]; then
    test -z "$(find "$LEGACY/interhub_traj_lane" -mindepth 1 -maxdepth 1 \
      ! -name 0_raw_data ! -name 1_ipv_estimation_results -print -quit)"
    if [[ -e "$LEGACY/interhub_traj_lane/0_raw_data" ]]; then
      test -L "$LEGACY/interhub_traj_lane/0_raw_data"
      test "$(readlink -f "$LEGACY/interhub_traj_lane/0_raw_data")" = "$RAW_SNAPSHOT"
    fi
    if [[ -e "$LEGACY/interhub_traj_lane/1_ipv_estimation_results" ]]; then
      test -L "$LEGACY/interhub_traj_lane/1_ipv_estimation_results"
      test "$(readlink -f "$LEGACY/interhub_traj_lane/1_ipv_estimation_results")" = "$RESULTS_SNAPSHOT"
    fi
  fi
  rm -rf -- "$LEGACY"
}

if [[ -e "$RETIREMENT_MARKER" ]]; then
  test -d "$ROOT_QUARANTINE"
  test -f "$LEGACY/TOMBSTONE.md"
  test "$(readlink -f "$RAW_LINK")" = "$RAW_SNAPSHOT"
  test "$(readlink -f "$RESULTS_LINK")" = "$RESULTS_SNAPSHOT"
  printf 'Legacy checkout retirement is already complete.\n'
  exit 0
fi

# Recover an interrupted earlier attempt before inspecting or archiving source.
if [[ -d "$ROOT_QUARANTINE" ]]; then
  if [[ -e "$LEGACY" ]]; then
    safe_remove_stub
  fi
  mv "$ROOT_QUARANTINE" "$LEGACY"
fi

test -d "$LEGACY/.git"
test "$(git -C "$LEGACY" rev-parse HEAD)" = "$EXPECTED_LEGACY_HEAD"
test -L "$RAW_LINK"
test -L "$RESULTS_LINK"
test "$(readlink -f "$RAW_LINK")" = "$RAW_SNAPSHOT"
test "$(readlink -f "$RESULTS_LINK")" = "$RESULTS_SNAPSHOT"
test -d "$LEGACY/interhub_traj_lane/0_raw_data.quarantine_$SNAPSHOT_ID"
test -d "$LEGACY/interhub_traj_lane/1_ipv_estimation_results.quarantine_$SNAPSHOT_ID"
test ! -e "$ARCHIVE_INCOMING"

SOURCE_AUDIT="$OUT/retirement_source_audit.${SLURM_JOB_ID:-$$}"
test ! -e "$SOURCE_AUDIT"
mkdir "$SOURCE_AUDIT"

git -C "$LEGACY" status --porcelain=v1 --untracked-files=all \
  > "$SOURCE_AUDIT/current_status.txt"
git -C "$LEGACY" diff --binary > "$SOURCE_AUDIT/current_tracked.patch"
cmp "$SOURCE_AUDIT/current_status.txt" "$OUT/legacy_status.txt"
cmp "$SOURCE_AUDIT/current_tracked.patch" "$OUT/legacy_tracked.patch"

printf 'sha256\tsize_bytes\tpath\n' > "$SOURCE_AUDIT/current_tracked_files.tsv"
while IFS= read -r -d '' rel; do
  path="$LEGACY/$rel"
  if [[ -f "$path" ]]; then
    printf '%s\t%s\t%s\n' \
      "$(sha256sum "$path" | awk '{print $1}')" \
      "$(stat -c '%s' "$path")" "$rel" \
      >> "$SOURCE_AUDIT/current_tracked_files.tsv"
  fi
done < <(git -C "$LEGACY" ls-files -z)
cmp "$SOURCE_AUDIT/current_tracked_files.tsv" "$OUT/legacy_tracked_files.tsv"

printf 'class\tsize_bytes\tmtime_epoch\tpath\n' \
  > "$SOURCE_AUDIT/current_untracked_inventory.tsv"
while IFS= read -r -d '' rel; do
  path="$LEGACY/$rel"
  [[ -e "$path" ]] || continue
  class=other
  case "$rel" in
    *.out|*.err|*.log) class=log ;;
    __pycache__/*|*/__pycache__/*|*.pyc|.pytest_cache/*|*/.pytest_cache/*) class=cache ;;
    *.py|*.sh|*.sbatch|*.json|*.yaml|*.yml|*.md|*.txt) class=code_or_metadata ;;
  esac
  printf '%s\t%s\t%s\t%s\n' "$class" \
    "$(stat -c '%s' "$path")" "$(stat -c '%Y' "$path")" "$rel" \
    >> "$SOURCE_AUDIT/current_untracked_inventory.tsv"
done < <(git -C "$LEGACY" ls-files --others --exclude-standard -z)
cmp "$SOURCE_AUDIT/current_untracked_inventory.tsv" \
  "$OUT/legacy_untracked_inventory.tsv"

git -C "$LEGACY" ls-files -z > "$SOURCE_AUDIT/include_paths.nul"
while IFS=$'\t' read -r class _size _mtime rel; do
  [[ "$class" = code_or_metadata ]] || continue
  [[ "$rel" = rq009_code_parity_tmp/* ]] && continue
  printf '%s\0' "$rel" >> "$SOURCE_AUDIT/include_paths.nul"
done < <(tail -n +2 "$OUT/legacy_untracked_inventory.tsv")
sort -zu "$SOURCE_AUDIT/include_paths.nul" > "$SOURCE_AUDIT/include_paths_sorted.nul"
: > "$SOURCE_AUDIT/expected_files.sha256"
: > "$SOURCE_AUDIT/expected_symlinks.tsv"
while IFS= read -r -d '' rel; do
  if [[ -L "$LEGACY/$rel" ]]; then
    printf '%s\t%s\n' "$rel" "$(readlink "$LEGACY/$rel")" \
      >> "$SOURCE_AUDIT/expected_symlinks.tsv"
  elif [[ -f "$LEGACY/$rel" ]]; then
    (cd "$LEGACY" && sha256sum "$rel") >> "$SOURCE_AUDIT/expected_files.sha256"
  else
    printf 'Expected archive input is missing: %s\n' "$rel" >&2
    false
  fi
done < "$SOURCE_AUDIT/include_paths_sorted.nul"

verify_archive() {
  test -f "$ARCHIVE/archive_sha256.txt"
  (cd "$ARCHIVE" && sha256sum -c archive_sha256.txt >/dev/null)
  cmp "$ARCHIVE/inventory_status.txt" "$OUT/legacy_status.txt"
  cmp "$ARCHIVE/inventory_tracked.patch" "$OUT/legacy_tracked.patch"
  cmp "$ARCHIVE/inventory_tracked_files.tsv" "$OUT/legacy_tracked_files.tsv"
  cmp "$ARCHIVE/inventory_untracked_inventory.tsv" \
    "$OUT/legacy_untracked_inventory.tsv"
  cmp "$ARCHIVE/inventory_checksums.sha256" "$OUT/inventory_checksums.sha256"
  cmp "$ARCHIVE/current_status.txt" "$SOURCE_AUDIT/current_status.txt"
  cmp "$ARCHIVE/current_tracked.patch" "$SOURCE_AUDIT/current_tracked.patch"
  cmp "$ARCHIVE/current_tracked_files.tsv" "$SOURCE_AUDIT/current_tracked_files.tsv"
  cmp "$ARCHIVE/current_untracked_inventory.tsv" \
    "$SOURCE_AUDIT/current_untracked_inventory.tsv"
  cmp "$ARCHIVE/include_paths_sorted.nul" "$SOURCE_AUDIT/include_paths_sorted.nul"
  (cd "$ARCHIVE/files" && \
    sha256sum -c "$SOURCE_AUDIT/expected_files.sha256" >/dev/null)
  while IFS=$'\t' read -r rel target; do
    test -L "$ARCHIVE/files/$rel"
    test "$(readlink "$ARCHIVE/files/$rel")" = "$target"
  done < "$SOURCE_AUDIT/expected_symlinks.tsv"
  local actual_links
  actual_links="$(mktemp)"
  (cd "$ARCHIVE" && find . -type l -printf '%p\t%l\n' | sort) > "$actual_links"
  cmp "$actual_links" "$ARCHIVE/archive_symlinks.tsv"
  rm -f "$actual_links"
  git bundle verify "$ARCHIVE/repository.bundle" >/dev/null
  git bundle list-heads "$ARCHIVE/repository.bundle" \
    | grep -F "$EXPECTED_LEGACY_HEAD" >/dev/null
}

if [[ -d "$ARCHIVE" ]]; then
  verify_archive
else
  mkdir -p "$ARCHIVE_INCOMING/files" "$ARCHIVE_PARENT"
  cleanup_incoming() {
    trap - ERR INT TERM
    chmod -R u+w "$ARCHIVE_INCOMING" 2>/dev/null || true
    rm -rf -- "$ARCHIVE_INCOMING"
    rm -rf -- "$SOURCE_AUDIT"
    exit 8
  }
  trap cleanup_incoming ERR INT TERM

  git -C "$LEGACY" status --porcelain=v1 --untracked-files=all \
    > "$ARCHIVE_INCOMING/current_status.txt"
  git -C "$LEGACY" diff --binary > "$ARCHIVE_INCOMING/current_tracked.patch"
  cmp "$ARCHIVE_INCOMING/current_status.txt" "$OUT/legacy_status.txt"
  cmp "$ARCHIVE_INCOMING/current_tracked.patch" "$OUT/legacy_tracked.patch"

  printf 'sha256\tsize_bytes\tpath\n' > "$ARCHIVE_INCOMING/current_tracked_files.tsv"
  while IFS= read -r -d '' rel; do
    path="$LEGACY/$rel"
    if [[ -f "$path" ]]; then
      printf '%s\t%s\t%s\n' \
        "$(sha256sum "$path" | awk '{print $1}')" \
        "$(stat -c '%s' "$path")" "$rel" \
        >> "$ARCHIVE_INCOMING/current_tracked_files.tsv"
    fi
  done < <(git -C "$LEGACY" ls-files -z)
  cmp "$ARCHIVE_INCOMING/current_tracked_files.tsv" "$OUT/legacy_tracked_files.tsv"

  printf 'class\tsize_bytes\tmtime_epoch\tpath\n' \
    > "$ARCHIVE_INCOMING/current_untracked_inventory.tsv"
  while IFS= read -r -d '' rel; do
    path="$LEGACY/$rel"
    [[ -e "$path" ]] || continue
    class=other
    case "$rel" in
      *.out|*.err|*.log) class=log ;;
      __pycache__/*|*/__pycache__/*|*.pyc|.pytest_cache/*|*/.pytest_cache/*) class=cache ;;
      *.py|*.sh|*.sbatch|*.json|*.yaml|*.yml|*.md|*.txt) class=code_or_metadata ;;
    esac
    printf '%s\t%s\t%s\t%s\n' "$class" \
      "$(stat -c '%s' "$path")" "$(stat -c '%Y' "$path")" "$rel" \
      >> "$ARCHIVE_INCOMING/current_untracked_inventory.tsv"
  done < <(git -C "$LEGACY" ls-files --others --exclude-standard -z)
  cmp "$ARCHIVE_INCOMING/current_untracked_inventory.tsv" \
    "$OUT/legacy_untracked_inventory.tsv"

  git -C "$LEGACY" ls-files -z > "$ARCHIVE_INCOMING/include_paths.nul"
  while IFS=$'\t' read -r class _size _mtime rel; do
    [[ "$class" = code_or_metadata ]] || continue
    [[ "$rel" = rq009_code_parity_tmp/* ]] && continue
    printf '%s\0' "$rel" >> "$ARCHIVE_INCOMING/include_paths.nul"
  done < <(tail -n +2 "$OUT/legacy_untracked_inventory.tsv")
  sort -zu "$ARCHIVE_INCOMING/include_paths.nul" \
    > "$ARCHIVE_INCOMING/include_paths_sorted.nul"
  rsync -a --from0 --files-from="$ARCHIVE_INCOMING/include_paths_sorted.nul" \
    "$LEGACY/" "$ARCHIVE_INCOMING/files/"

  : > "$ARCHIVE_INCOMING/expected_files.sha256"
  : > "$ARCHIVE_INCOMING/expected_symlinks.tsv"
  while IFS= read -r -d '' rel; do
    if [[ -L "$LEGACY/$rel" ]]; then
      printf '%s\t%s\n' "$rel" "$(readlink "$LEGACY/$rel")" \
        >> "$ARCHIVE_INCOMING/expected_symlinks.tsv"
    elif [[ -f "$LEGACY/$rel" ]]; then
      (cd "$LEGACY" && sha256sum "$rel") \
        >> "$ARCHIVE_INCOMING/expected_files.sha256"
    else
      printf 'Expected archive input is missing: %s\n' "$rel" >&2
      false
    fi
  done < "$ARCHIVE_INCOMING/include_paths_sorted.nul"
  (cd "$ARCHIVE_INCOMING/files" && \
    sha256sum -c ../expected_files.sha256 >/dev/null)
  while IFS=$'\t' read -r rel target; do
    test -L "$ARCHIVE_INCOMING/files/$rel"
    test "$(readlink "$ARCHIVE_INCOMING/files/$rel")" = "$target"
  done < "$ARCHIVE_INCOMING/expected_symlinks.tsv"

  cp "$OUT/legacy_repo.bundle" "$ARCHIVE_INCOMING/repository.bundle"
  cp "$OUT/legacy_tracked.patch" "$ARCHIVE_INCOMING/inventory_tracked.patch"
  cp "$OUT/legacy_status.txt" "$ARCHIVE_INCOMING/inventory_status.txt"
  cp "$OUT/legacy_tracked_files.tsv" "$ARCHIVE_INCOMING/inventory_tracked_files.tsv"
  cp "$OUT/legacy_untracked_inventory.tsv" \
    "$ARCHIVE_INCOMING/inventory_untracked_inventory.tsv"
  cp "$OUT/inventory_checksums.sha256" \
    "$ARCHIVE_INCOMING/inventory_checksums.sha256"
  git bundle verify "$ARCHIVE_INCOMING/repository.bundle" \
    > "$ARCHIVE_INCOMING/bundle_verify.txt"
  git bundle list-heads "$ARCHIVE_INCOMING/repository.bundle" \
    | grep -F "$EXPECTED_LEGACY_HEAD" >/dev/null
  {
    printf '# Retired HPC checkout archive\n\n'
    printf 'Source: `%s`\n\n' "$LEGACY"
    printf 'Frozen HEAD: `%s`\n\n' "$EXPECTED_LEGACY_HEAD"
    printf 'Included: every tracked file plus pre-inventory untracked code/metadata outside `rq009_code_parity_tmp`. Excluded by policy: raw/results payloads and their quarantine copies, logs, caches, and parity scratch outputs. The Git bundle, dirty patch, inventories, exact include list, file hashes, and symlink targets make the archive auditable. This is not a production checkout.\n'
  } > "$ARCHIVE_INCOMING/README.md"
  (cd "$ARCHIVE_INCOMING" && \
    find . -type l -printf '%p\t%l\n' | sort > archive_symlinks.tsv && \
    find . -type f ! -name archive_sha256.txt -print0 | sort -z \
      | xargs -0 sha256sum > archive_sha256.txt && \
    sha256sum -c archive_sha256.txt >/dev/null)
  chmod -R a-w "$ARCHIVE_INCOMING"
  mv "$ARCHIVE_INCOMING" "$ARCHIVE"
  trap - ERR INT TERM
  verify_archive
fi

rollback_retirement() {
  status=$?
  trap - EXIT INT TERM
  if [[ -e "$RETIREMENT_MARKER" ]]; then
    exit "$status"
  fi
  if [[ -d "$ROOT_QUARANTINE" ]]; then
    if [[ -e "$LEGACY" ]]; then
      safe_remove_stub
    fi
    mv "$ROOT_QUARANTINE" "$LEGACY"
  fi
  rm -f -- "$OUT/CODE_RETIREMENT_COMPLETE.incoming.${SLURM_JOB_ID:-$$}"
  rm -rf -- "$SOURCE_AUDIT"
  exit "$status"
}
trap rollback_retirement EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mv "$LEGACY" "$ROOT_QUARANTINE"
mkdir -p "$LEGACY/interhub_traj_lane"
ln -s "$RAW_SNAPSHOT" "$RAW_LINK"
ln -s "$RESULTS_SNAPSHOT" "$RESULTS_LINK"
{
  printf '# Retired legacy checkout\n\n'
  printf 'Executable code was retired after byte-verified data migration and post-switch preflight.\n\n'
  printf 'Use the Git-managed checkout at `%s/code/repo`.\n\n' "$BASE"
  printf 'Frozen legacy code archive: `%s`.\n\n' "$ARCHIVE"
  printf 'Rollback quarantine: `%s`.\n' "$ROOT_QUARANTINE"
} > "$LEGACY/TOMBSTONE.md"

test "$(readlink -f "$RAW_LINK")" = "$RAW_SNAPSHOT"
test "$(readlink -f "$RESULTS_LINK")" = "$RESULTS_SNAPSHOT"
test -z "$(find "$LEGACY" -type f \( -name '*.py' -o -name '*.sh' -o -name '*.sbatch' \) -print -quit)"
verify_archive
rm -rf -- "$SOURCE_AUDIT"

marker_incoming="$OUT/CODE_RETIREMENT_COMPLETE.incoming.${SLURM_JOB_ID:-$$}"
{
  printf 'legacy_head=%s\n' "$EXPECTED_LEGACY_HEAD"
  printf 'archive=%s\n' "$ARCHIVE"
  printf 'quarantine=%s\n' "$ROOT_QUARANTINE"
  printf 'raw_snapshot=%s\n' "$RAW_SNAPSHOT"
  printf 'results_snapshot=%s\n' "$RESULTS_SNAPSHOT"
  printf 'retired_at=%s\n' "$(date --iso-8601=seconds)"
} > "$marker_incoming"
mv "$marker_incoming" "$RETIREMENT_MARKER"
trap - EXIT INT TERM
printf 'Legacy checkout retired: %s\n' "$LEGACY"
