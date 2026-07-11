#!/usr/bin/env bash
set -euo pipefail

BASE="${HPC_SOCIALITY_ROOT:-/share/home/u25310231/ZXC/sociality_estimation}"
LEGACY="${LEGACY_IPV_ROOT:-/share/home/u25310231/ZXC/ipv_estimation}"
RUN_ID="${1:-pre_migration_20260711}"

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid inventory id: $RUN_ID" >&2
  exit 2
fi

OUT="$BASE/manifests/legacy_migration/$RUN_ID"
RAW="$LEGACY/interhub_traj_lane/0_raw_data"
RESULTS="$LEGACY/interhub_traj_lane/1_ipv_estimation_results"

test -d "$LEGACY/.git"
test -d "$RAW"
test -d "$RESULTS"
if [[ -e "$OUT/COMPLETE" ]]; then
  echo "Inventory is already complete: $OUT" >&2
  exit 3
fi
mkdir -p "$OUT"

exec > >(tee "$OUT/inventory.log") 2>&1

printf 'inventory_id=%s\n' "$RUN_ID"
printf 'created_at=%s\n' "$(date --iso-8601=seconds)"
printf 'host=%s\n' "$(hostname)"
printf 'legacy_root=%s\n' "$LEGACY"
df -h "$BASE" > "$OUT/filesystem.txt"
squeue -u "$(id -un)" -o '%.18i %.40j %.2t %.10M %.6D %R' > "$OUT/queue.txt"

git -C "$LEGACY" rev-parse HEAD > "$OUT/legacy_head.txt"
git -C "$LEGACY" remote -v > "$OUT/legacy_remotes.txt"
git -C "$LEGACY" status --porcelain=v1 --untracked-files=all > "$OUT/legacy_status.txt"
git -C "$LEGACY" diff --binary > "$OUT/legacy_tracked.patch"
git -C "$LEGACY" bundle create "$OUT/legacy_repo.bundle" --all

printf 'sha256\tsize_bytes\tpath\n' > "$OUT/legacy_tracked_files.tsv"
while IFS= read -r -d '' rel; do
  path="$LEGACY/$rel"
  if [[ -f "$path" ]]; then
    printf '%s\t%s\t%s\n' \
      "$(sha256sum "$path" | awk '{print $1}')" \
      "$(stat -c '%s' "$path")" "$rel" >> "$OUT/legacy_tracked_files.tsv"
  fi
done < <(git -C "$LEGACY" ls-files -z)

printf 'class\tsize_bytes\tmtime_epoch\tpath\n' > "$OUT/legacy_untracked_inventory.tsv"
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
    >> "$OUT/legacy_untracked_inventory.tsv"
done < <(git -C "$LEGACY" ls-files --others --exclude-standard -z)

find "$RAW" -type l -printf '%p\t%l\n' | sort > "$OUT/raw_symlinks.tsv"
find "$RAW" -type f -printf '%s\t%T@\t%p\n' | sort -k3,3 > "$OUT/raw_inventory.tsv"
find "$RAW" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/raw_sha256.txt"

find "$RESULTS" -type l -printf '%p\t%l\n' | sort > "$OUT/results_symlinks.tsv"
find "$RESULTS" -type f -printf '%s\t%T@\t%p\n' | sort -k3,3 > "$OUT/results_inventory.tsv"
find "$RESULTS" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/results_sha256.txt"

wc -l \
  "$OUT/legacy_tracked_files.tsv" \
  "$OUT/legacy_untracked_inventory.tsv" \
  "$OUT/raw_inventory.tsv" \
  "$OUT/results_inventory.tsv" > "$OUT/counts.txt"

find "$OUT" -maxdepth 1 -type f ! -name inventory_checksums.sha256 \
  ! -name inventory.log ! -name COMPLETE -print0 \
  | sort -z | xargs -0 sha256sum > "$OUT/inventory_checksums.sha256"
touch "$OUT/COMPLETE"
printf 'Inventory complete: %s\n' "$OUT"
