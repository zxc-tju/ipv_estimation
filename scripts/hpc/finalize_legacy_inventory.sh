#!/usr/bin/env bash
set -euo pipefail

BASE="${HPC_SOCIALITY_ROOT:-/share/home/u25310231/ZXC/sociality_estimation}"
RUN_ID="${1:-pre_migration_20260711}"
HASH_WORKERS="${HASH_WORKERS:-8}"
OUT="$BASE/manifests/legacy_migration/$RUN_ID"

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid inventory id: $RUN_ID" >&2
  exit 2
fi
if [[ ! "$HASH_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "HASH_WORKERS must be a positive integer" >&2
  exit 2
fi

test -d "$OUT"
test ! -e "$OUT/COMPLETE"
for required in \
  legacy_repo.bundle legacy_head.txt legacy_status.txt legacy_tracked.patch \
  legacy_tracked_files.tsv legacy_untracked_inventory.tsv \
  raw_inventory.tsv raw_sha256.txt raw_symlinks.tsv \
  results_inventory.tsv results_symlinks.tsv; do
  test -f "$OUT/$required"
done

expected="$(wc -l < "$OUT/results_inventory.tsv")"
test "$expected" -gt 0
cut -f3- "$OUT/results_inventory.tsv" > "$OUT/results_paths.txt"
test "$(wc -l < "$OUT/results_paths.txt")" -eq "$expected"

rm -f "$OUT/results_sha256.incoming" "$OUT/results_sha256.sorted"
xargs -d '\n' -r -n 32 -P "$HASH_WORKERS" sha256sum \
  < "$OUT/results_paths.txt" > "$OUT/results_sha256.incoming"
test "$(wc -l < "$OUT/results_sha256.incoming")" -eq "$expected"
LC_ALL=C sort -k2 "$OUT/results_sha256.incoming" > "$OUT/results_sha256.sorted"
mv "$OUT/results_sha256.sorted" "$OUT/results_sha256.txt"
rm -f "$OUT/results_sha256.incoming" "$OUT/results_paths.txt"

wc -l \
  "$OUT/legacy_tracked_files.tsv" \
  "$OUT/legacy_untracked_inventory.tsv" \
  "$OUT/raw_inventory.tsv" \
  "$OUT/results_inventory.tsv" \
  "$OUT/raw_sha256.txt" \
  "$OUT/results_sha256.txt" > "$OUT/counts.txt"

find "$OUT" -maxdepth 1 -type f ! -name inventory_checksums.sha256 \
  ! -name inventory.log ! -name COMPLETE -print0 \
  | sort -z | xargs -0 sha256sum > "$OUT/inventory_checksums.sha256"
touch "$OUT/COMPLETE"
printf 'Inventory finalized: %s (%s result files)\n' "$OUT" "$expected"
