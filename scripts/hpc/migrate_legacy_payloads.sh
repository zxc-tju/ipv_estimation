#!/usr/bin/env bash
set -euo pipefail

BASE="${HPC_SOCIALITY_ROOT:-/share/home/u25310231/ZXC/sociality_estimation}"
LEGACY="${LEGACY_IPV_ROOT:-/share/home/u25310231/ZXC/ipv_estimation}"
INVENTORY_ID="${INVENTORY_ID:-pre_migration_20260711_v1}"
SNAPSHOT_ID="${SNAPSHOT_ID:-interhub_legacy_20260711_v1}"
OUT="$BASE/manifests/legacy_migration/$INVENTORY_ID"

RAW_SOURCE="$LEGACY/interhub_traj_lane/0_raw_data"
RESULTS_SOURCE="$LEGACY/interhub_traj_lane/1_ipv_estimation_results"
RAW_PARENT="$BASE/data/interhub/snapshots"
RESULTS_PARENT="$BASE/archives/historical-results"
RAW_INCOMING="$RAW_PARENT/$SNAPSHOT_ID.incoming"
RESULTS_INCOMING="$RESULTS_PARENT/$SNAPSHOT_ID.incoming"
RAW_SNAPSHOT="$RAW_PARENT/$SNAPSHOT_ID"
RESULTS_SNAPSHOT="$RESULTS_PARENT/$SNAPSHOT_ID"
RAW_QUARANTINE="$LEGACY/interhub_traj_lane/0_raw_data.quarantine_$SNAPSHOT_ID"
RESULTS_QUARANTINE="$LEGACY/interhub_traj_lane/1_ipv_estimation_results.quarantine_$SNAPSHOT_ID"
NEW_RAW_LINK="$BASE/data/interhub/raw"
RESUME_MIGRATION="${RESUME_MIGRATION:-0}"
VERIFY_WORKERS="${VERIFY_WORKERS:-8}"

test -e "$OUT/COMPLETE"
command -v flock >/dev/null
exec 9>"$OUT/migration.lock"
flock -n 9 || {
  printf 'Another legacy payload migration is already running: %s\n' "$OUT/migration.lock" >&2
  exit 4
}
test -d "$RAW_SOURCE"
test ! -L "$RAW_SOURCE"
test -d "$RESULTS_SOURCE"
test ! -L "$RESULTS_SOURCE"
test -L "$NEW_RAW_LINK"
test "$(readlink -f "$NEW_RAW_LINK")" = "$(readlink -f "$RAW_SOURCE")"
for path in "$RAW_SNAPSHOT" "$RESULTS_SNAPSHOT" "$RAW_QUARANTINE" "$RESULTS_QUARANTINE"; do
  test ! -e "$path"
done
if [[ "$RESUME_MIGRATION" != 1 ]]; then
  test ! -e "$RAW_INCOMING"
  test ! -e "$RESULTS_INCOMING"
fi

mkdir -p "$RAW_PARENT" "$RESULTS_PARENT"
rsync -aH --links "$RAW_SOURCE/" "$RAW_INCOMING/"
rsync -aH --links "$RESULTS_SOURCE/" "$RESULTS_INCOMING/"

sed "s#  $RAW_SOURCE/#  $RAW_INCOMING/#" "$OUT/raw_sha256.txt" \
  | sha256sum -c - > "$OUT/raw_copy_verify.txt"
VERIFY_DIR="$OUT/results_verify_chunks"
rm -rf "$VERIFY_DIR"
mkdir "$VERIFY_DIR"
sed "s#  $RESULTS_SOURCE/#  $RESULTS_INCOMING/#" "$OUT/results_sha256.txt" \
  | split -n "l/$VERIFY_WORKERS" - "$VERIFY_DIR/chunk_"
find "$VERIFY_DIR" -type f -print0 \
  | xargs -0 -r -n 1 -P "$VERIFY_WORKERS" sha256sum -c \
  > "$OUT/results_copy_verify.txt"
rm -rf "$VERIFY_DIR"
test "$(grep -c ': OK$' "$OUT/raw_copy_verify.txt")" -eq "$(wc -l < "$OUT/raw_sha256.txt")"
test "$(grep -c ': OK$' "$OUT/results_copy_verify.txt")" -eq "$(wc -l < "$OUT/results_sha256.txt")"

test -L "$RAW_INCOMING/full_datasets/pkl"
test -d "$(readlink -f "$RAW_INCOMING/full_datasets/pkl")"
test -f "$RAW_INCOMING/full_datasets/index.csv"
test -f "$RAW_INCOMING/subsets_for_yiru/selected_interactive_segments_equalized.csv"

chmod -R a-w "$RAW_INCOMING" "$RESULTS_INCOMING"
mv "$RAW_INCOMING" "$RAW_SNAPSHOT"
mv "$RESULTS_INCOMING" "$RESULTS_SNAPSHOT"

rollback() {
  set +e
  if [[ -L "$RAW_SOURCE" && -d "$RAW_QUARANTINE" ]]; then
    rm "$RAW_SOURCE"
    mv "$RAW_QUARANTINE" "$RAW_SOURCE"
  fi
  if [[ -L "$RESULTS_SOURCE" && -d "$RESULTS_QUARANTINE" ]]; then
    rm "$RESULTS_SOURCE"
    mv "$RESULTS_QUARANTINE" "$RESULTS_SOURCE"
  fi
  if [[ ! -L "$NEW_RAW_LINK" || "$(readlink -f "$NEW_RAW_LINK" 2>/dev/null)" != "$(readlink -f "$RAW_SOURCE" 2>/dev/null)" ]]; then
    rm -f "$NEW_RAW_LINK"
    ln -s "$RAW_SOURCE" "$NEW_RAW_LINK"
  fi
}
trap rollback ERR

rm "$NEW_RAW_LINK"
mv "$RAW_SOURCE" "$RAW_QUARANTINE"
ln -s "$RAW_SNAPSHOT" "$RAW_SOURCE"
ln -s "$RAW_SNAPSHOT" "$NEW_RAW_LINK"

mv "$RESULTS_SOURCE" "$RESULTS_QUARANTINE"
ln -s "$RESULTS_SNAPSHOT" "$RESULTS_SOURCE"

test "$(readlink -f "$NEW_RAW_LINK")" = "$RAW_SNAPSHOT"
test "$(readlink -f "$RAW_SOURCE")" = "$RAW_SNAPSHOT"
test "$(readlink -f "$RESULTS_SOURCE")" = "$RESULTS_SNAPSHOT"
test -d "$RAW_QUARANTINE"
test -d "$RESULTS_QUARANTINE"
trap - ERR

{
  printf 'snapshot_id=%s\n' "$SNAPSHOT_ID"
  printf 'raw_snapshot=%s\n' "$RAW_SNAPSHOT"
  printf 'results_snapshot=%s\n' "$RESULTS_SNAPSHOT"
  printf 'raw_quarantine=%s\n' "$RAW_QUARANTINE"
  printf 'results_quarantine=%s\n' "$RESULTS_QUARANTINE"
  printf 'switched_at=%s\n' "$(date --iso-8601=seconds)"
} > "$OUT/SWITCH_COMPLETE"
printf 'Legacy payload migration complete: %s\n' "$SNAPSHOT_ID"
