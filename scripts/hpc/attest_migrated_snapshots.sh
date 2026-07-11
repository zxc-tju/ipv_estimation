#!/usr/bin/env bash
set -euo pipefail

BASE="/share/home/u25310231/ZXC/sociality_estimation"
LEGACY="/share/home/u25310231/ZXC/ipv_estimation"
INVENTORY_ID="pre_migration_20260711_v1"
SNAPSHOT_ID="interhub_legacy_20260711_v1"
OUT="$BASE/manifests/legacy_migration/$INVENTORY_ID"
RAW_SNAPSHOT="$BASE/data/interhub/snapshots/$SNAPSHOT_ID"
RESULTS_SNAPSHOT="$BASE/archives/historical-results/$SNAPSHOT_ID"
RAW_SOURCE="$LEGACY/interhub_traj_lane/0_raw_data"
RESULTS_SOURCE="$LEGACY/interhub_traj_lane/1_ipv_estimation_results"
WORKERS="${SLURM_CPUS_PER_TASK:-8}"
VERIFY_FINAL="$OUT/snapshot_reverification_$SNAPSHOT_ID"
ATTESTATION="$VERIFY_FINAL/attestation.json"
VERIFY_INCOMING="$OUT/snapshot_reverification.incoming.${SLURM_JOB_ID:-$$}"

test -f "$OUT/SWITCH_COMPLETE"
test ! -e "$ATTESTATION"
test ! -e "$VERIFY_FINAL"
test ! -e "$VERIFY_INCOMING"
test -d "$RAW_SNAPSHOT"
test -d "$RESULTS_SNAPSHOT"
test ! -L "$RAW_SNAPSHOT"
test ! -L "$RESULTS_SNAPSHOT"

command -v flock >/dev/null
exec 9>"$OUT/migration.lock"
flock -n 9 || {
  printf 'The payload migration is still active.\n' >&2
  exit 4
}
exec 8>"$BASE/manifests/runtime_maintenance.lock"
flock -x -n 8 || {
  printf 'A research run or maintenance operation is active.\n' >&2
  exit 5
}

cleanup() {
  status=$?
  trap - EXIT INT TERM
  chmod -R u+w "$VERIFY_INCOMING" 2>/dev/null || true
  rm -rf -- "$VERIFY_INCOMING"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir "$VERIFY_INCOMING"
sed "s#  $RAW_SOURCE/#  $RAW_SNAPSHOT/#" "$OUT/raw_sha256.txt" \
  > "$VERIFY_INCOMING/raw_snapshot_sha256.txt"
sha256sum -c "$VERIFY_INCOMING/raw_snapshot_sha256.txt" \
  > "$VERIFY_INCOMING/raw_snapshot_verify.txt"

sed "s#  $RESULTS_SOURCE/#  $RESULTS_SNAPSHOT/#" "$OUT/results_sha256.txt" \
  > "$VERIFY_INCOMING/results_snapshot_sha256.txt"
split -n "l/$WORKERS" "$VERIFY_INCOMING/results_snapshot_sha256.txt" \
  "$VERIFY_INCOMING/results_chunk_"
find "$VERIFY_INCOMING" -type f -name 'results_chunk_*' -print0 \
  | xargs -0 -r -n 1 -P "$WORKERS" sha256sum -c \
  > "$VERIFY_INCOMING/results_snapshot_verify.txt"
rm "$VERIFY_INCOMING"/results_chunk_*

raw_count="$(wc -l < "$OUT/raw_sha256.txt")"
results_count="$(wc -l < "$OUT/results_sha256.txt")"
test "$raw_count" -eq 51
test "$results_count" -eq 173034
test "$(grep -c ': OK$' "$VERIFY_INCOMING/raw_snapshot_verify.txt")" \
  -eq "$raw_count"
test "$(grep -c ': OK$' "$VERIFY_INCOMING/results_snapshot_verify.txt")" \
  -eq "$results_count"
test -z "$(find "$RAW_SNAPSHOT" "$RESULTS_SNAPSHOT" ! -type l -perm /222 -print -quit)"

python3 - "$OUT/SWITCH_COMPLETE" "$OUT/raw_sha256.txt" \
  "$OUT/results_sha256.txt" "$VERIFY_INCOMING/raw_snapshot_verify.txt" \
  "$VERIFY_INCOMING/results_snapshot_verify.txt" "$SNAPSHOT_ID" \
  "$RAW_SNAPSHOT" "$RESULTS_SNAPSHOT" "$raw_count" "$results_count" \
  "$VERIFY_INCOMING/attestation.json" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = {
    "schema_version": 1,
    "snapshot_id": sys.argv[6],
    "raw_snapshot": sys.argv[7],
    "results_snapshot": sys.argv[8],
    "switch_marker_sha256": sha256(sys.argv[1]),
    "source_manifest_sha256": {
        "raw": sha256(sys.argv[2]),
        "results": sha256(sys.argv[3]),
    },
    "snapshot_verification_sha256": {
        "raw": sha256(sys.argv[4]),
        "results": sha256(sys.argv[5]),
    },
    "verification_counts": {
        "raw": int(sys.argv[9]),
        "results": int(sys.argv[10]),
    },
    "verification_mode": "fresh_manifest_to_final_snapshot_sha256",
    "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
    "verified_at": datetime.now(timezone.utc).isoformat(),
}
with open(sys.argv[11], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

chmod -R a-w "$VERIFY_INCOMING"
mv "$VERIFY_INCOMING" "$VERIFY_FINAL"
trap - EXIT INT TERM
printf 'Snapshot verification attested: %s\n' "$ATTESTATION"
