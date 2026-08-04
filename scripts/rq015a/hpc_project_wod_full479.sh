#!/usr/bin/env bash
# RQ015A — WOD full479 CSV HPC-side column projection only.
#
# This script creates a new staging directory, projects the single authorized
# source CSV by column name, and writes a sanitization receipt. It does not copy
# data off HPC, submit jobs, or print cell values.
#
# Optional environment:
#   RQ015A_OUTPUT_DIR=/share/home/u25310231/ZXC/RQ015A_wod_readonly_retrieval/staging/<new-dir>

set -euo pipefail

TOOL_VERSION="rq015a-hpc-project-wod-full479-v2"
SOURCE_PATH="/share/home/u25310231/ZXC/RQ010B_wod_e2e/results/rq010b_wod_e2e_multiframe_tracking_ipv_full479_scored_audited_20260630T063600/rq010b_wod_e2e_multiframe_tracking_ipv_full479_audited_candidate_ipv_rating.csv"
EXPECTED_SOURCE_SHA256="290ef593460d613491cac9a9d8e3de67f10384091869db5401a2ea608039653c"
EXPECTED_SOURCE_ROW_COUNT="906"
PROJECTED_FILENAME="rq010b_wod_full479_audited_candidate_ipv_projected.csv"
RECEIPT_FILENAME="sanitization_receipt.json"
STAGE_ROOT="${RQ015A_STAGE_ROOT:-/share/home/u25310231/ZXC/RQ015A_wod_readonly_retrieval/staging}"
RUN_ID="${RQ015A_STAGE_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${RQ015A_OUTPUT_DIR:-$STAGE_ROOT/rq015a_wod_full479_projected_${RUN_ID}}"

case "$OUT_DIR" in
  /share/home/u25310231/ZXC/RQ015A_wod_readonly_retrieval/*) ;;
  *)
    printf 'error_count=1\n' >&2
    exit 2
    ;;
esac

if [ -e "$OUT_DIR" ]; then
  printf 'error_count=1\n' >&2
  exit 2
fi

mkdir -p "$(dirname "$OUT_DIR")"
mkdir "$OUT_DIR"

cleanup_on_error() {
  status=$?
  if [ "$status" -ne 0 ]; then
    rm -rf "$OUT_DIR"
  fi
  exit "$status"
}
trap cleanup_on_error EXIT

export TOOL_VERSION
export SOURCE_PATH
export EXPECTED_SOURCE_SHA256
export EXPECTED_SOURCE_ROW_COUNT
export PROJECTED_FILENAME
export RECEIPT_FILENAME
export OUT_DIR

python3 - <<'PY'
import csv
import hashlib
import json
import os
import re
import socket
import sys
from datetime import datetime


WHITELIST = (
    "segment_key",
    "candidate_index",
    "ego_ipv",
    "ego_ipv_error",
)
FORBIDDEN_RE = re.compile(r"(rating|preference|human|score)", re.IGNORECASE)
REQUIRED_RECEIPT_FIELDS = (
    "source_path",
    "source_sha256",
    "source_row_count",
    "projected_path",
    "projected_sha256",
    "projected_row_count",
    "column_whitelist",
    "columns_dropped",
    "forbidden_column_scan",
    "generated_at",
    "host",
    "tool_version",
)


def fail(message):
    print("error_count=1", file=sys.stderr)
    print("error_kind=%s" % message, file=sys.stderr)
    raise SystemExit(2)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


source_path = os.environ["SOURCE_PATH"]
out_dir = os.environ["OUT_DIR"]
projected_path = os.path.join(out_dir, os.environ["PROJECTED_FILENAME"])
receipt_path = os.path.join(out_dir, os.environ["RECEIPT_FILENAME"])
expected_source_sha256 = os.environ["EXPECTED_SOURCE_SHA256"]
expected_source_row_count = int(os.environ["EXPECTED_SOURCE_ROW_COUNT"])

if not os.path.isfile(source_path):
    fail("source_missing")

source_sha256 = sha256_file(source_path)
if source_sha256 != expected_source_sha256:
    fail("source_sha256_mismatch")

with open(source_path, "r", encoding="utf-8", newline="") as source_file:
    reader = csv.reader(source_file)
    try:
        header = next(reader)
    except StopIteration:
        fail("source_empty")

    if len(header) != len(set(header)):
        fail("duplicate_source_columns")

    missing = [name for name in WHITELIST if name not in header]
    if missing:
        fail("missing_whitelist_columns")

    if any(FORBIDDEN_RE.search(name) for name in WHITELIST):
        fail("forbidden_whitelist_column")

    index_by_name = {name: index for index, name in enumerate(header)}
    keep_indices = [index_by_name[name] for name in WHITELIST]
    columns_dropped = [name for name in header if name not in WHITELIST]
    if "rating" not in columns_dropped:
        fail("rating_not_confirmed_dropped")

    source_forbidden_columns = [name for name in header if FORBIDDEN_RE.search(name)]
    if not source_forbidden_columns:
        fail("source_forbidden_columns_not_observed")

    source_row_count = 0
    with open(projected_path, "w", encoding="utf-8", newline="") as projected_file:
        writer = csv.writer(projected_file, lineterminator="\n")
        writer.writerow(WHITELIST)
        for row in reader:
            if len(row) != len(header):
                fail("source_row_width_mismatch")
            writer.writerow([row[index] for index in keep_indices])
            source_row_count += 1

if source_row_count != expected_source_row_count:
    fail("source_row_count_mismatch")

projected_sha256 = sha256_file(projected_path)
with open(projected_path, "r", encoding="utf-8", newline="") as projected_file:
    projected_reader = csv.reader(projected_file)
    try:
        projected_header = next(projected_reader)
    except StopIteration:
        fail("projected_empty")
    projected_row_count = sum(1 for _ in projected_reader)

if projected_header != list(WHITELIST):
    fail("projected_header_mismatch")
if projected_row_count != source_row_count:
    fail("projected_row_count_mismatch")

forbidden_projected_columns = [
    name for name in projected_header if FORBIDDEN_RE.search(name)
]
if forbidden_projected_columns:
    fail("forbidden_projected_column")

receipt = {
    "schema_version": "rq015a-wod-full479-sanitization-receipt-v1",
    "source_path": source_path,
    "source_sha256": source_sha256,
    "source_row_count": source_row_count,
    "projected_path": projected_path,
    "projected_sha256": projected_sha256,
    "projected_row_count": projected_row_count,
    "column_whitelist": list(WHITELIST),
    "columns_dropped": columns_dropped,
    "source_forbidden_columns_observed": source_forbidden_columns,
    "forbidden_column_scan": {
        "pattern": "(?i)(rating|preference|human|score)",
        "match_count": len(forbidden_projected_columns),
        "matched_columns": forbidden_projected_columns,
        "required_match_count": 0,
    },
    "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    "host": socket.gethostname(),
    "tool_version": os.environ["TOOL_VERSION"],
    "transfer_performed": False,
}

with open(receipt_path, "w", encoding="utf-8") as receipt_file:
    json.dump(receipt, receipt_file, ensure_ascii=False, indent=2, sort_keys=True)
    receipt_file.write("\n")

with open(receipt_path, "r", encoding="utf-8") as receipt_file:
    loaded = json.load(receipt_file)

missing_receipt_fields = [
    field for field in REQUIRED_RECEIPT_FIELDS if field not in loaded
]
if missing_receipt_fields:
    fail("receipt_missing_required_fields")
if loaded["column_whitelist"] != list(WHITELIST):
    fail("receipt_column_whitelist_mismatch")
if loaded["projected_sha256"] != sha256_file(projected_path):
    fail("receipt_projected_sha256_mismatch")
if loaded["projected_row_count"] != projected_row_count:
    fail("receipt_projected_row_count_mismatch")
if loaded["forbidden_column_scan"]["match_count"] != 0:
    fail("receipt_forbidden_scan_mismatch")

print("projected_column_count=%d" % len(WHITELIST))
print("projected_columns=%s" % ",".join(WHITELIST))
print("columns_dropped_count=%d" % len(columns_dropped))
print("columns_dropped=%s" % ",".join(columns_dropped))
print("source_forbidden_column_count=%d" % len(source_forbidden_columns))
print("source_forbidden_columns=%s" % ",".join(source_forbidden_columns))
print("source_row_count=%d" % source_row_count)
print("projected_row_count=%d" % projected_row_count)
print("forbidden_projected_column_count=0")
print("output_file_count=2")
PY

trap - EXIT
