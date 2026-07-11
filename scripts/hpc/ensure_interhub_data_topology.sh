#!/usr/bin/env bash
set -euo pipefail

# Expose the retained, hash-audited InterHub payload through the managed
# deployment root without copying or mutating historical data.
BASE="${HPC_SOCIALITY_ROOT:-/share/home/u25310231/ZXC/sociality_estimation}"
LEGACY_RAW_ROOT="${INTERHUB_LEGACY_RAW_ROOT:-/share/home/u25310231/ZXC/ipv_estimation/interhub_traj_lane/0_raw_data}"
TARGET="$BASE/data/interhub/raw"

DEFAULT_CSV="$LEGACY_RAW_ROOT/subsets_for_yiru/selected_interactive_segments_equalized.csv"
DEFAULT_PKL_ROOT="$LEGACY_RAW_ROOT/subsets_for_yiru/pkl"
FULL_INDEX="$LEGACY_RAW_ROOT/full_datasets/index.csv"
FULL_PKL_ROOT="$LEGACY_RAW_ROOT/full_datasets/pkl"

test -f "$DEFAULT_CSV"
test -d "$DEFAULT_PKL_ROOT"
test -f "$FULL_INDEX"
test -d "$FULL_PKL_ROOT"

mkdir -p "$BASE/data/interhub"

if [[ -L "$TARGET" ]]; then
  [[ "$(readlink -f "$TARGET")" == "$(readlink -f "$LEGACY_RAW_ROOT")" ]] || {
    echo "Refusing to replace data link with a different target: $TARGET" >&2
    exit 2
  }
elif [[ -e "$TARGET" ]]; then
  echo "Refusing to replace an existing non-symlink path: $TARGET" >&2
  exit 3
else
  ln -s "$LEGACY_RAW_ROOT" "$TARGET"
fi

test -f "$TARGET/subsets_for_yiru/selected_interactive_segments_equalized.csv"
test -d "$TARGET/subsets_for_yiru/pkl"
test -f "$TARGET/full_datasets/index.csv"
test -d "$TARGET/full_datasets/pkl"

printf 'InterHub data topology ready: %s -> %s\n' "$TARGET" "$(readlink -f "$TARGET")"
printf 'Default CLI inputs: %s ; %s\n' \
  "$TARGET/subsets_for_yiru/selected_interactive_segments_equalized.csv" \
  "$TARGET/subsets_for_yiru/pkl"
