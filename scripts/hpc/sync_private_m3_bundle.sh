#!/usr/bin/env bash
set -euo pipefail

SSH_ALIAS="${SSH_ALIAS:-tongji-hpc}"
LOCAL_DIR="${LOCAL_M3_DIR:-models/rq009_m3}"
REMOTE_DIR="${REMOTE_M3_DIR:-/share/home/u25310231/ZXC/sociality_estimation/checkpoints/rq009_m3}"
MODEL="$LOCAL_DIR/m3_scorer.joblib"
CONTRACT="$LOCAL_DIR/feature_spec_contract.json"
MANIFEST="$LOCAL_DIR/manifest.json"

for path in "$MODEL" "$CONTRACT" "$MANIFEST"; do
  test -f "$path"
done

expected="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["artifact"]["sha256"])' "$MANIFEST")"
observed="$(shasum -a 256 "$MODEL" | awk '{print $1}')"
[[ "$observed" == "$expected" ]]

ssh -o BatchMode=yes -o ConnectTimeout=20 "$SSH_ALIAS" \
  "mkdir -p '$REMOTE_DIR/.incoming'"
scp -p -o BatchMode=yes -o ConnectTimeout=30 \
  "$MODEL" "$CONTRACT" "$MANIFEST" \
  "$SSH_ALIAS:$REMOTE_DIR/.incoming/"
ssh -o BatchMode=yes -o ConnectTimeout=20 "$SSH_ALIAS" "
  set -euo pipefail
  test \"\$(sha256sum '$REMOTE_DIR/.incoming/m3_scorer.joblib' | awk '{print \\$1}')\" = '$expected'
  mv '$REMOTE_DIR/.incoming/m3_scorer.joblib' '$REMOTE_DIR/m3_scorer.joblib'
  mv '$REMOTE_DIR/.incoming/feature_spec_contract.json' '$REMOTE_DIR/feature_spec_contract.json'
  mv '$REMOTE_DIR/.incoming/manifest.json' '$REMOTE_DIR/manifest.json'
  rmdir '$REMOTE_DIR/.incoming'
  printf 'M3 bundle synchronized: %s\\n' '$expected'
"
