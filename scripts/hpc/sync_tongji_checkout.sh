#!/usr/bin/env bash
set -euo pipefail

BRANCH="${BRANCH:-main}"
COMMIT="${COMMIT:-}"
BASE="${HPC_SOCIALITY_ROOT:-/share/home/u25310231/ZXC/sociality_estimation}"
REPO="$BASE/code/repo"
ORIGIN="https://github.com/zxc-tju/ipv_estimation.git"
BOOTSTRAP_BUNDLE="${BOOTSTRAP_BUNDLE:-}"

if [[ -z "$COMMIT" ]]; then
  echo "COMMIT must be an exact published Git commit" >&2
  exit 2
fi

mkdir -p \
  "$BASE/code" "$BASE/data" "$BASE/envs" "$BASE/scripts" \
  "$BASE/logs" "$BASE/work_dirs" "$BASE/manifests" "$BASE/checkpoints"

new_clone=0
if [[ ! -d "$REPO/.git" ]]; then
  if [[ -n "$BOOTSTRAP_BUNDLE" ]]; then
    test -f "$BOOTSTRAP_BUNDLE"
    git clone --no-checkout "$BOOTSTRAP_BUNDLE" "$REPO"
  else
    git clone --filter=blob:none --no-checkout "$ORIGIN" "$REPO"
  fi
  new_clone=1
fi

if [[ "$new_clone" -eq 0 && -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "Refusing to sync a dirty HPC checkout: $REPO" >&2
  exit 3
fi

git -C "$REPO" remote set-url origin "$ORIGIN"
if ! git -C "$REPO" cat-file -e "${COMMIT}^{commit}" 2>/dev/null; then
  git -C "$REPO" fetch --prune origin "$BRANCH"
fi
git -C "$REPO" cat-file -e "${COMMIT}^{commit}"
git -C "$REPO" checkout --detach "$COMMIT"

observed="$(git -C "$REPO" rev-parse HEAD)"
[[ "$observed" == "$COMMIT" ]]
[[ -z "$(git -C "$REPO" status --porcelain)" ]]
printf 'HPC checkout synchronized: %s\n' "$observed"
