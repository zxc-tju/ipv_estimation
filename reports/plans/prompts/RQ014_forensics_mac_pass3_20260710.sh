#!/usr/bin/env bash
# RQ014 Phase F pass 3 — last cheap surfaces: sibling projects (external drive), Obsidian,
# paper-repo git history, Cowork stores. READ-ONLY, bounded, fast.
# Run on Mac from repo root:
#   bash reports/plans/prompts/RQ014_forensics_mac_pass3_20260710.sh \
#     > reports/studies/RQ014_wod_e2e_rating_recovery/00_forensics/mac_forensics_pass3_20260710.txt 2>&1
set -u
export LC_ALL=C
KW='preference_score|IPV[ _-]?envelope|envelope[ _-]?deviat|rating.{0,40}deviat|deviat.{0,40}rating'
KWZH='评分|偏离|包络'

echo "=== RQ014 Mac pass3 $(date) ==="

echo "== [A] Sibling projects on external drive =="
P="/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes"
ls -lat "$P" | head -30
find "$P" -maxdepth 5 \( -iname "*wod*" -o -iname "*rating*" -o -iname "*rater*" -o -iname "*preference*" \) \
  -not -path "*/2_sociality_estimation/*" -not -path "*venv*" -not -path "*site-packages*" -not -path "*/.git/*" 2>/dev/null | head -60
echo "-- preference_score / rater in code outside this repo --"
grep -r -l -s --include="*.py" --include="*.ipynb" --include="*.md" -E "preference_score|rater[ _]score" "$P" 2>/dev/null | grep -v 2_sociality_estimation | head -40

echo "== [B] Obsidian vault: WOD/E2E notes + any IPV-rating experiment notes =="
for OB in "$HOME/Library/CloudStorage/OneDrive-个人/Obsidian" "/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Obsidian"; do
  [ -d "$OB" ] || continue
  echo "-- vault: $OB"
  find "$OB" -iname "*WOD*" -o -iname "*E2E*" 2>/dev/null | head -20
  grep -r -l -s -E "$KW" "$OB" 2>/dev/null | head -20
  grep -r -l -s -E "WOD.{0,200}($KWZH)" "$OB" 2>/dev/null | head -20
done
echo "-- print the WODE2E note if found --"
for f in "$HOME/Library/CloudStorage/OneDrive-个人/Obsidian/论文笔记/_inbox/WODE2E.md" "$HOME/Library/CloudStorage/OneDrive-个人/Obsidian/PaperNotes/_inbox/WODE2E.md"; do
  [ -f "$f" ] && echo "--- $f" && head -c 4000 "$f" && echo
done

echo "== [C] Paper repo git history (9_overleaf) =="
PR="/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/9_overleaf/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle"
if [ -d "$PR/.git" ]; then
  cd "$PR"
  echo "-- all branches/commits (last 30) --"; git log --all --oneline | head -30
  echo "-- pickaxe: commits ever adding/removing WOD-E2E / preference_score / rater --"
  git log --all --oneline -S"WOD-E2E" | head -10
  git log --all --oneline -S"preference_score" | head -10
  git log --all --oneline -S"rater" | head -10
  git log --all --oneline -S"评分" | head -10
  echo "-- current tree mentions --"; git grep -n -i -E "wod-?e2e|preference_score|rater score" $(git branch -r --format="%(refname:short)" | head -5) -- 2>/dev/null | head -20
else echo "paper repo not found at $PR"; fi

echo "== [D] Cowork session stores (.jsonl only, exclude today's self-matches) =="
S="$HOME/Library/Application Support/Claude/local-agent-mode-sessions/f6f565d5-642e-4c3a-b743-f76e64c2b293"
find "$S" -name "*.jsonl" ! -newermt "2026-07-10" -print0 2>/dev/null | xargs -0 grep -l -s -E "$KW" 2>/dev/null | head -20
echo "=== end pass3 ==="
