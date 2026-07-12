#!/usr/bin/env bash
# RQ014 Phase F pass 3b — no deep find on CloudStorage (avoids OneDrive hydration hangs).
# Run on Mac from repo root:
#   bash reports/plans/prompts/RQ014_forensics_mac_pass3b_20260710.sh \
#     > reports/studies/RQ014_wod_e2e_rating_recovery/00_forensics/mac_forensics_pass3b_20260710.txt 2>&1
set -u
export LC_ALL=C
B="/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes"
echo "=== RQ014 Mac pass3b $(date) ==="

echo "== [A] 1_Codes/archived — one-level listing only =="
ls -lat "$B/archived" 2>/dev/null | head -25
for d in "$B/archived"/*/; do echo "-- $d"; ls "$d" 2>/dev/null | head -10; done

echo "== [B] paper repo git history (local .git, no hydration) =="
PR="$B/9_overleaf/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle"
if [ -d "$PR/.git" ]; then
  git -C "$PR" log --all --oneline | head -40
  echo "-- pickaxe WOD-E2E --";        git -C "$PR" log --all --oneline -S"WOD-E2E" | head
  echo "-- pickaxe preference_score --"; git -C "$PR" log --all --oneline -S"preference_score" | head
  echo "-- pickaxe rater --";           git -C "$PR" log --all --oneline -S"rater" | head
  echo "-- HEAD tree mentions --";      git -C "$PR" grep -n -i -E "wod-?e2e|rater|preference score" HEAD 2>/dev/null | head -20
else echo "no git at $PR"; ls "$B/9_overleaf" 2>/dev/null; fi

echo "== [C] Obsidian WODE2E notes (targeted reads only) =="
for f in \
  "$HOME/Library/CloudStorage/OneDrive-个人/Obsidian/论文笔记/_inbox/WODE2E.md" \
  "$HOME/Library/CloudStorage/OneDrive-个人/Obsidian/PaperNotes/_inbox/WODE2E.md" \
  "/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Obsidian/论文笔记/_inbox/WODE2E.md" \
  "/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Obsidian/PaperNotes/_inbox/WODE2E.md"; do
  [ -f "$f" ] && { echo "--- $f"; head -c 5000 "$f"; echo; }
done

echo "== [D] Cowork session jsonl (local ~/Library, pre-today) =="
S="$HOME/Library/Application Support/Claude/local-agent-mode-sessions/f6f565d5-642e-4c3a-b743-f76e64c2b293"
find "$S" -name "*.jsonl" ! -newermt "2026-07-10" -print0 2>/dev/null \
 | xargs -0 grep -l -s -E "preference_score|IPV[ _-]?envelope|rating.{0,40}deviat|deviat.{0,40}rating" 2>/dev/null | head -20
echo "=== end pass3b ==="
