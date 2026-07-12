#!/usr/bin/env bash
# RQ014 Phase F pass 2 — tight-regex triage with content snippets. READ-ONLY.
# Run on Mac from repo root:
#   bash reports/plans/prompts/RQ014_forensics_mac_pass2_20260710.sh \
#     > reports/studies/RQ014_wod_e2e_rating_recovery/00_forensics/mac_forensics_pass2_20260710.txt 2>&1
set -u
# Tight patterns: real study vocabulary only (no bare "e2e"/"wod" which false-match hex).
KW='preference_score|rater[ _]score|human[ _]rating|IPV[ _-]?envelope|envelope[ _-]?deviat|deviation.{0,40}rating|rating.{0,40}deviation|评分.{0,60}(IPV|包络|偏离)|(包络|偏离).{0,60}评分|waymo.{0,40}(rating|preference)|WOD[- _]?E2E'
snip() { # file list on stdin -> filename + up to 3 bounded snippets each
  while IFS= read -r f; do
    echo "--- $f"
    LC_ALL=C grep -a -o -m3 -E ".{0,100}($KW).{0,160}" "$f" 2>/dev/null | cut -c1-320
  done
}
echo "=== RQ014 Mac pass2 $(date) ==="

echo "== [A] codex sessions (ALL rollouts, tight regex, with snippets) =="
grep -r -l -a -E "$KW" "$HOME/.codex/sessions" 2>/dev/null | snip

echo "== [B] codex history.jsonl =="
LC_ALL=C grep -a -o -E ".{0,100}($KW).{0,160}" "$HOME/.codex/history.jsonl" 2>/dev/null | cut -c1-320 | head -40

echo "== [C] Claude Code transcripts: old interhub_traj_lane project dir (full keyword hits) =="
grep -r -l -a -E "$KW" "$HOME/.claude/projects/-Users-xiaocong-Library-CloudStorage-OneDrive----Desktop-Projects-1-Codes-2-sociality-estimation-interhub-traj-lane-1-ipv-estimation-results" 2>/dev/null | snip

echo "== [D] Claude Code transcripts: main repo project dir, hits BEFORE June 2026 only (mtime) =="
find "$HOME/.claude/projects/-Users-xiaocong-Library-CloudStorage-OneDrive----Desktop-Projects-1-Codes-2-sociality-estimation" -name "*.jsonl" ! -newermt 2026-06-01 2>/dev/null | while IFS= read -r f; do
  if LC_ALL=C grep -q -a -E "$KW" "$f" 2>/dev/null; then echo "$f"; fi
done | snip

echo "== [E] Cowork session stores (tight regex, snippets) =="
grep -r -l -a -E "$KW" "$HOME/Library/Application Support/Claude/local-agent-mode-sessions/f6f565d5-642e-4c3a-b743-f76e64c2b293" 2>/dev/null | grep -v -E "skills|fonts|node_modules|\.png|\.jpg|\.ttf|\.xsd|\.pdf" | snip

echo "== [F] Sibling projects on external drive (corrected root) =="
PROJ="/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects"
ls -lat "$PROJ/1_Codes" 2>/dev/null | head -30
find "$PROJ/1_Codes" -maxdepth 5 \( -iname "*wod*" -o -iname "*rating*" -o -iname "*rater*" -o -iname "*preference*" \) \
  -not -path "*/2_sociality_estimation/*" -not -path "*venv*" -not -path "*site-packages*" -not -path "*/.git/*" 2>/dev/null | head -60
echo "-- py/ipynb mentioning preference_score or rater outside this repo --"
grep -r -l --include="*.py" --include="*.ipynb" --include="*.md" -E "preference_score|rater[ _]score" "$PROJ/1_Codes" 2>/dev/null | grep -v 2_sociality_estimation | head -40

echo "== [G] Old OneDrive location (if the ~/Library/CloudStorage path also exists) =="
for R in "$HOME/Library/CloudStorage"/OneDrive*; do
  [ -d "$R/Desktop/Projects/1_Codes" ] && ls -lat "$R/Desktop/Projects/1_Codes" | head -20
done
echo "=== end pass2 ==="
