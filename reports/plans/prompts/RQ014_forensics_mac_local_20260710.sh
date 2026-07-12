#!/usr/bin/env bash
# RQ014 Phase F — Mac-local read-only forensics (session transcripts, sibling projects, agent stores).
# STRICTLY READ-ONLY. Run on the Mac from the repo root:
#   bash reports/plans/prompts/RQ014_forensics_mac_local_20260710.sh \
#     > reports/studies/RQ014_wod_e2e_rating_recovery/00_forensics/mac_forensics_raw_$(date +%Y%m%d).txt 2>&1
set -u
KW='wod|e2e|rater|rating.{0,60}(ipv|envelope|deviat)|(ipv|envelope|deviat).{0,60}rating|preference_score|评分.{0,40}(偏离|包络|IPV)|(偏离|包络).{0,40}评分'
PROJ="$HOME/Library/CloudStorage/OneDrive-个人/Desktop/Projects"
echo "=== RQ014 Mac forensics $(date -Is) ==="

echo "== [1] Cowork session transcripts (all sessions, all spaces) =="
SESS="$HOME/Library/Application Support/Claude/local-agent-mode-sessions"
grep -r -l -i -E "$KW" "$SESS" 2>/dev/null | head -60

echo "== [2] Claude Code project transcripts =="
grep -r -l -i -E "$KW" "$HOME/.claude/projects" 2>/dev/null | head -60

echo "== [3] codex CLI session/log stores =="
for d in "$HOME/.codex/sessions" "$HOME/.codex/log" "$HOME/.codex/history.jsonl"; do
  [ -e "$d" ] && grep -r -l -i -E "$KW" "$d" 2>/dev/null | head -30
done

echo "== [4] Sibling project folders under 1_Codes (names + mtimes) =="
ls -lat --time-style=+%Y-%m-%d "$PROJ/1_Codes" 2>/dev/null || ls -lat "$PROJ/1_Codes" | head -30

echo "== [5] WOD/e2e/rating-named files across ALL sibling projects (excl. this repo + venvs) =="
find "$PROJ/1_Codes" -maxdepth 6 \( -iname "*wod*" -o -iname "*rating*" -o -iname "*rater*" \) \
  -not -path "*/2_sociality_estimation/*" -not -path "*venv*" -not -path "*site-packages*" \
  -printf "%TY-%Tm-%Td %12s  %p\n" 2>/dev/null | head -80

echo "== [6] Python files mentioning preference_score outside this repo =="
grep -r -l --include="*.py" "preference_score" "$PROJ/1_Codes" 2>/dev/null \
  | grep -v "2_sociality_estimation" | head -40

echo "== [7] ~/.rq009_codex_fleet residue =="
find "$HOME/.rq009_codex_fleet" -maxdepth 3 -newermt 2025-01-01 \
  -printf "%TY-%Tm-%Td  %p\n" 2>/dev/null | grep -v site-packages | head -40

echo "== [8] Spotlight full-text probe (may be slow; ctrl-C safe) =="
mdfind -onlyin "$PROJ" 'kMDItemTextContent == "*preference_score*"c' 2>/dev/null | head -30

echo "=== end RQ014 Mac forensics ==="
echo "MANUAL follow-ups (cannot be scripted): OneDrive web version-history/recycle-bin for deleted"
echo "analysis dirs; the old Windows machine C:\\Users\\xiaocongzhao\\OneDrive\\...\\2_sociality_estimation."
