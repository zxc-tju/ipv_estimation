#!/usr/bin/env bash
# RQ014 Phase F pass 2 — HPC remainder (sections that were cut off in pass 1). READ-ONLY, bounded.
# Run from Mac:
#   ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc 'bash -s' \
#     < reports/plans/prompts/RQ014_forensics_hpc_pass2_20260710.sh \
#     > reports/studies/RQ014_wod_e2e_rating_recovery/00_forensics/hpc_forensics_pass2_20260710.txt 2>&1
set -u
echo "=== RQ014 HPC pass2 $(date) ==="
echo "== [9] Home top-level (anything predating June 2026 => candidate) =="
ls -lat --time-style=+%Y-%m-%d ~ | head -50
echo "== [10] Home dirs NOT under /ZXC, oldest first (maxdepth 2) =="
find ~ -maxdepth 2 -type d -not -path "$HOME/ZXC*" -not -path "$HOME/.*" -printf "%TY-%Tm-%Td  %p\n" 2>/dev/null | sort | head -60
echo "== [11] bash_history size + any wod/ipv/waymo lines (loose) =="
wc -l ~/.bash_history 2>/dev/null
grep -n -i -E "waymo|ipv|wod" ~/.bash_history 2>/dev/null | head -40
echo "== [12] rating/preference-named files, home-wide but bounded depth =="
find ~ -maxdepth 4 \( -iname "*rating*" -o -iname "*rater*" -o -iname "*preference*" \) \
  -not -path "*/envs/*" -not -path "*conda*" -not -path "*site-packages*" \
  -printf "%TY-%Tm-%Td %12s  %p\n" 2>/dev/null | head -60
echo "== [13] preference_score in code, /ZXC only, excluding known RQ010B code dirs =="
grep -r -l --include="*.py" --include="*.sbatch" "preference_score" ~/ZXC 2>/dev/null | grep -v "RQ010B_wod_e2e/code" | head -40
echo "== [14] anything in /ZXC modified before 2026-06-20 (pre-RQ010B) =="
find ~/ZXC -maxdepth 3 ! -newermt 2026-06-20 -printf "%TY-%Tm-%Td  %p\n" 2>/dev/null | sort | head -60
echo "=== end HPC pass2 ==="
