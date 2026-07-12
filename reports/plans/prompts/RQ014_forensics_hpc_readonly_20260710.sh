#!/usr/bin/env bash
# RQ014 Phase F — HPC read-only forensics for the lost WOD-E2E rating~IPV-deviation study.
# STRICTLY READ-ONLY. Run from Mac:
#   ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc 'bash -s' \
#     < reports/plans/prompts/RQ014_forensics_hpc_readonly_20260710.sh \
#     > reports/studies/RQ014_wod_e2e_rating_recovery/00_forensics/hpc_forensics_raw_$(date +%Y%m%d).txt 2>&1
set -u
Z=/share/home/u25310231/ZXC
echo "=== RQ014 HPC forensics $(date -Is) host=$(hostname) user=$(whoami) ==="

echo "== [1] Slurm accounting: ALL jobs 2025-01-01..now (name/submit/workdir) =="
sacct -u u25310231 -S 2025-01-01 -E now -X \
  --format=JobID%14,JobName%45,Submit%20,State%12,WorkDir%140 2>/dev/null | head -400

echo "== [2] Jobs whose name/workdir mentions wod|e2e|rating|pref (any date) =="
sacct -u u25310231 -S 2025-01-01 -E now -X \
  --format=JobID%14,JobName%45,Submit%20,State%12,WorkDir%140 2>/dev/null \
  | grep -i -E "wod|e2e|rating|rater|pref" | head -100

echo "== [3] bash history: wod/e2e/rating/preference/envelope commands =="
grep -n -i -E "wod|e2e|rating|rater|preference|envelope" ~/.bash_history 2>/dev/null | head -120

echo "== [4] /ZXC top-2-level directory map with mtimes =="
find "$Z" -maxdepth 2 -type d -printf "%TY-%Tm-%Td  %p\n" 2>/dev/null | sort | head -120

echo "== [5] Files named *rating*|*rater*|*preference* anywhere under /ZXC =="
find "$Z" \( -iname "*rating*" -o -iname "*rater*" -o -iname "*preference*" \) \
  -printf "%TY-%Tm-%Td %12s  %p\n" 2>/dev/null | head -120

echo "== [6] Python/scripts reading preference_score OUTSIDE the known RQ010B code dirs =="
grep -r -l --include="*.py" --include="*.sh" --include="*.sbatch" "preference_score" "$Z" 2>/dev/null \
  | grep -v "RQ010B_wod_e2e/code" | head -60

echo "== [7] Result CSV/JSON mentioning both rating and ipv (filenames) =="
find "$Z" \( -iname "*.csv" -o -iname "*.json" \) -iname "*ipv*" \
  -printf "%TY-%Tm-%Td %12s  %p\n" 2>/dev/null | grep -i -E "rating|rater|pref|score" | head -60

echo "== [8] Any WOD-ish artifacts dated BEFORE 2026-06-22 (pre-RQ010B => lost-study candidates) =="
find "$Z" \( -iname "*wod*" -o -iname "*e2e*" \) -newermt "2025-01-01" ! -newermt "2026-06-22" \
  -printf "%TY-%Tm-%Td %12s  %p\n" 2>/dev/null | head -80

echo "== [9] Home dir stray items outside /ZXC (top level only) =="
ls -lat --time-style=+%Y-%m-%d ~ 2>/dev/null | head -40

echo "=== end RQ014 HPC forensics ==="
