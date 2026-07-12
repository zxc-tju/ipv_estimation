#!/usr/bin/env bash
# RQ014 FL05 — read-only index of ALL recorded correlation statistics from RQ010B
# intermediate outputs on HPC (reframed_pref_analysis + results). Login-node safe.
# Run from Mac:
#   ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc 'bash -s' \
#     < reports/plans/prompts/RQ014_forensics_hpc_fl05_20260710.sh \
#     > reports/studies/RQ014_wod_e2e_rating_recovery/00_forensics/hpc_fl05_stats_index_20260710.txt 2>&1
set -u
W=/share/home/u25310231/ZXC/RQ010B_wod_e2e
echo "=== RQ014 FL05 $(date) ==="
echo "== file inventory (csv/json/md under reframed_pref_analysis + results) =="
find "$W/reframed_pref_analysis" "$W/results" \( -name "*.csv" -o -name "*.json" -o -name "*.md" \) \
  -printf "%TY-%Tm-%Td %12s  %p\n" 2>/dev/null | sort | head -200
echo "== recorded correlation statistics (with source path) =="
grep -r -s -H -o -E ".{0,60}(spearman|pearson|kendall|rho|correlation)[^,;|]{0,140}" \
  --include="*.csv" --include="*.json" --include="*.md" \
  "$W/reframed_pref_analysis" "$W/results" 2>/dev/null | cut -c1-320 | head -400
echo "== strongly negative candidates (le -0.30 heuristic: -0.3..-0.9 patterns) =="
grep -r -s -H -E "\-0\.[3-9][0-9]*" --include="*.csv" --include="*.json" --include="*.md" \
  "$W/reframed_pref_analysis" "$W/results" 2>/dev/null \
  | grep -i -E "rho|spearman|corr|kendall|pearson" | cut -c1-320 | head -120
echo "== sha256 of every hit-bearing file (for fingerprint registry) =="
grep -r -l -s -E "spearman|rho|correlation" --include="*.csv" --include="*.json" --include="*.md" \
  "$W/reframed_pref_analysis" "$W/results" 2>/dev/null | head -80 | xargs -r sha256sum 2>/dev/null | head -80
echo "=== end FL05 ==="
