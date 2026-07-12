#!/usr/bin/env bash
# RQ014 Phase F pass 3 — HPC: exhume the June-29 IPV-rating pilot chain (all 4 result dirs)
# and the reframed phase3 stats, printing every correlation number ever produced. READ-ONLY.
# Run from Mac:
#   ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc 'bash -s' \
#     < reports/plans/prompts/RQ014_forensics_hpc_pass3_20260710.sh \
#     > reports/studies/RQ014_wod_e2e_rating_recovery/00_forensics/hpc_forensics_pass3_20260710.txt 2>&1
set -u
R=/share/home/u25310231/ZXC/RQ010B_wod_e2e/results
P=/share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis
echo "=== RQ014 HPC pass3 $(date) ==="
for d in \
  "$R/rq010b_wod_e2e_ipv_rating_pilot_20260629" \
  "$R/rq010b_wod_e2e_ipv_rating_pilot_dtfix_20260629T123954" \
  "$R/rq010b_wod_e2e_ipv_rating_pilot_fixed_20260629T124417" \
  "$R/rq010b_wod_e2e_ipv_rating_pilot_routefix_20260629T124941" ; do
  echo "===== $d"
  ls -la "$d" 2>/dev/null | head -20
  for f in "$d"/*.json "$d"/*.md "$d"/*.csv; do
    [ -f "$f" ] || continue
    sz=$(stat -c%s "$f")
    echo "--- $f ($sz bytes)"
    if [ "$sz" -lt 6000 ]; then cat "$f"; else head -c 2500 "$f"; echo; echo "...[truncated]"; fi
  done
done
echo "===== correlation numbers anywhere in the pilot chain + phase3"
grep -r -s -o -E ".{0,80}(spearman|rho|correlation|pearson).{0,120}" "$R"/rq010b_wod_e2e_ipv_rating_pilot* "$P/phase3_preference_test" 2>/dev/null | cut -c1-260 | head -60
echo "===== zxc-rq010b-ipv-rating job log (first run, 2026-06-29)"
head -c 4000 /share/home/u25310231/ZXC/RQ010B_wod_e2e/logs/zxc-rq010b-ipv-rating_1746009.out 2>/dev/null
echo "=== end HPC pass3 ==="
