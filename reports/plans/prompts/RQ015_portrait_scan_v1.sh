#!/usr/bin/env bash
# RQ015 触发画像扫描（只读、rating-free）。剔除 warm-up 占位行 frame_index<4。
# 用法: bash rq015_scan.sh <sigma01_hw4_ipv_timeseries.csv>
set -euo pipefail
CSV="${1:?need csv path}"
sed 's/"[^"]*"/Q/g' "$CSV" | awk -F, '
NR>1 && $22>=4 {
  for(a=0;a<2;a++){ i=(a==0?$26:$33); e=(a==0?$27:$34);
    if(i==""||e=="")continue; e+=0; ai=(i<0?-i:i); T++;
    if(ai<1e-9)Z++; if(ai<1e-6)Z6++; if(e>=0.61)U++; if(e<=0.5)G++; if(e>0.62204)X++;
    if(ai<1e-9&&e>=0.61)ZU++ } }
END{ printf "valid_rows(frame>=4) values=%d\n", T;
 printf "  |IPV|<1e-9      : %.4f%%\n", 100*Z/T;
 printf "  |IPV|<1e-6      : %.4f%%\n", 100*Z6/T;
 printf "  err>=0.61       : %.4f%%\n", 100*U/T;
 printf "  err<=0.50       : %.4f%%\n", 100*G/T;
 printf "  err>0.62204     : %.4f%%  (应为 0)\n", 100*X/T;
 printf "  P(zero|err>=.61): %.4f%%\n", 100*ZU/U;
 printf "  zero 中 err>=.61: %.4f%%\n", 100*ZU/Z }'
