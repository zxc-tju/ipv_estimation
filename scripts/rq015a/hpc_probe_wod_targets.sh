#!/usr/bin/env bash
# RQ015A — WOD 取回第 1 阶段：**只读探测**。
#
# 在 HPC 登录节点上运行。本脚本只做 ls / head -1 / wc -l / sha256sum，
# 不写入任何路径、不提交任何作业、不复制任何数据。
# 输出一段 JSON，请整段粘回本地会话。
#
#   bash hpc_probe_wod_targets.sh > rq015a_wod_probe.json
#
# 目的：确认 (a) 数据是否还在，(b) 列结构，(c) 是否存在 error 列，
# (d) 是否存在必须在传输前投影掉的 rating/preference 列。

set -uo pipefail

R=/share/home/u25310231/ZXC/RQ010B_wod_e2e
W=/share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ014

TARGETS=(
  "$R/results/rq010b_wod_e2e_multiframe_tracking_ipv_full479_scored_audited_20260630T063600/rq010b_wod_e2e_multiframe_tracking_ipv_full479_audited_candidate_ipv_rating.csv"
  "$R/reframed_pref_analysis/phase1_ipv_build/candidate_ipv.csv"
  "$R/reframed_pref_analysis/phase1_ipv_build/candidate_estimability_audit.csv"
  "$R/reframed_pref_analysis/phase1b_ipv_build_subwindow/candidate_ipv_subwindow.csv"
  "$R/reframed_pref_analysis/phase1b_ipv_build_subwindow/candidate_estimability_subwindow_audit.csv"
  "$R/reframed_pref_analysis/phase_schemeB_effectiveN/schemeB_candidate_ipv.csv"
)

emit_file() {
  local p="$1"
  printf '    {\n      "path": "%s",\n' "$p"
  if [ ! -f "$p" ]; then
    printf '      "exists": false\n    }'
    return
  fi
  local hdr rows sz sha
  hdr=$(head -1 "$p" | tr -d '\r')
  rows=$(( $(wc -l < "$p") - 1 ))
  sz=$(stat -c%s "$p" 2>/dev/null || stat -f%z "$p")
  sha=$(sha256sum "$p" 2>/dev/null | cut -d' ' -f1)
  # 只报告列名是否命中模式，不报告任何单元格取值
  local has_err has_rating
  has_err=$(printf '%s' "$hdr"    | grep -ciE '(^|,)[^,]*err[^,]*(,|$)' || true)
  has_rating=$(printf '%s' "$hdr" | grep -ciE '(^|,)[^,]*(rating|preference|human)[^,]*(,|$)' || true)
  printf '      "exists": true,\n      "data_rows": %s,\n      "bytes": %s,\n' "$rows" "$sz"
  printf '      "sha256": "%s",\n' "$sha"
  printf '      "header": "%s",\n' "$(printf '%s' "$hdr" | sed 's/"/\\"/g')"
  printf '      "has_error_like_column": %s,\n' "$([ "$has_err" -gt 0 ] && echo true || echo false)"
  printf '      "has_rating_like_column": %s\n    }' "$([ "$has_rating" -gt 0 ] && echo true || echo false)"
}

printf '{\n  "probe_kind": "READ_ONLY_NO_TRANSFER",\n'
printf '  "host": "%s",\n  "generated_at": "%s",\n' "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '  "wod_targets": [\n'
for i in "${!TARGETS[@]}"; do
  emit_file "${TARGETS[$i]}"
  [ "$i" -lt $(( ${#TARGETS[@]} - 1 )) ] && printf ',\n' || printf '\n'
done
printf '  ],\n'

# RQ014：只列出 run 目录与其中 jsonl/csv 的存在性与大小，不读表头
printf '  "rq014_work_dirs": [\n'
if [ -d "$W" ]; then
  first=1
  for d in "$W"/RQ014_1_wod_rating_recovery_*; do
    [ -d "$d" ] || continue
    n=$(find "$d" -type f \( -name '*.jsonl' -o -name '*.jsonl.gz' -o -name '*.csv' \) 2>/dev/null | wc -l)
    sz=$(du -sb "$d" 2>/dev/null | cut -f1)
    [ $first -eq 1 ] || printf ',\n'; first=0
    printf '    {"run_dir": "%s", "row_files": %s, "bytes": %s}' "$(basename "$d")" "$n" "${sz:-0}"
  done
  printf '\n'
else
  printf '    {"note": "work_dirs root not found: %s"}\n' "$W"
fi
printf '  ],\n'
printf '  "reminder": "本探测未传输任何数据。下一步须先做列投影 + sanitization receipt，再谈传输。"\n}\n'
