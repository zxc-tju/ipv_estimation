# STATUS — track I（rq015i-underflow-regimes）

state: DONE
released_by: commander 于 2026-08-03T02:36:44Z 放行（结论早已裁定，此前仅未翻牌）
updated_at: 2026-08-03T02:36:44Z
phase: 结项轮完成，已回复 commander 五条要求，等待放行
leader_pid: 9917

summary:
  交付 `board/reports/I1_underflow_regimes.md`（生成时间 2026-08-01T13:17:47Z）。
  I1 三轮（初版 → 分母订正 → 结项修改），leader 三轮独立复算。

  核心结果：
  - 区间① 全语料**实测**：严格 929,488（6.42%）/ 容差 966,227（6.68%）/ 14,473,982。
    明确结论：**能**用精确签名普查，推荐容差口径；36,739 行差额为纯浮点尾差。
  - 区间② 主口径按 signature 分层：**N 层 Mac 29.00% / HPC 28.60%**，
    U 层 0.83%/0.00%，Z 层 1.50%/1.17%。混合分母值已降为附注。
  - `nzero==6`（下溢制造的事实 hard argmax）：N 层两版**各 7 行、1.40%、全为 waymo**，
    两环境完全一致；此前对外的 Mac/HPC 分歧全部来自 U 层伪影。
  - 可识别性：**识别不了**完整集合（高特异度规则灵敏度仅 31–33%）。

  本轮推翻了 kickoff 的 3 处基准 + 1 处口径：q_eff 签名写错、区间③ 实为 7/9 行、
  partial_underflow 实为 ②∪③、zero_postwarm_scope 恒等于 signature∈{U,Z}
  （恰把问题所在的 N 层排除在分母外，commander 已采纳并更正对 PI 的口径）。

  交付纪律：progress.log 曾出现前瞻时间戳（最大 +14 分钟），已按 mtime 重锚修正，
  原件留存 progress.log.prefix_uncorrected.bak，报告方法节已声明。

next:
  等待 commander 放行。待裁决三项：
  (1) HPC 侧 N 层 ht_weight 为占位符，是否补权重以给出 HPC 设计基估计；
  (2) main_workflow.log 是否由 leader 补记（H 轨并发，本轮未写）；
  (3) 附注串被机械套用到区间③ 等无关行，纯排版冗余，是否要求清理。
  未自行转 DONE，未 commit。
