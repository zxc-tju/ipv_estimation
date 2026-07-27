# RQ015 — IPV 可估计性契约与估计器数值修复（知识层）

计划（当前，文件名沿用 v1、内部版本为 v1.1）：
`reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md`
状态：`BLOCKED / REQUEST_CHANGES`；`formal_g1_eligible=false`；无计算授权；
未修改任何已冻结 decision。

## 起因

RQ014 后继诊断（2026-07-25）在 sigma=0.1 全量 InterHub 时间序列上实测：
（**已按 v1 更正**，剔除 estimator warm-up 占位行 `frame_index<4` 后，
有效行 7,086,138 个 agent-值）：`|IPV|<1e-9` 有 **41.28%**，其中 **91.27%** 的行
`ipv_error>=0.61`；
全样本 **52.58%** 的 `ipv_error ≥ 0.61`（有效候选数 ≥ 6.6/7，实质无辨识力）。
源码确认存在 `sum(var)==0 → 均匀权重` 的兜底分支，使"无信息"与"中性"在产物中不可区分。

## 触发证据

- `reports/knowledge/RQ014_wod_e2e_rating_recovery/reviews/claude_center_collapse_diagnostic_20260725.md`
- `reports/knowledge/RQ014_wod_e2e_rating_recovery/reviews/m3_center_dispersion_probe.py`
- 本 README §起因 的全量扫描数字（见计划 §2）

## v1.1 双路独立复审（2026-07-26）

冻结 plan SHA-256：`de68bd15eb560a428d3146b4f68a88263eaaf168d3e7880f53989d692a0f8d21`。
两名正式复审者首次均给出 `PASS_WITH_CONDITIONS`；在不读取对方意见的前提下，分别核验
同一批新增科学/数值证据后，最终都改判为 **`BLOCKED`**：

- 统计/科学：`reviews/codex_stats_review_v1p1_20260726.md`；
- 执行/代码/治理：`reviews/codex_execution_review_v1p1_20260726.md`；
- 图文交叉裁决与数据可视化：`reviews/codex_dual_review_synthesis_v1p1_20260726.md`。

当前四类阻断为：权威 B2 schema/status/flags 未冻结；`min_mse_misfit` 的科学选择规则未冻结；
gate-pass M3 覆盖审计没有可执行通过合同；fail-closed 仍可被非有限 IPV grid、无效 ratio/tolerance
和极小正 sigma 探针击穿。冻结 RQ009 M3 test 的 nominal-0.90 支持域中，近零目标
`|y|<1e-6` 有 `520,826/522,219=99.7333%` 的区间包含 0，说明纠缠极强，但不能写成
“每个测不出的帧都判合规”，也不能从旧产物把所有近零值认定为不可估计。

旧接口兼容桥和 RQ015 HPC lane 仍是明确的未来执行 stop gate；当前
`execution_authorized=false`，不得启动 Phase、Formal G1、HPC validate 或 submit。

## 待核查项（不在本 RQ 内裁定）

- RQ007-KC-C2「高集中度指数 ≠ IPV 为 0」与实测 `P(IPV=0 | error≥0.61)=71.65%`（v1 有效行口径） 的张力。
- RQ009 中 M2/M3 相对 M0 的 −42% 区间宽度增益，有多少来自对回落行的窄区间预测
  （零值行 width=0.248 vs 非零行 0.881）。
- ~~时间序列中 `ipv_error > 1−1/√7` 的 9.47% 行意味着混入了 K≥9 的候选网格。~~
  **已被三方独立复算证伪**：这 305,824 行（=38,228 case × 4 帧 × 2 agent）是估计器
  warm-up 占位（`ipv_estimation.py:247-252` 初始化 zeros/ones，仅 `t≥MIN_OBSERVATION=4`
  才写回）；剔除后 `error>0.62204` 的行为 0。已在 v1 中改记为状态 `NOT_ATTEMPTED`(D0)。

在本 RQ 的 Phase A 完成前，不创建 `decision.md`。

## 版本

- v0（2026-07-25）：`reports/plans/RQ015_plan_v0_*.md` — 含已证伪的 K≥9 结论，仅作历史记录。
- **v1.1（2026-07-26，当前；文件名沿用 v1）**：`reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md`
  — 按 PI 决策 D-1~D-4 与三方复审定稿：剔除封存集重算、K_eff 直接采用、
  估计器三步走修复（log 域改写 / 充分统计量+状态码 / σ 重推）、重估移出改交付资格矩阵；
  当前双路复审 `BLOCKED / REQUEST_CHANGES`，不得将其视为已批准计划。
