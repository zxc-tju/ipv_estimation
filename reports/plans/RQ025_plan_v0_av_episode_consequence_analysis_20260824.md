# RQ025 Plan v0（WP7 AV-only existing-data consequence analysis）— episode/matching freeze

状态：`PLANNING / EXECUTING`｜`protected_data=NONE`｜`human_collection=denied`｜`causal_claim=denied`
日期：2026-08-24

本计划冻结 WP7 已批准的 episode 与 matching estimands，只做 AV-only existing-data consequence analysis。
主方案是 `any_side_combined + gap_frames=10 + same_scenario + with_replacement + q50`；敏感性方案保持同一
episode 定义，只把 pool 换成 `same_run`，其余维持 `with_replacement + q50`。这里不做 human collection，
不做 NI / equivalence，不做 positive incremental-value claim，也不写 paper claim。

## 1. 研究问题

### RQ025-1

在 accepted AV-only episode onsets 上，冻结后的 same-scenario matched design 能给出什么描述性的
consequence contrast？

### RQ025-2

same-run 敏感性分支是否保持同一方向的描述，而不需要重新打开 episode、matching 或 caliper 合同？

## 2. 统计单位与来源

- 主单位：episode-onset row。
- 嵌套结构：frames nested inside cases and runs。
- 聚类单位：`case_id`。
- Episode 冻结：`policy=any_side_combined`、`side=any`、`gap_frames=10`。
- Primary match freeze：`same_scenario`、`with_replacement`、`q50`。
- Sensitivity match freeze：`same_run`、`with_replacement`、`q50`。
- Source paths：
  - `.codex-fleet/nmi-revision-research-lead/work/WP7_av_existing_analysis/episode_sweep/STATUS.md`
  - `.codex-fleet/nmi-revision-research-lead/work/WP7_av_existing_analysis/episode_sweep/episode_candidates.parquet`
  - `.codex-fleet/nmi-revision-research-lead/work/WP7_av_existing_analysis/episode_sweep/REPORT.md`
  - `.codex-fleet/nmi-revision-research-lead/work/WP7_av_existing_analysis/matching_sweep/STATUS.md`
  - `.codex-fleet/nmi-revision-research-lead/work/WP7_av_existing_analysis/matching_sweep/matching_feasibility.csv`
  - `.codex-fleet/nmi-revision-research-lead/work/WP7_av_existing_analysis/matching_sweep/REPORT.md`
  - `.codex-fleet/nmi-revision-research-lead/work/WP7_av_existing_analysis/schema_v2/EXACT_SCHEMA.csv`
  - `.codex-fleet/nmi-revision-research-lead/work/WP7_av_existing_analysis/schema_v2/EXECUTOR_BRIEF.md`
  - `.codex-fleet/nmi-revision-research-lead/work/WP7_av_existing_analysis/schema/EXECUTION_SPEC.json`
  - `.codex-fleet/nmi-revision-research-lead/work/WP7_av_existing_analysis/schema/ESTIMAND_GATE.md`
  - `.codex-fleet/nmi-revision-research-lead/work/WP7_av_existing_analysis/design_choice_review_v2/DECISION_BRIEF.md`
  - `reports/knowledge/RQ021_contemporaneous_envelope/decision.md`
  - `reports/knowledge/RQ018_abnormal_ipv_degradation/decision.md`
  - `reports/knowledge/RQ019_counterpart_burden/decision.md`

## 3. 纳入与排除

### 纳入

- accepted AV-only rows with `status == OK AND mechanism2_gate_ok == true`
- episode-onset rows emitted by the frozen `episode_sweep`
- pre-outcome covariates already verified in `schema_v2`
- same-scenario primary branch and same-run sensitivity branch

### 排除

- human collection
- human labels
- `rq007_split`
- any protected field
- any episode that crosses case or run
- any redefinition of episode, matching pool, caliper, or replacement rule
- any positive incremental-value, NI, equivalence, or causal claim

## 4. 变量与估计量

- Episode variables：`scenario_id`、`priority_role`、`relative_distance_anchor`、`closing_rate_anchor`、
  `width_90`、`session_id`、`case_id`、`episode_start_frame`、`episode_end_frame`、
  `episode_frame_count`、`n_flagged_frames`、`n_unjudgeable_frames_inside_gap`。
- Outcome family：`official_safety`、`official_efficiency`、`official_comfort`、`official_compliance`、
  `official_coordination`、`official_comprehensive`、`mission_status`、`collision_flag_score0`。
- Primary estimand：same-scenario matched consequence contrast on the frozen `any_side_combined +
  gap_frames=10` episode population, summarized at the episode-onset row and clustered by `case_id`.
- Sensitivity estimand：same-run branch with the same episode freeze, same replacement rule, and the same `q50`
  caliper, again clustered by `case_id`.

## 5. 预期输出

- `.codex-fleet/nmi-revision-research-lead/work/WP7_av_existing_analysis/schema/t7_1_onset_ledger.parquet`
- `.codex-fleet/nmi-revision-research-lead/work/WP7_av_existing_analysis/schema/t7_2_matched_controls.parquet`
- `.codex-fleet/nmi-revision-research-lead/work/WP7_av_existing_analysis/REPORT.md`
- `.codex-fleet/nmi-revision-research-lead/work/WP7_av_existing_analysis/RUN_RECEIPT.json`

## 6. 一轮式验证

1. 核对 episode_sweep 的 A+B+C join 保真，且 frozen episode 定义仍是 `any_side_combined + gap_frames=10`。
2. 核对 matching_sweep 的 same-scenario primary 和 same-run sensitivity 都基于 `with_replacement + q50`。
3. 核对所有输出都保持 onset-row 粒度，frames 只作为嵌套层，不向上改写 unit。
4. 核对秒级长度仍不可辩护时只报 frames，不硬写 seconds。
5. 只做这一轮。若要换 episode、换 pool、换 caliper、换 replacement rule，必须停下并另起新合同。

## 7. Claim boundaries

- 只允许描述 AV-only existing-data 的 consequence contrast。
- 不允许 human collection。
- 不允许因果措辞。
- 不允许 NI / equivalence。
- 不允许 positive incremental-value claim。
- 不允许把 bounded-null 边界升级成 paper-ready 主张。

## 8. Stop gates

- episode merge 跨 case 或跨 run。
- `same_scenario` / `same_run` 之外出现新的 pool。
- `with_replacement`、`q50` 或 `gap_frames=10` 被改写。
- 需要 protected data 或 human data 才能继续。
- 需要重新解释 episode 才能继续。
- 需要改 caliper、改匹配单位或改 outcome family 才能继续。
