# RQ024 Plan v0（Approved）
状态：`APPROVED / EXECUTING` ｜ user_approval_date=2026-08-24 ｜ `protected_data=NONE` ｜ `Tier2=blocked` ｜ `threshold_retuning=denied`

本文件只定义 sealed Tier1 synthetic 的只读诊断合同，不做修复、不做阈值调整、不进入 Tier2，也不触碰任何受保护数据。

## 1. Context

当前已观察到的事实仅作为前置背景，不作为本计划的新结果：

- `.codex-fleet/nmi-revision-research-lead/work/WP2_recovery/gate_adjudication_v2/VERDICT.json` 已给出 `CONTRACT_FAIL`。
- `risk_coverage.csv` 共 54 行，42 个相邻更严格阈值比较中有 36 个违反 `risk_mae_rad` 单调性。
- `.codex-fleet/nmi-revision-research-lead/work/RQ024_wp2_diagnostic/STATUS.md` 已标记 `completed` 且最终标签为 `MIXED_DIAGNOSTIC`。

这些事实是 prior context，不是本计划要重新证明的结论。

## 2. Research Question

在 sealed Tier1 synthetic pilot 中，哪一类 deterministic contract mismatch 可以解释：为什么更严格的 `q_eff_lt` / `ipv_error_le` gate 没有降低 `risk_mae_rad`，反而触发 Gate A 失败。

## 3. Deterministic Unit

- 分析单位是 sealed Tier1 synthetic pilot 的单个 cell/row。
- 总单位数固定为 288。
- 不允许重采样、重跑、补点、删点或 synthetic regeneration。
- 288 个 cell 构成本次诊断的完整宇宙。

## 4. Variables

Primary outcome:

- `risk_mae_rad`

Gate / contract variables:

- `q_eff_lt`
- `ipv_error_le`
- `coverage`
- `pass_rule`

Diagnostic variables:

- `q_eff`
- `k_eff`
- `ipv_error`
- `abs_err_rad`
- `noise_pos_m`
- `boundary_saturation`
- `on_grid/off_grid`
- `pass/abstain`

Grouping variables:

- comparator family
- threshold level
- subset / stratum
- noise level
- grid status

## 5. Inputs

只读输入如下：

- `.codex-fleet/nmi-revision-research-lead/work/WP2_recovery/exec_v2/tier1_results.csv`
- `.codex-fleet/nmi-revision-research-lead/work/WP2_recovery/exec_v2/risk_coverage.csv`
- `.codex-fleet/nmi-revision-research-lead/work/WP2_recovery/design/ACCEPTANCE.md`
- `.codex-fleet/nmi-revision-research-lead/work/WP2_recovery/design/OUTPUT_SCHEMA.json`
- `.codex-fleet/nmi-revision-research-lead/work/WP2_recovery/exec_v2/RUN_RECEIPT.json`
- `.codex-fleet/nmi-revision-research-lead/work/WP2_recovery/exec_v2/REPORT.md`
- `.codex-fleet/nmi-revision-research-lead/work/WP2_recovery/gate_adjudication_v2/ADJUDICATION.md`
- `.codex-fleet/nmi-revision-research-lead/work/WP2_recovery/gate_adjudication_v2/VERDICT.json`
- `.codex-fleet/nmi-revision-research-lead/work/RQ024_wp2_diagnostic/STATUS.md`

## 6. Approved Scope

允许的工作只包括：

- schema 对齐检查
- 288 行保真检查
- 42 个相邻更严格阈值比较的单调性复核
- 按 comparator family、threshold、noise、on/off-grid、pass/abstain 做描述性分层
- boundary saturation 与 candidate-grid edge effect 的只读诊断
- mismatch taxonomy 汇总

不允许的工作包括：

- Tier2
- threshold retuning
- new synthetic generation
- estimator repair
- 真实数据或 protected data
- 任何其他文件修改，包括 `START_HERE.md`、`STUDIES.md`、paper repo、`decision.md`

## 7. Diagnostic Tasks

1. 核对 `OUTPUT_SCHEMA.json` 与两份 CSV 的表头、字段类型和必需列是否一致。
2. 核对 `tier1_results.csv` 是否严格保留 288 个 cell，没有静默扩行、截行或替换。
3. 复核 `risk_coverage.csv` 的 54 行阈值表，并确认 42 个相邻更严格阈值比较的单调性违例位置。
4. 按 comparator family、noise level、on/off-grid、pass/abstain 归类 mismatch。
5. 汇总 boundary saturation、近边界单元和异常噪声单元的分布，只做描述，不做阈值再选择。

## 8. Outputs

执行层输出保留在：

- `.codex-fleet/nmi-revision-research-lead/work/RQ024_wp2_diagnostic/REPORT.md`
- `.codex-fleet/nmi-revision-research-lead/work/RQ024_wp2_diagnostic/COMMAND_LOG.md`
- `.codex-fleet/nmi-revision-research-lead/work/RQ024_wp2_diagnostic/diagnostic_summary.json`
- `.codex-fleet/nmi-revision-research-lead/work/RQ024_wp2_diagnostic/boundary_summary.json`
- `.codex-fleet/nmi-revision-research-lead/work/RQ024_wp2_diagnostic/metric_association.csv`
- `.codex-fleet/nmi-revision-research-lead/work/RQ024_wp2_diagnostic/stratum_diagnostics.csv`
- `.codex-fleet/nmi-revision-research-lead/work/RQ024_wp2_diagnostic/paired_noise_deltas.csv`
- `.codex-fleet/nmi-revision-research-lead/work/RQ024_wp2_diagnostic/gate_selection_profile.csv`
- `.codex-fleet/nmi-revision-research-lead/work/RQ024_wp2_diagnostic/boundary_argmax_diagnostic.csv`
- `.codex-fleet/nmi-revision-research-lead/work/RQ024_wp2_diagnostic/RUN_RECEIPT.json`
- `.codex-fleet/nmi-revision-research-lead/work/RQ024_wp2_diagnostic/STATISTICS_REVIEW.md`
- `.codex-fleet/nmi-revision-research-lead/work/RQ024_wp2_diagnostic/BOUNDARY_DIAGNOSTIC.md`

若后续需要对外归档，才在 `reports/studies/RQ024_wp2_diagnostic/` 生成对应报告包；本计划本身不创建额外文件。

## 9. One-pass Checks

只做一轮，按以下顺序完成：

1. 解析 `OUTPUT_SCHEMA.json`，确认所有必需字段都能在两份 CSV 中找到。
2. 复核 288 行单位数，确认没有重复、缺行、补行或重抽样痕迹。
3. 复核 54 行 `risk_coverage.csv`，确认 42 个相邻更严格阈值比较的违例计数与位置。
4. 复核 `on_grid/off_grid`、`pass/abstain`、`boundary_saturation` 的分层分布。
5. 输出 mismatch taxonomy，并停止，不进入任何第二轮阈值探索。

## 10. Stop Conditions

出现任一情形即停止：

- row count != 288
- `coverage`、`risk_mae_rad` 或其他 contract 字段缺失
- schema drift 无法由 frozen contract 解释
- 发现 Tier2 输入、真实数据或 synthetic regeneration 痕迹
- 发现需要 retune threshold 才能解释 mismatch
- 发现任何 protected data
- 发现必须改写已接受的 RQ017+ 结论或 paper 文件

## 11. Claim Boundaries

本计划只允许声明：

- sealed Tier1 synthetic pilot 的 contract mismatch 诊断结果
- Gate A 失败的描述性解释
- 分层 mismatch taxonomy
- boundary / grid / noise 相关的只读证据

本计划不允许声明：

- Tier2 ready
- estimator improved
- real-world generalization
- causal effect
- 对生产系统的直接结论
