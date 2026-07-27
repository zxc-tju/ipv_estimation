# RQ015A Plan v3（定稿候选）— IPV 估计尝试状态与候选权重集中度回溯审计

状态：`FINAL_CANDIDATE / AWAITING_INDEPENDENT_REVIEW`｜`formal_g1_eligible=false`｜`execution_authorized=false`
日期：2026-07-26 ｜ 起草：Claude（PI 角色）
取代：v2（保留为历史）。响应 v2 三路复审（R1 5B/3M/2m、R2 2B/4M/2m、R3 6B/4M/2m）。

**本版与前几版的根本差别**：v0–v2 是"承诺将会冻结"的文档；v3 把承诺**变成了实文件**。
五条阻断中有三条是"承诺的文件不存在"，只能通过写出文件来关闭，不可能靠改正文关闭。

## 0. 绑定的实文件（全部纳入 manifest）

| 文件 | 作用 | 状态 |
|---|---|---|
| `reports/plans/RQ015A_ledger_schema_v1.json` | inventory + ledger schema + role crosswalk + 展开/折叠因子 + split 过滤合同 | **已写** |
| `scripts/rq015a/rq015a_contracts.py` | 唯一算法实现：守恒三恒等式、OnSite 局部序号、L1→L2→L3、episode 摘要、bins 稳定性、C0 路由 | **已写** |
| `tests/test_rq015a_contracts.py` | 16 项 fixtures，全部通过 | **已写** |
| `reports/plans/RQ015A_run_spec_v1.json` | immutable run spec：命令、环境、output root、receipt、授权对象 | **已写** |
| 本文件 | 计划正文 | — |

## 1. 关闭 blocker 1 + 4-冲突：下游一律不吃 bins

v2 自相矛盾：§1.1 说"分档不得进入下游判定"，但 §3 的 episode 摘要与 §4 的 C0 路由
都用了 `q_lo/q_hi`。v3 彻底切断：

- **episode 摘要**改用连续权重 `w = 1 − q_eff`（越集中权重越大），与等权版并列，
  只报 definition sensitivity，不称哪种更准；**不出现任何 bin**；
- **C0 路由**改用三个连续量：`unknown_share`、`unavailable_share`
  （= NOT_ATTEMPTED + UNKNOWN 占比）、`mean_q_eff_attempted`；
  路由用的 triage 阈值（0.20 / 0.05 / 0.80）是**运维分流阈值**，与报告用 bins
  完全无关，且另报三组敏感性与 `stable` 标记；
- **PI 的"从 dev+guard 重导 4/7 与 0.93"条件正式解除**：那两个数已从
  "科学阈值"降为"报告用 policy bins"，不进入任何判定，因此不再需要科学导出；
  它们连同九组敏感性一起披露，任一档占比极差 > 10pp 即
  `BINS_WITHHELD_UNSTABLE`、只发连续分布。此项解除记入 §7 的裁定沿革。
- 由 `tests/…::test_c0_routing_never_consumes_report_bins` 强制。

## 2. 关闭 blocker 3：守恒三恒等式（含展开与折叠）

v2 的 `input_rows = Σ terminal_rows` 在展开/折叠存在时不成立。v3 冻结：

```text
identity_1  measurement_rows = physical_rows × expansion / collapse
identity_2  measurement_rows = n(ATTEMPTED) + n(NOT_ATTEMPTED) + n(UNKNOWN)
identity_3  measurement_rows = Σ over recoverability n(recoverability)
```

逐产物因子（schema 中冻结）：sigma01 `E=2`（agent_1/2）；OnSite dense `E=4`
（ego/counterpart × hw4/hw10）；M3 `C=3`（3 个 alpha 行 → 1 anchor）；WOD `E=C=1`。
不可整除即 fail closed。三条恒等式与四类失败路径均有 fixture。

## 3. 关闭 blocker 4：唯一算法（已实现并测试）

- **OnSite 局部序号**：冻结 filtering → 按 `(timestamp_ms, frame_index)` 稳定升序 →
  `local_position = row_number − 1`。fixture 证明朴素 `frame_index − min` 会给出 13
  而正确答案是 3。
- **L1→L2**：分组键 `(case_id, perspective, configuration)`；`< 5` 个 L1 记
  `INSUFFICIENT_SUPPORT`；均值只取 `ATTEMPTED ∧ q_eff≠None`。
- **L2→L3**：只有 `OK ∧ mean≠None` 的 L2 等权参与；无合格 L2 → `ZERO_SUPPORT`，
  `mean=None`，**绝不以 0 参与平均**。
- **逐位确定性**：所有均值改为 `sorted + math.fsum`。这条是 fixture 迫出来的——
  朴素求和在不同输入顺序下给出 `0.3` 与 `0.30000000000000004`，违反"唯一算法"要求。
  现有测试断言聚合与 episode 摘要在 5 种随机置换下**逐位相同**。
- **因素分析**：仅描述性——逐候选因素报告 `q_eff` 的 Spearman 秩相关及
  case-cluster bootstrap（B=2000，seed `20260726`，按 `case_id` 聚类）95% CI；
  **不下因果结论**，不做变量选择。
- **C0 路由**：四态互斥穷尽，优先级
  `INDETERMINATE > OWNER_REANALYSIS_REQUIRED > NO_AUDIT_TRIGGER_DETECTED > NOT_APPLICABLE`，
  每态附 reason code；映射非 1:1 直接判 `INDETERMINATE`。

## 4. 关闭 blocker 2 + 5：manifest 绑定与运行合同

- **manifest**（`RQ015A_plan_v3_checksums_20260726.sha256`）现绑定：本文、schema、
  contracts 实现、fixtures、run spec、暴露登记，全部为**存在的文件**；
- **run spec**（`RQ015A_run_spec_v1.json`）冻结 `operation_id`、精确命令、
  Python 版本与环境约束、input roots、output root、no-overwrite、
  validate-only 先行、single-use receipt 的机器 PASS/FAIL 字段、以及授权对象名；
- receipt 必须包含：三条守恒恒等式逐产物结果、`held_out_parsed_rows = 0` 断言、
  duplicate / unmapped role / K-unknown 计数、bins 稳定性判定、C0 路由稳定性。

## 5. 主量与产物（不变）

主产物是连续 `q_eff = K_eff/K` 的分布（逐产物、逐层、逐分层）；
`q_eff = 1/((1−ipv_error)²·K)`；`ipv_error ≥ 1`（含 warm-up 占位）→ `q_eff=None` 且
`attempt_status=UNKNOWN`；每行记录实际 K，K 不明则 `UNKNOWN`，禁止套用 K=7。
`ipv_error` 只作集中度描述量；**全文禁止 estimability / "测出 IPV" 表述**。

## 6. 数据边界（不变）

split 白名单 `{development, guard}`，先按 case ID 过滤再读 measurement 字段，
断言 `held_out_parsed_rows = 0`；外部产物（OnSite/WOD）标
`RQ007_SPLIT_NOT_APPLICABLE`；**禁止跨产物 pooling**（M3 与 RQ009 current/target 均为
sigma01 派生，pooling 会重复加权同一观测），语料级数字唯一来源是 sigma01 原表。

## 7. 裁定沿革

- 2026-07-26 PI 裁定 held_out 聚合级暴露为**判读 A：豁免**，RQ007 确认路径不受影响；
  措辞已精确化为"程序解析并聚合了逐行字段，未显示/导出/人工检视任何单行值"。
- 同批附加条件"从 dev+guard 重导两个阈值"**于本版正式解除**，原因见 §1：
  两个数已不再是科学阈值。此解除属计划范围内的降级，不改变任何 held_out 边界。

## 8. 已知限制（必须随报告）

不区分近均匀权重的成因（下溢 / 当前网格与模型下平坦 / 模型失配）——属 RQ015B；
不衡量估计准确度，只衡量集中度；完整 estimability（RQ007 合取）不在范围内；
报告用 bins 是政策选择，不是数据中发现的边界。

## 9. 边界与生效条件

不改估计器、不部署闸门、不重训 M3、不覆盖任何冻结产物、不修改任何 owning RQ 的
`decision.md`、不做 replay；纯本地 CPU，无 HPC。
须经独立复审（≥2 路，身份互异且均非起草者）无 blocker 方可进入 Formal G1。
