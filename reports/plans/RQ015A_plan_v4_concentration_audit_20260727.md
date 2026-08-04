# RQ015A Plan v4（独立复审候选）— IPV 估计尝试状态与候选权重集中度回溯审计

状态：`FINAL_CANDIDATE / AWAITING_INDEPENDENT_REVIEW`｜`formal_g1_eligible=false`｜`execution_authorized=false`  
日期：2026-07-27｜取代：v3（v3 保留为历史，不覆写）

本版吸收 v3 以来的 PI 裁定、真实文件核验、HPC 只读探测、独立盲算复核与实现状态。RQ015A 的对象仍是
`attempt_status` 与连续 `q_eff = K_eff / K` 的集中度画像；它不衡量 IPV 精度，不区分近均匀权重的成因，
也不构成 RQ007 的完整合取条件。

铁律不变：不改估计器；不覆盖冻结产物；不改任何 `decision.md`；不 replay；不真正运行本审计；
不读取、转移、登记任何评分、偏好或人工分数字段；不翻转 `execution_authorized`。

## 0. 绑定来源与 v4 的写作边界

本正文以以下实文件为依据：v3 计划正文、2026-07-26 预执行合同核验、ledger schema v2、
封存暴露登记、WOD 只读取回规格 v1、v3 三路独立复审综合，以及 2026-07-27 的 HPC 探测与盲算复核记录。
本版只写计划正文；T5 入口、T6 授权对象、T7 run spec v2 由并行 agent 交付，本文仅记录其作用与依赖。

## 1. v3 已接受骨架：保留但收紧

- 主量继续是连续 `q_eff`，报告用 policy bins 仅作描述性展示，不进入 episode 摘要、C0 路由、
  `machine_verdict` 或任何判定。
- episode 摘要继续用连续量；C0 路由继续用 `unknown_share`、`unavailable_share`、
  `mean_q_eff_attempted` 与稳定性结论。
- 三恒等式、OnSite 局部序号、`sorted + math.fsum`、L3 `ZERO_SUPPORT` 不填 0、禁止跨产物 pooling
  均保留为必须实现的机器约束。
- `ATTEMPTED`、`NOT_ATTEMPTED`、`UNKNOWN`、q availability、recoverability 与 owning-RQ action
  是不同状态轴，不得互相替代。

## 2. C1-C14 预执行核验修正

| 项 | v4 合同 |
|---|---|
| C1 | sigma01 的 `2,490,992` 是已排除 D0 后的数，不能作 identity_1 基数。冻结为 physical `2,598,536`、measurement `5,197,072`、`NOT_ATTEMPTED` `215,088`。 |
| C2 | split join key 明确为 `case_id == scene_unique_id == case_key`，运行时必须断言未映射为 0。 |
| C3 | `rq009_m3_predictions` 的 15 列无任何 `ipv_error`，从 ledger 移除，仅作 provenance。 |
| C4 | 3× alpha 折叠只属于 predictions；feature matrix 为 `expansion_factor=2`、`collapse_factor=1`。 |
| C5 | v1 的 `anchors_dev_guard=1,778,594` 与 `cases_dev_guard=10,580` 无法复现，删除；feature matrix dev+guard 为 `4,497,368` 行。 |
| C6 | RQ009 fold `{train, guard_tune, calibration, test}` 与 RQ007 split `{development, guard, held_out}` 正交；每个 fold 含约 29% held_out。逐 fold held_out 行：train `751,826`、guard_tune `389,818`、calibration `376,580`、test `381,674`，合计 `1,899,898`。必须先按 `case_id` 白名单过滤，再读 measurement。 |
| C7 | OnSite 主键列是 `case_key`，不是 `case_id`；局部序号必须在 case 内按 `(timestamp_ms, frame_index)` 稳定排序后生成。 |
| C8 | OnSite 的 K 由 provenance 与 vendored estimator 确定为 `7`，grid 为 `legacy7_pi_over_8`。 |
| C9 | `rq014_g2r_anchor_scores` 本地无数据，v1 三字段 key 错；真实逻辑 key 来自 schema 的多字段顺序。 |
| C10 | run 环境不得再写“纯标准库”：parquet 产物要求 pandas/numpy 与 pyarrow 或 fastparquet。 |
| C11 | warm-up 占位 `error=1.0` 与均匀回退不同；`NOT_ATTEMPTED` 优先于 `UNKNOWN`。 |
| C12 | K 不是全局常量；代码库同时存在 K=7 与 K=5，未经 provenance 确认不得外推。 |
| C13 | OnSite 空串不能读作 0；空串且 `local_position < 4` 为 D0，空串且之后为 `UNKNOWN`。 |
| C14 | 本地可审计范围为 3/6：`interhub_sigma01_hw4_timeseries`、`rq009_feature_matrix`、`onsite_dense_timeseries`；三个 WOD/RQ014 产物本地缺失，按覆盖缺口披露。 |

## 3. RQ009 feature matrix：排除 M4_ONLY 通道

feature matrix 的 `M4_ONLY_ego_self_anchor_ipv_current` 与
`M4_ONLY_ego_self_anchor_ipv_error_current` 在本 RQ 中排除。`M4_ONLY` 前缀即禁用标记；
纳入会把 M4 自锚通道混入 M3 语境。`expansion_factor` 固定为 2，只包含
`counterpart_current` 与 `target_future` 两条通道。若未来需要分析 M4 自锚，必须另建独立
`artifact_id`，绝不与这两条通道合并聚合。

## 4. §9 修订：HPC 只读取回边界

v3 §9 的“无 HPC”改为：允许对三个 WOD/RQ014 产物做只读取回。取回是独立操作，必须有自己的
授权对象、SHA 清单、no-overwrite 目标目录与 sanitization receipt。取回失败或数据已不在 HPC 时，
自动退回缩减范围，并按覆盖缺口披露。探测不等于取回；本 RQ 迄今未取回任何数据。

强制顺序：HPC 侧先做列投影与列名 denylist 检查，确认无评分、偏好、人工分数字段后，才允许传输。
本地永不出现这些字段。任何投影 receipt 缺失、SHA 不一致、目标目录已存在、列名越界或写入既有 HPC
产物的迹象，均立即 abort。

## 5. HPC 只读探测结果（2026-07-27）

六个 WOD 目标在 HPC 上全部存在，本地仍未取回。

| 目标 | 行数 | error 列 | 评分列 |
|---|---:|---|---|
| RQ010B full479 audited candidate CSV | 906 | 有：`ego_ipv_error`、`ego_ipv_driven_error` | 有，传输前必须投影移除 |
| phase1 candidate CSV | 1,428 | 无 | 无 |
| phase1 audit CSV | 142 | 无 | 无 |
| phase1b subwindow candidate CSV | 1,428 | 无 | 无 |
| phase1b subwindow audit CSV | 1,428 | 无 | 无 |
| schemeB effective-N candidate CSV | 476 | 无 | 无 |

结论：

- `wod_rq010b_full479_audited` 不是 L4。其 906 行与 schema v2 的 unverified expected rows 完全吻合，
  且含直接 error 列；若经 HPC 侧列投影与 receipt 后取回，可恢复为 `L1_DIRECT`，WOD 一支可从零覆盖
  变为 906 行可审计。
- `wod_phase1_phase1b_10hz_schemeb` 的 `ipv_error_source: absent` 定性确认无误；四个相关表头均无
  error 列，schemeB 目标亦无 error 列。
- RQ014 仍按只读规格处理；若探测确认无 error 字段，不应为本 RQ 增加取回风险面。

## 6. 独立盲算复核（2026-07-27）

独立 agent 对原实现与冻结数字保持盲态，使用 `pyarrow.csv` 列投影流式读与 `pyarrow.dataset`
重算三个本地产物的行数口径，未用 pandas。结果与冻结事实逐位吻合，零分歧。

两项新增证据写入 v4：

- feature matrix 的 `anchor_frame_index` 最小值为 `7`，已在全部 138 parts / `6,397,266` 行上验证；
  schema v2 中“抽样实测”升级为全量结论。
- OnSite 的 `local_position < 4` 行数为 `1,068`（`267 cases × 4`）。错误使用全局
  `frame_index < 4` 只有 `53` 行，两者相差 `1,015` 行。该结果同时冻结 OnSite identity_2 的
  `NOT_ATTEMPTED` 基数，并量化支持“禁止全局 frame_index 规则”。

## 7. schema v2 内部待决项

`rq014_g2r_anchor_scores` 与另两个 `ARTIFACT_NOT_PRESENT_LOCALLY` 条目不一致：缺
`rq007_split_applicable`、`rq007_split_value`、`expansion_factor` / `collapse_factor`；
recoverability 又写为 `L4_UNRECOVERABLE`。identity_3 定义为
`measurement_rows = Σ_over_recoverability n(recoverability)`，同一批行同时有两个竞争的
recoverability 值会造成求和歧义。

当前实现强制取 `ARTIFACT_NOT_PRESENT_LOCALLY`。v4 不修改 schema v2，先把此项列为待决：
建议在 schema v3 中修正字段一致性；若 v4 复审必须先行，则需在复审包中冻结判定优先级
`ARTIFACT_NOT_PRESENT_LOCALLY` > `L4_UNRECOVERABLE`，并把该优先级写入 validator receipt。

## 8. 暴露裁定 §8（2026-07-27）

封存暴露登记新增 §8，作为 append-only supersede，§6 原文未改。§8 正式解除 §6 附加条件的三条：
两阈值须从 dev+guard 重导出、导出规则须先冻结登记 SHA、重导出前不得产出结论画像；同时撤销
`PROVISIONAL_PENDING_DEVGUARD_REDERIVATION` 标记。

解除理由：`4/7`（对应 `ipv_error=0.5`）与 `0.93`（对应 `ipv_error=0.608069099165`）已由科学阈值
降为报告用 policy bins；R6 加上 `test_c0_routing_never_consumes_report_bins` 已在代码层强制 bins
不进入任何判定，该条件已丧失保护对象。

不在解除范围内：§6 判读 A 与记录豁免；§7 措辞精确化，即扫描程序确实解析并聚合过 held_out 逐行字段，
不得回退为“未读取任何 held-out 逐行测量值”；§7 三条治理动作；R1/R2/R3；以及
`execution_authorized=false`。本解除不构成任何执行授权，不替 PI 签署。登记文件 SHA-256 已从
`aabbd0d6...4ab24` 变为 `b4e2bcbf5ec37245a26cd738481983b5370768d0bc61323382efd8a77496864d`；
签署状态仍为 `PENDING_PI_SIGNATURE`。

## 9. 实现现状（仅记录，不授权运行）

当前分支 `rq015a-implementation`，commit `553c6c03`，已实现：

- `scripts/rq015a/rq015a_contracts.py`：唯一算法，已适配 schema v2；
- `scripts/rq015a/rq015a_types.py`：共享类型；
- `scripts/rq015a/build_ledger.py`：T1；
- `scripts/rq015a/validate_only.py`：T2；
- `scripts/rq015a/receipt.py`：T3；
- `scripts/rq015a/factor_analysis.py`：T4；
- `tests/test_rq015a_*.py`：RQ015A 合计 72 passed。

T5 入口、T6 授权对象、T7 run spec v2 由并行 agent 交付中。v4 的职责是记录它们必须闭合的
执行链位置，不预设其最终内容，也不把 T1-T4 的测试通过解释为执行授权。

## 10. 可审计范围与覆盖披露

本地存在且可审计：`interhub_sigma01_hw4_timeseries`、`rq009_feature_matrix`、
`onsite_dense_timeseries`。本地缺失：三个 WOD/RQ014 产物。

若 `wod_rq010b_full479_audited` 完成独立授权下的 HPC 侧投影、receipt、SHA 校验与本地隔离取回，
其 recoverability 可改为 `L1_DIRECT`，带来 906 行 WOD 覆盖。若最终未取回，报告必须在标题级披露
WOD 一支零覆盖，不得表述为“全语料”。

## 11. v4 进入最后独立复审前的 blocker

1. schema v2 的 RQ014 recoverability 冲突未在 schema 文件内修正；本版仅提出优先级建议。
2. WOD 取回仍缺单独签署的授权对象、HPC 侧投影 receipt 与取回 SHA 清单；未取回前 WOD 仍为覆盖缺口。
3. T5/T6/T7 的最终对象尚需纳入 v4 复审包与 checksum manifest；本文不代写、不冻结。
4. 暴露登记 §8 仍为 `PENDING_PI_SIGNATURE`；它解除了前置条件，但不授权本审计运行。

只有当上述对象被 checksum-bound、validate-only 通过、独立复审无 blocker，且 PI 对精确 package/run-spec
另签 scoped single-use authorization 后，才可能进入实际执行。
