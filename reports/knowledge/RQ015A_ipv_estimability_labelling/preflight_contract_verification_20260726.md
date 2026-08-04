# RQ015A 预执行合同核验（对真实文件）— 2026-07-26

状态：`PREFLIGHT_STRUCTURAL_ONLY`｜`execution_authorized=false`｜起草：Claude（PI 角色）

## 0. 本次读取的边界（重要）

本次核验**只读取结构性事实**：列名、主键、行数、split/fold 归属、文件存在性、
求解器候选网格常量。**没有解析、聚合或输出任何 `ipv_*` 数值字段**，因此不构成
RQ015A 的执行，也不构成对 held_out 的新暴露。行数按 split 分组的计数使用
`scene_unique_id` 与 `frame_index` 两列完成，未触及任何 measurement 字段。

复现脚本：`/tmp/struct_scan.py`、`/tmp/fold_crosswalk.py`（内容随本文归档于 §7）。

## 1. 结论摘要

对 `RQ015A_ledger_schema_v1.json` 的 13 项修正。其中 **C6 是安全性问题**：
按 `fold` 过滤会解析约 1,899,898 行 held_out 数据。**C3/C4/C5 是科学性问题**：
被指定为主要 M3 产物的文件里根本不存在 `ipv_error`。

| # | schema 原文 | 实测 | 严重度 |
|---|---|---|---|
| C1 | sigma01 identity_1 用 `physical_rows_dev_guard_post_d0 = 2490992` | dev+guard 物理行 **2,598,536**；D0 行 107,544；差为 2,490,992 | 阻断（恒等式不自洽） |
| C2 | split join key 未声明 | `case_id ≡ scene_unique_id ≡ case_key`，0 行未映射 | 阻断（合同缺失） |
| C3 | `rq009_m3_predictions` 经 join 取 `target_ipv_error_future` | **predictions.parquet 无任何 ipv_error 列** | 阻断（产物选错） |
| C4 | M3 `collapse_factor = 3` | 3× 是 predictions 的 alpha；feature matrix 每 anchor×perspective 1 行 | 阻断 |
| C5 | `anchors_dev_guard=1778594`、`cases_dev_guard=10580` | feature matrix dev+guard = **4,497,368** 行 | 阻断（期望值无来源） |
| C6 | 隐含 fold 与 RQ007 split 对齐 | **正交**：每个 fold 都含约 29% held_out | **安全阻断** |
| C7 | OnSite `row_key_fields` 含 `case_id` | 实际列名为 **`case_key`**，无 `case_id` | 阻断 |
| C8 | OnSite `K_source` 可能为 null | 可确定：vendored `agent.py` 网格 K=7 | 修正（收紧） |
| C9 | `rq014_g2r_anchor_scores` 带期望行数 | **本地不存在数据**，仅 schema 与 HPC 回执；且真实主键与 schema 所写不同 | 阻断（范围） |
| C10 | run spec 声明"纯标准库" | feature matrix / OnSite / predictions 均为 **parquet** | 阻断（环境） |
| C11 | warm-up 占位与 D0 规则的优先级未定义 | 占位为 `error = 1.0` 精确值 | 阻断（判定歧义） |
| C12 | K 视作常量 7 | 代码库存在 realtime 网格 **K=5** | 修正（禁止外推） |
| C13 | OnSite 空值单元无规则 | 首行四个 ipv 字段均为**空串** | 阻断（判定歧义） |

## 2. 逐条实测证据

### C1 — sigma01 守恒基数错位

`sigma01_hw4_ipv_timeseries.csv`，3,695,981 数据行，按 RQ007 split 精确划分：

```text
development 1,865,625   guard 732,911   held_out 1,097,445   未映射 0
其中 frame_index < 4（D0）  dev 77,032   guard 30,512   held_out 45,368
dev+guard 物理行 = 2,598,536 ；减 D0 107,544 = 2,490,992
```

schema 的 2,490,992 是**排除 D0 之后**的数。但 `identity_2` 要求
`measurement_rows = n(ATTEMPTED) + n(NOT_ATTEMPTED) + n(UNKNOWN)`；若基数已排除 D0，
`NOT_ATTEMPTED` 恒为 0，该恒等式退化为空断言。

**修正**：`physical_rows = 2,598,536`，`measurement_rows = 5,197,072`，
其中 `NOT_ATTEMPTED = 215,088`（107,544 × 2）。
现有 fixture `test_conservation_expansion_1_to_2` 用 2,490,992 搭配非零 NOT_ATTEMPTED，
**内部不自洽，必须改**。

附带确证：每个 case 恰有 1 行 `frame_index == 0`
（19,258 / 7,628 / 11,342，与 case 数逐一相等），故 sigma01 的 `frame_index`
逐 case 从 0 起且连续，**全局规则 `frame_index < 4` 对 sigma01 有效**
（与 OnSite 相反，见 C7）。

### C2 — split join key

`case_split_assignment.csv` 列为 `case_id,split`，值形如 `ipv_000001`；
sigma01 的 `scene_unique_id` 与 RQ009 feature matrix 的 `case_key` / `scene_unique_id`
取同一值域。全表 3,695,981 行**零未映射**。

**修正**：schema 增加 `split_join_key`，声明三者等价，并要求运行时断言未映射数为 0。

### C3 / C4 / C5 — M3 产物指认错误

`04_calibration/predictions/tier=M3/fold={test,calibration}/predictions.parquet` 列为：

```text
case_key, anchor_frame_index, perspective, source_dataset, fold, tier,
alpha, nominal, q_lo, q_hi, lo_cal, hi_cal, width, abstain, y
```

**没有 `ipv_error`，也没有 `current` / `counterpart` / `target` 三个角色。**
三个角色实际对应 `03_features/matrix/**/features_part_*.parquet` 中的三列：

| ledger role | 实际列 | 备注 |
|---|---|---|
| `counterpart` | `counterpart_ipv_error_current` | |
| `target` | `target_ipv_error_future` | |
| `current`（ego 自锚） | `M4_ONLY_ego_self_anchor_ipv_error_current` | **M4 专用**，列名前缀即禁用标记 |

`predictions.parquet` 行数 = feature matrix 行数 × 3（alpha 80/90/95）：
test 3,811,698 = 1,270,566 × 3；calibration 3,798,846 = 1,266,282 × 3。
故 `collapse_factor = 3` 只属于 predictions，**feature matrix 的折叠因子为 1**。

**修正**：把 `rq009_m3_predictions` 替换为 `rq009_feature_matrix`，
`expansion_factor = 2`（`counterpart` + `target`；`M4_ONLY_*` 默认排除，
若纳入须显式声明并单列，不得混入 M3 语境），`collapse_factor = 1`。
`predictions.parquet` 从台账中移除，仅保留为 provenance 引用。
`anchors_dev_guard=1778594` / `cases_dev_guard=10580` **无法从任何实测口径复现**，
按未经证实的数处理并删除。

### C6 — fold 与 split 正交（安全阻断）

feature matrix 全量 6,397,266 行，fold × RQ007 split 交叉计数：

```text
                development     guard    held_out
train             1,305,680   500,868     751,826
guard_tune          649,848   262,378     389,818
calibration         640,390   249,312     376,580
test                633,924   254,968     381,674
合计              3,229,842 1,267,526   1,899,898
```

**每一个 fold 都包含 held_out 行**，占比稳定在 29% 上下。
RQ009 的 fold 词表 `{train, guard_tune, calibration, test}` 与 RQ007 的
`{development, guard, held_out}` 是两套独立划分。

任何"按 `fold=guard_tune` 近似 RQ007 guard"的写法都会解析 389,818 行 held_out。
**修正**：schema 明写 `fold_is_not_split`，并要求先按 `case_key → split` 白名单过滤，
再读任何 measurement 列；`held_out_parsed_rows` 断言保持为 0。

dev+guard 可用行数实测 **4,497,368**（dev 3,229,842 + guard 1,267,526）。

### C7 — OnSite 主键列名

`onsite_ipv_timeseries.csv` 70,317 数据行，主键列为 `case_key`
（形如 `onsite:beijing:T17:A1:native_case:2344`），**不存在 `case_id` 列**。
首行 `frame_index = 101`，再次确证局部序号规则的必要性
（`frame_index − min` 与全局 `frame_index < 4` 均不可用）。

### C8 — OnSite K 可确定

`onsite_ipv/provenance.json` 记 `legacy_slsqp: true`、`sigma: 0.1`、
`min_observation: 4`，无 `solver_mode` 覆写；vendored `agent.py:57`
`virtual_agent_IPV_range = np.array([-3,-2,-1,0,1,2,3]) * math.pi / 8`。
**K = 7，grid_id `legacy7_pi_over_8`**，由 provenance 冻结而非假设。

### C9 — RQ014 产物本地缺失

仓库内只有 `configs/artifact_schemas/rq014_g2r_anchor_score_row_v1.schema.json`
与 HPC 回执 / slurm 日志；**没有任何 anchor-score 数据文件**。
且真实逻辑主键为
`(segment_id, feature_index/feature_id, sampling_id, temporal_id, horizon_id, tau_tick, candidate_ordinal/candidate_id)`，
schema 所写的 `["segment_key","cell_id","tau_tick"]` 三字段**均不存在**。
该 schema 亦确证行内无任何 error 字段（只有 `candidate_ipv`），
故 `ipv_error_source: absent` / `L4_UNRECOVERABLE` 的定性正确。

**修正**：`rq014_g2r_anchor_scores` 标记 `ARTIFACT_NOT_PRESENT_LOCALLY`，
移出可执行范围，并在报告中作为覆盖缺口披露。计划 §9 禁用 HPC，故本 RQ 内不可补。

### C10 — 环境声明错误

run spec 声明 `required_modules: ["json","math","dataclasses","pathlib","csv"]`
与"纯标准库"。但 feature matrix、OnSite anchors、predictions 均为 **parquet**，
标准库无法读取。沙箱实测：`pandas 2.3.3` 与 `numpy 2.2.6` 存在，
`pyarrow` 安装超时失败，`fastparquet` 安装成功并可读全部目标 parquet。

**修正**：环境改为 `python>=3.9, pandas, numpy, fastparquet`（或 pyarrow 二选一），
并在 receipt 中记录实际 engine 与版本。"纯标准库"表述删除。
OnSite 另有 CSV 孪生文件，可作为 parquet 不可用时的降级路径（须校验两者一致）。

### C11 — warm-up 占位精确值与优先级

`frame_index = 0` 行实测 `ipv_key_agent_1 = 0.0`、`ipv_key_agent_1_error = 1.0`。
占位是 **`error = 1.0` 精确值**，与"均匀回退"的 `1 − 1/√7 = 0.622036` 是两件事：
前者估计器从未运行，后者运行了但权重摊平。

D0 规则（`frame_index < 4`）与退化规则（`error ≥ 1` → UNKNOWN）在同一行同时触发。

**修正**：冻结优先级 —— **`NOT_ATTEMPTED` 优先于 `UNKNOWN`**。
理由：D0 是已知成因，UNKNOWN 是"成因不明"的兜底；已知成因不得被兜底吞掉。
`q_eff` 在两种情形下同为 `None`，仅 `attempt_status` 与 `reason_code` 不同。

### C12 — K 非常量

`src/sociality_estimation/core/agent.py:63-64`：

```python
virtual_agent_IPV_range  = np.array([-3, -2, -1, 0, 1, 2, 3]) * math.pi / 8   # K=7
realtime_agent_IPV_range = np.array([-3, -1, 0, 1, 3]) * math.pi / 8          # K=5
```

`configs/ipv_sigma01_exact.json` 的 `solver_mode: "exact"` 走 virtual 分支 → K=7。
但 K=5 网格在代码库中确实存在，其均匀回退值为 `1 − 1/√5 = 0.552786`，
与 K=7 的 0.622036 不同。**禁止把 K=7 外推到未经 provenance 确认的产物。**

### C13 — OnSite 空值单元

`onsite_ipv_timeseries.csv` 首行四个 IPV 字段（`ipv_ego_hw10`、`ipv_ego_hw10_error`、
`ipv_counterpart_hw10`…）**均为空串**，非 0 亦非 1.0。

**修正**：冻结三分规则 ——
空值 ∧ `local_position < 4` → `NOT_ATTEMPTED`（reason `D0_WARMUP`）；
空值 ∧ `local_position ≥ 4` → `UNKNOWN`（reason `EMPTY_CELL_UNEXPLAINED`）；
非空 → 按数值走常规判定。空串**绝不得**被读作 0。

## 3. 对既有 fixture 的影响

`tests/test_rq015a_contracts.py::test_conservation_expansion_1_to_2` 使用
`physical_rows = 2_490_992` 且 `NOT_ATTEMPTED = 1_000_000`，按 C1 属不自洽构造，
需改为 `physical_rows = 2_598_536`、`measurement_rows = 5_197_072`、
`NOT_ATTEMPTED = 215_088`。

`test_conservation_collapse_3_to_1` 所测的 3→1 折叠对 feature-matrix 产物不再适用；
应保留为通用能力测试，另加 feature matrix 的 `E=2, C=1` 实测用例。

## 4. 尚未核验

- feature matrix 中 `M4_ONLY_ego_self_anchor_ipv_error_current` 是否应完全排除
  （倾向排除，需 PI 确认）；
- WOD 两个产物（`wod_rq010b_full479_audited`、`wod_phase1_phase1b_10hz_schemeb`）
  的实际文件位置与主键尚未定位；
- OnSite `local_position` 的冻结 filtering 具体定义需与 RQ012B 的
  `corrected_clean_mask.csv` 对齐。

## 5. 建议的下一步

1. 按 C1–C13 出 `RQ015A_ledger_schema_v2.json`（不覆写 v1）；
2. 同步修 fixture 与 run spec 环境段；
3. 定位 WOD 两产物后补 §4 第二项；
4. 再进入实现（ledger builder / validator / receipt writer）。

## 6. 与暴露裁定的关系

本次未读取任何 measurement 字段，**不构成新的 held_out 暴露**。
C6 的发现反向说明：若无此次核验即按 fold 过滤执行，将造成约 1,899,898 行
held_out 解析——这正是 2026-07-26 裁定所要防止的事件类别。

## 7. 复现脚本

见 `scripts/rq015a/preflight_structural_scan.py`（由本文同批提交）。
