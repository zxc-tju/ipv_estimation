# RQ015A Plan v2 — IPV 估计尝试状态与候选权重集中度回溯审计

状态：`PROPOSED / AWAITING_INDEPENDENT_REVIEW`｜`formal_g1_eligible=false`｜`execution_authorized=false`
日期：2026-07-26 ｜ 起草：Claude（PI 角色）
取代：v1（`RQ015A_plan_v1_attempt_status_and_weight_concentration_audit_20260726.md`，保留为历史）
响应：`reviews/rq015a_three_reviewer_synthesis_v1_20260726.md`（三路一致 `BLOCKED`：
R1 4B/4M/2m、R2 2B/4M/3m、R3 5B/5M/2m），逐条关闭 G2–G7。

---

## 0. 本版两项定性改变

**(1) 构念收窄维持不变**（复审确认成立，不得回退）：本 RQ 只描述
"是否尝试过估计"与"候选权重的集中程度"，**不得**使用 estimability / "是否测出 IPV" 表述。

**(2) 放弃把分档作为主产物**（关闭 G2）。三路独立指出：数据分布不会自动给出两个唯一的
语义分界点，"以后冻结一条规则并登记 SHA"不等于现在有科学规则。因此：

| | v1 | **v2** |
|---|---|---|
| 主产物 | 三档标签 `CONCENTRATED/INTERMEDIATE/NEAR_UNIFORM` | **连续归一化集中度 `q_eff` 的分布**（逐产物、逐单位、逐分层） |
| 分档 | 主判据，需科学阈值 | **降级为 secondary policy summary**：边界是**显式的政策选择**，不声称是数据中发现的边界 |
| 阈值导出 | 待"从 dev+guard 重新导出" | **不再需要科学导出**；政策边界与其敏感性一起报告，另设 withheld 失败路径 |

这消解了 G2 的根本困难：描述性审计不需要语义阈值。

## 1. 主量定义（连续，无阈值）

```text
ipv_error  = 1 − √(Σ wᵢ²)              # 已存字段（RQ007 集中度指数）
K_eff      = 1 / (1 − ipv_error)²       # 有效候选数 ∈ (0, K]
q_eff      = K_eff / K                  # 归一化集中度 ∈ (0, 1]，主量
```

- **主产物 = `q_eff` 的完整分布**：逐产物报告 min/分位数(1,5,10,25,50,75,90,95,99)/max、
  直方图（固定 50 等宽 bin，边界写死 `[0,1]`）、以及按 §4 三层单位的分布；
- 每行必须记录**实际 K**；K 不可确定 ⇒ `q_eff` 记 `unknown`，**不得**套用 K=7；
- `ipv_error` 的语义边界照旧如实记录：它是 RQ007 冻结的 *identifiability proxy*，
  本 RQ 只当描述量用。

### 1.1 Secondary policy summary（分箱）与其失败路径

- 政策边界固定为 `q_lo = 4/7`、`q_hi = 0.93`，**明确标注为 policy choice，非数据发现**；
- 必须同时报告敏感性：`q_lo ∈ {0.45, 4/7, 0.65}` × `q_hi ∈ {0.90, 0.93, 0.96}` 的
  九组占比表；
- **withheld 路径**：若九组之间任一产物的三档占比极差 > 10 个百分点，则该产物的分箱摘要
  标记 `BINS_WITHHELD_UNSTABLE`，只发布连续分布，不发布分档结论；
- 分档**不得**用于任何下游判定；C0 路由（§5）只用连续量与暴露度。

## 2. 权威输入清单与 split 合同（关闭 G3）

### 2.1 RQ007 split 的真实标签（v1 用词错误，此处更正）

`case_split_assignment.csv` 的标签是 **`development / guard / held_out`**
（**不是** "sealed"），数量 **19,258 / 7,628 / 11,342**。

- 连接键：`case_id = case_key`；
- **禁止**用 `split != "sealed"` 这类写法排除 held-out（会全量放行）；必须用
  白名单 `split ∈ {development, guard}`；
- **RQ009 的 `train / guard_tune / calibration / test` 是另一套分割**，
  `guard_tune ≠ RQ007 guard`，两者只能通过 `case_id = case_key` 连接，不得混用；
- **读取顺序（fail-closed）**：先按 case ID 白名单过滤，**再**读取任何 measurement 字段；
  验收要求 `held_out parsed rows = 0` 且 `held_out conclusion rows = 0`。

### 2.2 逐产物合同（数字为复审独立核查所得）

| 产物 | 真实规模（RQ007 dev+guard 内） | `NOT_ATTEMPTED`（D0）判据 | 主键 / measurement role | 已冻结注意事项 |
|---|---|---|---|---|
| sigma01 hw4 时间序列 | D0 后 **2,490,992** 宽表 frame rows → **4,981,984** agent-values | 全局 `frame_index < MIN_OBSERVATION(=4)` | 原表**无** `agent_slot` 列；须冻结 `ipv_key_agent_1/2 → role∈{agent_1, agent_2}` 的展开规则 | 展开规则写入 crosswalk 并做 fixture |
| RQ009 M3 预测 | calibration+test 共 2,536,848 anchors / 7,610,544 alpha rows；限 dev+guard 后 **1,778,594 anchors / 10,580 cases** | 继承来源行**不足以**冻结；须按 role 分别声明（current 取 anchor `t*`，target 取 `t*+6`） | prediction 表**无** `ipv_error`；须 join feature matrix 的 `target_ipv_error_future`，键为 **`(case_key, anchor_frame_index, perspective, source_dataset)`**（v1 遗漏 `source_dataset`） | 每 anchor 3 个 alpha 行须按 anchor 去重；join cardinality 必须 1:1，否则 fail closed |
| OnSite dense timeseries | 70,317 physical rows / 267 cases | **局部位置 0–3 = 1,068 rows**；全局 `frame_index<4` 只命中 **53** 行（错） | 一个 dense row 含 **ego/counterpart × hw4/hw10 四通道**；role 未冻结时 D0 分母可为 1,068 / 2,136 / 4,272 | 恢复规则：冻结 filtering 后按 `(timestamp_ms, frame_index)` 排序取局部 `row_number−1`；**不得**用 `frame_index − min(frame_index)`（254/267 cases 首帧非 0，36 cases 不连续） |
| OnSite anchor table | — | current role 有 **252** 个 anchors 落在局部 position 3 且 `error=1` | current / target role 分列 | 套全局帧规则会把这 252 行错标 `ATTEMPTED`；target role 最小局部位置 9，**无 D0** |
| WOD RQ010B full479 audited | **906 = 302 × 3** rows | 逐产物确认 | 该表**保留** error | 可直接回填（L1） |
| WOD Phase1 / Phase1b / 10Hz / SchemeB | — | — | **丢弃 error**，但可由 replay 恢复 | 本 RQ 禁止 replay ⇒ 标 `RECOVERABLE_BY_REPLAY_OUT_OF_SCOPE`，**不得**笼统归为"无法恢复" |
| RQ014 `g2r_anchor_scores` | **37,242** 唯一 candidate-tau rows | — | 无 error | 其中 **18,516** solver-budget + **3,060** undefined-heading 是 scene-wide fail-closed 终态；这 **21,576** 行 ⇒ `attempt_status = UNKNOWN`，**不得**由终态推断逐行状态 |

### 2.3 禁止跨产物 pooling（新增，关闭 G2/G3 交叉项）

RQ009 的 current/target 与 M3 均是 sigma01/hw4 的**复制或 join**。若把多个产物直接
pooling 计算分布或阈值，会对**同一原始 observation 重复加权**。因此：

- `q_eff` 分布**逐产物分别报告**，禁止跨产物合并；
- 若需一个"语料级"数字，唯一合法来源是 **sigma01 hw4 原表**（其余产物皆为其派生）；
- 派生产物的分布只用于回答"该产物自身的暴露度"。

### 2.4 Ledger schema（先冻结再打标）

`concentration_ledger` 冻结项：long 格式；主键 `(artifact_id, product_row_key, measurement_role)`；
每字段 role 标注（`observed` / `derived` / `verdict` / `provenance`）；
`measurement-role crosswalk`（各产物字段 → 统一 role 名）；`unknown` 的显式取值；
**守恒规则**：`input_rows = Σ terminal_rows`，逐产物核对，不符即 fail closed；
duplicate key / unmapped role / K unknown / 守恒不符 → 全部 fail closed 并写 reason code。
schema 与 crosswalk 须附 fixture（每产物至少 1 个小样例，输入输出双向校验）。

## 3. 三层分析单位与 SAP（关闭 G4）

```text
L1 measurement      : (artifact_id, product_row_key, measurement_role)
L2 case-persp-config: (case_id, perspective, configuration)   # configuration = hw / window / rate 等
L3 case             : case_id
```

- **分母逐层显式**：每个报告数字必须标注其 L1/L2/L3 分母及构成；
- **minimum support**：L2 聚合要求该单元 ≥ 5 个 L1 measurement，否则该单元记
  `INSUFFICIENT_SUPPORT` 且不进入 L3 平均；
- **zero support**：L3 单元若所有 L2 均 insufficient，记 `ZERO_SUPPORT`，单独报数，
  不以 0 参与任何平均；
- **unknown 传播**：任一层含 `unknown` 时不得静默丢弃，须单独一列计数；
- **聚类不确定度**：L3 层比例的不确定度用 case-cluster bootstrap（B=2000，seed `20260726`，
  percentile 95% CI，按 `case_id` 聚类）；
- **episode 摘要**：只报告"仅 `q_eff ≤ q_lo` 帧"与"以 `1−q_eff` 为权重"两种定义的
  **definition sensitivity**，**不得**声称哪一种更准确（RQ007-KC-C3 的差异
  0.26 rad / 22% 符号翻转 / 加权后 7% 只作背景引用）；
- 每个公式须有唯一 fixture 结果（输入 → 期望输出），复审可逐条重算。

## 4. C0 确定性路由（关闭 G5）

v1 的"低资格风险"没有任务级证据支撑，作废。改为**确定性路由**，四个终态互斥穷尽：

| 终态 | 必要充分条件 |
|---|---|
| `NOT_APPLICABLE` | 该 owning RQ 的分析不使用任何 IPV/`ipv_error` 派生量 |
| `NO_AUDIT_TRIGGER_DETECTED` | 使用了，且其分析行中 `q_eff ≥ q_hi` 与 `NOT_ATTEMPTED` 与 `unknown` 的合计占比 **< 5%**（阈值为 policy，需同报敏感性） |
| `OWNER_REANALYSIS_REQUIRED` | 使用了，且上述合计占比 **≥ 5%** |
| `INDETERMINATE_UNKNOWN_PROVENANCE` | `unknown` 占比 ≥ 20%，或无法建立该 RQ 分析行与 ledger 的 1:1 映射 |

- 优先级：`INDETERMINATE` > `OWNER_REANALYSIS_REQUIRED` > `NO_AUDIT_TRIGGER_DETECTED` > `NOT_APPLICABLE`；
- 每个终态附 reason code 与 owner action（"由 owning RQ 自行发起 amendment"）；
- **本 RQ 不自动修改任何 owning RQ 的 decision**；
- 覆盖 RQ003 / RQ009 / RQ010B / RQ011B / RQ012B / RQ014。

## 5. held_out 治理的精确记录（关闭 G6）

PI 已于 2026-07-26 裁定豁免（判读 A），该裁定保留。但措辞须精确化：

- 精确事实：**扫描程序解析并聚合了 held_out 的逐行字段**；
  **未显示、未导出、未人工检视任何 held_out 单行数值**；未在 held_out 上做估计/拟合/检验。
  v1 与裁定书中"未读取逐行值"的表述据此更正；
- RQ007 的权威入口（`decision.md` / knowledge README）须加 **append-only pointer**
  指向本暴露登记，不得声称 held_out 仍 "untouched"；
- **外部产物（OnSite / WOD）不属于 RQ007 split**，须显式标注
  `RQ007_SPLIT_NOT_APPLICABLE`，不得套用 dev/guard/held_out 过滤；
- 本 RQ 结论一律只用 `split ∈ {development, guard}` 的 InterHub 系产物 +
  标注了 `RQ007_SPLIT_NOT_APPLICABLE` 的外部产物。

## 6. 运行合同与验收链（关闭 G7）

- `operation_id = rq015a_concentration_audit`；
- **immutable run spec**：绑定 plan SHA、SAP SHA、inventory SHA、schema/crosswalk SHA、
  code SHA、env manifest SHA、exact command、output root；
- **validate-only 先行**，再单次执行；
- **no-overwrite**：output root 已存在即中止；
- **single-use receipt**：含机器可判的 `PASS/FAIL`、各产物行守恒核对、
  `held_out parsed rows = 0` 断言、duplicate/unmapped/K-unknown 计数；
- 任一条件不符（输入哈希不匹配、越界读取、覆盖、split leakage、守恒失败）→ **中止**；
- 本 RQ 纯本地 CPU，无 HPC，不受 run-spec `rq_id` 常量限制影响。

## 7. 交付物

1. 冻结的 inventory + ledger schema + measurement-role crosswalk + fixtures；
2. 逐产物 `q_eff` **连续分布**（主产物）+ 三层单位分布 + 分层（source/geometry/window）；
3. secondary policy 分箱摘要 + 九组敏感性（含 `BINS_WITHHELD_UNSTABLE` 判定）；
4. D0 计数逐产物核对表（含 OnSite 局部序号、M3 anchor 去重、WOD/RQ014 unknown 清单）；
5. 集中度相关因素探索性分析（不下因果结论）；
6. episode 摘要 definition sensitivity；
7. C0 确定性路由表；
8. held_out 治理精确记录 + RQ007 pointer；
9. run receipt + 有界报告 + 独立复审记录。

## 8. 验收标准

全文无 estimability / "测出 IPV" 表述；主产物是连续 `q_eff` 分布而非分档；
分箱明示为 policy 且附敏感性与 withheld 路径；split 用白名单且
`held_out parsed rows = 0`；M3 join 键含 `source_dataset` 且 1:1；
OnSite D0 = 1,068（局部规则）而非 53（全局规则）；RQ014 的 21,576 行为 `UNKNOWN`；
WOD 可 replay 产物标 `RECOVERABLE_BY_REPLAY_OUT_OF_SCOPE`；
跨产物无 pooling；三层分母、minimum/zero support、unknown、聚类 CI 全部显式；
C0 四终态互斥穷尽且附 reason code；行守恒逐产物通过；receipt 机器判定 PASS。

## 9. 已知限制（必须随报告）

- 不区分近均匀权重的成因（下溢 / 当前网格与模型下平坦 / 模型失配）——属 RQ015B；
- 不衡量估计准确度，只衡量集中度；集中度高 ≠ 估计正确；
- 完整 estimability（RQ007 合取）不在本 RQ 范围内；
- 分箱边界是政策选择，不是数据中发现的语义边界。

## 10. 边界与生效条件

不改估计器、不部署闸门、不重训 M3、不覆盖任何冻结产物、不修改任何 owning RQ 的
`decision.md`、不做 replay；不得以任何下游关联（含评分）作为任何选择准则。
须经独立复审（≥2 路，身份互异且均非起草者）无 blocker 方可进入 Formal G1。
