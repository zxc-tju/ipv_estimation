# RQ015A Plan v1 — IPV 估计尝试状态与候选权重集中度回溯审计

状态：`PROPOSED / AWAITING_INDEPENDENT_REVIEW`｜`formal_g1_eligible=false`｜`execution_authorized=false`
日期：2026-07-26 ｜ 起草：Claude（PI 角色）
取代：`RQ015A_plan_v0_ipv_estimability_labelling_20260726.md`（v0 保留为历史，**其
"可估计性打标"命名与 `ESTIMABLE/NOT_ESTIMABLE` 标签已判定为构念越界，不得再引用**）

响应复审（三路独立 + 统一反证质询，结论一致 `BLOCKED / REQUEST_CHANGES`）：
`reports/knowledge/RQ015A_ipv_estimability_labelling/reviews/rq015a_three_reviewer_synthesis_v0_20260726.md`
及三份 reviewer 意见（R1 5B/4M/1m、R2 4B/3M/4m、R3 6B/5M/2m）。

---

## 0. 本版最重要的改变：**构念收窄与更名**

复审的中心判定成立：`K_eff` 只能描述**候选权重的集中程度**，不能推出"是否真正测出
IPV"。RQ007 的完整 estimability 是一个**合取**（交互机会 + 连续低集中度 + 机械对照 +
case health），而本 RQ 只有其中一项。因此：

| v0 | v1 |
|---|---|
| RQ 名："可估计性回溯打标与画像" | **"IPV 估计尝试状态与候选权重集中度回溯审计"** |
| 标签 `ESTIMABLE / WEAK / NOT_ESTIMABLE` | **`CONCENTRATED / INTERMEDIATE / NEAR_UNIFORM`** |
| 断言"是否测出 IPV" | 只断言"权重集中还是近均匀"，以及"是否尝试过估计" |
| "帧级研究直接受损、case 级受损较轻" | **"暴露于低集中度状态、存在资格风险"**（见 §6） |

**全文禁止**把本 RQ 的任何输出称为 estimability / 可估计性；亦不得声称"未测出 IPV"。
若将来要恢复 estimability 表述，必须重建 RQ007 的完整合取，工作量显著扩大，
届时另立 RQ，不在本 RQ 内。

## 1. 研究问题（收窄后）

**RQ015A**：在现有 IPV 产物中，逐行回答两个**可直接观测**的问题：

1. **是否尝试过估计**（`ATTEMPTED` / `NOT_ATTEMPTED`）；
2. 若尝试过，**候选权重的集中程度**落在哪一档（`CONCENTRATED` / `INTERMEDIATE` /
   `NEAR_UNIFORM`）。

并给出帧级与 case 级两套画像、跨产物执行合同、以及下游**资格风险**矩阵。
本 RQ 是描述性审计，不改数值、不改估计器、不部署闸门、不重估任何既有结论。

## 2. 集中度分档（修正阈值数学）

### 2.1 归一化集中度

复审指出 `K_eff ≤ 4` 与 `K_eff ≥ 0.93·K` 在 `K ≤ 4` 时**重叠**（K=4 时区间
`[3.72, 4]` 同时满足两侧），且 K=7 的精确 error 边界是 **0.608069099165**，
不是"0.61"。改用**归一化集中度**，对任意 K 互斥且可比：

```text
ipv_error = 1 − √(Σ wᵢ²)            # 已存字段
K_eff     = 1 / (1 − ipv_error)²     # 有效候选数，∈ (0, K]
c         = K_eff / K                # 归一化集中度 ∈ (0, 1]

CONCENTRATED   : c ≤ c_lo
INTERMEDIATE   : c_lo < c < c_hi
NEAR_UNIFORM   : c ≥ c_hi
NOT_ATTEMPTED  : 见 §3，先分流，不参与分档

c_lo, c_hi : **占位值 4/7 (=0.571428571428…) 与 0.93**，状态
             `PROVISIONAL_PENDING_DEVGUARD_REDERIVATION`（见 §2.3）
```

- 三档**按构造互斥且穷尽**（对一切 `K ≥ 1`）；
- K=7 时与 v0 数值等价：`c ≤ 4/7 ⇔ K_eff ≤ 4`；`c ≥ 0.93 ⇔ K_eff ≥ 6.51`；
- 边界只以 `c` 表述。**禁止**引用"error ≥ 0.61"这类近似值；如需 error 形式，
  必须写精确式 `1 − 1/√(0.93·K)`（K=7 时 = 0.608069099165，K=5 时 = 0.536261104240，
  K=9 时 = 0.654349435090）；
- 每行必须记录**实际 K**；K 不可确定则该行标 `unknown`，不得套用 K=7。

### 2.2 依据与边界（如实记录）

`ipv_error` 即 RQ007 已冻结的"估计器集中度指数"，RQ007 `decision.md` 称其为
*identifiability proxy*，且 C1–C3 为 dev/guard 边界结论。本 RQ 只把它当作
**集中度描述量**使用，不作任何 estimability 推断。

### 2.3 阈值须从 dev+guard 重新导出（PI 裁定 2026-07-26 的附加条件）

封存集历史暴露经 PI 裁定为**判读 A：不构成实质破坏，记录豁免**
（`reports/knowledge/RQ015A_ipv_estimability_labelling/sealed_exposure_disclosure_20260726.md` §6，
RQ007 知识层有同步副本）。同批批准的附加条件为：

- 现有 `4/7` 与 `0.93` 是**看过含 sealed 的全语料聚合后写下的**，故降级为占位值；
- **Phase A 的第一项计算任务**：在 dev(19,258)+guard(7,628)、
  `frame_index ≥ MIN_OBSERVATION`、且已按 §3 逐产物计数单位去重后的行上，
  重新导出 `c_lo` 与 `c_hi`；导出规则在计算前冻结并登记 SHA-256；
- **重导出前不得用占位值产出任何结论画像**；
- 重导出值与占位值并列报告，以示可审计。

## 3. 跨产物执行合同（复审 blocker 3，逐产物冻结）

v0 笼统使用 `frame_index < 4` 判 `NOT_ATTEMPTED`，跨产物不成立。逐产物冻结如下：

| 产物 | `NOT_ATTEMPTED` 判据 | 计数单位 / 主键 | 已知缺失 |
|---|---|---|---|
| sigma01 时间序列（InterHub） | 全局 `frame_index < MIN_OBSERVATION(=4)` | `(scene_unique_id, frame_index, agent_slot)`；单位 = **agent-值** | — |
| RQ009 M3 预测 | 继承来源行 | **`(case_key, anchor_frame_index, perspective)`；每 anchor 有 3 个 `nominal/alpha` 行（80/90/95），必须按 anchor 去重，否则三倍计数** | — |
| OnSite（RQ012B） | **按估计器局部序号**（该产物的估计起始偏移），**不得**套用全局 `frame_index<4` | 逐产物冻结 | 局部序号定义须在 Phase A 前确认 |
| WOD（RQ010B / RQ014） | 逐产物确认 | 逐产物冻结 | **部分产物未保留 `ipv_error`** ⇒ 该部分只能标 `unknown`，不得推断分档 |

### 3.1 Ledger 冻结项（先冻结 schema，再打标）

`concentration_ledger` 必须先冻结：**主键、计数单位、每字段角色（观测 / 派生 / 判定）、
行守恒规则**（输入行数 = 各终态行数之和，逐产物核对），以及 `unknown` 的显式取值。
未冻结前不得开始打标。

## 4. 封存集历史暴露（复审 blocker 2；**PI 已于 2026-07-26 裁定**）

**事实**：v0 立项证据中的全语料聚合画像（7,086,138 agent-值：零值 41.2794%、
`err≥0.61` 52.5810%、`err≤0.50` 24.1688% 等）**包含 RQ007 封存集的 11,342 个 case**，
且 v0 的阈值（`K_eff≤4`、`0.93·K`）是在看过这些聚合分布之后写下的。

因此**不能**再声称"sealed 从未参与阈值选择"。本版处置：

1. **如实登记历史暴露**：暴露形式为**语料级聚合统计**（分布分位与比例），
   未读取任何 held-out 的**逐行测量值**、未做任何 held-out 上的效应估计；
2. 暴露记录写入本 RQ 与 RQ007 两侧知识层；
3. **PI 裁定（2026-07-26）= 判读 A**：暴露不构成对 RQ007 封存目的的实质破坏，
   记录豁免；**RQ007 的 held-out 确认路径不受影响**。裁定全文见
   `reports/knowledge/RQ015A_ipv_estimability_labelling/sealed_exposure_disclosure_20260726.md` §6
   （RQ007 知识层有同步副本）；
4. **同批批准的附加条件**：两条集中度边界必须从 dev+guard 重新导出（见 §2.3），
   重导出前不得用占位值产出结论画像；
5. 本 RQ 的最终画像与任何阈值一律**只在 dev(19,258)+guard(7,628) 上计算与冻结**，
   sealed 不进入结论。

## 5. Phase A — 打标与画像

### 5.1 四层可行性（禁止笼统声称"无需重算"）

| 层 | 判定 | 处置 |
|---|---|---|
| L1 可直接回填 | 已存 `ipv_error` 且 K 已知 | 直接判 `NOT_ATTEMPTED` + 三档 |
| L2 可由 provenance 重建 | K / 网格 ID 可由运行配置确定 | 重建后回填并记录依据 |
| L3 需受控重算 | 需 `min_mse` / `mse_per_candidate` 等未保存量 | **移交 RQ015B**，本 RQ 标 `pending_RQ015B` |
| L4 无法恢复 | 无逐行 `ipv_error`（部分 WOD 产物） | 标 `unknown` 并列明后果 |

### 5.2 双口径画像

帧级与 case 级并列给出（数字待按 §3 单位契约 + 剔除 sealed 后重算；
v0 的全语料先验值仅供对照，不得进入结论）。case 级须报告：
每 case 的 `CONCENTRATED` 计数分布、完全无 `CONCENTRATED` 的 case 比例、
以及 case 内比例的分布。

### 5.3 集中度的相关因素（探索性）

先验证据显示集中度**不**主要由接近程度决定（<5 m 档比例明显更高但仅占 3.8% 的帧）。
候选因素：观测窗长度、轨迹曲率/减速幅度、路径类型、数据源、采样率。
**不下因果结论**；其价值是为 RQ015B 的机制拆分提供先验。

### 5.4 episode 摘要规则

沿用并扩展 RQ007-KC-C3 的加权（全帧 vs 交互期摘要均值相差约 0.26 rad、
约 22% case 严格符号翻转、加权后降至约 7%），给出
"仅 `CONCENTRATED` 帧"与"集中度加权"两种摘要的对照。

## 6. Phase C0 — 下游**资格风险**矩阵（措辞收窄，复审 blocker 5）

**不得**在未做任务级重估前宣布"帧级研究直接受损"或"case 级受损较轻"。
本 RQ 只能给出**暴露度**与**资格风险**：

对 RQ003 / RQ009 / RQ010B / RQ011B / RQ012B / RQ014 逐项给出：

- 真实分析单位与聚合结构；
- **暴露度**：其分析行中落入 `NEAR_UNIFORM` / `NOT_ATTEMPTED` / `unknown` 的比例；
- 按集中度筛选是否会改变估计量与暴露定义；
- **选择偏倚风险**（集中度可能与结果变量相关）与所需额外控制；
- 结论限定为三档之一：`低资格风险 / 存在资格风险，需任务级重估 / 不适用`。

是否重开由各 RQ 自行按既有流程发起 amendment。

## 7. 交付物与验收

1. 冻结的 `concentration_ledger` schema（主键/单位/字段角色/行守恒）+ 四层分类表；
2. 逐产物执行合同（含 OnSite 局部序号、M3 anchor 去重、WOD 缺失清单）；
3. 封存集历史暴露登记 + PI/RQ007 裁定记录；
4. 剔除 sealed 的双口径画像 + 分层；
5. 集中度相关因素探索性分析；
6. episode 摘要规则对照；
7. 下游资格风险矩阵；
8. 有界报告 + 独立复审记录。

**验收**：全文无 estimability / "测出 IPV" 表述；三档按 `c` 定义且互斥；
每行记录实际 K；M3 anchor 未三倍计数；OnSite 用局部序号；行守恒逐产物核对通过；
sealed 未进入任何结论且暴露已登记裁定；矩阵结论只用三档措辞。

## 8. 已知限制（必须随报告）

- 本 RQ **无法**区分近均匀权重的成因（数值下溢 / 当前网格与模型下平坦 / 模型失配），
  相关统计量未保存；该拆分属 **RQ015B**；
- 本 RQ **不**衡量估计准确度，只衡量集中度；集中度高不等于估计正确；
- 完整 estimability（RQ007 合取）不在本 RQ 范围内。

## 9. 边界与禁止事项

不改估计器、不部署闸门、不重训 M3、不覆盖任何冻结产物、不修改任何已接受
`decision.md`；不得以任何下游关联（含评分）作为阈值选择准则。
RQ007-KC-C2 与实测 `P(IPV=0 | c≥0.93)` 的张力登记为待核查项，交回 RQ007 流程。

## 10. 资源与生效条件

纯打标与统计，本地 CPU 分钟级，无 HPC 需求。
须经独立复审（≥2 路，身份互异且均非起草者）无 blocker 方可进入 Formal G1。
