# RQ015B Plan v0 — IPV 估计器数值修复与 verifier 非信息性估计闸


**术语更新（2026-07-26，依 RQ015A v1 三路复审）**：全文原“可估计性闸”一律改称**非信息性估计闸**（non-informative-estimate gate）；分层取值改用 RQ015A v1 的集中度分档 `CONCENTRATED / INTERMEDIATE / NEAR_UNIFORM`（归一化 `c = K_eff/K`）；阈值**不得**引用近似值 `error >= 0.61`，须用 `c` 或精确式 `1 - 1/sqrt(0.93*K)`（K=7 时 = 0.608069099165）。本 RQ **不得**声称“IPV 未测出”或使用 estimability 表述；可辩护的说法是：**权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息 ⇒ 弃权**。完整 estimability 是 RQ007 的合接，不在本 RQ 范围内。

状态：`PROPOSED / AWAITING_INDEPENDENT_DUAL_REVIEW`｜`execution_authorized=false`
日期：2026-07-26 ｜ 起草：Claude（PI 角色）
来源：PI 决策 2026-07-26 将原 RQ015 拆分为 A/B（依据 `RQ015_plan_v1p2_amendment_20260726.md` §A6）。

- **RQ015A**：估计尝试状态与候选权重集中度回溯审计（当前版本 `RQ015A_plan_v1_attempt_status_and_weight_concentration_audit_20260726.md`；v0 的 estimability 命名已因构念越界作废）。
- **RQ015B（本文）**：估计器修复、状态契约、生产兼容层、覆盖审计、弃权闸部署决策。

继承：原合并计划的 §4（Phase B）与 §4.6（覆盖审计），以及 v1.2 amendment 的
A1（字段表）、A2（阈值规则与 D2 命名）、A3（覆盖审计规格）、A5（fail-closed）。
原合并计划保留为历史记录，不再作为执行依据。

**依赖**：RQ015B 的 L3 层重算服务 RQ015A 的四层分类；RQ015A 的
集中度相关因素分析为本 RQ 的"可修 vs 固有"判定提供先验。两者可并行起草，
但本 RQ 的 B3 与覆盖审计**不得**在 RQ015A 的 fold 归属定位完成前冻结阈值。

---

## 1. 研究问题

**RQ015B**：把"无信息被静默写成中性 0"的数值缺陷修掉；把零值成因拆成
D1/D2/D3/D4；并在统计有效性得到审计的前提下，给 verifier 装上**非信息性估计闸**。

## 2. 待修缺陷（已确立）

`src/sociality_estimation/core/agent.py::cal_traj_reliability` 在概率域连乘后
`if sum(var): w=var/sum(var) else: w=ones(K)/K`。两条不同性质的路径产生同一个 `IPV=0`：
数值下溢与真实平坦，且现有产物**无法区分**。

**关键数学事实**：`var_i = [∏_k φ(d_ik)]^{1/n}` 取对数后
`log var_i = −log(σ√2π) − MSE_i/(2σ²)`，首项归一化时约掉，故现行计算等价于
`w = softmax(−MSE_i/(2σ²))`。连乘是绕路，也正是下溢之处。

**下溢临界 RMS**（乘积在开 1/n 次方**之前**下溢；两个边界分别命名）：

| n | 进入 subnormal | 舍入为精确 0（触发兜底） |
|---:|---:|---:|
| 5（hw=4） | 1.6915 m | 1.7336 m |
| 11（hw=10） | 1.1470 m | 1.1752 m |

## 3. B1 — log 域改写（σ 不变，数学等价）

```python
mse  = np.mean(rel_dis**2, axis=1)
logw = -mse / (2 * sigma**2); logw -= logw.max()
w    = np.exp(logw); w /= w.sum()          # 分母 ≥ 1，兜底分支不可达
```

**平价门（硬性、不可豁免）**：未下溢行上新旧逐位一致 `max_abs_diff ≤ 1e-12`。
依据 `reports/knowledge/_analysis/ipv_estimator_divergence_investigation.md`
（一次纯向量化重构曾使 sigma01 数值偏移 0.281）。

## 4. B2 — 正交结果契约与生产兼容层

### 4.1 唯一权威字段表（与 `ReliabilityResult` 逐字段一致）

```text
weights[K], ipv, ipv_error, k_eff, status, flags, reason_code,
K, grid_id, min_mse, loglike_gap, mse_per_candidate[K],
step_sq_residuals|None, schema_version, estimator_version, sufficiency_scope
```

- **`STATUS_SET`（互斥，恰好一个）**：`OK / NOT_ATTEMPTED / NON_FINITE_INPUT /
  SOLVER_FAILURE / MODEL_MISFIT / FLAT_LIKELIHOOD`；
- **`flags`（可并存）**：`AT_GRID_BOUNDARY / LEGACY_PARTIAL_UNDERFLOW /
  LEGACY_TOTAL_UNDERFLOW`。**`AT_GRID_BOUNDARY` 永为 flag，绝非 status。**
- `status ≠ OK ⇒ ipv 必为 NaN`；`status = OK ⇒ ipv/ipv_error/k_eff/weights 全部有限`。

### 4.2 充分性边界

`mse_per_candidate` **仅**对当前 Gaussian squared-error 似然及其 σ 变更充分。
存下它之后改 σ、重算 `K_eff` 无需重解虚拟轨迹（重解才是昂贵步骤）。
**它不足以换核**——反例：逐步平方残差 `(0,2)` 与 `(1,1)` 的 MSE 同为 1，
但 ν=3 的 Student-t 下对数似然为 **−1.02165** 与 **−1.15073**。
换核须另存 `step_sq_residuals`（实现已提供开关，默认关闭，抽样层可开启）。

### 4.3 生产兼容层（**本 RQ 的关键未交付项**）

现有 `estimate_ipv_pair()`、InterHub 导出/绘图、M3 构建、verifier anchor 均假设纯数值。
必须交付：

1. 版本化适配层：把 `ReliabilityResult` 映射到旧接口，**弃权行不得以 0 参与任何聚合**；
2. `NOT_ATTEMPTED`/`SOLVER_FAILURE` 的上游发射点；
3. 全链路 abstention integration tests（从估计器到 verifier 判定）。

**在此三项交付并通过测试前，不得接线生产路径。**

## 5. B3 — σ 重新推导（outcome-blind，且可能判定为"不需要"）

权重锐度由 `Δlog w = ΔMSE/(2σ²)` 决定；σ=0.1 使系数达 50。
**但坍缩与否取决于候选间实际 MSE 间距**（实测特征化）：

| 候选间距 RMS | ΔMSE | σ=0.1 下 K_eff | 判读 |
|---|---:|---:|---|
| ~0.1 m | ~0.01 | ≈2.2 | 未坍缩 |
| ~1.0 m | ~1.0 | <1.05 | 坍缩为硬 argmax = 虚假自信 |

因此 **σ 是否需要重定是经验问题**，答案在 B2 落盘的 `mse_per_candidate` 里；
**B2 必须先于 B3**，且 B3 可能被证据判定为"不需要"。

若需重定：以 `sqrt(min_mse)` 的稳健中心值（中位数）为 σ。
**定性与边界**：这是 **heuristic**，不是观测噪声标准差——它混合真实噪声、
候选网格离散误差与模型偏差。必须声明：(a) 选择目标是"使权重锐度与候选可分辨性相称"；
(b) **拟合只在 dev+guard 上进行，RQ007 sealed 禁止参与**；(c) 若证据显示未坍缩则不动 σ。

**治理**：B3 改变数值 ⇒ 新 `estimator_version`；下游要么重新派生要么显式标注版本；
**绝不覆盖**已冻结的 sigma01 产物与 pinned legacy checkout；新旧并存、不做就地替换。

## 6. 零值机制的可执行分类器（D1–D4）

优先级 **D4 > D1 > D3 > D2 > OK**。D0（未尝试）由 `frame_index` 上游分流。

| 机制 | 判据 |
|---|---|
| `D4_SOLVER_OR_INPUT_FAILURE` | 非有限输入 / 平方距离溢出 |
| `D1_NUMERICAL_UNDERFLOW` | **legacy 与新实现的权重最大绝对差 > 1e-6**，即 legacy 结果已被下溢污染 |
| `D3_MODEL_MISFIT` | `min_mse > min_mse_misfit`（阈值见 §6.1） |
| `D2_FLAT_UNDER_CURRENT_GRID_AND_MODEL` | `c = K_eff/K >= 0.93` 且 `min_mse` 未超阈（阈值以 `c` 表述，禁用近似值 0.61） |
| `OK` | 其余 |

**D1 的定义是"legacy 结果是否被改变"，不是"是否发生过下溢"**：远距候选被清零属
无害部分下溢（判 `OK`，flag 仍如实记录）；只有候选跨越下溢边界、被清零者本应携带
可观权重（构造用例中相对权重 ≈ e^{−0.87} ≈ 0.42）时才判 D1。

**D2 不得称"固有不可辨识"**：平坦是相对于当前候选网格分辨率与当前行为模型的性质；
是否为固有性质需另做网格/模型敏感性，不在本 RQ 断定。

### 6.1 `min_mse_misfit` 的选择与冻结规则（outcome-blind）

```text
min_mse_misfit := Q_p( min_mse )
拟合集 : frame_index ≥ 4 且 status ∉ {NON_FINITE_INPUT, SOLVER_FAILURE}
范围   : dev(19,258) + guard(7,628)；RQ007 sealed(11,342) 禁止参与
p      : 0.99（primary）；0.95 / 0.999 作登记敏感性
单位   : m²
```

分层报告 source × geometry × window；**预先冻结的规则**：任一层的 `Q_0.99` 与全局相差
超过 2 倍则改用分层阈值（分层定义同时冻结）。阈值与拟合集清单须在任何机制分类运行
**之前**计算并登记 SHA-256。实现侧 `min_mse_misfit` 为必填且须有限正数，
杜绝 D3 被静默关闭。

## 7. 覆盖审计（部署前置，不可绕过）

"非信息性估计闸与 OOD 门正交"只在概念上成立，**不蕴含统计独立**：新弃权规则改变被评分总体；
B1 会改变 legacy 下溢行的 IPV，可能同时改变 M3 输入特征与偏离度；冻结 M3/CQR 是在
legacy 标签与原 OOD 门下训练/校准/评估的。**RQ009 报告的约 0.899 覆盖只能作为
历史 ungated 边际结果保留。**

| 项 | 冻结值 |
|---|---|
| split | **RQ009 冻结 test fold 独用**，且禁止参与任何阈值/σ 选择；RQ007 sealed 全程不动 |
| 分层 | `concentration_band ∈ {CONCENTRATED, INTERMEDIATE}` × `estimator_version ∈ {legacy, B1}`（4 格逐格报告） |
| 目标 | primary nominal **90%**；secondary 80% / 95% |
| 估计量 | 经验覆盖率 = `y ∈ [lo_cal, hi_cal]` 的行占比 |
| 不确定度 | case-cluster bootstrap，B=2000，seed `20260726`，percentile 95% CI（按 `case_key` 聚类） |
| 通过标准 | 点估计在 nominal ± **3 pp** 内 **且** CI 下界 ≥ nominal − **5 pp** |
| 失败后果 | 弃权闸**不得部署**；重校准/重训需**单独授权**（重训 M3 明确在本 RQ 范围外） |

**必须一并报告的既有基线**：冻结 M3 test fold、90% nominal 支持域内，`|y|<1e-6` 的近零行
有 **520,826 / 522,219 = 99.7333%** 的区间包含 0。因此机制风险有实据，
但**不成立"每个测不出的帧都判合规"这一全称表述**（约 0.27%、1,393 行不包含 0）。

## 8. 与 M3 / verifier 的边界

| 层 | 本 RQ |
|---|---|
| IPV 估计器 | **修复**（新版本并存） |
| M3 训练标签 | 不改数值 |
| M3 模型（7 分位数 + 共形半径 + OOD 门） | **完全不动，不重训** |
| verifier 判定链 | 新增上游非信息性估计闸，**部署待 §7 审计通过** |

M3 既有的 OOD 门问"我知不知道**这里的人类规范**"；本闸问"这一行的候选权重是否近均匀、从而该 IPV 数值是否携带候选间的判别信息"。二者正交，且非信息性估计闸逻辑上**先于**包络闸。
**禁止表述**：不得称本闸为 estimability gate、不得声称"IPV 未测出"；可辩护表述为"权重近均匀 ⇒ 数值不携带判别信息 ⇒ 弃权"。"按集中度筛选后重训 M3"**不在本 RQ 内**，届时作为独立下游决策项提出。

## 9. 实现现状（`BUILD_WHILE_DENY / B1_PROTOTYPE / B2_SCAFFOLD_NOT_WIRED`）

已交付（未接入任何生产路径、未改 legacy、未碰真实数据、未提交作业）：

- `src/sociality_estimation/core/reliability_logdomain.py`：log 域权重、正交结果契约、
  充分统计量与可选逐步残差、`classify_zero_mechanism()`、fail-closed 校验器
  （σ/ratio/tol/shape/grid_id，禁 `bool`，`2σ²` 下溢检查）、结果不变量
  （禁 `OK+NaN`、非 OK 必 NaN、status 必属 `STATUS_SET`、flags 不重复）、
  双定义 `underflow_rms_threshold`、legacy 忠实复刻（仅供平价与机制判别）。
- `tests/test_rq015_reliability_logdomain.py`：**36/36 通过**，含
  50 组平价门、兜底不可达、D1 全/部分下溢（含"无害部分下溢判 OK"的反向用例）、
  D2/D3 分离、D4 双路径、换核反例、两个下溢阈值（精度 5e-4）、
  以及 `python -O` 剥离 assert 后守卫仍有效的子进程测试。

**未交付**：§4.3 的生产兼容层三项。

## 10. 阶段顺序与验收

1. **B1**：平价门 ≤1e-12；兜底不可达（有测试）。
2. **B2**：契约冻结 + 生产兼容层 + 全链路 abstention integration tests；充分统计量落盘。
3. **机制拆分**：§6.1 阈值先冻结并登记 SHA，再跑分类；给出 D1/D2/D3/D4 比例。
   **抽样精度合同**：冻结抽样框、各层分配、随机种子、代表性层与疑似下溢加密层、
   目标比例的 CI/误差上限、case/scene 聚类处理；"数万锚点"本身不构成规格。
4. **B3**：仅在 §5 证据支持时执行；新版本号。
5. **覆盖审计**：§7 全部通过方可考虑部署弃权闸。

## 11. 资源与执行面前置

B1 代码改动小，但**全量重跑代价等于当年 sigma01 全量重跑**（虚拟轨迹须重解），需 HPC；
**先做分层抽样**验证平价门与机制比例，再申请全量预算。B3 因有充分统计量几乎免费。

**执行面硬伤（须在提交前解决）**：现行 run-spec schema 将 `rq_id` 钉死为常量 `"RQ014"`，
launcher 带 `--rq014-only`，生成 sigma01 的 pinned checkout `5edd2810` 已退役为 tombstone
⇒ HPC 路径**当前字面不可执行**，需先扩展执行面或另开通道。
另：两份既有 log 域实现（`scripts/rq014/wod_ipv_adapter.py` 与
`archived/report_process/RQ010B_ipv_rating_pilot_20260629/` 下副本）**均保留均匀兜底**，
不可作为干净基线或参考实现。作业名前缀 `zxc-rq015b-`。

## 12. 边界与禁止事项

- 不重训 M3；不覆盖 sigma01 冻结产物、RQ009 特征矩阵或 pinned legacy；
- 不得以任何下游关联（含评分）作为 σ、阈值或网格的选择准则；
- 不得修改任何已接受 `decision.md`；
- 新估计器**不得**进入 RQ014 lane（其 R3 重跑存在时序冲突）。

## 13. 生效条件

须经独立双路复审无 blocker 方可进入 Formal G1；其后逐 operation 走
scoped decision → allowlist → immutable spec → validate-only → 单次提交。
