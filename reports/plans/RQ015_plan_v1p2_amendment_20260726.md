# RQ015 Plan v1.2 Amendment — 关闭 v1.1 双路复审的 4 blocker

状态：`AWAITING_RE_REVIEW`｜`formal_g1_eligible=false`｜`execution_authorized=false`
日期：2026-07-26 ｜ 起草：Claude（PI 角色）
基底：`RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md`（v1.1，**不覆写、保留原样**）
本文按复审要求另存为新文件，仅**取代** v1.1 中下列小节：§4.2 字段表、§4.4 的 D2 命名、
§4.6 覆盖审计、§7 的部署性表述，并**新增** §A2 的阈值选择规则。其余条款照旧。

响应复审：

- `reports/knowledge/RQ015_ipv_estimability_contract/reviews/codex_dual_review_synthesis_v1p1_20260726.md`
- `.../codex_stats_review_v1p1_20260726.md`（2 blocker / 3 major / 1 minor）
- `.../codex_execution_review_v1p1_20260726.md`（3 blocker / 3 major）

两路均先给 `PASS_WITH_CONDITIONS`、在互不通信的情况下核验新增证据后**独立改判 `BLOCKED`**。
这一事实本身被记录为一个警示：v1.1 的补充材料引入了新的不一致，说明**新增规格的速度
超过了闭合速度**（另见 §A5 的范围建议）。

---

## A1（blocker 1）B2 字段表、status/flags 语义统一

v1.1 §4.2 的字段表与实现不一致：字段表列 `at_grid_boundary`（布尔）却又把
`AT_GRID_BOUNDARY` 列入 status 取值，且遗漏 `flags`/`k_eff`/`schema_version` 等。
**以下为唯一权威字段表**，与 `ReliabilityResult` 逐字段一致：

```text
weights[K]            float64  各候选归一化权重
ipv                   float64  status=OK 时为有限值；否则**必为 NaN**
ipv_error             float64  1 − √(Σwᵢ²)（RQ007 集中度指数）
k_eff                 float64  1/(Σwᵢ²)
status                str      互斥主状态，取值 ∈ STATUS_SET（见下）
flags                 tuple    可并存诊断，与 status 正交
reason_code           str|None
K                     int      实际候选数
grid_id               str      非空
min_mse               float64  minᵢ MSEᵢ（单位 m²）
loglike_gap           float64  (MSE₂−MSE₁)/(2σ²)
mse_per_candidate[K]  float64  充分统计量（仅对 Gaussian/σ 充分）
step_sq_residuals     [K][n]|None  换核所需；默认不存
schema_version        str      "rq015-reliability-result-v2"
estimator_version     str
sufficiency_scope     str      "gaussian_sigma_only"
```

- **`STATUS_SET`（互斥，恰好一个）**：`OK / NOT_ATTEMPTED / NON_FINITE_INPUT /
  SOLVER_FAILURE / MODEL_MISFIT / FLAT_LIKELIHOOD`。
- **`flags`（可并存）**：`AT_GRID_BOUNDARY / LEGACY_PARTIAL_UNDERFLOW /
  LEGACY_TOTAL_UNDERFLOW`。
- **`AT_GRID_BOUNDARY` 永远是 flag，绝不是 status**。
  由 `tests/...::test_status_set_excludes_grid_boundary` 强制。

## A2（blocker 2）`min_mse_misfit` 的科学选择与冻结规则；D2 语义降级

### A2.1 阈值选择规则（outcome-blind，冻结后不可动）

`MODEL_MISFIT` 要捕捉的是"**没有任何候选**能解释观测"这一尾部，因此阈值必须是
语料相对量而非拍脑袋常数：

```text
min_mse_misfit := Q_p( min_mse )   over 冻结拟合集
拟合集   : frame_index ≥ 4 且 status ∉ {NON_FINITE_INPUT, SOLVER_FAILURE}
数据范围 : dev(19,258) + guard(7,628) 的 cases —— **RQ007 sealed(11,342) 禁止参与**
p        : 0.99（primary）；0.95 与 0.999 作为登记敏感性，不改变 primary
单位     : m²（均方距离），与 `min_mse` 同尺
```

分层报告 source × geometry × window。**冻结规则（非事后选择）**：若任一层的
`Q_0.99` 与全局值相差超过 2 倍，则改用**分层阈值**，分层定义同样在此冻结。
阈值与拟合集清单在任何机制分类运行**之前**计算并登记 SHA-256。

实现侧已强制 `min_mse_misfit` 为必填且须为有限正数（缺省即 `TypeError`，
非法值即 `EstimatorInputError`），杜绝 D3 被静默关闭。

### A2.2 D2 不得称"固有不可辨识"

平坦是**相对于当前候选网格分辨率与当前行为模型**的性质：更细的网格或不同的模型
可能把它解开。因此机制标签由 `D2_INTRINSIC_FLAT` **更名为**
`D2_FLAT_UNDER_CURRENT_GRID_AND_MODEL`，并在全部文本中禁止"固有不可辨识"表述。
（由 `tests/...::test_d2_label_is_not_intrinsic` 强制。）
相应地，v1.1 §4.4 判据表中"D2 固有不可辨识（IPV 对轨迹无杠杆——真实发现）"
改为"**在当前网格与模型下平坦**；是否为固有性质需另做网格/模型敏感性，
不在本 RQ 断定"。

## A3（blocker 3）M3 gate-pass 覆盖审计的完整规格

v1.1 §4.6 只说了"要审计"，未给 split/目标/CI/容差/通过标准。**以下为完整规格**：

| 项 | 冻结值 |
|---|---|
| 数据 split | **RQ009 冻结 test fold 独用**；该 fold 不得参与任何阈值/σ 选择；RQ007 sealed 全程不动 |
| 分层 | `gate_status ∈ {ESTIMABLE, WEAK}` × `estimator_version ∈ {legacy, B1}`（4 格，逐格报告） |
| 目标 | primary：nominal **90%**；secondary：80% 与 95% |
| 覆盖估计量 | 经验覆盖率 = `y ∈ [lo_cal, hi_cal]` 的行占比 |
| 不确定度 | **case-cluster bootstrap**，B=2000，seed `20260726`，percentile 95% CI（按 `case_key` 聚类） |
| 通过标准 | 点估计落在 nominal ± **3 pp** 内 **且** CI 下界 ≥ nominal − **5 pp** |
| 失败后果 | 弃权闸**不得部署**；重校准或重训需**单独授权**（重训 M3 明确在本 RQ 范围外） |

**必须一并报告的既有基线（复审实测）**：在冻结 M3 test fold、90% nominal 的支持域内，
`|y|<1e-6` 的近零行有 **520,826 / 522,219 = 99.7333%** 的区间包含 0。
这个数既说明机制担忧有实据，也**订正了我此前的全称表述**（见 A4）。

## A4 对既有表述的订正

v1.1 §7 与我此前的会话表述写作"**每个**测不出的帧都被静默判为合规"。
这是**错误的全称量化**。准确表述为：

> 在冻结 M3 test fold、90% nominal 支持域内，近零 IPV 行（`|y|<1e-6`）中
> **99.7333%（520,826/522,219）** 的区间包含 0，因此**绝大多数**测不出的帧
> 会落入包络内而被判为合规；其余约 0.27%（1,393 行）不会。
> 机制风险由该比例支撑，但不成立"全部"。

同一订正适用于 `START_HERE.md` 与本 RQ 的 README。

## A5（major 汇总）fail-closed 收口与实现现状

已在实现与测试中关闭（`tests/test_rq015_reliability_logdomain.py`，**36/36 通过**）：

- **`OK + NaN` 不可能**：`ReliabilityResult.__post_init__` 强制
  `status=OK ⇒ ipv/ipv_error/k_eff/weights 全部有限`；非 OK 状态 ⇒ `ipv` 必为 NaN；
  status 必属 `STATUS_SET`；`flags` 不得重复。违反抛 `ResultInvariantError`。
- **非法 ratio / tolerance 被拒**：`k_eff_flat_ratio ∈ (0,1]`、
  `legacy_divergence_tol > 0` 有限，且拒绝 `bool`；参数化测试覆盖
  `0 / 负值 / >1 / NaN / inf / True`。
- **极小正 σ 不再泄漏 `AssertionError`**：新增 `2σ²` 下溢检查与 log-weight 有限性检查，
  统一抛 `EstimatorInputError`；`1e-200`、`5e-324`、`True` 均有测试。
- **`assert` 剥离后守卫仍有效**：softmax 兜底守卫由 `assert` 改为显式
  `raise ResultInvariantError`，并有 `python -O` 子进程测试证明其不失效。
- 另补 `grid_id` 非空、`mse` 形状/非空校验。

实现状态维持 `BUILD_WHILE_DENY / B1_PROTOTYPE / B2_SCAFFOLD_NOT_WIRED`：
**生产兼容层、`NOT_ATTEMPTED`/`SOLVER_FAILURE` 上游发射点、全链路 abstention
integration tests 仍未交付**，不得接线生产路径。

## A6 PI 给用户的范围建议（需决策，非本文自行决定）

三轮复审均 BLOCKED，且每轮的阻断都真实存在。观察到的模式是：
本计划同时试图冻结 (a) 回溯打标/诊断研究，与 (b) 完整的估计器再工程 + 统计有效性契约；
复审正确地要求两者都完整规格化，于是规格面积持续扩张。

因此建议考虑拆分（**由 PI 决定，本文不自行执行**）：

- **RQ015a — 回溯打标与画像**：D0 更正、剔除 sealed 的双口径画像、
  `estimability_ledger`、可估计性预测子分析、C0 下游资格矩阵。
  **不改估计器、不部署任何闸门 ⇒ 不需要覆盖审计**，规格面积小、可较快通过复审并产出
  真实价值（诚实的测量图景）。
- **RQ015b — 估计器修复与弃权闸**：B1/B2/B3、生产兼容层、
  gate-pass 覆盖审计、部署决策。规格面积大，但目标单一。

若维持单一 RQ，则接受"闭合周期较长"这一代价；两种选择都合法，只是不应混合。

## A7 生效条件

本 amendment 须经同口径独立双路复审无 blocker，方可与 v1.1 合并计入 Formal G1 资格。
`execution_authorized` 保持 `false`；未接线生产路径、未修改任何冻结产物、未提交 HPC 作业。
