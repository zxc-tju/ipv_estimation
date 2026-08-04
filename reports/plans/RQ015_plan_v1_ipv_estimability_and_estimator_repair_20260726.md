# RQ015 Plan v1 — IPV 可估计性打标与估计器数值修复

状态：`v1.1 / AWAITING_RE_REVIEW`（无计算授权；不得开始任何 Phase；`execution_authorized=false`）
本稿已按 2026-07-26 双路复审（`BLOCKED/REQUEST_CHANGES`）的九项关闭清单修订。
日期：2026-07-26 ｜ 起草：Claude（PI 角色）
取代：`RQ015_plan_v0_ipv_estimability_contract_and_estimator_repair_20260725.md`
（v0 保留为历史记录，**其 §2.5「K≥9 网格混入」结论已被证伪，不得再引用**）

复审依据（三方独立、结论一致 BLOCKED/REQUEST_CHANGES）：

- `reports/knowledge/RQ015_ipv_estimability_contract/reviews/claude_stats_review_v0_20260725.md`
- `reports/knowledge/RQ015_ipv_estimability_contract/reviews/claude_execution_review_v0_20260725.md`
- 三路复审记录：`main_workflow.log:1888`

PI 决策（2026-07-26，本文按此定稿）：D-1 剔除封存集重算；D-2 `K_eff` 直接使用，
不新增校准前置；D-3 **修复估计器**（v0 曾拟暂缓，PI 改为执行）；D-4 Phase C 移出。

---

## 0. 一句话

把"这一帧到底有没有测出 IPV"变成显式、可审计的状态量；修掉使"无信息"被静默写成
"中性 0"的数值缺陷；并给 verifier 装上第二道弃权闸。**本 RQ 不重训 M3、不修改任何
已冻结产物、不重估既有结论。**

## 1. v0 的事实错误更正（三方独立复算一致）

| v0 的说法 | 实测 | 处置 |
|---|---|---|
| 9.47% 的行 `error>0.62204` ⇒ 混入 K≥9 候选网格 | **证伪**。这些是估计器 warm-up 占位：`ipv_estimation.py:247-252` 把数组初始化为 `zeros`/`ones`，仅 `t ≥ MIN_OBSERVATION=4` 才写回。`38,228 × 4 × 2 = 305,824` 行 `frame_index<4` 恒为 `IPV=0, error=1`；剔除后 `error>0.62204` 的行为 **0** | 删除该结论；新增 **D0=未尝试估计** 状态 |
| 零值率 43.71%、`err≥0.61` 54.54% | 含占位行。剔除后：**41.28%** / **52.58%** | 全文重算（见 §2） |
| 下溢临界 RMS ≈ 3.8 m | 该式误用 `n=1`。实现先连乘再开 `1/n` 次方，**乘积在开方前即下溢**：hw=4（n=5）≈ **1.73 m**，hw=10（n=11）≈ **1.18 m** | 更正；此更正**加重**而非减轻 D1 |
| 32.4% 的零值精确等于 `1−1/√7` ⇒ 走了均匀兜底 | 1e-9 容差下仅 **9.88%**（32.4% 需 2e-5 容差）；且现有产物未保存 `fallback_hit`/候选对数似然，**无法**从 `error` 值证明走过兜底分支 | 删除该推断；改由 §4 的新旧对比获得 |
| D1/D2 二分即完备 | 至少还有 D0（未尝试）、D3（模型失配：所有候选都拟合很差）、D4（求解失败/输入非有限/历史不足） | 改为 **D0–D4 五状态** |

上述错误已扩散至 `START_HERE.md` 与 `reports/knowledge/RQ015_ipv_estimability_contract/README.md`，
**必须与本计划同批更正**。

## 2. 更正后的触发画像（`frame_index ≥ 4` 有效行，7,086,138 agent-值）

复现脚本：`reports/plans/prompts/RQ015_portrait_scan_v1.sh`（纳入校验和；
注意源 CSV 第 9 字段含引号内逗号，朴素 `awk -F,` 会给出量级错误的结果）。

| 指标 | 值 |
|---|---:|
| `\|IPV\|<1e-9` | **41.2794%** |
| `\|IPV\|<1e-6` | 41.4086% |
| `ipv_error ≥ 0.61`（K_eff ≥ 6.6/7，实质无辨识力） | **52.5810%** |
| `ipv_error ≤ 0.50`（K_eff ≤ 4） | **24.1688%** |
| `ipv_error > 0.62204` | **0.0000%**（占位行剔除后归零，印证 §1） |
| `P(IPV=0 \| error ≥ 0.61)` | 71.6527% |
| 零值中 `error ≥ 0.61` 的比例 | 91.2702% |

**D-1 待办**：以上为全语料口径。Phase A 必须**剔除 RQ007 封存集（11,342 cases）
后重算全部画像**，且 §5 阈值只在 dev(19,258)+guard(7,628) 上冻结。封存集的画像
留待 RQ007 自行解封后补做。若 case 级 fold 归属文件不可得，Phase A 第一步即为定位或
重建该归属，未定位前不得冻结阈值。

case 级与按数据源、按距离的画像（v0 §2.4b/2.4c）同样须在剔除占位行与封存集后重算；
v0 的结论方向（case 级约九成可用、可估计性不主要由接近程度决定）预期不变，但数字会变。

## 3. 可估计性判据（PI 决策 D-2：直接采用，不新增校准前置）

```text
K_eff = 1 / (1 − ipv_error)²      # ipv_error = 1 − √(Σ wᵢ²)，即权重集中度
K     = 该行实际候选数（须显式记录）

ESTIMABLE      : K_eff ≤ 4
WEAK           : 4 < K_eff < 0.93·K
NOT_ESTIMABLE  : K_eff ≥ 0.93·K          # K=7 时对应 ipv_error ≥ 0.61
NOT_ATTEMPTED  : frame_index < MIN_OBSERVATION（不参与上述判定）
```

**依据与边界（如实记录）**：该量即 RQ007 已接受并冻结的"估计器集中度指数"，
RQ007 `decision.md` 将其表述为 *identifiability proxy*，且其 C1–C3 为 dev/guard 边界结论、
held-out 仍封存。本 RQ 依 PI 决策直接采用，不再新增仿真恢复校准。相应地，
本 RQ 的一切基于阈值的陈述均继承该 proxy 边界，**不得表述为"IPV 估计误差"的直接度量**。

`NOT_ATTEMPTED` 行 `ipv_error=1`，`K_eff` 公式在此除零，故必须在判定前先分流。

## 4. Phase B — 估计器修复（PI 决策 D-3；本 RQ 的核心工程交付）

### 4.0 关键数学事实

现行 `cal_traj_reliability` 计算 `varᵢ = [∏ₖ φ(d_ik)]^{1/n}`，
`φ(d) = (1/(σ√2π))·exp(−d²/2σ²)`。取对数：

```
log varᵢ = −log(σ√2π) − MSEᵢ / (2σ²) ，  MSEᵢ = (1/n)·Σₖ d_ik²
```

首项对所有候选相同、归一化时约掉。因此**现行计算等价于**：

```
w = softmax( −MSEᵢ / (2σ²) )
```

连乘、开方与常数因子全是绕路，而连乘正是下溢发生处。

### 4.1 B1 — log 域改写（σ 不变，数学等价）

```python
mse  = np.mean(rel_dis**2, axis=1)   # (K,)
logw = -mse / (2 * sigma**2)
logw -= logw.max()                   # 稳定化：最大项恒为 exp(0)=1
w     = np.exp(logw); w /= w.sum()   # 分母 ≥ 1，永不为零
```

均匀兜底分支变为不可达 → 改为断言 + 状态码。

**平价门（硬性）**：在**未发生下溢**的行上，新旧实现必须逐位一致
（`max_abs_diff ≤ 1e-12`）。依据 `reports/knowledge/_analysis/ipv_estimator_divergence_investigation.md`
（一次纯向量化重构曾使 sigma01 数值偏移 0.281），此门不可豁免。

**B1 不改变任何科学结论**，只使"算崩了"不再伪装成"中性"。

### 4.2 B2 — 输出充分统计量与状态码

每行输出（版本化 schema，先冻结再实现）：

```
ipv, ipv_error, status, reason_code, K, grid_id,
min_mse, loglike_gap, at_grid_boundary, mse_per_candidate[K],
estimator_version
```

- `min_mse = minᵢ MSEᵢ`：**区分 D2 与 D3 的关键**——"大家一样好"还是"大家一样烂"；
- `at_grid_boundary`：argmax 落在网格端点（对应 v0 观察到的 k=±3 堆积，
  意味着真值可能在网格之外）；
- `mse_per_candidate`：**对当前 Gaussian squared-error 似然及其 σ 变更是充分统计量**。
  存下它之后，改 σ、重算 `K_eff` 均无需重新求解虚拟轨迹（重解才是昂贵步骤，
  当年 sigma01 全量重跑需 HPC），这是本次修复中性价比最高的一项。
  **充分性边界（复审 C-BLOCK-4.1，v1 初稿在此处错误泛化）**：它**不足以**支持换核。
  反例：逐步平方残差 `(0, 2)` 与 `(1, 1)` 的 MSE 同为 1，但在 ν=3 的 Student-t 下
  去掉公共常数后的对数似然分别为 **−1.02165** 与 **−1.15073**。若要支持重尾核，
  必须另存逐步残差 `step_sq_residuals[K][n]`（实现已提供 `keep_step_residuals` 开关，
  默认关闭，抽样层可开启）。"由 MSE 支持任意换核"的表述**作废**。

状态码取值：`OK / NOT_ATTEMPTED / SOLVER_FAILURE / NON_FINITE_INPUT /
MODEL_MISFIT / FLAT_LIKELIHOOD / AT_GRID_BOUNDARY`。
**禁止**以单一 `NOT_ESTIMABLE` 概括。`ipv` 在非 `OK` 状态下取 `NaN`；
消费方必须显式处理，不得将弃权当作 0 参与任何聚合（现有 InterHub 汇总、绘图、
M3 构建与 WOD 固定输出均假设数值结果，兼容层须在 B2 内一并交付）。

### 4.3 B3 — σ 重新推导（outcome-blind）

修好数值后，权重锐度完全由 `Δlog w = ΔMSE/(2σ²)` 决定；σ=0.1 使系数达 50。

**更正（2026-07-26，实现落地后由测试特征化）**：v1 初稿断言"σ=0.1 必然坍缩为
one-hot"，这是**绝对化的错误表述**。坍缩与否取决于候选之间的实际 MSE 间距：

| 候选间距（RMS） | ΔMSE | σ=0.1 下的 K_eff | 判读 |
|---|---:|---:|---|
| ~0.1 m | ~0.01 | ≈ 2.2 | 未坍缩，权重仍反映证据强弱 |
| ~1.0 m | ~1.0 | < 1.05 | 完全坍缩为硬 argmax = 虚假自信 |

见 `tests/test_rq015_reliability_logdomain.py::test_sigma_sharpness_is_conditional_on_candidate_spread`。

因此 **σ 是否需要重定是一个经验问题，而不是先验断言**——答案就在 B2 落盘的
`mse_per_candidate` 里：真实候选轨迹之间的典型 MSE 间距一旦测出来，σ 该不该动、
该动到多少就自明了。**这正是 B2 必须先于 B3 的理由**，也意味着 B3 有可能在证据面前
被判定为"不需要"。

根因：σ=0.1 m 从来不是按"轨迹预测误差有多大"设定的（真实误差为米级），
而是为迁就旧数值行为调出来的。

**重定方法（不看任何下游关联、不接触评分）**：以 B2 存下的
`sqrt(min_mse)`（获胜候选的残差尺度）的稳健中心值（中位数）作为 σ。
同时报告分位数与按数据源/几何的分层，作为敏感性。

**定性与边界（复审 C-BLOCK-4.2）**：`median(sqrt(min_mse))` **不是**自动成立的
观测噪声标准差——它混合了真实观测噪声、候选网格离散误差与模型偏差，
只能作为**heuristic**。因此必须显式声明：(a) 选择目标是"使权重锐度与候选可分辨性
相称"，而非"估计噪声真值"；(b) **拟合只在 dev+guard 上进行，RQ007 sealed 同样禁止
参与**（该禁令与 §2 的 Phase A 阈值一致，v1 初稿仅对阈值声明、遗漏了 σ）；
(c) 若 §4.4b 的 MSE 间距证据显示 σ=0.1 未造成坍缩，B3 可判定为"不需要"。

**治理**：B3 **改变数值** ⇒ 新 `estimator_version`；所有下游产物要么重新派生、
要么显式标注版本；**绝不覆盖**已冻结的 sigma01 产物与 pinned legacy checkout。
新旧实现并存、版本标识，不做就地替换。

### 4.4 修复本身即为 D1/D2/D3 判据（替代 v0 的"影子仪表"设想）

在同一批输入上跑新旧两版并对比：

| 修前 | 修后 | 判定 |
|---|---|---|
| 权重近乎均匀 | 变得集中 | **D1 数值下溢**（伪不可辨识，已消除） |
| 权重近乎均匀 | 仍均匀，`min_mse` 小 | **D2 固有不可辨识**（IPV 对轨迹无杠杆——真实发现） |
| 权重近乎均匀 | 仍均匀，`min_mse` 大 | **D3 模型失配**（参考线/噪声/模型不适用） |

因此 v0 §2.4e"log 域后残留即可归因于 D2"的断言被**替换**为上表；
D0 由 `frame_index` 直接分流，D4 由状态码捕获。

### 4.4b 实现现状（`BUILD_WHILE_DENY / B1_PROTOTYPE / B2_SCAFFOLD_NOT_WIRED`）

按仓库既有的 build-while-deny 惯例先行落地。**如实定性（复审 C-BLOCK-1）：
B1 为原型、B2 仅为 scaffold；生产接线尚未交付，不得据此声称"可部署弃权链"。**

已交付：

- `src/sociality_estimation/core/reliability_logdomain.py`
  —— log 域权重；**正交结果合同**（互斥主状态 `status` + 可并存诊断 `flags` +
  `reason_code` + `schema_version`/`grid_id`/`estimator_version` 三元组）；
  充分统计量 `mse_per_candidate` 与可选 `step_sq_residuals`；
  **可执行的机制分类器** `classify_zero_mechanism()`；fail-closed 入口校验；
  双定义的 `underflow_rms_threshold(boundary="subnormal"|"zero")`；
  以及 legacy 概率域算法的忠实复刻（**仅供平价测试与机制判别**）。
- `tests/test_rq015_reliability_logdomain.py` —— **18 项全部通过**。

**尚未交付（Phase B 明列交付物，必须在启用前完成）**：
`estimate_ipv_pair()` 与 InterHub 导出/绘图、M3 构建、verifier anchor 的
数值接口兼容层；`NOT_ATTEMPTED`/`SOLVER_FAILURE` 的上游发射点；
全链路 abstention integration tests。**在此之前不得接线生产路径。**

关键测试结论（除平价门外）：

- **D1 的定义被钉死为"legacy 结果是否被改变"，而非"是否发生过下溢"**。
  测试证明：远距候选被清零属**无害**部分下溢（判 `OK`，但 flag 如实记录）；
  只有候选跨越下溢边界、被清零者本应携带可观权重（相对权重 ≈ e^{−0.87} ≈ 0.42）时
  才判 D1。这修正了复审指出的"D1 只看是否命中均匀兜底"的漏洞。
- D2 与 D3 由 `min_mse` 分开且**阈值为必填参数**——缺省即报错，
  杜绝 v1 初稿 `DEFAULT_MIN_MSE_MISFIT=None` 导致 D3 永不可达的问题。
- 两个下溢阈值分别命名并分别测试：进入 subnormal 区
  `n=5: 1.6915 m / n=11: 1.1470 m`；乘积舍入为精确 0（触发兜底）
  `n=5: 1.7336 m / n=11: 1.1752 m`。

### 4.5 成本与分期

B1 代码改动极小，但**全量重跑的代价等于当年 sigma01 全量重跑**（虚拟轨迹须重新求解），
需 HPC。**先做分层抽样**（按数据源 × 几何 × 窗长分层，量级数万锚点）验证平价门并
估计 D1/D2/D3 比例；据此再申请全量预算。B3 因有 B2 的充分统计量，几乎免费。

### 4.6 新增可估计性闸后的 M3 条件覆盖审计（复审 C-BLOCK-3）

"可估计性闸与 OOD 门正交"只在**概念**上成立，**不蕴含统计独立**：

- 新弃权规则改变了被评分总体；
- B1 会在 legacy 下溢行上改变观测 IPV，可能同时改变 M3 的输入特征与最终偏离度；
- 冻结 M3/CQR 是在 legacy 标签与原 OOD 门下训练、校准、评估的；RQ009 报告的
  约 **0.899** 90% 覆盖是**该原始总体上的边际结果**，且其子组覆盖与 LODO 范围本就不均匀。

因此：冻结 M3 可以字节不变，但**其覆盖声明只能作为历史 ungated 结果保留**，
不得移植为"新增可估计性闸后 verifier 仍近名义覆盖"。

强制要求：

1. 在 `gate-pass × estimator_version` 人群上做 **outcome-blind 的选择后条件覆盖审计**；
2. §7 的"可部署交付"由既成事实**降级为待审候选**，审计通过前不得部署；
3. 若审计显示覆盖偏离名义值，重校准或重训需**单独授权**，
   RQ015 不得静默决定（重训 M3 明确在本 RQ 范围之外）。

## 5. Phase A — 回溯打标（四层可行性）

历史产物并非都能直接打标，须分层处理，禁止笼统声称"无需重算"：

1. **可直接回填**：已存 `ipv_error` 且 K 已知的产物（sigma01 时间序列、RQ009 目标/特征矩阵）
   → 直接判定 D0 与三态；
2. **可由 provenance 重建**：K、网格 ID 未直接记录但可由运行配置确定者；
3. **必须受控重算**：需要 `min_mse` / `mse_per_candidate` 才能定 D2 vs D3 者
   → 归入 Phase B 的抽样重跑。**抽样须有精度合同（复审 5.3）**：冻结抽样框、
   各层分配、随机种子、代表性层与疑似下溢加密层、目标比例的置信区间/误差上限，
   以及 case/scene 聚类处理；"数万锚点"本身不构成可复现规格；
4. **无法恢复**：标 `unknown`，明确列出并说明后果（部分 WOD 产物无逐行 IPV/error）。

交付：`estimability_ledger`（schema 先冻结）、双口径（帧级/case 级）画像、
按数据源与几何的分层、以及"什么预测了帧级可估计性"的探索性分析
（v0 §2.4c 显示可估计性**不**主要由接近程度决定：<5 m 档可估计率 44.5% 但仅占 3.8% 帧，
5 m 以外各档平坦于约 22%——数字待按 §2 口径重算）。

## 6. Phase C0 — 下游资格矩阵（PI 决策 D-4：重估移出本 RQ）

**不在 RQ015 内重估任何既有研究。** v0"重估四个既有阴性"的表述作废——
其中 RQ011B 为 `PROVISIONAL_NULL / UNDER_IDENTIFIED`（非冻结阴性），
RQ003 为 Tier-B 边界证据（非通用 IPV 阴性）。

本 RQ 仅交付一份**资格矩阵**，对每个下游研究说明：

- 其分析单位与聚合结构（如 RQ010B 场景内三候选比较；RQ012B 锚点→单元 + 聚类推断）；
- 按可估计性筛选是否会改变估计量与暴露定义；
- **选择偏倚风险**：可估计性可能与结果变量相关（如混乱交互既难估计又更易出事故），
  故筛选并非中性缩样；需要何种额外控制或敏感性；
- 结论：`可重估 / 需重新设计 / 不适用`。

各 RQ 是否重开，由其自身按既有流程发起 amendment，RQ015 不代为决定。

## 7. 与 M3 / verifier 的关系（明确边界）

| 层 | 本 RQ 是否触动 |
|---|---|
| IPV 估计器（`cal_traj_reliability`） | **修复**（Phase B，新版本并存） |
| M3 训练标签（估计器输出） | 不改数值，**只贴出处标签** |
| M3 模型（7 分位数 + 共形半径 + OOD 门） | **完全不动**，不重训 |
| verifier 判定链 | **新增上游弃权闸**（见下） |

M3 已有的 OOD 支持门回答的是"我知不知道**这里的人类规范**"（InterHub 弃权率 4.78%）；
它**不**回答"这辆车的 IPV **测出来没有**"。二者正交，且可估计性闸在逻辑上**先于**包络闸：
偏离度 = |实测 IPV − 规范中心|，左项没测出来时右项再准也无意义。

**当前行为（缺陷）**：测不出 → `IPV=0` → 落在中心≈0 的人类包络内 → 判定"符合规范"。
即每个测不出的帧都被静默判为合规。本 RQ 的可部署交付即是把它改为"无法判定"。

**明确不在本 RQ 内**：用可估计标签筛选后重训 M3。Phase A/B 会产出逼问该问题的证据，
届时作为独立的下游决策项提出。

## 8. 验收标准

- **A 通过**：四层可行性分类完成；封存集已剔除且阈值只在 dev/guard 冻结；
  双口径画像与分层经独立复核；`estimability_ledger` schema 已冻结。
- **B1 通过**：平价门 `≤1e-12`；兜底分支不可达（有测试）；D1/D2/D3 抽样比例给出。
- **B2 通过**：正交结果合同冻结并实现；**旧接口兼容层与全链路 abstention integration tests 交付**；充分统计量落盘（并明示其仅对 Gaussian/σ 充分）。
- **B3 通过**：σ 由 outcome-blind 规则确定并记录；新版本号；无任何冻结产物被覆盖。
- **C0 通过**：资格矩阵覆盖 RQ003/RQ009/RQ010B/RQ011B/RQ012B/RQ014，逐项给出结论。
- **覆盖审计通过**：gate-pass 人群上的条件覆盖审计完成（§4.6），未通过则弃权闸不得部署。

## 9. 边界与禁止事项

- 本 RQ **不**提出任何行为学主张，是测量学研究；
- 不得以任何下游关联（含评分）作为 σ、阈值或网格的选择准则；
- 不得修改任何已接受 `decision.md`；RQ007-KC-C2（"高指数 ≠ IPV 0"）
  与实测 `P(IPV=0 | error≥0.61)=71.65%` 的张力，登记为待核查项交回 RQ007 流程；
- 不得覆盖 sigma01 已冻结产物、RQ009 特征矩阵或 pinned legacy estimator；
- 与 RQ014 的关系：RQ014 冻结 lane 不受影响；其 R3 重跑存在时序冲突风险，
  Phase B 的新估计器**不得**进入 RQ014 lane。

## 10. 执行面前置（v0 的硬伤，须在启动前解决）

- 现行 run-spec schema 将 `rq_id` 钉死为常量 `"RQ014"`，launcher 带 `--rq014-only`，
  生成 sigma01 的 pinned checkout `5edd2810` 已退役为 tombstone
  ⇒ **§4 的 HPC 执行路径当前字面不可执行**，需先扩展执行面或另立通道；
- 两份既有 log 域实现（`scripts/rq014/wod_ipv_adapter.py`，以及 `archived/report_process/RQ010B_ipv_rating_pilot_20260629/` 下的副本）
  **均保留均匀兜底**，不可作为干净基线或参考实现；
- 本计划与校验和须提交进 merged main，否则"冻结"不成立；
- 作业名前缀 `zxc-rq015-`，durable root 按 HPC 复用指南；重计算一律 sbatch。

## 11. 生效条件

本 v1 须经独立双路复审（统计口径 / 执行治理，身份互异且均非起草者）无 blocker，
方可进入 Formal G1；G1 通过后仍须逐 operation 走 scoped decision → allowlist →
immutable spec → validate-only → 单次提交。当前 `execution_authorized=false`。
