# RQ015A v0 独立技术复审——Reviewer 1

## Review setup

- 复审对象：`reports/plans/RQ015A_plan_v0_ipv_estimability_labelling_20260726.md`
- 冻结 SHA-256：`cd352390d816c12c77942211ad73479a5d3b71c43c8c801150ab3e25cfa9fea8`（现场复算一致）
- 复审日期：2026-07-26
- 评审重心：technical soundness / technical failings；同时简要覆盖 originality、significance、interdisciplinary interest 和 readability。
- 独立性声明：本评审未读取、未引用任何新产生的 RQ015A review 文件，也未与其他 Reviewer 交换意见。
- 评审边界：本次是计划及可执行性复审，不是对已完成 RQ015A 结果的复审。所有核查均为本地只读；未改估计器、未改数据、未启动 HPC，也未启动 Formal G1。

## Overall assessment

**Verdict: `BLOCKED`**

**Finding counts: 4 blocker / 4 major / 1 minor.**

RQ015A 的拆分方向是正确的：它把“旧产物到底测到了什么”与“如何修估计器并部署弃权闸”分开，且明确禁止行为学、因果和部署性外推（计划 `:18-32`, `:144-150`）。这一范围收缩使研究具有实际测量治理价值。

但是，v0 还不能唯一、无泄漏地实施。四个问题直接影响中心交付物：

1. `NOT_ATTEMPTED` 用原始 `frame_index < 4` 判定，在已列入覆盖面的 OnSite 产物中可被证实地误标；
2. 计划已承认立项数字扫描了 RQ007 held-out，但仍没有裁定这对“sealed untouched / evaluate once”的后果；
3. `estimability_ledger` 没有冻结 observation unit、字段、主键和不变式，宽表中多视角/多窗口测量及 M3 多 nominal 重复行会产生非唯一计数；
4. case 聚合与“可估计性加权”未冻结，且被引用的 RQ007 公式硬编码 `K=7`，与本计划允许逐行 K 变化不相容。

因此，本 Reviewer 不认为 v0 已满足其自定的“无 blocker 方可进入 Formal G1”条件（计划 `:157-160`）。

## Who would be interested in the results, and why

- 自动驾驶与人–机交互研究者：本研究能区分“数值已存在”与“候选偏好在当前网格下真正被区分”。
- 逆向规划、参数识别和不确定性研究者：结果可作为“估计器集中度不等于观测误差”的实例性测量审计。
- RQ003/RQ009/RQ010B/RQ011B/RQ012B/RQ014 的下游分析者：资格矩阵可防止在选择机制不明时直接“过滤后重跑”。

当前材料能证明领域内的治理价值，但尚不能证明具有跨学科的杰出科学重要性。

## Major strengths

1. **研究问题单一且结论边界清楚。** 计划将 RQ015A 定义为测量学描述研究，不改数值、不提行为学主张（`:16-27`）。
2. **对可识别性代理的语义边界较诚实。** 计划明确 `ipv_error` 派生量是 RQ007 的 identifiability proxy，不是 IPV 估计误差的直接度量（`:55-70`）；RQ007 冻结决策也明确保留这一限定（`reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/decision.md:28-30`）。
3. **如实保留 D1/D2/D3 不可区分的负结果。** 计划没有用未保存的 `min_mse` 猜测零值机制（`:29-32`）。
4. **产物可行性不被简化为“都能直接回填”。** L1–L4 分层可容纳 provenance 重建、移交 RQ015B 和真正无法恢复的产物（`:80-90`）。
5. **下游筛选的选择偏差被正面提出。** 计划没有预设“只保留可估计帧就一定更真实”，而是要求每个下游 RQ 评估暴露定义变化与选择偏差（`:115-129`）。

## Major concerns

### B1 — `NOT_ATTEMPTED` 的序号定义在 OnSite 上会系统性误标（blocker）

计划把 D0 冻结为 `frame_index < MIN_OBSERVATION`（`:61-70`），同时明确要覆盖 RQ012B OnSite（`:89-90`）。但 OnSite 生成链的估计器是对“筛出交互对象后的保留序列位置”从 0 开始填充前四行，而交付表再写回原始帧号：

- 生成脚本在原始序列上 `enumerate` 后，会跳过当前交互对象不存在的帧：`reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/hpc_onsite_ipv/build_onsite_m3_anchors_hpc.py:494-532`。
- `estimate_ipv_pair` 对保留后 dataframe 的行序列执行 `min_observation=4`：同文件 `:713-736`。
- 输出却保留 `row.frame_index`，而不保存估计器局部序号：同文件 `:1045-1089`。

现场只读查验 `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_ipv_timeseries_multi_allvalid.parquet` 可见，例如 `onsite:beijing:T17:A1:native_case:2344` 的前四个 D0 占位位于原始 `frame_index=101..104`，数值为 `IPV=0, ipv_error=1`。按 v0 规则，它们不会被先分流为 `NOT_ATTEMPTED`，而会落入有效判定并被误记为 `NOT_ESTIMABLE`。

**必须关闭：** 将 D0 定义为估计调用内的 `sequence_position < min_observation`，并为每个 `case_key × perspective × estimator_configuration` 冻结序列排序键、重复/断帧处置和 provenance。不得通用地把源 `frame_index` 当作估计器局部序号。

### B2 — RQ007 held-out 的“sealed untouched”声明已与现有扫描事实冲突（blocker）

v0 明确写明其表中数字是“全语料口径，含 RQ007 封存集”（`:34-53`）。被引用的扫描脚本只排除 `frame_index<4`，没有读取 fold 归属，也没有 held-out 排除（`reports/plans/prompts/RQ015_portrait_scan_v1.sh:1-18`）。

这与 RQ007 冻结合同不一致：

- `split_freeze.json` 把 `held_out` 定义为合同冻结后“single confirmatory evaluation / evaluate once only”：`reports/studies/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_process/00_meta/split_freeze.json:99-114`。
- 已接受决策仍声明 held-out “untouched”，且确认 19,258 / 7,628 / 11,342 边界：`reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/decision.md:1-7`, `:18-30`。

v0 的“从最终报告中剔除”能防止进一步泄漏，但不能把已发生的全集端点扫描变回“untouched”。

**必须关闭：** Formal G1 前应完成一项显式 PI/治理裁定：记录已读的端点、判定 RQ007 held-out 是否已被污染，并更新其未来确认性资格。RQ015A 运行应直接钉死现有分割产物 `case_split_assignment.csv` 及 SHA-256 `90d8bb91e68f9b5e0596cf1ae915eb22b01a5c4ccffbad00c0b446efa46d537d`，将其实际值 `held_out` 显式映射为 sealed，并对所有 InterHub/RQ009 表执行 `0 held_out IDs after join` 的硬失败检查。不应再提供“定位或重建”两种自由路径，因为现有已冻结归属表可用（计划 `:74-78`）。

### B3 — `estimability_ledger` 没有可执行 schema，中心观测单位不唯一（blocker）

计划把问题表述为“每一行到底有没有测出 IPV”（`:16-21`），要求“schema 先冻结”（`:131-142`），但文件中没有字段表、主键、状态集、去重规则或不变式。这不是局部实现细节，因为已点名的产物根本不共享“一行=一次测量”的语义：

- InterHub 时序一行含两个 agent 视角；现行估计器也把两视角写回两列：`src/sociality_estimation/core/ipv_estimation.py:334-352`, `:355-372`。
- OnSite 一行同时含 `ego/counterpart × hw10/hw4` 四套 IPV/error：`reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/hpc_onsite_ipv/build_onsite_m3_anchors_hpc.py:1083-1089`。
- RQ009 feature matrix 同时有 current counterpart、future target 和 M4-only current ego，它们是不同时点/用途的测量：`reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/03_features/feature_dictionary.csv:52-60`。
- 现场只读查验 M3 test predictions 发现 3,811,698 行只有 1,270,566 个唯一 `case_key × anchor_frame_index × perspective` 测量：同一 `y` 因 80/90/95% 三个 nominal 水平恰好重复 3 次。如果按 predictions 物理行计数，画像会被三倍重复加权。

**必须关闭：** Formal G1 前冻结 normalized ledger schema，至少包含：

`source_artifact_id/path_sha256`, `source_row_key`, `case_key`, `sequence_key`, `perspective`, `measurement_role` (current/future/M4-only), `estimator_version`, `sigma`, `history_window`, `min_observation`, `solver_mode`, `grid_id`, `K`, `K_source`, `source_frame_index`, `sequence_position`, `timestamp`, `ipv` (rad), `ipv_error` (dimensionless), `k_eff` (effective candidates), `attempt_status`, `estimability_label`, `feasibility_level`, `reason_code`, `rq007_split`, `sealed_excluded`, `schema_version`。

必须冻结唯一主键，且将 L1–L4 的“恢复可行性”与 `NOT_ATTEMPTED/ESTIMABLE/WEAK/NOT_ESTIMABLE/UNKNOWN` 的“测量状态”设为正交字段，不得在一个 label 列中混用 `pending_RQ015B`、`unknown` 和三态标签（计划 `:80-87`）。

### B4 — case 聚合规则及 K-aware 加权未冻结（blocker）

计划要求帧级和 case 级画像并列，并称将“沿用并扩展” RQ007 加权方案（`:92-105`）。但：

1. “case 至少 1/5 个可估计值”没有说是对两个 perspective 合并、对每个 case-agent 分别计算，还是要求两个 perspective 都达标。对冻结 hw4 产物的只读复算，合并两视角时“至少 1 个”为 96.01%，但 case-agent 口径为 90.72%；两者不能互换。
2. RQ007 原加权公式硬编码 `C_max = 1 - 1/sqrt(7)`，再以 `max(C_max-c_i(t),0)` 加权：`reports/studies/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_process/07_summary_sensitivity/summary_sensitivity_method.md:19-31`。本计划却允许逐行 K 变化或 unknown（`:55-64`）。如直接沿用 K=7 权重，K=5 的完全均匀行仍会得到正权重，与“对不可估计行降权”的语义冲突。
3. 计划没有定义零个 `ESTIMABLE` 帧的 case 如何交付“仅可估计帧”摘要，也没有定义加权分母为 0、窗口并存、不规则采样与各 case 时长不同时的处置。

**必须关闭：** 冻结至少三层单位（measurement、case-perspective-configuration、case-configuration）和每层分母；分别报告两视角，然后才可给出显式定义的 pair-level 聚合。加权必须冻结为 K-aware 公式并定义 K unknown/零分母行为，或者把加权对照限定在有证据的 K=7 InterHub 产物。

### M1 — K/K_eff 数值合同不完整（major）

核心公式维度一致：`ipv_error` 无量纲，`K_eff=1/(1-ipv_error)^2` 是有效候选数，也无量纲。但以下边界没有冻结：

- K=7 时 `K_eff >= 0.93K` 的精确 error 阈值是 `0.6080690991651898`，而不是文字可被按字面实现的 `0.61`（计划 `:61-64`）。对冻结 hw4 表，两者会相差 52,776 个 agent-frame（0.7448 percentage point）。执行应始终计算 K_eff/精确边界，不应使用四舍五入快捷阈值。
- K<=4 时 `K_eff<=4` 与 `K_eff>=0.93K` 发生重叠；现行 API 又允许任意 `candidate_ipv_values`：`src/sociality_estimation/core/agent.py:156-161` 与 `src/sociality_estimation/core/ipv_estimation.py:181-195`。应冻结支持的 K 集合（当前证据主要为 5/7），其他网格标为 unsupported/unknown，或明确优先级。
- 应冻结 `ipv_error` 为 NaN/inf、超出理论域 `[0, 1-1/sqrt(K)]`、K 缺失/非整数、以及浮点容差时的 fail-closed 标签。

### M2 — 立项数字缺少产物/估计器版本限定（major）

v0 把 41.2794% 零值、24.1688% `error<=0.50` 和 52.5810% `error>=0.61` 写成“全语料口径”（`:34-53`），但这组数字精确匹配 RQ009 `history_window=4` 重估时序，并不代表计划所列入的所有产物。现场只读对照显示：

- RQ009 hw4：`|IPV|<1e-9 = 41.2794%`，`error<=0.50 = 24.1688%`；
- 原 sigma=0.1 主时序：`|IPV|<1e-9 = 35.8447%`，`error<=0.50 = 41.9411%`。

这是研究对象之间的真实差异，不是可忽略的文字问题。RQ009 hw4 产物已有可钉死的 manifest 和字节哈希：`data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/target_hw4/target_hw4_manifest.json:1-41`。

**要求：** 将 §3 降级为“冻结 RQ009-hw4 产物的立项基线”，并对每个覆盖产物分别钉死路径、SHA-256、估计器版本、sigma、history window、K/grid 和测量角色。禁止把一个 hw4 画像描述为跨产物的“全语料”事实。

### M3 — “什么预测了可估计性”仍是分析题目，不是可复现 SAP（major）

计划列出了候选预测子并说明“探索性、不下因果结论”（`:107-113`），这一边界是正确的；但仍缺少：

- primary outcome（三态 ordinal、`ESTIMABLE` vs rest，还是 K_eff 连续值）；
- 预测时间支撑（仅使用当前/历史轨迹，还是允许全 episode 特征）；
- case-level blocked 拟合/验证、重复帧的聚类不确定性、case-balanced 权重、类别不均衡和缺失值规则；
- dev 与 guard 的角色，以及多预测子探索的选择/多重性处置；
- 效应量、区间、最小层样本量和不支持结论的判定。

**要求：** Formal G1 前冻结一个有界 SAP。一个可接受的轻量方案是：以 K_eff 连续值和预先定义的 `ESTIMABLE` 指示为两个描述性终点，仅在 RQ007 development 拟合/选择，guard 做一次 case-blocked 复现，按 case 聚类给出效应量和区间，sealed 不读；“候选轨迹可分离度”若未存储则直接归 L3/pending_RQ015B，不在 RQ015A 临时重算。

### M4 — C0 下游资格矩阵的结论准则不可复现（major）

计划要求为六个 RQ 输出 `可重估 / 需重新设计 / 不适用`（`:115-129`），但未定义这三个结论的必要/充分条件。不同执行者可以对同一个“筛选后改变暴露定义”做出不同分类，而验收只检查“逐项有结论”（`:140-142`），不检查结论是否按同一规则产生。

**要求：** 为矩阵冻结 decision rubric，至少包含 `downstream artifact/version`, `analysis unit`, `IPV role`, `ledger coverage`, `K known share`, `estimability positivity by required stratum`, `estimand/exposure changed?`, `selection-outcome dependence assessable?`, `required sensitivity/control`, `evidence file:line`, `qualification`, `reason_code`, `not_assessable/missing evidence`。“可重估”不应仅表示代码能跑；它应表示现有设计在重新定义可估计集合后仍能识别目标估计量。

### m1 — 画像的不确定性和分母报告规则未写入验收（minor）

帧级样本数极大，但帧在 case 内强相关。计划要求“独立复核”双口径画像（`:131-142`），却没有要求逐表报告分母、case/perspective/configuration 数、缺失与 unknown 比例，也没有要求 case-clustered 区间。建议把这些列入所有主表/主图的硬验收项，并明确帧加权与 case 等权的差别。

## Technical failings that must be addressed before the case is established

| ID | Severity | 技术失败 | 可验收关闭条件 |
|---|---|---|---|
| B1 | blocker | D0 使用源 `frame_index`，OnSite 可证实误标 | 冻结 `sequence_position` 及源别排序/断帧合同；以 OnSite 首四保留行测试强制 D0 |
| B2 | blocker | 已有全集扫描与 sealed untouched 冲突 | PI 裁定污染后果；钉死已有 split 文件/哈希；所有运行在解析 IPV/error 前先做 case allowlist，最终 `held_out=0` |
| B3 | blocker | ledger 没有唯一 observation unit/schema | 冻结 schema、主键、去重和不变式；M3 三 nominal 行只产生一个 target 测量标签 |
| B4 | blocker | case/perspective 聚合和 K-aware 加权不唯一 | 冻结三层分母、K-aware 权重、零分母/零可估计帧处置，并用手工 fixture 验证 |
| M1 | major | K/K_eff 精确边界、支持 K 域、invalid/unknown 未定义 | 冻结数学域、容差和 fail-closed 状态；边界测试不使用 `0.61` 快捷值 |
| M2 | major | hw4 基线被表述为跨产物“全语料”事实 | 逐产物冻结 manifest、estimator lineage/configuration 和单独画像 |
| M3 | major | 预测子探索无终点、split、聚类不确定性和选择规则 | 冻结有界 SAP，dev 探索 / guard 一次复现 / sealed 不读 |
| M4 | major | 下游资格三分类没有 decision rubric | 冻结证据列、必要条件、`NOT_ASSESSABLE` 路径和 reason codes |

## Assessment against Nature-style criteria

| Axis | Assessment |
|---|---|
| Originality | **Moderate, not yet demonstrated as a new scientific advance.** 将已存 `ipv_error` 系统地正规化为跨产物 ledger 及下游资格矩阵有方法学新意；但核心集中度构念和 episode-summary 敏感性已来自 RQ007（计划 `:67-69`, `:100-105`）。与现有文献/工程实践的新颖性差异不能从本计划判定。 |
| Scientific importance / significance | **High internal importance; broad importance not yet established.** 这项审计可能改变多个下游 RQ 对 IPV 的可解释边界，因而具有强领域内价值。但它不更新数值、不验证行为真值，也不直接证明部署性改进，应保持为有界测量学贡献。 |
| Interdisciplinary readership | **Potentially relevant but presently field-internal.** 参数可识别性、逆向规划和选择偏差具有跨学科共性；当前文本以 RQ 编号、D0–D3、M3 和内部产物为主，尚未将结论提炼为一个非专业读者可直观理解的一般测量问题。 |
| Technical soundness | **Not established.** 研究范围和语义限定是健全的，但 D0 可证实误标、sealed 治理未裁定、ledger 单位未冻结以及 case/K-aware 聚合不唯一，使中心交付物无法重现。 |
| Readability for nonspecialists | **Clear structure, insufficient standalone explanation.** 计划的“单一 RQ—边界—交付—验收”结构清楚；但 `ipv_error` 为何反映候选权重集中度、宽表如何展开为测量行、帧级与 case 级分母为何改变结论，均需一幅机制图和字段/单位示意图才能让非专业读者准确理解。 |

## Recommendation posture

**`BLOCKED / REQUEST_CHANGES`**

这不是对 RQ015A 拆分方向的否定；相反，回溯打标与估计器修复分离后，研究目标已经可管理。阻断来自实施合同还没有把这个目标变成唯一算法，以及已发生的 held-out 聚合扫描尚未被治理裁定。

建议以新的 v0.1/amendment 闭合 B1–B4 后重启同一双路复审。在那之前：

- `formal_g1_eligible=false`
- `execution_authorized=false`
- 不得产生最终画像或下游资格结论
- 不得触发 RQ007 held-out 的任何进一步 IPV/error 解析

## Risk / unsupported claims

- 不能由 RQ015A 区分 D1/D2/D3；计划已正确承认。
- 不能把 `ipv_error`/K_eff 类别表述为真实 IPV 误差或行为真值。
- 不能把 RQ009-hw4 的 41.2794% 零值基线泛化为所有估计器配置/所有产物。
- 不能在未冻结选择机制分析前声称“某几何/数据源预测可估计性”，更不能作因果解释。
- 不能在无资格 rubric 时把 `可重估` 当作下游结论仍有效的证明。

## Uniform post-review evidence challenge

### Challenge protocol

本节是在首轮报告冻结后接收的统一证据质询。本 Reviewer 仅重读了自己的报告，并查验指定的 RQ007 一手 binding contract 与 RQ015A 冻结计划；未读取 reviewer2/reviewer3 的任何文件。首轮原始记录保持不变：

- **Original verdict:** `BLOCKED`
- **Original counts:** `4 blocker / 4 major / 1 minor`

### Evidence adjudication

| Challenge evidence | Independent adjudication | Effect on findings |
|---|---|---|
| A/B: RQ007 把当前 IPV、concentration index、interaction opportunity、estimability 和 behavioural dynamics 分为五个不得混淆的概念；`g_i(t)` 是五条件的 conjunction，concentration-only 不得命名或解释为 estimability。RQ015A 却用 K_eff 单独三分回答“有没有测出 IPV”。 | **New blocker B5.** 首轮报告已识别“proxy 不是误差/真值”，但没有识别更严格的冻结命名禁令。这不是措辞软化就能关闭的问题；它改变中心构念、label 集、报告标题和 C0 矩阵所审计的暴露。 | 新增 `B5` blocker；技术 verdict 不变，blocker 数 +1。 |
| C: v0 的 cutoffs/先验数字来自含 sealed 的全语料，但同一计划又称找到 split 前不得冻结阈值、sealed 全程不参与任何阈值。 | **Upgrades the evidentiary basis and closure burden of existing B2; no new count.** 计划 `:52-64` 已经把全集画像和冻结 cutoffs 写入同一个基线，而 `:76-77`, `:140-142` 又声称事前 split/sealed-free 程序。即使 cutoffs 被称为 PI 政策而非数据校准，现有证据也不能证明它们未受已观测全集分布影响。 | B2 仍为 blocker，但关闭条件从“裁定是否打开”升级为“裁定污染 + 废止 sealed-free threshold provenance 声明 + 重建可审计阈值资格”。 |
| D1: M3 predictions 每个 anchor 有 3 个 alpha/nominal 行。 | **Corroborates existing B3.** 它证明 ledger 必须先归一化 measurement key，不能按物理 prediction row 计数。 | 不新增 finding，不改变 B3 severity。 |
| D2: OnSite D0 依赖 estimator-local sequence position。 | **Corroborates existing B1.** 源 `frame_index` 与估计器局部序号并非通用同义词。 | 不新增 finding，不改变 B1 severity。 |

### B5 — RQ015A 违反 RQ007 冻结的 concentration-versus-estimability 构念分离（new blocker）

RQ007 的 binding execution contract 并非仅要求“不把 `ipv_error` 叫作标准差”。它进一步冻结了五个正交概念：

- `theta_hat_i(t)` 是当前 IPV 估计，不是真实人格；
- `c_i(t)` 是七个候选权重上的无量纲 concentration/identifiability index；
- `o(t)` 是 interaction opportunity；
- `g_i(t)` 才是合同意义下的 estimability；
- `d_theta_i(t)` 是行为 IPV 动力学。

证据：`reports/studies/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_process/00_meta/binding_execution_contract.md:53-69`。

同一合同定义 `g_i(t)=1` 须同时满足 sustained low concentration、warm-up 排除、active opportunity、mechanical controls 存活和 case-health 合格（同文件 `:71-79`），并在 `:81` 明文禁止把 concentration-index-only 输出命名或解释为 estimability。

RQ015A v0 则以 K_eff/concentration 单独把每行命名为 `ESTIMABLE/WEAK/NOT_ESTIMABLE`，并用它回答“到底有没有测出 IPV”（`reports/plans/RQ015A_plan_v0_ipv_estimability_labelling_20260726.md:16-21`, `:55-70`）。这与所引用 RQ007 合同的禁令直接冲突。一行权重很集中，并不足以证明它处于有效交互机会、通过机械对照且 case health 合格；反之，权重近均匀也不能证明“没有测出 IPV”，只能证明当前候选网格/估计器的权重未集中。

#### Minimal closure condition for B5

**最小且不扩张 RQ015A 范围的关闭路径是重命名构念，而不是临时重建完整 `g_i(t)`：**

1. 将研究问题改为“旧产物每个测量行的候选 IPV 权重在当前网格下有多集中？”，不再回答“有没有测出 IPV”。
2. 将 `ESTIMABLE / WEAK / NOT_ESTIMABLE` 重命名为不含 estimability 语义的诊断标签，例如 `CONCENTRATED / INTERMEDIATE_CONCENTRATION / NEAR_UNIFORM_WEIGHTS`；`NOT_ATTEMPTED` 仍作为正交 attempt status。
3. 所有 artifact 增加 `concepts_measured`，仅声明 `current IPV estimate` 与 `IPV estimator concentration/identifiability index`；明文声明 `g_i(t)` **not measured / not reconstructed**。
4. 把“仅可估计帧”、“可估计性预测子”和“按可估计性筛选”分别改为“concentration-qualified frames”、“concentration 关联因素”和“concentration-based selection”；C0 矩阵只能评估这个更弱诊断的下游影响。
5. 不得再称 K 可变的三分阈值是 RQ007 已冻结的 `g_i(t)`；RQ007 合同中的 `c_i(t)` 明确基于七个候选权重（binding contract `:57-60`）。K=5 或其他网格的归一化是 RQ015A 新的诊断扩展，必须单独标记并按首轮 M1 关闭。

若作者坚持保留 `ESTIMABLE` 命名，则不再是最小 RQ015A：必须逐产物重建并审计 RQ007 `g_i(t)` 的全部 conjunction，无法重建的产物只能标为 `ESTIMABILITY_UNKNOWN`。

### Upgraded minimal closure condition for B2/C

1. 在任何新阈值冻结或 IPV/error 解析前，钉死现存 RQ007 split assignment 路径与 SHA-256，先建 dev/guard allowlist，以 fail-closed 方式拒绝 `held_out`、未匹配和重复 case key。
2. 删除 `:76-77` 中与已发生事实不符的“找到 split 前不得冻结”未来时叙事，不得在验收中声称 sealed “全程未参与任何阈值”（`:140-142`）。
3. 建立时间线/污染 ledger，记录全语料扫描发生在 cutoffs 冻结之前还是之后，以及 PI 是否看到分布后才定义 `K_eff<=4`/`0.93K`。证据不足时，必须默认 cutoff provenance **not sealed-independent**。
4. 对 RQ015A，将 RQ007 held-out 标记为已被该终点的聚合查看污染，不再将其保留为 RQ015A 的未来确认集。若 RQ007 治理层对“未分层聚合查看”有不同裁定，必须以显式 PI amendment 记录，不得由执行者默认。
5. 当前 cutoffs 只能被声明为“在已观测全集分布后冻结的 PI 诊断规则”，除非能提供早于全集扫描的带时间戳冻结证据。它们不得获得 sealed-confirmatory 标签。

### Final post-challenge recommendation

- **Final verdict after challenge:** `BLOCKED` — 与首轮相同。
- **Final counts after challenge:** `5 blocker / 4 major / 1 minor` — 新增 B5；B2 证据和关闭负担升级但 severity 原本已为 blocker；D 仅交叉确认 B1/B3。
- `formal_g1_eligible=false`
- `execution_authorized=false`

在 B5 未关闭前，即使 D0、ledger 与聚合的机械问题全部修复，RQ015A 仍不能以 `estimability` 的名义发布 K_eff-only 画像。
