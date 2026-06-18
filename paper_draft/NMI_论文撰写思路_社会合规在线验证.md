# NMI 论文撰写思路：自动驾驶社会合规的在线运行时验证

> 整合自 Overleaf 草稿（`main.tex` / `structure.md`）+ 两份证据报告（Round 5/6 整合稿）+ 你的国自然实车挑战赛数据集。
> 定位选择：**框架优先 + NSFC 外部验证（稳健路线）**。
> 目标期刊：*Nature Machine Intelligence*（IMRaD，摘要 ≤150 词，正文 ≤4000 词）。

---

## 0. 一页结论（TL;DR）

**一句话故事**：我们把"自动驾驶是否社会合规"从一个事后的平均分，重述为一个**在线运行时验证（runtime verification）**问题——从大规模人类交互数据中学到**状态条件化的经验社会规范带（norm envelope）**，在线判断 AV 的交互行为是否落在"人类胜任者在该状态下会做的事"的范围内；并用一个**独立的实车挑战赛**作为外部检验，证明"偏离规范带"确实对应"更差的、且常规安全检查看不见的"后果。

**当前证据能撑什么、不能撑什么**（两份报告已冻结）：

| 结论 | 证据强度 | 能否进主文 |
|---|---|---|
| 社会规范是**状态依赖**的经验带（priority−nonpriority IPV 随 PET 翻转 +0.058→+0.001→−0.034） | **强**，多重稳健 | ✅ 主文核心 |
| 动态 IPV（社会信号）**因果/在线估计**（仅用历史轨迹窗口，strict-online） | 强（已实现，report 低估） | ✅ 主文 |
| 验证器架构：在线查规范带 + 打偏离分 | 中（可行性，gap 仅剩"风险态索引"） | ✅ 主文（含诚实 runtime gap） |
| 状态条件验证器**预测性能全面优于** scalar baseline | 弱（增益微小：宽度 −2.1%、AUC 0.814 vs 0.797） | ⬇️ 降级为"规范参照/接口优势" |
| 在线 early-warning 高召回 | 弱（5%FPR 下 recall≥1s 仅 7.4%，AUC 0.648） | ⬇️ proof-of-concept |
| 闭环 planner 性能提升 / 真车部署 | **不支持**（只有合成违规 + 接口 demo） | ❌ future work |

**最关键的杠杆 = 你的 NSFC 挑战赛数据。** 它一次性补上当前稿件的三个致命空洞（见 §1）。整篇思路的重心，就是**把这套数据从"额外素材"提升为论文的第三幕和验证闭环**。

---

## 1. 诊断：梦想稿 vs. 证据稿的落差

你的 `structure.md` 写的是"梦想稿"：>95% 验证准确率、38% 碰撞风险下降、12% 效率提升、真车部署、社会合规轨迹重构闭环。但两份证据报告在 5 轮分析后**冻结的证据**是：只有 Claim 1（状态依赖）强，其余全部降级，没有闭环，违规标签是合成注入的。

这中间的落差，正是你说的"还没想好怎么验证和闭环"。审稿人会用三刀致命：

1. **社会合规没有 ground truth。** 安全有客观坏结果（碰撞），社会合规没有。你现在用**合成违规注入**来算 AUC——审稿人一眼看穿："你只是证明了对你自己造的扰动敏感。"（报告里 Figure 5 的 caption 已经自认这点。）
2. **没有真实被验对象。** InterHub 里 AV/HV 的划分被 Waymo 主导（占 61%、占 AV 案例 85%），你验证的几乎是"日志回放"，不是真正在跑的规划算法。
3. **没有"偏离→后果"的闭环。** 合成 counterfactual（触发率 3.0%→12.9%）只是接口演示，不是真实代价。

**好消息**：你说 NSFC 数据带有"排名 + 分场景评分 + 安全事件"。这三块标签**精确对应上面三刀**——它是真实算法（补②）、有独立外部后果标签（补①③）、且多个获奖算法可横向对比。下面整套思路就是围绕"用 NSFC 把这三个洞补上"来组织的。

---

## 2. 核心叙事重构：三幕结构

把你的直觉（"社会偏好可量化、可在线估计、人类模式可分析 → verification 框架"）落成清晰的三幕。每一幕回答审稿人的一个"凭什么"：

### 幕一 · 规范的存在性与可估计性（DISCOVERY，强证据）
**主张**：社会偏好（IPV）可量化、可在线估计，且**不是一个平均分，而是随 `risk × geometry × role × time` 改变的经验规范带**。
**证据**：InterHub 38,228 case；PET-gated priority 翻转（+0.058→−0.034）在 drop-Waymo、3/5/分位分箱下方向稳健；几何先验 4/4 数据源同向。
**回答的直觉**："社会偏好可量化、人类模式可分析。" → 这是你全篇最硬的地基。

### 幕二 · 在线验证器（METHOD）
**主张**：可以构造一个四层在线验证器——state recognizer → 经验规范带 → deviation scorer → planner interface。
**关键更正（report 低估了这点）**：**被监测的社会信号——动态 IPV——本来就是因果/在线的**。它只用一段*历史轨迹*窗口估计 ego 与对手的时变 IPV，不含未来信息，因此是 **strict-online**，不是报告里标的"conditional-online / 需证明 causal"。所以幕二不是"在线可行性 proof-of-concept"，而是"在线估计器已存在且因果运行"。
**重新切分 runtime gap**（别再笼统说"在线信息不完整"）：

| 量 | 用未来信息？ | 在线状态 | 用途 |
|---|---|---|---|
| **动态 IPV**（历史窗口估计 ego + 对手） | 否，纯因果 | ✅ **strict-online**（被监测信号） | 在线输入 |
| 风险态 PET（索引规范带） | 是 | ⚠️ 需 causal proxy（已恢复 ~55%、phase IoU 0.30） | 在线需替代 |
| 全窗 / case-mean IPV（建带标定） | 是 | ❌ offline-only | 仅离线标定 |

真正还需在线化的**只剩"用什么风险态去索引规范带"**，不是社会信号本身。
**必须处理的方法学陷阱**：规范带现在用**全窗 IPV** 标定，而在线比较的是**动态 IPV**（滚动窗口、更噪、早期未收敛）——拿苹果比橘子，偏离分可能被估计瞬态污染。**修法**：用同一个动态 IPV 过程、按匹配的"窗口长度/交互进度"在人类数据上建带，即**把规范带做成随交互进度变化的动态带**——正好用上状态空间里已有的 `time` 维度，动态 IPV(t) 对比进度 t 处的规范带，逻辑闭合。
**顺势加强**：你同时估计 ego 与对手的动态 IPV，正好接上 **P(IPV_i | IPV_j)** 条件规范——在线给定对手当前 IPV，检验 ego 是否落在条件合理区间；两个 IPV 都因果，准则完全在线。
**关键姿态**：剩余 runtime gap（风险态索引）仍**当作概念贡献而非缺陷**——"风险态在线不可完美观测，正是需要运行时 verifier 的理由"。
**回答的直觉**："社会偏好可**在线**估计 → 形成 verification 框架。"

### 幕三 · 外部验证与闭环（VALIDATION，★用 NSFC 数据）★全文新重心
**主张**：把验证器套到**实车挑战赛的获奖算法**上，用竞赛**排名 / 分场景评分 / 安全事件**作为**独立 ground-truth**，证明：(i) 社会合规偏离与独立评判一致（外部准则效度）；(ii) 偏离预测了**常规安全检查看不见**的失分与事件（区分效度 + 增量价值）。
**回答的直觉**："怎么验证、怎么闭环、相对安全验证的优势。"

> **一句话的范式转变**：从"自己造违规标签算 AUC"→"真实算法 + 真实独立后果"。这一步把稿件从"一个自洽但自证的框架"升级为"一个被外部世界检验过的框架"，是能不能上 NMI 的分水岭。

---

## 3. 逐一回答你的四个问题

### Q1 ——"怎么证明我的 verification 是准的？"

**先把难点说清楚**：社会合规是一个**潜在构念（latent construct）**，世界上不存在单一金标准告诉你"这次交互到底合不合规"。所以你**不能**像验证一个分类器那样"对答案"。当前用合成违规注入，本质是在回避这个问题，也是最大软肋。

**正确解法 = 构念效度三角验证（construct validity triangulation）。** 不靠单一金标准，靠四条腿互相印证——这恰恰是社会科学/心理测量学里验证一个不可直接观测构念的标准做法，写进 Methods 会显得方法论扎实：

1. **预测效度（predictive）**：偏离规范带是否**预测客观坏结果**（冲突升级、被迫让行、急刹、PET/TTC 下降、效率损失）。InterHub 和 NSFC 都能测。
2. **准则效度（criterion）★**：偏离分数是否与 **NSFC 竞赛的独立排名/评分**一致。**这是合成标签永远给不了的外部锚**，是 Q1 的最强答案。
3. **收敛效度（convergent）**：与一小批人类主观评分（"这个操作激进/得体吗"）一致。可选补充，成本低、加分高。
4. **区分效度（discriminant）**：验证器能捕捉到**安全指标捕捉不到**的东西（见 Q3）。

**写作建议**：在论文里**明确告诉审稿人**"我们不假装社会合规有金标准；我们用多源 convergent evidence 来建立构念效度"。这种诚实在 NMI 是加分项，而不是示弱。

---

### Q2 ——"如果一辆 AV 偏离了合理范围，会怎么样？"（后果链）

你需要一条清晰的**后果链**，三个层级递进，把"偏离"和"代价"焊死：

1. **预测性后果**：偏离在统计上**先于/预测**客观退化。用 InterHub + NSFC 量化（偏离分数高的窗口 → 后续冲突、急刹、被迫让行概率上升）。
2. **评价性后果 ★**：偏离 ↔ **更差的独立评判**（NSFC 排名更低、分场景评分更低、安全事件更多）。**这是把"社会偏离"翻译成"真实代价"的关键一跳**——而且代价是别人（竞赛评委/规则）打的分，不是你自己定义的。
3. **可执行后果（验证→动作的闭环）**：验证器输出 warning / soft-cost / fallback；在 **counterfactual 重规划**里把行为拉回规范带，展示结果改善。**用 NSFC 真实失分案例做 counterfactual**，比报告里现在的合成 demo 强一个量级。

**诚实边界**（一定要写进 Limitations，否则被打）：完整真车闭环仍是 future work；但"**真实算法的真实后果** + counterfactual 修正演示"已足以撑起"后果"叙事，不需要谎称做了闭环部署。

---

### Q3 ——"相对普通安全性检验，额外优势是什么？怎么体现？"

**这是全篇的 NMI 卖点，本质是区分效度（discriminant validity）。**

**对照基线（你 repo 里已经有 Pek et al. 2020 这篇在线安全验证作 foil，正好用上）**：

| | 形式化安全验证（Pek 2020 / RSS / reachability / CBF） | 社会合规验证（本文） |
|---|---|---|
| 问的问题 | "会不会碰撞 / 越硬约束？" | "在当前状态下，行为是否落在**人类胜任者的行为带**内？" |
| 触发时机 | 临近失败才触发（被动） | **更早**触发（行为还安全、但已偏离规范时） |
| 输出 | 二值（safe / unsafe） | **分级**偏离分 |
| 盲区 | 看不见"安全但反常"的行为 | 专门捕捉 **safe-but-bad**：安全但激进地抢 gap 逼别人让；安全但过度保守冻结车流 |

**怎么把优势"体现"出来（这是必须做的实验，不能只在 Discussion 里说）**：

1. **找出"安全验证通过、但社会验证报警"的案例与统计**，再证明这些案例**下游后果更差**（更多诱发冲突 / NSFC 排名更低 / 安全事件更多）。→ 这就量化了"安全之外的增量信息"。
2. **增量回归 / 部分相关**：控制住传统安全与规则合规指标后，社会偏离对"竞赛失分 / 安全事件"是否**仍有显著解释力**。有，就证明它不是安全指标的同义反复。
3. **lead-time 对比**：社会偏离预测下游冲突的提前量 vs. 纯安全监测器。
4. **定位措辞**：**互补而非替代**——一个新的监测层，捕捉形式化安全监测器在结构上看不见的失效模式（个体上无碰撞、却违反人类协商规范的行为）。

> 这条线一旦做实，审稿人问"这不就是换个名字的安全检查吗？"你有数据正面回答：不是，它在安全检查全过的样本里仍能预测真实的竞赛失分和事件。

---

### Q4 ——如何用 NSFC 挑战赛数据（你的杀手锏）

**它是什么**：多个获奖算法在真车挑战赛中的真实交互轨迹 + 排名 + 分场景评分 + 安全事件。

**为什么是杀手锏**：① 真实 AV 行为（非合成）；② 多个不同算法（横向对比）；③ 独立外部 ground-truth（排名/评分/事件）。这三点正好补上 §1 的三刀。

**三个核心分析（对应 Q1–Q3，建议各出一张主图）**：

- **A. 准则效度（→ Q1）**：每个算法的"社会合规偏离分数"vs. 竞赛**排名 / 分场景评分**。排名越高，是否越贴近人类规范带？散点 + 相关系数 + 分场景拆解。
- **B. 后果链（→ Q2）**：高偏离场景是否对应更多**安全事件 / 更低评分**。可做 per-scenario 的偏离 → 事件率曲线。
- **C. 区分价值（→ Q3）**：控制住传统安全/规则指标后，社会偏离对失分/事件的**增量解释力**；以及"安全过、社会报警"案例的下游后果。

**加分项**：你选了"还包含人类对照轨迹"——如果属实，直接把**获奖算法 vs 人类**放进同一规范带，画一条"谁更像人 / 更得体"的谱系图。这是极强的视觉证据，也直接呼应 Shirado et al. 2023（人机混合交通中的互惠）那条人因线。

**必须写清的边界**：NSFC 是受控赛道、算法/场景数量有限 → 把它定位成**独立外部验证集**，而非"大规模泛化证明"。和 InterHub 形成干净分工：

> **InterHub = 建立规范（establish the norm）；NSFC = 外部检验规范（test the norm against an independent world）。** 两套数据来源、采集协议完全独立，这种"建规范/验规范"分离本身就是强方法论卖点。

---

## 4. 修订后的逐节大纲（替换 structure.md）

对齐 NMI（IMRaD、摘要 ≤150 词、正文 ≤4000 词）与已冻结证据。**粗体 ★ = 用 NSFC 数据的新内容。**

### Abstract（≤150 词）
背景（AV 不只要安全，还要社会可理解）→ 问题（缺在线、状态条件的合规判定）→ 方法（state-conditioned 经验规范带 + 在线验证器）→ 结果（状态依赖强证据；**★在独立实车挑战赛上，社会偏离与竞赛排名/安全事件一致，且在安全检查之外有增量预测力**）→ 意义（面向人机共存的可审计运行时监测）。**删掉** 95%/38%/12% 这类未被证据支撑的数字。

### 1. Introduction（≤1500 词）
- 社会智能是 AV 安全之上缺失的一层（引 Shirado 2023）。
- Gap：现有评估是离线、总体、平均分；安全验证（引 Pek 2020 / RSS）只答"会不会撞"，答不了"在此状态下是否得体"。
- 贡献四点：(i) 把社会合规形式化为 runtime verification；(ii) 经验证明规范强状态依赖、envelope 可估计；(iii) 在线验证器 + 诚实 runtime gap；(iv) **★用独立实车挑战赛做外部验证，并证明相对安全检查的增量价值**。

### 2. Results
- **R1 社会规范是状态依赖的经验带**（InterHub，强）。PET-gated priority 翻转 + 几何先验 + 稳健性。→ Fig 2、Fig 3。
- **R2 在线社会合规验证器**（InterHub，方法 + 诚实 gap）。四层架构；**动态 IPV 因果/在线估计（strict-online，仅历史窗口）**；动态规范带（按交互进度匹配）+ P(IPV_i\|IPV_j) 条件准则；runtime gap 重新切分为"仅风险态索引待在线化"。→ Fig 1、Fig 4。
- **R3 ★外部验证：在 NSFC 获奖算法上，社会偏离 ↔ 独立排名/评分/安全事件**（NSFC，准则 + 预测效度）。→ Fig 5。**新核心结果。**
- **R4 ★区分价值与后果**：安全过、社会报警的案例及其下游代价；增量回归；counterfactual 修正 demo。→ Fig 6、Fig 7。

### 3. Discussion
- 把社会合规研究从离线评价推进到运行时验证；与安全验证互补的新监测层。
- Limitations（直接照搬两份报告里冻结的边界）：构念效度而非金标准；Waymo 主导；NSFC 是受控外部集而非泛化证明；synthetic-free 但样本有限；完整真车闭环属 future work。
- Broader impact：人机共存、可认证社会感知自治。

### 4. Methods
IPV 数学定义与**因果在线估计器**（历史窗口 → ego + 对手动态 IPV；明确窗口长度、是否用预测意图、收敛性/偏差刻画）；状态空间构建与 high-support cell 准则；**动态规范带**（按交互进度/窗口长度匹配标定，分层 partial-pooling + conformal——刻意与在线动态 IPV 同分布，避免全窗-vs-滚动失配）；deviation scorer 与 P(IPV_i\|IPV_j) 条件准则；**leakage contract（关键区分：动态/滚动 IPV = strict-online 允许；全窗 IPV / observed PET / phase 标签 = offline-only 禁止）**；**★NSFC 验证协议（如何对齐场景、计算偏离、与排名/评分/事件做相关与增量回归）**；planner interface 与 counterfactual 协议。

### Extended Data
数据清单与健康审计；完整状态空间支持矩阵与稀疏 fallback；baseline ladder + 负控制（shuffled-state ≈ scalar）；early-warning 阈值/lead-time 敏感性；跨源 LODO；**★NSFC 分算法/分场景细表**；over-conservatism 控制。

---

## 5. 主图计划（6–7 张主图）

| 图 | 内容 | 数据/来源 | 边界标注 |
|---|---|---|---|
| Fig 1 | 在线社会合规验证器框架（离线校准层 / 在线监测层 + runtime gap 标注） | 架构示意 | 非 formal proof |
| Fig 2 | 经验状态空间支持度 + 规范带 envelope | InterHub Round2 | observed PET 仅离线 |
| Fig 3 | **状态依赖翻转**（PET-gated priority + 几何先验，4 源 forest） | InterHub Round1/2，**强** | Waymo 梯度偏平 |
| Fig 4 | 在线验证器逐帧 deviation（**动态 IPV 因果在线**）+ 风险态索引可行性（55% 恢复、IoU 0.30） | InterHub Round2/4 | gap 仅在风险态索引，非社会信号 |
| **Fig 5 ★** | **NSFC 外部验证：偏离分数 vs 排名/分场景评分/安全事件** | **NSFC** | 受控外部集，非泛化 |
| **Fig 6 ★** | **区分价值：安全过但社会报警的案例 + 下游后果 + 增量回归** | **NSFC + InterHub** | 互补非替代 |
| **Fig 7 ★** | **后果与 counterfactual 修正 demo（真实失分案例）** | **NSFC** | 非真车闭环部署 |

> 与现有报告的 8 图相比：删掉"baseline ladder 作为主图"（降级进 ED，因增益微弱），把腾出的主文版面让给 NSFC 的 Fig 5–7。这是把"自证框架"改成"被外部验证的框架"的版面级体现。

---

## 6. 投稿前必做清单与风险预案

| 优先级 | 工作 | 解决什么 | 你是否已有数据 |
|---|---|---|---|
| **P0** | NSFC pipeline：把验证器套上获奖算法，算偏离 vs 排名/评分/事件（Fig 5） | 外部准则效度 = Q1 最强答案 | ✅ 已有 |
| **P0** | 区分效度实验：安全过但社会报警 + 下游后果 + 增量回归（Fig 6） | 相对安全验证的优势 = NMI 卖点 | ✅ 已有（+安全指标） |
| **P1** | NSFC counterfactual 修正 demo（Fig 7） | 后果链第三层 = Q2 闭环 | ✅ 轨迹已有 |
| **P1** | 用 predicted-risk proxy（APET/TTC-like）替换 observed PET 重跑在线验证器 | 解最核心 leakage blocker | 需补实验 |
| **P2** | 小规模人类主观评分研究（收敛效度） | 加固构念效度 | 可选 |
| **P2** | causal phase/conflict detector + alarm/event timestamp | 才能把 replay 升级为在线预警候选 | 需补实验 |

**审稿风险预案（高频拷问 → 你的答案）**：
- "社会合规没金标准，你怎么验证？"→ 构念效度三角 + **NSFC 外部排名锚**。
- "这不就是换皮的安全检查？"→ Fig 6 增量回归：安全全过样本里仍预测真实失分/事件。
- "合成违规标签不可信。"→ R3/R4 改用 **NSFC 真实算法 + 真实事件**，合成只留在 ED 做敏感性。
- "Waymo 主导、泛化存疑。"→ 明确把 InterHub 定位为建规范、NSFC 定位为独立外部验证集；不声称全局泛化。

---

## 7. 定位句、标题与摘要骨架

**一句话定位（cover letter 开头可直接用）**：
> Autonomous vehicles are certified for collision safety, yet they operate in human social traffic where competent behavior is contextual and implicit. We formulate social compliance as **state-conditioned online runtime verification**, learn empirical normative envelopes from multi-source human interactions, and—using an **independent real-vehicle challenge**—show that deviation from these envelopes tracks independent performance rankings and predicts safety-relevant outcomes that conventional safety checks miss.

**标题候选**：
1. *Online runtime verification of socially compliant autonomous driving*（稳）
2. *Social compliance is a state, not a score: runtime verification of human-aligned autonomy*（更有观点）
3. *Beyond collision safety: verifying autonomous vehicles against human social-norm envelopes*（突出与安全验证的对比）

**摘要骨架（填 NSFC 数字后即可用）**：
> AVs must be not only safe but socially intelligible. We recast social compliance as state-conditioned runtime verification: from N human interactions we estimate empirical normative envelopes over `risk × geometry × role × time`, and build an online verifier that scores deviation in real time. Social compliance is strongly state-dependent (priority-related preference reverses with collision risk, +0.058→−0.034, robust across sources and binnings). On an **independent real-vehicle challenge**, verifier deviation aligns with official rankings and per-scenario safety events, and retains predictive value after controlling for conventional safety metrics—evidence that social-norm verification is complementary to, not a relabeling of, collision safety. We explicitly quantify the online/offline runtime gap and position the verifier as an auditable monitoring layer toward certifiable, human-aligned autonomy.

---

## 附：你原 `structure.md` 需要改动的地方（一键对照）

| 原 structure.md | 问题 | 改成 |
|---|---|---|
| ">95% verification accuracy" | 无金标准，无法成立 | 删；改为构念效度 + NSFC 外部一致性 |
| "38% collision-risk reduction / 12% efficiency" | 来自不存在的闭环 | 删，或仅作 NSFC counterfactual 的"潜在改善"且严格限定 |
| "Real Vehicle Deployment（commercial AV）" | 你做的是回放/外部数据集，非部署 | 改为"独立实车挑战赛外部验证" |
| "Socially Compliant Trajectory Reconstruction"作为主结果 | 只有合成 demo | 降为 counterfactual demo + future work |
| Results 以"重构/真车"收尾 | 把最弱证据放最重位置 | 改以 **R3 NSFC 外部验证 + R4 区分价值**收尾（最强外部证据压轴） |
