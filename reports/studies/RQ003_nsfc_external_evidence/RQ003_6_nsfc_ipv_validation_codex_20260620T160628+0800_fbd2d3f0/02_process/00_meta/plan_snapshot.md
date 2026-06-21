# NSFC 实车挑战赛 · 自锚 IPV 验证器 — 冻结式确认分析规范
### (Frozen Confirmatory Analysis Specification, based on prior exploration)

> **本地过程文档（LOCAL ONLY）。不进 Overleaf（CLAUDE.md）。**
> 这**不是**严格意义的预注册：H1–H4 已受前期探索（评分级 150 结果、signed≈0.255 / abs≈0、
> safe-but-bad 分布）启发。本文件是在既有探索之上**冻结分析口径**的规范。
> 真正可算作**新确认性证据**的只有：①尚未计算的严格 rolling-to-rolling 条件 IPV；
> ②新产生的盲法行为标注；③冻结后才执行的 held-out 预测；④独立实现复现。
> 其余一律标 exploratory，不得包装成"未看结果先注册"。
> 日期：2026-06-19 · 配套：`paper/main.tex`、`methods_revision_memo_online_verifier.md`。

---

## 1. 目标与中心问题

论文把 NSFC 设为**外部验证集**，检验自锚 / conformal IPV 验证器是否与碰撞安全**互补、非重标**。
中心问题（不再问"谁离人类均值最近"）：

> **在状态条件化、对手条件化、严格同口径的动态 IPV 下，谁向"错误方向"且"持续地"偏离人类
> 条件规范；这种错向偏离能否解释已过碰撞安全检查的算法仍被判协同较差，并在原始运动学与常规
> 安全指标之外提供增量预测效用？**

只有过 Gate + 冻结主分析 + 红队 + 独立复现的结果，才建议写入 `main.tex`。

---

## 2. 既有数据事实（前期探索已确立，附来源，作 exploratory 先验）

- 10 队 × 15 场景 × 六维(0–100)=150 行；replay 仅 13–14 session（覆盖不全）；安全原语取自 8 份诊断
  PDF；对手为脚本 NPC（`mvSimulation`），**无人类对照**；全集仅 2 撞（均 A1）。
- safety 98.7% 顶格 vs **official coordination score** 2.7% 顶格（社会维未饱和）。
- coordination 是"状态"多于"特质"：方差分解 team 0.057 / scenario 0.443；队内 sd≈12.1 ≫ 队间 3.3；
  A–C 族秩相关 ρ=−0.04。
- signed 方向携带信号、绝对幅度不携带：场景内秩相关 signed≈0.255、abs≈0。
- safe-but-low-coordination 集存在：28 cell（safety≥95 且 coordination 落底 20%），C3/B1/A4/A3 高发。
- 综合分由效率支配：corr(综合,效率)=0.94 ＞ (综合,协同)=0.74。
- A1 是碰撞触发"安全否决"关（2 撞、全距 93.25）；连续交互场景 C4/B2/A4/B1 更适合算 IPV。

> 口径提醒：来源审计（Gate −1）确认前，全文一律称 **official coordination score**，
> **不**写 "expert-rated coordination"。来源 CSV 见 `ws2-outcomes-tierA/`、`ws3-stats/`、
> `competition_evidence_from_raw_data/01_results/`。

---

## 3. Gate −1 — 数据与 outcome 来源审计（最先做，硬门）

主统计单元写 150 个 `team×scenario`，但 replay 仅 13–14 session → 有效样本很可能 <150，且缺失可能与
团队/赛区/日志质量相关。**来源未澄清前不得进入任何 IPV 关联。** Gate −1 必须回答：

1. 每个官方评分 cell 能否**唯一映射**到 replay；
2. 实际可算 IPV 的 cell 数，及 `team × scenario × area` **覆盖矩阵**；
3. 缺失 cell 与已观测 cell 在 coordination / efficiency / team / area 上是否**系统性差异**；
4. coordination 究竟是**人工专家评分、规则公式、还是某些运动学原语自动产生**；
5. 北京/上海是否**同评分尺度、同评委、同 rubric**。

产出 `provenance_audit.md` + 覆盖矩阵 CSV。第 4 点确认前，outcome 一律称 official coordination score。

---

## 4. Gate 0 — IPV 测量口径审计（硬门）

> 未过 Gate 0 禁止任何 NSFC 关联/显著性搜索；失败则停 criterion-validity，转 domain-gap（Tier C）。

**4.1 动态同口径**：NSFC 与 InterHub 用**同一估计器、同窗 Δ、同采样率、同 progress 定义**；
**禁止** rolling-IPV 比 full-window mean envelope（rolling-to-rolling only）；Δ∈{短,中,长}预设敏感性，
主窗看 outcome 前冻结；报 rolling vs full-window 偏差/收敛/有效帧比例。

**4.2 形式化的规范与符号契约**（θ>0=prosocial）。验证用的**人类条件规范**（不自锚、不加护栏）：

```text
m(t) = Q_0.5( θ_ego | θ_npc, s, τ )                      # 人类条件中位数（s=state, τ=progress）
w(t) = max{ (Q_high − Q_low)/2 ,  w_min }                # 预冻结最小宽度，避免被极窄区间放大
competitive_shortfall  D_comp(t) = max(0, [Q_low(θ_npc,s,τ) − θ_ego] / w)
over_yielding_excess   D_yield(t) = max(0, [θ_ego − Q_high(θ_npc,s,τ)] / w)
```

canonical 单测（100% 通过）：明显抢行→D_comp 正确；明显过度礼让→D_yield 正确；同动作不同 risk/role
条件判定可不同；角色交换/镜像/时间截断后无符号翻转、无未来泄漏。

**4.3 三个概念的边界（执行前必须钉死）**：
- **self-anchor 定义**：ego 交互前/trailing 历史（仅用 t 之前信息）的因果早窗 IPV，决策时刻冻结。
  在**部署 verifier**（InterHub）里它只用于**收窄区间**；**外部验证里不以 ego 自锚定义"期望值"**
  ——期望值用**人类条件规范 m(t)**，否则会抹掉要检测的队间真实差异并自我洗白。
- **两路输出分离**：(1) **经验 verifier**：完全由人类数据估计的 D_comp / D_yield；
  (2) **安全策略 guard**：高风险下额外保守下限（情境下限）。**外部验证首先只测 (1)**；不得先加护栏、
  再把护栏产生的结果当"人类规范有效"。
- **conformal 边界**：阈值在 **InterHub calibration split 上冻结**。InterHub 人类轨迹与 NSFC 算法轨迹
  **不满足可交换性** → NSFC 上**不得宣称 nominal conformal coverage**，只报 empirical coverage / OOD /
  abstention；**不得**用 NSFC coordination 调阈值，**不得**用参赛算法自身分布重定义"正常"（否则把普遍
  非合规校准成正常）。

**4.4 支持/域外门控**：逐帧记 risk-proxy 置信、geometry/role 来源、progress 置信、InterHub cell
support、estimator uncertainty、fallback 层级。主分析只用达预注册 support 的帧；低支持仅 monitor-only。

**4.5 对手条件 IPV**：对 NPC 同样算动态 IPV，主检验用 m(t) 条件规范 + simultaneous-competition rate +
reciprocity mismatch + violation onset/persistence/AUC。这是区分"IPV 社会机制"与"一般运动学距离"的关键。

**Gate 0 验收**：符号单测 100% 通过；无未来信息入在线指标；rolling-to-rolling 完成；主分析帧
high-support 覆盖达预设线；marginal/conditional/scalar 三套可复算；产 `ipv_measurement_audit.md` +
`ipv_trace.csv` + `unit_test_results.csv`。

---

## 5. 假设（修订）

| 编号 | 假设 | 检验 | 对接 |
|---|---|---|---|
| **H1a** 域迁移审计 | 冻结的 InterHub envelope 在 NSFC 哪些 high-support 状态可用、哪些必须 abstain | OOD/support map；不是成功门槛 | R1-ext |
| **H1b** 状态条件化效度 | state-conditioned deviation 比 scalar deviation 更能对应独立 outcome | 条件 vs scalar 的 held-out 对比 | R1-ext / 准则 |
| **H2** 方向 > 绝对、条件 > 边际 | D_comp/D_yield 的关联与 CV > 绝对偏离；`P(θ_ego|θ_npc,s,τ)` > 边际 > scalar | 预注册收敛比较 | 准则效度 |
| **H3** 双尾后果机制 | competitive shortfall ↔ 抢行/逼迫/短时距；over-yielding ↔ 冻结/停走震荡；状态依赖非对称双尾 | 盲法事件 ↔ D_comp/D_yield | 后果链 |
| **H4** 容量匹配增量效用 | **在相同在线信息预算、容量匹配模型下，加入两个预注册 IPV 表征是否稳定改善 held-out 表现** | 见 §6 主比较 | 判别力 |

**关于 risk-gated priority reversal**：InterHub 的 `+0.058→−0.034` 是 **priority−non-priority 的 IPV
差值随风险反转**，不是"所有个体随风险的整体亲社会梯度"。NSFC 是 ego 算法 + 脚本 NPC，**不能用来重估
人类规范**，故 priority reversal 只作**方向性外部检查**，**不作必须命中的成功门槛**。

**关于 H4 的措辞（重要）**：IPV 是轨迹运动学的**确定性函数**，信息论上不可能含轨迹里没有的信息；足够
灵活的运动学模型理论上可重构 IPV。因此即便主模型⑦>④，也只能写
**"incremental predictive utility relative to the prespecified kinematic baseline"**，
**不得**写成 "new information beyond kinematics"。

---

## 6. 最小主分析（冻结，confirmatory family 仅此一组）

为防 150 cell 过拟合，confirmatory family **只含**下列一组；其余全部降级到 sensitivity/exploratory。

- **主样本**：成功映射、high-support、**非 A1**、collision-free 的 cell。
- **主 outcome**：去除 `scenario + area` 固定效应后的 **coordination residual**。
- **主 IPV predictor（双尾 AUC）**：conflict-window、time-normalized 的 **D_comp AUC** 与 **D_yield AUC**。
- **主比较（容量匹配的一次增量）**：
  `state + causal kinematics + safety`  vs  `state + causal kinematics + safety + D_comp + D_yield`。
- **主泛化**：leave-one-team-out；**次级**：leave-one-scenario-out；
  leave-one-family-out（仅 3 fold）只作**迁移边界描述**，不作显著性 headline。
- 其余（p90、max、onset、response latency/gain、不同窗口/phase、三套 safe subset 的全组合）→
  **sensitivity / exploratory family**，FDR 校正并标 exploratory。

---

## 7. 分析单元、端点、排除

- **单元**：主统计单元 = `team×scenario`（经 Gate −1 后的有效子集）；帧/交互级仅用于动态 IPV 与事件 onset。
- **端点**：Primary = §6 的 coordination residual；Secondary = 盲法行为标签（§8 W3）+ 连续 safety
  primitives；Exploratory only = comprehensive / area rank / overall rank。
- **A1/碰撞**：2 碰撞 cell 单列 catastrophic safety failure，不进连续主回归；主结果同时报"全样本/去 A1"。
- **safe subset（≥3 种，不依赖 outcome）**：S1 collision=0；S2 safety=100 且 collision=0；
  S3 collision/takeover/line-crossing=0 且 TTC/lateral-gap 过预设阈。主结论须 ≥2 种定义方向一致。

---

## 8. 工作流（W−1 … W7）

- **W−1 Provenance audit（= Gate −1）**：replay↔score 映射、覆盖矩阵、缺失偏差、coordination 来源、
  京沪 rubric 一致性。
- **W0 Freeze & spec**：写 `claims_register.md`（标 supported/exploratory/rejected）、
  `primary_endpoints.md`、`exclusion_and_safe_subset.md`、`ipv_sign_contract.md`；冻结 §6 主分析。
- **W1 IPV measurement audit（= Gate 0）**：执行 §4；失败转 domain-gap。
- **W2 Directional conditional IPV signature**：开放找"错向+持续"最稳模式（不看总排名）；预注册收敛比较
  absolute vs signed / scalar vs conditional / marginal vs counterpart-conditioned / max vs persistent-AUC。
- **W3 盲法现象学（两样本，避免选择偏差）**：
  - **机制样本**：极端分层抽样（高 D_comp/低 coord、高 D_yield/低 coord、反例、对照）→ 仅用于 case cards
    与发现抢行/冻结两类机制，**不估自然发生率**。
  - **验证样本**：从可用 replay **随机 / 场景分层随机**抽取（抽样概率已知）→ 用于正式 事件–IPV 检验。
  - ≥2 名盲标注者（不显队名/官方分/IPV）；标签：aggressive intrusion / appropriate assertiveness /
    over-yielding-freeze / oscillation / deadlock / smooth reciprocal negotiation / unrelated failure；
    报 inter-rater agreement。措辞：**"独立于官方分与 IPV 输出的盲法行为 criterion reference"**
    （非"完全独立真值"——标注者仍看与 IPV 同源的轨迹行为）。
- **W4 Incremental validity（容量匹配 ladder）**：①state ②safety ③kinematics ④state+kin+safety
  ⑤IPV ⑥state+IPV ⑦state+kin+safety+IPV ⑧IPV-removed/shuffled-IPV/wrong-state 负对照。判据见 §6 主比较。
- **W5 State dependence 与迁移边界**：风险/几何/角色/相位异质性；leave-team/scene/family；京沪分区；
  high-support-only vs fallback-inclusive；找"高效率但绝对偏离大、方向却合适"的反例。目标=识别 verifier
  在哪些状态可用、哪些必须 abstain。
- **W6 NPC quasi-controlled（次级，只匹配 exposure-onset 之前）**：**只能匹配**触发前初始位置/速度/姿态、
  脚本版本/seed/actor identity、**ego 尚未影响 NPC 之前**的状态。**禁止**匹配交互开始后的 NPC 前 3–5s
  轨迹（已是 post-treatment）。NPC 闭环响应 ego 时只称 **matched opportunity structure**，不称同刺激/因果。
- **W7 红队 / 复现 / story editor**：杀符号错、未来泄漏、proxy、评分循环、多重比较、NPC 伪因果、低支持强
  判违规；独立实现主指标与 CV；**只在 Gate 通过后**映射到正文。

---

## 9. 统计与验证协议

- **场景内主检验**：场景内 10 队 predictor↔coordination residual 秩关联 → 跨场景汇总（median
  scenario-wise Spearman、方向一致场景数、场景分层 permutation、cluster bootstrap over scenarios）。
- **交叉验证**：leave-team（主）/ leave-scene（次）/ leave-family（边界描述）；主指标 rank corr / MAE /
  CV-R²，看 ⑦ vs ④ 增量。注：leave-one-scenario 总排名 rho 0.976 系机械重叠，不作泛化证据。
- **置换/多重比较**：team n=10 用精确/MC label permutation，保留 scenario/area 结构；confirmatory family
  只含 §6；其余入 discovery family，FDR；null/反向入 `tried.md`。
- **负对照/反事实**：state shuffle、IPV time shuffle、counterpart swap、role flip、sign flip、
  wrong-envelope-cell、kinematics-only、IPV-removed；future-leaky full-window IPV 仅作 optimistic
  upper-bound，**不可作 deployed result**。
- **结果分层**：
  - **Tier A（进主文）**：方向化条件 IPV 在 safe subset 与 coordination/盲法行为一致，且跨团队/跨场景 CV
    对 kinematic+safety baseline 有稳定增量；过红队+独立复现。
  - **Tier B（进 ED/Discussion）**："interpretable diagnostic alignment"，不声称独立预测价值。
  - **Tier C（负结果/future work）**：只在单场景/单赛区/低支持有效，或被运动学解释；保留为 transfer
    boundary。

---

## 10. 与论文叙事对接（按 label 引用，编号待你确认）

> 同步文件 `paper/main.tex` 当前为 8 图：planner-demo=Fig 6、`fig:nsfc-validation`=Fig 7、
> `fig:discriminant`=Fig 8，"External validation"=第 5 个 Results 子节。你提到的 Fig 8/9、R6/R7 对应一个
> 我这里未同步到的新版本。本计划一律按 **label/角色** 引用，编号确认后再钉死。

- W1b/W5 → **R1 外部复现 / 状态依赖**。
- W2/W3/W4 → **Fig.`nsfc-validation`（准则效度 + 后果链）**。
- W4/W5 → **Fig.`discriminant`（判别力，最高价值点）**。

**预写主文句式（仅 Tier 达标才用，EN；注意已改 official coordination score）**：
- Tier A：*"In an independent real-vehicle challenge, global closeness to a human IPV median did not track
  overall ranking. State- and counterpart-conditioned directional nonconformity instead separated two failure
  modes—excess competition and over-yielding—and carried incremental predictive utility for the official
  coordination score among collision-free runs, relative to a prespecified capacity-matched kinematic and
  safety baseline."*
- Tier B：*"Directional IPV nonconformity aligned descriptively with the official coordination score but did
  not add robust out-of-sample utility beyond the prespecified kinematic baseline."*
- Tier C：*"The transferred human envelope did not validate on the challenge data, revealing a domain-alignment
  requirement for IPV-based runtime verification."*

---

## 11. 需同步修改 `main.tex`（正式分析前）

1. **主端点改写**：现正文"higher-ranked algorithms sit closer to the human reasonable interval"
   （L203 一带）已被本规范放弃，改为**方向化条件非合规 + 容量匹配增量效用**。
2. **修符号 bug（真正替换，不只留在计划里）**：θ>0=prosocial 时，δ=(θ−m)/w **正侧实为 over-yielding**，
   但现正文（L187–189 的 one-sided soft cost / L280 deviation score）对正侧触发并称"更竞争 violation"
   ——符号反了。用 **D_comp / D_yield 双尾**替换单侧 δ cost（Algorithm 2 与 Results「planner-facing」段）。
3. **编号**：按你的现稿把 NSFC→Fig 8/R6、discriminant→Fig 9/R7（我这边同步版仍是 Fig 7/8）；先确认再改。

---

## 12. 启动顺序 + 执行前四个硬修订（checklist）

启动顺序：Gate −1 → Gate 0 → 冻结 §6 → 并行重算 directional IPV / kinematics baseline / support →
盲法两样本 → leave-team/scene/family + 负对照 → NPC matching（不通过则不做因果）→ 红队+复现 → Tier 定位 →
最后才动正文。

四个硬修订（全部完成方可作正式执行蓝图）：
- [x] 命名改为**冻结式确认分析规范**（§标题/§1 已改）；
- [x] 加入 **Gate −1** replay 映射与 coordination 来源审计（§3）；
- [x] 明确 **self-anchor / conformal / guarded policy** 边界与两路输出分离（§4.3）；
- [x] 主分析压缩为**双尾 AUC + 一次容量匹配增量比较**（§6）。

---

## 13. Paste-ready briefs（精简）

**IPV measurement auditor（含两路分离与 conformal 边界）**
```text
不看 NSFC outcome。验证 NSFC 动态 IPV 与 InterHub 规范带可比。查：①θ 符号（>0=prosocial；
competitive/over-yielding 单测）②online leakage（仅 trailing history；禁 full-window/observed PET/
realized order/post-hoc phase）③rolling-to-rolling 同窗同采样同 progress ④人类条件规范 m=Q0.5(θ_ego|
θ_npc,s,τ)、w=max{(Qhigh−Qlow)/2,wmin}；经验 verifier(D_comp/D_yield) 与安全 guard 分两路输出
⑤conformal 阈值在 InterHub split 冻结，NSFC 不claim nominal coverage、不调阈、不用参赛分布重定义正常
⑥逐帧 support/fallback/uncertainty/abstention。产 ipv_measurement_audit.md/unit_test_results.csv/
ipv_trace_sample.csv。符号不一致或未来泄漏=blocking。
```

**Directional IPV experimenter（最小主分析）**
```text
confirmatory family 只含：主样本=mapped+high-support+非A1+collision-free；主 outcome=去 scenario+area
固定效应的 coordination residual；主 predictor=D_comp/D_yield 的 conflict-window time-normalized AUC；
主比较=state+causal-kin+safety vs +D_comp+D_yield（容量匹配）；主泛化=leave-one-team-out（次 leave-scene；
family 仅边界）。报 effect size、cluster CI、方向一致场景数、null/反向。其余指标入 exploratory+FDR。
```

**Blind phenomenology annotator（两样本）**
```text
只看匿名 replay（无队名/官方分/IPV）。两样本：机制样本=极端分层（仅 case cards/机制）；验证样本=随机/
场景分层随机（已知抽样概率，做正式 事件–IPV 检验）。标签：aggressive intrusion/appropriate assertiveness/
over-yielding-freeze/oscillation/deadlock/smooth reciprocal negotiation/unrelated failure。≥2 人报 agreement。
定位="独立于官方分与 IPV 的盲法行为 criterion reference"，非完全独立真值。
```

**Red team**
```text
攻击：δ 符号/单侧 cost 反了（正侧实为 over-yielding）；rolling 比 full-window envelope；IPV 只是
speed/TTC/stop 重编码（容量匹配 baseline 检验）；coordination 被效率/场景难度混淆；team/area 嵌套与帧级
伪重复；A1 zero-score veto；NPC 是否同刺激（post-treatment 匹配）；选窗口/阈值/样本是否看过 outcome；
低支持态强判违规；conformal 非交换性下宣称 coverage。攻击后仍成立才许 Tier A。
```
