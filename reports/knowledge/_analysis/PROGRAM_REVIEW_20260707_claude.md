# 研究进展全面分析与后续工作建议（程序级综述）

作者角色：Claude（research planner / reviewer / repository integrator）
日期：2026-07-07
性质：解释层综述与建议，**不引入任何新证据、不冻结任何新论断**。所有论断的权威来源仍是各 RQ 的 `decision.md`。
目的：以"更好地为论文研究提供有利证据"为目标，总结当前研究状态、分析证据结构、给出优先级排序的后续工作建议。

---

## 1. 一句话现状

v4.1 论文主线（"在线的、可估计性感知的、情境条件化的动态 IPV 校验器"）的**内部证据链已经闭合且质量很高**（RQ007 可估计性门 + RQ009 R3 情境条件化保形包络），但**所有外部验证面（WOD-E2E、OnSite）以及所有"IPV 特异性增量"检验均返回有界 null**。论文当前最强的正向证据全部来自 InterHub 内部；外部腿是"注册的负向/边界结果"。下一阶段的核心任务不是继续扩面，而是：(a) 用已具备条件的 RQ007 封存 held-out 确认把内部主线从 provisional 升级为 confirmed；(b) 按照"情境条件化包络 + 可估计性门控 + 弃权"的真实证据结构重述论文价值主张；(c) 用两个已识别的靶向杠杆（OnSite 交互失败段检索、E16 被动性→死锁定向检验）争取程序内第一个外部正向结果。

## 2. 当前状态总结（截至 2026-07-07）

### 2.1 证据链与各环节状态

主证据链（来自 `RQ_PROGRESS_DASHBOARD.md`）：

```text
在线 IPV 时间序列
→ 交互条件化可估计性        [RQ007 accepted，dev/guard 边界，held-out 封存]
→ 可估计性感知动态包络      [RQ009 accepted，R3 情境条件化保形包络]
→ OnSite 匹配场景效度       [RQ011 accepted 就绪；RQ011B closed-out 暂定null/欠识别；RQ012B accepted 有界null]
→ WOD-E2E 人类偏好效度      [RQ010 可行性 accepted；RQ010B 2026-07-03 完成 = 有界NULL，10Hz敏感性07-04关闭]
→ 超越安全基线的增量价值    [RQ013 planning，S0 占位]
```

### 2.2 已冻结的正向（可入稿）证据

| 来源 | 核心内容 | 边界 |
|---|---|---|
| RQ009-KC-R3 | 可估计性感知、情境条件化 split-conformal 动态包络：90% 下宽度 −42.3%、Winkler −35.6%、覆盖 ≈0.899、弃权 4.78%；排除观测 risk/PET，支持域外弃权 | InterHub σ=0.1；经验监视器非形式证明；LODO 覆盖 0.749–0.991（迁移非无条件）；目标零原子 ~21.6% |
| RQ007 C1–C3 | 可估计性是交互条件化的（总差 ≈−0.13，其中邻近性 ≈−0.096，冲突几何残差 −0.032~−0.036 且 CI 不含零）；可估计性≠行为稳定；episode 汇总规则依赖定义 | 仅 development(19,258)/guard(7,628)；sealed(11,342) 未触碰 → 措辞必须 provisional |
| RQ011 | OnSite 全域就绪 `READY_WITH_FROZEN_EXCLUSIONS`：full_300 结果 + clean_285 回放（T19 回放排除） | 仅范围/就绪决策，无 run 级/因果论断 |
| RQ001（legacy） | 严格前缀 map-lane 因果滚动 IPV + split-conformal 是最锐的在线区间且唯一达 ~0.90 LWO 覆盖（0.902/0.628）；conformal 必要（raw ~0.86 欠覆盖） | 工程先验 / M4 消融边界，不建立新 M3 规范 |
| RQ004 | IPV 是 风险×几何×角色×时间 的状态条件化响应面，非全局标量；粗糙路面几何是稳定先验 | episode 级组织性描述 |
| RQ005 | 运行时验证框架 = 经验/概率监视器（四层）；泄漏契约（观测 PET、实现顺序、事后相位等仅限离线）是主要贡献 | 治理性论断；planner 接口仅 n=6 演示 |
| RQ002 | 自锚必要（disposition 增量 R²≈0.45）但**不足以**构成规范权威；norm laundering 实在（不良子集富集 ≈1.507×）→ 混合设计方向 | 证伪/边界性接受 |

### 2.3 系统性 null 模式（程序中最重要的经验事实）

五条相互独立的证据线在"IPV 特异性外部增量"上全部返回 null 或有界 null：

| 线 | 结果 | 关键数字 | 功效/测量边界 |
|---|---|---|---|
| RQ003 NSFC | 无稳健 IPV 特异性增量（Tier B 边界） | — | 先验 null |
| RQ008 InterHub 时序发现 | 负向边界：0/24 方向性结构幸存控制 | — | 发现层，确认层未开 |
| RQ009 M3/M4 内部消融 | IPV 条件化通道无增量：M3≈M2（配对 90% Winkler 差 −0.0002，p=0.863）；M4 ≈ −2% 宽度 | 内部消融，非手稿论断 | InterHub 内部 |
| RQ010B WOD-E2E 偏好 | 候选 IPV 不预测人类偏好、不及物理特征（IPV strength=0 vs 物理 ρ≈0.16–0.26；max-stat p=1.0）；4Hz→10Hz 敏感性关闭（null 保持） | S1 n=75 ρ=0.148 p=0.10；S2 n=98 ρ=0.031 p=0.69 | N 上限 75–98，\|ρ\|<0.28 以下欠功效；对手车相机测距 ~1.2 m；M3 不迁移（≤15% in-support） |
| RQ011B OnSite 时刻级监视 | PROVISIONAL_NULL / UNDER_IDENTIFIED（测量受限） | C2 AUC=0.493；固定警报 54.2 次/交互分钟 @ recall 0.20 | 主对照 C1 对照数 = 0；交互失败段检索/分割不足 → **不是干净否证** |
| RQ012B OnSite 偏差→危害 | 全行为电池 BOUNDED，0 个 SUPPORTED 端点 | near-miss IRR 1.2239 [1.031,1.345] p=0.0018 q=0.051 但败于 M2（情境可解释）；E16 死锁 IRR 1.4967 p=0.0026 q=0.051 通过全部对照但欠功效 | n=245 单元 / 19 队；聚类置换 + BH-FDR |

### 2.4 手稿与治理状态

- 论文基线：paper 仓库 `main` 合并提交 `c6783577`（v4.1，estimability-aware dynamic norm 叙事；`structure.md`/claims register 同步）。手稿仓库不在本工作区，本报告未直接核验其当前内容。
- PAPER002 ledger（`decision.md`/`synthesis.md`）**仍将 R4 WOD-E2E 腿标注为 pending RQ010B**——已滞后于 2026-07-03 的 RQ010B 有界 null 决策，需要更新（见 §5 建议 B）。
- 治理同步缺口（已核实）：`RQ_PROGRESS_DASHBOARD.md` 头部"Last synchronized: 2026-06-29"，且其 blocker 清单仍称"RQ001/002/004/005 缺 accepted decision.md"——但四者的 `decision.md` 均已于 2026-06-24 人工冻结为 ACCEPTED；`STUDIES.md` 与 `rq_progress_registry.csv` 对应行也仍是 `review`。
- 工程侧：IPV 估计器 sigma01 数值兼容性问题已通过 `solver_mode {exact(默认)/fast/realtime}` 修复（恢复 sigma01 生成一致性）；HPC 上 RQ010B 全管线保留于 `/ZXC/RQ010B_wod_e2e/`。
- 明确暂停项：RQ008B 不执行；RQ012 双人标注弃用；RQ007 sealed 未开。

## 3. 关键分析

### 3.1 证据结构是"内强外空"，论文叙事必须与之对齐

程序的正向证据（锐利、近名义覆盖、可估计性门、弃权机制、泄漏契约）全部建立在 InterHub 内部；两个外部面交付的都是注册边界。这不是执行失败——每个外部 null 都经过预注册、独立评审、红队、盲复制，方法学质量高——而是证据的真实形状。对 NMI 级投稿而言，这个形状支持的论文是："一个诚实的、带弃权与可估计性门控的经验运行时监视器 + 严格的外部效度边界地图"，而不是"IPV 校验在外部面被验证有效"。任何暗示后者的措辞都会在评审中被 RQ010B/RQ012B 的注册 null 直接反驳（且这些 null 已在仓库/报告中留痕，不可撤回，也不应撤回）。

### 3.2 五重 null 的两种解释仍未被区分，这是论文的软肋也是机会

现有数据无法区分：(i) 标量 IPV 在外部结果上确实没有特异性信号（真 null）；(ii) 测量链衰减（相机测距 ~1.2 m、M3 支持域 ≤15%、OnSite 失败段检索不足、N≤98）把真实效应压到功效线以下。每条 null 各自的功效/测量边界（§2.3 右列）在各 decision.md 中记录良好，但**缺一份跨 RQ 的统一衰减/功效综合分析**：把每条线"可检出的最小效应量"与"测量误差预算"放进同一张表，回答"若 IPV 特异性效应存在，其上界是多少"。这份分析成本低（全部输入已在库中）、可直接进补充材料，并把审稿人问题"你们为什么相信这是边界而不是构念失败？"转化为定量回答。建议作为一个小型分析工单执行（见 §5 建议 F）。

### 3.3 最便宜的强正向证据就在手里：RQ007 封存 held-out

RQ007 的 C1–C3 目前只能以 provisional 措辞入稿，因为 sealed 分割（11,342）未开。PI 早前设定的开启前提是"RQ009 完全冻结"——该前提自 2026-06-29 起已满足（RQ009 accepted，M3-vs-M4 无升级，下游 RQ010B/011B/012B 消费也已完成）。开启是一次性、不可逆的，因此顺序必须是：先冻结确认协议（端点、阈值、通过/失败判据、以及 R3 包络是否同时在 sealed 上做确认性重估），独立评审协议，然后请求 PI 的单独开启决策。若确认成功，论文内部主线（R1/R2/R3 对应的可估计性 + 包络）从"开发/守卫集 provisional"升级为"预注册封存确认"——这是当前所有可选动作中证据增益/成本比最高的一项。风险对称性也可接受：若确认失败，越早知道越好，避免在 provisional 措辞上继续加码。

### 3.4 两个已识别的外部正向杠杆，都是靶向而非扩面

1. **OnSite 交互失败段检索/分割（RQ011B 的注册阻塞）**：RQ011B 的 null 被明确标注为"测量受限、非干净否证"——主对照组为 0 使 C1 无法判定。一个专门 RQ（如 RQ014）解决失败段检索/分割后重访监视器判定，是 OnSite 面唯一被治理记录认可的重启路径。它同时服务两头：若重访转正，得到程序第一个外部正向结果；若仍 null，把"测量受限"升级为更干净的否证，边界叙事更硬。
2. **E16 被动性→死锁的定向功效化检验（RQ012B 的 bounded hint）**：E16 是全电池中唯一通过全部对照（含 placebo/label/M2/exposure）的通道，仅因欠功效 + BH 边缘而未 SUPPORTED；且 Fig.4 显示 9/10 后果中过度被动尾部偏差大于过度激进尾部——方向一致。一个预注册的、单端点的定向检验（不再将 α 摊薄于全电池多端点）是低成本的确认机会。注意诚实性约束：这是 RQ012B 明示允许的"future powered test"，不是 p-hacking 式重测；功效来源须是新增单元/新赛季数据或预注册的单端点设计，而非同数据重算。

### 3.5 RQ013 若按原定义执行，先验上大概率再产出一个 null

RQ013（"IPV 校验器相对安全/运动学基线的增量"）的直接前身证据——RQ003 增量 null、RQ009 IPV 通道 null、RQ012B 偏差→危害 null、RQ010B 偏好 null——给了它一个很差的先验。按"IPV 增量"定标执行几乎注定复制 null 模式。建议重新定标为**包络效用**：情境条件化包络（而非 IPV 通道）相对预先指定的运动学基线，在锐度、弃权质量（弃权时段的风险富集）、OnSite 官方结果分层上的增量。这与已接受的 R3 论断同构，是程序里唯一有正向先验的增量问题。

### 3.6 预期审稿风险清单

| 预期质疑 | 现有弹药 | 缺口 |
|---|---|---|
| "IPV 条件化无增量，为何还叫 sociality verification？" | RQ005-KC-FRAMING：验证对象是估计出的 IPV 相对人类包络的偏差，不是 IPV 的预测增量；RQ007 给出何时可估计 | 论文措辞需系统排查，避免残留 v3 自锚叙事 |
| "外部效度呢？" | 两个注册边界 + 各自的测量/功效上界 | §3.2 的统一衰减分析尚未做 |
| "包络只是情境统计，无社会性内容？" | RQ004 状态条件化响应面 + RQ002 laundering 证伪链条说明为何必须情境化 | 可在 discussion 中显式串联 |
| "provisional 措辞太多" | — | RQ007 sealed 确认（§3.3）是唯一解 |
| "σ=0.1 参数敏感" | RQ006 鲁棒性附录 | 已足够，保持引用 |

## 4. 有价值的补充观察

- **null 结果的资产化**：本程序对 null 的治理（预注册、红队、盲复制、注册边界）本身达到发表级方法学标准。无论主线论文如何，RQ010B（相机-only 追踪管线 + 偏好 null + M3 迁移失败的 OOD 分析）具备独立的 negative-results/benchmark 论文价值；可作为主线论文的姊妹篇或补充材料，同时增强主线论文"我们认真检验过"的可信度。
- **估计器一致性纪律**：sigma01 数值漂移事故（`_analysis/ipv_estimator_divergence_investigation.md`）→ `solver_mode=exact` 修复的闭环，说明"pinned 数值基线 + 兼容性开关"应成为常设约定；任何未来估计器加速都应先过 sigma01 parity 测试再合入。
- **仓库治理总体健康**：三层结构（plans/studies/knowledge）、evidence-first、状态词汇表执行得相当一致；主要问题只是 §2.4 所列的索引滞后，属低成本修复。

## 5. 后续工作建议（按优先级）

| # | 建议 | 优先级 | 成本 | 预期证据收益 | 前置 |
|---|---|---|---|---|---|
| A | **冻结 RQ007 sealed 确认协议 → 独立评审 → 请求 PI 开启决策**：端点限 C1–C3 + （可选）R3 包络确认性重估；开启后不可逆 | P0 | 低（数据已在，协议为主） | 内部主线 provisional → confirmed，论文最大措辞升级 | RQ009 已冻结（已满足）；PI 单独授权 |
| B | **PAPER002 ledger 与手稿 pending 标记同步**：R4 WOD-E2E 由 `\externalpending` 改为注册有界 null；spine 措辞全面对齐"情境条件化包络 + 可估计性门 + 弃权"；排查残留 IPV-增量暗示 | P0 | 低 | 消除稿-证不一致这一最大评审风险 | 无 |
| C | **Scope RQ014：OnSite 交互失败段检索/分割**，成功后重访 RQ011B 监视器判定 | P1 | 中 | 唯一被认可的 OnSite 重启路径；正向或更干净的否证皆有价值 | 计划评审 |
| D | **E16 被动性→死锁预注册定向检验**（单端点、明确功效来源） | P1 | 中 | 程序第一个潜在外部正向后果链 | 新增数据或预注册单端点设计 |
| E | **RQ013 重新定标为包络效用 vs 运动学基线**（锐度/弃权质量/官方结果分层），弃用"IPV 增量"定标 | P1 | 中 | 唯一有正向先验的增量问题；补齐证据链末环 | A 之后更佳 |
| F | **跨 RQ 衰减/功效综合分析**（§3.2）：每条 null 的最小可检出效应 + 测量误差预算一张表 | P1 | 低 | 把"边界 vs 构念失败"转为定量回答，直接进 SI | 无（输入齐备） |
| G | **治理同步修复**：dashboard 头部日期与 blocker 清单、STUDIES.md 与 registry 中 RQ001/002/004/005 状态（review→accepted）、PAPER002 与 RQ010B 状态一致化 | P2 | 极低 | 防止未来代理/评审被过期索引误导 | 无 |
| H | WOD-E2E 360° 追踪扩 N（唯一未用杠杆，N 75–98 → 更高） | P3 | 高 | 仅当论文需要更紧的偏好效应上界时才值得 | 明确的效应上界需求 |
| I | 维持关闭：RQ008B 不执行；RQ012 人工标注保持弃用 | — | — | 避免稀释 | PI 已决策 |

**建议的执行顺序**：B、G（本周，纯文档）→ A 协议冻结与评审（并行）→ PI 开启决策 → F（并行，低成本）→ C/D scope 与计划评审 → E。

## 6. 报告边界

- 本报告未访问 paper 仓库（`../9_overleaf/...`），关于手稿内容的陈述以本仓库内 PAPER002 记录为准。
- 本报告不改变任何 RQ 状态；§5 各项均需按共享研究协议走计划/评审/PI 决策流程。
- 所有数字引自各 `decision.md`、`RQ_PROGRESS_DASHBOARD.md`、`START_HERE.md` 与 `main_workflow.log`（2026-07-07 版本）。

## 来源索引

- `STUDIES.md`；`reports/knowledge/RQ_PROGRESS_DASHBOARD.md`；`reports/knowledge/rq_progress_registry.csv`
- `reports/knowledge/{RQ001,RQ002,RQ004,RQ005,RQ007,RQ009,RQ010,RQ012}_*/decision.md`
- `reports/knowledge/PAPER002_dynamic_ipv_evidence_architecture/{synthesis.md,decision.md}`
- `reports/knowledge/RQ013_beyond_safety_increment/README.md`
- `reports/knowledge/_analysis/{ipv_estimator_divergence_investigation.md,ipv_accel_hyperparam_finding.md}`
- `START_HERE.md`（2026-07-03 版）；`main_workflow.log`（尾部条目至 2026-07-07）；git log（至 `7bd6dcdc`）
