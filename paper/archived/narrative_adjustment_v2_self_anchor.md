# 叙述调整 v2：自锚 conformal 取代 PET-bin 包络查询

> 触发：两份新报告（`在线当前 IPV 合理性区间估计`、`自锚 conformal 取代 PET-bin 包络查询`，2026-06-19，13-agent codex fleet 产出）。
> 本文件只讲"叙述如何调整"。已 pull 最新论文（Overleaf `c2f1de6`，你只做了去 `\emph` 等表层改动，正文内容未变）。

---

## 0. 新结果一句话

在线判断"当前 IPV 是否落在人类合理区间"时，**预测 risk/PET 再查经验包络几乎没用**（oracle PET 仅比全局 floor 窄约 3%）；真正让区间又窄又可校准、还能跨数据源迁移的，是**读取驾驶人自身早窗的因果 rolling-IPV（自锚）+ conformal 校准**。

锁定数字（balanced lane_ids 切片）：

| 方法 | TEST cov/width | Leave-Waymo-Out cov/width |
|---|---|---|
| FLOOR（全局） | 0.889 / 0.898 | 0.868 / 0.871 |
| oracle PET（CEILING） | 0.889 / 0.867 | 0.860 / 0.840（仍欠覆盖）|
| no-roll（因果运动学） | 0.896 / 0.738 | 0.857 / 0.743 |
| **causal-roll（自锚，推荐）** | **0.899 / 0.591** | **0.902 / 0.628（唯一 ≥0.90）** |

Verifier A/B（在线包络直替 PET-bin，同一 deviation 接口）：LWO 下 Online 0.823/0.488/误报0.114 全面优于 PET-bin 0.786/0.678/0.142。泄漏已排除：用地图车道中心线 reference，因果前缀 IPV 与离线 IPV 的 corr=0.993（observed-prefix reference 只有 0.281，是错误参照所致）。

---

## 1. 最大改动：verifier 的"参照对象"换了

**旧链路（现稿 R2/R3）**：state recognizer（risk×geometry×role×time）→ PET-bin 经验包络 → deviation。risk 是核心维度，runtime gap（在线 risk 只恢复 ~55%）是头号软肋。

**新链路**：RealtimeIPVEstimator（prefix + map-lane）→ 自身早窗 causal rolling-IPV →〔rolling-IPV + 几何 + 路权 + 运动学，**不含 PET**〕→ split-conformal → [q05,q50,q95] 区间 → deviation。paired-IPV（sum/diff）作**第二层联合合理性 gate**；source-guard 处理跨域漂移；无车道时退回 no-roll（可部署切片约 74%）。

这条新链路一举把原稿的三个弱点变成卖点：① 绕开 risk 估计瓶颈；② 覆盖率达名义 0.90（原来 0.823 欠覆盖）；③ 跨 Waymo 留出 0.902（原来 source-dependent）；④ 区间窄 ~42%。

---

## 2. 诚实对账：和 Claim 1、以及我上一轮"必须估计 risk"的关系

**这条新结果直接修正了我上一轮的回答。** 上轮我论证"online 必须估计 risk，因为规范随 risk 翻转"。要分清两个不同的估计目标：

1. **规范的位置（中心/中位数）**：随 risk 移动（Claim 1，+0.058→−0.034 的中位数摆动）——这是**人群层面的科学事实**。
2. **某个具体 agent 在线的合理区间（验证真正要的东西）**：由**状态内、驾驶人之间**的方差主导，最佳预测来自 agent 自身早窗 IPV（时间自一致性）。risk 在这里贡献很小。

为什么不矛盾：risk 驱动的中位数摆动（~0.09）相对区间宽度（~0.5–0.86）和驾驶人间散布很小。所以条件化 risk 把中心挪一点，但**不收窄散布**；而验证器要拿捏的正是散布，自锚才能收窄它。

**一句话新口径**：*risk 决定人群规范"在哪里"，但不决定单个体合理行为"有多紧"；在线验证关心后者，所以自锚于驾驶人自身因果 IPV，而不是估计 risk。*

由此：
- 我上轮"必须估计 risk"**只在"识别人群规范场"意义上成立**（即 R1 的描述性发现）；对**可部署 verifier 不成立**——它自锚、绕开 risk。这是好消息：它消掉了我标记的头号 runtime 瓶颈。
- risk 在 verifier 里**降级**：从"区间的核心条件变量"降为"可选的升级/escalation gate"（高风险下同样的偏离更紧急→更易触发 fallback）。它不再进入区间估计的 online feature path。
- 额外好处：verifier 刻意**不用 PET/risk**，所以更难被审稿人说成"安全检验换皮"——它测的是与碰撞风险正交的东西（相对人类条件分布的社会自一致性）。这把上一轮 Q3 的区分效度论证变得更干净。

---

## 3. 新的叙述主线（修订 Results 顺序）

把原来"状态条件包络仅微弱胜 scalar（降级）"的负结果，**升级成一个反直觉的方法学发现**：

1. **R1（不变，强）**：社会规范在人群层面是状态依赖的（priority gap 随 risk 翻转）。— 关于人类行为的科学发现。
2. **R2（反直觉负结果，新升级为亮点）**：构建 verifier 最直觉的做法是"预测 risk→查包络"，但它几乎不比全局 floor 窄——**risk/PET 不是 IPV 合理区间的主要信息源**。这解释了原稿"状态条件仅微弱胜 scalar"的旧负结果：不是状态条件没用，而是**选错了条件变量**。
3. **R3（新方法学核心）**：换用**自锚 causal rolling-IPV + conformal**，区间窄 ~42%、覆盖达名义 0.90、且**跨 Waymo 留出 0.902**——sharpness + calibration + 跨源迁移三者兼得。paired-IPV 作第二层联合 gate。
4. **R4（verifier A/B）**：在线自锚包络 vs PET-bin 包络，同一 deviation 接口：更窄、误报不升（LWO 下降）。"我们的 verifier 成立且胜过显然基线"。
5. **R5/R6**：原 early-warning（POC）、planner 接口（demo）保留但后置。
6. **R7/R8（NSFC，planned）**：外部验证 + 区分价值。

这条弧线非常 NMI：**一个反直觉、强证据、且方法学新颖的发现**（"别预测风险，去读驾驶人自己"），并顺手解决了原稿最大的两个技术弱点。

---

## 4. 原弱点 → 现卖点（更新 Limitations）

| 原稿 limitation | 新证据 | 现表述 |
|---|---|---|
| coverage 欠覆盖（0.823），不能称 calibrated | conformal 后 TEST 0.90、LWO 0.902 | 名义覆盖达成（在锁定切片上）|
| 在线校准 source-dependent（Waymo 0.706；Lyft FPR 19.8%）| 自锚 causal-roll 跨 Waymo 留出 0.902 | 跨源迁移由自锚完成（限 lane 可用切片）|
| runtime gap：在线 risk 只恢复 ~55% | verifier 不依赖 risk | 该 gap 对 interval 不再是瓶颈 |
| 预测优势仅微弱胜 scalar（Claim 2 降级）| 自锚窄 42%、显著胜 floor 与 oracle PET | 升级为方法学亮点 |

**保留的诚实边界**（必须写进 Limitations，否则被打）：① 可部署切片约 74%，依赖 `lane_ids`，其余退回 no-roll；② route-conditioned 假设（决策时车道/路线已知）；③ 硬约束部署前仍需目标域小样本再校准（LWO 仍 <0.90 的硬阈值线）；④ 锁定数字来自 balanced 5k、sigma=0.1，conditional/per-cell validity 不能无条件保证。

---

## 5. 具体改哪里（落到 main.tex）

- **Abstract**：把"state-conditioned envelope + 诚实 runtime gap"的关键结果句，换成"自锚 causal rolling-IPV + conformal 给出更窄、名义覆盖、跨 Waymo 留出迁移的在线区间；并发现 risk/PET 不是合理区间的主要信息源"。
- **Intro 贡献点 (iii)**：从"在线验证器 + 诚实 runtime gap"改为"一个自锚、distribution-free 校准的在线 verifier，并给出反直觉发现：合理区间的信息来自驾驶人自身早窗 IPV 而非风险分箱"。
- **Results**：按 §3 重排；把原 R3 baseline ladder 改写为"reframe 负结果（risk 不收窄区间）+ 自锚方法胜出"，并入新的 Fig（reframe + cross-dataset + causality + verifier A/B 四联图，对应报告 Fig 1–4）。
- **Fig 1 架构图**：重画 online path = RealtimeIPVEstimator(prefix+map-lane) → rolling-IPV + 几何/路权/运动学（标注"PET excluded"）→ conformal → 区间 → deviation；加 source-guard 切换与 lane-fallback 分支；risk 仅作可选 escalation gate。
- **Methods**：① IPV 估计补"自锚 causal rolling-IPV，prefix + map-lane centerline reference，history_window=10"；② 把上轮写的"动态规范带"改成"**conformalized quantile regression（CQR）+ split-conformal**，self-anchor 为主特征；③ 把 P(IPV_i|IPV_j) 条件准则**重定位为第二层 paired 联合 gate**（pair-sum/pair-diff），而非主判据（报告显示 paired-only 仅 +0.46pp 覆盖、略变宽）；④ 加 source-guard / calibration-mode 切换与 lane fallback；⑤ 因果落定段（map-lane reference corr 0.993）写进 leakage 论证。
- **输出字段**：可在 Methods 或 ED 列出报告第 5 节的接口字段（ipv_interval_p50/80/90/95、ipv_reasonable_p90、ipv_tail_score、pair_sum/diff_interval_p90、pair_joint_reasonable_p90、calibration_mode、source_health）。
- **Limitations**：按 §4 改写。

---

## 6. 待确认 / 保持诚实

- **NSFC 衔接**：新自锚 verifier 同样套到 NSFC 获奖算法上——而且更好，因为它已跨源校准。Fig 8/9 的分析思路不变，但 deviation 来自自锚区间。
- **route/lane 依赖**：NSFC 实车场景是否有可靠 lane/route 语义？若无，落到 no-roll 兜底——这会影响可部署切片论证，需在数据确认时核对。
- **硬约束 vs soft 接入**：LWO 仍 <0.90 硬线，论文口径继续保持 soft-cost / warning / monitor 优先，硬约束留作目标域再校准后的 future work。
- **别过度泛化**："跨源稳健"只在 balanced lane_ids 锁定切片成立，不是对任意未知地图的无条件保证——措辞照报告口径。
