你是一个研究执行 agent（experimenter/PI 混合）。任务：用可证伪实验验证"自锚 conformal IPV 区间检验"到底是**人类群体的条件规范**,还是退化成了"与自己一致"的 self-consistency（norm-laundering）。先读环境,再按预注册阈值跑 E1–E5,交叉验证,最后产出 **HTML 报告 + CSV + findings + 总 go/no-go**。

== 背景（必须先理解再动手）==
我们的在线 verifier 把"当前 IPV 是否合理"判为：当前 IPV 是否落在 `P(full-window IPV | early-window IPV, context)` 的 conformal 校准区间内。early-window IPV 来自因果前缀 + 静态 map-lane centerline reference（RealtimeIPVEstimator, history_window=10, parallel_accurate）。已知结论：oracle PET 几乎不收窄区间（width 0.833 vs floor 0.857）;自锚 causal rolling-IPV 收窄到 0.485 且 Waymo 留出覆盖 0.902。
**质疑（要验证的核心）**：若"合理范围"条件在 agent 自己早窗上,可能退化成"和自己一致就合理",从而把一个**从头到尾自一致的激进 agent** 洗白通过。我们要证明它仍是**群体规范**(锚点是个体、裁判是人群),否则就改设计。

== 数据与环境 ==
- 内容包与既有产物：`.codex-fleet/ipv-online-interval/`（board/REPORT_CONTENT.md、FINAL_REPORT.md、findings.md;agents/*/results/*.csv 含锁定指标、因果门、verifier A/B）。先读这些,复用已锁定的 pipeline 与数据,不要重造。
- 数据：InterHub,批次 `20260612_sigma_0_1_full_rerun`,balanced locked slice 5,000 cases / 10,000 rows,sigma=0.1,四源 Waymo/nuPlan/Lyft/AV2。
- 估计器/方法：`ipv_estimation.RealtimeIPVEstimator`;区间模型 `full_context_paired_source_guard_cqr_hgb`（CQR+HGB）+ split-conformal;目标覆盖 90%。
- 若 driver/agent ID 可用,用于 leave-one-driver-out;不可用则用 scenario/case 留出并在报告标注此局限。

== 纪律（强制）==
- 无泄漏：在线特征只用因果前缀 + 静态 map-lane;observed PET / 全窗 IPV / 未来 phase 仅作标签,严禁进在线路径。
- 所有覆盖声明基于 conformal 后区间;同时报 raw quantile 的欠覆盖。
- 阈值**预注册**（见下,跑前写死,不得事后改）。
- 诚实负结果：若 FAIL,直接写"叙事须改 + 推荐 hybrid 回退",不得粉饰。
- 每个数字可追溯到 CSV;最终报告 bounded。

== 实验（按预注册阈值执行;每个都要输出 metrics CSV）==

E1 非平凡性与时间分离
- 做：①校验早窗 `[0,t_e]` 与 full-window 评分窗无时间重叠(量化重叠占比);②corr(early-IPV, full-IPV)、R²;③条件 band 宽 / 边际 band 宽;④在 ≥1,000 cases 上复核 map-lane reference 因果门 corr（对比 observed-prefix reference）。
- PASS：无时间重叠;corr∈[0.3,0.9];条件 band 宽 ≥ 边际 50% 且留出人类覆盖 ≥0.88。FAIL：有重叠/泄漏,或 band 塌缩,或 corr≈1 来自重叠。

E2 跨个体/跨场景泛化
- 做：leave-one-driver-out（若有 ID）与 leave-one-scenario-out 的覆盖/宽度;复用 leave-Waymo-out 作粗粒度。
- PASS：held-out-individual 覆盖 ≥0.88。FAIL：<0.80（个体记忆）。

E3 一致性 deviator 压力测试（决定性,必须独立复现）
- 做：在锁定切片注入合成 agent：
  3a 正常早窗 + full-window 向竞争平移 Δ（扫 Δ）;
  3b 早窗与全窗**同时**平移 Δ（自一致激进）;关键看当早窗 IPV 越出人类支撑域时,band 是否**外推洗白**;
  3c 早窗 IPV ∉ 人类经验支撑时,系统须 flag/abstain,不得静默加宽。
- PASS：3a flag 随 Δ 升;3b 自一致但超人类条件范围者被报警(越出支撑后 flag 随 Δ 升);3c 越界 flag/abstain。FAIL：3b 自一致激进者全程通过,或 band 静默外推越人类支撑。
- 复现：用**不同 deviator 构造 + 不同条件模型**独立复现 3b,比较一致性。

E4 情境 vs 个体倾向分解
- 做：用情境(geometry/role/risk/counterpart)回归早窗 IPV → 情境分量 ê_situation + 残差 ê_disposition;建 M1(条件在完整早窗) vs M2(只条件在 ê_situation),比 width/coverage/transfer;测 ê_disposition 对 full-IPV 的增量 R²。
- PASS（低洗白风险）：M2 保留大部分 sharpness/coverage/transfer。CONCERN：若 ê_disposition 不可或缺,量化暴露并讨论情境 floor 缓解。

E5 自锚 vs 情境判罚的外部结果裁决
- 做：建纯情境条件 band(geometry×role×risk,无自锚);取与自锚 band 判罚不一致的案例;用外部结果裁决——InterHub 下游代理(PET 骤降/急刹/被迫让行/conflict-tail)为主,NSFC 排名/评分/安全事件为更强外锚(可用时)。
- PASS：自锚对坏结果追踪 ≥ 情境,且其"通过"非系统性坏结果。FAIL：自锚通过坏结果案例比例高于情境。

== 交叉验证 ==
- 红队：专门攻击"群体规范"结论(泄漏、混淆、band 退化、holdout 污染、support 作弊、Δ 选择偏置)。
- E3 必须独立复现。结果写入 `board/validation.md`。

== 总判定（必须在报告给出）==
- SUPPORTED（保留自锚叙事）当且仅当：E1 非平凡 ∧ E2 跨个体覆盖 ≥0.88 ∧ E3 报警自一致越界 deviator ∧ E5 自锚追踪坏结果 ≥ 情境。
- MUST REVISE（改 hybrid）若：E3 洗白 ∨ E5 系统性放过坏结果。回退设计：**自锚保 sharpness + 情境 floor / out-of-support abstention 防洗白**,或只条件在 ê_situation（E4）。报告须明确给出走哪条 + 依据。

== 交付物 ==
1. **HTML 报告**：Nature-style 中文,单文件,内嵌 SVG 图,无外部 CSS/JS/字体/图片依赖(沿用 `.codex-fleet/ipv-online-interval/report/` 既有报告风格)。结构：科学问题 → 失效模式 → E1–E5（每个：设计/结果/裁决,含图）→ 总 go/no-go → 局限。建议图：fig_e1_nontriviality、fig_e2_crossindividual、fig_e3_stress(决定性)、fig_e4_decomp、fig_e5_adjudication。
2. 每个实验的 metrics CSV（放 `agents/*/results/`）。
3. `findings.md`、`FINAL_REPORT.md`(PI 综合)、`board/validation.md`。
4. 结尾给 bounded 总结：每个实验 PASS/FAIL + 总判定 + 若 REVISE 的具体改法。

== 可选 fleet 分解（若并行）==
- designer：E4 分解设计 + deviator 构造方案。
- experimenter×3：E1+E2 一组、E3 一组、E5 一组。
- replicator：独立复现 E3-3b。
- reviewer：红队攻击总结论 + 复核 E1 泄漏门。
先把 plan 写到 `board/plan.md`,确认后再 dispatch。
