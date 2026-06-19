# 验证实验计划：自锚区间是"群体规范"而非"个体自证"

> 目的：用可证伪的实验,确认 self-anchored conformal IPV 区间检验的是**人类群体的条件规范**,而不是退化成"与自己一致"的 self-consistency（即排除 **norm-laundering** 失效）。
> 产出由 codex 执行（见配套 `codex_prompt_self_anchor_validation.md`）。报告用 HTML。
> 数据：InterHub,批次 `20260612_sigma_0_1_full_rerun`,balanced locked slice 5,000 cases / 10,000 rows,sigma=0.1,四源（Waymo/nuPlan/Lyft/AV2）。内容包 `.codex-fleet/ipv-online-interval/`。

---

## 0. 中心问题与判定逻辑

**中心问题**：把"合理区间"条件在 agent 自身早窗 IPV 上,是否构成**群体条件规范** `P(full-window IPV | early-window IPV, context)`(锚点是个体、裁判是人群),还是退化为"个体自定义正常"(self-consistency 洗白)？

**要捍卫的论点**：早窗 IPV 是**协变量**,合理范围仍是**人类人群的条件分布**,且 verifier 仍能对落在人类条件范围之外的行为报警。

**要排除的失效模式（norm-laundering）**：因为条件在 agent 自己的早窗上,一个**从头到尾自一致的激进 agent** 的 band 会随它自己平移,从而永远"通过"。

**总判定（go / no-go,实验结束时必须给出）**：

- **SUPPORTED（保留 R2/R3 自锚叙事）** 当且仅当：E1 非平凡 ∧ E2 跨个体覆盖成立 ∧ E3 能报警"自一致但超出人类范围"的 deviator ∧ E5 自锚判罚对真实坏结果的追踪 ≥ 情境基线。E4 用于量化并约束残余暴露。
- **MUST REVISE（降级/改混合）** 若：E3 出现洗白(自一致激进者全程通过) ∨ E5 自锚系统性放过坏结果案例。回退设计：**hybrid = 自锚保 sharpness + 情境 floor / out-of-support abstention** 防洗白；或只条件在"情境可解释的早窗分量"上(见 E4)。

每个实验都给 PASS/FAIL 阈值;阈值是预注册的,先写死再跑。

---

## 1. 实验清单（E1–E5）

### E1 — 非平凡性与时间分离（angle：是不是真的"预测",不是窗口重叠/泄漏）
- **目标**：确认早窗→全窗是真实预测规律,而非时间重叠或未来泄漏;且条件 band 仍保留人群散布。
- **方法**：
  1. 校验早窗 `[0, t_e]` 与作为 target 的 full-window IPV 评分窗**无时间重叠**,量化重叠占比。
  2. 计算 corr(early-IPV, full-IPV)、R²。
  3. 条件 band 宽 vs 边际 band 宽（width ratio）。
  4. 在 ≥1,000 cases（非仅 n=48）上复核因果重建门：map-lane reference 的 corr vs observed-prefix reference 的 corr。
- **主指标**：时间重叠占比;corr/R²;width ratio;leakage-gate corr。
- **PASS（非平凡群体规范）**：无时间重叠;corr 落在非退化区间(约 0.3–0.9);条件 band 宽 ≥ 边际的 ~50%(未塌缩) **且** 留出人类覆盖 ≥0.88。
- **FAIL（平凡）**：存在重叠/泄漏;或 band 塌缩(width ratio≪);或 corr≈1 来自重叠。
- **agent role**：experimenter（+ 复核 leakage 由 reviewer）。

### E2 — 跨个体/跨群体泛化（angle：band 是共享人群函数,不是记住个体）
- **目标**：在一部分人类上拟合的条件 band,能覆盖**没见过的个体/场景**。
- **方法**：leave-one-driver-out（若有 driver/agent ID）与 leave-one-scenario-out 的覆盖率;复用 leave-Waymo-out 作粗粒度。报告留出覆盖、宽度。
- **主指标**：held-out 覆盖（Wilson CI）、宽度,按 holdout 类型分。
- **PASS**：held-out-individual 覆盖 ≥ ~0.88（接近名义 0.90）→ 泛化、非记忆。
- **FAIL**：held-out-individual 覆盖塌（如 <0.80）→ band 个体特异(记忆)。
- **备注**：若无 driver ID,用 scenario/case 留出并标注此局限。
- **agent role**：experimenter。

### E3 — 一致性 deviator 压力测试（**决定性**;angle：能否报警自一致但超人类范围者）
- **目标**：直接检验 norm-laundering。若只查自一致,自一致激进者会通过。
- **方法**（合成/counterfactual agent,在锁定切片上注入）：
  - **3a 正常早窗 + 后段偏离**：早窗 IPV 正常,full-window 向竞争方向平移 Δ。预期 flag 率随 Δ 升。
  - **3b 自一致平移（洗白探针）**：早窗与全窗**同时**平移 Δ（从一开始就激进、内部自一致）。关键：当 Δ 把早窗 IPV 推到**人类支撑域之外**时,band **不得外推洗白**,verifier 必须报警(经由 out-of-support 检测 / band 截断 / abstention)。
  - **3c out-of-support 检测**：当早窗 IPV ∉ 人类经验支撑时,系统须 flag/abstain,而非静默加宽 band。
- **主指标**：3a/3b 的 flag 率 vs Δ;flag 率过 0.5 的 Δ;早窗越界时的 flag/abstain 率。
- **PASS（群体规范）**：3a flag 随 Δ 升;3b 自一致但超出人类条件范围者**被报警**(flag 随 Δ 升,一旦越出人类支撑);3c 越界触发 flag/abstain。
- **FAIL（洗白）**：3b 自一致激进者在所有 Δ 都通过(flag≈基线),或 band 静默外推越过人类支撑 → 叙事必须改(加情境 floor / out-of-support abstention)。
- **agent roles**：experimenter（执行）+ **replicator（用不同 deviator 构造/不同条件模型独立复现 3b）** + reviewer（红队找漏洞）。

### E4 — 情境 vs 个体倾向分解（angle：量化并约束洗白暴露）
- **目标**：拆出早窗信号里"情境可解释"与"个体倾向残差"两部分,量化洗白暴露面。
- **方法**：用情境（geometry/role/risk/counterpart）回归早窗 IPV,得情境分量 ê_situation 与残差 ê_disposition。建两个区间模型:M1 条件在完整早窗;M2 只条件在 ê_situation。比较 sharpness/coverage/transfer;并测 ê_disposition 对 full-IPV 的增量 R²。
- **主指标**：M1 vs M2 的 width/coverage/transfer;ΔR²(ê_disposition);M2 保留的收益比例。
- **PASS（低洗白风险）**：M2（只用情境分量）保留大部分 sharpness/coverage/transfer → 收益主要来自情境而非个体自证。
- **CONCERN（须约束）**：若 ê_disposition 对收益不可或缺 → 标注洗白暴露,量化并讨论缓解(情境 floor)。
- **agent role**：experimenter + designer（设计分解）。

### E5 — 自锚 vs 情境判罚的外部结果裁决（angle：分歧由真实后果裁决）
- **目标**：在自锚 band 与"纯情境 band"判罚**不一致**的案例上,哪种判罚更贴合真实坏结果？若自锚系统性放过情境会报警的坏结果案例 → 洗白。
- **方法**：建纯情境条件 band(geometry×role×risk,无自锚)。取分歧案例。用外部结果裁决：InterHub 下游代理(PET 骤降、急刹、被迫让行、conflict-tail)为主(即取即用);NSFC 排名/分场景评分/安全事件为更强外锚(可用时)。
- **主指标**：分歧案例中各判罚与坏结果的关联(AUC / odds ratio);自锚"通过"案例里坏结果的比例。
- **PASS**：自锚判罚对坏结果追踪 ≥ 情境;自锚的"通过"非系统性坏结果。
- **FAIL**：自锚通过坏结果案例的比例高于情境(后果层面证实洗白)。
- **agent role**：experimenter + reviewer。

---

## 2. 交叉验证（fleet 规范）

- **红队（reviewer）**：专门攻击"群体规范"结论——找泄漏、混淆、band 退化、holdout 污染、support 作弊、Δ 选择偏置。任何 headline 结论须过红队。
- **独立复现（replicator）**：对 **E3(决定性)** 用不同路线独立复现(不同 deviator 构造 + 不同条件模型),比较一致性。
- 记录在 `board/validation.md`。

---

## 3. 交付物（codex 产出）

- **HTML 报告**（Nature-style 中文,内嵌 SVG 图,无外部 CSS/JS/字体依赖,沿用现有报告风格）：科学问题 → 每个实验的设计/结果/裁决 → 总 go/no-go → 局限。
- 每个实验的 **metrics CSV**。
- `findings.md`、`FINAL_REPORT.md`(PI 综合)、`board/validation.md`。
- 嵌入图(建议)：fig_e1_nontriviality、fig_e2_crossindividual、fig_e3_stress(决定性)、fig_e4_decomp、fig_e5_adjudication。

---

## 4. 纪律（必须遵守）

- **无泄漏**：在线特征只用因果前缀 + 静态 map-lane;observed PET / 全窗 IPV / 未来 phase 只能做标签,不进在线路径。
- **conformal 校准**：所有覆盖声明基于 split-conformal 后区间;raw quantile 的欠覆盖须同时报。
- **预注册阈值**：PASS/FAIL 阈值跑前写死(本文件 §1)。
- **诚实负结果**：若 FAIL,直接写"叙事须改 + 推荐 hybrid 回退",不得粉饰。
- **可追溯**：报告每个数字对应到 CSV;最终报告 bounded。

---

## 5. 与论文的对接

- **若 SUPPORTED**：E2/E3/E5 进主文或 Extended Data,直接堵住"合理性掌握在个体手里"的审稿质疑;R2/R3 自锚叙事成立。
- **若 MUST REVISE**：把 verifier 改为 **hybrid（自锚 + 情境 floor / out-of-support abstention）**,R2 改写为"自锚收窄区间,情境 floor 保证非洗白",并用 E3/E5 的结果支撑这个更稳的设计。

> 立刻可做(不依赖 NSFC)：E1、E2、E3、E4 与 E5 的 InterHub 下游代理部分。NSFC 仅用于 E5 的更强外锚,等数据 pipeline 就绪再补。
