# NSFC 竞赛数据·开放探索执行计划（PI + 执行 subagent 多 agent 方案）

> 目标：对 NSFC 实车挑战赛数据做**进一步、开放式**分析。不预设要证明论文哪条 claim;把"10 个精英算法 × 15 场景 × 完整 replay + 专家六维评分"当作一个**尚未被充分挖掘的自然实验**,让 agent 发散发现结构/异常/新现象/新指标,再收敛证伪、连回论文。
> 本计划**不进 Overleaf**(见 CLAUDE.md),研究过程产物,留工作文件夹。报告用单文件 HTML。

---

## 0. 与上一轮的根本区别（为什么再做一轮）

上一轮（11 agent / 4 波,见 `02_process/{plan,tried,knowledge}.md`、`01_results/FINDINGS.md`）是**确认式**:预设 H1–H7、测"验证器偏离 ↔ 官方分"(Tier-B,结论 **null/欠功效**)、做 InterHub do-sim 因果。强结果只有 Tier-A(安全饱和、coordination 仍区分)与 do-sim 因果头条。

**这一轮是开放探索式**:
- 不从"证明论文 X"出发,而从"**这批数据里到底有什么**"出发。
- 鼓励发现**复杂化甚至反驳现有叙事**的结果。
- 把上轮当 limitation 的东西(NPC=脚本对手)**反转成角度**(受控刺激)。
- 守则:不 p-hack;dead-end 记 `tried.md`;惊喜度优先于"支持论文"。

---

## 1. 中心探索问题（开放,不预设答案）

**"这 10 个精英算法在真实社会交互场景里,到底做了什么、彼此怎么不同、为什么有的被专家判为协同差?"**

这是发现问题,不是验证问题。围绕它播种 territory,让 explorer 自由挖掘,PI 按"惊喜度 × 稳健潜力 × 对论文价值"策展。

---

## 2. 数据资产（已确认,可复用）

- **官方评分**:150 行 = 10 队 × 15 场景(A1–A7/B1–B4/C1–C4),六维 0–100(safety/efficiency/comfort/compliance/coordination/comprehensive)+ area_rank。
- **Replay logs**(每 session):`vehicle_perception_simulation_trajectory.log`(caseId+角色,15 段锚点)、`vehicle_trajectory.log`(ego)、`simulation_trajectory.log`(周围 agent,无 caseId)、`monitor.log`(健康)。
- **诊断 PDF**:每场景安全原语(碰撞、接管、压线、TTC 低于阈值时长、横向间距、违规)。全集仅 2 次碰撞(均 Beijing A1)→ 用连续安全原语,不用碰撞计数。
- **规模**:~4,587 raw tracks;冲突交互需 extractor 提取。
- **已知约束**:对手 = 脚本 NPC(`mvSimulation`);无人类对照轨迹;状态格(risk/geometry/role/phase)是推断非官方语义。
- **可复用产物**:`round2_envelope_for_verifier.csv`(规范带)、`src/sociality_estimation/core/ipv_estimation.py`(estimator)、`master_outcome_table.csv`、graded deviation CSV、do-sim 脚本与产物、Tier-A 表。

---

## 3. 多 agent 组织（PI + 执行 subagent）

- **PI（主控,judgment-only）**:写探索宪章、播种 territory、读 bounded 报告、**策展惊喜度/可发表性**、防过早收敛、决定深挖/证伪/收敛、最终综合。不亲自跑 pipeline。
- **执行 subagent（codex,并行,各自隔离工作区）**:
  - **explorer**（发散）：拿一片 territory + 开放 brief("surprise me"),产"发现卡 + 排序假设 + 证据指针",**不测固定假设**。
  - **experimenter**（收敛）：对 PI 选中的 lead 做严格统计检验。
  - **reviewer**（红队）：杀伪发现——混淆、泄漏、NPC 伪影、多重比较、口径循环。
  - **replicator**（独立复现）：换路线/换定义复算 headline 发现。
  - **designer**（可选）：为发散结果设计严谨检验或新指标。

隔离:写代码/改共享脚本的 agent 用 `--worktree`;读多型 explorer 共享 board。模型默认最强 + 最高 reasoning;analysis agent 给 network-enabled sandbox(`--sandbox workspace-write -c sandbox_workspace_write.network_access=true`)以便 `pip install`。

---

## 4. 波次（发散 → 策展 → 收敛 → 综合）

**Wave 0（PI）**:建新 run dir `./.codex-fleet/nsfc-open-explore/`;写探索宪章 + 数据地图到 `board/plan.md`;播 6 个开放 territory;写死 anti-confirmation 守则。

**Wave 1 — 发散开放探索（并行 explorer,开放 brief）**。territory 候选(开放,explorer 可自行扩展):
- **T1 算法行为表型**:对 10 个算法做交互签名聚类——是否浮现"驾驶人格"/失效家族?谁系统性激进/保守/犹豫?
- **T2 基准结构**:15 场景的交互需求与区分力——哪些场景最能分开算法?A1 为何出现全集仅有的 2 次碰撞?给 benchmark 一张潜在结构图。
- **T3 "安全但协同差"现象学**:直接在 replay 里**编目真实行为事件**(僵局、过度礼让冻结、抢行、震荡、犹豫、谁先走的博弈),建 safe-but-bad 的**现象学目录**,而非只靠 friction proxy。锚点已知:T6(thicv2025)总分高却在 C3/A3/B4 协同垫底。
- **T4 NPC 作为受控刺激（关键新角度）**:对手是同一套脚本 NPC → 这是**天然的"控制对手"刺激-响应实验**。比较不同 ego 算法对**同一 NPC 行为**的社会响应差异。把上轮的 limitation 反转成 opportunity:在控制了对手的前提下,算法间响应差异更干净。
- **T5 数据驱动社会指标发现**:不强加 IPV,直接从轨迹特征学一个**可解释模型**预测专家 coordination 分,看哪些行为原语最 load-bearing,再回看它们与 IPV/偏离的关系(自下而上,而非自上而下套 IPV)。
- **T6 跨场景一致性**:算法社会签名是稳定特质还是场景特异?可迁移性如何?

每个 explorer 产:发现卡(现象 + 证据指针 + 量级)、**排序假设**、以及**≥1 个"未被上轮叙事覆盖"的发现**。

**Wave 2 — PI 策展**:汇总发现,按"惊喜度 × 稳健潜力 × 论文价值"排序;挑 2–4 个深挖 lead;**显式保留反叙事/复杂化发现**;写 `board/knowledge.md`。这是 openness 的闸门——只有 PI gate 后才进收敛。

**Wave 3 — 收敛检验（experimenter + reviewer + replicator）**:对选中 lead 预注册主指标后严格检验;红队找伪;headline 发现独立复现;NPC 伪影/泄漏/多重比较显式处理;记 `board/validation.md`。

**Wave 4 — 综合（PI）**:产出 HTML 报告 + findings;每个存活发现标注对论文三幕叙事的影响(支持 / 复杂化 / 新方向)。

---

## 5. 引导开放探索的具体机制（回应"注意引导开放探索"）

这是本计划的核心,防止 agent 退化成"再确认一遍论文":
1. **explorer prompt 必须开放式**:"探索 {territory};告诉我**最意外、最可发表**的结构;给排序假设 + 证据指针;**不要只确认已有结论**;若发现与现有叙事冲突,照报。"(模板见 §8)
2. **发散配额**:每个 explorer 至少报 1 个未被上轮覆盖的发现;否则判为未完成。
3. **PI 策展看惊喜度,不看是否支持论文**;反叙事发现强制保留进 `knowledge.md`。
4. **两阶段闸门**:Wave 1 严禁过早收敛(不准在发散阶段就下结论);PI 显式 gate 后才进 Wave 3。
5. **anti-confirmation 守则**(写进每个 prompt):不得为支持论文选择性报告;dead-end 入 `tried.md`;NPC 伪影/泄漏/几何混淆先排除再信;n=10 团队级只用置换/精确检验。
6. **PI 自检**:每次策展问"我是不是在挑支持论文的、忽略复杂化的?"——若是,纠偏。

---

## 6. 交付物

- **HTML 报告**(Nature 风中文,单文件,内嵌 SVG,无外部 CSS/JS/字体依赖,沿用现有报告风格):探索宪章 → 发现地图 → 存活发现(证据/边界/复现)→ **反叙事/复杂化发现** → 对论文三幕的影响 + future work。
- `board/{plan.md, tried.md, knowledge.md, validation.md}`、`agents/*/results/`、`findings.md`。
- 每个发现可追溯到 CSV;最终报告 bounded。

---

## 7. 验收（开放探索版——过程与质量门,不是 PASS/FAIL 阈值）

- **覆盖**:≥5 个 territory 被开放挖掘,各有发现卡 + 排序假设。
- **新颖**:≥3 个"未被上轮覆盖"的发现进入策展;≥1 个反叙事/复杂化发现被诚实记录。
- **稳健**:进主报告的 headline 必须过红队 + 独立复现;NPC 伪影/泄漏/多重比较显式处理;团队级只用置换/精确检验。
- **诚实**:null/dead-end 在案;不 p-hack;惊喜度优先于"支持论文"。
- **连回**:每个存活发现标注对论文叙事的影响(支持/复杂化/新方向),并给"是否、如何"进主文/ED/future work 的建议。

---

## 8. Explorer dispatch brief 模板（paste-ready,开放式）

```
角色：explorer（发散开放探索）。你是 PI 领导的研究舰队的一只手,只看本 prompt。
任务：开放探索 NSFC 竞赛数据的 territory = {T?: 一句话}。目标是**发现**,不是验证。

数据(绝对路径)：
- 官方评分 / master_outcome_table：{path}
- replay logs（caseId 锚点在 vehicle_perception_simulation_trajectory.log）：{path}
- 诊断 PDF 安全原语：{path}
- 可复用产物（envelope/estimator/deviation/do-sim）：{paths}
约束：对手=脚本 NPC（mvSimulation）;无人类对照;状态推断非官方;团队 n=10 欠功效;仅 2 次碰撞。
环境：可 pip install;network sandbox 已开。

要求：
1) 开放挖掘——告诉我这片 territory 里**最意外、最可发表**的结构/现象/异常。
2) 不要只确认论文已有结论;若发现与"安全饱和+coordination 区分""自锚 verifier"等叙事冲突,照报。
3) 至少给 1 个"未被上一轮(H1-H7/Tier-A/B/do-sim)覆盖"的发现。
4) 排除 NPC 伪影/泄漏/几何混淆后再断言;n=10 用置换/精确检验。
5) dead-end 也要报(写进你的 results/tried_local.md)。

产物（bounded final report,≤~400 字 + CSV）：
- 发现卡：现象 / 证据指针(文件:行或图) / 量级 / 稳健性自评。
- 排序假设：3-5 条,标"惊喜度"和"下一步最便宜的严格检验"。
- 中间 CSV 落 agents/<name>/results/。
```

收敛阶段(experimenter/reviewer/replicator)沿用 `codex-research-fleet` 的对应模板。

---

## 9. 与上轮的衔接（复用 / 不重复 / 边界）

- **复用**:Tier-A、do-sim、deviation pipeline、envelope、estimator、master_outcome_table。
- **不重复**:不再跑 H1–H7 确认;Tier-B null 已知,不 p-hack;以发现为先。
- **边界**:NPC=脚本、无人类对照、状态推断——要么绕开,要么变角度(如 T4);所有跨 InterHub 迁移的断言须标"未验证迁移"。

---

## 10. 启动顺序

1. PI 写 `board/plan.md`(本文件即蓝本)+ 数据地图,与用户确认一次。
2. 一批并行 dispatch T1–T6 explorer(开放 brief)。
3. `fleet_status.sh --results` 收 bounded 报告 → PI 策展(Wave 2)。
4. dispatch 收敛 agent(experimenter/reviewer/replicator)查存活 lead。
5. PI 综合 → HTML 报告 + findings;标注对论文的影响。
