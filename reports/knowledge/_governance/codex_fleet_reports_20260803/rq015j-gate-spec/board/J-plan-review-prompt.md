# 独立复审任务：审查 track J 的计划，并核查计划里所有被当作既定事实的数字

你是**独立复审方**。被审对象是一份研究计划书，以及写在其中、尚未被任何人复核过的一批数字。

**关键背景：这些数字全部由同一个人（监督方 Cowork Claude）独立计算，直接写进了任务书，
没有经过第二方核对。** 你的首要价值就在这里。

**默认立场是找问题。** 如果你读完只写"计划合理"，这次复审就是失败的。
每一条结论都必须给出你自己算出的数字或引用到具体文件行，不接受"看起来没问题"。

---

## 你要审的东西

主文件：`.codex-fleet/rq015j-gate-spec/board/J-leader-kickoff.md`

上下文（按需读，不必全读）：
- `.codex-fleet/rq015h-abstain-gate/board/reports/H_FINAL_leader_synthesis.md`
- `.codex-fleet/rq015i-underflow-regimes/board/reports/I1_underflow_regimes.md`
- `.codex-fleet/rq015h-abstain-gate/board/commander_notes.md`（监督方的裁决与自认的错误）
- `STUDIES.md`（研究全貌）、`AGENTS.md` 的 Research Velocity Principle 一节

原始数据（你自己算，不要采信任何转述）：
- `.codex-fleet/rq015b-repair/work/anchor_mse.csv` —— 2,300 行、36 列，
  含 `mse_per_candidate[7]`、`w_log[7]`、`k_eff_log`、`legacy_fallback_triggered`、
  `at_grid_boundary`、`partial_underflow`、`ipv_log`、`source`、`signature`、`n_band`、`n_obs`
- `.codex-fleet/rq015b-repair/work/mechanism_split.csv` —— 含 `ht_weight`、`zero_postwarm_scope`
- `.codex-fleet/rq015g-hpc-resolve/work/anchor_mse_hpc.csv` —— 同结构的 HPC 版

解释器钉死 `<local-rq009-venv>/bin/python`。

---

## 第一部分【必做】核查这些被当作既定事实的数字

计划书把下列数字当作已确认事实写入，**逐条自己算一遍**，给出你的值与是否一致。
不一致的地方**以你的计算为准**，并指出监督方可能错在哪。

| # | 被断言的事实 |
|---|---|
| 1 | `spread(mse_per_candidate[7]) == 0` 的锚点有 **400 个（17.4%）**，且**全部来自 nuplan、waymo 零行** |
| 2 | `k_eff_log` 落在 6.75–7.00 这一格的有 **766 个**锚点，是分布中最大的一个模态 |
| 3 | `k_eff = 6.75` 对应最佳候选权重 **0.2027**（即 `max(w_log)` 与 `k_eff` 的换算） |
| 4 | 门（`spread==0` 或 `max(w_log)<0.20` 弃权）在样本上留下 **1,017 个 = 44.2%** |
| 5 | 门后按最佳候选权重分带，nuplan 与 waymo 的 \|ipv\| 均值分别为：0.20–0.25 带 0.131/0.051；0.25–0.35 带 0.178/0.200；0.35–0.50 带 0.273/0.356；0.50–0.75 带 0.438/0.571；0.75–1.01 带 0.755/0.970 |
| 6 | 上述最高带的样本数是 nuplan **39** vs waymo **352** |
| 7 | 门后汇总时 nuplan/waymo 的 \|ipv\| 均值为 **0.293 / 0.633**，边界饱和占比 **1% / 20%** |
| 8 | `zero_postwarm_scope == True` **等价于** `signature ∈ {U, Z}`，会把 signature N 整层排除 |
| 9 | 全语料台账（4 份 parquet 合计 **14,473,982** 行）**不含** `mse_per_candidate[7]` 与 `w_log[7]`，且其中的 `k_eff` 是**连乘域**的 |

---

## 第二部分【必做】审查门的设计本身

不要预设它是对的。至少回答：

1. **θ = 0.20 的依据成立吗？** 计划说它取在"近均匀模态的边缘"。
   你自己看 `k_eff_log` / `max(w_log)` 的分布，这个边缘是否真的存在、是否真的在 0.20 附近？
   换一个合理的 θ 会让下游结论有多大变化？
2. **两条判据是否有重叠或遗漏？** `spread(mse)==0` 的行在 log 域下 `max(w_log)` 是多少？
   判据 1 是否已被判据 2 完全包含（若是，判据 1 就是冗余的，应指出）？
3. **`max(w_log)` 是不是合适的统计量？** 与 `k_eff`、熵、前二名权重比等替代量相比，
   在这个用途下有没有明显更好的选择？如果有，说明理由与代价。
4. **在线可计算性**：门只允许用单次运行内可得的量。
   计划里的定义是否真的满足这一条？有没有隐含依赖离线信息？
5. **弃权时返回什么**：计划要求"不得返回 ipv = 0"。这个约定在工程上够不够明确？

---

## 第三部分【必做】审查范围与方法

1. **"全语料无法普查、只能做设计基估计"这个判断对吗？** 台账真的缺 log 域权重吗（自己去看 schema）？
   有没有别的产物保存了足够的信息，使得部分普查成为可能？
2. **设计基估计的做法是否成立？** 2,300 锚点的抽样设计（配额 U300/Z150/N125 × 4 单元）
   支持外推到全语料吗？`ht_weight` 的构造是否与该设计一致？
   计划要求用"全域分母"而非 `zero_postwarm_scope` 分母，这个选择对吗？
3. **bootstrap 设定**（B=2000、seed=20260731、N 层 491 个 cluster）是否恰当？
4. **是否有违反 `AGENTS.md` 速度原则的成分**——即这一轮有没有做过头？
   PI 明确说过这两个弃权机制"只要有即可，不做重点，设计上不必苛求细节"。
   反过来也要看：**有没有为了省事而漏掉必要的东西？**

---

## 第四部分【必做】审查它是否服务于真实用途

最终用途是 online verification：判断自动驾驶车辆的 IPV 是否落在人类分布内。
门的下游是 RQ009 的上下文条件 conformal envelope。PI 已定：**envelope 按场景上下文分格，
不按数据源拆分**（理由：社会倾向应是人类群体的固有属性）。

1. 这个门的输出形式，RQ009 能直接用吗？接口上缺什么？
2. 计划要求"按 RQ009 实际用的上下文变量分格"。
   去 `reports/knowledge/RQ009_dynamic_counterpart_conditioned_envelope/` 查明那些变量到底是什么，
   并判断 2,300 锚点样本里**是否具备**这些变量。**如果不具备，这条要求就是无法执行的，必须指出。**
3. 计划断言"门筛出的样本系统性更极端，但 AV 侧用同一把尺子，偏移两边抵消"。
   这个论证成立吗？有没有反例情形（例如 AV 的分布形状与人类不同时）？

---

## 你的产出

写到 `.codex-fleet/rq015j-gate-spec/board/reports/J_plan_review.md`，结构：

1. **核查表**：第一部分 9 条，逐条给"断言值 / 我的复算值 / 一致或不一致 / 若不一致的原因"
2. **发现的问题**：按严重程度排序。每条写清楚：是什么问题、证据、若不改会导致什么后果、建议怎么改
3. **明确判定**：`计划可执行` / `需修改后执行` / `不应执行`，并给出理由
4. **你认为计划里最薄弱的一条**，单列

写作要求：不用比喻和行话；每个数字带来源文件与列名；
**若你与监督方的计算不一致，直接说"监督方算错了"并给出正确值**，不要客气。

## 硬约束

- 只读不写（除了你自己的报告与 `work/` 下的临时脚本）
- **不改** `agent.py` / `ipv_estimation.py` / `process_interhub.py` / `reliability_logdomain.py`
- **不提交 git commit**，**不跑 HPC 作业**，**不重解任何锚点**
- 不解析 RQ007 held_out，不读 RQ014 致盲字段
- 不对 `reports/` 做全仓库 `rg`（会把 RQ003 controlled-access 行拉进上下文）
- 时间戳一律 `date -u +%Y-%m-%dT%H:%M:%SZ`，不要前瞻估计
- **你是复审方，不执行 track J 本身。** 不要开始做 J 的三件事。
