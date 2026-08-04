# 独立审查 B：从数据与产物出发，独立推导需求，最后再对照计划

你是**独立审查方 B**。同时还有另一位审查方在独立工作，你们互不可见——
**不要试图寻找或读取对方的产物。** 最终由监督方比对你们的分歧。

## 你的入口：**先不要读计划书。** 先自己推导

**执行顺序是强制的**，请按此进行并在报告里体现：

### 第一阶段（读计划书之前完成）

目标：**在不看方案的前提下，自己回答"要把下面这道门套到全语料上，到底需要做什么"。**

门的规格（这是唯一预先给你的东西）：
```text
w_log      = softmax(log_score)
mse_spread = max(mse_per_candidate) - min(mse_per_candidate)
max_w_log  = max(w_log)

if mse_spread == 0:      ABSTAIN, reason = NO_IPV_EFFECT
elif max_w_log < 0.20:   ABSTAIN, reason = NEAR_UNIFORM
else:                    OK, ipv_log = sum(candidate_ipv_i * w_log_i)
```

请自己去查并回答：
1. 全语料的规模与真实计算单元是什么？（台账 4 份 parquet 在
   `reports/studies/RQ015A_ipv_estimability_labelling/RQ015A_1_concentration_audit_20260731T093746Z_e82091ce/concentration_ledger/`）
2. `mse_per_candidate[7]` 与 `log_score[7]` 目前在整个仓库里**有没有任何地方存着**？
   把你搜过的目录与产物全部列出，包括没有的。**这一条是本次审查最有价值的部分。**
3. 若必须重算，最小的重算是什么？要动哪些代码路径？成本量级多少？
   （可参照 `.codex-fleet/rq015g-hpc-resolve/board/reports/G_leader_adjudication.md` 里的实测）
4. 产出应该长什么样，才能让下游 RQ009 直接用？

### 第二阶段（完成第一阶段后才读）

现在读 `.codex-fleet/rq015k-fullcorpus-gate/board/K-leader-kickoff.md`，
并逐条对照你自己的结论：

- 你推出来而计划**没有**要求的 → 计划的遗漏
- 计划要求而你认为**不必要**的 → 计划的冗余
- 两边结论**冲突**的 → 逐条给出你的证据

**请在报告里明确标出哪些结论是第一阶段得出的（未看计划书）**，
这是本次双盲的关键，不要事后回填。


## 共同的三个收尾问题（两位审查方都必须回答，措辞一致，便于比对）

**Q1. 这轮重算是否必要？** 有没有任何现成产物能替代它、或大幅降低它的成本？
把你查过的路径全部列出，**包括查了但没有的**。

**Q2. K1 的勘察范围，是否足以支撑「要不要投 K2」这个决定？**
缺什么？多了什么？

**Q3. 明确判定：`可执行` / `需修改后执行` / `不应执行`。** 给理由。
另单列一条：**你认为这份计划里最可能造成实际损失的一处。**

## 硬约束（违反即为审查失败）

- **你是独立审查方，不执行 K1 或 K2 本身。** 不要开始勘察、不要投任何作业
- **禁止读取另一位审查方的任何文件**：不得打开 `board/reviews/` 下不属于你的报告，
  不得读不属于你的 `.log`。你不知道对方在看什么，也不需要知道
- 只读；除你自己的报告与 `work/` 下你自己的临时脚本外不写任何文件
- **不改** `agent.py` / `ipv_estimation.py` / `process_interhub.py` / `reliability_logdomain.py`
- **不提交 git commit**，**不提交任何 HPC 作业**，不重解任何锚点
- 不解析 RQ007 held_out，不读 RQ014 致盲字段
- 不对 `reports/` 做全仓库 `rg`（会把 RQ003 controlled-access 行拉进上下文）
- 时间戳一律 `date -u +%Y-%m-%dT%H:%M:%SZ`，**不要前瞻估计**
- 解释器钉死 `<local-rq009-venv>/bin/python`

写作要求：不用比喻和行话；每个数字带来源文件与列名或行号；
**若你的计算与计划书或监督方不一致，直接写"计划错了"并给出正确值，不要客气。**

产出写到 `.codex-fleet/rq015k-fullcorpus-gate/board/reviews/KR_B_review.md`。
