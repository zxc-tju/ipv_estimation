# 独立审查 A：从计划书出发，审查全语料重算方案

你是**独立审查方 A**。同时还有另一位审查方在独立工作，你们互不可见——
**不要试图寻找或读取对方的产物。** 最终由监督方比对你们的分歧。

## 你的入口：先读计划书，再验它

被审对象：`.codex-fleet/rq015k-fullcorpus-gate/board/K-leader-kickoff.md`

它继承的规格与结论（属既有记录，你**可以**读）：
- `.codex-fleet/rq015j-gate-spec/board/reports/J1_gate_spec_and_impact.md`（门规格定稿）
- `.codex-fleet/rq015j-gate-spec/board/reports/J_plan_review.md`（上一轮的独立复审）
- `.codex-fleet/rq015h-abstain-gate/board/reports/H_FINAL_leader_synthesis.md`
- `.codex-fleet/rq015g-hpc-resolve/board/reports/G_leader_adjudication.md`（HPC 通道与成本参照）
- `AGENTS.md` 的 Research Velocity Principle 与「自带上下文的汇报」两节

## 你要审的四件事

### 1. 前提是否成立
计划的立论是「现有台账没有 `mse_per_candidate[7]` 与 `w_log[7]`，
且已有的 `k_eff` 由连乘域 `ipv_error` 派生、不能替代，所以必须重算」。
**自己去核这个立论**，不要采信转述。若它不成立，整轮就不该做。

### 2. 门的规格在全语料上是否可无歧义执行
J1 定稿的规格是针对单帧写的。搬到千万行的批处理上，有没有单帧场景下不会暴露的歧义？
至少检查：`mse_spread == 0` 的浮点相等在批处理里怎么判、
`log_score` 从哪来、`softmax` 的数值实现、缺列或 NaN 行怎么处理、
以及互斥 reason 的判定顺序会不会在向量化实现里被写反。

### 3. 成本模型与资源方案的要求是否足够
计划要求 K1 实测单位成本与内存后再给方案。
**这个要求本身够不够？** 参照 G 轨：2,300 锚点 / 6 worker / 500.6s；
24 worker 曾因每进程各载一份 PKL 而 OOM（TRES mem=160992M 仍不够）。
用这些数外推到全语料，你算出的量级是多少？计划有没有低估？
断点续算、失败重投、产物校验这三件，计划要求得够不够具体？

### 4. 分段设计与速度原则
计划把 K1（勘察）与 K2（执行）分开，K1 后必须停下。
这个分段是必要的谨慎，还是多余的一道关？
反过来：有没有该做而计划没要求的？
特别看第五条（查清 RQ009 三个 IPV 输入列的填充规则）——
它是必要的，还是不该塞进这一轮？


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

产出写到 `.codex-fleet/rq015k-fullcorpus-gate/board/reviews/KR_A_review.md`。
