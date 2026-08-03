# 独立审查 B（K2 执行方案）：先自己把方案推一遍，最后再对照

你是**独立审查方 B**。同时另有一位审查方在独立工作，你们互不可见——
**不要试图寻找或读取对方的产物。** 最终由监督方比对你们的分歧。

被审对象是一份**已获 PI 授权、尚未派发**的全语料重算执行任务书。
本轮复审出结论、监督方裁定之后才会启动。你的判断是有后果的。

## 你的入口：**先不要读任何方案文档。** 先自己推一遍

**执行顺序是强制的**，请按此进行并在报告里体现。

### 第一阶段（在读任何方案/建议之前完成）

第一阶段**禁止打开**这三样东西：
- `.codex-fleet/rq015k-fullcorpus-gate/board/K2-leader-kickoff.md`（被审的任务书）
- `.codex-fleet/rq015k-fullcorpus-gate/board/reports/K1b_memory_pilot.md`（它含一条资源**建议**，会锚定你）
- `.codex-fleet/rq015k-fullcorpus-gate/board/commander_notes.md`（含监督方的指令）

第一阶段**可以**读的：
- 原始实测产物：`.codex-fleet/rq015k-fullcorpus-gate/work/k1b_memory_pilot/`（各 config 的 `k1b_pilot_summary.json`、`k1b_sample_summary.json`、`k1b_progress.log`、`k1b_sacct_final.txt`、`cluster_snapshot_pre_submit.txt`）
  与 `.codex-fleet/rq015k-fullcorpus-gate/work/k1_pilot_summary.json`、`hpc_frozen_pkl_listing.tsv`
- `.codex-fleet/rq015k-fullcorpus-gate/board/reports/K1_preflight_and_plan.md`（勘察结论：范围与单元数）
- `.codex-fleet/rq015j-gate-spec/board/reports/J1_gate_spec_and_impact.md`（门规格定稿与设计基估计）
- `.codex-fleet/rq015g-hpc-resolve/board/reports/G_leader_adjudication.md`（HPC 通道与确定性证据）
- 台账：`reports/studies/RQ015A_ipv_estimability_labelling/RQ015A_1_concentration_audit_20260731T093746Z_e82091ce/concentration_ledger/` 下 4 份 parquet
- 代码：`src/sociality_estimation/core/reliability_logdomain.py` 等

门的规格（**冻结，不许改**，这是预先给你的唯一方案性输入）：

```text
log_score_i = -mse_i / (2 * sigma^2)      sigma = 0.1
w_log       = softmax(log_score)          用 log-sum-exp
mse_spread  = max(mse_per_candidate) - min(mse_per_candidate)
max_w_log   = max(w_log)
k_eff_log   = 1 / sum(w_log_i^2)

if 输入非有限 / 缺列 / 求解失败:  ENGINEERING_FAILURE (NON_FINITE_INPUT | SOLVER_FAILURE)
elif mse_spread == 0:            ABSTAIN, NO_IPV_EFFECT
elif max_w_log < 0.20:           ABSTAIN, NEAR_UNIFORM
else:                            OK, ipv_log = sum(candidate_ipv_i * w_log_i)
```
候选网格 `legacy7_pi_over_8`（7 点 `[-3..3]·π/8`），`K = 7`，`theta = 0.20` 是政策阈值不是数据断点。

任务：**在不看任何方案的前提下，自己回答「把这道门物化到 InterHub 全语料上，该怎么做」。**

1. **规模**：全量真实求解单元数是多少？回到台账与 K1 的原始产物核，不要采信任何转述。
   RQ009 那 8,994,736 行需要重解吗，还是能由 InterHub 结果 join 回填？你自己判。
2. **资源**：给定 K1b 实测的每 worker 峰值 RSS 与吞吐（P6/P10/P16 三点），
   以及**你自己现查的**集群实况（只读查询方法见下），
   给出你认为最优的（片数 / 每片 worker 数 / `--mem` / 总核·小时 / 墙钟）。
   **明确写出到底是哪个约束在起作用**：QOS 的 4,000 核合计上限、实际空闲核、实际空闲内存、还是单节点内存上限。
   `--mem` 的余量按什么规则定，为什么。
3. **分片与幂等**：分片键怎么选，才能保证不重不漏、可断点续算、重投幂等？
   「一个分片已完成」的判据应该由哪几项共同构成？各类失败该怎么分类、各自重投几次、什么条件下必须整体停止？
4. **验收**：这是一次**普查**（不是抽样）。普查该用什么判据验收？
   J 轨用抽样给出过一个设计基点估计与置信区间（域、权重、B、seed 见 J1 报告）。
   **普查结果该不该跟那个区间比？两边的分母与域是不是同一个东西？** 自己去核，给结论。
   除此之外还该有哪些判据？
5. **交付**：产出要长什么样，下游 RQ009 才能直接用？
   有没有哪一条**必须随台账一起交付的警告**，否则 RQ009 一定会用错？

### 第二阶段（第一阶段全部写完之后才做）

现在读被审的任务书 `.codex-fleet/rq015k-fullcorpus-gate/board/K2-leader-kickoff.md`，
以及 `.codex-fleet/rq015k-fullcorpus-gate/board/reports/K1b_memory_pilot.md`（含它自己的资源建议）。

注意：任务书第四节的资源配置是**监督方推翻 K1b 建议后自己算的**，
两者结论不同。**你的第一阶段结果是独立的第三方计算。** 三方逐项对照：
- 你推出来而任务书**没有**要求的 → 遗漏
- 任务书要求而你认为**不必要**的 → 冗余
- 三方结论**冲突**的 → 逐条给证据，指明谁错、错在哪一步

**报告里必须明确标出哪些结论是第一阶段得出的（未看方案）。** 这是本次双盲的关键，不要事后回填。

## 共同的三个收尾问题（两位审查方措辞一致，便于比对）

**Q1. 第四节的资源配置正确吗？** 给出你独立算出的
（片数 / 每片 worker / `--mem` / 总核·小时 / 墙钟），并指明监督方错在哪一步（若有）。
另说明：**按你的配置投出去，最坏情况是什么。**

**Q2. 第六节的三条验收判据，能不能真正判定 K2 成功？**
逐条给出漏判情形与误报情形。缺哪条？多哪条？

**Q3. 明确判定：`可执行` / `需修改后执行` / `不应执行`。** 给理由。
另单列一条：**你认为这份任务书里最可能造成实际损失的一处。**

## 硬约束（违反即为审查失败）

- **你是审查方，不执行 K2。** 不得 `sbatch` / `srun` / `salloc`，不得重解任何锚点，不得建 K2 的 work_dir
- 允许**只读**的集群查询：`ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc "<命令>"`，
  命令限于 `sinfo` / `squeue` / `sacct` / `sacctmgr show assoc` / `ls` / `du`。
  查到的原始输出连同 `date -u +%Y-%m-%dT%H:%M:%SZ` 存进你自己的工作文件，报告里引用它
- **禁止读取另一位审查方的任何文件**：不得打开 `board/reviews/` 下不属于你的报告，不得读不属于你的 `.log`
- 只读；除你自己的报告与 `work/` 下你自己的临时脚本外不写任何文件
- **不改** `agent.py` / `ipv_estimation.py` / `process_interhub.py` / `reliability_logdomain.py` / `configs/ipv_sigma01_exact.json`
- 不提交 git commit；禁止 `git checkout -- .` / `git restore .` / `git stash` / `git reset --hard` / `git clean -fd`
- 工作区非空是**预期状态**（此前轨道留下的文件仍在），清洁性只查你自己创建的文件
- 不解析 RQ007 held_out，不读 RQ014 致盲字段
- 不对 `reports/` 做全仓库 `rg`（会把 RQ003 controlled-access 行拉进上下文）
- 时间戳一律 `date -u +%Y-%m-%dT%H:%M:%SZ`，**不要前瞻估计**
- 解释器钉死 `<local-rq009-venv>/bin/python`
- 全文禁用 `estimability` 与「测出/未测出 IPV」的说法

写作要求：不用比喻和行话；每个数字带来源文件与字段名或行号；
**与任务书或监督方不一致时，直接写「计划错了」并给出正确值，不要客气。**
一轮到底：不写第二版规格，不提替代判据，不做阈值扫描。

产出写到 `.codex-fleet/rq015k-fullcorpus-gate/board/reviews/K2R_B_review.md`。
