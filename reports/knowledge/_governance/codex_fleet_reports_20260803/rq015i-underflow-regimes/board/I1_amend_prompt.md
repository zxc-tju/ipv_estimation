你是 I1，刚交付了 `board/reports/I1_underflow_regimes.md`。leader 自查通过了第 1 部分与第 3 部分，
但在第 2 部分发现一个**口径缺陷**，需要你做**一轮定向订正**。这不是复审，不要重做已经对的部分。

仓库根（$REPO）：
.
解释器：<local-rq009-venv>/bin/python

# 已通过 leader 独立复算、不要改动的部分

- 第 1 部分区间① 全语料实测：leader 独立重扫 4 份 parquet，逐 artifact 与你完全一致
  （interhub 172,094/187,340；rq009 756,805/778,298；onsite 366/366；wod 223/223；
  合计严格 929,488、容差 966,227 / 14,473,982）。**保留原样。**
- 样本基准复现、`partial_underflow == 区间②∪区间③` 的反推、区间③ = Mac 7 / HPC 9 行：
  leader 独立复算一致。**保留原样。**
- 第 3 部分可识别性结论：**保留原样。**

# 缺陷（必须修）

`zero_postwarm_scope == True` **恒等于 `signature ∈ {U, Z}`**，signature `N`（500 个锚点）
被整体排除在分母之外。经 leader 复算：

| signature | 锚点数 | 区间② Mac | 区间② HPC | nzero==6 Mac | nzero==6 HPC |
|---|---|---|---|---|---|
| **N（被排除）** | 500 | **145（29.0%）** | **143（28.6%）** | **7（1.40%）** | **7（1.40%）** |
| U | 1200 | 10（0.83%） | 0（0%） | 8（0.67%） | 0（0%） |
| Z | 600 | 9（1.50%） | 7（1.17%） | 2（0.33%） | 2（0.33%） |

**后果**：你报告的头条「区间② 设计基估计 Mac 1.417% / HPC 0.953%」实际是
**U/Z 域内**的占比，落掉了 88%（Mac 145/164）、95%（HPC 143/150）的已观测区间② 行。
区间② 主要是一个 **signature N 现象**，而现有头条数字对 N 一无所知。
读者若把它读成「区间② 在全语料的占比」会被严重误导。

同理，`nzero==6` 的 Mac/HPC 分歧（0.807% vs 0.372%）**全部来自 U 层**
（Mac U 有 8 行、HPC U 有 0 行）；在 N 层两版**完全一致**（各 7 行、1.40%，且 7 行全为 waymo）。
你原报告把这个分歧归因描述得不准确。

# 要做的四件事

## 1. 给所有既有第 2 部分数字重贴标签

把现有表格的每一行标题/表头明确改成 **「U/Z 域内（zero_postwarm_scope==True，等价于 signature∈{U,Z}）」**，
并在该节开头加一段醒目说明：该分母**整体排除 signature N**，而 N 恰是区间② 的主要发生层。
数值不用重算，只改标签与说明。

## 2. 新增 signature N 层的估计

- **Mac**：N 层的 `ht_weight` 是真实设计权重（4 个取值，对应 2 source × 2 n_band 单元），
  做 HT 比率估计 + cluster bootstrap CI。**B=2000、seed=20260731 不变**；
  cluster 仍为 `scene_unique_id`，但 N 层内只有 **491** 个 cluster（不是 1459）——
  这不是改 bootstrap 规格，是把同一套程序应用到另一个域，**在报告里写明这一点**。
- **HPC**：N 层的 `ht_weight` **全部为 1.0，是占位符不是设计权重**（Mac 对应权重和 2,111,119，
  HPC 只有 500）。因此 **HPC 的 N 层不得给 HT 加权估计**；只报未加权样本占比，
  并明确标注「HPC 侧 N 层缺设计权重，无法给出设计基估计」。
- 指标：区间②、区间③（第 0 节定义）、`nzero==6`，各按 source 分列（waymo/nuplan 分开）+ 合并行。

## 3. 新增全域（U∪Z∪N）估计，并标注其地位

用 Mac 权重做一个覆盖全部 2300 锚点的 HT 估计（cluster = 全部 1909 个 scene_unique_id，
B/seed 不变）。**明确标注**：这个分母是 D0–D4 从未使用过的口径，属于本轮新增的辅助口径，
不能与 D0–D4 的既有结论直接并列。HPC 侧同样因 N 层权重是占位符而不可给出，如实写明。

## 4. 改写「Mac/HPC 分歧」的结论段

按分层重述：
- N 层：两版区间② 占比几乎相同（29.0% vs 28.6%），`nzero==6` 完全相同（各 7 行，全 waymo）
- U 层：分歧集中在此（Mac 区间② 10 行 / HPC 0 行）
- 结论应是「分歧是 U 层现象，N 层两环境高度一致」，而不是笼统的「加权后差距缩小、CI 重叠」

# 报告结构要求

在报告**摘要部分**必须出现这句话的等价表述（不要写成因果主张）：
**区间② 在样本中集中于 signature N 层（Mac 29.0%、HPC 28.6% 的 N 锚点命中），
而 D0–D4 的既有分母口径整体排除了该层。**

`nzero==6`（七个候选被下溢删掉六个、只剩一个存活的事实 hard argmax）的 N 层数字
（Mac/HPC 各 7 行、1.40%、全为 waymo）要在摘要里单独点出。

保持原有诚实性要求：一切外推写明「设计基估计（design-based estimate）」并附 CI，
样本不足就写「CI 不可给出」，不要挤数。

# 硬约束（不变）

1. `rq007_split == 'held_out'` 断言保留
2. 不改 `agent.py`/`ipv_estimation.py`/`process_interhub.py`/`reliability_logdomain.py`
3. 禁止 `git checkout -- .` / `git restore .` / `git stash` / `git reset --hard` / `git clean -fd` / `git commit`
   —— 另一条 track 的 agent 在同一仓库工作，工作区非空是预期状态
4. 只写 `.codex-fleet/rq015i-underflow-regimes/work/` 与 `board/reports/` 下你自己的文件
5. 不重算锚点、不提交 HPC 作业、不提议全量重跑、不重新抽样
6. 全文禁用 `estimability` 与「测出/未测出 IPV」；可辩护表述是
   **权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息**
7. 描述性结果不得写成因果主张
8. 不要对 `reports/` 做全仓库 `rg`

直接原地更新 `board/reports/I1_underflow_regimes.md`（保留已通过的部分），
脚本增量写在 `work/i1_*.py`。时间戳用 `date -u +%Y-%m-%dT%H:%M:%SZ`。做完即止，不要问我问题。
