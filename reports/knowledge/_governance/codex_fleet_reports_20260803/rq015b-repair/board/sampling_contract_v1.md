# RQ015B 抽样合同 v1 — 冻结于 2026-07-31T09:00Z（B1 派出之前）

冻结人：track B leader。冻结依据：`board/plan.md` §抽样合同、
`reports/plans/RQ015B_plan_v0_estimator_repair_and_abstain_gate_20260726.md` §6.1/§10。

**本文件在任何重解运行开始之前写定。** 此后 B1/B2 对本文件**只读**。
若执行中发现合同不可满足，正确动作是**停下并报告偏差**，不是就地改合同。

---

## 0. 已确立的估计器常量（勘察自源码，B1 须逐条复核）

| 量 | 值 | 出处 |
|---|---|---|
| 候选数 `K` | 7 | `agent.py:63` `virtual_agent_IPV_range` |
| 候选网格 | `[-3,-2,-1,0,1,2,3] × π/8` | 同上，**关于 0 对称** |
| `sigma` | 0.1 | `agent.py:95` |
| `min_observation` | 4 | `configs/ipv_sigma01_exact.json` |
| `history_window` | 10 | 同上 (`current_ipv_history_window`) |
| 观测长度 `n` | `min(frame_index, 10) + 1` | 即 fi=4 → n=5；fi≥10 → n=11 |
| `ipv` | `Σ ipv_range · w` | `agent.py:253` |
| `ipv_error` | `1 − sqrt(Σ w²)` | `agent.py:254` |

**推论（本轮抽样设计的支点）**：网格关于 0 对称 ⇒ 均匀权重 `w = 1/7`
必然给出 `ipv = 0.0`（精确）且 `ipv_error = 1 − 1/√7 = 0.622035526990773`。
因此**兜底分支留下了可先验识别的指纹**，无需重解即可定位疑似兜底行。

**下溢临界 RMS（计划已算出，直接采用，B1 只做数值复核不重导）**

| n | 进入 subnormal | 舍入为精确 0（触发兜底） |
|---:|---:|---:|
| 5 | 1.6915 m | 1.7336 m |
| 11 | 1.1470 m | 1.1752 m |

---

## 1. 分析单元

**锚点 (anchor)** = `(scene_unique_id, frame_index, agent_slot)`，`agent_slot ∈ {1, 2}`。
一行 CSV 贡献 2 个锚点。**聚类单元** = `scene_unique_id`（与 RQ007 split 的划分单元一致）。

## 2. 抽样框 F（frame）及其覆盖边界

**来源表**：`data/derived/interhub/20260612_sigma_0_1_full_rerun/00_hpc_outputs/sigma01_ipv_timeseries.csv`
（3,695,981 数据行）。

**纳入条件（全部满足）**

1. `split(scene_unique_id) ∈ {development, guard}`，split 取自
   `data/derived/interhub/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_outputs/splits/case_split_assignment.csv`；
2. `frame_index ≥ 4`（`frame_index < 4` 为 D0 warm-up，按**普查**计数，不抽样）；
3. `dataset` 归一化后 ∈ `{nuplan, waymo}`；
4. 该案例的原始 pkl **本地存在**于 `data/interhub/raw/full_datasets/pkl/`。

**held_out 铁律的实现方式**：流式读取时**先且仅先**取字段 1 (`scene_unique_id`)
判 split；非 development/guard 的行**立即丢弃，不解析其余任何字段**。

**覆盖边界（必须在报告里显式声明，禁止静默截断）**

- `lyft_train_full`（dev+guard 后热身 785,080 锚点）与 `av2_motion_forecasting`（同口径 234,546 锚点）
  **无本地原始数据，本轮完全不可达**；
- waymo 缺 `500-799` 分片；
- 因此本轮 D0–D4 比例的外推范围是"**本地可达的 nuplan + waymo 子集**"，
  不是 InterHub 全体。报告标题与结论句必须带上这一限定。

**已扫描到的总体（dev+guard，`frame_index ≥ 4`，尚未施加 pkl 可达性筛选）**

| source | 签名 | n 带 | 锚点数 |
|---|---|---|---:|
| nuplan | zero | fi≥10 (n=11) | 145,241 |
| nuplan | zero | fi 4–9 | 5,193 |
| nuplan | nonzero | fi≥10 | 1,132,547 |
| nuplan | nonzero | fi 4–9 | 58,407 |
| waymo | zero | fi≥10 | 728,705 |
| waymo | zero | fi 4–9 | 33,085 |
| waymo | nonzero | fi≥10 | 1,696,335 |
| waymo | nonzero | fi 4–9 | 162,845 |

D0 warm-up（普查，不抽样）：nuplan 42,432；waymo 130,624。

## 3. 分层（12 格）

`stratum = source × n_band × signature`

- `source ∈ {nuplan, waymo}`
- `n_band ∈ {RAMP (frame_index 4–9, n=5..10), FULL (frame_index ≥ 10, n=11)}`
- `signature ∈ {U, Z, N}`：
  - **U（疑似均匀兜底，加密层）**：`ipv == 0.0` 且 `|ipv_error − 0.622035526990773| ≤ 1e-6`
  - **Z（其余零值）**：`ipv == 0.0` 且不属 U
  - **N（非零）**：`ipv != 0.0`

**U 的语义必须写对**：U 同时容纳"数值下溢触发兜底"与"似然真平坦"两种情形
（两者都给均匀权重）。**U 不等于 D1**。区分它们正是重解要回答的问题。

## 4. 配额（固定数，不随执行调整）

| signature | 每格配额 | 格数 | 小计 |
|---|---:|---:|---:|
| U | 300 | 4 | 1,200 |
| Z | 150 | 4 | 600 |
| N | 125 | 4 | 500 |
| **合计** | | 12 | **2,300** |

**每案例上限**：同一 `(scene_unique_id, stratum)` 内最多取 **3** 个锚点（压低设计效应）。

**亏格规则（唯一合法处置，不得再分配）**：若某格可用总体 < 配额，则**全取**该格，
差额**不转移到其他格**，并在报告中逐格登记 `quota / drawn / shortfall`。

**n=5 覆盖硬要求**：RAMP 各格所抽锚点中，`frame_index == 4`（n=5）者
**不得少于该格实抽数的 25%**；按 §5 排序在 RAMP 格内**先填满 n=5 的 25% 名额，再填其余**。
理由：n=5 的下溢临界（1.6915 / 1.7336 m）必须被样本跨越。

## 5. 抽样规则（确定性，不用 RNG）

对每个锚点计算
`h = sha256("RQ015B_parity_sample_v1::20260731::outcome_blind::" + scene_unique_id + "::" + frame_index + "::" + agent_slot)`，
在每格内按 `h` 的十六进制升序排列，依次取入，跳过已达每案例上限（3）的案例，
直到填满配额。**seed salt 即上述字符串常量，不使用任何随机数发生器**，
因此结果与 numpy / python 版本无关，可逐位复现。

## 6. 精度合同

- **主估计量**：零值总体（U ∪ Z）中 D1/D2/D3/D4 各自的**框加权**占比。
  框加权 = 每格权 `N_cell / n_cell`（Horvitz–Thompson），**因 U 被刻意加密，
  未加权比例不得作为结论数字**。
- **不确定度**：case-cluster bootstrap，按 `scene_unique_id` 重抽，
  **B = 2000，seed = 20260731**，percentile 95% CI。
- **精度上限**：主估计量的 95% CI 半宽 **≤ 3 pp**。
  超出则**照实报告并标注**，**不得**追加抽样或改配额（追加会破坏冻结）。

## 7. 后分层（RMS 带）

残差 RMS 只有重解后才知道，故**不作抽样分层，只作后分层**。冻结带界（m）：

`[0, 1.1470) | [1.1470, 1.1752) | [1.1752, 1.6915) | [1.6915, 1.7336) | [1.7336, ∞)`

即两组临界值的并集，n=5 与 n=11 的两个临界都是带边界。逐带报格数与机制构成。

## 8. 机制判据（沿用计划 §6，不新造）

优先级 **D4 > D1 > D3 > D2 > OK**；D0 由 `frame_index < 4` 上游分流。

| 机制 | 判据 |
|---|---|
| `D4_SOLVER_OR_INPUT_FAILURE` | 非有限输入 / 平方距离溢出 / 求解器异常 |
| `D1_NUMERICAL_UNDERFLOW` | **legacy 与 log 域权重的最大绝对差 > 1e-6**（即 legacy 结果已被下溢改变） |
| `D3_MODEL_MISFIT` | `min_mse > min_mse_misfit`（阈值见下） |
| `D2_FLAT_UNDER_CURRENT_GRID_AND_MODEL` | `c = K_eff/K ≥ 0.93` 且 `min_mse` 未超阈 |
| `OK` | 其余 |

**D1 的定义是"legacy 结果是否被改变"，不是"是否发生过下溢"**：远距候选被清零
属无害部分下溢，判 `OK`（flag 仍如实记录）。

**`min_mse_misfit` 的本轮处置**：计划 §6.1 要求 `Q_0.99(min_mse)` 在分类**之前**冻结并登记 SHA。
本轮该分位数只能由本抽样样本估计（全量不可得），故：
B2 **先**在样本上算 `Q_0.99`、写入独立文件、登记 SHA-256，**再**跑分类；
并在报告中标注**该阈值为样本估计、属本轮临时口径，不得据此冻结生产阈值**。
同时报 `p = 0.95 / 0.999` 的敏感性。

## 9. 禁止读取的字段

`PET`、`intensity`、`priority_label`、`turn_label`、`path_category`、`path_relation`、
`actual_order`，以及任何 rating / preference / score 字段。
**允许读取的列**仅限：`scene_unique_id, dataset, folder, scenario_idx, track_id,
frame_index, timestamp, key_agent_1, key_agent_2, ipv_key_agent_1, ipv_key_agent_1_error,
ipv_key_agent_2, ipv_key_agent_2_error` 及两个 agent 的 `px, py, vx, vy, heading`。
（重解所需的 target 标签由 `pipelines/interhub/process_interhub.py` 自身的
`classify_heading` 从几何重新导出，**不得**读 `turn_label` 列。）

## 10. 与冻结产物的关系

只读 `sigma01_ipv_timeseries.csv` 与 pkl，**不写、不覆盖任何 `data/derived/` 下产物**。
所有输出写入 `.codex-fleet/rq015b-repair/`（该目录被 `.gitignore` 第 23 行忽略）。
