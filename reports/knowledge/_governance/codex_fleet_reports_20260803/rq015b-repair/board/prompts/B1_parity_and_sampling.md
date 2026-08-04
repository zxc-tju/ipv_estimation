# B1 — 抽样框物化、复现门、平价门（RQ015B，诊断性一轮）

你是 RQ015B 的执行 agent B1。项目根目录（下称 `$ROOT`）：
`.`

你的产出被下游 B2 直接使用。**本任务是诊断性的，一轮做完，不要做多版规格、不要开授权闸、
不要写治理文书。** 把结果跑出来、自查一遍数值健康与覆盖、出报告。

---

## 0. 铁律（违反即作废，全文照抄自上游，不要去别处找）

1. **RQ007 held_out 集不得被解析。** 抽样框只能取 `development` + `guard`。
   实现方式：流式读取时**先且仅先**取第 1 个字段（0-based index 1，`scene_unique_id`）判 split；
   split 不属 `{development, guard}` 的行**立即丢弃，不得解析其余任何字段**。
2. **不得读取任何 rating / preference / score 字段**，也不得读
   `PET`、`intensity`、`priority_label`、`turn_label`、`path_category`、`path_relation`、`actual_order`。
3. **不得接线生产估计器。** `src/sociality_estimation/core/reliability_logdomain.py`
   保持 `BUILD_WHILE_DENY`，**不得**被 import 进 `agent.py` / `ipv_estimation.py` /
   `pipelines/` 的任何生产路径。你只能**只读地调用**它。
   **不得修改 `$ROOT` 下任何 git-tracked 文件**（`src/`、`pipelines/`、`configs/`、
   `reports/`、`*.md` 一律不许改）。
4. **不得覆盖任何冻结产物。** `data/derived/` 下只读。你的**全部**输出只能写进
   `$ROOT/.codex-fleet/rq015b-repair/`（该目录已被 `.gitignore` 忽略）。
5. **不写因果措辞**；**禁止**使用 estimability 一词，**禁止**"测出 / 未测出 IPV"的表述。
   唯一可辩护的说法：**权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息**。

**派发时刻的基线（leader 于 2026-07-31T09:11Z 实测记录，以此为准）**：
```
HEAD                     = e82091ceaa2586bdb09b6153dfbed3be24d6bf98
git status --porcelain   = 空（工作区完全干净，0 行）
sha256 agent.py               = bde0f58258e915feb90eeb89d716632db95051d5b9d0a98abe9898cacd1da9f7
sha256 reliability_logdomain.py = 8f740677eb2c3cfd0cba7e9785db9b1fba5cd4a40c0f6e0584bab5747eb8f830
```

**完工前自检（三条都要做，结果写进报告 §11）**：
1. `git status --short` 输出必须**仍然为空**。出现任何一行都说明你违反了铁律 3/4，
   必须自行回滚后再交付（你的产物全在 `.codex-fleet/` 内，该目录被 `.gitignore` 忽略，
   正常情况下不会出现在 `git status` 里）。
2. `git rev-parse HEAD` 必须仍是 `e82091ce...`。**不得 commit、不得 stash、不得切分支。**
3. 上面两个文件的 sha256 必须逐位不变。**`agent.py` 一字不动是本轮的硬条款**：
   本轮只测量那个兜底缺陷，**不修它**。若你"顺手修好了"，回滚并在报告 §11 写明。

---

## 1. 已确立的事实（直接用，不要重新推导）

### 1.1 核心数学恒等式

legacy 的可靠性权重是
```
var_i = [ ∏_k φ(d_ik) ]^(1/n) ,   φ(d) = (1/(σ√2π))·exp(−d²/(2σ²))
```
取对数：`log var_i = −log(σ√2π) − MSE_i/(2σ²)`，其中 `MSE_i = mean_k(d_ik²)`。
首项对所有候选相同、归一化时约掉，故 **legacy 恒等于 `w = softmax(−MSE_i/(2σ²))`**。
连乘是绕路，也**正是下溢发生之处**。

**因此平价检验不需要重解虚拟轨迹两次**：对每个抽样锚点**只解一次**拿到逐候选
`rel_dis`（进而 `MSE_i`），再用**同一组 MSE** 分别算"legacy 连乘"与"log 域"两种权重做对比。
这把数值问题与求解器问题彻底隔开。**你必须走这条路，不许为两种权重各解一次。**

### 1.2 估计器常量（已勘察，你须复核一次）

| 量 | 值 | 出处 |
|---|---|---|
| `K` | 7 | `src/sociality_estimation/core/agent.py:63` `virtual_agent_IPV_range` |
| 候选网格 | `np.array([-3,-2,-1,0,1,2,3]) * math.pi / 8`，**关于 0 对称** | 同上 |
| `sigma` | 0.1 | `agent.py:95` |
| `min_observation` | 4 | `configs/ipv_sigma01_exact.json` |
| `history_window` | 10 | 同上（`current_ipv_history_window`） |
| 观测长度 `n` | `min(frame_index, 10) + 1`（fi=4 → n=5；fi≥10 → n=11） | |
| `ipv` | `sum(ipv_range * weight)` | `agent.py:253` |
| `ipv_error` | `1 - sqrt(sum(weight**2))` | `agent.py:254` |

**待修缺陷原文**（`agent.py:1136-1139`，本轮**不改它**，只测量它）：
```python
if sum(var):
    weight = var / (sum(var))
else:
    weight = np.ones(candidates_num) / candidates_num   # 无信息被写成中性
```

**关键推论**：网格关于 0 对称 ⇒ 均匀权重必然给出 `ipv = 0.0`（精确）且
`ipv_error = 1 − 1/√7 = 0.622035526990773`。**兜底分支留下了可先验识别的指纹。**

### 1.3 下溢临界 RMS（已算出，直接采用；只做数值复核，不要重导）

| n | 进入 subnormal | 舍入为精确 0（触发兜底） |
|---:|---:|---:|
| 5 | 1.6915 m | 1.7336 m |
| 11 | 1.1470 m | 1.1752 m |

### 1.4 已扫描到的总体（dev+guard，`frame_index ≥ 4`，锚点 = 行 × agent_slot）

| source | 签名 | n 带 | 锚点数 |
|---|---|---|---:|
| nuplan | zero | fi≥10 | 145,241 |
| nuplan | zero | fi 4–9 | 5,193 |
| nuplan | nonzero | fi≥10 | 1,132,547 |
| nuplan | nonzero | fi 4–9 | 58,407 |
| waymo | zero | fi≥10 | 728,705 |
| waymo | zero | fi 4–9 | 33,085 |
| waymo | nonzero | fi≥10 | 1,696,335 |
| waymo | nonzero | fi 4–9 | 162,845 |

D0 warm-up（`frame_index < 4`，普查不抽样）：nuplan 42,432；waymo 130,624。
全 dev+guard 后热身锚点 4,981,984 中 `ipv == 0.0` 占 1,200,636 = **24.1%**。
`ipv_error` 恰为均匀签名 0.622036 的锚点共 **676,405** 个。

### 1.5 关键路径

- 抽样框主表：`$ROOT/data/derived/interhub/20260612_sigma_0_1_full_rerun/00_hpc_outputs/sigma01_ipv_timeseries.csv`（2.1 GB，3,695,981 数据行，39 列）
- split 表：`$ROOT/data/derived/interhub/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_outputs/splits/case_split_assignment.csv`（列：`case_id,split`）
- 原始 pkl：`$ROOT/data/interhub/raw/full_datasets/pkl/`
  （**本地只有** `train_singapore, train_vegas1..6`（nuplan）与 `waymo_0-299, waymo_300-499, waymo_800-999`）
- 执行档：`$ROOT/configs/ipv_sigma01_exact.json`（生成 sigma01 产物的那一份）
- 管线：`$ROOT/pipelines/interhub/process_interhub.py`
- log 域原型：`$ROOT/src/sociality_estimation/core/reliability_logdomain.py`（+ `tests/test_rq015_reliability_logdomain.py`，36 tests）
- **抽样合同（只读，不得修改）**：`$ROOT/.codex-fleet/rq015b-repair/board/sampling_contract_v1.md`

主表列序（0-based）：`0 source_row, 1 scene_unique_id, 2 dataset, 3 folder, 4 scenario_idx,
5 track_id, ..., 21 frame_index, 22 timestamp, 23 key_agent_1, 24 key_agent_2,
25 ipv_key_agent_1, 26 ipv_key_agent_1_error, 27..31 key_agent_1_{px,py,vx,vy,heading},
32 ipv_key_agent_2, 33 ipv_key_agent_2_error, 34..38 key_agent_2_{px,py,vx,vy,heading}`。

---

## 2. 任务（按顺序，每步有硬门）

### T1 — 物化抽样框

按 `sampling_contract_v1.md` §2 的纳入条件建 frame 索引，落盘
`$ROOT/.codex-fleet/rq015b-repair/work/frame_index.csv`，列：
`anchor_id, scene_unique_id, dataset, source, folder, scenario_idx, frame_index,
agent_slot, n_obs, legacy_ipv, legacy_ipv_error, signature, n_band, split, pkl_available`。
`anchor_id = f"{scene_unique_id}|{frame_index}|{agent_slot}"`。

签名判定（合同 §3）：
- `U`：`legacy_ipv == 0.0` 且 `|legacy_ipv_error − 0.622035526990773| ≤ 1e-6`
- `Z`：`legacy_ipv == 0.0` 且非 U
- `N`：`legacy_ipv != 0.0`

**同一遍扫描里额外统计并报告**（用于对齐"约四成零值"的既有认知，这是上游明确要的）：
`ipv == 0.0`、`|ipv| < 1e-6`、`|ipv| < 1e-3` 三种口径各自在
(a) 全体锚点 (b) 后热身锚点 上的占比，dev+guard 与全体（含 held_out 的**计数**可由
总行数减出，**不要为此解析 held_out 行**）分别给出。若某口径能解释"四成"，明说是哪个。

**pkl 可达性**：把 `folder` 映射到 pkl 文件名，标记 `pkl_available`。
不可达的源（lyft / av2 / waymo 500-799）**保留在 frame_index.csv 里并标 False**，
但**不进入抽样**。逐源报可达 / 不可达锚点数。

### T2 — 抽样（确定性，无 RNG）

按合同 §4/§5 抽 2,300 个锚点：配额 U=300、Z=150、N=125 **每格**
（格 = source{nuplan,waymo} × n_band{RAMP=fi 4–9, FULL=fi≥10} × signature{U,Z,N}，共 12 格）；
每 `(scene_unique_id, stratum)` 最多 3 个锚点；
排序键 `h = sha256("RQ015B_parity_sample_v1::20260731::outcome_blind::" + scene_unique_id + "::" + str(frame_index) + "::" + str(agent_slot)).hexdigest()` 升序。
RAMP 各格内**先填满 `frame_index == 4` 的 25% 名额再填其余**。
亏格**全取且不转移**，逐格登记 `quota / drawn / shortfall`。

落盘 `work/sample_v1.csv`，并把 `frame_index.csv` 与 `sample_v1.csv` 的
**SHA-256 写进 `work/sample_v1.sha256`**。

### T3 — 复现门（**硬门；不过就停**）

先取样本中的 **40 个锚点**（跨 12 格均匀取，nuplan/waymo 各半）做试点：
用 `configs/ipv_sigma01_exact.json` 的参数、经
`pipelines/interhub/process_interhub.py` **自身的**案例装载/对齐路径重建
`MotionSequence`（**target 标签必须由该模块的 `classify_heading` 从几何重新导出，
不许读 `turn_label` 列**），调用 `estimate_ipv_pair(..., return_diagnostics=True,
diagnostic_steps=[目标 frame])` 取回该帧的 `virtual_tracks / weights / ipv_range`。

**门 A（外部一致）**：重解得到的 `ipv`、`ipv_error` 与主表存档的
`legacy_ipv`、`legacy_ipv_error` 相比，**≥ 39/40 个锚点**满足两者绝对差 ≤ 1e-6。
**门 B（内部一致）**：你自己从 `virtual_tracks` 与实际轨迹窗口算出的 `rel_dis`
再走 legacy 连乘公式所得权重，与诊断里返回的 `weights` **逐位一致（≤ 1e-15）**——
这证明你的 `act_track` 窗口切法与生产一致。

**任一门不过：立即停止，不要继续 T4/T5**，写
`board/BLOCKED_B1.md` 说明失配的具体形态（差多少、系统性偏移还是个别案例、
最可能的参数分歧点），并把已跑通的部分落盘。这是真 blocker：
复现不了存档，后续所有机制比例都无意义。

### T4 — 吞吐量与规模决策

在试点上测 `秒/锚点`（含 7 个候选的 SLSQP 求解），报可用核数与拟用并行度，
外推 2,300 锚点的墙钟。
- 若外推 **≤ 6 小时**：直接跑全量 2,300。
- 若 **> 6 小时**：**不要自行缩减样本量**（缩减会破坏冻结合同）。停下，写
  `board/BLOCKED_B1.md` 报出实测吞吐与建议方案（更高并行度 / 缩配额 / 上 HPC），等指示。

### T5 — 解算与平价门

对 2,300 个锚点各**解一次**，落盘
`work/anchor_mse.parquet`（或 csv），每锚点一行，含：
`anchor_id, source, n_band, signature, frame_index, n_obs, K,
mse_per_candidate[7], rms_per_candidate[7], min_mse, argmin_candidate,
legacy_prod_sum (即 sum(var)), legacy_fallback_triggered (bool),
w_legacy[7], w_log[7], max_abs_diff, ipv_legacy, ipv_log,
ipv_error_legacy, ipv_error_log, k_eff_legacy, k_eff_log,
at_grid_boundary (argmax 落在候选网格端点), any_nonfinite`。

- `w_legacy` 必须由**生产函数本身**（`agent.cal_traj_reliability`，只读调用）产生，
  不许你重写一份"等价实现"——平价必须对着真实 legacy 比。
- `w_log` 由 `reliability_logdomain.py` 产生。
- 两者**共用同一组重解结果**。

**平价门判据（硬性）**

1. **未触发兜底且未部分下溢的锚点**上，`max_abs_diff ≤ 1e-12`。逐格报
   `max / p99 / median` 的 `max_abs_diff`，并给出全样本最大值。
2. **已触发兜底的锚点**上，legacy 走均匀兜底，而 log 域给出**非均匀**权重
   （报这些锚点上 log 域的 `k_eff` 分布——若 log 域也近均匀，说明这些行本就平坦，
   这是重要发现，如实报）。
3. **兜底不可达证明**：log 域中 `logw -= logw.max()` 后至少一项为 0 ⇒ `exp` 至少一项为 1
   ⇒ 分母 ≥ 1，除零分支不可达。给出**论证 + 一个数值实验**
   （构造 RMS 远超 1.7336 m 的极端输入，验证 log 域仍产出有限非均匀权重）。
4. **临界值复核**：用实测数据验证 §1.3 的四个临界 RMS，报实际观察到的
   兜底触发起点与理论值之差（容差 5e-4）。

### T6 — 数值健康自查（一轮，不要展开成审计）

至少查：`mse_per_candidate` 有无 NaN/Inf/病态常数；`min_mse` 分布；
每格实抽数与案例数；同一案例贡献的锚点数分布；
`legacy_fallback_triggered` 与签名 `U` 的一致性（**理论上 U ⊇ 兜底行**，
若出现"触发兜底但签名不是 U"或大量"签名 U 但未触发兜底"，如实报并解释——
后者恰恰说明那些行是**真平坦**而非下溢，是本轮最有价值的信号之一）。

---

## 3. 结项报告（**有界**：≤ 250 行 Markdown）

写到 `$ROOT/.codex-fleet/rq015b-repair/board/reports/B1_report.md`，**严格按此结构**：

```
# B1 结项报告 — 抽样框、复现门、平价门
## 1. 一句话结论（平价门过 / 不过；兜底是否不可达）
## 2. 抽样框与覆盖（表：逐源可达/不可达锚点数；零值口径对齐表）
## 3. 冻结样本（表：12 格 quota/drawn/shortfall/案例数；两个 SHA-256）
## 4. 复现门（门 A、门 B 的实测数字）
## 5. 吞吐量与实际规模
## 6. 平价门（4 条判据逐条给数字；未下溢锚点 max_abs_diff 的 max/p99/median）
## 7. 兜底行上 log 域的表现（k_eff 分布 —— 修得好还是本就平坦的第一个信号）
## 8. 临界 RMS 复核（理论 vs 实测）
## 9. 数值健康自查（发现的问题，没有就写"无"）
## 10. 交给 B2 的产物清单（路径 + SHA-256 + 每列含义一行说明）
## 11. 偏差与未做的事（有就写，没有写"无"；不得静默截断）
```

**报告纪律**：不写因果措辞；不用 estimability / "测出 IPV"；
不把描述性结果上升为对方法的断言；每个数字给出它的分母。

## 4. 输出位置汇总

- `$ROOT/.codex-fleet/rq015b-repair/work/` — frame_index.csv、sample_v1.csv、
  sample_v1.sha256、anchor_mse.parquet、你写的脚本
- `$ROOT/.codex-fleet/rq015b-repair/board/reports/B1_report.md` — 结项报告
- `$ROOT/.codex-fleet/rq015b-repair/board/BLOCKED_B1.md` — 仅在触发停止条件时写

## 4b. 心跳（必做，leader 靠它判断你是否存活）

每完成一个 T 步（T1…T6），**立即**向 `$ROOT/.codex-fleet/rq015b-repair/board/B1_heartbeat.log`
**追加**一行：`<UTC ISO8601> | T<k> | <一句话结果或关键数字>`。
T5 是长任务，额外要求：每处理约 200 个锚点追加一行
`<UTC> | T5-progress | <已完成锚点数>/2300`。
心跳只是给 leader 看的，不要写成报告，不要因为写心跳而中断计算。

## 5. 停止条件（触发即停，写 BLOCKED_B1.md）

1. 任何 held_out 行被解析（哪怕一次）
2. 需要读取评分 / 偏好字段才能继续
3. 需要修改 `src/` 或 `pipelines/` 下任何 tracked 文件才能继续
4. T3 复现门不过
5. T4 外推墙钟 > 6 小时
6. 抽样结果与"零值约占四成"的既有认知出现**数量级**冲突
   （提示：已知精确零值口径为 24.1%，这**不算**数量级冲突，不要为此停）

开工。
