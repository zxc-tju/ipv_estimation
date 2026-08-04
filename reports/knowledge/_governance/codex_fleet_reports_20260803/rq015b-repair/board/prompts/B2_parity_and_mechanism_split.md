# B2 — 平价门 + 机制拆分 + 结项报告（RQ015B，诊断性一轮，收尾）

你是 RQ015B 的执行 agent B2。项目根目录（下称 `$ROOT`）：
`.`

上游 B1 已完成抽样框物化与抽样，**在 T3 复现门失败后按设计停下**。
监督方已裁定：**放行，继续 T5**。你接着往下做，**不要重做 B1 已做的部分**。

**本任务是诊断性的，一轮做完。** 不要做多版规格、不开授权闸、不写治理文书、
不自己给自己加审计轮次。把结果跑出来、自查一遍数值健康与覆盖、出报告，结束。

---

## 0. 铁律（违反即作废，全文照抄，不要去别处找）

1. **RQ007 held_out 不得被解析。** 你只用 B1 冻结的样本，样本内全部是
   `development` / `guard`。**不得**回到主表去捞任何新行。
2. **不得读取任何 rating / preference / score 字段**，也不得读
   `PET`、`intensity`、`priority_label`、`turn_label`、`path_category`、
   `path_relation`、`actual_order`。target 标签由几何重新导出，**不得**读 `turn_label`。
3. **不得接线生产估计器。** `src/sociality_estimation/core/reliability_logdomain.py`
   保持 `BUILD_WHILE_DENY`，**不得**被 import 进 `agent.py` / `ipv_estimation.py` /
   `pipelines/` 任何生产路径。你只能**只读地调用**它。
   **`src/sociality_estimation/core/agent.py` 必须一字不动**——本轮只测量那个兜底缺陷，**不修它**。
4. **不得修改 `$ROOT` 下任何 git-tracked 文件。** `src/`、`pipelines/`、`configs/`、
   `reports/`、`tests/`、`scripts/`、`*.md` 一律不许改、不许新增。
   你的**全部**输出只能写进 `$ROOT/.codex-fleet/rq015b-repair/`（该目录已被 `.gitignore` 忽略）。
   **包括你要写的 pytest 测试**——写进 `.codex-fleet/rq015b-repair/work/`，**不要**放进 `$ROOT/tests/`。
5. **【最高优先级】禁止任何形式的工作区变更或回滚。** 明令禁止执行：
   ```
   git checkout（含 checkout 任何历史提交/分支）   git restore     git stash
   git reset（任何模式）   git clean    git add    git commit    git rm    git mv
   ```
   **工作区非空是【预期状态】**：track A 的 agent 正在**同一棵工作树**里并发工作，
   已产出多个 run 目录与十余个 dirty 条目。
   **一次历史 checkout 会当场毁掉 track A 这一轮的全部工作。**
   你只对**自己创建的文件**负责；其他文件一律不碰、不回滚、不提交。
   需要清洁性自检时，**只列你自己写的文件清单**，不要因为 `git status` 非空就采取任何动作。
6. **不写因果措辞**；**禁止**使用 estimability 一词；**禁止**"测出 / 未测出 IPV"的表述。
   唯一可辩护的说法：**权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息**。

**完工自检（写进报告 §10）**：`git rev-parse HEAD` 必须仍是
`e82091ceaa2586bdb09b6153dfbed3be24d6bf98`；
`shasum -a 256 src/sociality_estimation/core/agent.py` 必须仍是
`bde0f58258e915feb90eeb89d716632db95051d5b9d0a98abe9898cacd1da9f7`；
`src/sociality_estimation/core/reliability_logdomain.py` 必须仍是
`8f740677eb2c3cfd0cba7e9785db9b1fba5cd4a40c0f6e0584bab5747eb8f830`。
三者任一变化 = 你违反了铁律，必须在报告里明写（**但仍然不许回滚**）。

---

## 1. 已确立的事实（直接用，**不要重新推导、不要复核第三遍**）

监督方已独立复算并背书以下常数，**不要再验一遍**：

```
候选数 K = 7 ；网格 = [-3,-2,-1,0,1,2,3]·π/8（关于 0 对称）；sigma = 0.1
min_observation = 4 ；history_window = 10 ；n_obs = min(frame_index,10)+1
ipv       = Σ(ipv_range · weight)
ipv_error = 1 − sqrt(Σ weight²)

均匀兜底指纹： ipv = 0.0（精确） 且 ipv_error = 1 − 1/√7 = 0.6220355269907728
softmax 恒等式（**恒等，不是近似**）：
    var_i = [∏_k φ(d_ik)]^(1/n) ,  φ(d) = (1/(σ√2π))·exp(−d²/(2σ²))
    ⇒ log var_i = −log(σ√2π) − MSE_i/(2σ²) ，首项对所有候选相同、归一化时约掉
    ⇒ legacy 恒等于  w = softmax(−MSE_i/(2σ²))
    连乘是绕路，也正是下溢发生之处。
下溢临界 RMS（闭式）： n=5 → subnormal 1.691526 m / 归零 1.733619 m
                       n=11 → subnormal 1.147025 m / 归零 1.175245 m
```

**待修缺陷原文**（`agent.py:1136-1139`，本轮**只测量不修改**）：
```python
if sum(var):
    weight = var / (sum(var))
else:
    weight = np.ones(candidates_num) / candidates_num   # 无信息被写成中性
```

### 1.1 B1 已交付、你直接用的产物

| 路径（相对 `$ROOT`） | 内容 |
|---|---|
| `.codex-fleet/rq015b-repair/work/sample_v1.csv` | **冻结样本 2,300 锚点**，sha256 `d27f10907b7ca8da5815a6b832859d64a40b7fbf41aa0e5587c51bec8466759e` |
| `.codex-fleet/rq015b-repair/work/t2_summary.json` | 12 格 `available / drawn / quota / shortfall / case_count_*`（HT 权重的分母来源） |
| `.codex-fleet/rq015b-repair/work/frame_index.csv` | 4,981,984 行全量锚点索引（664 MB，按需流式读） |
| `.codex-fleet/rq015b-repair/work/run_b1_rq015b.py` | **B1 的可复用模块**：`build_sequences()`、`diagnostic_for_anchor()`、`HISTORY_WINDOW`、`MIN_OBSERVATION` |
| `.codex-fleet/rq015b-repair/board/sampling_contract_v1.md` | **冻结合同，只读，不得修改，不得出 v2** |
| `.codex-fleet/rq015b-repair/board/BLOCKED_B1.md` | T3 复现门 40 行明细 |

**开工第一件事**：核对 `sample_v1.csv` 的 sha256 与上表一致。不一致立即停下写 `BLOCKED_B2.md`。

**必须复用 `run_b1_rq015b.py` 的 `build_sequences` / `diagnostic_for_anchor`**，
不要自己重写一份装载/对齐逻辑：B1 的门 B 已证明该路径复算 legacy 权重
与生产诊断**逐位一致（weight_diff = 0.0，40/40）**。重写会把这个保证弄丢。

### 1.2 T3 复现门失败——这是**已裁定的边界，不是你要解决的问题**

```
gate_a 12/40（阈值 39/40，失败）   gate_b 40/40（通过，最大权重差 0）
按签名： U 12/14 过 ｜ Z 0/14 全败 ｜ N 0/12 全败
28 个失败中：15 个"仅 ipv_error 差"（中位 3.4e-05）；13 个两者都差（最大 ipv_diff = 3π/8 = 整格错位）
B1 已排除"窗口化取值口径"这一解释（完整 pair-call 与窗口单帧调用给出相同重解结果）
```

**监督方裁定，你必须照此执行**：
- **禁止追查失配根因**（求解器差异 / 参考线处理 / 存档 checkout 漂移都不追）。
  代码漂移假说记进报告的 known-issues 一节，留待独立一轮。
- **禁止 `git checkout` 任何历史提交去验证漂移假说**（见铁律 5，这条会毁掉 track A）。
- 理由：RQ015B 要回答的不是"存档当年怎么来的"，而是
  **"改成 log 域能不能把兜底锚点救回来、救回多少"**——后者是
  **当前代码在这批数据上的性质**，与能否重现 2026-06-12 的存档值无关。

---

## 2. 任务

### T5 — 解算与平价门（核心）

对冻结样本的 **2,300 个锚点各解一次**（B1 实测 2.05 s/锚点，串行约 1.31 h；
可用多进程加速，但须记录并行度且保证结果与串行一致）。

落盘 `.codex-fleet/rq015b-repair/work/anchor_mse.csv`（或 parquet），每锚点一行：
```
anchor_id, source, n_band, signature, frame_index, n_obs, K,
mse_per_candidate[7], rms_per_candidate[7], min_mse, argmin_candidate,
legacy_prod_sum (= sum(var)), legacy_fallback_triggered (bool),
w_legacy[7], w_log[7], max_abs_diff,
ipv_legacy, ipv_log, ipv_error_legacy, ipv_error_log,
k_eff_legacy, k_eff_log, at_grid_boundary, any_nonfinite,
partial_underflow (至少一个候选的 φ 连乘下溢但 sum(var) 仍非 0)
```

- `w_legacy` 必须由**生产函数本身**（`agent.cal_traj_reliability`，**只读调用**）产生，
  **不许**你重写一份"等价实现"——平价必须对着真实 legacy 比。
- `w_log` 由 `reliability_logdomain.py` 产生（只读调用，不接线）。
- **两者共用同一组重解结果**（同一组 MSE），**不许为两种权重各解一次**。
  这正是把数值问题与求解器问题隔开的关键。
- `k_eff` 定义：`k_eff = 1 / Σ(w²)`（有效候选数，均匀时 = K = 7）。

**平价门四条判据（逐条给数字）**

1. **未触发兜底、且未发生部分下溢的锚点**上，`max_abs_diff ≤ 1e-12`。
   逐格报 `max / p99 / median`，并给全样本最大值。
2. **已触发兜底的锚点**上：legacy 走均匀兜底，而 log 域给出**非均匀**权重。
   报这些锚点上 log 域的 `k_eff` 分布。
   **若 log 域也近均匀，说明这些行本就平坦——这是重要发现，如实报，不要藏。**
3. **兜底不可达证明**：log 域中 `logw -= logw.max()` 后至少一项为 0
   ⇒ `exp` 至少一项为 1 ⇒ 分母 ≥ 1 ⇒ 除零分支不可达。
   给出**论证 + 数值实验**（构造 RMS 远超 1.7336 m 的极端输入，
   验证 log 域仍产出有限非均匀权重）+ **一个 pytest 测试**
   （写进 `.codex-fleet/rq015b-repair/work/test_b2_fallback_unreachable.py`，
   **不要**放进 `$ROOT/tests/`）。
4. **临界值复核**：用实测数据验证 §1 的四个临界 RMS，报实际观察到的兜底触发起点与闭式值之差。
   **【监督方已批准的判定口径，照此执行】**：逐项 `np.prod` 在 subnormal 区间累积舍入，
   实际归零点比闭式**晚** 0.58–0.82 mm 属**预期口径差，不算失败**。
   **仅当偏差方向为负（比闭式更早归零）或量级达厘米级才算判据 4 失败。**

### T6 — 机制拆分 D0–D4（本轮最有价值的产出）

判据沿用计划 §6，**不新造**。优先级 **D4 > D1 > D3 > D2 > OK**；
D0 由 `frame_index < 4` 上游分流（**普查计数，不抽样**）。

| 机制 | 判据 |
|---|---|
| `D0_NOT_ATTEMPTED` | `frame_index < 4` 的 warm-up 占位（估计器从未运行）。已知普查数：nuplan 42,432 / waymo 130,624 |
| `D4_SOLVER_OR_INPUT_FAILURE` | 非有限输入 / 平方距离溢出 / 求解器异常 |
| `D1_NUMERICAL_UNDERFLOW` | **legacy 与 log 域权重的最大绝对差 > 1e-6**（即 legacy 结果已被下溢改变） |
| `D3_MODEL_MISFIT` | `min_mse > min_mse_misfit`（阈值处置见下） |
| `D2_FLAT_UNDER_CURRENT_GRID_AND_MODEL` | `c = k_eff/K ≥ 0.93` 且 `min_mse` 未超阈 |
| `OK` | 其余 |

**`D1` 的定义是"legacy 结果是否被改变"，不是"是否发生过下溢"**：
远距候选被清零属无害部分下溢，判 `OK`（`partial_underflow` 标志仍如实记录）。

**`min_mse_misfit` 的处置（顺序不可颠倒）**：
先在样本上算 `Q_0.99(min_mse)` → 写入独立文件
`.codex-fleet/rq015b-repair/work/min_mse_misfit_threshold.json` → **登记 sha256** →
**然后才**跑分类。报告中必须标注：**该阈值为本抽样样本估计，属本轮临时口径，
不得据此冻结生产阈值**。同时报 `p = 0.95 / 0.999` 的敏感性。

**"修得好的"是本轮要回答的那个分叉，必须单独量化**：
在零值锚点（U ∪ Z）中，给出 log 域改写后**权重变为实质非均匀**的比例
（判据：`k_eff_log / K < 0.93`），逐层给出。
- 若 **D1 占主导** ⇒ 改写能直接救回大量样本 ⇒ 支持申请全量重跑预算；
- 若 **D2 占主导** ⇒ 问题在网格/模型，改数值没用 ⇒ 不值得重跑。
**这个分叉是本轮存在的理由，结论段必须正面回答，不许含糊。**

### T7 — 加权与不确定度（合同 §6，不得改）

- **主估计量**：零值总体（U ∪ Z）中 D1/D2/D3/D4 各自的**框加权**占比。
  框加权 = 每格权 `N_cell / n_cell`（Horvitz–Thompson），
  `N_cell` 取 `t2_summary.json` 的 `available`，`n_cell` 取 `drawn`。
  **因 U 被刻意加密，未加权比例不得作为结论数字。**
- **不确定度**：case-cluster bootstrap，按 `scene_unique_id` 重抽，
  **B = 2000，seed = 20260731**，percentile 95% CI。
- **精度上限**：主估计量 95% CI 半宽 **≤ 3 pp**。
  超出则**照实报告并标注**，**不得**追加抽样或改配额（追加会破坏冻结）。

### T8 — 数值健康自查（**一轮，不要展开成审计**）

至少查：`mse_per_candidate` 有无 NaN/Inf/病态常数；`min_mse` 分布；
每格实抽数与案例数；同一案例贡献的锚点数分布；
`legacy_fallback_triggered` 与签名 `U` 的一致性
（**理论上 U ⊇ 兜底行**；若出现"触发兜底但签名非 U"或大量"签名 U 但未触发兜底"，
如实报并解释——**后者恰恰说明那些行是真平坦而非下溢，是本轮最有价值的信号之一**）。

---

## 3. 结项报告（**有界**：≤ 250 行 Markdown）

写到 `$ROOT/.codex-fleet/rq015b-repair/board/reports/B2_report.md`，严格按此结构：

```
# RQ015B 结项报告 — 平价门与机制拆分（本地可达 nuplan + waymo 子集，分层抽样 2,300 锚点）
## 1. 两句话结论（(a) 平价门过/不过、兜底是否不可达；(b) D0–D4 构成与"修得好的"占多少）
## 2. 覆盖边界与口径（三项边界 + 两个分母）
## 3. 平价门（4 条判据逐条给数字）
## 4. 兜底行上 log 域的表现（k_eff 分布 —— 修得好还是本就平坦）
## 5. 临界 RMS 复核（闭式 vs 实测，含已批准的容差口径）
## 6. 机制拆分 D0–D4（框加权占比 + 95% CI + 逐层表）
## 7. 复现门未过：一等边界（不是脚注）
## 8. 数值健康自查（没有问题就写"无"）
## 9. known issues（代码漂移假说，留待独立一轮）
## 10. 产物清单（路径 + sha256 + 每列一行说明）+ 完工自检三项 SHA
## 11. 偏差与未做的事（有就写，没有写"无"；不得静默截断）
```

### 3.1 §2 必须写进去的口径与边界

**两个分母（已对齐，直接用）**：
| 口径 | 分子 / 分母 | 占比 |
|---|---|---|
| 精确零，dev+guard **后热身** | 1,200,636 / 4,981,984 | **24.10%** |
| `\|ipv\| < 1e-6`，dev+guard **全体**（含 warm-up） | 2,008,902 / 5,197,072 | **38.65%** |

**38.65% 就是既有认知里"约四成"的口径**，与 24.10% 分母不同，不是冲突。

**覆盖边界三项（禁止静默截断）**：
1. `lyft_train_full`（785,080 锚点）与 `av2_motion_forecasting`（234,546 锚点）
   **无本地原始数据，本轮完全不可达**；
2. waymo 缺 `500-799` 分片；
3. **`waymo_300-499.pkl` 文件存在但读取报 `pickle data was truncated`**，
   已按 `pkl_available=False` 排除（监督方裁定：跳过 + 登记，不修复、不重取）。

⇒ 结论外推范围是"**本地可达的 nuplan + waymo 子集**"，不是 InterHub 全体。
**报告标题与结论句必须带上这一限定。**

### 3.2 §6 必须附的"各层能支持什么"表（监督方指定口径，照抄）

| 层 | 复现门 | 能支持的说法 | **不能**支持的说法 |
|---|---|---|---|
| U（676,405） | 12/14 | 兜底触发本身可复现；当前代码下该层的 D1/D2 拆分 | "存档这 676,405 行里 X% 是 D1"——不动点不承载权重信息 |
| Z（**524,231** = 1,200,636 − 676,405） | **0/14** | 仅"当前代码下这批锚点的机制构成" | 任何向存档 Z 层的外推 |
| N | 0/12 | 同上 | 同上 |

**监督方点名要求**：最值钱的 **Z 层（524,231 行，精确零但无均匀签名，正是 D1/D2 的分界所在）
恰恰是复现率为 0 的那一层。不许把它藏在分层表里，必须在结论段明写。**

### 3.3 §1 与 §7 必须原文包含的边界句（一等边界，不是脚注）

> 本轮结论刻画的是**当前代码在抽样锚点上的行为**；存档主表（2026-06-12）的
> 逐锚点值在本轮未能复现（复现门 gate_a 12/40，阈值 39/40），
> 故上述比例**不得读作存档 M3 标签的成分构成**。

### 3.4 额外产出：known-issue 片段（写进 board，**不要**去改 track A 的文件）

把"存档不可复现 / 代码漂移假说"写成一段可直接并入的 Markdown，落盘
`.codex-fleet/rq015b-repair/board/reports/known_issue_snippet_repro_gate.md`。
**不要**去写 `reports/knowledge/RQ015A_.../known_issues_and_audit_boundary_20260730.md`
——那是 tracked 文件且属 track A，此刻正被并发修改（铁律 4/5）。合并由上游安排。

### 3.5 报告纪律

不写因果措辞；不用 estimability / "测出 IPV"；不把描述性结果上升为对方法的断言；
**每个数字都要给出它的分母**；不得静默截断。

---

## 4. 心跳（必做，leader 靠它判断你是否存活）

每完成一个 T 步，**立即**向
`$ROOT/.codex-fleet/rq015b-repair/board/B2_heartbeat.log` **追加**一行：
`<UTC ISO8601> | T<k> | <一句话结果或关键数字>`。
**开工第一行必须在任何长任务之前就写**（B1 的教训：扫描期静默 10 分钟无记录）。
T5 是长任务，额外要求：每处理约 200 个锚点追加
`<UTC> | T5-progress | <已完成>/2300`。
心跳只给 leader 看，不要写成报告，不要因为写心跳而中断计算。

## 5. 输出位置汇总

- `.codex-fleet/rq015b-repair/work/` — `anchor_mse.csv`、
  `min_mse_misfit_threshold.json`、`test_b2_fallback_unreachable.py`、你写的脚本
- `.codex-fleet/rq015b-repair/board/reports/B2_report.md` — 结项报告
- `.codex-fleet/rq015b-repair/board/reports/known_issue_snippet_repro_gate.md` — known-issue 片段
- `.codex-fleet/rq015b-repair/board/BLOCKED_B2.md` — 仅在触发停止条件时写

## 6. 停止条件（触发即停，写 `BLOCKED_B2.md`，**不要自行变通**）

1. `sample_v1.csv` 的 sha256 与 §1.1 不符
2. 需要读取评分 / 偏好字段才能继续
3. 需要修改 `src/` / `pipelines/` / `tests/` / `scripts/` 下任何 tracked 文件才能继续
4. 需要接线生产估计器才能继续
5. T5 实测吞吐外推 > 6 小时（**不要自行缩减样本量**，缩减会破坏冻结合同）
6. 发现任何 held_out 行被读入

**注意：复现门失败已由监督方裁定为边界，不是停止条件——不要因为它而停。**

开工。
