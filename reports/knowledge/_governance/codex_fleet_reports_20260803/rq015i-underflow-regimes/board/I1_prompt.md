你是 RQ015I 的执行 agent（代号 I1）。任务：量化 IPV 权重计算在**连乘域（legacy product domain）**
下的三个失效区间的规模。一轮做完，出一份报告，不要问我问题，不要等确认。

仓库根（下称 $REPO）：
.

**Python 解释器必须钉死用这个**（系统 python3 缺包）：
<local-rq009-venv>/bin/python

你的产出目录：$REPO/.codex-fleet/rq015i-underflow-regimes/work/
最终报告写到：  $REPO/.codex-fleet/rq015i-underflow-regimes/board/reports/I1_underflow_regimes.md

---

# 0. 三个失效区间的定义

对样本文件里的每一行，`legacy_density_product[7]` 是一个 7 元素的浮点数组（以字符串形式存储的
Python list，用 `ast.literal_eval` 解析）。定义（`nzero` = 该数组中**恰等于 0.0** 的分量个数）：

| 区间 | 判据 | 现状 | 后果 |
|---|---|---|---|
| ① 全下溢 | `nzero == 7` ⇔ `legacy_prod_sum == 0` | 已标记 `legacy_fallback_triggered=True` | 回退均匀权重 ⇒ `ipv_legacy` 恰为 0、`ipv_error_legacy` 恰为 0.6220355269907728 |
| ② 部分下溢 | `1 <= nzero <= 6` 且 `legacy_prod_sum > 0` | **未标记** | 这些候选权重恰为 0 —— 不是降权，是被静默删除出加权平均 |
| ③ 次正规 | `nzero == 0` 但 `min(legacy_density_product) < 2.2250738585072014e-308` | **未标记** | 落入 denormal，尾数位被吃掉，精度静默退化 |

0.6220355269907728 = 1 - 1/sqrt(7)。下称 `TGT`。

---

# 1. 输入文件（只读这些，不要在 $REPO 里做宽泛检索）

样本层（2300 行锚点，两个环境各一份，36 列同名同序）：
```
Mac: $REPO/.codex-fleet/rq015b-repair/work/anchor_mse.csv
HPC: $REPO/.codex-fleet/rq015g-hpc-resolve/work/anchor_mse_hpc.csv
```
关键列：`anchor_id, scene_unique_id, source, n_band, signature, split, frame_index, n_obs, K,
legacy_density_product[7], legacy_prod_sum, legacy_fallback_triggered, w_legacy[7], w_log[7],
max_abs_diff, ipv_legacy, ipv_log, ipv_error_legacy, ipv_error_log, k_eff_legacy, k_eff_log,
at_grid_boundary, any_nonfinite, partial_underflow, solve_error`

设计权重与机制层（HT 权重）：
```
Mac: $REPO/.codex-fleet/rq015b-repair/work/mechanism_split.csv
HPC: $REPO/.codex-fleet/rq015g-hpc-resolve/work/mechanism_split_hpc.csv
```
列：`anchor_id, scene_unique_id, source, n_band, signature, frame_index, n_obs, ht_weight,
zero_postwarm_scope, mechanism_p95, mechanism_p99, mechanism_p999, repair_good_k_eff_log_lt_0p93K`

全语料台账（**4 份 parquet，schema 同构，合计 14,473,982 行**）：
```
$REPO/reports/studies/RQ015A_ipv_estimability_labelling/RQ015A_1_concentration_audit_20260731T093746Z_e82091ce/concentration_ledger/
  interhub_sigma01_hw4_timeseries.parquet   5,197,072 行  (137 MB)
  rq009_feature_matrix.parquet              8,994,736 行  (236 MB)
  onsite_dense_timeseries.parquet             281,268 行
  wod_rq010b_full479_audited.parquet              906 行
```
列：`artifact_id, product_row_key, measurement_role, case_id, rq007_split, ipv_error, K,
candidate_grid_id, k_eff, q_eff, attempt_status, reason_code, recoverability,
ledger_schema_version, aggregation_perspective, aggregation_configuration`

⚠ 全语料台账**没有保存 `legacy_density_product[7]`**，也**没有 `n_obs`**。因此区间②/③
无法在全语料上直接分类。**不要重算任何锚点，不要提交 HPC 作业，不要提议全量重跑。**

---

# 2. leader 已勘察的 4 项事实（直接用，不要再自己推翻重来；但若你的计算与之矛盾，如实写出来）

**(a) `q_eff = k_eff / K`，不是 1/k_eff。** 实测 `max|q_eff - k_eff/K| = 8.88e-16`。
所以**均匀兜底（区间①）在台账上的签名是 `k_eff ≈ 7`（= K）且 `q_eff ≈ 1.0`**。
任务书初稿写的「q_eff 恰等于 1/7」是错的，按那个筛会返回 0 行。**不要用 1/7。**

**(b) 严格相等在 `k_eff` 上不成立。** 在 interhub 第 0 个 row group（10 万行）上实测：
`|ipv_error - TGT| <= 1e-15` 命中 3621 行；其中 `q_eff == 1.0` 严格命中 3209 行，
而 `k_eff == 7.0` 严格命中 **0** 行（k_eff 有浮点尾差）。
所以严格 `==` 口径必须建立在 `q_eff` 上，`k_eff` 只能用容差。
且 3621 − 3209 = 412 行只满足 ipv_error 而不满足 q_eff——**这部分必须解释清楚**
（它们是不是兜底？ipv_error 是否被量化？看 `attempt_status`/`reason_code`/`k_eff` 分布）。

**(c) `anchor_mse.csv` 里已有的 `partial_underflow` 列 ≠ 区间②。**
Mac 上该列 True 有 171 行，而按区间② 定义重算是 164 行；7 行分歧**全部是 `nzero == 0`**，
其中 2 行正是区间③ 的那两行（`ipv_034135|21|2`、`ipv_034642|27|2`）。
**你必须刻画这个已有列到底是什么语义**（读它的生成逻辑或从数值反推），
并在报告里明确「不得把它当作区间② 的标志位」。区间②/③ 一律以第 0 节定义重算为准。

**(d) RQ007 硬约束结构性安全，但仍要断言。** 4 份 ledger 的 `rq007_split` 取值只有
`development` / `guard` / `RQ007_SPLIT_NOT_APPLICABLE`，**没有 `held_out`**。
你的脚本里**必须**加一条断言：若任何 ledger 行的 `rq007_split == 'held_out'`，立即中止并报错退出。
样本文件的 `split` 列取值为 `development`/`guard`，同样断言。

---

# 3. 第 1 部分 —— 区间① 在全语料上直接计数（实测，不是估计）

区间① 有精确签名，可在台账上识别。**分块读，按 row group 逐块处理，不要整表 load。**
只读需要的列（`pyarrow.parquet.ParquetFile.read_row_group(i, columns=[...])`）。

对 **4 份 parquet 全部**执行，报告**每份 artifact 单列 + 合计**：

- 严格口径：`(ipv_error == TGT) & (q_eff == 1.0)`（浮点严格相等）
- 容差口径：`(abs(ipv_error - TGT) <= 1e-15) & (abs(q_eff - 1.0) <= 1e-12) & (abs(k_eff - K) <= 1e-9)`
- **两个口径都要报，差值单独列出并解释**
- 分层：按 `artifact_id` × `rq007_split` × `measurement_role`
- 另外单独报「只满足 ipv_error 条件但不满足 q_eff 条件」的行数及其 `attempt_status` /
  `reason_code` / `k_eff` 分布（即上面 (b) 的 412 行现象在全语料上的规模）

**交叉验证**：这个计数应与 `attempt_status` / `reason_code` 里既有的失败标记大体对应。
给出区间① 命中行 × `reason_code` 的交叉表，以及 `reason_code` 各取值中有多少比例命中区间①。
对不上的部分要解释，不要一笔带过。

注意 `attempt_status == 'NOT_ATTEMPTED'` 的行（interhub 215,088 行、onsite 4,272 行）
和 `UNKNOWN`（onsite 274,022 行）——这些行的 `ipv_error` / `q_eff` 可能是 NaN，
要说明它们在分母里怎么处理的。

---

# 4. 第 2 部分 —— 区间②/③ 的设计基外推（本轮主产出）

2300 个锚点是**分层概率样本**（配额 U300/Z150/N125 × 4 个 source×n_band 单元）。
RQ015B 已建好 HT 权重与 bootstrap，**直接复用，不要另造**。

**leader 已核实的口径（照做）**：
- 分母口径与 D0–D4 一致 = `zero_postwarm_scope == True`，样本内 1800 行
- 该口径内 distinct `scene_unique_id` = **1459**，正是 bootstrap 的 clusters 数
- 该口径内 `ht_weight` 合计 = 534,939，**Mac 与 HPC 完全相同** ⇒ 两版估计在同一分母上
  直接可比（全表 ht_weight 合计两版不同，那是域外 500 行造成的，不要用全表）
- bootstrap：**B=2000、seed=20260731、cluster = scene_unique_id、聚类重抽（cluster bootstrap）**。
  **B / seed / clusters 一律不得改**。CI 用 percentile 法（2.5%/97.5%）；
  若某单元 bootstrap 分布退化（例如样本内命中数为 0 或 1），**如实写「CI 不可给出」**。

用 `anchor_id` 把 `anchor_mse*.csv` 与 `mechanism_split*.csv` 连接（两边 anchor_id 都是 2300 且唯一）。

要交付**带 CI 的设计基估计**（HT 比率估计：`sum(w*1{regime}) / sum(w)`）：

1. **区间② 占比**，按 `source` 分列（**waymo / nuplan 必须分开**），合并值只能作为附加行
2. **区间③ 占比**，同样分 source（注意样本里只有 2 行，CI 会极宽或不可给出——**宽就写宽**）
3. **「`nzero == 6`」那一格单独给估计与 CI** —— 七个候选被下溢删掉六个、只剩一个存活，
   这是**由下溢制造的、而非由证据支持的事实上的 hard argmax**，且不带任何标记。
   这一格是本轮最重要的数字。
4. 可选补充：按 `signature`（U/Z/N）分层的估计，若样本量允许
5. **Mac 与 HPC 两版各做一遍并对照**。已知样本上区间② 的严重程度 Mac 远高于 HPC
   （`max_abs_diff` max 0.324 vs 4.08e-3，p99 5.78e-2 vs 1.04e-4）。
   **要说明这个差距在加权外推后还剩多少**——即分别给两版的加权占比与 CI，并说明 CI 是否重叠。

**诚实性要求（硬性）**：这是外推不是普查。报告标题与每一处引用都必须写明
**「设计基估计（design-based estimate）」并附 CI**，不得写成实测计数。
不要为了好看而挤出一个数：样本量不足就写不足。

---

# 5. 第 3 部分 —— 可识别性：能不能不重算就直接识别区间②/③？

在 2300 样本上我们同时有「真值（区间归属，按第 0 节定义重算）」与「全语料可得的统计量」。

**全语料可得的列只有**：`ipv_error`、`K`、`k_eff`、`q_eff`、`candidate_grid_id`、
`measurement_role`、`rq007_split`、`attempt_status`、`reason_code`、`recoverability`。
样本里另有 `n_obs`、`source`、`n_band`、`signature`、`at_grid_boundary` 等，
但 **`n_obs` 在全语料台账里不存在**。

注意：样本 CSV 的对应列名是 `ipv_error_legacy` / `k_eff_legacy`（连乘域），
台账的 `ipv_error` / `k_eff` 对应哪一域要先确认（对照数值分布判断），**在报告里写明你的判断依据**。

要做的：

- 拟合判别函数，目标 = 「是否属于区间②（或②∪③）」，特征**只用全语料可得列**
- 报告灵敏度 / 特异度 / ROC-AUC，**Mac 与 HPC 两版分别验证**
- 同时试一个**简单可解释规则**（例如 `k_eff` 阈值单独、或 `k_eff` + `q_eff` 组合）；
  若某简单规则能达到高特异度，写明阈值，全语料可直接套用，**全量重跑就没必要了**
- 若识别不了，**如实写「识别不了」**，并说明这意味着：要拿到区间②/③ 的实测计数只能重算全量。
  **不要为了给出「有用的答案」而美化一个弱分类器。** 这个结论会直接进 PI 的决策。
- 若你的判别函数用到了 `n_obs`，**必须明确指出它不能直接套用到全语料**，
  并说明能否用台账已有列替代（试一下，给出替代后的性能损失）

---

# 6. 报告要求

写到 `$REPO/.codex-fleet/rq015i-underflow-regimes/board/reports/I1_underflow_regimes.md`，必须含：

1. **样本基准复现**：逐项对照下表，不一致处必须解释

   | 量 | Mac 预期 | HPC 预期 |
   |---|---|---|
   | 非兜底行数 | 1697 | 1687 |
   | 区间② 行数 | 164（占非兜底 9.7%） | 150（8.9%） |
   | nzero 直方图（非兜底） | {0:1533,1:94,2:29,3:8,4:9,5:7,6:17} | {0:1537,1:89,2:30,3:7,4:9,5:6,6:9} |
   | 区间② 上 max_abs_diff | p99 5.78e-2，max 0.324 | p99 1.04e-4，max 4.08e-3 |
   | 区间③ 行数 | 2 | 2 |
   | 区间③ 具体行 | ipv_034135\|21\|2、ipv_034642\|27\|2（均 waymo、signature Z、n_obs=11） | 同 |

   （leader 已独立复现 Mac 的 nzero 直方图，与上表逐格一致。）

2. **区间① 全语料实测**：分层计数、严格 vs 容差两口径、与 `reason_code` 交叉验证
3. **区间②/③ 设计基估计**：占比 + CI，分 source，`nzero==6` 单独一行，Mac/HPC 双版对照
4. **可识别性结论**：判别函数存在与否、性能指标、能否套用到全语料
5. **每个数字须可复算**：写明用了哪个文件的哪一列、什么筛选条件、什么权重、什么口径

把所有计算脚本留在 `work/` 下（命名 `i1_*.py`），中间产物存 CSV，报告里引用文件名。
报告用中文写。

---

# 7. 硬约束（不得放松）

1. `rq007_split == 'held_out'` 不得被解析（脚本里加断言，出现即中止）
2. RQ014 致盲相关的评分字段不得读取
3. 不得覆盖冻结产物或已接受的 `decision.md`；**只写你自己 work/ 与 board/reports/ 下的文件**
4. 描述性结果不得写成因果主张
5. **全文禁用 `estimability` 一词，禁用「测出/未测出 IPV」的表述。**
   可辩护的表述是：**权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息**
6. 不改 `agent.py` / `ipv_estimation.py` / `process_interhub.py` / `reliability_logdomain.py`
7. **禁止** `git checkout -- .` / `git restore .` / `git stash` / `git reset --hard` / `git clean -fd` /
   `git commit` / `git checkout` 任何历史提交。另一条 track 的 agent 正在同一仓库工作，
   工作区非空是**预期状态**。你只对自己创建的文件负责。
8. 不要对 `reports/` 做全仓库 `rg`（会把 RQ003 controlled-access 行拉进上下文）；
   只读第 1 节列出的路径
9. 不重算锚点、不提交 HPC 作业、不提议全量重跑、不重新抽样、不改 bootstrap 的 B/seed/clusters

时间戳一律用 `date -u +%Y-%m-%dT%H:%M:%SZ` 取真实墙钟，不要前瞻估计。
