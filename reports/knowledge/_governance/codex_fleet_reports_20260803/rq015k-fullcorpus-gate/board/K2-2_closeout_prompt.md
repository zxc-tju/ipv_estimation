# K2-2：K2 结项收尾（四件事，做完就结束，不要扩展范围）

你是 K2-2。K2 的全量计算**已经完成且产物没有问题**——监督方已复核。
你的任务**只有四件**，做完写报告更新、转 `WAITING_ON_COMMANDER`，**不要重跑求解、不要重跑 join、不要改任何判据阈值**。

## 背景（监督方 19:12Z 裁定，已记入 `board/commander_notes.md`，先读那一节）

K2-1 以 `final_status=FAIL`、`blockers=g_anchor, solver_failure_threshold` 结项。
**监督方逐条复核后判定两条 blocker 都不成立**：

1. **`g_anchor` 被推翻。** `k2_fullcorpus_materializer.py` 第 1468 行读的是
   `.codex-fleet/rq015b-repair/work/anchor_mse.csv`（**RQ015B 的 Mac 产物**），
   而任务书第 6.2 节指定的是
   `.codex-fleet/rq015g-hpc-resolve/work/anchor_mse_hpc.csv`（**G 轨的 HPC 产物**）。
   监督方已实测锚点 `ipv_007137|46|1`：
   K2 的 L1 值与 **G-HPC 基线逐位相同（最大绝对差 = 0）**；
   与 Mac 基线差 `0.013332352186283258`，**恰等于 validator 报出的 `max_abs_diff`**。
   两份基线本身在 2,300 个锚点里有 1,867 个（81.2%）不同——这正是 G 轨既有结论
   （Mac↔HPC 差异来自软件栈；HPC↔HPC 逐位相同）。**产物无问题，是比错了对象。**
2. **`solver_failure_threshold` 被监督方撤销。** 那条「单片超 100 行即停」的阈值是监督方
   按 pilot 的 0/1,120 拍的，**从未在 nuPlan Vegas 数据上校准**。全局 1,934/4,981,984 = 0.0388%，
   `non_finite` 全为 0，且被正确记为工程失败、未污染两个科学 reason。
   **改为「必须刻画并报告」，不重跑、不判 FAIL。**

## 你要做的四件事

### 1. 修基线路径并重跑 G 锚点比对（**只跑这一个校验**）

把第 1468 行改为 `.codex-fleet/rq015g-hpc-resolve/work/anchor_mse_hpc.csv`。
**比较口径一个字不许改**（float64、按 canonical key 对齐、`diff != 0.0` 即失败）。
只重跑 `validate_g_anchor`，可直接对本地已回收的
`data/derived/rq015k_logdomain_gate/l1_v1` 跑，**不必再投 Slurm**。

必须报出：`anchor_rows`、**`compared_rows`**、`max_abs_diff`、`first_mismatch`。
预期 `max_abs_diff = 0`。**若 `compared_rows` 远小于 2,300，说明 `anchor_id` 与
`canonical_key` 对齐有问题，要说明命中率并解释原因**，不得含糊带过。
结果写 `work/k2_fullcorpus/validation/g_anchor_hpc_baseline.json`。

### 2. 补测 RQ009 join 行的 canonical_key 唯一性（**这是唯一从未被测过的判据**）

`finalize()` 里 `join_counts["duplicates"] = 0` 是**硬编码**，`validate_outputs()` 的 blocker
也**没有检查它**；`seen_interhub` 去重集合只覆盖 InterHub 行、不含 join 行。
所以「RQ009 exact-one join, duplicates 0」至今**没有任何实测依据**。

对本地 `l1_v1/artifact_id=rq009_feature_matrix` 的 8,994,736 行**实测** `canonical_key` 去重：
报 `rows`、`unique_keys`、`duplicates`、若有重复给前 10 个例子。
结果写 `work/k2_fullcorpus/validation/rq009_join_key_uniqueness.json`。
**`new_solve_rows = 0` 可保留为「by construction」，但报告里必须这样标注，不得写成实测。**

### 3. 刻画 1,934 行 SOLVER_FAILURE（不重跑）

给出：按 PKL / source 的分布；这些行的共同特征（`n_obs`、几何、是否与源共线等你能从产物读到的）；
与既有结论「400 个退化锚点全部来自 nuplan、与源共线」是否同源的判断。
结果写 `work/k2_fullcorpus/validation/solver_failure_characterization.json` 与报告正文一节。

### 4. 更新报告与看板

- `board/reports/K2_fullcorpus_gate_ledger.md`：把 `final_status` 由 FAIL 改为
  **PASS（附监督方裁定依据）**，两条原 blocker 分别改写为「基线选错，已在正确基线下复核」
  与「阈值由监督方撤销，改为刻画」。**不得删除失败史。**
- 报告必须包含**完整失败史**，solve 三投与 finalize 两投**分开列**，各自根因：
  ① font-cache 并发锁 ② PyArrow 定长 list 写不了 null 行 ③ 逐行重算源 parquet SHA-256
  ——并明写**没有一个触碰数值路径**。
- 报告必须写两条方法学结论：
  **(a)** 1 行 canary 测不到「多 worker 并发」与「工程失败行写盘」两条真实路径；
  **(b)** 每一条验收判据都必须有一次**故意让它失败**的验证，证明它真的会 FAIL——
  本轮出现两例「看起来在检查、实际没检查该检查的东西」（RQ009 `duplicates` 硬编码、
  G 锚点比错基线）。
- `rq009_array_restore_1000` 在 summary 里是 FAIL、18:44 的 corrected 版是 PASS：
  **说明以哪一版为准、原版为何 FAIL。**
- `INTERFACE_NOTE.md` 必须含那条警告：**`ipv_log = 0` 是合法且高频的通过门估计值
  （门后 23.40%），判别字段只能是 `status` 与 `reason_code`，不是 `ipv_log` 的数值。**
- **刷新 `board/STATUS.md`**（自 12:47:23Z 起未更新，监督方已催六次）。
- manifest 补「真正决定归属的行索引区间」字段说明（分片是按输入 CSV 行顺序切的连续块，
  `row_key_min/max` 是观测到的键范围、不构成不相交证明）。

## 科学结论（已由监督方核定，直接引用，不要重算）

InterHub 全量 4,981,984 个求解单元：
`OK` **3,502,340 = 70.3001%**；`NEAR_UNIFORM` 1,457,746 = 29.2604%；
`NO_IPV_EFFECT` 19,964 = 0.4007%；`SOLVER_FAILURE` 1,934 = 0.0388%。
J 轨设计基点估计 71.2695%、CI [67.1729%, 75.2135%]（2,300 锚点、HT 权重、1,909 cluster）
→ **普查值落在 CI 内，与点估计相差 0.97 个百分点**。
**这是解释性对照，不是判据**（域与分母不同：普查分母是 4,981,984 个求解单元，
J 的分母是 2,646,058 的 HT 权重）。**不得写成"验证通过"。**

## 硬边界

- **不改** `agent.py` / `ipv_estimation.py` / `process_interhub.py` / `reliability_logdomain.py` /
  `configs/ipv_sigma01_exact.json`；`k2_fullcorpus_materializer.py` 只许改第 1468 行那个路径
- 不提交 git commit；禁止 `git checkout -- .` / `restore` / `stash` / `reset --hard` / `clean -fd`
- 不投 Slurm（本轮全部可在本地产物上完成）
- 不解析 RQ007 held_out，不读 RQ014 致盲字段
- 不做阈值扫描、不提替代判据、不写规格 v2
- 时间戳一律 `date -u +%Y-%m-%dT%H:%M:%SZ`，不要前瞻估计
- 结项自证：四个受保护文件 SHA、`git --no-optional-locks status --porcelain`

做完写 `state: WAITING_ON_COMMANDER`，**不要自行 DONE**。
