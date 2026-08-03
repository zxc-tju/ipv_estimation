# K1 — 全语料 log 域门台账的勘察与资源规划（不投全量作业）

你是 track K 的唯一执行 agent。仓库根：
`.`

**先读完本文件全部内容再动手。本文件已经把你需要读的文件路径全部列出，不要花时间摸索仓库结构。**
上一轮有 agent 把全部执行预算花在探索仓库上，一行计算都没跑就被杀掉。

---

## 0. 这项工作在整个研究里的位置（你必须先理解，否则会做错方向）

**最终用途**：online verification —— 判断自动驾驶车辆当前的 IPV 是否符合人类的分布。
IPV（Interaction Preference Value）是一个标量参数，表示驾驶主体的社会互动倾向。

**已裁定的两级弃权机制，串联执行：**

```
机制一（RQ015，本轮要物化的）：这一帧的 IPV 能不能估出有判别力的数值
    ├─ ABSTAIN → 直接结束，不进机制二
    └─ OK      → 进入机制二
机制二（RQ009，已 accepted）：当前场景的人类数据够不够判定 AV 是否偏离
```

**机制一的规格已经定稿**（由 track J 完成并经监督方复算），见
`.codex-fleet/rq015j-gate-spec/board/reports/J1_gate_spec_and_impact.md` 的 §1.1、§1.2、§1.3。
**本轮不改这个规格，一个字都不改。**

**为什么现在必须重算**：现有的全语料台账（4 份 parquet，合计 14,473,982 行）里
**没有** `mse_per_candidate[7]`（7 个候选 IPV 各自的均方误差）与
`w_log[7]`（log 域归一化权重）。台账里已有的 `k_eff` 是从旧的连乘域 `ipv_error` 派生的
（`reports/plans/RQ015A_ledger_schema_v4_20260731.json` 第 143-148 行：
`k_eff = 1.0 / (1.0 - ipv_error) ** 2`），**不能替代 log 域权重**。
所以机制一目前无法在现有产物上逐行判定。

**本轮（track K）分两段**：

- **K1 = 你要做的这一段**：勘察 + 资源规划。**不投全量作业**，只投一个小批测成本。
- **K2 = 尚未授权**：按 K1 的方案重算并物化全语料台账。需监督方另行放行。

**你只做 K1。K1 结束就停。**

---

## 1. 铁律（不可协商，违反即本轮作废）

```
1. 不得解析 RQ007 的 held_out 划分数据（污染不可恢复）。
   凡涉及 rq007_split 列，只允许统计计数，不得把 held_out 行的内容读进分析。
   任何输出必须能报告 held_out_parsed_rows = 0。
2. 不得读取 RQ014 致盲相关的评分字段。
3. 不得静默覆盖任何冻结产物或已接受的 decision.md。
4. 描述性结果不得写成因果主张。
5. 不得修改这四个文件：
   src/sociality_estimation/core/agent.py
   src/sociality_estimation/core/ipv_estimation.py
   src/sociality_estimation/core/reliability_logdomain.py
   pipelines/interhub/process_interhub.py
6. 禁止：git checkout -- . / git restore . / git stash / git reset --hard / git clean -fd
   禁止：git checkout 任何历史提交到主工作区
   禁止：git commit（本轮产物由 PI 统一提交）
   工作区非空是【预期状态】，此前轨道留下的文件仍在。你只对自己创建/修改的文件负责。
7. 不得对 reports/ 做全仓库 rg —— 宽泛检索会把 controlled-access 的标注文件整行拉进上下文。
   检索必须限定到本文件列出的具体目录。
8. 不投全量作业。HPC 上只允许投一个小批 pilot（规模见 T4）。
9. 全文禁用 "estimability" 一词和"测出/未测出 IPV"的说法。
   可辩护的表述是：权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息。
10. 时间戳一律用 `date -u +%Y-%m-%dT%H:%M:%SZ` 实测，不得前瞻估计或手写。
11. 若任何环节出现密码提示，立即停止并上报，不得输入、存储或打印密码。
```

**Python 解释器钉死**：本地一切 Python 检查用
`<local-rq009-venv>/bin/python`
（系统 python3 缺 pytest 等依赖，会把基线判错）。

---

## 2. 已经查明的事实 —— 不要重复劳动

以下事实已由两位互不可见的独立审查方（KR-A 从计划书出发、KR-B 从数据反推）各自独立确认，
监督方已复核关键数字。**你直接引用，不要重新验证一遍。** 你的时间要花在第 3 节的新工作上。

### 2.1 台账规模与状态拆分（三者相加恰等于总数）

| artifact（parquet 文件名） | 总行数 | ATTEMPTED | NOT_ATTEMPTED | UNKNOWN |
|---|---:|---:|---:|---:|
| `interhub_sigma01_hw4_timeseries.parquet` | 5,197,072 | 4,981,984 | 215,088 | 0 |
| `onsite_dense_timeseries.parquet` | 281,268 | 2,974 | 4,272 | 274,022 |
| `rq009_feature_matrix.parquet` | 8,994,736 | 8,994,736 | 0 | 0 |
| `wod_rq010b_full479_audited.parquet` | 906 | 906 | 0 | 0 |
| **合计** | **14,473,982** | **13,980,600** | **219,360** | **274,022** |

目录：
`reports/studies/RQ015A_ipv_estimability_labelling/RQ015A_1_concentration_audit_20260731T093746Z_e82091ce/concentration_ledger/`

来源：同一 run 目录下 `bounded_report.md` 第 18-24 行、`concentration_ledger_summary.csv`
（列名 `artifact` / `attempt_status` / `rows`）。

OnSite 的 274,022 行 UNKNOWN 是 `ipv_error` 为 NULL 的空单元，
`reason_code = EMPTY_CELL_UNEXPLAINED`（`bounded_report.md` 第 8、28 行）。

4 份 parquet 的列集合完全相同，共 16 列：
`artifact_id, product_row_key, measurement_role, case_id, rq007_split, ipv_error, K,
candidate_grid_id, k_eff, q_eff, attempt_status, reason_code, recoverability,
ledger_schema_version, aggregation_perspective, aggregation_configuration`。
**没有** `mse_per_candidate`、`log_score`、`w_log`、`ipv_log`、`max_w_log`、`mse_spread`。

每份 parquet 内 `product_row_key × measurement_role` 的 distinct 数等于该文件行数（KR-B 已验，重复为 0）。

### 2.2 已查过、确认没有逐候选量的路径（不要再查这些）

- 上述 4 份 concentration ledger parquet
- `reports/plans/RQ015A_ledger_schema_v4_20260731.json`（`k_eff` 由旧 `ipv_error` 派生）
- RQ015A run 目录内全部文本产物（`bounded_report.md` / `run_receipt.json` /
  `concentration_ledger_summary.csv` / `usable_subset.csv` / `portraits.json`）
- `data/derived/.../03_features/matrix/fold=*/source_dataset=*/features_part_*.parquet`（138 个 part，59 列）
- `data/derived/.../04_calibration/predictions/tier=M3/fold={calibration,test}/predictions.parquet`
- `data/derived/.../03_features/target_hw4/sigma01_hw4_ipv_timeseries.csv`
- `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors/onsite_ipv_timeseries.csv`
- `data/derived/wod_e2e/rq015a_full479_projected/rq010b_wod_full479_audited_candidate_ipv_projected.csv`
- `data/derived/interhub/RQ009.../03_features/parity/**/ipv_results.xlsx`（48 个，只有 legacy IPV/error）
- `data/` 全目录按 `*mse*`、`*log*score*`、`*w_log*`、`*candidate*score*`、`*anchor_mse*`、`*ipv_results*` 的文件名搜索
- 代码：`agent.py`（第 1078-1118 行算 legacy 似然但不持久化）、
  `ipv_estimation.py`（第 340-367 行 diagnostics 只存 `virtual_tracks` / legacy `weights` / `ipv_range` / `ipv` / `ipv_error`）、
  `process_interhub.py`（第 1168-1175 行调用 `estimate_ipv_pair` 未启用 diagnostics；第 854-885 行写出的 xlsx 只有 legacy IPV/error 和运动列）

### 2.3 已查到确实含逐候选量、但只是 2,300 锚点样本、不能替代全量的路径

- `.codex-fleet/rq015b-repair/work/anchor_mse.csv`（Mac 侧，2,300 行）
- `.codex-fleet/rq015g-hpc-resolve/work/anchor_mse_hpc.csv`（HPC 侧，2,300 行）
- `.codex-fleet/rq015f-estimability-contract/work/q4b_anchor_joined.csv`（2,300 行 join 表）
- `.codex-fleet/rq015h-abstain-gate/work/w_log_consistency_sample.csv`（10 行）
- `.codex-fleet/rq015c-drift-forensics/work/gate_legacy_vs_current.csv`（40 行）

`anchor_mse_hpc.csv` 的表头（36 列，你实现时按它对齐字段命名）：
```
sample_order,anchor_id,scene_unique_id,dataset,source,folder,n_band,signature,split,
frame_index,agent_slot,n_obs,K,mse_per_candidate[7],rms_per_candidate[7],legacy_var[7],
legacy_density_product[7],min_mse,min_rms,argmin_candidate,legacy_prod_sum,
legacy_fallback_triggered,w_legacy[7],w_log[7],max_abs_diff,manual_legacy_weight_diff,
ipv_legacy,ipv_log,ipv_error_legacy,ipv_error_log,k_eff_legacy,k_eff_log,
at_grid_boundary,any_nonfinite,partial_underflow,solve_error
```
**这两份 2,300 行产物是你的 pilot 验收锚点**：pilot 若覆盖到其中的 anchor，
新算出的 `mse_per_candidate[7]` 必须与之逐位一致。

### 2.4 G 轨（rq015g）已跑通的 HPC 通道

- 冻结环境：`/share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/bin/python`
  （Python 3.9.24，numpy 1.21.6 / scipy 1.7.3，OpenBLAS）
- 冻结 PKL 快照：
  `/share/home/u25310231/ZXC/sociality_estimation/data/interhub/snapshots/interhub_legacy_20260711_v1/full_datasets/pkl`
  **禁用** `subsets_for_yiru/pkl`（那是更小的 legacy 子集，会得到错误结果）
- 线程钉死：`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1`
  （不设会破坏确定性）
- managed checkout：`6bdcc2e6`
- 已验证 fata02（AMD EPYC 9654）与 cpui158（Intel）对本计算**逐位相同**
  （Slurm 2024766，348/348 float64 bitwise equal），故分区可按队列可得性选择
- 现成脚本（**复用，不要另造**）：
  - `.codex-fleet/rq015g-hpc-resolve/work/stage_and_submit_g1_hpc.sh`（暂存 + 提交）
  - `.codex-fleet/rq015g-hpc-resolve/work/submit_rq015g_anchor_resolve.sbatch`（sbatch 模板）
  - `.codex-fleet/rq015g-hpc-resolve/work/run_g1_hpc.py`（driver）
  - `.codex-fleet/rq015g-hpc-resolve/work/fetch_g1_hpc_outputs.sh`（回取）
  - `.codex-fleet/rq015g-hpc-resolve/work/local_input_manifest.json`（9 个 PKL 的 SHA-256）
- 上一次的远端 work_dir：
  `/share/home/u25310231/ZXC/sociality_estimation/work_dirs/INFRA/rq015g_anchor_resolve_20260801T014419Z`
  **不得覆盖它。你必须新建自己的 work_dir。**
- SSH 别名 `tongji-hpc`，用户 `u25310231`。作业名必须以 `zxc-` 开头。
- 跨项目 HPC 指南：`../HPC_TONGJI_USAGE_GUIDE.md`（相对本仓库根的上一级目录）

### 2.5 G 轨的三个时间口径（**三个都要报，不得只用其中一个**）

监督方已逐一核过来源，三个数字都真实存在，指的是不同的东西：

| 口径 | 数字 | 来源 |
|---|---:|---|
| 求解循环 | 500.6 s | `.codex-fleet/rq015g-hpc-resolve/board/progress.log` 第 28 行 `completed=2300/2300 elapsed=500.6s` |
| driver 完整耗时 | 702.9936774782836 s | `.codex-fleet/rq015g-hpc-resolve/work/g1_hpc_summary.json` 的 `t5.elapsed_seconds`（同处 `seconds_per_anchor_wall=0.3056494249905581`） |
| Slurm 作业墙钟 | 862 s（00:14:22） | `.codex-fleet/rq015g-hpc-resolve/board/reports/G_leader_adjudication.md` 第 5 行 |

三者都是 2,300 锚点 / 6 worker / 单节点。线性外推到 13,980,600 个 ATTEMPTED 行，
分别约 **35 / 49 / 61 天**，不含排队与重投。

**内存风险不是理论风险**：24 worker 曾因每进程各载一份 PKL 而 OOM，
TRES `mem=160992M` 仍不够（`G_leader_adjudication.md` 第 242-255 行记录 7 次投递）；
最终 `fata` 分区 6 worker 才完成。

### 2.6 J1 已给出的对照结果（K2 验收用，K1 不必重算）

- 全域可估率 design-based estimate：**71.2695%**，95% CI `[67.1729%, 75.2135%]`
  （HT 分母 2,646,058；门后保留权重 1,885,831.096；B=2000、seed=20260731、cluster 数 1,909）
- 两条弃权原因的全域权重占比：`NO_IPV_EFFECT` 0.5095%、`NEAR_UNIFORM` 28.2210%
- 门后（`status=OK`）的行里有 **23.40%** 的 `ipv_log` 恰好为零（判据 `|ipv_log| <= 1e-9`；
  分 signature 为 N 12/363、U 91/511、Z 135/143；占门后 HT 权重 10.2788%）

---

## 3. 你要交付的东西

**唯一交付物**：`.codex-fleet/rq015k-fullcorpus-gate/board/reports/K1_preflight_and_plan.md`

支撑数据、脚本、中间产物全部写进
`.codex-fleet/rq015k-fullcorpus-gate/work/`（这个目录现在是空的，归你用）。

**进度日志**（leader 靠它判断你是否存活，**必须写**）：
每完成一个任务、以及每次 HPC 状态变化，向
`.codex-fleet/rq015k-fullcorpus-gate/board/K1_progress.log` **追加**一行：
```
<UTC 时间戳> | <任务号> | 做了什么 | 结论
```
HPC 排队等待期间至少每 10 分钟追加一行心跳，否则 leader 会判定你已死。

---

## 4. 七项任务

### T1 — 求解单元收敛到规范键，给出真实求解量【最可能省掉大量成本】

14,473,982 是**台账行数**，不等于需要做 7 候选求解的**单元数**。

RQ015A schema（`reports/plans/RQ015A_ledger_schema_v4_20260731.json` 第 676-680 行）写明：
`rq009_feature_matrix`（8,994,736 行）是 `interhub_sigma01_hw4_timeseries`（5,197,072 行）的
**派生**产物，且 `cross_artifact_pooling` 被禁止。

**你要做的**：

1. 把「需判门的单元」定义到规范键一层，建议形如
   `(artifact 或 source case, frame, agent/role, candidate_grid_id)`。
   你必须自己从 `product_row_key` 的实际构造中确定这个键到底是什么 ——
   读 `scripts/rq015a/build_ledger.py`（`product_row_key` 与 aggregation key 相关代码在第 1022-1076 行附近）
   与 `scripts/rq015a/run_rq015a.py`（第 495-528 行是当前 parquet writer schema）。
2. 给出**去重后的确切求解单元数**，以及分 artifact 的拆分表。
3. 明确写出：`rq009_feature_matrix` 的 8,994,736 行中，有多少能 join 回
   `interhub_sigma01_hw4_timeseries` 的已解单元而**不需要重复求解**，多少是新单元。
4. 给出「台账行数 → 求解单元数」的压缩比，以及它把第 2.5 节的 35/49/61 天外推改成了多少。

**这一条若能把求解量从 1,398 万压到 500 万量级，是本轮最大的成本节省。务必做实。**
不要靠推理下结论，要用实际 join 验证并给出计数。

### T2 — HPC 侧是否已有逐候选量落盘（本地侧已查完，别重查）

第 2.2 / 2.3 节已经把**本地**路径查完了。**两位审查方都没有查 HPC 侧。**

**你要做的**：只查 HPC 侧，范围限定在
`/share/home/u25310231/ZXC/` 下的
`sociality_estimation/{work_dirs, checkpoints, archives, inputs, manifests, data, logs}`
以及 `RQ010B_wod_e2e`、`rq009_hw4_submit_20260626`、`rq012b_onsite_ipv_20260627T202508`、
`RQ015A_wod_readonly_retrieval`、`ipv_estimation`。

按文件名搜 `*mse*`、`*log_score*`、`*w_log*`、`*candidate*`、`*anchor*`，
并对命中的表格类文件抽查表头。**只读，不删不改远端任何东西。**

产出一张表：路径 / 查法 / 结论。查了但没有的也要列出，便于监督方复核你没漏。
**如果查到某份 HPC 产物已含全量或大批量 `mse_per_candidate[7]`，
立刻在 `K1_progress.log` 写明并把它放在报告最前面 —— 那会改变整轮的成本结构。**

### T3 — 逐 artifact 的原始数据可重算性

对 4 个 artifact **逐一**证明：能否从现有 raw / snapshot 数据重建 7 条候选轨迹。

- InterHub（`interhub_sigma01_hw4_timeseries`，4,981,984 ATTEMPTED）：
  路径较清楚，冻结 PKL 快照见 2.4 节。确认覆盖率：快照里的 PKL 是否覆盖全部 case。
- OnSite（`onsite_dense_timeseries`，2,974 ATTEMPTED）：**必须明确回答有没有重算入口。**
- WOD（`wod_rq010b_full479_audited`，906 ATTEMPTED）：**必须明确回答有没有重算入口。**
  注意 `bounded_report.md` 第 14-16、155-158 行说明本轮只覆盖 full479 的 906 行。
- `rq009_feature_matrix`：取决于 T1 的 join 结论。

**若某个 artifact 无法重算，如实写出来 —— 这是一个范围结论，不是失败。**

### T4 — 分层 pilot（唯一允许投的 HPC 作业）

**规模 2,000–5,000 单元。必须分层，不得随机一把抓。**

分层维度（**每一层分别报数字**）：
`artifact` × `source` × `measurement_role` × 轨迹长度（`n_obs` 分档）× PKL 分组，
且**必须覆盖 OnSite 与 WOD**。

注意：OnSite ATTEMPTED 只有 2,974 行、WOD 只有 906 行，
你完全可以在 pilot 里**全量覆盖这两个 artifact**，这比抽样更省事也更有说服力。

**每层必须报的硬指标**（只报平均秒/单元是不合格的）：

| 指标 | 要求 |
|---|---|
| worker 数 | 实际使用值 |
| PKL 分片方式 | 每 worker 载哪些 PKL、是否复用 |
| 每 worker 常驻内存峰值 RSS | 实测，MB |
| PKL 载入的内存放大倍数 | 磁盘大小 → 常驻内存 |
| 墙钟 | solve-loop / driver / Slurm 三个口径都报 |
| 每单元耗时 | 分层给出，含长尾（P50 / P90 / P99 / max） |
| Slurm `--mem` 建议值 | 由实测 RSS 推出，留出裕度 |
| 失败率与失败类型 | 见 T5 的分类 |

**HPC 操作要求**：
- 新建 work_dir，形如
  `/share/home/u25310231/ZXC/sociality_estimation/work_dirs/INFRA/rq015k_pilot_<UTC时间戳>`，
  **不得覆盖任何既有目录**
- 作业名 `#SBATCH --job-name=zxc-rq015k-pilot`，日志放该 work_dir 的 `logs/`
- 线程环境变量按 2.4 节钉死
- 只能用 `sbatch` 提交，**绝不在登录节点跑计算**
- 复用 2.4 节列出的现成脚本，按需改造成 K 版本放进
  `.codex-fleet/rq015k-fullcorpus-gate/work/`，**不要修改 G 轨的原文件**
- 用 `squeue -u u25310231` / `sacct` 监控，记录 job id 和全部投递尝试

**排队期间不要干等。** 提交后立即回到本地做 T1 / T5 / T6 / T7，轮询 HPC 状态。
若队列拥塞超过 2 小时仍未起跑，在 `K1_progress.log` 写明，
先把其余任务全部做完并写进报告，pilot 结果留待补充 —— 不要因为排队而空转。

### T5 — 资源方案与失败恢复设计（从原则变成验收项）

给出：分区、节点数、每节点 worker 数、每作业单元数、预计墙钟（三口径）、总 worker-hours。

并把以下四项写成**可验收的具体规则**，不是原则性描述：

1. **分片**：每个 shard 固定输入范围、PKL 清单、行键范围、行数、预期输出行数。
   同一个规范键不得因为派生行而被重复求解。
2. **断点续算**：输出先写临时文件，校验通过后原子 rename。
   已完成 shard 由 manifest 的「输入 SHA + 代码 SHA + 命令 + 行数 + 输出 SHA」判定，
   **不能只看文件存不存在**。重复投递必须幂等。
3. **失败重投**：只重跑失败 shard。失败原因至少分
   `OOM / TIMEOUT / SOLVER_FAILURE / NON_FINITE_INPUT / SCHEMA_MISMATCH`。
   超过阈值必须停下上报，**不得扩大资源硬跑**（给出你建议的阈值）。
4. **产物校验**：主键唯一、无缺失或重复分片、`K=7`、数组长度为 7、有限性规则、
   reason 互斥顺序、`ipv_log` null 规则、各状态计数、抽样重跑一致性、
   输入清单 SHA、`held_out_parsed_rows = 0`。

### T6 — RQ009 join 干跑【go/no-go 条件之一】

抽样解析 `rq009_feature_matrix.parquet` 的 `product_row_key`，
证明 `context_cell_key`（或其生成输入：`source_dataset` / `anchor_frame_index` /
`perspective` / `measurement_role` 等）能**一对一**取到。

**若取不到，K2 的产物接不上下游 —— 这必须作为 go/no-go 结论明确写出来。**

参考：RQ009 的 calibration 代码在
`reports/studies/RQ009_.../02_process/04_calibration/calibration.py`
（gate numeric features 在第 141-157 行，OOD gate 在第 704-715 行）。
**检索限定在这一个 RQ009 目录内，不要对 reports/ 做全仓库搜索。**

### T7 — 附录（不阻断 K2，只需照抄说明，不要量化）

RQ009 的分布外判据用了 `counterpart_ipv_current`、`counterpart_ipv_error_current`、
`counterpart_ipv_slope_pre_anchor` 三列。当对手车的 IPV 不可估时，这三列被填成什么？

**规则已经查明，你直接引用即可，不要重新调查、不要量化污染规模、不要重算 RQ009 的 4.78% 弃权率：**

> `build_features.py` 第 774-776 行直接沿用上游 legacy IPV / error 数值；
> slope 由 `theil_sen_slope()` 计算，有效历史点少于 2 个时返回 NaN（第 583-597 行）；
> 之后被 calibration 的中位数填补吸收（`Preprocessor` 第 211-235 行、gate 第 704-715 行）。

写成报告附录一节，一段话说清即可。

---

## 5. K2 的输出契约 —— 你要在 K1 报告里把它冻结下来

K1 报告必须包含一节完整的 K2 输出契约。以下 6 条是强制内容，**照写，不得"优化"**：

### 5.1 门适用性契约【最重要】

**只对 `attempt_status == ATTEMPTED` 的行计算本门**（13,980,600 行）。

其余 219,360 行 `NOT_ATTEMPTED` 与 274,022 行 `UNKNOWN` 必须写：
`gate_applicable = false`、全部门字段为 null，并保留 `source_attempt_status` 与 `source_reason_code`。

**绝不允许把"上游没有有效输入"写成 `NO_IPV_EFFECT` 或 `NEAR_UNIFORM`** ——
那会把工程缺失混进科学弃权原因，直接污染 RQ009 的弃权分布与 context cell 通过率。
这是两位审查方共同点名的头号风险。

### 5.2 工程失败必须有独立状态

只有 `OK` / `ABSTAIN` 两个状态是不够的。
`src/sociality_estimation/core/reliability_logdomain.py` 第 36-49 行已有
`NON_FINITE_INPUT`、`SOLVER_FAILURE` 等互斥状态，**复用它们，不要另造名字**。
这类行 `ipv_log = null`、`w_log[7]` 全 null，**不得归入两个科学 reason**。

### 5.3 σ 必须钉死在批处理规格里

```
log_score_i = -mse_i / (2 * sigma^2)，sigma = 0.1
```
（B1 代码 `SIGMA=0.1`、七点候选网格 `legacy7_pi_over_8`）。
若不同 artifact 误用不同 σ，`max_w_log` 与 `k_eff_log` 会整体偏移、门判定整体失真。
**输出必须持久化 `log_score[7]`，或记录 σ 与公式，使读者能自行复算 `w_log[7]`。**

### 5.4 `mse_spread == 0` 是精确浮点相等，不是容差

先断言 7 个 MSE 均为有限 float64、长度为 7，再判 `max - min == 0.0`。
**不得改成 `np.isclose`** —— 这是精确退化标签
（Mac / HPC 两环境对这 400 行 MSE 逐位相同已在 G 轨证实），不是阈值。
非有限的行走 5.2 的状态，不进这两个科学 reason。

### 5.5 softmax 复用现成实现

`reliability_logdomain.py` 第 172-188 行的 `weights_from_mse()` 已是稳定实现
（减最大值后 exp 再归一，并检查分母有限）。第 167-169 行是 `candidate_mse`。
**直接调用，不要另写一份。**

### 5.6 互斥 reason 必须有序赋值

样本内 400 个 `mse_spread == 0` 的行**全部**也满足 `max_w_log < 0.20`。
向量化实现必须用有序 `np.select`，或先写 `NO_IPV_EFFECT`、再只在 `reason is null` 的行写 `NEAR_UNIFORM`。
**两个布尔掩码顺序覆盖会写反。**

### 5.7 门规格本身（照抄，不得改动）

```text
输入（单帧内可得）: frame_id, candidate_grid_id, K=7, candidate_ipv[7],
                    mse_per_candidate[7], log_score[7], context_cell_key

w_log      = softmax(log_score)  用 log-sum-exp
mse_spread = max(mse_per_candidate) - min(mse_per_candidate)
max_w_log  = max(w_log)
k_eff_log  = 1 / sum(w_log_i^2)

if mse_spread == 0:      status=ABSTAIN, reason_code=NO_IPV_EFFECT,  ipv_log=null
elif max_w_log < 0.20:   status=ABSTAIN, reason_code=NEAR_UNIFORM,   ipv_log=null
else:                    status=OK,      reason_code=null,
                         ipv_log = sum(candidate_ipv_i * w_log_i)
```

- **必须在 log 域算。** 连乘域下溢会把可估的行错判为不可估。
- `theta = 0.20` 是政策阈值，**不是数据自然断点**，不得据此调参、不得做阈值扫描。
- 两条判据**互斥且有序**。判据 1 的额外筛出量为 0，它保留为语义标签，
  **不得与判据 2 相加当作两份贡献**。
- 弃权时 `ipv_log` 必须为 **null**，不得为 0、NaN 或缺列。

### 5.8 台账要物化的列

至少：`mse_per_candidate[7]`、`log_score[7]`、`w_log[7]`、`candidate_ipv[7]`、
`max_w_log`、`mse_spread`、`k_eff_log`、`status`、`reason_code`、`ipv_log`（弃权为 null）、
`candidate_grid_id`、`K`、帧/行键、`context_cell_key`、
`gate_applicable`、`source_attempt_status`、`source_reason_code`，
以及审计用的 legacy `ipv_error` / `k_eff` / `q_eff`。

**建议拆成两份 parquet**：L1 行级门台账 + 聚合表。
`gate_pass_rate` 是聚合字段，不应重复塞进每一条行级记录。
你要在报告里给出具体的 schema 决定：array 列还是标量列、partition 方案、
schema version、候选顺序声明、nullability。

### 5.9 必须随台账交付的下游警告（写进接口说明，不能只留在报告正文）

门后（`status = OK`）的行里有 **23.40%** 的 `ipv_log` 恰好为零。
这是本门要消除的那处混淆的**镜像情形**：本门保证「弃权」不再被写成 `ipv = 0`，
但反过来**不成立** —— `ipv_log = 0` 仍是合法且高频的**通过门**的估计值（中性社会倾向）。

**判别字段只能是 `status` 与 `reason_code`，不是 `ipv_log` 的数值。**

旁证：RQ009 自己的 `decision.md` 的 Boundaries 一节写着
`Target exact-zero atom ~21.6%`，与本轮的 23.40% 量级一致，两者来源独立。

---

## 6. 报告写作要求（硬性，不合格要返工）

**原则：这份报告必须能被一个完全没有跟进过程的读者一次读懂。
上下文重建的成本由你承担，不由读者承担。**

1. **先定位，再讲进度。** 开头必须交代三件事：这项工作要解决什么问题、
   整体已经走到哪一步、本次是其中哪一环。不得直接从增量讲起。
2. **不用黑话、不用比喻。** 必须使用项目专有名词时，当场用一句话说明它是什么
   （例如 IPV、context_cell_key、product_row_key、HT 权重都要解释）。
3. **结论与待决事项分开。** 需要监督方拍板的事必须**单独成节**，
   写清选项、判断依据、以及不做的后果，不得藏在叙述中间当陈述句带过。
4. **数字自带口径。** 每个百分比必须同时给分子、分母、筛选条件、来源文件与列名。
   读者无法自行复算的数字等于没给。
5. **措辞两处更正，必须遵守**：
   - **"全语料"必须限定为「RQ015A 当前 4 份本地可审计 L1 parquet ledger 的全行」**，
     不得写成"全 WOD"或"全项目全语料"。本轮覆盖 4/6 产物，WOD 只含 full479 的 906 行。
   - **不要写"只增加持久化输出列、不改任何计算"** —— 这会被误读为没有实现工作。
     事实是：`process_interhub.py` 第 1168-1175 行未启用 diagnostics、
     第 854-885 行只写 legacy IPV/error；`ipv_estimation.py` 第 340-367 行的 diagnostics 也不含 MSE/log-score。
     **K2 需要新增 materializer 或 writer。四个受保护文件仍然不许改，但不要假设零实现工作。**
6. 报告末尾必须有一节 **「K2 go / no-go 判定」**，明确写出：
   - 建议投 K2 / 建议不投 / 建议先补什么再投
   - 依据是什么
   - 若投，总资源预算（三口径墙钟 + worker-hours + 建议 `--mem`）
   - 阻断条件是否解除（T6 的 join 干跑是其中之一）

---

## 7. 结项动作

1. 写完 `board/reports/K1_preflight_and_plan.md`。
2. 在 `board/K1_progress.log` 追加最后一行，写明报告已落盘和 HPC job id。
3. 自查一遍：铁律 11 条有没有违反；报告里每个数字有没有带口径；
   第 5 节的 6 条强制内容有没有全部写进去。
4. **停下。不要提交 git commit。不要投全量作业。不要写"规格 v2"。**

**速度原则**：本轮是诊断性 / 规划性产出。一轮做完，出报告，结束。
不做多版本方案、不做盲审、不加授权闸门。发现自己在写第二版规格就是跑偏了，停下。
