# K2R_A 独立审查报告

审查对象：`.codex-fleet/rq015k-fullcorpus-gate/board/K2-leader-kickoff.md`。
本报告只判断这份 K2 任务书能否机械执行；没有提交 Slurm 作业，没有重解锚点，没有创建 K2 work_dir，没有读取其他审查方报告或日志。

结论：**需修改后执行**。第四节资源裁定**计划错了**：P6 方向可辩护，但片数、可用核心、墙钟和内存公式均不能按任务书原文执行。第二节和第五节也存在批处理实现歧义，足以造成产物看似完成但口径不可审。

本轮 A 方新增证据只在：

- `.codex-fleet/rq015k-fullcorpus-gate/work/k2r_a_review/cluster_snapshot_raw.txt`
- `.codex-fleet/rq015k-fullcorpus-gate/work/k2r_a_review/k2r_a_recompute.py`
- `.codex-fleet/rq015k-fullcorpus-gate/work/k2r_a_review/K2R_A_recalc.json`

## 1. 第四节资源裁定

### 1.1 独立重算结果

我只把 K2 任务书第八节允许的 `intel` 与 `fata` 分区计入资源池；`amd` 分区虽然在当前 `sinfo` 中存在，但任务书未把它列入 K2 通道。

当前集群快照时间为 `2026-08-02T12:15:01Z`，来源为 `cluster_snapshot_raw.txt:1`。`intel` 分区当前汇总为 `41` 个 mix 节点、`7` 个 alloc 节点、`2,517` 个 idle CPU，来源为 `cluster_snapshot_raw.txt:591-594`；`fata` 分区当前为 `2` 个 mix 节点、`1` 个 idle 节点、`321` 个 idle CPU，来源为 `cluster_snapshot_raw.txt:600-603`。合计可用节点口径为非 down/drain/inval 的 `51` 个节点、`2,838` 个 idle CPU、`33,786,878 MB` free memory，来源为 `K2R_A_recalc.json` 字段 `current_wall_by_shape.P6_fixed_48G.{usable_nodes,idle_cores,free_memory_mb}`。

K2 真正要重解的 InterHub 求解单元为 `4,981,984`，来源为 `k1_t1_t6_local_analysis.json` 字段 `per_artifact[artifact=interhub_sigma01_hw4_timeseries].canonical_solve_units_charged_to_this_artifact`，同数也见 `t1_solve_units_by_artifact.csv` 的 `interhub_sigma01_hw4_timeseries` 行、列 `canonical_solve_units_charged_to_this_artifact`。

| 配置 | 每片 worker | `--mem` | 同时可开片数 | 活跃核心 | 总核·小时 | 墙钟 | 来源 |
|---|---:|---:|---:|---:|---:|---:|---|
| **我采用的保守配置** | 6 | 48G | 447 | 2,682 | 2,893.75 | 1.079 h | `K2R_A_recalc.json` 字段 `current_wall_by_shape.P6_fixed_48G`；吞吐来自 `results/P6/k1b_pilot_summary.json` 字段 `interhub_rows=1120`, `solve_loop_elapsed_seconds=390.3266615867615` |
| P6 按任务书公式 | 6 | 24G | 457 | 2,742 | 2,893.75 | 1.055 h | `K2R_A_recalc.json` 字段 `current_wall_by_shape.P6_formula_24G` |
| P10 按 K1b 公式 | 10 | 40G | 264 | 2,640 | 3,072.86 | 1.164 h | `K2R_A_recalc.json` 字段 `current_wall_by_shape.P10_formula_40G` |
| P16 按 K1b 公式 | 16 | 64G | 157 | 2,512 | 3,423.42 | 1.363 h | `K2R_A_recalc.json` 字段 `current_wall_by_shape.P16_formula_64G` |

因此，在当前快照下，如果沿用 K2 固定的 48G 安全余量，正确值是：

**447 片 / 每片 6 worker / `--mem=48G` / 2,893.75 核·小时 / 1.079 小时。**

任务书第四节写的是 `418` 片、`2,508` 核、`2,893` 核·小时、`1.15` 小时，来源为 `K2-leader-kickoff.md:74-77`。**计划错了**：核·小时基本对，片数、活跃核心和墙钟不对。

### 1.2 任务书旧快照下也算错

K2 第四节引用的是 K1b 投递前快照：`intel` idle CPU `2,384`、`fata` idle CPU `130`、合计 `2,514`，来源为 `K2-leader-kickoff.md:69-72` 与 `cluster_snapshot_pre_submit.txt:127-195,242-245`。

按该旧快照逐节点装箱，而不是把全分区 idle CPU 直接相除，P6/48G 的正确同时可开片数是 `402`、活跃核心 `2,412`、墙钟 `1.200 h`；P16/64G 的正确同时可开片数是 `134`、活跃核心 `2,144`、墙钟 `1.597 h`。来源为 `K2R_A_recalc.json` 字段 `pre_submit_wall_by_shape.{P6_fixed_48G,P16_formula_64G}`。

任务书给 P6 `418` 片、P16 `157` 片，来源为 `K2-leader-kickoff.md:74-77`。**计划错了**：它把跨节点总 idle CPU 当成一个连续资源池，没有按 Slurm 单任务必须落在单节点上的约束逐节点取 `floor(idle_cpus / workers)`，也没有把 P6/48G 在若干节点上的 free-memory 限制纳入装箱。

### 1.3 起作用的约束

QOS 不是约束。K1b 记录的 QOS 名称为 `cpu-4000_core-l40-16_card-a800-16_card`，来源为 `K1b_memory_pilot.md:39`；本轮 `sacctmgr show assoc` 原始输出仍显示该 QOS 名称，来源为 `cluster_snapshot_raw.txt:620-621`。我采用的 P6/48G 配置活跃核心为 `2,682`，低于 `4,000`，来源为 `K2R_A_recalc.json` 字段 `current_wall_by_shape.P6_fixed_48G.active_cores`。

单节点内存上限也不是约束。P6/48G 单片要求 `48G`，而 `intel` 节点内存为 `644000 MB`、`fata` 节点内存为 `3094000 MB`，来源为 `cluster_snapshot_raw.txt:401-469,516-519` 的节点字段 `MEMORY`。

真正约束是**逐节点空闲资源装箱**。在当前 P6/48G 下，CPU-only slot 为 `457`，memory-only slot 为 `664`，最终 slot 为 `447`；其中 `31` 个节点 CPU 更紧、`8` 个节点内存更紧、`12` 个节点两者并列，来源为 `K2R_A_recalc.json` 字段 `current_wall_by_shape.P6_fixed_48G.{cpu_only_slots,memory_only_slots,slots,node_binding_counts}`。所以不能只看全分区 idle CPU 总数。

### 1.4 内存公式内部不自洽

K2 第四节固定采用 P6/`--mem=48G`，来源为 `K2-leader-kickoff.md:76,79`。同一节又给出自适应公式 `ceil_to_8G(2.789 GiB × workers × 1.3)`，来源为 `K2-leader-kickoff.md:85-87`。

把 `workers=6` 代入该公式，得到 `ceil_to_8G(2.789 × 6 × 1.3)=24G`；来源为 `K2R_A_recalc.json` 字段 `k1b_config_summary.P6.mem_g_from_k2_formula_2p789gib`。用 K1b 原始 RSS 求和也得到 `24G`，来源为 `K2R_A_recalc.json` 字段 `k1b_config_summary.P6.{rss_sum_gib=16.33676528930664,mem_g_from_raw_rss_30pct=24}`。**计划错了**：固定 48G 与本节公式不一致。

我仍把 48G 作为保守执行资源重算，是因为任务书明确说 OOM 是已发生过的主要失败模式，且 48G 对 P6 实测 `16.3368 GiB` RSS sum 是 `2.94x` 余量；来源为 `K2-leader-kickoff.md:82-83` 与 `K2R_A_recalc.json` 字段 `k1b_config_summary.P6.rss_sum_gib`。

### 1.5 墙钟外推口径

K2 第四节的核效率比较用的是每 worker 吞吐：P6 `2.8693914872403545 / 6 = 0.4782319145400591` units/s/worker，P16 `6.467853906643005 / 16 = 0.4042408691651878` units/s/worker；来源为 `K2R_A_recalc.json` 字段 `k1b_config_summary.{P6,P16}.throughput_units_per_worker_second`。这部分方向正确。

墙钟外推用的是每片吞吐和同时可开片数，不是每 worker 吞吐。P6 原始 `solve_loop_elapsed_seconds=390.3266615867615` 与 `driver_elapsed_seconds=390.383811712265` 基本相同，来源为 `results/P6/k1b_pilot_summary.json` 同名字段；`k1b_progress.log:1-14` 记录 P6 从 driver started 到 driver complete。因此 pilot 的进程启动/载入时间已经计入 P6 吞吐。这个外推对 1,120 行 pilot 的冷载入开销是保守的；但它只测了 `waymo_0-299.pkl` 单个 PKL，来源为 `K1b_memory_pilot.md:22-29,47-53`，不能证明所有 PKL 行耗时分布一致。

## 2. 第二节门规格搬到批处理的可执行性

### 2.1 `weights_from_mse()` 与公式一致，但调用方必须显式传 sigma

`weights_from_mse(mse, sigma)` 的函数签名有两个必填参数，`sigma` 没有默认值，来源为 `src/sociality_estimation/core/reliability_logdomain.py:172`。函数先把 `mse` 转为 `dtype=float`，调用 `_check_sigma(sigma)`，再计算 `logw = -mse / (2.0 * sigma ** 2)`，来源为 `src/sociality_estimation/core/reliability_logdomain.py:174-180`。`_check_sigma()` 只检查有限正数，不内置 `0.1`，来源为 `src/sociality_estimation/core/reliability_logdomain.py:115-125`。

结论：函数实现与 K2 第二节 `log_score_i=-mse_i/(2*sigma^2), sigma=0.1` 一致，来源为 `K2-leader-kickoff.md:27-28,52-53`。风险在调用方：任务书必须保证所有 writer/materializer 都显式传 `sigma=0.1` 并把输出 `sigma` 写入 manifest；函数本身不会替调用方钉死 `0.1`。

### 2.2 `mse_spread == 0` 的 float64 条件没有贯穿到输出 schema

K2 第二节要求先断言 7 个 MSE 均为有限 float64 且长度为 7，再用精确相等 `max-min==0.0`，来源为 `K2-leader-kickoff.md:45-46`。如果输入和 parquet schema 都保持 float64，parquet 往返不会破坏精确零 spread。若 writer 经过 float32 降精度，小差值可能被压成相等，或者边界判断发生变化。

任务书第九节只说 Arrow fixed-size list 或标量列 fallback，来源为 `K2-leader-kickoff.md:291-297`，没有把 element dtype 写成强制 double。结论：第二节表达了正确要求，但批处理输出要求仍有缺口，不能机械阻止 float32 writer。

### 2.3 reason 顺序写得明确，但工程失败状态枚举不清

科学 reason 顺序写得足够明确：先 `NO_IPV_EFFECT`，否则再 `NEAR_UNIFORM`，并提示两个 boolean mask 顺序覆盖会写反，来源为 `K2-leader-kickoff.md:45-51`。这能防住最常见的写反错误。

但工程失败状态不够清楚。第二节伪代码写 `status=ENGINEERING_FAILURE, reason_code=NON_FINITE_INPUT 或 SOLVER_FAILURE`，来源为 `K2-leader-kickoff.md:33-35`；第五节又把 `SOLVER_FAILURE`、`NON_FINITE_INPUT` 当失败类型处理，来源为 `K2-leader-kickoff.md:105-110`；既有 log-domain 结果合同使用 `STATUS_NON_FINITE_INPUT` 与 `STATUS_SOLVER_FAILURE` 作为主状态，来源为 `src/sociality_estimation/core/reliability_logdomain.py:36-49`。结论：工程失败与科学 reason 的隔离原则正确，但 schema 枚举不唯一，批处理实现可能出现两套状态写法。

### 2.4 缺列、NaN、长度错误、单行缺数组

覆盖情况如下：

| 情况 | K2 任务书覆盖 | 审查结论 |
|---|---|---|
| NaN / inf | `输入非有限` 走工程失败，来源 `K2-leader-kickoff.md:33-35`；第五节要求科学 reason 前 MSE 有限，来源 `K2-leader-kickoff.md:112-114` | 覆盖 |
| 数组长度不是 7 | 第二节要求长度 7，来源 `K2-leader-kickoff.md:24-25,45-46`；第五节要求数组长度恰为 7，来源 `K2-leader-kickoff.md:112-114` | 覆盖 |
| 整个必需列缺失 | 第二节写“缺列”走工程失败，来源 `K2-leader-kickoff.md:33-35`；第五节写 `SCHEMA_MISMATCH` 立即整体停止，来源 `K2-leader-kickoff.md:105-107` | **不清楚**：表级缺列应整体停止，不能作为每行工程失败继续产出 |
| 某一行没有 `mse_per_candidate` | 第二节没有把 null list、缺 list、空 list 分开；第五节只写长度和有限性校验，来源 `K2-leader-kickoff.md:112-114` | **不清楚**：应落到哪个 failure_type 不能机械判定 |

## 3. 第五节分片、续算、重投、校验

### 3.1 `expected_output_rows`

对 InterHub solve shard，`expected_output_rows` 可以从分片时的 canonical unit table 直接得到：等于该 shard 的 canonical key 行数。来源口径是 `k1_t1_t6_local_analysis.json` 字段 `per_artifact[interhub].canonical_solve_units_charged_to_this_artifact=4981984` 与 K1 报告 canonical key 定义 `K1_preflight_and_plan.md:21-40`。

但 K2 还要求 RQ009 的 `8,994,736` 行由 InterHub 结果 join 回填，来源为 `K2-leader-kickoff.md:59-61`。任务书没有定义 RQ009 join shard 的 manifest，也没有说明非 `ATTEMPTED` 行和 OnSite/WOD pass-through 行由哪个 shard 输出。结论：`expected_output_rows` 对 solve shard 可算；对完整 K2 L1 产物不可机械判定。

### 3.2 canonical key 唯一性

K1 定义的 InterHub canonical key 是 `scene_unique_id | frame_index | measurement_role | candidate_grid_id`，来源为 `K1_preflight_and_plan.md:23-30`。这个 key 不含 PKL 文件名。按 `(单个 PKL, 行键区间)` 切，只能在以下条件同时成立时保证唯一：源 unit table 已按 canonical key 去重；同一 `scene_unique_id/frame_index/role/grid` 不会出现在多个 PKL；row-key 区间严格不重叠。

任务书只写“同一 canonical key 只能出现在一个分片里”，来源为 `K2-leader-kickoff.md:93-99`，没有要求生成 manifest 时输出全局去重证明。因此这条不是自动成立，需要额外机器证明。

### 3.3 重投阈值

K1 InterHub pilot 为 `1,120 / 1,120 OK`，来源为 `K1_preflight_and_plan.md:112` 与 `k1_pilot_summary.json` 字段 `interhub_rows`、`failure_counts_interhub.OK`。K1b 三个配置各 `1,120 / 1,120 OK`，来源为 `K1b_memory_pilot.md:63-69` 与 `results/{P6,P10,P16}/k1b_pilot_summary.json` 字段 `failure_counts_interhub.OK`。K1b 三配置是同一批 1,120 行，不应当按 3 个独立样本扩大分母。

按我采用的 P6/48G 当前配置，平均每片 `4,981,984 / 447 = 11,145.38` 行，来源为 `K2R_A_recalc.json` 字段 `current_wall_by_shape.P6_fixed_48G.mean_rows_per_slot`。阈值换算为：

- `SOLVER_FAILURE`：`min(100, 2.0% × 11,145.38)=100` 行，即平均片约 `0.897%`；来源为 `K2R_A_recalc.json` 字段 `current_wall_by_shape.P6_fixed_48G.solver_failure_stop_rows_at_mean_shard`。
- `NON_FINITE_INPUT`：`0.1% × 11,145.38=11.15` 行，实际超过阈值大约从 `12` 行开始；来源为 `K2R_A_recalc.json` 字段 `current_wall_by_shape.P6_fixed_48G.non_finite_0p1pct_rows_at_mean_shard`。

按 K1/K1b 实测失败率 `0 / 1,120` 与 `0 / 1,120`，正常情况下不会触发这两个阈值。`SOLVER_FAILURE` 阈值相对 pilot 很松；`NON_FINITE_INPUT` 阈值较紧，但它针对输入污染，紧是合理的。真正问题不是阈值，而是前述工程失败枚举和表级缺列处理不清。

### 3.4 `.tmp` 与原子 rename

同目录内 `rename` 在 POSIX 语义下是原子的；K2 要求 `<shard>.tmp.parquet` 与 `.tmp.manifest.json` 校验后 rename，且文件存在不等于完成，来源为 `K2-leader-kickoff.md:100-103`。这条方向正确。

但 parquet 与 manifest 是两个文件，两个 rename 不是一个事务。若 parquet 已 rename、manifest 未 rename 时进程退出，只要完成判据以最终 manifest 为唯一完成标记，就不会误判完成；若下游扫描 final parquet 而不先核 manifest，就会误判。任务书没有明确 rename 顺序与“manifest last”规则，所以这条还不能完全机械执行。

## 4. 第六节验收判据

### 4.1 与 J 设计基 CI 对照

J1 给出的全域设计基保留率为 `1,885,831.096 / 2,646,058 = 71.2695%`，CI 为 `[67.1729%, 75.2135%]`，`B=2000`、seed `20260731`、cluster `1,909`，来源为 `j1_gate_spec_evidence.json` 字段 `global_design_based_not_census.{ht_retained_weight,ht_denominator,ht_gate_pass_rate_pct,cluster_bootstrap}`，同见 `J1_gate_spec_and_impact.md:113-123`。

漏判情形：K2 如果把 reason 顺序写反、漏掉 RQ009 join 行、或把工程失败混进科学 reason，只要总保留率仍落在 CI 内，这条会放过。K2 如果只覆盖 InterHub/RQ009，而 J1 的设计基目标域与 K2 普查域不完全一致，这条也无法定位错误。

误报情形：CI 是概率样本的估计区间，不是 K2 普查结果的机械等值判据；即使 K2 正确，真实普查值也可能因抽样误差、域差异或 source 构成差异落在 CI 外。任务书写的是“必须重点解释”，来源为 `K2-leader-kickoff.md:121-124`，所以它可以作为异常提示，不能作为成功/失败判据。

缺少的判据：从 K2 产物逐行重算 `mse_spread`、`max_w_log`、`k_eff_log`、`status`、`reason_code`、`ipv_log null` 规则，并与产物逐字段一致。现有第六节没有这个全量机械复算。

### 4.2 G 锚点逐位一致

G 锚点要求与 `.codex-fleet/rq015g-hpc-resolve/work/anchor_mse_hpc.csv` 的重叠单元逐位一致，来源为 `K2-leader-kickoff.md:125-126`。G 报告显示 2,300 锚点在冻结环境中完成，HPC solve_errors/nonfinite 为 `0 / 0`，来源为 `G_leader_adjudication.md:80-94`；跨 AMD/Intel 348 个值逐位相同，来源为 `G_leader_adjudication.md:48-72`。

漏判情形：2,300 锚点不能覆盖所有 15 个 InterHub PKL、所有 source、所有 row-key 区间、RQ009 join、OnSite/WOD pass-through、非 `ATTEMPTED` 行、parquet schema 和续算逻辑。2,300 锚点来源见 `G_leader_adjudication.md:80-94`；15 个 PKL 来源见 `K1_preflight_and_plan.md:64-67`。重叠锚点一致只能证明核心求解在这些锚点上没有漂移。

误报情形：如果比较用 CSV 字符串而不是解析后的 float64 数组，格式化差异会被误报；如果 key 对齐或候选顺序错误，也会出现不是数值本身导致的失败。K1b 的比较字段是 `mse_per_candidate[7]` CSV 字符串精确相等，来源为 `K1b_memory_pilot.md:61-67`，K2 验收必须先定义解析后逐位比较还是字符串比较。

### 4.3 worker 数不改变数值

K1b 证明 P6/P10/P16 在同一 `waymo_0-299.pkl` 样本 1,120 行上 `mse_per_candidate[7]` 零不一致，来源为 `K1b_memory_pilot.md:57-69` 与 `k1b_consistency_summary.json` 字段 `config_pair_checks`、`k1_overlap`。

漏判情形：样本只覆盖一个 PKL 和 1,120 行，不能证明其他 PKL、其他 source、失败路径、join 路径或输出 writer 在不同 worker 数下不变。若 K2 worker 数不变，这条甚至不会被触发，但仍可能有其他实现错误。

误报情形：若重跑抽查时写出格式、行顺序、候选列序列化发生变化，字符串比较可能误报；若抽查样本改变而含有之前未覆盖的合法边界行，也可能被解释成 worker 数导致的变化。

缺少的判据：全量覆盖对账和 schema 对账。至少要机械确认输入行数、输出行数、主键唯一、RQ009 join `8,994,736 / 8,994,736`、非 `ATTEMPTED` 行 `219,360 + 274,022` 的 null 规则、OnSite/WOD 的非科学状态、`held_out_parsed_rows=0`、固定候选顺序、数组 double 类型、以及 manifest SHA/command/code 一致。第六节只列了其中少数项。

多出的判据：J 设计基 CI 不应作为硬成功判据；它只能触发解释。worker 数抽查只在 worker 数变更时有意义，不能替代全量逐行规则校验。

## 5. 第三节与第七节范围数字核验

### 5.1 第三节范围数字

| 任务书数字 | 核验结果 | 来源 |
|---|---:|---|
| InterHub 求解单元 `4,981,984` | 正确 | `k1_t1_t6_local_analysis.json` 字段 `per_artifact[interhub_sigma01_hw4_timeseries].canonical_solve_units_charged_to_this_artifact`；`t1_solve_units_by_artifact.csv` 同 artifact 行同列 |
| RQ009 行数 `8,994,736`，join 回填 `8,994,736`，0 漏 0 重 | 正确 | `k1_t1_t6_local_analysis.json` 字段 `per_artifact[rq009_feature_matrix].{attempted_rows,joined_rows_to_sigma01,new_rows_not_in_sigma01}`；干跑样本 `t6_rq009_context_join_dry_run.exact_one_matches=512,misses=0,duplicates=0` |
| OnSite `2,974` | 正确 | `k1_t1_t6_local_analysis.json` 字段 `per_artifact[onsite_dense_timeseries].attempted_rows` |
| WOD `906` | 正确 | `k1_t1_t6_local_analysis.json` 字段 `per_artifact[wod_rq010b_full479_audited].attempted_rows` |
| `NOT_ATTEMPTED=219,360` | 正确 | `K2R_A_recalc.json` 字段 `not_attempted_total`，由 InterHub `215,088` 加 OnSite `4,272` 得出；原字段为 `per_artifact[*].not_attempted_rows` |
| `UNKNOWN=274,022` | 正确 | `K2R_A_recalc.json` 字段 `unknown_total`；原字段为 `per_artifact[onsite_dense_timeseries].unknown_rows` |

注意：K1 的四产物 canonical solve units 合计为 `4,985,864`，来源为 `k1_t1_t6_local_analysis.json` 字段 `canonical_solve_units_total`；K2 第三节已经裁定 OnSite/WOD 本轮不重解，来源为 `K2-leader-kickoff.md:59-62`，所以资源核算应使用 InterHub `4,981,984`，不是 `4,985,864`。

### 5.2 第七节 23.40% 与 signature 拆分

我用 `anchor_mse.csv` 的列 `mse_per_candidate[7]`、`w_log[7]`、`ipv_log`、`signature` 复算，筛选 `mse_spread != 0` 且 `max(w_log) >= 0.20`；零判定为 `abs(ipv_log) <= 1e-9`。HT 权重只用于补充权重占比，来源为 `mechanism_split.csv` 列 `ht_weight`。输入文件行数均为 `2,301`（含 header），来源为 `wc -l .codex-fleet/rq015b-repair/work/{anchor_mse.csv,mechanism_split.csv}`。

复算结果：门后 `1,017` 行中 `238` 行 `abs(ipv_log)<=1e-9`，比例 `238 / 1,017 = 23.4022%`；分 signature 为 N `12 / 363`、U `91 / 511`、Z `135 / 143`；门后 HT 权重中零点质量为 `193,840.21400000024 / 1,885,831.0959999831 = 10.2788%`。来源为 `K2R_A_recalc.json` 字段 `anchor_zero_recalc.{ok_rows,zero_ipv_log_rows_abs_le_1e_minus_9,zero_ipv_log_fraction_of_ok_pct,ok_by_signature,zero_by_signature,zero_ht_weight,retained_ht_weight,zero_ht_share_of_retained_pct}`。

因此 K2 第七节的 `23.40%` 与 N/U/Z 拆分正确，来源对照为 `K2-leader-kickoff.md:130-139`。

## 6. 共同收尾问题

### Q1. 第四节的资源配置正确吗？

**不正确。计划错了。**

按当前集群快照和 K2 允许的 `intel`/`fata` 分区，我的保守配置是：

**447 片 / 每片 6 worker / `--mem=48G` / 2,893.75 核·小时 / 1.079 小时。**

来源：`cluster_snapshot_raw.txt:1,591-603,620-621`；`K2R_A_recalc.json` 字段 `current_wall_by_shape.P6_fixed_48G`；K1b P6 吞吐来源 `results/P6/k1b_pilot_summary.json` 字段 `interhub_rows` 与 `solve_loop_elapsed_seconds`；InterHub 分母来源 `k1_t1_t6_local_analysis.json` 字段 `per_artifact[interhub].canonical_solve_units_charged_to_this_artifact`。

监督方错在三步：

1. 把分区总 idle CPU 当成连续资源池，没有逐节点装箱。旧快照下 P6/48G 是 `402` 片，不是 `418` 片；P16/64G 是 `134` 片，不是 `157` 片。来源为 `K2R_A_recalc.json` 字段 `pre_submit_wall_by_shape`。
2. 固定 `--mem=48G` 与本节公式不一致。公式代入 P6 得 `24G`，不是 `48G`。来源为 `K2R_A_recalc.json` 字段 `k1b_config_summary.P6.{mem_g_from_k2_formula_2p789gib,mem_g_from_raw_rss_30pct}`。
3. 墙钟把错误片数当成活跃核心。`2,893.75 / 2,508 = 1.154 h` 能复现任务书的量级，但 `2,508` 核本身不是当前逐节点装箱结果，也不是 K1b 旧快照逐节点装箱结果。

按我的 P6/48G 配置投出去，最坏情况是：如果投递时空闲资源低于本轮快照，Slurm 会把 array 分成更多波次，墙钟按实际活跃核心增加；如果某些未测 PKL 的 RSS 高于 `waymo_0-299.pkl` pilot，OOM 会触发 3-worker 重投，重投规则来源为 `K2-leader-kickoff.md:105-108`。资源层最坏结果是排队/拖时，不是数值污染。真正会造成产物损失的风险来自第二节和第五节的 schema/manifest 歧义。

### Q2. 第六节的三条验收判据，能不能真正判定 K2 成功？

**不能。**

1. J 设计基 CI：能提示异常，不能判定成功。漏判：总保留率在 CI 内但 reason 顺序、工程失败隔离、RQ009 join 或非 `ATTEMPTED` null 规则错误。误报：K2 普查值正确但因抽样误差或目标域差异落在 CI 外。
2. G 锚点逐位一致：能证明重叠锚点求解未漂移，不能覆盖全量批处理。漏判：其他 PKL/source、join、pass-through、schema、续算错误。误报：字符串格式、候选顺序、key 对齐问题造成的非数值失败。
3. worker 数抽查：能检查单 PKL 样本的并发确定性，不能证明全量确定性。漏判：未覆盖的 PKL、失败路径、writer 逻辑。误报：格式或行顺序差异。

缺少的判据是全量机械复算与对账：逐行重算门字段、schema/dtype、主键唯一、分片覆盖、RQ009 join 行数、非 `ATTEMPTED` 与 OnSite/WOD 的非科学状态、manifest SHA/command/code、`held_out_parsed_rows=0`。多出的硬判据是把 J CI 当成功/失败；它只能要求解释。

### Q3. 明确判定：可执行 / 需修改后执行 / 不应执行

**需修改后执行。**

理由：

- 第四节资源裁定计划错了，且固定内存与自适应公式互相矛盾。
- 第二节在工程失败状态枚举、表级缺列、单行缺数组、float64 输出 schema 上仍有歧义。
- 第五节没有把 `expected_output_rows`、RQ009 join shard、非 `ATTEMPTED` 输出、canonical key 全局唯一证明写成机械规则。
- 第六节的三条验收只能发现部分错误，不能判定 K2 完整成功。

最可能造成实际损失的一处：**第五节没有定义完整 L1 产物的分片与完成判据**。如果只按 InterHub solve shard 的 manifest 判完成，RQ009 的 `8,994,736` 回填行、非 `ATTEMPTED` 的 `219,360` 行、`UNKNOWN` 的 `274,022` 行、以及 OnSite/WOD 的非科学状态行可能缺失或由人工解释补齐，产物仍可能被误认为完成。这会直接污染交付给 RQ009 的接口台账。

## 7. 边界自查

- 未提交 `sbatch` / `srun` / `salloc`；本轮集群访问只有 `sinfo`、`squeue`、`sacctmgr show assoc`，原始输出保存在 `cluster_snapshot_raw.txt`。
- 未创建 K2 work_dir；只创建 A 方审查工作目录 `work/k2r_a_review/`。
- 未修改 `agent.py` / `ipv_estimation.py` / `process_interhub.py` / `reliability_logdomain.py` / `configs/ipv_sigma01_exact.json`；本轮 SHA 保存在 `work/k2r_a_review/protected_sha256.txt`，与 K1b closeout 的对应 SHA 一致，K1b 来源为 `K1b_memory_pilot.md:105-113`。
- 未提交 git commit；未执行 `git checkout -- .` / `git restore .` / `git stash` / `git reset --hard` / `git clean -fd`。
- 未对 `reports/` 做全仓库检索；只读取任务书指定的具体背景材料与具体输入文件。
- 本报告未触发任务书列出的两类措辞禁令。
