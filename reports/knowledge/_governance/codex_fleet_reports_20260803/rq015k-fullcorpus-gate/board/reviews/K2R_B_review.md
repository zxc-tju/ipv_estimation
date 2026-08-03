# K2R_B 独立审查报告

审查方：B  
报告时间：2026-08-02T12:15:07Z 的集群只读查询之后  
结论：**需修改后执行**

本轮要审的是 K2 任务书：把冻结的 log 域门判据物化到 InterHub 全语料，并让 RQ009 能直接 join 使用。整体状态是：K1 已确认 InterHub/RQ009 范围，K1b 已做单 PKL 资源 pilot；K2 还没有派发。本报告先给出我在未看任务书和 K1b 建议前的独立推导，再逐项对照任务书。

我没有执行 K2，没有 `sbatch` / `srun` / `salloc`，没有重解锚点，没有创建 K2 work_dir，没有读取其他审查方文件。第一阶段冻结笔记在 `work/K2R_B_review/phase1_independent_notes.md`；集群只读原始输出在 `work/K2R_B_review/cluster_readonly_raw_20260802T121507Z.md`。

## 第一阶段结论（未看方案）

### 1. 规模

InterHub 真正需要求解的是 `4,981,984` 个 canonical solve units。来源：K1 `k1_t1_t6_local_analysis.json` 的 `per_artifact[artifact=interhub_sigma01_hw4_timeseries].canonical_solve_units_charged_to_this_artifact`，K1 报告第 34-40 行同值。

RQ009 的 `8,994,736` 行不应重解，应由 InterHub 结果 join 回填。来源：同一 JSON 的 `per_artifact[artifact=rq009_feature_matrix].joined_rows_to_sigma01=8,994,736`、`attempted_rows=8,994,736`、`new_rows_not_in_sigma01=0`，K1 报告第 38、42、66-67 行同值。

OnSite `2,974` 行和 WOD `906` 行本轮不能当作可正常求解的 InterHub 范围。K1 报告第 68-71 行说明当前没有能产出 `mse_per_candidate[7]`、`log_score[7]`、`w_log[7]` 的 materializer；若写入 K2 交付，只能是明确的工程范围状态，不能写成 `NO_IPV_EFFECT` 或 `NEAR_UNIFORM`。

### 2. 资源

第一阶段我按 K1b 原始 JSON 独立计算，而不是采纳 K1b 报告建议。K1b 三点如下，来源为 `results/P*/k1b_pilot_summary.json` 的 `workers`、`interhub_rows`、`driver_elapsed_seconds`、`worker_memory.*.peak_rss_mb`，第一阶段笔记第 17-25 行已冻结：

| 配置 | driver 秒 | 总吞吐 rows/s | 每核吞吐 rows/s/core | RSS sum |
|---|---:|---:|---:|---:|
| P6 | 390.383811712265 | 2.868971 | 0.4781619 | 16,728.84765625 MB |
| P10 | 248.7333538532257 | 4.502814 | 0.4502814 | 27,881.87890625 MB |
| P16 | 173.2081036567688 | 6.466210 | 0.4041381 | 44,629.078125 MB |

独立资源结论：**P6 最优**。P16 单片最快，但每核吞吐低；在核数上限约束下，P6 总核时更低，墙钟也更短。

我第一阶段推导出的配置是：

| 项 | 值 | 来源/口径 |
|---|---:|---|
| 片数 | 665 | `ceil(4,981,984 / 7,500)`，见冻结笔记第 27-34 行 |
| 每片 worker | 6 | P6 每核吞吐最高，见冻结笔记第 21-25 行 |
| `--mem` | 32G | P6 RSS sum × 1.30 = 21,747.50 MB 后向上留调度余量，见冻结笔记第 37 行 |
| 总核·小时 | 2,894.18 | `(4,981,984 / (1120 / 390.383811712265) / 3600) * 6`，见冻结笔记第 34 行 |
| 纯 driver 墙钟 | 0.7254 h | `665` 片一波跑完，见冻结笔记第 33 行 |
| 操作预算 | 约 1.0 h | 给 Slurm dispatch、PKL 冷加载、尾片留余量，见冻结笔记第 35 行 |

按我现场只读查询，如果允许使用 `sinfo` 中显示的 CPU 分区，**约束是 QOS 的 4,000 核合计上限**，不是实际空闲核、实际空闲内存或单节点内存。证据：`sacctmgr` QOS 行是 `cpu-4000_core-l40-16_card-a800-16_card`（集群 raw 第 23-24 行）；`sinfo` 显示 `amd` 空闲核 `9,104 + 9,984`、`intel` 空闲核 `2,517`、`fata` 空闲核 `129 + 192`，CPU 分区合计空闲核 `21,926`（集群 raw 第 62、66、68、71-72 行），高于 `665*6=3,990`；`amd` idle 行单独有 52 个节点、每节点至少 `682,931 MB` free（集群 raw 第 68 行），内存不成为约束。

必须说明：K1 和任务书都只明确提到 `fata/intel` 这个数值通道。若监督方不允许泛化到 `amd` 分区，则任务书必须把分区限制写清楚，并按投递前实时 `intel+fata` 空闲核重算并发数。不能一边说“必须重查集群”，一边把旧的 418 片写成固定配置。

### 3. 分片与幂等

分片键应是 `(单个 PKL, 行键区间)`，行键为 `scene_unique_id | frame_index | measurement_role | candidate_grid_id`。manifest 必须显式列 `shard_id`、`artifact_scope`、`pkl_file_list`、`source_dataset`、`row_key_min`、`row_key_max`、`canonical_key_count`、`expected_output_rows`、输入 parquet SHA、PKL SHA、代码 SHA、命令、`sigma=0.1`、`candidate_grid_id=legacy7_pi_over_8`、`K=7`、UTC 创建时间。第一阶段冻结笔记第 43-55 行给出完整规则；K1 报告第 161-177 行也有相同框架。

一个分片“完成”必须同时满足：final parquet 和 final manifest 都存在；临时文件不存在；manifest 中输入 SHA、代码 SHA、命令、行数、输出 SHA 全匹配；canonical key 唯一且数量等于预期；validator PASS。文件存在不等于完成。

失败分类：`SCHEMA_MISMATCH` / 缺文件 / SHA 不符 / 重复键 / 缺键，重试 0 次并整体停止；`OOM` 重试 1 次；`TIMEOUT` 切半或加时重试 1 次；`SOLVER_FAILURE` 只重跑失败行 1 次，超过单片 `100` 行或 `2.0%` 取小者则停止；`NON_FINITE_INPUT` 不盲目重试，单片超过 `0.1%` 则停止。

### 4. 验收

这是普查，不是抽样。第一验收应是覆盖和行级不变量：

- InterHub 输出必须覆盖 `4,981,984` 个 canonical solve units，重复键 `0`、缺失键 `0`。来源：K1 JSON 字段和 K1 报告第 36 行。
- RQ009 join 输出必须覆盖 `8,994,736` 行，exact-one join，new solve rows `0`。来源：K1 JSON 字段和 K1 报告第 38、42 行。
- `held_out_parsed_rows=0` 必须保持。来源：K1 JSON 顶层字段和 K1 报告第 21 行。
- 行级门规则必须逐行成立：`K=7`、grid 正确、数组长度 7、权重和为 1、reason 顺序正确、非 OK 的 `ipv_log` 为 null、OK 的 `ipv_log` 有限。

J1 的区间不能判定 K2 成功。J1 是设计权重估计，分母为 `2,646,058` HT 权重、`2,300` 锚点、`1,909` clusters、`B=2,000`、seed `20260731`（J1 报告第 115、246-248 行）。K2 是 InterHub 行级普查，分母是 `4,981,984` canonical solve units，并另有 `8,994,736` 个 RQ009 join rows。两边分母和域不是同一个东西。K2 结果可以和 J1 做解释性对照，但不能用是否落在 J1 CI 内来判定成功或失败。

### 5. 交付

交付至少三件：

1. `interhub_gate_ledger.parquet`：InterHub 每个 canonical solve unit 一行，含键、PKL/source 字段、`candidate_ipv[7]`、`mse_per_candidate[7]`、`log_score[7]`、`w_log[7]`、`max_w_log`、`mse_spread`、`k_eff_log`、`status`、`reason_code`、`ipv_log`、`K`、grid、sigma、provenance。
2. `rq009_gate_join.parquet`：RQ009 每个 feature-matrix row 一行，保留 RQ009 row key，并附 joined InterHub 门字段和 `context_cell_key`。
3. manifest/summary 包：分片 manifest、checksum manifest、status/reason/source/context summary、validator report、overlap report、命令与环境记录。

必须随台账给 RQ009 的警告：`ipv_log == 0` 是合法 OK 点值。J1 报告第 159-169 行给出 `238 / 1,017 = 23.40%` 的门后零点值，并明确只能用 `status` 与 `reason_code` 判别，不能从数值 `0` 反推弃权或缺失。

## 第二阶段对照

### K1b 建议

K1b 建议 P16 / `--mem=64G` / 228 并发片 / 1.02 小时，来源为 K1b 报告第 10-16、71-83、85-99 行。K1b 计划错了：它只优化单片速度和节点总容量，没有按核上限下的每核吞吐重算。P6 的每核吞吐 `0.4781619` 高于 P16 的 `0.4041381`，同一核预算下 P6 更快且总核时更低。

K1b 对内存的原始测量是有用的：P6/P10/P16 每 worker RSS 都约 `2,789 MB`，P16 `--mem=64G` 来自 RSS sum `43.583 GiB * 1.30` 后上取整（K1b 报告第 47-53 行）。问题在资源结论，不在 pilot 数据。

### K2 任务书

任务书的范围部分基本正确：InterHub 做 `4,981,984`，RQ009 `8,994,736` 行 join，OnSite/WOD 本轮不做且不得写成科学 reason（任务书第 55-65 行）。这与第一阶段一致。

任务书第四节部分正确、部分错误：

- 正确：推翻 K1b P16，采用 P6。任务书第 74-81 行的方向与第一阶段一致。
- 计划错了：把 `418` 片、`48G`、`1.15` 小时写成固定资源配置。它的数字来自旧快照中的 `intel 2,384 + fata 130 = 2,514` 空闲核（任务书第 69-77 行），不是本审查现场查询。现场 raw 显示 CPU 分区空闲核合计 `21,926`，QOS 才是我的配置下的约束（集群 raw 第 62、66、68、71-72 行）。
- 计划错了：`--mem=48G` 没有按 K1b 的 30% 规则得到。P6 RSS sum `16,728.84765625 MB`，30% 后为 `21,747.50 MB`；我第一阶段给出 `32G`，K1b 表中按 `ceil_to_8G` 会是 `24G`（K1b 报告第 47-53 行）。`48G` 是 约 3 倍余量，不是线性外推。若监督方坚持 `48G`，应把理由写成保守策略，而不是公式结果。
- 漏项：任务书没有把“是否允许 `amd` 分区”作为待决事项写清楚。若只允许 `fata/intel`，就不能使用我按全 CPU 分区算出的 `665` 并发；若允许 `amd`，就必须加上分区级 canary，避免把 G 轨 `fata02`/`cpui158` 的证据静默泛化到所有 CPU 分区。

任务书第五节的分片、续算、失败分类大体可执行（任务书第 93-117 行），与第一阶段基本一致。需要补强的是：`sum(w_log)`、`max_w_log` 范围、`k_eff_log` 范围、RQ009 final join 行数应进入最终 validator，而不是只在正文范围里出现。

任务书第六节不能真正判定 K2 成功：

1. “与 J1 区间对照”（任务书第 121-124 行）只能作为解释性检查，不能作为成功判据。漏判情形：K2 缺失一批 key 或 RQ009 join 重复，但聚合比例仍落在区间内。误报情形：K2 完整且正确，但由于 InterHub 行级普查分母与 J1 HT 分母不同而落在区间外。
2. “G 锚点重叠逐位一致”（任务书第 125-126 行）是必要 canary，但不充分。漏判情形：重叠行全对，非重叠 PKL 缺片、重复片、reason 顺序错、RQ009 join 错。误报情形较少，但若只比 CSV 字符串而不对齐 canonical key，可能把排序或序列化问题误当数值问题。
3. “worker 数不改变数值”（任务书第 127-128 行）只覆盖并行一致性，不覆盖普查完整性。漏判情形：worker 一致性样本通过，但 K2 产物少 shard、错 join、状态/null 规则错。冗余情形：若 K2 继续用已经测过的 P6，不需要重复证明 P6/P10/P16；若改 worker 或新增分区，才需要重跑。

缺的验收项：`4,981,984` InterHub canonical keys 完整覆盖；`8,994,736` RQ009 rows exact-one join；`held_out_parsed_rows=0`；所有 shard manifest SHA/row count/output SHA 匹配；行级 gate invariant；`ipv_log == 0` 警告已进入接口说明；OnSite/WOD 与非 ATTEMPTED 行不进入两个科学 reason。

## 三个共同问题

### Q1. 第四节资源配置正确吗？

**不正确。计划错了。**

我的第一阶段独立配置是：`665` 片 / 每片 `6` worker / `--mem=32G` / `2,894.18` 核·小时 / 纯 driver 墙钟 `0.7254 h`，操作预算约 `1.0 h`。来源：冻结笔记第 27-39 行；K1b 原始 JSON字段 `driver_elapsed_seconds`、`interhub_rows`、`worker_memory.*.peak_rss_mb`。

监督方错在两步：

1. 把旧的 `intel+fata` 空闲核快照固化为 `418` 片（任务书第 69-77 行）。现场 raw 显示若 CPU 分区均允许，约束是 `cpu-4000...` QOS，而不是实际空闲核（集群 raw 第 23-24、62、66、68、71-72 行）。
2. 把 `48G` 写成资源配置，但它不是 K1b 线性内存规则的结果。P6 RSS sum `16,728.84765625 MB`，30% 后 `21,747.50 MB`；`48G` 是额外保守策略，需要明说，否则后续会误复算。

按我的配置投出去，最坏情况有三类：第一，若 `amd` 分区未获数值通道确认，可能出现分区级数值差异；这必须通过分区 canary 防住。第二，Slurm 不愿一次放 `%665` 小片，墙钟拉长但不应损坏结果。第三，并发 PKL 读取造成 I/O 尾延迟，墙钟超过 `1.0 h`；幂等 manifest 能保证这只是时间损失，不是数据损坏。

### Q2. 第六节三条验收判据能不能真正判定 K2 成功？

**不能。**

J1 区间不是同域同分母，不能判 K2 成功。G 重叠和 worker 一致性是 canary，不是普查验收。第六节缺少真正的普查判据：InterHub `4,981,984` 全覆盖、RQ009 `8,994,736` exact-one join、manifest SHA/row count/output SHA、行级 gate invariant、`held_out_parsed_rows=0`、OnSite/WOD/非 ATTEMPTED 的状态隔离。

多余或会误导的是 J1 CI gate。它可以放在“解释性对照”，不能放在“成功/失败”的验收判据里。

### Q3. 明确判定

判定：**需修改后执行**。

理由：范围、门规格、分片幂等、零点值警告总体正确；但第四节资源值不是我现场独立计算的当前最优值，且没有决策 `amd` 分区是否允许；第六节把 J1 区间放进验收，会导致正确普查被误判或错误产物被漏判。

最可能造成实际损失的一处：**第六节第 1 条把 J1 设计权重区间当作 K2 成功判据**。这会把两个不同分母、不同域的数字混在一起，既可能让正确 K2 被要求解释成问题，也可能让缺片或 join 错误的 K2 因聚合比例“看起来在区间内”而通过。

## 修改清单

执行前至少改三处：

1. 第四节资源改成“投递前按允许分区重查并发”。若允许全 CPU 分区，采用 `665 / P6 / --mem=32G / 2,894.18 core-h / 约 1.0h`，并加分区 canary；若只允许 `fata/intel`，删除全 CPU 解释并用实时 `fata+intel` 空闲核重算。
2. 第六节删除“落在 J1 CI 内/外”作为验收；改为解释性对照。
3. 第六节加入普查验收：`4,981,984` InterHub key 覆盖、`8,994,736` RQ009 exact-one join、manifest SHA/row count/output SHA、行级 gate invariant、`held_out_parsed_rows=0`。
