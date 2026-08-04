# RQ017-R2A A 号独立复审报告

## 收尾问答

**Q1.** 本轮应当复用哪些现成组件与先例？逐个给出文件路径与行号（HPC 上的给绝对路径），并说明它在链路中承担什么。

应复用这些组件与先例：

| 组件/先例 | 文件与行号 | 作用 |
|---|---|---|
| 七候选网格与 `sigma=0.1` | `src/sociality_estimation/core/agent.py:63-64`, `src/sociality_estimation/core/agent.py:95`, `src/sociality_estimation/core/agent.py:156-161` | 冻结候选网格为 `[-3,-2,-1,0,1,2,3] * pi/8`；`exact`/`fast` 默认七候选，`realtime` 才是五候选，故本轮必须显式阻断五候选路径。 |
| 当前求解包装 | `src/sociality_estimation/core/ipv_estimation.py:37-49`, `src/sociality_estimation/core/ipv_estimation.py:181-194`, `src/sociality_estimation/core/ipv_estimation.py:271-284` | `MotionSequence` 承载 `[x,y,vx,vy,heading]`、目标标签和参考线；`estimate_ipv_pair` 调用 protected estimator。注意代码里 `history_window=10` 时切片为 `t-10 : t+1`，即最多 11 行含当前行。 |
| log 域权重 | `src/sociality_estimation/core/reliability_logdomain.py:172-188` | 从七候选 MSE 与 `sigma=0.1` 计算稳定 softmax 权重；门判据必须调用这一实现或逐位等价实现。 |
| K2 冻结门判据 | `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:53-59`, `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:649-689` | `mse_spread == 0.0` 先判 `NO_IPV_EFFECT`，再以 `max_w_log < 0.20` 判 `NEAR_UNIFORM`，否则 `OK` 并写 `ipv_log`。 |
| K2 L1 schema 形状 | `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:776-817`, `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:820-847` | 输出 parquet 的字段顺序、七候选数组展开、null scalar 写法与原子写出先例。 |
| K2 并发 cache 修复 | `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:161-166`, `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:175-184` | 真实修复是每个 Slurm task/PID 设置独立 `MPLCONFIGDIR` 与 `XDG_CACHE_HOME`，不是任务书里一句注释。新 OnSite 入口必须迁移同等机制。 |
| K2 Slurm wrapper | `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/submit_k2_solve_array.sbatch:1-30` | 已验证的作业名、分区、CPU、内存、时间、array 输出路径、`PYTHONPATH`、`cd repo_stage` 与 `--workers 6` 先例。 |
| K2 staging/pydeps | `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/stage_and_submit_k2_fullcorpus.sh:39-69` | 远端 run dir 必须 `test ! -e`，并在冻结 env 缺 `pyarrow` 时安装到 run-dir `pydeps`；sbatch 必须把 `pydeps` 加入 `PYTHONPATH`。 |
| K2 失败先例 | `.codex-fleet/rq015k-fullcorpus-gate/board/reports/K2_fullcorpus_gate_ledger.md:166-181`, `.codex-fleet/rq017-review2-a/work/rq017_r2a_readonly_audit.json:85-205` | 三类真实落地失败：font-cache 并发锁、PyArrow null array 写出、逐行 SHA 过慢；canary 必须覆盖这些路径。 |
| G/K2 HPC 同源锚点 | `.codex-fleet/rq015g-hpc-resolve/work/anchor_mse_hpc.csv:1-5`, `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/validation/g_anchor_hpc_baseline.json:1-7` | 2,300 个 HPC 版锚点基线；本轮应在同一有效环境下复算小批锚点，要求七候选 MSE 逐位相同。 |
| G2 跨节点同源先例 | `.codex-fleet/rq015g-hpc-resolve/board/reports/G2_crossnode_gate.md:39-60`, `.codex-fleet/rq015g-hpc-resolve/board/reports/G2_crossnode_gate.md:63-79`, `.codex-fleet/rq015g-hpc-resolve/board/reports/G2_crossnode_gate.md:137-140` | 给出同 checkout、同 env、同 profile、同线程环境、跨 `cpui158`/`fata02` 逐位一致的验收形状。HPC 绝对路径包括 `/share/home/u25310231/ZXC/sociality_estimation/code/repo` 与 `/share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/bin/python`。 |
| OnSite fallback 参考线与窗口先例 | `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/hpc_onsite_ipv/build_onsite_m3_anchors_hpc.py:705-710`, 同文件 `:739-775`, `:836-847`, `:930-997` | 已有 OnSite 观测轨迹 fallback、按行位置取 anchor 与窗口、角色方向、target window end 与 `history_row_count` 写出合同。 |
| RQ016C product key 与机制二 dry-run | `.codex-fleet/rq016c-human-only-envelope/work/H2/run_rq016c_h2_human_only_envelope.py:136-144`, 同文件 `:419-424`, `.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun_summary.json:4-8`, `.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun_summary.json:74-108` | `product_row_key` 构造与 dry-run 连接源。机制二交叉应连接既有 dry-run 的 `mechanism2_gate_ok`，不要重新给 anchor 表打分。 |
| Tongji HPC 共享规范 | `../HPC_TONGJI_USAGE_GUIDE.md:14-23`, `../HPC_TONGJI_USAGE_GUIDE.md:83-101`, `../HPC_TONGJI_USAGE_GUIDE.md:166-177`, `../HPC_TONGJI_USAGE_GUIDE.md:217-223`, `../HPC_TONGJI_USAGE_GUIDE.md:243-253` | HPC 根、作业名前缀、run 目录、提交前检查、sigma01 兼容路径与冻结参数。绝对工作根是 `/share/home/u25310231/ZXC`；本轮 proposed run dir 是 `/share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ017/<run_id>/`。 |

**Q2.** 把 venue 从本机改到 HPC 之后，**新增**了哪些会静默失败的风险？逐条说明错了会产生什么结果，以及为什么现有检查抓不到。

1. 有效 Python 栈不是任务书以为的栈。只读探针显示 `/share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/bin/python` 当前 Python 3.9.24、numpy 1.21.6、scipy 1.7.3，但直接 import `pyarrow` 失败，来源 `.codex-fleet/rq017-review2-a/work/hpc_env_probe_20260804T064654Z.txt:1-5`。K2 依赖 run-dir `pydeps` 注入，来源 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/stage_and_submit_k2_fullcorpus.sh:65-69` 与 sbatch `PYTHONPATH` `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/submit_k2_solve_array.sbatch:24-25`。v3 只写了 `pydeps/` 目录和记录版本，没有要求安装/注入/断言实际 import origin；结果可能是直接失败，也可能导入别处 pyarrow，现有“记录版本”不会抓住路径错误。
2. 新入口可能导入了错的代码。K2 的 sbatch 明确 `cd repo_stage` 并设置 `PYTHONPATH="${SCRIPT_DIR}/pydeps:${PWD}/src:..."`，来源 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/submit_k2_solve_array.sbatch:20-30`。v3 模板只有 `PYTHON=...`，来源 `.codex-fleet/rq017-onsite-materializer/board/RQ017_M1_kickoff_v3.md:197-215`，未要求打印 `module.__file__` 或保护文件 SHA blocker。结果是看起来在 HPC 上跑，实际用错 checkout 或旧兼容路径；门判据复算仍可能自洽。
3. 并发 cache 风险没有被实际实现。v3 在 `.codex-fleet/rq017-onsite-materializer/board/RQ017_M1_kickoff_v3.md:209-214` 写了线程环境和 font-cache 注释，但没有给出 `MPLCONFIGDIR`/`XDG_CACHE_HOME` 赋值。K2 真正实现见 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:175-184`。如果复发，array task 会失败或互相污染 cache；若 canary 没覆盖真实导入/写出路径，现有数值检查抓不到。
4. 作业实际分区可能被提交命令覆盖。v3 固定模板写 `intel,fata`，但没有要求事后用 `sacct` assert 所有主作业与子任务 partition 不含 `amd`。K2 报告把所有 job 分区作为证据列出，来源 `.codex-fleet/rq015k-fullcorpus-gate/board/reports/K2_fullcorpus_gate_ledger.md:181`。若有人用命令行覆盖分区，数值可自洽但与 K2 来源不再可比。
5. 远端 run 目录碰撞或相对路径错误。K2 staging 用 `test ! -e '${HPC_WORKDIR}'` 再创建，来源 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/stage_and_submit_k2_fullcorpus.sh:39-40`；v3 只描述 run_id 与目录结构，来源 `.codex-fleet/rq017-onsite-materializer/board/RQ017_M1_kickoff_v3.md:169-182`。若 timestamp 重复或提交目录错，可能复用旧 inputs/outputs；最终行数与门判据仍可通过。
6. 机制二交叉若调用 `score_external_rows.py` 会读整张输入表。该脚本 `pd.read_parquet(path)` 读全列并把 scored 输出写出，来源 `.codex-fleet/rq016c-human-only-envelope/work/H2/score_external_rows.py:22-28`, `:51-53`；anchor schema 中有 `target_ipv_future` 与 `target_ipv_error_future`，来源 `.codex-fleet/rq017-review2-a/work/rq017_r2a_readonly_audit.json:338-339`。v3 第 8 节要求“与 RQ016C 支持门交叉”，来源 `.codex-fleet/rq017-onsite-materializer/board/RQ017_M1_kickoff_v3.md:317-318`，但未明确只能连接 dry-run 的 `product_row_key, mechanism2_gate_ok`。结果可能违反输入列白名单，既有机制一数值检查抓不到。

**Q3.** 「与既有 InterHub 台账处于同一软件栈」这件事，怎样才算被证明？给出你认为充分的验收做法，并指出仅仅"也在 HPC 上跑"为什么不够。

充分证明必须同时满足四层：

1. 有效导入层：记录 `sys.executable`、Python/numpy/scipy/pyarrow 版本、`pyarrow.__file__`、`sociality_estimation.__file__`、`agent.__file__`、`ipv_estimation.__file__`、`reliability_logdomain.__file__`，并断言都来自本轮 `repo_stage/src` 或 run-dir `pydeps`。当前远端 env 直接没有 pyarrow，来源 `.codex-fleet/rq017-review2-a/work/hpc_env_probe_20260804T064654Z.txt:1-5`，所以只写 env 路径不充分。
2. 代码与配置哈希层：把 protected files SHA 与 K2/G 清单对齐。G manifest 已记录 `agent.py`、`ipv_estimation.py`、`reliability_logdomain.py`、`process_interhub.py`、`configs/ipv_sigma01_exact.json` 的 SHA match，来源 `.codex-fleet/rq015g-hpc-resolve/work/hpc_environment_manifest.json:9-33`；K2 报告也列出同组 SHA，来源 `.codex-fleet/rq015k-fullcorpus-gate/board/reports/K2_fullcorpus_gate_ledger.md:212-220`。
3. 正面对照层：在本轮 run dir 内复算 G-HPC anchor 小批，与 `.codex-fleet/rq015g-hpc-resolve/work/anchor_mse_hpc.csv` 对齐，要求七候选 MSE 逐位相同；K2 修正后的完整对照是 `anchor_rows=2300, compared_rows=2300, max_abs_diff=0.0`，来源 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/validation/g_anchor_hpc_baseline.json:1-7`。
4. Slurm/节点层：记录实际 partition/node/thread env，并确认不含 `amd`。G2 已给出同 checkout、同 env、同 profile、同线程环境，跨节点 `max|Δ|=0.0`，来源 `.codex-fleet/rq015g-hpc-resolve/board/reports/G2_crossnode_gate.md:39-60`, `:63-79`, `:137-140`。

仅仅“也在 HPC 上跑”不够，因为 HPC 上至少有多个变量会改变结果或写法：不同 checkout、不同 `PYTHONPATH`、run-dir `pydeps` 有无 pyarrow、不同 BLAS/线程环境、以及被禁止的分区。v3 自己也承认 Mac 与 HPC 差异很大，来源 `.codex-fleet/rq017-onsite-materializer/board/RQ017_M1_kickoff_v3.md:8-13`；同理，HPC 内部如果有效栈不同，也不能视为同源。

**Q4.** 作业形状（分片数、并发上限、`cpus-per-task`、`mem`、`time`）应当怎么定？给出你的算法与依据。

算法如下：

1. 先用 canary 实测本轮 OnSite 入口的每行耗时和 task 峰值 RSS；canary 必须 sbatch、至少两个 array task、含工程失败写出、含 7 行坐标异常和参考线 fail-closed 合成行，依据 v3 `.codex-fleet/rq017-onsite-materializer/board/RQ017_M1_kickoff_v3.md:283-300`。
2. 初始 `cpus-per-task=6, mem=48G, time=04:00:00`，因为 K2 成功 solve array 用同一资源，来源 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/submit_k2_solve_array.sbatch:1-10`，且 K2 Slurm 证据显示成功 job `2070433` 完成 460 tasks，来源 `.codex-fleet/rq017-review2-a/work/rq017_r2a_readonly_audit.json:140-162`。
3. 分片数按 `ceil(67861 / rows_per_shard)` 定。`rows_per_shard` 用 canary 的 p95 每行耗时算：`floor(0.70 * time_limit_seconds / p95_seconds_per_row_per_task)`，再受内存安全约束 `peak_rss <= 0.70 * mem`。若 canary 不差于 K2，可沿 K2 的 shard 上限 11,000 行，K2 rollup `a_shard_max_rows=11000`，来源 `.codex-fleet/rq017-review2-a/work/rq017_r2a_readonly_audit.json:14-16`；本轮 67,861 行则是 7 个分片，前 6 个约 11,000 行、末片约 1,861 行。这里不写百分数，避免把行数比误当成资源口径。
4. 并发上限按逐节点装箱，不按分区空闲核总数直接除以 6。对每个可用 `intel/fata` 节点计算 `min(floor(idle_cpus/6), floor(free_mem_mb/49152))`，再求和并取 `min(N, slot_sum, policy_cap)`。K2 的 per-node 装箱 summary 是 `workers=6, mem_mb_per_shard=49152, slots_sum=427, concurrency=427`，来源 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/cluster_snapshot_calculation.json:107-119`。当前只读 `sinfo` 快照显示 `intel` 有 6 个 idle 节点、`fata` 无 idle 节点但有 3 个 mix 节点，来源 `.codex-fleet/rq017-review2-a/work/hpc_sinfo_20260804T064654Z.txt:1-14`；本轮若 N=7，并发可先取 7，但仍要用提交前 per-node 快照重新计算。
5. 若 canary 显示 OnSite 入口明显更慢或 RSS 更高，只调 `rows_per_shard`、`mem` 或 `time`，不改 `sigma`、候选网格、分区，也不改投 `amd`。

**Q5.** 有哪些验收判据**必须存在**，否则错误会静默通过？逐条列出，并说明缺了它会漏掉什么。

必须存在以下脚本 blocker：

1. `product_row_key` 三方一对一：anchor 构造键、输出键、RQ016C dry-run 键均唯一且交集为 67,861。缺失会漏掉错行、错车、错时刻；我实测 anchor/dry-run 交集 67,861/67,861，筛选为两表全部行，来源 `.codex-fleet/rq017-review2-a/work/rq017_r2a_readonly_audit.json:208-213`, `:246-253`，列为 `case_key/anchor_frame_index/perspective/source_dataset/product_row_key`。
2. 行位置窗口等价断言：`valid_anchor_positions()` 用 `pos` 和 `wx_start=max(0,pos-10+1)`，来源 `build_onsite_m3_anchors_hpc.py:836-847`；但求解包装里 `history_window=10` 实际切 `t-10:t+1`，来源 `src/sociality_estimation/core/ipv_estimation.py:271-284`。缺失会产生 off-by-one，MSE、权重、恒等式都仍自洽。
3. 输入列白名单和实际读取列日志：anchor 表存在 `target_ipv_future`、`target_ipv_error_future`，来源 `.codex-fleet/rq017-review2-a/work/rq017_r2a_readonly_audit.json:338-339`；我核数只读列见 `.codex-fleet/rq017-review2-a/work/rq017_r2a_readonly_audit.json:214-228`, `:249-252`。缺失会漏掉目标值或 outcome 字段被新入口、交叉脚本误读。
4. 有效导入路径、protected SHA 与环境同源 blocker：缺失会漏掉错 checkout、错 pydeps、错 pyarrow。当前远端 env 无 pyarrow，来源 `.codex-fleet/rq017-review2-a/work/hpc_env_probe_20260804T064654Z.txt:1-5`。
5. 门判据复算与负对照：从 `mse_0..mse_6` 重推 `status/reason_code/ipv_log`；注入 `mse_spread==0.0` 和 `theta=0.20/0.22` sentinel，确保检查会 FAIL。缺失会漏掉 `np.isclose` 替换精确相等、阈值漂移或 reason 顺序反转；K2 规格见 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:649-689`。
6. 工程失败隔离：工程失败不得写成 `NEAR_UNIFORM` 或 `NO_IPV_EFFECT`；缺失会把求解/输入失败混入科学 reason。K2 失败行写法见 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:720-758`。
7. 写出后读回 schema、null scalar 和 SHA：缺失会漏掉 PyArrow null 写出问题。K2 原子写 parquet 与 manifest hash 逻辑见 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:840-847`, `:859-876`。
8. Slurm 事后证据：`sacct`/summary 必须断言 job name `zxc-`、partition 只含 `intel/fata`、array N/M 与 task 数一致、无失败 task。缺失会漏掉提交参数被命令行覆盖或部分 task 失败。
9. no-overwrite：远端 run dir 与本地 `data/derived/rq017_onsite_gate/l1_v1/` 若存在即停止。缺失会把旧输出当新输出；K2 staging 先例是 `test ! -e`，来源 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/stage_and_submit_k2_fullcorpus.sh:39-40`。
10. 机制二交叉只连接 dry-run：用 `product_row_key, mechanism2_gate_ok` 计算 `OK ∩ support_gate_ok`。缺失会诱导执行方重新打分并读整表，见 `score_external_rows.py:51-53`。

**Q6.** 这一轮有没有会污染下游、或不可逆的风险？逐条列出并给出规避办法。

有，但都可规避：

1. 错行/错时刻产物污染下游在线验证。规避：C1 三方键一对一、输出写 `solve_frame_index/anchor_frame_index/target_window_end_frame_index/history_window_used`，并用抽样行人工重建窗口。
2. 目标值或评分字段被读入。规避：新入口用显式 `columns=[...]` 读 parquet，记录实际读取列集合；机制二交叉只读 dry-run `product_row_key, mechanism2_gate_ok`。
3. 旧 285 行通道混入。规避：输入路径和 SHA 只允许 67,861 行 allvalid anchor 与 70,317 行 timeseries；我实测两个文件 SHA 分别为 `98f27564407daa2b3e9984d64b43282c764dad771e2ee0d2f49bc4869052acc5` 与 `86bced1a2e7d3e9a1c1378e55b76043703fede6c00d459c00ae10db45e4ed864`，来源 `.codex-fleet/rq017-review2-a/work/rq017_r2a_readonly_audit.json:346-350`, `:415-419`。
4. 放松 RQ007 护栏。规避：不要改 `.codex-fleet/rq015b-repair/work/run_b2_rq015b.py:71-72` 与 `:268-273`；OnSite 走独立入口，并断言参与求解的 InterHub 行数为 0。
5. 写入错误 HPC 目录或旧退役目录。规避：所有远端写入限制在 `/share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ017/<run_id>/`；禁止 `/share/home/u25310231/ZXC/ipv_estimation`，该路径退役见 `../HPC_TONGJI_USAGE_GUIDE.md:236-241`。
6. 使用 `amd` 导致产物不可比。规避：提交前脚本 grep partition，提交后 `sacct` assert 全 task partition 不含 `amd`。
7. 坐标异常被静默丢弃。规避：7 行 `relative_distance_anchor > 500000` 必须出现在输出并单列状态；我实测 7/67,861，筛选为 anchor 全部行、列 `relative_distance_anchor`，来源 `.codex-fleet/rq017-review2-a/work/rq017_r2a_readonly_audit.json:241-245`。
8. 本地新产物覆盖。规避：若 `data/derived/rq017_onsite_gate/l1_v1/` 或远端 run dir 已存在内容，停止并报告，不覆盖。

**Q7.** 若你只能提**一条**改动意见，是哪一条？为什么是它而不是别的。

把 HPC staging 与 sbatch 从“文字模板”改成可执行、可审计、会失败的 wrapper 合同：必须包含 `set -euo pipefail`、`cd "${SCRIPT_DIR}/repo_stage"`、`PYTHONPATH="${SCRIPT_DIR}/pydeps:${PWD}/src:..."`、run-dir `pyarrow` 安装/版本断言、`MPLCONFIGDIR`/`XDG_CACHE_HOME` 隔离、有效导入路径与 protected SHA blocker、`sacct` 分区 blocker。理由是 venue 改 HPC 是 v3 相比 v2 的核心变化；当前远端冻结 env 直接没有 pyarrow，来源 `.codex-fleet/rq017-review2-a/work/hpc_env_probe_20260804T064654Z.txt:1-5`，而 v3 的 sbatch 片段没有 `PYTHONPATH`，来源 `.codex-fleet/rq017-onsite-materializer/board/RQ017_M1_kickoff_v3.md:197-215`。如果这条不修，后面的科学验收可能根本跑不到，或跑在看起来相同但实际不同的栈上。

**Q8.** 这一轮该不该开跑？只答 `GO` / `GO_WITH_CHANGES` / `NO_GO` 三者之一，并给出不超过三句话的理由。若答 `GO_WITH_CHANGES`，把必须先改的条目按重要性排序列出。

`GO_WITH_CHANGES`。v3 的科学范围、复用求解组件和 K2 门判据方向正确，但 HPC wrapper 与同源性证明还不足以保证照做可复现 K2 软件栈。必须先改：

1. 补成可执行 fail-closed HPC wrapper：pydeps/pyarrow、`PYTHONPATH`、cache 隔离、import origin、protected SHA、`sacct` 无 `amd`。
2. 把 OnSite 新入口的 C1 键、行位置窗口、角色方向、输入列白名单和冻结 profile 对齐做成脚本 blocker。
3. 明确机制二交叉只连接 RQ016C dry-run 的 `product_row_key, mechanism2_gate_ok`，不得调用会读整表的重新打分入口。
4. canary 必须覆盖并发、工程失败写出读回、7 行坐标异常、参考线 fail-closed 和两个负对照 sentinel。

## 逐条审查详细依据

本次复审对象是 `.codex-fleet/rq017-onsite-materializer/board/RQ017_M1_kickoff_v3.md`。这项工作要补齐 OnSite 自动驾驶车 anchor 行的机制一判据：机制一用七候选 MSE 与 log 域权重判断当前 IPV 数值是否携带候选间判别信息；机制二的人类参照 envelope 已另行完成。本次只审任务书照做是否会出错、错误能否暴露；未运行求解器，未投 Slurm，未训练模型，未写 `data/derived/`。

结论：`GO_WITH_CHANGES`。v3 的范围、冻结配置、复用核心求解与门判据方向基本正确，但 HPC 执行模板与同源性验收还不够硬；按原文开跑会在 `pyarrow`/`PYTHONPATH`/并发 cache、实际导入代码来源、输入列白名单和机制二交叉方式上留下可静默通过或执行期失败的缺口。

### 1. 求解链路正确性

结论：核心复用对象指认基本正确，但 v3 对窗口参数的文字必须更精确。

- `MotionSequence` 和 `estimate_ipv_pair` 是合适入口，见 `src/sociality_estimation/core/ipv_estimation.py:37-49`, `:181-194`。
- `history_window=10` 在代码里不是“只取 10 行历史再加当前行”的参数名语义，而是 `start=max(0,t-history_window)` 且切片 `start:t+1`，见 `src/sociality_estimation/core/ipv_estimation.py:271-284`。OnSite anchor 构造的 `history_row_count` 是 `wx_start=max(0,pos-10+1)` 到当前行，见 `build_onsite_m3_anchors_hpc.py:938-940` 与写出 `history_row_count` 的 `:997`。这两个概念如果不在 preflight 中逐行比对，会出现 off-by-one。
- 门判据复用 K2 的 `gate_from_mse()` 正确，见 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:649-689`；不应整体复用 K2 `validate_outputs()`，因为它硬编码 InterHub/RQ009/全 L1 总数，见同文件 `:1327-1338`。
- 不应复用 `run_b2_rq015b.py solve_anchor_task()`，因为 `ALLOWED_SPLITS={"development","guard"}` 与 split 检查在 `.codex-fleet/rq015b-repair/work/run_b2_rq015b.py:71-72`, `:268-273`。

### 2. HPC 落地

结论：v3 方向正确，但模板不够可执行。

- 共享指南要求 `/share/home/u25310231/ZXC` 下工作、重计算走 sbatch、job name 以 `zxc-` 开头，见 `../HPC_TONGJI_USAGE_GUIDE.md:14-23`。
- v3 run dir 在 `/share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ017/<run_id>/`，来源 `.codex-fleet/rq017-onsite-materializer/board/RQ017_M1_kickoff_v3.md:169-182`，符合共享根。
- 但 v3 sbatch 片段缺少 K2 实际有的 `set -euo pipefail`、`SCRIPT_DIR`、`cd repo_stage`、`PYTHONPATH`，对照 K2 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/submit_k2_solve_array.sbatch:13-30`。
- 当前 `sinfo` 只读快照显示 `intel` 有 6 个 idle 节点、183 个 down 节点，`fata` 13 个 drain、3 个 mix，来源 `.codex-fleet/rq017-review2-a/work/hpc_sinfo_20260804T064654Z.txt:1-14`；排队正常，但不能因此改投 `amd`。

### 3. 同源性

结论：v3 说“必须正面证明”是对的，但验收应增加有效导入路径与 pydeps 断言。

- K2 最终用 G-HPC baseline 修正了旧 Mac baseline 错比，`max_abs_diff=0.0`，见 `.codex-fleet/rq015k-fullcorpus-gate/board/reports/K2_fullcorpus_gate_ledger.md:104-120`。
- G2 的严格同源证据包括 checkout、env、profile、线程环境与跨节点逐位一致，见 `.codex-fleet/rq015g-hpc-resolve/board/reports/G2_crossnode_gate.md:39-60`, `:63-79`, `:137-140`。
- 当前冻结 env 直接没有 pyarrow，来源 `.codex-fleet/rq017-review2-a/work/hpc_env_probe_20260804T064654Z.txt:1-5`；因此本轮必须记录 run-dir `pydeps` 实际 import 版本和 `__file__`，否则“同一 env 路径”不足。

### 4. 口径一致性

结论：冻结值核对一致，但新入口必须 fail-closed。

- `configs/ipv_sigma01_exact.json:4-16` 固定 `solver_mode=exact`、`sigma=0.1`、`min_observation=4`、参考线参数 `60/40/40`、窗口 `10/4/6`。
- `process_interhub.py` 现有 profile 校验会比对命令参数与 config，并比对 `agent_module.sigma`，冲突即抛错，见 `pipelines/interhub/process_interhub.py:2038-2078`。RQ017 新入口需要同等级校验。
- 七候选网格从 `agent.py` 与 K2 常量两处一致，见 `src/sociality_estimation/core/agent.py:63-64`, `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:53-59`。

### 5. 安全边界

结论：边界列得对，但至少两条需要脚本化。

- RQ007 护栏不得改，代码位置是 `.codex-fleet/rq015b-repair/work/run_b2_rq015b.py:71-72`, `:268-273`。
- 输入列白名单必须脚本化，因为 anchor 表实际含目标列，来源 `.codex-fleet/rq017-review2-a/work/rq017_r2a_readonly_audit.json:338-339`。
- OnSite timeseries 源表没有任何 map/lane/route/reference-like 列，来源 `.codex-fleet/rq017-review2-a/work/rq017_r2a_readonly_audit.json:272-274`；因此 fallback 后果必须靠参考线点数 fail-closed 和 canary 暴露。

### 6. 验收判据逐条有效性

| v3 自查项 | 判定 | 依据与缺口 |
|---|---|---|
| 行数守恒与 C1 三方交集 | 有效，必须保留 | 我实测 anchor 67,861 行、dry-run 67,861 行、构造键唯一 67,861、交集 67,861，来源 `.codex-fleet/rq017-review2-a/work/rq017_r2a_readonly_audit.json:208-213`, `:246-253`。 |
| 状态守恒 | 有效但需定义工程失败集合 | K2 schema 中 `status/reason_code/failure_type/solver_status` 是分列写法，见 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:776-817`。 |
| 门判据可复算 | 有效 | K2 `gate_from_mse()` 是冻结依据，见 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:649-689`。 |
| 恒等式 + K/grid 断言 | 有效但不充分 | 能抓权重/IPV 算错，抓不到 wrong row/window；必须与 C1/C3/C5/C6 联合。 |
| 工程失败隔离 | 有效 | K2 失败写法为 null metrics + status/reason failure，见 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:720-758`。 |
| 护栏 blocker 与环境同源 | v3 方向对，细节不足 | 需增加 import origin、pydeps、protected SHA、`sacct` partition blocker。 |
| 两条负对照 | 有效，但必须注入 sentinel | v3 自己指出自然输出不保证含敏感行，见 `.codex-fleet/rq017-onsite-materializer/board/RQ017_M1_kickoff_v3.md:331-334`。 |
| 数值健康 | 有效但只能抓数值域 | 抓不到错对象、错时刻、目标列泄漏。 |
| 取回完整性 | 有效 | 需远端/local 分片行数与 SHA，K2 manifest hash 先例见 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:859-876`。 |
| 不覆盖检查 | 有效但需覆盖远端 staging | v3 只写本地与远端产物目录存在则停，K2 远端 `test ! -e` 先例见 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/stage_and_submit_k2_fullcorpus.sh:39-40`。 |

### 7. 失败路径覆盖

canary 列表方向正确，但应补充两个要求：一是 canary 使用与全量完全相同的 sbatch wrapper、`PYTHONPATH`、cache 设置和 parquet writer；二是 canary 把“自然样本”和“合成 sentinel”分文件输出，正式产物 manifest 明确排除 sentinel。K2 的教训是 one-row canary 没覆盖并发和失败写出，来源 `.codex-fleet/rq015k-fullcorpus-gate/board/reports/K2_fullcorpus_gate_ledger.md:204-208`。

### 8. 分母与计数抽查

我用 `.codex-fleet/rq017-review2-a/work/rq017_r2a_readonly_audit.py` 只读生成 `.codex-fleet/rq017-review2-a/work/rq017_r2a_readonly_audit.json`。抽查结果如下：

| 数字 | 我的实测 | 来源 |
|---|---:|---|
| OnSite anchor 表行列 | 67,861 行、66 列 | `.codex-fleet/rq017-review2-a/work/rq017_r2a_readonly_audit.json:346-350`；源文件 `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet`。 |
| C1 dry-run 连接 | anchor 构造键唯一 67,861；dry-run 键唯一 67,861；交集 67,861 | `.codex-fleet/rq017-review2-a/work/rq017_r2a_readonly_audit.json:208-213`, `:246-253`；筛选为两表全部行，列见 `:214-228`, `:249-252`。 |
| OnSite K2 台账缺口 | 281,268 行；`gate_applicable=True` 为 0；`mse_0..mse_6/max_w_log/mse_spread/status/reason_code` 非空均为 0 | `.codex-fleet/rq017-review2-a/work/rq017_r2a_readonly_audit.json:50-63`, `:79-84`；筛选 `artifact_id=onsite_dense_timeseries` 分区，列见 `:64-78`。 |
| 历史短于 10 行 | 1,572/67,861，筛选 anchor 全部行，列 `history_row_count` | `.codex-fleet/rq017-review2-a/work/rq017_r2a_readonly_audit.json:257-271`。 |
| 机制二支持门 dry-run | 21,936/67,861 = 32.3249%，筛选 dry-run 全部行，列 `mechanism2_gate_ok`，源 `.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet` | `.codex-fleet/rq017-review2-a/work/rq017_r2a_readonly_audit.json:266-267`, `:352-373`。 |
| InterHub K2 solve 分母与状态 | gate_applicable true 4,981,984；OK 3,502,340；ABSTAIN 1,477,710；SOLVER_FAILURE 1,934 | `.codex-fleet/rq017-review2-a/work/rq017_r2a_readonly_audit.json:31-49`；筛选 `artifact_id=interhub_sigma01_hw4_timeseries` 分区，列见 `:32-36`。 |
| 7 行坐标异常 | 7/67,861 = 0.0103%，筛选 anchor 全部行、`relative_distance_anchor > 500000`，列 `relative_distance_anchor/relative_dx_anchor/relative_dy_anchor` | `.codex-fleet/rq017-review2-a/work/rq017_r2a_readonly_audit.json:241-245`。 |

### 9. 遗漏

1. 缺少完整 HPC wrapper 作为任务书交付物。v3 给模板，但没有要求提交前先生成并静态检查 wrapper。
2. 缺少 import-origin 审计：只比版本不比 `__file__` 会漏掉错 checkout 或错 pydeps。
3. 缺少机制二交叉的安全做法：应只连接 dry-run 两列，不重新打分。
4. 缺少 per-node 装箱脚本输出。v3 要现算，但没有要求保存 node-level slots 计算文件；K2 有 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/cluster_snapshot_calculation.json:107-119`。
5. 缺少 `sacct` 事后 FAIL 规则：实际分区、失败 task、array task 总数、elapsed、NodeList 应作为 blocker。

state: WAITING_ON_COMMANDER
timestamp_utc: 2026-08-04T06:49:46Z
