# RQ017-R2B review

reviewer: B
phase1_closed_utc: 2026-08-04T06:47:40Z
phase1_blinding_note: 阶段一完成前未打开 `.codex-fleet/rq017-onsite-materializer/**`；未读取 A 号目录、上一轮复审目录、`START_HERE.md`、`STUDIES.md`、`main_workflow.log`、`reports/knowledge/**/README.md` 或 `commander_notes.md`。

## 阶段一：独立推导

### 1. 位置与目标

这项工作要补的是在线验证的第一道弃权机制：对 OnSite 的 AV anchor frame 重新跑冻结 IPV 求解，产出七个候选 IPV 的 MSE、log-domain 权重和 `status`/`reason_code`，使后续人类参照 envelope 只接收携带候选区分信息的帧。整体流程已经有 InterHub 的冻结求解、K2 L1 台账和 OnSite envelope；本次是把 OnSite scope B 的 67,861 个 timing-valid anchor frame 接到同一第一道机制。

PI 已定三件事我不再论证：范围是 `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet` 的 67,861 行；OnSite 参考线合同是观测轨迹 fallback；运行地点是同济 HPC，Slurm 分区只用 `intel,fata`，不用 `amd`。

### 2. InterHub 求解链路现在长什么样

现有 InterHub 入口是 `pipelines/interhub/process_interhub.py`。默认 profile 指向 `configs/ipv_sigma01_exact.json`，硬编码窗口与参考线参数是 `HISTORY_WINDOW=10`、`MIN_OBSERVATION=4`、`REFERENCE_CLIP_MARGIN_M=60.0`、`REFERENCE_MAX_POINTS=40`、`REFERENCE_SMOOTH_POINTS=40`，见 `process_interhub.py:45-55`。CSV 输出列只包含均值、error、状态、case dir、pkl、segment 和 reference source，不包含七候选 MSE 或 log-domain 权重，见 `process_interhub.py:58-70`。

InterHub 数据定位链路如下：

- CSV key 是 `folder/scenario_idx/key_agents/track_id`，`parse_key_agents()` 要求两个 key agent，见 `process_interhub.py:236-240`；`csv_key()` 生成四元 key，见 `process_interhub.py:252-258`。
- pkl event index 由 `build_event_index()` 扫描 pkl 并按 metadata 或 CSV scene id 对齐，重复 key 直接报错，见 `process_interhub.py:284-310`。
- pkl event 通过 `load_event()` 读取并规范化为 `metadata/vehicles/road_info`，见 `process_interhub.py:332-372`。
- 参考线优先来自 lane/frame/reference/current/possible lane ids；没有可用 lane centerline 时回退到 observed positions，回退值名是 `observed_trajectory_fallback`，见 `process_interhub.py:375-410`。
- 运动输入必须能形成 `[x, y, vx, vy, heading]`；非有限行会被 `_motion_arrays()` 过滤，见 `process_interhub.py:485-513`。
- 两车运动按共同 timestamp 对齐，少于 `MIN_OBSERVATION + 1` 行报错，见 `process_interhub.py:529-553`；nuPlan 会按 `DATASET_DOWNSAMPLE_FACTORS={"nuplan_train": 2}` 降采样，见 `process_interhub.py:556-580`。
- `process_case()` 装载 event、对齐、构造 reference、分类 heading、生成两个 `MotionSequence`，再调用 `estimate_ipv_pair()`，见 `process_interhub.py:1049-1175`。成功后只写 Excel、plot 和 metadata，见 `process_interhub.py:1182-1238`。

求解核心在 `src/sociality_estimation/core/ipv_estimation.py` 和 `src/sociality_estimation/core/agent.py`：

- `MotionSequence` 的数据接口是二维数组，列至少为 `[x, y, vx, vy, heading]`，另带 target 和可选 reference，见 `ipv_estimation.py:36-55`。
- `estimate_ipv_pair()` 的默认参数是 `history_window=10`、`min_observation=4`、`solver_mode="exact"`；文档说明 exact/fast 都使用七候选，realtime 使用五候选，见 `ipv_estimation.py:181-241`。
- 每个 t 从 `start=max(0,t-history_window)` 截窗口；分别从 primary 和 counterpart 视角创建 `Agent`，调用 `_estimate_agent_ipv()`，再把当步 IPV 和 error 写入数组，见 `ipv_estimation.py:247-338`。
- diagnostics 会保存 observed、interacting、virtual_tracks、weights、ipv_range、ipv、ipv_error，见 `ipv_estimation.py:340-371`。这正是取七候选 MSE 所需的中间物。
- 当前帧接口 `estimate_ipv_current()` 只是把 `min_observation` 设成最后一步并调用 `estimate_ipv_pair()`，见 `ipv_estimation.py:375-427`。
- 七候选网格是 `[-3,-2,-1,0,1,2,3] * pi/8`，见 `agent.py:61-64`；exact 模式走 legacy loop cost backend，见 `agent.py:164-170`。
- `Agent.estimate_self_ipv()` 对每个候选构造任务、求 virtual track，再调用 `_apply_candidate_ipv_tracks()` 算权重，见 `agent.py:769-832`；候选任务含 position、velocity、heading、target、reference、candidate IPV、inter_track、solver options、solver mode，见 `agent.py:209-231`。
- legacy 权重是 `cal_traj_reliability()`：对每个候选计算 virtual track 与 observed track 的逐步距离，使用 `sigma=0.1` 的 Gaussian 密度乘积，若所有候选 var 和为 0 则返回均匀权重，见 `agent.py:1097-1141`。

七候选 MSE 与第一道机制的冻结实现分两层：

- 数学合同在 `src/sociality_estimation/core/reliability_logdomain.py`：legacy 连乘等价于 `softmax(-MSE_i/(2*sigma^2))`，见 `reliability_logdomain.py:14-18`；`candidate_mse()` 和 `weights_from_mse()` 分别定义每候选 MSE 和稳定 softmax，见 `reliability_logdomain.py:167-188`。
- 该模块文件头同时声明它“未被任何生产路径导入”，见 `reliability_logdomain.py:1-7`。所以它能作为冻结公式和校验器依据，不能单独证明 InterHub 主入口已经自动产出 K2 L1 台账。
- K2 全语料 materializer 在 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py` 将冻结字段落到 L1：`SCHEMA_VERSION=rq015k_logdomain_gate_l1_v1`、`SIGMA=0.1`、`THETA=0.20`、`K=7`、`GRID_ID=legacy7_pi_over_8`、`IPV_GRID=[-3..3]*pi/8`，见 `k2_fullcorpus_materializer.py:53-69`。
- K2 `gate_from_mse()` 对 MSE 做 log score 和 `weights_from_mse()`；若 `mse_spread == 0.0`，则 `status=ABSTAIN/reason_code=NO_IPV_EFFECT`；否则若 `max_w_log < 0.20`，则 `status=ABSTAIN/reason_code=NEAR_UNIFORM`；否则 `status=OK` 并写 `ipv_log`，见 `k2_fullcorpus_materializer.py:649-689`。
- K2 L1 行把 `candidate_ipv_0..6/mse_0..6/log_score_0..6/w_log_0..6/max_w_log/mse_spread/k_eff_log/status/reason_code/ipv_log` 写入 parquet schema，见 `k2_fullcorpus_materializer.py:692-817`。

因此，我认为 OnSite 版不应改 InterHub `process_case()` 输出；应新写一个 OnSite materializer，复用求解核心和 K2 L1 门控字段，但把 OnSite anchor/timeseries 转为同一个 `MotionSequence`/diagnostics 输入。

### 3. 冻结口径

求解口径：

- 候选网格：`[-3,-2,-1,0,1,2,3] * pi/8`，代码定义见 `agent.py:63`；K2 落库常量见 `k2_fullcorpus_materializer.py:57-59`。
- K：7，K2 常量见 `k2_fullcorpus_materializer.py:56`；OnSite K2 dense ledger 实测 `K` 非空 281,268 行，唯一值 7，来源 `[RQ015A concentration ledger]/concentration_ledger/onsite_dense_timeseries.parquet` 的 `K` 列。
- sigma：0.1，profile 定义见 `configs/ipv_sigma01_exact.json:4-7`；legacy likelihood 全局变量见 `agent.py:94-96`；K2 常量见 `k2_fullcorpus_materializer.py:53-55`。
- history window：10，profile 定义见 `configs/ipv_sigma01_exact.json:12-15`；InterHub 入口常量见 `process_interhub.py:47-48`；`estimate_ipv_pair()` 默认参数见 `ipv_estimation.py:181-187`。
- min observation：4，profile 定义见 `configs/ipv_sigma01_exact.json:4-8`；InterHub 常量见 `process_interhub.py:47-48`；早于该行的 InterHub 输出填充不是有效机制一输入，见 `ipv_estimation.py:211-214`。
- reference 处理：clip margin 60m、max points 40、smooth points 40，profile 定义见 `configs/ipv_sigma01_exact.json:7-10`；InterHub 常量见 `process_interhub.py:49-51`。

门控口径：

- K2 L1 生产字段与 downstream interface 是 `status` 和 `reason_code`，接口说明见 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/INTERFACE_NOTE.md:35-52`。
- K2 生产判定顺序是：非有限/solver failure 先由 `classify_failure()` 和 `l1_row_from_solve()` 转为 `NON_FINITE_INPUT` 或 `SOLVER_FAILURE`，见 `k2_fullcorpus_materializer.py:620-630` 与 `k2_fullcorpus_materializer.py:692-758`；数值可用后先判 `mse_spread==0.0`，再判 `max_w_log<0.20`，否则 OK，见 `k2_fullcorpus_materializer.py:649-689`。
- `ipv_log=0` 不能当作弃权；接口说明明确 discriminator 是 `status` 加 `reason_code`，不是 `ipv_log` 数值，见 `INTERFACE_NOTE.md:54-82`。
- `reliability_logdomain.py` 另有一个结果合同：`STATUS_NON_FINITE_INPUT/SOLVER_FAILURE/MODEL_MISFIT/FLAT_LIKELIHOOD/OK`、`K_EFF_FLAT_RATIO=0.93`，并在 `estimate_reliability()` 先判 `min_mse > min_mse_misfit`，再判 `k_eff >= ratio*K`，见 `reliability_logdomain.py:33-64` 与 `reliability_logdomain.py:215-291`。这与 K2 L1 的 `THETA=0.20/max_w_log` 表达不是同一列名合同。OnSite 若要写入 K2 L1，必须遵循 K2 L1 字段和 reason 值；若另写研发诊断，可同时保留 k_eff 诊断，但不能让 downstream 读错字段。

### 4. OnSite 现在有什么、缺什么

实测来源一：`data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet`。

- 该表用 `pyarrow.parquet.ParquetFile` 实测 67,861 行、66 列、1 个 row group。关键列包括 `case_key`、`scene_unique_id`、`unit_composite_key`、`frame_index`、`anchor_timestamp`、`perspective`、`ego_key_agent`、`counterpart_key_agent`、`source_dataset`、`scenario_idx`、`track_id`、`key_agents`、anchor 处 ego/counterpart 的 velocity/heading、若干相对运动特征、`counterpart_ipv_current`、`target_ipv_future`、`M4_ONLY_ego_self_anchor_ipv_current`、`history_row_count`、几何映射列。
- 只读统计：`source_dataset` 全部为 `onsite_competition_clean_285`，67,861/67,861，来源该表 `source_dataset` 列；`perspective` 全部为 `onsite_av_primary`，67,861/67,861，来源该表 `perspective` 列。
- `history_row_count` 最小 4、最大 10；分布为 `{4:267,5:265,6:264,7:261,8:258,9:257,10:66289}`，来源该表 `history_row_count` 列。

实测来源二：`data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_ipv_timeseries_multi_allvalid.parquet`。

- 该表用 `pyarrow.parquet.ParquetFile` 实测 70,317 行、37 列、1 个 row group。关键列包括 `case_key`、`scene_unique_id`、`unit_composite_key`、`frame_index`、`timestamp_ms`、`time_s`、ego/counterpart 的 `x/y/vx/vy/heading`，以及已有 hw4/hw10 IPV/error。
- 67,861 个 anchor 全部能按 `(unit_composite_key, frame_index)` 找到当前 timeseries 行，匹配 67,861/67,861，来源 anchor 表与 timeseries 表的 `unit_composite_key/frame_index` 列。
- OnSite stage3plus 两个当前 parquet 的 reference-like 列实测：anchor 表只有 `geometry_mapping_status` 命中 `map/lane/route/reference/refline/centerline` 字符串，timeseries 表无命中列；来源两个 parquet schema names。映射说明也写明 replay logs 不提供等价 InterHub 的 map/path metadata，几何字段是运动学启发式，见 `onsite_to_m3_categorical_mapping.md:3-17`。

实测来源三：K2 dense OnSite ledger `[RQ015A concentration ledger]/concentration_ledger/onsite_dense_timeseries.parquet`，该目录由 `k2_fullcorpus_materializer.py` 的 `LEDGER_PARENT` 定义，见 `k2_fullcorpus_materializer.py:39-51`。

- 该表用 `pyarrow.parquet.ParquetFile` 实测 281,268 行、16 列、3 个 row group。
- `artifact_id` 全部是 `onsite_dense_timeseries`，281,268/281,268，来源 `artifact_id` 列。
- `measurement_role` 四类各 70,317 行：`ego_hw4`、`ego_hw10`、`counterpart_hw4`、`counterpart_hw10`，来源 `measurement_role` 列。
- `attempt_status` 分布为 `NOT_ATTEMPTED=4,272`、`UNKNOWN=274,022`、`ATTEMPTED=2,974`，来源 `attempt_status` 列；`reason_code` 中 `D0_WARMUP=4,272`、`EMPTY_CELL_UNEXPLAINED=274,022`、null 2,974，来源 `reason_code` 列。
- 该 K2 source ledger 自身没有 `mse_0` 列；本地 L1 pass-through 也把 OnSite/WOD `ATTEMPTED` 设置为 `out_of_scope_reason=NO_MATERIALIZER_IN_SCOPE`，见 `k2_fullcorpus_materializer.py:1147-1177`。这解释了为什么 OnSite dense 目前没有七候选 MSE 和第一道机制字段。

接入求解器所需输入逐项对照：

- `MotionSequence.data=[x,y,vx,vy,heading]`：OnSite timeseries 表有 ego/counterpart 的对应列，满足接口形状，来源 `onsite_ipv_timeseries_multi_allvalid.parquet` schema；接口定义见 `ipv_estimation.py:36-55`。
- `target`：InterHub 用 `classify_heading()` 从全 aligned motion 的 heading delta 得到 `lt/rt/gs`，见 `process_interhub.py:583-590` 与 `process_interhub.py:1105-1115`。OnSite 没有 lane/route target；必须复用 heading-delta 分类，且 canary 要核对同一 unit 中 ego/counterpart label 是否稳定、是否被 `ensure_unique_labels()` 改名，见 `process_interhub.py:593-599`。
- `reference`：OnSite 没有可用 reference-line 字段；PI 合同是观测轨迹 fallback。InterHub fallback 函数已经有 observed trajectory fallback 逻辑，见 `process_interhub.py:407-410`，但它依赖 pkl event schema。OnSite 版应直接从 timeseries 的 observed `[x,y]` 构造 reference，然后用同一 clip/smooth 参数处理，见 `process_interhub.py:423-473`。
- `history window`：不能只信 `history_row_count`。只读核查显示，多数 anchor 的 `history_row_count=10` 对应 last-11 含当前帧窗口，但也有早期/缺帧单元不满足简单等式；来源 anchor 表 `history_row_count` 与 timeseries 表 `unit_composite_key/frame_index` 分组统计。OnSite materializer 必须按 timeseries 的实际帧序列取 `current plus previous up to 10`，并记录实际窗口长度。
- `canonical key`：K2 dense ledger 的 `product_row_key` 形如 `case_key=...|frame_index=...|timestamp_ms=...`，来源 `[RQ015A concentration ledger]/concentration_ledger/onsite_dense_timeseries.parquet` 的 `product_row_key`、`measurement_role`、`aggregation_configuration` 列。OnSite anchor 表有 `case_key/frame_index` 和 `anchor_timestamp`，timeseries 表有 `timestamp_ms`，所以必须明确用哪个 timestamp 参与 K2 key，并做 exact-one join 验收。

缺口的精确边界：

1. 缺一个 OnSite 输入 builder：读取 scope B anchor 表和同目录 timeseries 表，按 `unit_composite_key` 建序列，按 anchor 当前帧取窗口，生成一行一个 solve unit 的 manifest/input CSV。
2. 缺一个 OnSite solver adapter：不走 InterHub pkl/index，直接构造 ego/counterpart `MotionSequence`，按 PI 合同用 observed trajectory fallback reference，调用 `Agent.estimate_self_ipv(..., return_details=True, solver_mode="exact", candidate_ipv_values=legacy7)` 或等价 `estimate_ipv_pair()` diagnostic。
3. 缺一个 OnSite K2 writer：从 diagnostics 取 observed 与 7 条 virtual track，计算 `mse_0..6/log_score_0..6/w_log_0..6/max_w_log/mse_spread/k_eff_log/status/reason_code/ipv_log`，字段名与 K2 L1 一致。
4. 缺一个 OnSite integration/validation：把 67,861 anchor 输出与 K2 dense ledger 或新 artifact key exact-one 对齐，明确是否只写 AV ego anchor role，还是需要扩展到 K2 dense 的四个 role。PI 口径写的是 67,861 行 source anchor，不是 281,268 dense-role 行；这里必须在开跑前固定输出域，否则 downstream 会把行数不一致误读成缺失。

### 5. HPC 应该怎么用

共享 HPC 指南给出的硬规则：

- SSH alias 是 `tongji-hpc`，HPC home 是 `/share/home/u25310231`，工作根是 `/share/home/u25310231/ZXC`，见 `../HPC_TONGJI_USAGE_GUIDE.md:14-20`。
- 重计算必须通过 Slurm `sbatch`，job name 必须以 `zxc-` 开头，见 `../HPC_TONGJI_USAGE_GUIDE.md:21-46`。
- 读身份、work root、queue、sacct、sinfo 的推荐命令见 `../HPC_TONGJI_USAGE_GUIDE.md:48-81`。
- 默认目录布局是 `/share/home/u25310231/ZXC/<project_or_rq>/{code,data,envs,scripts,logs,work_dirs,manifests,checkpoints}`，见 `../HPC_TONGJI_USAGE_GUIDE.md:83-101`。
- CPU sbatch 模板包含 `--nodes=1`、`--ntasks=1`、`--cpus-per-task`、`--time`、logs、`set -euo pipefail` 和 `cd /share/home/u25310231/ZXC/<project_or_rq>`，见 `../HPC_TONGJI_USAGE_GUIDE.md:103-126`。

RQ015K 可复用先例：

- K2 stage script 用 `/share/home/u25310231/ZXC/sociality_estimation` 作为 remote project，run dir 是 `${REMOTE_PROJECT}/work_dirs/INFRA/rq015k_k2_fullcorpus_<timestamp>`，见 `stage_and_submit_k2_fullcorpus.sh:15-20`。
- 它复制 `src/sociality_estimation`、`pipelines/interhub/process_interhub.py`、`configs`、RQ015B repair solver scripts、K2 materializer、manifest、validation、inputs，并建立 frozen PKL root symlink，见 `stage_and_submit_k2_fullcorpus.sh:39-63`。
- 它在 HPC work dir 下装本地 pyarrow target，然后 `cd` 到 run dir 提交 sbatch，见 `stage_and_submit_k2_fullcorpus.sh:65-72`。
- Solve array sbatch 名为 `zxc-rq015k-k2`，分区 `intel,fata`，`cpus-per-task=6`，`mem=48G`，`time=04:00:00`，`array=1-460%427`，logs 在 `logs/`，见 `submit_k2_solve_array.sbatch:1-11`。
- Solve sbatch 把 `OMP/MKL/OPENBLAS/NUMEXPR` 全部设为 1，切到 `repo_stage`，使用 `/share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/bin/python`，设置 `PYTHONPATH`，每个 array task 取 manifest list 对应行并执行 `run-shard --workers 6`，见 `submit_k2_solve_array.sbatch:13-30`。
- K2 final successful solve job `2070433` 完成 460/460 task，`alloc_cpus=6`、`req_mem=48G`、分区只有 `fata/intel`，elapsed 范围 `00:01:22` 到 `01:23:09`，来源 `slurm_jobs_summary.json:56-79`。
- K2 报告记录 remote authoritative work dir 为 `/share/home/u25310231/ZXC/sociality_estimation/work_dirs/INFRA/rq015k_k2_fullcorpus_finalize_20260802T175006Z/`，本地 L1/aggregates/manifest/validation 位置见 `K2_fullcorpus_gate_ledger.md:24-34`。

我本轮只读 HPC 快照：

- `ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc 'hostname; whoami; pwd'` 返回 `logini01 / u25310231 / /share/home/u25310231`。输出还带 known_hosts 更新失败提示，原因是本地沙箱禁止写 `~/.ssh/known_hosts`，但命令本身成功。
- `sinfo -h -o "%P %D %c %t"` 显示 `intel` 当前有 mix/alloc/idle 节点，`fata` 有 inval/drain/mix，`amd` 虽有 mix/alloc/idle 但本任务禁用。HPC script 应保留 `BatchMode=yes`，并可像 K2 先例一样使用显式 SSH options，见 `stage_and_submit_k2_fullcorpus.sh:11-14`。

OnSite 版运行目录建议：

- Remote root：`/share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ017/rq017_onsite_r2b_<UTC>/`，下面放 `repo_stage/`、`logs/`、`pydeps/`、`manifests/`、`outputs/`。
- 上传最小源：`src/sociality_estimation`、`pipelines/interhub/process_interhub.py` 中的 reference/heading helper 或等价复制、`configs/ipv_sigma01_exact.json`、新 OnSite materializer、需要复用的 RQ015B/K2 helper（若引用）。
- 上传数据：anchor parquet 16M、timeseries parquet 13M、K2 OnSite dense source parquet 716K，来源 `ls -lh` 对上述三个文件；不要上传整个 `.codex-fleet/rq015b-repair/work`，因为本地目录实测 3.5G，来源 `du -sh .codex-fleet/rq015b-repair/work`。只复制 `run_b2_rq015b.py` 58K 或直接把需要函数移入 OnSite materializer。
- 取回：每分片 parquet、manifest、validation JSON、Slurm stdout/stderr、sacct 摘要；若输出小于 8GB 可全量取回，K2 fetch 脚本就是按 remote L1 size 决定全量或部分取回，见 `fetch_k2_outputs.sh:14-31`。

作业形状建议：

- 先做 canary，再 full array。canary 不应只跑一行；至少包含 early window、full window、缺帧窗口、两个城市/多个 task、预期成功行、预期工程失败注入行。K2 closeout 的教训是 one-row canary 没覆盖多 worker 并发和工程失败写出，见 `K2_fullcorpus_gate_ledger.md:200-208`。
- Full run 初始形状我会用 68 个分片，每片约 1,000 个 anchor，array `1-68%24`，每 task `cpus-per-task=6`、`mem=48G`、`time=04:00:00`、`--partition=intel,fata`。依据是：K2 对 4,981,984 个 InterHub solve unit 使用 460 分片、每片目标 11,000，见 `prepare_inputs_summary.json:6-9` 与 `prepare_inputs_summary.json:27-29`；OnSite scope 只有 67,861 个 anchor；使用 `%24` 足以避免为了小作业占用几百个 task 槽，并避开当前 `fata` 多数不可用状态。
- canary 后按实测速率调整：`rows_per_shard = clamp(floor(target_wall_seconds / p95_seconds_per_row), 500, 4000)`，`target_wall_seconds` 取 3 小时以内，full `time` 仍留 4 小时；并发上限取 `min(num_shards, 24, floor(allowed_idle_or_mix_cpu_slots/6))`，若 live `fata` 仍 inval/drain 居多则等效只按 `intel` 估算。

### 6. 怎样证明与既有 InterHub 台账处于同一软件栈

仅仅“也在 HPC 上跑”不够，因为同一 HPC 上可能使用不同 checkout、不同 `PYTHONPATH`、不同 env、不同 materializer 脚本、或 `configs/ipv_sigma01_exact.json:22-25` 里保留的 legacy remote path。充分验收应该至少包含：

1. 每个 OnSite shard manifest 写入 git HEAD、核心文件 SHA、materializer SHA、python executable、`PYTHONPATH`、env package versions、thread env。K2 的 `code_sha_label()` 已对 `agent.py`、`ipv_estimation.py`、`reliability_logdomain.py`、`process_interhub.py`、`ipv_sigma01_exact.json` 和 K2 script 做 SHA 汇总，见 `k2_fullcorpus_materializer.py:202-225`。
2. OnSite 开跑前比较当前核心 SHA 与 K2 accepted report 记录：`agent.py=bde0f582...`、`ipv_estimation.py=e2c84e62...`、`reliability_logdomain.py=8f74067...`、`process_interhub.py=2010433...`、`ipv_sigma01_exact.json=3add56c...`，K2 报告来源 `K2_fullcorpus_gate_ledger.md:212-220`。我本轮本地只读 `shasum -a 256` 得到相同五个核心 SHA；当前 git HEAD 是 `406e7a6595ab145f6b102cf461f8a72b4923c424`。
3. 对同一小批 InterHub G-HPC anchor 再跑 OnSite materializer 的 shared solver adapter，或把 OnSite adapter 对 InterHub row 的输出和 K2/G-HPC MSE 做 float64 exact equality。K2 自身的正确 canary是 2,300 个 G-HPC anchor `max_abs_diff=0.0`，见 `K2_fullcorpus_gate_ledger.md:104-120`。
4. 验收导入路径：HPC stdout/manifest 必须证明 import 的 `sociality_estimation.core.agent` 来自 `repo_stage/src/...`，不是 retired `/share/home/u25310231/ZXC/ipv_estimation` 或用户 site-packages。
5. 输出 schema 和 reason 值必须和 K2 interface 一致：候选顺序、`mse_0..6`、`w_log_0..6`、`max_w_log`、`mse_spread`、`status/reason_code`，接口字段见 `INTERFACE_NOTE.md:13-24` 与 `INTERFACE_NOTE.md:35-52`。

### 7. 安全边界

- RQ007 held-out 不得解析。K2 input builder 显式把 held-out token 设为 `HELD_TOKENS`，若进入 solve input 直接报错，见 `k2_fullcorpus_materializer.py:71-75` 与 `k2_fullcorpus_materializer.py:525-539`。OnSite 源不应触碰 RQ007 split；若 join 到 K2 source ledger，不能读取 held-out 行的 downstream fields。
- RQ014 评分相关字段不得读取。本轮 OnSite anchor/timeseries schema 没有 RQ014 score/rating 列；若 materializer输入 builder 读取目录，必须列白名单，不使用 `*` 读全表。
- 不覆盖现有 K2 L1。K2 writer 用 `SCHEMA_MISMATCH` 和 manifest sha 防止同 manifest 覆盖，见 `k2_fullcorpus_materializer.py:850-879`。OnSite 版应写新 run id 目录，最终只在 commander 批准后发布或合并。
- `amd` 不得用。K2 先例全部使用 `intel,fata`，见 `submit_k2_solve_array.sbatch:2-10`；K2 report 也记录 all submitted jobs used `--partition=intel,fata`，见 `K2_fullcorpus_gate_ledger.md:181`。
- 不能把 `ipv_log=0` 当作弃权，见 `INTERFACE_NOTE.md:54-82`。
- 不能把 `gate_applicable=false` 的 pass-through 行计入 `NO_IPV_EFFECT/NEAR_UNIFORM`，见 `INTERFACE_NOTE.md:47-52` 与 `K2_fullcorpus_gate_ledger.md:198`。
- OnSite reference 合同必须固定为 observed trajectory fallback。已有 InterHub fallback 代码见 `process_interhub.py:407-410`；OnSite mapping 文件说明无等价 map/path metadata，见 `onsite_to_m3_categorical_mapping.md:3-17`。
- 不把描述性结果写成因果主张；本轮只产出第一道机制台账，不改第二道 envelope。

### 8. 我会设定的验收判据与 canary

Canary：

- C0 schema/key canary：100 行不跑求解，只生成 input manifest，验证 100/100 可 exact-one join 到 timeseries 当前帧，0 duplicate key，0 missing key，source 文件 sha 写入 manifest。
- C1 solver canary：至少 200 个 anchor，分层覆盖 `history_row_count=4..10`、early/full window、北京/上海/不同 task、几何关系类别；用同一 sbatch 形状 `workers=6` 跑，要求并发输出与 serial rerun 逐列一致。RQ015B 的 serial check 模式可复用，见 `run_b2_rq015b.py:409-488`。
- C2 failure canary：人为给一个非有限输入副本，要求输出 `NON_FINITE_INPUT` 且数值列全 null；人为给一个空/短窗口副本，要求 fail-closed，不得写 OK。
- C3 InterHub stack canary：同一 HPC run dir 对一小批 G-HPC anchor 运行 shared solver adapter，要求 MSE float64 exact equality，参考 K2 corrected G-HPC check `anchor_rows=2300/compared_rows=2300/max_abs_diff=0.0`，见 `K2_fullcorpus_gate_ledger.md:104-120`。

Full-run 验收：

- Scope coverage：输出 anchor rows 必须是 67,861/67,861，筛选条件为 source anchor parquet 全表，来源 `onsite_m3_av_anchors_multi_allvalid.parquet` 行数；duplicate canonical key 必须 0。
- Key join：输出每行必须 exact-one join 到 source anchor `(case_key, unit_composite_key, frame_index)` 和 timeseries 当前 row；missing 0，duplicate 0。
- K2 integration domain：若写回 dense ledger，必须声明并验证是 67,861 个 AV anchor rows 还是 271,444 个 four-role rows；该域不固定时不得开跑。
- Numeric invariants：每个 `status=OK/ABSTAIN` 行必须有 7 个 finite `mse/log_score/w_log/candidate_ipv`；`sum(w_log)` 与 1 的绝对差不超过 `1e-12`；`max_w_log` 在 `[1/7,1]`；`k_eff_log` 在 `[1,7]`。K2 invariant 逻辑见 `k2_fullcorpus_materializer.py:1367-1405`。
- Reason order：`mse_spread==0` 必须是 `NO_IPV_EFFECT`；`mse_spread!=0 and max_w_log<0.20` 必须是 `NEAR_UNIFORM`；其他可用数值行才可 `OK`，见 `k2_fullcorpus_materializer.py:1367-1405`。
- Engineering failures：`NON_FINITE_INPUT/SOLVER_FAILURE` 行的 `mse_0..6/w_log_0..6/max_w_log/mse_spread/k_eff_log/ipv_log` 必须全 null，见 `k2_fullcorpus_materializer.py:720-758`。
- Provenance：每个 shard manifest 必须有 input sha、output sha、code sha、row count、row key bounds、python path、Slurm job id；manifest output sha 必须逐文件复算通过，K2 manifest validator 见 `k2_fullcorpus_materializer.py:1511-1529`。
- HPC evidence：sacct 要证明 job names 以 `zxc-` 开头、分区只含 `intel/fata`、无 `amd`、array task 数等于 manifest 数、完成状态可解释。

阶段一结论：`GO_WITH_CHANGES`。必须先补的是 OnSite-specific materializer 和输出域合同；复用求解核心、K2 L1 字段、K2 Slurm 外壳和 K2 invariant 验收，但不能直接照搬 InterHub pkl builder，也不能把 K2 pass-through OnSite rows 当成已完成。

## 阶段二：对照方案

opened_plan_utc: 2026-08-04T06:50:29Z
opened_plan: `.codex-fleet/rq017-onsite-materializer/board/RQ017_M1_kickoff_v3.md`

### 1. 我独立推出来、方案里也有的内容

- 目标一致：方案明确本轮只补机制一，不做机制二打分，也不对任何车辆下结论，见 `RQ017_M1_kickoff_v3.md:42-46`；这与阶段一定位一致。
- 范围一致：方案把 scope 固定为 67,861 个 timing-valid anchor frame，见 `RQ017_M1_kickoff_v3.md:48-54`；我阶段一也认为 PI 口径是 67,861 行，不是直接照 K2 dense 四角色 281,268 行。
- 复用边界一致：方案只复用 `MotionSequence/estimate_ipv_pair` 和 K2 `gate_from_mse()`，见 `RQ017_M1_kickoff_v3.md:70-78`；我阶段一也认为新写边界只应是 OnSite 行到 `MotionSequence` 的入口和 L1 writer。
- 不复用边界一致：方案禁止把 OnSite 送进 `run_b2_rq015b.py:solve_anchor_task()`，也禁止整体复用 K2 `validate_outputs()`，见 `RQ017_M1_kickoff_v3.md:79-92`；这与我对 InterHub pkl builder、K2 hard-coded counts 的判断一致。
- 门判据一致：方案逐条冻结 `sigma=0.1`、7 点网格、`mse_spread==0`、`max(w_log)<0.20`、工程失败隔离，见 `RQ017_M1_kickoff_v3.md:99-115`；这对应阶段一从 `k2_fullcorpus_materializer.py:649-689` 推出的 K2 L1 口径。
- HPC venue 一致：方案按 K2 先例使用 `/share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ017/<run_id>/`、`intel,fata`、`cpus-per-task=6`、`mem=48G`、`time=04:00:00`，见 `RQ017_M1_kickoff_v3.md:167-227`；这与阶段一从 HPC guide 和 K2 sbatch 推出的外壳一致。
- 同源证明方向一致：方案要求环境版本记录和 K2 G-HPC anchor baseline 逐位复算，见 `RQ017_M1_kickoff_v3.md:231-245`；我阶段一也认为“也在 HPC 上跑”不够，必须给核心 SHA、import path、G-HPC canary 和 manifest 证据。

### 2. 方案里有、我阶段一没想到或没写到位的内容

- `product_row_key` 合同：方案要求输出键与 `.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet` 逐行一对一，见 `RQ017_M1_kickoff_v3.md:127-134`。我阶段一只提出要固定输出域，没有把 RQ016C dry-run 作为正式合同；阶段二实测 dry-run 67,861 行、`product_row_key` 唯一 67,861、与 anchor 构造键交集 67,861/67,861，来源 dry-run `product_row_key` 列和 anchor 表 `case_key/anchor_frame_index/perspective/source_dataset` 列。
- 旧通道 denylist：方案明确 `ONSITE_CHANNEL_EXACT_HW10` 旧通道列入 denylist，见 `RQ017_M1_kickoff_v3.md:58-69`。我阶段一没有识别这个命名风险；它必须保留，因为错误通道会让输出看似合法但分母和窗口语义错。
- 行位置窗口语义：方案引用 OnSite anchor 构建脚本，明确按 `pos` 而不是绝对 `frame_index` 取窗口，见 `RQ017_M1_kickoff_v3.md:136-148`。阶段二补读源码后确认：`valid_anchor_positions()` 用 `pos`、`TARGET_FINAL_OFFSET`、`FEATURE_HISTORY_WINDOW` 和 `frame_index >= MIN_OBSERVATION` 判定，见 `build_onsite_m3_anchors_hpc.py:836-847`；`build_anchor_rows()` 也用 `wx_start=max(0,pos-FEATURE_HISTORY_WINDOW+1)` 和 `wx_valid` 写 `history_row_count`，见 `build_onsite_m3_anchors_hpc.py:930-997`。
- 短历史不可剔除：方案把 1,572 行短于 10 的历史列为必须保留，见 `RQ017_M1_kickoff_v3.md:150-153`。我阶段一测到了分布，但没有明确写成“不得在输入端剔除”的 blocker。
- 参考线 fail-closed：方案指定去重后观测轨迹点数少于 2 必须 fail-closed，见 `RQ017_M1_kickoff_v3.md:268-272`；源码 `prepared_reference()` 正是少于 2 点抛 `ValueError("observed reference has fewer than two unique points")`，见 `build_onsite_m3_anchors_hpc.py:705-710`。我阶段一只说观测轨迹 fallback，没把 degenerate reference 作为 canary 用例。
- 7 行坐标异常：方案要求 7 行约 570,761m 的异常不能静默丢弃，见 `RQ017_M1_kickoff_v3.md:274-279`。阶段二实测 `relative_distance_anchor > 100000` 的行数为 7/67,861，来源 anchor 表 `relative_distance_anchor` 列；这 7 行都在 `onsite:shanghai:T10:C4:native_case:2311` 的 frame 144-150，`relative_dx_anchor` 约 -570,761，`relative_dy_anchor` 约 -8。
- canary 更具体：方案要求 canary 走 sbatch、至少 2 个并发 array task、覆盖四种状态、读回 null scalar，并逐一规避 K2 的三个真实失败，见 `RQ017_M1_kickoff_v3.md:283-300`。我阶段一提出了并发和失败 canary，但没有绑定 K2 job `2069424/2069818/2071368` 的具体失败模式。
- 强制负对照：方案要求把 `mse_spread==0` 换成 `np.isclose(atol=1e-12)` 和把 theta 从 0.20 改为 0.22 两条检查必须真的 FAIL，且用合成 sentinel 保证触发，见 `RQ017_M1_kickoff_v3.md:322-337`。我阶段一只写了正向 invariant，没有要求“检查自身能失败”。

### 3. 方案里有、但我认为可压缩或只作背景的内容

- 方案头部引用 Mac 与 HPC 在 2,300 个锚点上的差异，见 `RQ017_M1_kickoff_v3.md:8-13`。这对 venue 裁定有解释价值，但本轮 PI 已裁定 venue=HPC；执行报告里应把它作为背景，不应变成新的待复算任务。
- 方案 §6.1 要求若干冻结事实“直接引用，不要重算后报略不同的数”，见 `RQ017_M1_kickoff_v3.md:250-258`。我同意正式报告不要重开这些口径，但开跑前仍应允许用当前 L1 做只读存在性核查；阶段二实测本地 `data/derived/rq015k_logdomain_gate/l1_v1/artifact_id=onsite_dense_timeseries` 的 281,268 行中 `mse_0..6/max_w_log/mse_spread/status/reason_code` 非空计数全为 0，`gate_applicable=False` 为 281,268/281,268，来源对应 L1 parquet 列。

### 4. 我阶段一想到、方案没有或不够明确的内容

- SSH/known_hosts operational caveat：我只读 SSH 成功，但本地输出 `known_hosts` 更新失败，因为沙箱不能写 `~/.ssh/known_hosts`。K2 stage script 用 `UserKnownHostsFile=/dev/null` 和 `StrictHostKeyChecking=no`，见 `stage_and_submit_k2_fullcorpus.sh:11-14`；RQ017 stage script 应明确采用同等 SSH options，避免把本机 known_hosts 写失败误判为 HPC 不可达。
- Manifest shard ownership：K2 closeout 指出 `row_key_min/max` 不是全局不重叠证明，建议记录 source row index interval，见 `K2_fullcorpus_gate_ledger.md:200-208`。方案的 run receipt 要求逐分片行数与耗时，见 `RQ017_M1_kickoff_v3.md:310-315`，但我建议再强制写 `input_row_index_start_inclusive` / `input_row_index_end_exclusive`。
- Per-shard code SHA/import path：方案 §5 要环境同源，§8 要运行回执含输入与代码哈希，见 `RQ017_M1_kickoff_v3.md:231-245` 与 `RQ017_M1_kickoff_v3.md:310-315`。我建议把 K2 `code_sha_label()` 模式逐字复用到每个 shard manifest，并记录 import 源文件绝对路径，依据 `k2_fullcorpus_materializer.py:202-225`。

### 5. 是否有与方案结论相反的地方

没有实质相反结论。阶段一我把“如果要接 K2 dense ledger，必须先固定一行 anchor 还是四角色展开”列为待定风险；方案 §C2 已明确输出一行一 anchor、67,861 行，若判断必须四角色展开要停下上报，见 `RQ017_M1_kickoff_v3.md:133-134`。这不是冲突，而是方案比我的阶段一更明确。

阶段二结论：`GO_WITH_CHANGES`。方案方向正确，但开跑前必须把 `product_row_key`/输出粒度、行位置窗口、环境同源、失败型 canary、负对照和不覆盖检查全部落成脚本 blocker；否则错误会以“数值合法”的形式静默通过。

## Q1. 本轮应当复用哪些现成组件与先例？逐个给出文件路径与行号（HPC 上的给绝对路径），并说明它在链路中承担什么。

答：

- `src/sociality_estimation/core/ipv_estimation.py:36-55`：`MotionSequence` 输入合同，承接 OnSite ego/counterpart 的 `[x,y,vx,vy,heading]`。
- `src/sociality_estimation/core/ipv_estimation.py:181-241` 与 `src/sociality_estimation/core/ipv_estimation.py:247-371`：`estimate_ipv_pair()` 的窗口、求解和 diagnostics 输出，用于取得 virtual tracks 和 weights。
- `src/sociality_estimation/core/agent.py:61-64`、`src/sociality_estimation/core/agent.py:769-832`、`src/sociality_estimation/core/agent.py:1097-1141`：七候选网格、`Agent.estimate_self_ipv()` 和 legacy 权重路径。
- `src/sociality_estimation/core/reliability_logdomain.py:167-188`：从七候选 MSE 稳定计算 log-domain 权重。
- `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:53-69`、`:649-689`、`:692-817`、`:1367-1405`：K2 L1 常量、门判据、落库 schema 和 row-level invariant。
- `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/hpc_onsite_ipv/build_onsite_m3_anchors_hpc.py:705-710`、`:836-847`、`:930-997`：OnSite 观测轨迹 fallback fail-closed、行位置窗口语义、anchor row 字段生成。
- `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/submit_k2_solve_array.sbatch:1-30`：Slurm solve array 形状、线程环境、manifest list 调度。
- `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/stage_and_submit_k2_fullcorpus.sh:15-20`、`:39-72`：HPC run dir、staging、pydeps、提交方式。
- HPC absolute precedent: `/share/home/u25310231/ZXC/sociality_estimation/work_dirs/INFRA/rq015k_k2_fullcorpus_finalize_20260802T175006Z/`，来源 `K2_fullcorpus_gate_ledger.md:24-34`；用于对齐 RQ017 run dir 结构，而不是复写该目录。
- `../HPC_TONGJI_USAGE_GUIDE.md:14-46`、`:83-126`：同济 HPC 工作根、`zxc-` 作业名、Slurm CPU 模板和密码提示规则。

## Q2. 把 venue 从本机改到 HPC 之后，**新增**了哪些会静默失败的风险？逐条说明错了会产生什么结果，以及为什么现有检查抓不到。

答：

- 用错 checkout/env/import path：会得到一套内部自洽但与 InterHub K2 不可比的 MSE；权重和为 1、`max_w_log` 范围、`ipv_log` 恒等式都仍会通过，所以必须用 G-HPC baseline 和 core SHA 抓。
- 用了 retired `/share/home/u25310231/ZXC/ipv_estimation` 或用户 site-packages：会变成“在 HPC 上跑但不同源”；K2/OnSite 各自单独验收都可能 PASS，所以 manifest 必须记录 python executable、`PYTHONPATH`、import file path。
- 分区误用 `amd`：数据字段检查抓不到分区来源；只有 sbatch、sacct 和 run receipt 能证明 `partition` 不含 `amd`。
- SSH known_hosts 写失败或 pydeps 不一致被当作普通 warning 忽略：可能导致 staging 半成功、后续用旧依赖；现有 L1 数值检查只看输出，不看远端 staging 完整性。
- Matplotlib cache、PyArrow null array、逐行 SHA 慢路径：K2 已出现过并发或写出失败，见 `K2_fullcorpus_gate_ledger.md:166-181`；单行本机 canary 抓不到，必须有并发 sbatch canary 和 null scalar 读回。
- 远端已有目录被复用：会混入旧分片或旧 manifest；落盘数值 invariant 抓不到 provenance 混杂，必须 `test ! -e run_dir` 和 output sha/manifest 全量校验。

## Q3. 「与既有 InterHub 台账处于同一软件栈」这件事，怎样才算被证明？给出你认为充分的验收做法，并指出仅仅"也在 HPC 上跑"为什么不够。

答：

充分验收是四层证据同时成立：第一，manifest 记录并比对 core file SHA、materializer SHA、git HEAD、python executable、依赖版本和 import source path，K2 可复用 `code_sha_label()`，见 `k2_fullcorpus_materializer.py:202-225`；第二，在 RQ017 HPC run dir 用同一环境重算 K2 G-HPC baseline 的小批锚点，要求 `mse_per_candidate[7]` 逐位相同、`max_abs_diff=0.0`，方案要求见 `RQ017_M1_kickoff_v3.md:231-245`；第三，Slurm/sacct 证明分区只含 `intel,fata`、job name 以 `zxc-` 开头；第四，OnSite 输出 schema 和 K2 interface 字段完全一致，接口见 `INTERFACE_NOTE.md:13-24` 与 `INTERFACE_NOTE.md:35-52`。

仅仅“也在 HPC 上跑”不够，因为 HPC 上可以同时存在不同 work dir、不同 Python env、不同 `PYTHONPATH`、retired path 和用户 site-packages；这些差异会产出数值合法但不可与 K2 InterHub 台账并列解释的结果。

## Q4. 作业形状（分片数、并发上限、`cpus-per-task`、`mem`、`time`）应当怎么定？给出你的算法与依据。

答：

固定资源基线先沿用 K2：`cpus-per-task=6`、`mem=48G`、`time=04:00:00`、`partition=intel,fata`，依据 K2 sbatch `submit_k2_solve_array.sbatch:2-10` 和 K2 final solve job 460/460 completed、`alloc_cpus=6`、`req_mem=48G`、elapsed max `01:23:09`，来源 `slurm_jobs_summary.json:56-79`。

分片算法：先 sbatch canary 测 `p95_seconds_per_row`，再取 `rows_per_shard = clamp(floor(3*3600 / p95_seconds_per_row), 500, 4000)`；`N = ceil(67861 / rows_per_shard)`；并发 `M = min(N, 24, floor(allowed_idle_or_mix_cpu_slots / 6))`，其中 allowed slots 只从 `intel,fata` live `sinfo` 算，不含 `amd`。没有 canary 前的保守初值是约 68 片、每片约 1,000 行、`--array=1-68%24`；全量前必须用 canary 速率替换这个初值。

## Q5. 有哪些验收判据**必须存在**，否则错误会静默通过？逐条列出，并说明缺了它会漏掉什么。

答：

- `product_row_key` exact-one：缺了会漏掉错行、错车、错时刻；方案 C1 见 `RQ017_M1_kickoff_v3.md:127-134`，阶段二实测交集 67,861/67,861。
- 输出粒度 67,861 行：缺了会把 four-role dense rows 当成本轮分母；方案 C2 见 `RQ017_M1_kickoff_v3.md:133-134`。
- 行位置窗口一致：缺了会用绝对 frame_index 或错误历史长度求解；源码依据见 `build_onsite_m3_anchors_hpc.py:836-847` 与 `:930-997`。
- ego/counterpart 主体方向：缺了会把 counterpart 倾向当成 AV；方案 C5 见 `RQ017_M1_kickoff_v3.md:155-156`。
- 参考线 fallback fail-closed：缺了会把单点/重复点 reference 写成 OK 或未解释失败；源码见 `build_onsite_m3_anchors_hpc.py:705-710`。
- 门判据复算：缺了会漏掉 `np.isclose` 误用、theta 误改、reason 顺序颠倒；方案 negative controls 见 `RQ017_M1_kickoff_v3.md:322-337`。
- 工程失败隔离：缺了会把 solver/input 问题计入两个科学 reason；K2 failure null 写法见 `k2_fullcorpus_materializer.py:720-758`。
- 环境同源 blocker：缺了会让不同软件栈的合法数值进入下游；方案 §5 见 `RQ017_M1_kickoff_v3.md:231-245`。
- Manifest/sha/取回完整性：缺了会漏掉远端旧分片、缺分片或本地取回不完整；K2 manifest validator 见 `k2_fullcorpus_materializer.py:1511-1529`。
- Slurm evidence：缺了会漏掉 `amd` 或登录节点重计算；HPC guide 见 `../HPC_TONGJI_USAGE_GUIDE.md:21-46`。

## Q6. 这一轮有没有会污染下游、或不可逆的风险？逐条列出并给出规避办法。

答：

- 覆盖 K2 或 RQ016C 既有产物：会污染下游 source of truth；规避是只写 `data/derived/rq017_onsite_gate/` 和 HPC `/work_dirs/RQ017/<run_id>/`，存在即停，方案边界见 `RQ017_M1_kickoff_v3.md:341-360`。
- 错输出粒度进入 downstream：会改变 67,861 分母并影响 envelope 交叉；规避是 C1/C2 blocker 和 dry-run key 交集。
- 错主体方向进入 downstream：会把 AV 与 counterpart 对调；规避是输出记录 `ego_key_agent/counterpart_key_agent`、role、motion source columns，并做样本人工审计。
- 读取目标/评分/未来列：会破坏盲法或引入 leakage；规避是输入列白名单，方案 C7 见 `RQ017_M1_kickoff_v3.md:161-163`。
- 误把工程失败当科学 reason：会污染机制一分布；规避是工程失败 null invariant 和 failure canary。
- 使用不同软件栈：会让 OnSite 与 InterHub 对照不可解释；规避是 G-HPC baseline 逐位复算和 per-shard code SHA。
- 静默丢弃 7 行坐标异常：会让正式产物不守恒；规避是把 7 行列入 canary 和正式输出，方案见 `RQ017_M1_kickoff_v3.md:274-290`。

## Q7. 若你只能提**一条**改动意见，是哪一条？为什么是它而不是别的。

答：

把“测量合同 preflight”做成正式的、提交前必跑且失败即停止的机器脚本，特别是 C1 `product_row_key` exact-one、C2 67,861 行输出粒度、C3 行位置窗口、C5 主体方向和 C7 输入列白名单。原因是这些错误都会产生有限 MSE、合法权重和可复算门判据，后面的数值 invariant 抓不到；相比作业分片、报告格式或取回方式，这一条最直接决定产物是不是在回答同一个问题。

## Q8. 这一轮该不该开跑？只答 `GO` / `GO_WITH_CHANGES` / `NO_GO` 三者之一，并给出不超过三句话的理由。若答 `GO_WITH_CHANGES`，把必须先改的条目按重要性排序列出。

答：`GO_WITH_CHANGES`。

必须先改：1. 把 C1/C2/C3/C5/C7 测量合同写成 blocker 脚本；2. 把 §5 环境同源 G-HPC baseline 逐位复算写成 blocker；3. 把 canary 扩成至少 2 个 sbatch array task、四状态覆盖、7 行坐标异常、参考线 fail-closed 和两条必须 FAIL 的负对照。
这些改动完成后再投全量；否则错误会以“数值合法”的形式静默进入 downstream。

state: WAITING_ON_COMMANDER
timestamp_utc: 2026-08-04T06:52:04Z
