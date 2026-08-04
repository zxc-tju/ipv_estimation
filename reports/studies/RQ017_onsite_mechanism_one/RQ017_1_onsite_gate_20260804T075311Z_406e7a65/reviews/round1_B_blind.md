# RQ016D-RB Review

Reviewer: B

Blinding note: this Stage 1 section was written before opening `.codex-fleet/rq016d-onsite-materializer/board/RQ016D_M1_kickoff.md`. I did not read `.codex-fleet/rq016d-onsite-materializer/**`, `.codex-fleet/rq016d-review-a/**`, `START_HERE.md`, `STUDIES.md`, `main_workflow.log`, `reports/knowledge/**/README.md`, or any `commander_notes.md` before this section was frozen.

Stage 1 frozen at: 2026-08-04T03:26:48Z

## 阶段一：独立推导

### 0. 当前步骤定位

这项工作服务于在线验证：先判断当前帧的 IPV 数值是否携带候选间差异信息，再用人类 envelope 判断 AV 是否偏离。RQ015 已冻结机制一的配置和判定规则；OnSite 当前缺的是把 67,861 个 timing-valid anchor frames 接进同一套求解与机制一判据生成链路。

本节只基于源码、配置、允许的历史执行脚本、OnSite parquet 实测和 K2 L1 现状推导，不引用方案文档。

### 1. InterHub 求解链路

1. InterHub 常规入口把 pickle 事件加载为两车运动序列。`pipelines/interhub/process_interhub.py:1049-1077` 读取事件、解析 key agents、按共同 timestamp 对齐，并按数据集需要下采样；`pipelines/interhub/process_interhub.py:1078-1101` 构造两车参考线、裁剪、平滑；`pipelines/interhub/process_interhub.py:1105-1116` 生成两条 `MotionSequence`；`pipelines/interhub/process_interhub.py:1167-1175` 调用 `estimate_ipv_pair`。
2. 参考线合同已有观测轨迹兜底：`pipelines/interhub/process_interhub.py:375-410` 优先从 lane-like 字段取中心线，取不到时返回 observed trajectory fallback；`pipelines/interhub/process_interhub.py:423-464` 对参考线按运动范围裁剪和抽样；`pipelines/interhub/process_interhub.py:467-473` 做重复使用前的平滑。
3. `MotionSequence` 输入要求是至少 5 列 `[x, y, vx, vy, heading]`，见 `src/sociality_estimation/core/ipv_estimation.py:36-55`。`estimate_ipv_pair` 的窗口参数、诊断参数和候选网格参数在 `src/sociality_estimation/core/ipv_estimation.py:181-245` 定义；每个 timestep 用 `start=max(0,t-history_window)` 取含当前帧的窗口，见 `src/sociality_estimation/core/ipv_estimation.py:270-284`；主车与交互对象分别调用 `_estimate_agent_ipv`，诊断会保留 observed、interacting、virtual tracks 和 weights，见 `src/sociality_estimation/core/ipv_estimation.py:286-371`。
4. 七候选轨迹来自 `Agent`。候选网格解析在 `src/sociality_estimation/core/agent.py:156-161`，候选任务构造在 `src/sociality_estimation/core/agent.py:209-230`，候选轨迹应用与旧权重计算在 `src/sociality_estimation/core/agent.py:234-260`。旧权重本身是概率域连乘再归一化，分母为 0 时退到均匀权重，见 `src/sociality_estimation/core/agent.py:1078-1141`。
5. RQ015B 的修复脚本展示了从 anchor 到七候选 MSE 的可执行分解。`run_b1_rq015b.py:606-670` 从 InterHub row 构造两条 `MotionSequence`；`run_b1_rq015b.py:742-829` 对当前 anchor 生成候选轨迹、计算 `mse_per_candidate`、`w_log`、`max_abs_diff`、`k_eff_log` 和边界 flag。B2 并行脚本把同一逻辑用于样本，并把 MSE、RMS、旧权重、新权重写成 anchor 行，见 `.codex-fleet/rq015b-repair/work/run_b2_rq015b.py:268-338`。
6. 机制一在 log 域模块里有更干净的合同。`weights_from_mse` 用 `softmax(-MSE/(2*sigma^2))`，见 `src/sociality_estimation/core/reliability_logdomain.py:172-188`。`estimate_reliability` 先校验输入和候选数，再算 step residuals、MSE、权重、`k_eff`、`min_mse`、`loglike_gap`，然后按 `MODEL_MISFIT` 先于 `FLAT_LIKELIHOOD` 赋 status，见 `src/sociality_estimation/core/reliability_logdomain.py:215-291`。D1/D2/D3/D4 的机制分类顺序是 D4、D1、D3、D2、OK，见 `src/sociality_estimation/core/reliability_logdomain.py:294-326`。
7. K2 full-corpus materializer 已把 InterHub 结果写成 L1 台账。输入 CSV 字段合同在 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:76-113`；shard writer 的 manifest 字段和 sha 合同在 `:381-447`；InterHub 输入准备从 `iter_attempted_interhub()` 取 4,981,984 个 ATTEMPTED 单元，做 frame/meta alignment、held token 保护和 PKL sha 检查，见 `:451-617`；`run_shard()` 校验 input sha、行数、运行 solve、排序后写 L1，见 `:850-924`。
8. K2 的当前 gate 计算不需要新设计。`gate_from_mse()` 检查 K=7 和有限 MSE，计算 `log_score`、`w_log`、`max_w_log`、`mse_spread`、`k_eff_log`；`mse_spread==0` 先给 `ABSTAIN/NO_IPV_EFFECT`，否则 `max_w_log < 0.20` 给 `ABSTAIN/NEAR_UNIFORM`，其余为 `OK`，见 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:649-689`。L1 schema 含 `mse_0..mse_6`、`w_log_0..w_log_6`、`max_w_log`、`mse_spread`、`k_eff_log`、`status`、`reason_code`、`ipv_log`，见 `:776-817`。不变量检查覆盖 K、有限性、权重和、`max_w_log` 范围、`k_eff_log` 范围、reason 顺序和 OK/non-OK 的 `ipv_log` 空值关系，见 `:1367-1405`。

### 2. 冻结口径

1. 求解 profile 是 `ipv_sigma01_exact`，solver mode 为 `exact`、sigma 为 0.1、`min_observation=4`、参考线裁剪边界 60 m、参考点上限 40、平滑点数 40，见 `configs/ipv_sigma01_exact.json:3-10`。
2. 当前窗口为 10，future target 窗口为 4，future final offset 为 6，见 `configs/ipv_sigma01_exact.json:12-15`。OnSite 生成脚本也使用 `FEATURE_HISTORY_WINDOW=10`、`TARGET_HISTORY_WINDOW=4`、`MIN_OBSERVATION=4`、`TARGET_FINAL_OFFSET=6`，见 `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/hpc_onsite_ipv/build_onsite_m3_anchors_hpc.py:27-42` 和 `:1309-1319`。
3. 候选网格是 7 个值 `[-3,-2,-1,0,1,2,3] * pi/8`。该值在 B2 脚本中定义于 `.codex-fleet/rq015b-repair/work/run_b2_rq015b.py:65-68`，在 K2 materializer 中定义于 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:53-59`。
4. K2 的机制一阈值合同是 `THETA=0.20`，`mse_spread==0` 优先于 near-uniform，见 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:53-59` 和 `:649-689`。更完整的 RQ015B 机制拆分中，`K_EFF_FLAT_RATIO=0.93`、`LEGACY_DIVERGENCE_TOL=1e-6`，判定顺序是 D4、D1、D3、D2、OK，见 `.codex-fleet/rq015b-repair/work/run_b2_rq015b.py:65-72` 和 `:558-567`。
5. `min_mse` 模型不匹配阈值在 RQ015B 样本脚本里是从有限 `min_mse` 的 p95/p99/p999 派生，并选 p99 作为诊断阈值，见 `.codex-fleet/rq015b-repair/work/run_b2_rq015b.py:570-586`。若本轮只补 K2 L1 机制一字段，则应沿用 K2 `gate_from_mse()` 的 `mse_spread/max_w_log/status/reason_code` 合同；若要同时输出 D1/D3 机制标签，必须显式带上 RQ015B 阈值来源，不能在 OnSite 新拟一个未冻结阈值。

### 3. OnSite 现有什么、缺什么

1. PI 指定 anchor 表实测 67,861 行，来源为 `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet`；行数见 `.codex-fleet/rq016d-review-b/work/onsite_readonly_audit.json:4`，路径由核查脚本 `.codex-fleet/rq016d-review-b/work/inspect_onsite_readonly.py:13-17` 固定。`case_key + anchor_frame_index` 无重复，见 `.codex-fleet/rq016d-review-b/work/onsite_readonly_audit.json:486-492`。
2. anchor 表拥有求解所需身份与当前帧字段：`case_key`、`anchor_frame_index`、`anchor_timestamp`、`ego_key_agent`、`counterpart_key_agent`、`key_agents`、`track_id`、`ego_vx_anchor`、`ego_vy_anchor`、`ego_heading_anchor`、`counterpart_vx_anchor`、`counterpart_vy_anchor`、`counterpart_heading_anchor` 均为 67,861/67,861 非空；筛选条件为 PI 指定 allvalid anchor 表所有行，来源文件和列名见 `.codex-fleet/rq016d-review-b/work/onsite_readonly_audit.json:4` 与 `:452` 周边。
3. 同目录 OnSite 时序表实测 70,317 行，来源为 `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_ipv_timeseries_multi_allvalid.parquet`，路径由 `.codex-fleet/rq016d-review-b/work/inspect_onsite_readonly.py:18-21` 固定，行数见 `.codex-fleet/rq016d-review-b/work/onsite_readonly_audit.json:275`。该表有 `ego_x/y/vx/vy/heading` 与 `counterpart_x/y/vx/vy/heading` 列，生成脚本的 `IPV_COLUMNS` 明确列出这些字段，见 `build_onsite_m3_anchors_hpc.py:117-155`。
4. OnSite 生成脚本已经展示了如何从原始日志得到求解输入：`build_frame_dataframe()` 解析 AV 和 counterpart 位置、heading、速度、相对量，见 `build_onsite_m3_anchors_hpc.py:494-565`；`prepared_reference()` 用观测运动轨迹生成参考线，见 `:705-710`；`estimate_ipv_timeseries()` 构造两条 `MotionSequence` 并调用 `estimate_ipv_pair(history_window=10,min_observation=4)`，见 `:713-736`；`estimate_ipv_at()` 是单点等价调用，见 `:739-775`。
5. OnSite 真正缺的是机制一字段，不是原始成对轨迹。K2 L1 的 `artifact_id=onsite_dense_timeseries` 分区有 281,268 行，见 `.codex-fleet/rq016d-review-b/work/onsite_readonly_audit.json:587`；但 `mse_0..mse_6` 的非空计数为 0，见 `:614-621`；`w_log_0..w_log_6` 的非空计数为 0，见 `:622-628`；`max_w_log`、`mse_spread`、`k_eff_log`、`status`、`reason_code`、`ipv_log` 也都是 0 个非空，见 `:629-635`。筛选条件是 K2 L1 `artifact_id=onsite_dense_timeseries` 全部分区行，列名如上。
6. K2 源台账 `onsite_dense_timeseries` 只有 16 个列，没有 map/lane/route/reference-line 字段；实测 `map_like_non_null_counts` 为空，见 `.codex-fleet/rq016d-review-b/work/onsite_readonly_audit.json:654-684`。同目录 OnSite 时序表也没有 map/lane/route/reference-line 字段，见 `.codex-fleet/rq016d-review-b/work/onsite_readonly_audit.json:475`。因此 PI 给定的观测轨迹 fallback 是唯一可执行参考线合同。
7. 窗口语义不能照搬 InterHub 的绝对 `frame_index`。OnSite `valid_anchor_positions()` 按行位置 `pos` 检查 future offset 和历史行数，见 `build_onsite_m3_anchors_hpc.py:836-847`；`build_anchor_rows()` 用 `wx_start=max(0,pos-FEATURE_HISTORY_WINDOW+1)`，`wx=frames.iloc[wx_start:pos+1]`，并把 `history_row_count` 写成 `len(wx_valid)`，见 `:930-997`。实测 anchor 当前帧在时序表中 0 个缺失，见 `.codex-fleet/rq016d-review-b/work/onsite_readonly_audit.json:691-695`；`history_row_count=10` 有 66,289 行，见 `:696-703`；按绝对 `anchor_frame_index-10..anchor_frame_index` 取窗得到 11 行的 anchor 有 65,839 行，见 `:705-714`。这说明字段 `history_row_count` 是历史行语义，不应把它当成 `MotionSequence` 总行数。

### 4. 必须新写的部分及边界

必须新写的是 OnSite 输入 materializer，而不是新求解器、不是新 gate，也不是新 envelope。

边界应是：

1. 读取 PI 指定 anchor 表 67,861 行和同目录 OnSite 时序表；按 `case_key` 建立有序 frame table，找到每个 `anchor_frame_index` 对应的行位置 `pos`。
2. 对每个 anchor 构造两条 `MotionSequence`：ego motion 使用 `ego_x, ego_y, ego_vx, ego_vy, ego_heading`，counterpart motion 使用对应 counterpart 列；target label 复用 `classify_heading()` 与 `ensure_unique_labels()`，参考线复用观测轨迹 fallback 的 `prepared_reference()` 或仓库主线等价实现。
3. 对每个 anchor 只运行当前帧、当前窗口、冻结七候选网格和 `solver_mode="exact"`。窗口应按 OnSite 的行位置语义取含当前帧的最多 11 行；如果复用 `estimate_ipv_pair()`，应把窗口切片后设 `min_observation=len(window)-1` 或直接复用 `estimate_ipv_current(..., return_diagnostics=True)`，避免绝对 frame index 被误用。
4. 从 diagnostics 或等价 `Agent.estimate_self_ipv(return_details=True)` 中取 observed 与 virtual tracks，计算 `mse_per_candidate`，再调用 K2 的 `gate_from_mse()` 生成 `w_log`、`max_w_log`、`mse_spread`、`k_eff_log`、`status`、`reason_code` 和 `ipv_log`。
5. 写出与 K2 L1 schema 兼容的 OnSite 分区或一个可 join 回 K2 L1 的独立 run；schema、数组列展开和不变量应复用 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:776-817` 与 `:1367-1405`。

不应新写：候选网格、sigma、权重公式、near-uniform 阈值、参考线策略、机制二 envelope。

### 5. 必须守住的安全边界

1. 阶段内只读边界：本报告之外不改源码、配置、已有 run 目录，也不写 `data/derived/`。核查脚本只写在 `.codex-fleet/rq016d-review-b/work/`，路径见 `.codex-fleet/rq016d-review-b/work/inspect_onsite_readonly.py:13-35`。
2. RQ007 保护：K2 InterHub 输入准备遇到 held tokens 会立即失败，见 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:536-538`；OnSite materializer 不应调用会解析 held rows 的 InterHub source path，也不应放宽 InterHub 的 held-token 检查。
3. RQ014 致盲字段保护：本轮只需要轨迹、heading、velocity、anchor identity、K2 bookkeeping。核查脚本显式跳过字段名含 `rating/score/preference/intensity/priority_label` 的值读取，见 `.codex-fleet/rq016d-review-b/work/inspect_onsite_readonly.py:35` 与 `:49-56`。
4. 参考线合同：OnSite 没有真 map/lane/route/reference-line 字段，见 `.codex-fleet/rq016d-review-b/work/onsite_readonly_audit.json:475` 和 `:684`；只能使用观测轨迹 fallback，并在 manifest 中明示。
5. 输出不可静默覆盖：K2 shard runner 会用 input sha、ledger sha、code sha、sigma、grid、row count 和 output sha 判断已完成结果是否可跳过，不匹配则失败，见 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:850-879`。OnSite 新 materializer 应继承同等检查，不能覆盖既有 OnSite pass-through 分区后才发现错误。
6. 角色边界：K2 源 OnSite 台账有四种 `measurement_role`，每种 70,317 行；本轮 PI 范围是 anchor 表 67,861 行，不是四角色全 dense 表。若要回填 K2 L1，必须在 manifest 中明确定义是 AV 当前角色的一行一 anchor，还是四角色扩展，不能让 67,861 与 281,268 混用。

### 6. 我会设置的验收判据与 canary

1. 输入覆盖：67,861/67,861 anchor rows 都能在 OnSite 时序表中找到当前帧；筛选条件为 PI 指定 anchor 表所有行，来源列 `case_key,anchor_frame_index` 和时序表 `case_key,frame_index`，现有只读核查已经显示当前帧缺失为 0，见 `.codex-fleet/rq016d-review-b/work/onsite_readonly_audit.json:691-695`。
2. 唯一键：`case_key + anchor_frame_index` 必须 67,861 distinct、0 duplicates，来源 `.codex-fleet/rq016d-review-b/work/onsite_readonly_audit.json:486-492`；输出 canonical key 也必须同数无重复。
3. 窗口 canary：至少覆盖早期窗口 `history_row_count=4`、ramp 窗口、full window `history_row_count=10`、timestamp gap 非 100 ms 的 case。真实失败模式是把 `history_row_count` 当总行数、用绝对 frame index 切片、或默认 10 Hz 等间隔而忽略实际 timestamp。
4. 参考线 canary：至少一例直行、一例左/右转、一例重复/近重复坐标。如果观测轨迹 fallback 后去重不足 2 点，应 fail closed。真实失败模式是参考线为空但求解器仍返回数值，或 target label 不稳定导致候选轨迹方向错误。
5. 角色 canary：至少比较 ego 与 counterpart 两个方向的输入窗口，确认本轮只写 AV 当前角色时没有把 counterpart 轨迹写成 observed。真实失败模式是 MSE 有数但语义反了，现有 K2 gate 不会发现。
6. Gate order canary：手造或抽样保存三类 MSE：全相等、非全等但 `max_w_log < 0.20`、明显单峰；应分别得到 `ABSTAIN/NO_IPV_EFFECT`、`ABSTAIN/NEAR_UNIFORM`、`OK`，对应 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:649-689`。
7. L1 不变量：所有 OK 行 `mse_0..mse_6`、`w_log_0..w_log_6`、`max_w_log`、`mse_spread`、`k_eff_log`、`ipv_log` 必须有限；非 OK 行的 `ipv_log` 必须为空；`sum(w_log_i)` 必须等于 1；`K` 必须等于 7。现有检查函数见 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:1367-1405`。
8. Existing L1 gap closure：执行前 K2 L1 OnSite 281,268 行中 `mse_0..mse_6`、`w_log_0..w_log_6`、`max_w_log`、`mse_spread`、`k_eff_log`、`status`、`reason_code`、`ipv_log` 都是 0 个非空，见 `.codex-fleet/rq016d-review-b/work/onsite_readonly_audit.json:587-635`。执行后必须对本轮 scope 的目标行给出非空字段计数与行数口径；不能只报作业完成。

阶段一到此冻结；阶段二若改变判断，只在下一节说明，不修改本节。

## 阶段二：对照方案

方案打开时点：2026-08-04T03:29:03Z。打开文件仅为 `.codex-fleet/rq016d-onsite-materializer/board/RQ016D_M1_kickoff.md`。

### 1. 我独立推出来的东西，方案里有没有

有，核心方向一致。

1. 方案把本轮定位为补 OnSite 机制一字段，不做机制二结论，见 `.codex-fleet/rq016d-onsite-materializer/board/RQ016D_M1_kickoff.md:23-29`。这与阶段一“只补 67,861 个 anchor 的 MSE/gate，不新建 envelope”的判断一致。
2. 方案冻结范围为 67,861 行 allvalid anchor 表，见 `RQ016D_M1_kickoff.md:33-39`；我用 parquet 实测得到 67,861 行、`case_key + anchor_frame_index` 无重复，见 `.codex-fleet/rq016d-review-b/work/onsite_readonly_audit.json:4` 和 `:486-492`。
3. 方案冻结观测轨迹 fallback，见 `RQ016D_M1_kickoff.md:40-43`；我从 OnSite 生成脚本确认 `prepared_reference()` 就是用运动轨迹 xy 点去重、抽样、平滑，见 `build_onsite_m3_anchors_hpc.py:705-710`。
4. 方案要求复用求解链路和门判据，实质只写 OnSite 到 `MotionSequence` 的输入适配，见 `RQ016D_M1_kickoff.md:47-98`；这与阶段一第 4 节一致。
5. 方案给出的 gate 规则与 K2 `gate_from_mse()` 一致，见 `RQ016D_M1_kickoff.md:99-118` 和 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:649-689`。
6. 方案强调 `ALLOWED_SPLITS` 不得为 OnSite 放宽，而应走独立入口，见 `RQ016D_M1_kickoff.md:160-172`；这与阶段一安全边界一致。

### 2. 方案里有、而我没想到的是什么；哪些对，哪些多余

正确且应保留：

1. 7 行坐标异常必须单独定性，不得静默剔除。方案见 `RQ016D_M1_kickoff.md:186-193`；我复核为 7 行，筛选条件为 PI 指定 anchor 表中 `relative_distance_anchor > 100000`，来源列 `relative_distance_anchor,relative_dx_anchor,relative_dy_anchor`，见 `.codex-fleet/rq016d-review-b/work/inspect_onsite_readonly.py:163-200` 和 `.codex-fleet/rq016d-review-b/work/onsite_readonly_audit.json:772-826`。
2. canary 必须含四种状态、至少 2 个 worker、写后读回，见 `RQ016D_M1_kickoff.md:174-184`。我阶段一提出了 gate order canary，但没有把并发和写后读回写成硬门槛；方案更完整。
3. 两条负对照必须真的 FAIL，见 `RQ016D_M1_kickoff.md:208-211`。这是检查本身是否有牙齿，应该保留。
4. 全量本机速率若外推超过 6 小时则停下报告，不自行转 HPC，见 `RQ016D_M1_kickoff.md:245-246`。这与本任务只读复审边界不同，但对后续执行是正确的运行边界。
5. 方案指出旧 285 行通道不能用，见 `RQ016D_M1_kickoff.md:65-68`。我阶段一没有特别写这点；它能防止误把 267/285 的旧小样本当成 67,861 的目标范围。

可以作为报告上下文，但不是 materializer 本身的必要条件：

1. 与 InterHub 全语料状态分布对照，见 `RQ016D_M1_kickoff.md:136-140`。它有助于读者理解结果尺度，但 materializer 验收的第一优先级仍是 OnSite 67,861 行的输入、求解、gate 与 schema 自洽。
2. 与 RQ016C 支持门交叉，见 `RQ016D_M1_kickoff.md:136-143`。这对“下一轮还有多少行可判”有用，但不得让该交叉计算变成本轮是否产出机制一字段的阻塞项。

### 3. 我想到、方案里没有的是什么；哪些必须补

必须补：

1. 明确 OnSite 窗口是行位置语义，不是绝对 frame-index 语义。OnSite 生成脚本 `valid_anchor_positions()` 和 `build_anchor_rows()` 都按 `pos` 取窗，见 `build_onsite_m3_anchors_hpc.py:836-847` 和 `:930-997`；实测 `history_row_count=10` 有 66,289 行，但对应求解输入通常应是历史 10 + 当前 1 行，见 `.codex-fleet/rq016d-review-b/work/onsite_readonly_audit.json:696-724`。这必须进入 preflight 和 canary。
2. 明确 role/canonical key 合同。K2 输入字段需要 `measurement_role`、`canonical_key`、`frame_id` 等，见 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:76-113`；但 OnSite 源台账有四类 dense role，而 PI 范围是 67,861 个 AV anchor。执行前必须冻结输出是一行一 anchor 的 AV 当前角色，还是四角色展开，并给出行数等式。
3. OnSite-specific validator 必须独立于 K2 hard-coded InterHub/RQ009 总数。现有 `validate_outputs()` 写死 4,981,984、8,994,736、14,473,982 等常量，见 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:1327-1338`。本轮不能直接复用这整个 validator，只能复用其中的 row-level invariant。
4. 参考线 fallback 需要 fail-closed 条件：去重后的观测轨迹点数少于 2 时不能继续写 OK。生成脚本已有 `prepared_reference()` 的异常，见 `build_onsite_m3_anchors_hpc.py:705-710`，方案没有把它列成 canary。

可补但不阻塞：

1. 在 key_numbers 里单列 K2 当前 gap：L1 OnSite 281,268 行 `gate_applicable` 全 False，见 `.codex-fleet/rq016d-review-b/work/onsite_readonly_audit.json:652-653`。
2. 把 Arrow CPU cache 权限警告记为非阻塞运行环境噪声；本次只读脚本出现该警告但退出码为 0。

### 4. 有没有与方案结论相反的地方

没有根本相反的结论，但有一处需要防误读。

方案在 `RQ016D_M1_kickoff.md:70-97` 写“照抄求解链路”，并列出 `timed_solve()` 调 `solve_anchor_task()`。如果这被理解为直接把 OnSite row 送进 `.codex-fleet/rq015b-repair/work/run_b2_rq015b.py:268-338` 的 `solve_anchor_task()`，我不同意：该函数第 271-272 行会按 InterHub split 拒绝非 development/guard 行，且下游 `run_b1_rq015b.py:606-670` 从 InterHub PKL 构造序列。正确理解应是复用 `Agent`/`estimate_ipv_pair` 的求解与 K2 `gate_from_mse()`，但新写 OnSite row 到 `MotionSequence` 的入口；方案第 94-97 行也支持这个理解。

## 5. 收尾问答

**Q1.** 这个 materializer 应当复用哪些现成组件？逐个给出文件路径与行号，并说明它在链路中承担什么。

1. `configs/ipv_sigma01_exact.json:3-15`：冻结 solver/profile、sigma、窗口、参考线裁剪和平滑参数。
2. `src/sociality_estimation/core/ipv_estimation.py:36-55`：`MotionSequence` 输入合同，要求 `[x,y,vx,vy,heading]`。
3. `src/sociality_estimation/core/ipv_estimation.py:181-245` 与 `:270-371`：`estimate_ipv_pair()` 的窗口、候选诊断和双向求解循环。
4. `src/sociality_estimation/core/agent.py:209-260`：七候选任务构造、候选轨迹收集、旧权重与候选加权 IPV。
5. `src/sociality_estimation/core/reliability_logdomain.py:172-188`：从 MSE 到 log 域权重的稳定 softmax。
6. `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:649-689`：从 `mse_per_candidate` 到 `status/reason_code` 的冻结 gate。
7. `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:776-817` 与 `:1367-1405`：L1 schema 和 row-level invariant。
8. `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/hpc_onsite_ipv/build_onsite_m3_anchors_hpc.py:494-565`、`:705-736`、`:836-997`：OnSite 成对轨迹、观测轨迹 fallback、窗口和 anchor 生成语义。

**Q2.** 求解链路上最容易被做错、且做错后**不会自己暴露**的是哪一步？说明它错了会产生什么样的结果，以及为什么现有检查抓不到。

最危险的是 OnSite 输入适配：把 anchor 的行位置窗口、ego/counterpart 方向、参考线三者之一接错。它仍会产生有限 `mse_0..mse_6`、权重和为 1、`max_w_log/k_eff_log` 落在合法范围内的 L1 行，因此 K2 row-level invariant 会通过；但这些数值对应的是错误窗口或错误主体，机制二会在错误 IPV 上继续运行。

**Q3.** `run_b2_rq015b.py` 的 `ALLOWED_SPLITS` 护栏在 OnSite 场景下应当怎么处理？给出你的具体做法，并说明它为什么不会削弱 InterHub 那条路径的保护。

保持 `.codex-fleet/rq015b-repair/work/run_b2_rq015b.py:72` 和 `:271-272` 原样不动。OnSite 新建独立入口，入口只接受 PI 指定 anchor 表和 OnSite 时序表，写 `rq007_split="NOT_APPLICABLE_ONSITE"` 或同等非 held 标识，并断言参与求解的 InterHub 行数为 0。InterHub 路径继续由原函数和 K2 `prepare_inputs()` 的 held-token failure 保护，见 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:536-538`，所以不会削弱原保护。

**Q4.** canary 必须覆盖哪些路径才算有效？逐条列出，并说明每条对应的真实失败模式。

1. 自然 OK 行：证明 OnSite 到 solver 到 L1 的主路径能写出有限 MSE 和 OK `ipv_log`。
2. near-uniform 行，天然没有就用合成 MSE：防止 `max_w_log < 0.20` 未接线或阈值被改。
3. no-effect 行，使用七候选 MSE 完全相等的合成输入：防止把精确零 spread 写成 near-uniform。
4. 工程失败行，使用 NaN 或参考线不足 2 点的合成输入：防止工程失败被记成科学 reason。
5. 早期、ramp、full 三类窗口：防止 `history_row_count` 被当总行数或绝对 frame index 被误用。
6. ego/counterpart 方向互换检查：防止 MSE 语义反向但数值合法。
7. 7 行坐标异常至少抽 1 行：防止输入端静默过滤异常值。
8. 至少 2 worker 并发并写后读回：覆盖并发缓存、parquet 空数组、sha/manifest 读写等执行失败路径，方案列出的真实历史失败见 `RQ016D_M1_kickoff.md:174-184`。

**Q5.** 有哪些验收判据**必须存在**，否则错误会静默通过？逐条列出，并说明缺了它会漏掉什么。

1. 行数守恒：目标 scope 必须是 67,861 行；缺它会把 70,317 dense frames 或 281,268 四角色行混成目标。
2. 唯一键守恒：`case_key + anchor_frame_index` 和输出 canonical key 必须无重复；缺它会重复判断同一 anchor。
3. 当前帧覆盖：67,861/67,861 anchor rows 必须在时序表找到当前帧；缺它会把空窗或邻近帧当当前帧。
4. 窗口语义检查：按行位置取含当前帧窗口；缺它会把 `history_row_count=10` 截成 10 行或按绝对 frame index 错切。
5. role 检查：输出主体必须是 AV 当前角色；缺它会把 counterpart 行当 AV 行。
6. gate 可复算：从落盘 `mse_0..mse_6` 重新算 `status/reason_code` 必须零差异；缺它会漏掉阈值和 reason 顺序被改。
7. 数值不变量：`sum(w)=1`、`max_w_log in [1/7,1]`、`mse_spread>=0`、`k_eff_log in [1,7]`；缺它会漏掉列错位、空数组写坏和非有限传播。
8. 工程失败隔离：工程失败行不能进入两个科学 reason；缺它会让下游把工程问题当成可解释弃权。
9. 坐标异常纳入：7 行异常必须计数并保留；缺它会改变分母且不可追溯。
10. 负对照必须 fail：缺它无法证明验收脚本真的能抓住错误。

**Q6.** 这一轮有没有会污染下游、或不可逆的风险？逐条列出并给出规避办法。

1. 错误 L1 被机制二消费：先写 `data/derived/rq016d_onsite_gate/` 独立路径，通过验收后再决定是否 join/backfill。
2. 角色或窗口错接但数值合法：用 canary、row-position preflight、role audit 和少量手工轨迹对照规避。
3. 读取 RQ014 致盲评分字段：materializer 只读轨迹、anchor identity、K2 bookkeeping；保留字段黑名单。
4. 触碰 RQ007 held rows：OnSite 独立入口，不读取 InterHub held rows；InterHub guard 原样保留。
5. 静默剔除 7 行坐标异常：在输入审计中计数并保留，失败则工程失败。
6. 覆盖既有 K2 pass-through：新建输出目录，manifest 含 input/code sha，写后读回再发布。

**Q7.** 若你只能提**一条**改动意见，是哪一条？为什么是它而不是别的。

把“OnSite 输入合同”升成 canary 前的硬性 preflight：冻结 canonical key、measurement role、row-position 窗口、参考线 fallback、67,861 行数等式，并输出机器可读审计。原因是求解器和 gate 已经冻结，最可能静默通过的错误都发生在输入适配层；其他问题多数会被数值 invariant 或作业失败暴露。

**Q8.** 这一轮该不该开跑？只答 `GO` / `GO_WITH_CHANGES` / `NO_GO` 三者之一，并给出不超过三句话的理由。若答 `GO_WITH_CHANGES`，把必须先改的条目按重要性排序列出。

GO_WITH_CHANGES。
必须先改：1. 写明并实现 OnSite 独立输入入口，不直接使用 InterHub `solve_anchor_task()`；2. 把 row-position 窗口语义和 AV 当前 role/canonical key 合同纳入 preflight；3. canary 加入四状态、2 worker、写后读回、7 行坐标异常、负对照。方向正确，但这些输入合同不先钉牢，错误会生成合法形状的 MSE 并继续污染下游。

state: WAITING_ON_COMMANDER
timestamp_utc: 2026-08-04T03:31:53Z
