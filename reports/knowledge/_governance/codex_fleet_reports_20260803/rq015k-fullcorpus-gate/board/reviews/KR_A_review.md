# KR_A 独立审查：Track K 全语料重算方案

生成时间（UTC）：2026-08-02T07:32:20Z。

本次审查解决的问题是：从 K 计划书出发，判断“把 J1 已定稿的第一道 IPV 门落到全语料”是否有必要、是否能无歧义执行、是否具备足够的成本与失败恢复设计。整体研究已经走到 J1 门规格完成、现有 RQ015A 台账只保存旧 `ipv_error/q_eff/k_eff` 的阶段；本次是 K1/K2 执行前的独立审查，不执行 K1、不执行 K2、不提交 HPC 作业。

边界声明：我未读取另一审查方报告或日志；未修改 `agent.py` / `ipv_estimation.py` / `process_interhub.py` / `reliability_logdomain.py`；未提交 git commit；未做全仓库 `reports/` 检索作为证据。一次限定检索误命中了较大的 RQ009 CSV 行级内容，我停止后不采用其输出作为本报告证据。

## 总判定

**判定：需修改后执行。**

这轮重算的核心前提成立：现有 4 份 RQ015A concentration ledger parquet 合计 `14,473,982` 行，列集合只有 `artifact_id/product_row_key/measurement_role/case_id/rq007_split/ipv_error/K/candidate_grid_id/k_eff/q_eff/attempt_status/reason_code/recoverability/ledger_schema_version/aggregation_perspective/aggregation_configuration`；没有 `mse_per_candidate[7]`、`w_log[7]` 或 `log_score[7]`。`k_eff` 在 `reports/plans/RQ015A_ledger_schema_v4_20260731.json` 第 143-148 行明确定义为 `1.0 / (1.0 - ipv_error) ** 2`，不能替代 log 域权重。

但 K 计划还不能直接派 K1/K2：它需要补齐批处理状态语义、非有限/缺列行处理、`log_score` 的 σ 绑定、资源外推口径、分片断点续算与产物校验的具体要求。K1 与 K2 分段是必要的，不是多余关卡。

最可能造成实际损失的一处：**K1 的“小批 2,000-5,000 单元”没有要求按 PKL/数据源/角色/行复杂度分层抽样，也没有给出内存上限和分片重投验收标准。** G 轨已经证明 24 worker 因每进程各载一份 PKL 而 OOM，TRES `mem=160992M` 仍不够；若 K2 依据单一小批均值投全量，最现实的损失是几十天量级的 Slurm 时间被 OOM、超时或从头重跑消耗。

## 1. 前提是否成立

**成立。没有现成全语料产物可直接替代 J 门全语料台账。**

我核过的路径如下。

| 路径 | 核查结果 |
|---|---|
| `reports/studies/RQ015A_ipv_estimability_labelling/RQ015A_1_concentration_audit_20260731T093746Z_e82091ce/concentration_ledger/interhub_sigma01_hw4_timeseries.parquet` | pyarrow metadata：`5,197,072` 行；16 列；只有 `candidate_grid_id` 命中 candidate 关键词；无 `mse_per_candidate[7]`、`w_log[7]`、`log_score[7]`。 |
| 同目录 `onsite_dense_timeseries.parquet` | pyarrow metadata：`281,268` 行；同一 16 列；无逐候选量。 |
| 同目录 `rq009_feature_matrix.parquet` | pyarrow metadata：`8,994,736` 行；同一 16 列；无逐候选量。 |
| 同目录 `wod_rq010b_full479_audited.parquet` | pyarrow metadata：`906` 行；同一 16 列；无逐候选量。 |
| `reports/plans/RQ015A_ledger_schema_v4_20260731.json` | 第 143-148 行：`k_eff` 由旧 `ipv_error` 派生；第 149-154 行：`q_eff=k_eff/K`。 |
| `data/derived/.../03_features/matrix/fold=*/source_dataset=*/features_part_*.parquet` | 138 个 part；抽查 schema 59 列，含 `counterpart_ipv_current`、`counterpart_ipv_error_current`、`counterpart_ipv_slope_pre_anchor`、`target_ipv_future` 等，但无 `mse_per_candidate[7]`、`w_log[7]`、`log_score[7]`。 |
| `data/derived/.../04_calibration/predictions/tier=M3/fold=calibration/predictions.parquet` | pyarrow metadata：`3,798,846` 行；15 列为 `case_key/anchor_frame_index/perspective/source_dataset/fold/tier/alpha/nominal/q_lo/q_hi/lo_cal/hi_cal/width/abstain/y`；无 IPV 或逐候选量。 |
| `data/derived/.../04_calibration/predictions/tier=M3/fold=test/predictions.parquet` | pyarrow metadata：`3,811,698` 行；同一 15 列；无 IPV 或逐候选量。 |
| `data/derived/.../03_features/target_hw4/sigma01_hw4_ipv_timeseries.csv` | 表头 39 列，只有 `ipv_key_agent_1` / `ipv_key_agent_1_error` / `ipv_key_agent_2` / `ipv_key_agent_2_error`，无逐候选量。 |
| `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors/onsite_ipv_timeseries.csv` | 表头 37 列，只有四组 `ipv_*` / `ipv_*_error`，无逐候选量。 |
| `data/derived/wod_e2e/rq015a_full479_projected/rq010b_wod_full479_audited_candidate_ipv_projected.csv` | 表头 4 列：`segment_key/candidate_index/ego_ipv/ego_ipv_error`，无逐候选量。 |
| `.codex-fleet/rq015b-repair/work/anchor_mse.csv` | 表头有 `mse_per_candidate[7]`、`rms_per_candidate[7]`、`w_log[7]`；但分母是 J/G 使用的 2,300 锚点样本，不是全语料。 |
| `.codex-fleet/rq015g-hpc-resolve/work/anchor_mse_hpc.csv` | 同样有逐候选量；G1 报告第 48-52 行记录行数 `2,300`、`solve_errors=0`、`held_out_parsed_rows=0`；仍只是锚点样本。 |
| `.codex-fleet/rq015h-abstain-gate/work/w_log_consistency_sample.csv` | 表头含 `weight_sum`、`ipv_log_from_w`、`k_eff_from_w`；`wc -l` 为 11，即 10 行样本加表头，不是全语料缓存。 |
| `.codex-fleet/rq015h-abstain-gate/work/h2_key_anchor_check.csv` | `wc -l` 为 2，即 1 行检查加表头，不是全语料缓存。 |
| `data/derived/.../03_features/parity/hpc_hw4/code_parity_sample_cases_with_ipv.csv` 与 `local_hw4/code_parity_sample_cases_with_ipv.csv` | 表头只有样本级 `ipv_*_mean`、`ipv_*_error_mean` 与 provenance 字段，无逐候选量。 |

能降低成本的现成材料有两类，但不能替代重算：

1. `.codex-fleet/rq015b-repair/work/anchor_mse.csv` 与 `.codex-fleet/rq015g-hpc-resolve/work/anchor_mse_hpc.csv` 可以作为实现与验收样本；不能给全语料逐行门结果。
2. RQ015A schema 第 676-680 行写明 `rq009_feature_matrix` 是 `interhub_sigma01_hw4_timeseries` 的派生产物，`cross_artifact_pooling` 禁止，且 only corpus-level source 是 `interhub_sigma01_hw4_timeseries`。这不能省掉重算，但提示 K2 不应按 ledger 行盲目重复求解派生行，应先把“求解单元”定义到可复用的 `(case/frame/agent)` 或等价物，再把结果 join 回 RQ009 派生行。

## 2. J1 门规格搬到全语料时的歧义

J1 单帧规格本身是清楚的：J1 报告第 31-55 行定义 `mse_spread`、`max_w_log`、`k_eff_log`、先 `mse_spread==0` 后 `max_w_log<0.20` 的有序判定；第 61-75 行定义机器输出字段和空值规则。但 K 计划把它搬到千万行批处理时，仍缺几条必须写进 K1/K2 prompt 的工程规则。

1. **`mse_spread == 0` 必须限定为 7 个有限 float64 的精确相等，不应改成 `np.isclose`。** J1 报告第 57 行记录样本内 `spread(mse)==0` 为 `400/2,300`，且 `max(w_log)=1/7`；G 报告第 176-181 行记录 Mac/HPC 对这 400 行 MSE 字符串逐位相同。这说明这个 reason 是“精确退化标签”，不是容差阈值。K 计划第 74 行只有 `mse_spread == 0`，还应要求先断言 7 个 MSE 有限、长度为 7，再用 `max-min == 0.0`；非有限不进入这两个 ABSTAIN reason。
2. **`log_score` 来源必须固定。** J1 输入契约第 17-18 行允许 `log_score[7]` 或由 MSE 得到的等价量，并推荐 `log_score_i=-mse_i/(2*sigma^2)`。B1 代码第 75-77 行固定 `SIGMA=0.1`、`K=7`、七点网格；`reliability_logdomain.py` 第 14-18 行给出同一推导。K 计划第 65-78 行没有把 `sigma=0.1` 写入批处理规格；如果 K2 在不同 artifact 或不同历史配置中误用别的 σ，`max_w_log` 与 `k_eff_log` 会整体改变。
3. **softmax 数值实现有现成合同，应直接复用。** `reliability_logdomain.py` 第 172-188 行的 `weights_from_mse()` 是稳定 softmax：检查 sigma 与 MSE 有限，计算 `-mse/(2*sigma^2)`，减最大 log weight，再 `exp` 和归一化；分母要求有限且不坍塌。K 计划第 69 行写“用 log-sum-exp”方向正确，但应明确允许“减最大值”实现，且输出中保留 `log_score[7]` 或记录 `sigma` 与公式，否则读者无法复算 `w_log[7]`。
4. **缺列、NaN、inf、求解失败行没有状态定义。** J1 第 17-18 行要求 MSE/log_score 必须有限；`reliability_logdomain.py` 第 36-49 行已有 `NON_FINITE_INPUT`、`SOLVER_FAILURE` 等互斥状态，且第 244-258 行把非有限轨迹或平方距离溢出转成非 OK 结果。B1 的 `solve_worker()` 第 994-1030 行在异常时写 NaN 数组和 `any_nonfinite=True`。K 计划第 74-77 行只有 `ABSTAIN/OK`，这会迫使 K2 在失败行上二选一：要么 NaN 传播后没有 reason，要么被错误地归入 `NO_IPV_EFFECT/NEAR_UNIFORM`。必须新增批处理状态，例如 `INVALID_INPUT` / `SOLVER_FAILURE`，并规定这些行 `ipv_log=null`、`w_log[7]=null` 或全 null，而不是 `ABSTAIN` 的两种科学 reason。
5. **互斥 reason 的判定顺序必须防止向量化覆盖。** K 计划第 82-84 行写明先 `NO_IPV_EFFECT`、再 `NEAR_UNIFORM`，这是对的；但样本内 400 行 `mse_spread==0` 全部也满足 `max_w_log<0.20`，J1 evidence 第 2-8 行和第 69-81 行给了计数。因此 K2 实现必须使用有序 `np.select` 或先写 `NO_IPV_EFFECT` 后只在 `reason is null` 的行写 `NEAR_UNIFORM`；不能用两个布尔 mask 后让第二个覆盖第一个。

## 3. 成本模型与资源方案

K 计划第 42-47 行要求 K1 实测单位成本、每 worker 常驻内存、PKL 载入内存放大倍数，并给资源方案、断点续算和失败重投。这个方向是对的，但要求还不够具体。

G 轨有三个时间口径，必须分开：

| 口径 | 来源 | 外推到 `14,473,982` ledger 行、6 worker 的单节点墙钟 |
|---|---|---:|
| 求解循环 progress | `.codex-fleet/rq015g-hpc-resolve/board/progress.log` 第 5 行开始，`2,300` anchors / 6 workers；第 28 行完成求解，elapsed `500.6s` | `14,473,982 / 2,300 * 500.6 = 3,150,293.6s = 875.08h = 36.46d` |
| driver 记录的 T5 elapsed | `.codex-fleet/rq015g-hpc-resolve/work/g1_hpc_summary.json` 第 650-663 行：`anchors=2300`、`workers=6`、`elapsed_seconds=702.9936774782836` | `4,423,964.3s = 1,228.88h = 51.20d` |
| Slurm wall time | `G1_hpc_resolve_report.md` 第 5 行和 `slurm_attempts_sacct.txt` 第 16-18 行：`00:14:22` | `5,424,596.7s = 1,506.83h = 62.78d` |

如果只按现有 ledger 的 `ATTEMPTED` 行上限外推，分母为 `13,980,600` 行。来源为 `concentration_ledger_summary.csv` 的 `artifact/attempt_status/rows`：OnSite `ATTEMPTED=2,974`（第 2 行），WOD `906`（第 5 行），InterHub sigma01 `4,981,984`（第 8 行），RQ009 feature matrix `8,994,736`（第 11 行）。用同三种时间口径外推分别是 `35.22d`、`49.46d`、`60.64d`。这仍是单 6-worker 节点量级，不含队列等待和重投。

计划若把 `500.6s` 当总 walltime，**计划错了**。`500.6s` 是求解循环完成时间；同一 G 轨机器摘要显示完整 T5 为 `702.9937s`，Slurm 作业为 `00:14:22`。K2 的墙钟预算应至少同时报 solve-loop、driver、Slurm 三个口径。

内存风险不是理论风险。G adjudication 第 242-252 行记录 7 次投递，其中 Job `2023247` 在 24 worker 下 `OUT_OF_MEMORY`，原因是“24 进程各载一份 PKL”，TRES `mem=160992M` 仍不够；最终 Job `2023332` 改为 `fata/6 worker` 才完成。第 254-255 行还记录 leader 曾判断“worker/内存比其实没变”，commander 认可。K1 必须把 worker 数、PKL 分片、RSS 峰值和 Slurm `--mem` 写成硬指标，而不是只报平均秒/单元。

K1 还应补四个具体验收项：

1. 分片单位：每个 shard 的输入范围、PKL 列表、行键范围、行数、角色数、expected output rows 必须固定；同一个 `(case/frame/agent/grid)` 不得因为 `rq009_feature_matrix` 是派生行而重复求解。
2. 断点续算：输出先写临时文件，完成校验后原子 rename；已完成 shard 由 manifest 的 input SHA、code SHA、command、row count、output SHA 决定，不能只看文件是否存在。
3. 失败重投：重投只跑失败 shard；失败原因至少分 `OOM/TIMEOUT/SOLVER_FAILURE/NON_FINITE_INPUT/SCHEMA_MISMATCH`；超过阈值时停止而不是扩大资源硬跑。
4. 产物校验：全局校验必须包括唯一主键、缺失/重复 shard、`K=7`、数组长度 7、finite 规则、reason 互斥顺序、`ipv_log` null 规则、状态计数、抽样重跑一致性、输入清单 SHA、以及 `held_out_parsed_rows=0`。

## 4. K1/K2 分段与速度原则

K1 后停下是必要谨慎，不是多余关卡。AGENTS.md 第 34-40 行说过程强度要和主张风险成比例；第 54-61 行也列出 RQ007 held-out、RQ014 致盲字段、冻结产物和因果表述是不得放松的效度边界。K2 是多日量级 HPC 作业，而且会生成新的下游接口台账；在没有单位成本、内存、分片、断点续算和状态合同之前直接投全量，违反的不是“流程”，而是资源与数据完整性边界。

同时，K1 不应膨胀成新的多轮规格工程。AGENTS.md 第 46-52 行把“尚未运行前多轮计划”和“用治理文书替代实际产出”列为反模式；K 计划第 153-157 行也把本轮限定为一个 agent、一轮自查、出报告。我的建议是：K1 允许一次实测与一次自查，但必须在 K1 prompt 中补上上节硬指标，避免跑完后再争论口径。

K 计划第 48-55 行要求查清 RQ009 三个 IPV 输入列的填充规则。我的判断：

1. 这件事**有必要记录**，因为 calibration 第 141-157 行把 `counterpart_ipv_current`、`counterpart_ipv_error_current`、`counterpart_ipv_slope_pre_anchor` 列入 gate numeric features；第 704-715 行又用 train-fit median imputation、standardization 和 KNN distance 建 out-of-distribution gate。若第一道门输出没有清楚区分 `status/reason_code` 与合法 `ipv_log=0`，RQ009 会继续把占位零和合法零混在一起。
2. 但它**不应作为 K2 是否投递的 blocker**。K2 的 go/no-go 取决于是否有逐候选量、求解单元、成本、内存、失败恢复和输出合同。RQ009 三列的历史填充规则是下游解释问题，可以在 K1 报告附录中用代码路径说明，不应扩展成对 RQ009 4.78% 弃权率的重算。
3. 代码层面当前可确认的规则是：`build_features.py` 第 774-776 行直接从 sigma01 行写 `counterpart_ipv_current`、`counterpart_ipv_error_current`，并用 `theil_sen_slope()` 写 slope；`theil_sen_slope()` 第 583-597 行在少于 2 个有限历史点时返回 NaN；feature dictionary 第 1131-1133 行把三列定义为 t* 当前值、error 和最多 5 个有效 W_x 行的 Theil-Sen slope。Calibration 的 `Preprocessor` 第 211-235 行与 gate 第 704-715 行对数值缺失做 median imputation。因此 K1 只需说明“current/error 沿用上游旧 IPV/error 数值，slope 不足历史为 NaN 后被 median impute”，不要在本轮量化污染规模。

## 共同收尾问题

### Q1. 这轮重算是否必要？有没有现成产物能替代它、或大幅降低它的成本？

**必要。** 现成产物不能替代全语料 J 门台账。原因是 J 门需要 `mse_per_candidate[7]` 与 `w_log[7]` 或可复算 `w_log[7]` 的 `log_score[7]`；现有全语料台账只有旧 `ipv_error/q_eff/k_eff`，其中 `k_eff` 由旧 `ipv_error` 派生。

查过的路径已在第 1 节列出。查到有逐候选量但不能替代的路径只有三类：`.codex-fleet/rq015b-repair/work/anchor_mse.csv`、`.codex-fleet/rq015g-hpc-resolve/work/anchor_mse_hpc.csv`、H 轨 10 行/1 行级样本文件；这些都不是全语料。查了但没有逐候选量的路径包括 RQ015A 四个 concentration ledger parquet、RQ009 feature matrix 138 part、M3 calibration/test predictions、sigma01/onSite/WOD 源表、RQ009 parity sample CSV。

能大幅降低成本的机会不是“现成产物直接替代”，而是 K1 必须定义 canonical solve unit，避免对 `rq009_feature_matrix` 这类派生行重复求解同一底层轨迹-帧-角色。

### Q2. K1 的勘察范围，是否足以支撑“要不要投 K2”这个决定？缺什么？多了什么？

**方向足够，细节不足。**

缺的内容：

- 求解单元必须从 ledger 行进一步收敛到 canonical `(artifact/source case, frame, agent/role, grid)`，并说明派生行如何 join，不能把 `14,473,982` ledger 行直接当最终求解量。
- `log_score` 的来源必须固定为 `-mse/(2*0.1^2)` 或记录等价 log likelihood 的生成源；输出要能复算 `w_log[7]`。
- 非有限、缺列、求解失败必须有独立状态，不得塞进 `NO_IPV_EFFECT/NEAR_UNIFORM`。
- 小批成本必须分层抽样，至少按 PKL、source、n_obs/窗口长度、artifact/role 覆盖；否则无法从 G 轨非线性进度和 OOM 记录外推。
- 断点续算、失败重投、产物校验必须从原则变成 manifest 和 shard 级验收项。

多的内容：

- RQ009 三个 IPV 输入列的填充规则可以保留为 K1 附录，但不应成为 K2 投递前的阻断条件；若开始量化污染比例或重算 RQ009 弃权率，就超出本轮。

### Q3. 明确判定：可执行 / 需修改后执行 / 不应执行。

**需修改后执行。**

理由：

1. 必要性成立：没有全语料逐候选 MSE/log 权重；现有 `k_eff` 不能替代。
2. K1/K2 分段成立：单节点 6-worker 外推是 36.46-62.78 天量级，且 G 轨已有 24-worker OOM 事实。
3. 当前计划缺少批处理失败状态、NaN/缺列规则、σ 绑定、分片/重投/manifest 细则；这些缺口会直接影响 K2 产物可用性。
4. 速度原则不支持再开规格 v2 或多轮盲审，但支持在 K1 prompt 内一次性补齐上述执行合同后再跑。

最可能造成实际损失的一处：小批成本与内存测量若不按 PKL/数据源/角色/行复杂度分层，并且没有 shard manifest 与重投规则，会把 G 轨已经暴露的 OOM 与长尾耗时放大到全语料，造成多日 HPC 资源浪费和不可恢复的半成品台账。
