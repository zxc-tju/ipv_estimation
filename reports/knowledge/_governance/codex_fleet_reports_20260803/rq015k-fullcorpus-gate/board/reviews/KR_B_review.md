# KR-B 独立审查报告：RQ015K 全语料 log 域门台账计划

审查方：KR-B  
生成时间：2026-08-02T08:00:44Z  
边界：只读审查；未执行 K1、未执行 K2、未提交 HPC 作业、未修改受保护代码；未打开 `board/reviews/` 下其他报告，未读任何 `.log`。  
解释器：所有本地 Python 检查均使用 `<local-rq009-venv>/bin/python`。

## 0. 顺序声明

本报告按双盲要求分两阶段写：

1. **第一阶段结论**是在打开 `.codex-fleet/rq015k-fullcorpus-gate/board/K-leader-kickoff.md` 之前形成的。第一阶段只看指定 4 份 parquet、现有产物、代码路径和非 K 计划材料。
2. **第二阶段对照**是在第一阶段结论固定后才读取 K kickoff。

我没有对 `reports/` 做全仓库 `rg`。限定内容搜索只覆盖指定 RQ015A run 目录、非 `reports/` 的代码/`.codex-fleet` 路径，并排除了 K 计划目录、`board/reviews/` 和 `.log`。

## 1. 第一阶段结论（未读 K 计划前）

### 1.1 全语料规模与真实计算单元

指定台账目录：

`reports/studies/RQ015A_ipv_estimability_labelling/RQ015A_1_concentration_audit_20260731T093746Z_e82091ce/concentration_ledger/`

4 份 parquet 的 L1 行合计为 **14,473,982** 行。来源：`bounded_report.md` 第 18-24 行列出四产物合计和分 artifact 行数；`concentration_ledger_summary.csv` 第 1-13 行列出分 artifact、`attempt_status` 和 `rows` 列。

| artifact | ledger rows | ATTEMPTED | NOT_ATTEMPTED | UNKNOWN | 来源 |
|---|---:|---:|---:|---:|---|
| `interhub_sigma01_hw4_timeseries.parquet` | 5,197,072 | 4,981,984 | 215,088 | 0 | parquet metadata `num_rows`; columns `attempt_status`, `product_row_key`, `measurement_role`; `bounded_report.md` 18-24 |
| `onsite_dense_timeseries.parquet` | 281,268 | 2,974 | 4,272 | 274,022 | same columns; `bounded_report.md` 18-24 |
| `rq009_feature_matrix.parquet` | 8,994,736 | 8,994,736 | 0 | 0 | same columns; `bounded_report.md` 18-24 |
| `wod_rq010b_full479_audited.parquet` | 906 | 906 | 0 | 0 | same columns; `bounded_report.md` 18-24 |
| 合计 | 14,473,982 | 13,980,600 | 219,360 | 274,022 | `bounded_report.md` 18-24 |

我用 parquet 的 `product_row_key` 与 `measurement_role` 两列检查了唯一性：每份 parquet 中 `product_row_key × measurement_role` 的 distinct 数等于该文件行数，重复单元数为 0。这个结果说明，现有 L1 ledger 的一行就是一个可对齐的“产品行键 × measurement_role”测量行。

门本身只有在已有单帧候选轨迹、能得到 `mse_per_candidate[7]` 与 `log_score[7]` 时才可调用。因此实际需要重算候选量的最小分母不是全部 14,473,982 行，而是现有 `attempt_status == ATTEMPTED` 的 **13,980,600** 行。`NOT_ATTEMPTED` 的 219,360 行与 `UNKNOWN` 的 274,022 行不是门输入；它们必须在新产物中保留上游状态，不能硬塞成 `NO_IPV_EFFECT` 或 `NEAR_UNIFORM`。OnSite 的 `UNKNOWN` 主体来自 `EMPTY_CELL_UNEXPLAINED`，见 `bounded_report.md` 第 26-28 行。

### 1.2 `mse_per_candidate[7]` 与 `log_score[7]` 是否已有全量落盘

结论：**没有发现任何可替代全语料重算的全量产物。**

已查路径与结果：

| 路径或产物 | 查法 | 结论 |
|---|---|---|
| 4 份 `concentration_ledger/*.parquet` | parquet schema metadata | 列只有 `artifact_id`、`product_row_key`、`measurement_role`、`case_id`、`rq007_split`、`ipv_error`、`K`、`candidate_grid_id`、`k_eff`、`q_eff`、`attempt_status`、`reason_code`、`recoverability`、`ledger_schema_version`、`aggregation_perspective`、`aggregation_configuration`；没有 `mse_per_candidate[7]`、`log_score[7]`、`w_log[7]`、`ipv_log`、`max_w_log`、`mse_spread`。 |
| RQ015A run 目录 `.../RQ015A_1_concentration_audit_.../` | 限定 `rg 'mse_per_candidate|log_score|w_log|ipv_log|max_w_log|mse_spread'` | 无命中；该 run 的文本产物也没有这些字段。 |
| `.codex-fleet/rq015b-repair/work/anchor_mse.csv` | CSV header + row count | 有 `mse_per_candidate[7]`、`w_log[7]`、`ipv_log` 等列，但只有 2,300 行锚点样本，不是全语料。 |
| `.codex-fleet/rq015g-hpc-resolve/work/anchor_mse_hpc.csv` | CSV header + row count | 同上，2,300 行 HPC 重解锚点样本；可作验证锚点，不能替代全量。 |
| `.codex-fleet/rq015f-estimability-contract/work/q4b_anchor_joined.csv` | CSV header + row count | 2,300 行锚点 join 表，含样本级候选量，不是全语料。 |
| `.codex-fleet/rq015h-abstain-gate/work/w_log_consistency_sample.csv` | CSV header + row count | 10 行一致性样本，列为 `ipv_log_from_w`、`k_eff_from_w` 等，不含全量候选 MSE。 |
| `.codex-fleet/rq015c-drift-forensics/work/gate_legacy_vs_current.csv` | CSV header + row count | 40 行漂移样本，不含全量候选 MSE/log-score。 |
| `.codex-fleet/rq015j-gate-spec/work/*.json` | JSON top-level keys | 证据 JSON 记录设计基估计和 gate 定义，不是逐行候选量存储。 |
| `src/sociality_estimation/core/reliability_logdomain.py` | 代码搜索 | 有计算函数，不是存储产物。`candidate_mse` 在第 167-169 行；`weights_from_mse` 在第 172-188 行。 |
| `src/sociality_estimation/core/agent.py` | 代码搜索 | legacy 似然在第 1078-1118 行计算 `var` 并归一；没有持久化 MSE/log-score。 |
| `src/sociality_estimation/core/ipv_estimation.py` | 代码搜索 | diagnostics 在第 340-367 行只保存 `virtual_tracks`、legacy `weights`、`ipv_range`、`ipv`、`ipv_error`；没有 MSE/log-score。 |
| `pipelines/interhub/process_interhub.py` | 代码搜索 | 第 1168-1175 行调用 `estimate_ipv_pair` 且未启用 diagnostics；第 854-885 行写出的 xlsx 只有 legacy IPV、error 和运动列。 |
| `data/derived/interhub/RQ009.../03_features/parity/**/ipv_results.xlsx` | 文件名搜索 + 抽查 3 个 xlsx header | 共 48 个 parity xlsx，列为 `timestamp`、两个 agent 的 legacy IPV/error 和运动列；没有候选数组。 |
| `data/` 文件名搜索 | `find data -type f` 按 `*mse*`、`*log*score*`、`*w_log*`、`*candidate*score*`、`*anchor_mse*`、`*ipv_results*` | 未发现全量候选 MSE/log-score 命名产物；命中项主要是 RQ009 parity 的 `ipv_results.xlsx`。 |
| 非 `reports/` 仓库文件名搜索（排除 `.git`、`data`、K reviews） | `find` 按同一组名称 | 命中样本级 `anchor_mse*.csv`、少量检查样本和大量 legacy/archived `*_ipv_results.xlsx`；没有全语料 K 门候选量台账。 |

所以 Q1 的核心事实是：现有 2,300 锚点产物能降低“阈值设计、验收锚点、设计基估计”的成本，但不能替代 K2 所需的逐行全量门后台账。

### 1.3 如果必须重算，最小重算是什么

最小重算不是重新设计门，也不是重跑旧 RQ009。最小重算是：

1. 对现有 L1 ledger 中 `attempt_status == ATTEMPTED` 的 **13,980,600** 行，重新得到同序的 7 个候选轨迹或候选残差。
2. 计算并物化 `mse_per_candidate[7]`、`log_score[7] = -mse_i/(2*sigma^2)`、`w_log[7]`、`max_w_log`、`mse_spread`、`k_eff_log`、`ipv_log`、门状态与互斥 reason。
3. 对 219,360 个 `NOT_ATTEMPTED` 行和 274,022 个 `UNKNOWN` 行，只保留上游不可判门状态，不调用门，不把它们写成 `NO_IPV_EFFECT` 或 `NEAR_UNIFORM`。

可复用代码路径：

| 目的 | 路径与证据 |
|---|---|
| 候选 IPV 网格 | `src/sociality_estimation/core/agent.py` 第 63-64 行定义 7 候选网格和 5 候选 realtime 网格；本门要求 `K=7`。 |
| 候选求解 | `src/sociality_estimation/core/agent.py` 第 209-231 行构造候选任务，第 173-200 行求解候选轨迹。 |
| pair/time 循环 | `src/sociality_estimation/core/ipv_estimation.py` 第 181-372 行，现有 `estimate_ipv_pair` 已按时间步和双角色循环。 |
| MSE 与 log 权重计算 | `src/sociality_estimation/core/reliability_logdomain.py` 第 167-188 行已提供 `candidate_mse` 和稳定 softmax。 |
| InterHub case 载入、对齐、下采样 | `pipelines/interhub/process_interhub.py` 第 1049-1175 行。现有调用只写 legacy 输出；K 可以新增 materializer 或 wrapper，不必改旧 estimator 算法。 |
| 与现有 L1 ledger 对齐 | `scripts/rq015a/build_ledger.py` 中 `product_row_key` 和 aggregation key 相关路径，搜索命中第 1022-1076 行；`scripts/rq015a/run_rq015a.py` 第 495-528 行显示当前 parquet writer schema 不含 K 门列。 |

最小实现可以是不改 `agent.py` / `ipv_estimation.py` / `process_interhub.py` / `reliability_logdomain.py` 的新增 K materializer：调用现有估计器取得 diagnostics 或候选轨迹，再用 `reliability_logdomain.py` 计算 MSE/log 权重并写新 parquet。若实现者选择在旧 pipeline 里“加列”，那就会触及受保护文件；计划必须明确允许新脚本路线，避免把“不要改计算”误读成“无需实现新的持久化路径”。

成本量级只能先做线性下界估算，不能替代 K1 实测。G 轨可用三种口径：

| 口径 | 来源 | 数字 |
|---|---|---:|
| final Slurm wall time | `G_leader_adjudication.md` 第 5 行；`G1_hpc_resolve_report.md` 第 5、20 行 | 2,300 锚点，6 worker，00:14:22 = 862 秒 |
| G work JSON compute elapsed | `.codex-fleet/rq015g-hpc-resolve/work/g1_hpc_summary.json` 第 650-655 行 | 2,300 锚点，702.993677 秒，0.305649 秒/anchor wall |
| OOM 风险 | `G_leader_adjudication.md` 第 249、252-255 行 | 24 worker 曾因每进程载一份 PKL OOM，最终 6 worker 完成 |

用 13,980,600 个 `ATTEMPTED` 行线性外推：

| 外推基准 | 分子/分母 | 6-worker wall time | 6-worker worker-hours |
|---|---:|---:|---:|
| G JSON 702.993677 秒 / 2,300 anchor | 13,980,600 / 2,300 = 6,078.5217 倍 | 4,273,162 秒 = 49.46 天 | 7,121.94 |
| Slurm wall 862 秒 / 2,300 anchor | 同上 | 5,239,686 秒 = 60.64 天 | 8,732.81 |

这只是量级说明。真实 K1 必须按 case/PKL 复用、轨迹长度、source、artifact 和 status 分层实测；不能用 2,300 锚点均值直接投 K2。

### 1.4 产出应长什么样，才能让 RQ009 直接用

RQ009 计划要求当前 rolling IPV、有效 ego/counterpart 当前 IPV、冻结门、支持率和总弃权 reason 分布，见 `reports/plans/RQ009_plan_v0_dynamic_counterpart_conditioned_envelope_20260624.md` 第 83-93 行和第 189-201 行。RQ009 决策层已经接受 context-conditioned envelope，并要求下游不要期待独立 counterpart channel，见 `decision.md` 第 33-37 行；同时边界里有 exact-zero atom，见 `decision.md` 第 21-25 行。

因此 K 产物至少应是一个可 join 的 parquet package，而不是只给汇总表：

1. **L1 row-level gate parquet**：一行对应现有 `artifact_id × product_row_key × measurement_role`。保留 `case_id`、`candidate_grid_id`、`K`、`source_attempt_status`、`source_reason_code`、legacy `ipv_error/k_eff/q_eff` 以便审计。
2. **门输入与输出列**：`candidate_ipv_0..6`、`mse_0..6`、`log_score_0..6`、`w_log_0..6`、`max_w_log`、`mse_spread`、`k_eff_log`、`status`、`reason_code`、`ipv_log`。如果用 list/JSON 数组列，也必须同时记录候选顺序和 schema version；为了 pandas/RQ009 直接 join，标量列更稳。
3. **非门输入行处理**：对 `NOT_ATTEMPTED/UNKNOWN` 行应有 `gate_applicable=false`，门字段为 null，并用上游状态解释；不能伪装成门弃权 reason。
4. **RQ009 join 字段**：对 `rq009_feature_matrix` 至少解析并保存 `source_dataset`、`anchor_frame_index`、`perspective`、`measurement_role`、`context_cell_key` 或其生成输入；如果 `context_cell_key` 只能在 RQ009 现有 feature matrix 里生成，K1 应先做 join-key dry run。
5. **聚合表**：按 `context_cell_key`、case/scene、ego/counterpart 角色和 source 给出 pass rate、reason counts、分母、缺失/不可判门计数。`gate_pass_rate` 是聚合字段，不应只塞进每一条原始 L1 行。
6. **接口警告**：`status` 与 `reason_code` 是弃权唯一判别字段；`ipv_log == 0` 可以是合法 OK 值，不能反推缺失或弃权。J1 报告第 155-172 行给出门后 `ipv_log` exact-zero 的口径。

## 2. 第二阶段：对照 K kickoff

### 2.1 一致项

K kickoff 的主方向是对的：

| K 计划要求 | 我的第一阶段结论 | 对照 |
|---|---|---|
| 现有 4 份 parquet 合计 14,473,982 行，但缺 `mse_per_candidate[7]` 和 `w_log[7]`，不能逐行判门 | schema 检查和限定搜索得到同一结论 | 一致。K kickoff 第 23-26 行正确。 |
| K1 先查计算单元和现成逐候选产物 | 第一阶段认为这是最有价值的节省检查 | 一致。K kickoff 第 34-41 行正确。 |
| K1 要测小批单位成本和内存，不直接投 K2 | 第一阶段成本外推显示直接 K2 风险很高 | 一致。K kickoff 第 42-47 行方向正确。 |
| 台账至少包含 MSE、log 权重、门状态、reason、`ipv_log`、key 和 RQ009 对接字段 | 第一阶段产物形状也要求这些字段 | 一致。K kickoff 第 101-106 行正确。 |
| 不改门规格、不调 theta、不把两条 reason 相加 | 第一阶段没有推导出任何重设阈值的必要 | 一致。K kickoff 第 63-85 行正确。 |

### 2.2 计划遗漏

1. **遗漏了 `NOT_ATTEMPTED/UNKNOWN` 的门外状态契约。**  
   第一阶段发现，14,473,982 行中只有 13,980,600 行是 `ATTEMPTED`；另有 219,360 行 `NOT_ATTEMPTED`、274,022 行 `UNKNOWN`。来源：`bounded_report.md` 第 18-24 行，列为 `ATTEMPTED`、`NOT_ATTEMPTED`、`UNKNOWN`。K kickoff 第 101-106 行列了门字段，但没有规定非门输入行如何落盘。若 K2 把这些行写成 `ABSTAIN`，会把“没有输入”混成“输入可算但门弃权”，直接污染 RQ009 的弃权分布。

2. **遗漏了按 artifact 的 raw recomputability 检查。**  
   K kickoff 第 34-36 行要求确定计算单元，但没有明确要求每个 artifact 逐一证明可以从现有 raw/snapshot 数据重建候选轨迹。InterHub 路径较清楚；OnSite 和 WOD full479 是否有足够原始轨迹与候选重算入口，需要在 K1 明确列出。RQ015A 报告还提醒 WOD 只覆盖 full479 的 906 行，不能写成全 WOD，见 `bounded_report.md` 第 14-16 行和第 155-158 行。

3. **遗漏了 RQ009 join-key dry run 和 schema dry run。**  
   计划写了 `context_cell_key`，但 K1 没有要求先证明 `rq009_feature_matrix` 的 `product_row_key` 能无歧义解析并 join 到 RQ009 context cell。RQ009 的 primary gate 依赖可 join 的当前帧特征和支持率，见 RQ009 plan 第 83-93 行、第 189-201 行。

4. **遗漏了输出存储格式与验收不变量。**  
   对千万行、每行 7+7+7 个候选相关数值，必须在 K1 决定 parquet partition、array vs scalar columns、schema version、row-count conservation、nullability、候选顺序、SHA manifest 和断点续算规则。K kickoff 第 46-47 行提到断点续算，但没有把 schema 和 row-count 验收作为 K2 投放前的 stop gate。

5. **遗漏了小批实测必须分层。**  
   K kickoff 第 42-44 行建议 2,000-5,000 单元。这个数量本身可以，但必须分层覆盖 artifact、source、measurement_role、trajectory length、PKL reuse pattern、OnSite `UNKNOWN` 和 WOD full479；否则测到的是某一类 InterHub case 的成本，不足以预测 K2。

### 2.3 计划冗余或可降级项

1. K kickoff 第 48-55 行要求查 RQ009 的 `counterpart_ipv_current`、`counterpart_ipv_error_current`、`counterpart_ipv_slope_pre_anchor` 缺失填充值。这个检查对下游解释有价值，但它不是决定“是否需要 K2”的必要条件。建议保留为 K1 附属检查，不应拖住资源方案或 full-corpus gate ledger 的可执行判定。

2. K kickoff 第 101-106 行把 `gate_pass_rate` 与 row-level 字段放在同一处。`gate_pass_rate` 是聚合结果，不是单帧原始门字段。建议拆成 L1 row parquet 和 aggregation parquet，避免 downstream 把重复的聚合率当作行级变量。

### 2.4 冲突或计划错了

1. **计划错了：G 轨耗时 `500.6s` 没有在我检查的 G 正文和 JSON 中得到支持。**  
   K kickoff 第 44 行写“G 轨 2,300 锚点、6 worker、500.6s 完成求解”。我查到的来源是：
   - `G_leader_adjudication.md` 第 5 行：Slurm `2023332`，6 worker，elapsed `00:14:22`，即 862 秒。
   - `G1_hpc_resolve_report.md` 第 5、20 行：同一 job elapsed `00:14:22`。
   - `g1_hpc_summary.json` 第 650-655 行：`elapsed_seconds=702.9936774782836`，`seconds_per_anchor_wall=0.3056494249905581`。
   这三者都不是 500.6s。资源估算应改用 702.993677 秒的 compute elapsed 或 862 秒的 Slurm wall，并说明口径。

2. **计划有过度表述风险：“全语料”必须限定为这 4 份本地可审计 parquet。**  
   K kickoff 第 1 行和第 23 行使用“全语料”。但 RQ015A `bounded_report.md` 第 14-16 行和第 155-158 行说明本轮覆盖 4/6 产物，WOD 只含 full479 906 行，另外 WOD/RQ014 分支不在 ledger。若 K2 报告写成无条件“全 WOD”或“全项目全语料”，计划错了。正确写法应是“对 RQ015A 当前 4 份 L1 parquet ledger 的全行物化”。

3. **计划第 131-133 行“只增加持久化输出列，不改任何计算”容易被误读。**  
   第一阶段代码检查显示，现有 `process_interhub.py` 第 1168-1175 行不请求 diagnostics，第 854-885 行只写 legacy IPV/error；现有 `ipv_estimation.py` 第 340-367 行 diagnostics 也不直接写 MSE/log-score。K2 至少需要新增 materializer 或新增 writer。可以不改四个受保护文件，但不能假设“无实现工作”。

## 3. 共同收尾问题

### Q1. 这轮重算是否必要？

**必要，但只对需要逐行普查式 K 门台账的目标必要。**  
如果目标只是报告全域影响估计，现成 2,300 锚点的 design-based estimate 已可用；J1 报告第 113-127 行给出 HT 分母 2,646,058、保留权重 1,885,831.096、可估率 71.2695% 和 reason 权重。若目标是让 RQ009 直接逐行 join 并按任意 context cell 展开，现有产物不能替代 K2。

查过的路径：

- `reports/studies/RQ015A.../concentration_ledger/*.parquet`：无 `mse_per_candidate[7]`、`log_score[7]`、`w_log[7]`。
- `reports/studies/RQ015A.../bounded_report.md`、`run_receipt.json`、`concentration_ledger_summary.csv`：只有 q_eff/attempt/status 描述，无候选数组。
- `.codex-fleet/rq015b-repair/work/anchor_mse.csv`：2,300 锚点样本，有候选数组，不能替代全量。
- `.codex-fleet/rq015g-hpc-resolve/work/anchor_mse_hpc.csv`：2,300 HPC 锚点样本，有候选数组，不能替代全量。
- `.codex-fleet/rq015f-estimability-contract/work/q4b_anchor_joined.csv`：2,300 样本 join，不能替代全量。
- `.codex-fleet/rq015h-abstain-gate/work/w_log_consistency_sample.csv`：10 行一致性样本，不能替代全量。
- `.codex-fleet/rq015c-drift-forensics/work/gate_legacy_vs_current.csv`：40 行漂移样本，不能替代全量。
- `.codex-fleet/rq015j-gate-spec/work/*.json`：证据 JSON，不是逐行候选量。
- `src/sociality_estimation/core/*`、`pipelines/interhub/process_interhub.py`、`scripts/rq015a/*`：有计算和 ledger 代码，没有全量候选量存储。
- `data/derived/interhub/RQ009.../03_features/parity/**/ipv_results.xlsx`：48 个 parity xlsx，抽查列名只有 legacy IPV/error 和运动列。
- `data/` 文件名搜索：未发现全量候选 MSE/log-score 命名产物。

能大幅降低成本的现成产物：没有。能降低风险的现成产物：2,300 锚点的 `anchor_mse_hpc.csv` 可作为 K2 抽样验收锚点；J1 design-based estimate 可作为 K2 普查结果的解释参照，但不能替代 K2。

### Q2. K1 的勘察范围是否足以支撑「要不要投 K2」？

**方向足够，范围需补齐后才足够。**

已覆盖且必要：

- 计算单元和数量，见 K kickoff 第 34-36 行。
- 查现成逐候选产物，见第 37-41 行。
- HPC 小批成本和内存，见第 42-45 行。
- 资源方案和断点续算，见第 46-47 行。

缺少：

- 非门输入行的状态契约：219,360 个 `NOT_ATTEMPTED` 和 274,022 个 `UNKNOWN` 的输出规则。
- 每个 artifact 的 raw recomputability 证明，尤其 OnSite 与 WOD full479。
- RQ009 join-key dry run 和 `context_cell_key` 生成/对齐规则。
- 分层 pilot 设计，而不是泛泛 2,000-5,000 单元。
- schema、partition、row conservation、nullability、candidate order、manifest 的 K2 前置验收。
- 修正 G 轨耗时口径：500.6s 改为 702.993677s compute elapsed 或 862s Slurm wall。

多了但可保留：

- RQ009 counterpart 三列填充值检查。它不是 K2 投放必要条件，但可作为下游接口风险检查保留；不要让它扩大成 RQ009 重算。

### Q3. 明确判定

判定：**需修改后执行**。

理由：

- 计划的核心判断正确：现有 4 份 ledger 缺少全量候选 MSE/log 权重，逐行 K 门台账无法由现有产物替代。
- K1 先查现成产物、计算单元、成本和资源方案是正确顺序。
- 但计划现在缺少非门输入行契约、artifact raw recomputability、RQ009 join dry run、schema/partition 验收和分层 pilot；并且 G 轨 `500.6s` 成本口径无来源。

最可能造成实际损失的一处：

**把 `NOT_ATTEMPTED/UNKNOWN` 行或无法重建候选量的 artifact 行误写成门弃权。**  
这会把“上游没有有效门输入”混成 `NO_IPV_EFFECT` 或 `NEAR_UNIFORM`，直接污染 RQ009 的 abstention reason distribution 和 context cell pass rate。风险来源是 K kickoff 第 101-106 行只列门字段，未规定非门输入行输出契约；对应分母是 219,360 个 `NOT_ATTEMPTED` 行和 274,022 个 `UNKNOWN` 行，来源为 `bounded_report.md` 第 18-24 行的 `NOT_ATTEMPTED`、`UNKNOWN` 列。

## 4. 建议修改清单

1. 在 K1 任务中新增“门适用性契约”：只对 `attempt_status == ATTEMPTED` 行计算 K 门；其他行 `gate_applicable=false`，门字段 null，并保留 `source_attempt_status/source_reason_code`。
2. 把 K1 pilot 改成 stratified pilot：按 artifact、source、measurement_role、trajectory length、PKL group、OnSite/WOD 覆盖抽样，报告每层 wall time、worker RSS、PKL reuse、failure rate。
3. 修正 G 成本参照：写 702.993677 秒 compute elapsed 与 862 秒 Slurm wall 两个口径，删除 500.6s 或补其来源。
4. 增加 RQ009 join dry run：对 `rq009_feature_matrix` 抽样解析 `product_row_key`，证明 `context_cell_key` 或其生成输入一对一可得。
5. 增加 K2 前置 schema gate：row count conservation、候选顺序、nullability、schema version、parquet partition、SHA manifest、断点续算和重复投递幂等性。
6. 把“全语料”在标题或摘要中限定为“RQ015A 当前 4 份本地可审计 L1 parquet ledger”，不得扩写为全 WOD 或 RQ014。
