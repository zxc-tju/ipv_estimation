# RQ016D-RA review

## **Q1.** 这个 materializer 应当复用哪些现成组件？逐个给出文件路径与行号，并说明它在链路中承担什么。

本轮要补的是 OnSite 67,861 个自动驾驶车 anchor 的机制一判据；应当复用已冻结的求解核心、候选网格、log 域权重、门判据和 K2 L1 落盘合同，但不能直接复用 InterHub 的 `run_b2_rq015b.py::solve_anchor_task` 作为 OnSite 入口。

应复用或镜像的组件如下。

- `configs/ipv_sigma01_exact.json:5-15`：冻结 `solver_mode=exact`、`sigma=0.1`、`min_observation=4`、参考线裁剪/平滑参数、当前窗口 10、目标窗口 4、目标 offset 6。
- `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:53-59`：冻结 schema version、`SIGMA=0.1`、`THETA=0.20`、`K=7`、`candidate_grid_id=legacy7_pi_over_8`、7 点候选网格。
- `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:649-689`：`gate_from_mse`，从 7 个 MSE 生成 `log_score`、`w_log`、`max_w_log`、`mse_spread`、`k_eff_log`、`status`、`reason_code`、`ipv_log`。
- `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:780-847`：K2 L1 scalar schema、数组展平、atomic parquet 写法。这个可以规避 list/null 写出问题。
- `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:891-901`：多 worker 求解、按 `sample_order` 复原顺序、再生成 L1 行的执行模式，可作为 OnSite shard runner 的结构模板。
- `.codex-fleet/rq015b-repair/work/run_b2_rq015b.py:188-190`：`loads_array`，用于读取旧 CSV 中的 7 候选数组文本。
- `.codex-fleet/rq015b-repair/work/run_b2_rq015b.py:268-284`：InterHub `solve_anchor_task` 如何从 diagnostic 取 observed/virtual tracks 并计算 7 候选 MSE。OnSite 不能直接调用这个入口，因为同文件 `271-272` 会拒绝非 `development/guard` split。
- `.codex-fleet/rq015b-repair/work/run_b1_rq015b.py:606-670`：InterHub `build_sequences` 的等价性模板：事件加载、双车时间对齐、数据集降采样、参考线构建/裁剪/平滑、heading label、`MotionSequence`。OnSite 应写同等职责的版本。
- `.codex-fleet/rq015b-repair/work/run_b1_rq015b.py:742-784`：`solve_current_details` 展示当前代码如何用 `Agent.estimate_self_ipv(... solver_mode="exact", candidate_ipv_values=IPV_GRID)` 取 observed/virtual tracks 并计算 MSE。
- `src/sociality_estimation/core/ipv_estimation.py:37-55`：`MotionSequence` 的输入合同，`data` 至少为 `[x,y,vx,vy,heading]`。
- `src/sociality_estimation/core/ipv_estimation.py:181-194` 与 `247-328`：`estimate_ipv_pair` 的 exact 调用面、窗口、diagnostics、candidate grid 传递路径。
- `src/sociality_estimation/core/agent.py:63-68`、`156-161`：exact 默认 7 点网格；realtime 才是 5 点网格。
- `src/sociality_estimation/core/reliability_logdomain.py:172-188`：`weights_from_mse`，稳定 softmax，遇非有限 MSE 抛输入错误。
- `pipelines/interhub/process_interhub.py:375-410`、`423-473`、`529-580`、`583-599`：参考线 fallback、裁剪/采样/平滑、双车对齐、数据集降采样、heading label 规则。OnSite 不能照搬 pkl loader，但这些规则是等价性边界。
- `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/hpc_onsite_ipv/build_onsite_m3_anchors_hpc.py:705-735`、`778-833`：历史 OnSite all-valid builder 的轨迹列、观测轨迹参考线、当前窗口 10 与目标窗口 4 at `t*+6` 的实现线索。只能借鉴轨迹构造，不能复用旧 vendored 求解器本体。

## **Q2.** 求解链路上最容易被做错、且做错后**不会自己暴露**的是哪一步？说明它错了会产生什么样的结果，以及为什么现有检查抓不到。

最危险的是 OnSite 行到求解单元的测量合同：`product_row_key`、自动驾驶车/交互对象角色、anchor 帧还是 `target_window_end_frame_index`、以及窗口 10/4 的对应关系。任务书要求最终与 RQ016C 支持门交叉，RQ016C dry-run 的主键是 `product_row_key`，我实测 anchor 构造键与 dry-run 一对一：67,861/67,861 交集，筛选条件为两表全部行，来源 `.codex-fleet/rq016d-review-a/work/rq016d_ra_readonly_checks.json:261-263`，列为 anchor 构造键与 dry-run `product_row_key`；但任务书自查没有要求 materializer 输出也满足这个一对一合同。

如果执行方在局部帧位置、全局 `frame_index`、`anchor_frame_index`、`target_window_end_frame_index`、或 AV/HV 角色上偏一层，仍会产生 7 个有限 MSE，`gate_from_mse` 可复算、`ipv_log` 恒等式也会通过，状态分布也可能看起来正常。现有自查 `RQ016D_M1_kickoff.md:199-213` 主要验证落盘判据内部一致性，不能证明判据对应的是 RQ016C 那一行、那一方、那一个目标时间点。

## **Q3.** `run_b2_rq015b.py` 的 `ALLOWED_SPLITS` 护栏在 OnSite 场景下应当怎么处理？给出你的具体做法，并说明它为什么不会削弱 InterHub 那条路径的保护。

具体做法：保持 `.codex-fleet/rq015b-repair/work/run_b2_rq015b.py:72` 与 `271-272` 一字不动，不把 OnSite 行传进 `run_b2_rq015b.py::solve_anchor_task`。在 RQ016D 自己的 `work/M1/` 下写 `solve_onsite_anchor_task`，只接受 `source_dataset == "onsite_competition_clean_285"`、`perspective == "onsite_av_primary"`、`product_row_key` 可由 anchor 表字段重建、且输入列名中没有 `rq007` 或 `held` 标记；本次实测 OnSite anchor 表没有这类列，来源 `.codex-fleet/rq016d-review-a/work/rq016d_ra_readonly_checks.json:119-121`。

这不会削弱 InterHub 路径，因为 InterHub 的 split 护栏仍在原文件原函数内生效，K2 的 held token 检查也保留在 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:74` 与 `1306-1334`。OnSite 是独立入口，不能用“dataset 不在 RQ007 体系里”作为放宽 InterHub split 的理由。

## **Q4.** canary 必须覆盖哪些路径才算有效？逐条列出，并说明每条对应的真实失败模式。

有效 canary 至少要覆盖这些路径。

1. 真实 OnSite 正常 anchor：从 `onsite_ipv_timeseries_multi_allvalid.parquet` 读取 ego/counterpart `x,y,vx,vy,heading` 到 `MotionSequence`，跑完整求解和门判据，覆盖轨迹列选择、角色、帧定位、参考线构造。
2. 真实 OnSite 异常坐标 anchor：7 行 `relative_distance_anchor > 100000` 必须至少抽一行，筛选条件为 anchor 全表、列 `relative_distance_anchor`，实测 7/67,861 = 0.0103%，来源 `.codex-fleet/rq016d-review-a/work/rq016d_ra_readonly_checks.json:122-123`；覆盖坐标系异常进入求解后的工程失败隔离。
3. 真实 `OK` 与真实 `NEAR_UNIFORM`：若自然 canary 找不到，先小批只读探测后再固定样本；覆盖求解输出到 `gate_from_mse` 的正常路径。
4. 合成等 MSE 行：不混入正式产物，只用于触发 `NO_IPV_EFFECT`，覆盖 `mse_spread == 0.0` 的精确顺序。
5. 合成非有限输入行：不混入正式产物，只用于触发工程失败，覆盖工程失败不得写成科学 reason。
6. 至少 2 个 worker 并发、每个 worker 独立 cache 目录：覆盖 Matplotlib cache 并发锁与多进程 import/pickle 失败；K2 对 per-process cache 的做法在 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:175-184`。
7. parquet 写出后读回：覆盖 null 标量列、schema、排序和分区路径。K2 的展平 scalar schema 与 atomic 写法见 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:780-847`。
8. 输入 hash 只按文件或 shard 计算一次：覆盖逐行重算 SHA 导致吞吐崩掉的问题。K2 在 manifest 阶段和 shard 结束阶段计算 hash，见 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:421-435` 与 `899-918`。

## **Q5.** 有哪些验收判据**必须存在**，否则错误会静默通过？逐条列出，并说明缺了它会漏掉什么。

必须存在的验收判据如下。

1. `product_row_key` 一对一：materializer 输出、OnSite anchor 表、RQ016C dry-run 三者各 67,861 个唯一键，互相 miss=0、duplicate=0。缺了会把机制一判据交叉到错误的机制二行。
2. 测量角色与时间点断言：每行必须明确 `measurement_role`、subject 是 AV、counterpart 是 HV、求解帧是 anchor 还是 `target_window_end_frame_index`，并与 `future_target_history_window=4` / offset 6 合同一致。缺了会把当前值当成目标值，或把 HV 的判据当成 AV 的判据。
3. `MotionSequence` 重建审计：抽样行逐项核对输入 window 中的 `[x,y,vx,vy,heading]` 与 dense timeseries 相等，参考线唯一点数、最多 40 点、平滑 40 点。缺了会漏掉局部帧位置偏移、排序错、单位错、角色交换。
4. OnSite 来源 allowlist：只读 motion/key 字段，不读 outcome、评分、人工标注或 RQ014 相关字段；输入列名检查遇 `rq007`/`held` 标记直接 fail。缺了会把安全边界交给执行者口头遵守。
5. K、grid、sigma、theta、exact mode 写入并复核：`K=7`、`candidate_grid_id=legacy7_pi_over_8`、`sigma=0.1`、`theta=0.20`、`solver_mode=exact`。缺了会漏掉错用 realtime 5 点网格或调阈值。
6. 门判据 replay：从落盘 `mse_0..6` 重算 `status`/`reason_code` 零差异。缺了会漏掉门顺序或阈值实现错误。
7. 负对照必须由 sentinel 保证失败：构造 `0 < mse_spread <= 1e-12` 的 synthetic 行验证 `np.isclose` 版本会 fail；构造 `max_w_log` 落在 `[0.20,0.22)` 的 synthetic 行验证 `theta=0.22` 会 fail。缺了则自然 OnSite 分布可能没有敏感行，负对照会给出假通过。
8. 工程失败隔离：`failure_type != OK` 的行不得有 `reason_code in {NO_IPV_EFFECT, NEAR_UNIFORM}`，且数值列 null pattern 与 K2 schema 一致。缺了会把输入/求解失败伪装成科学 reason。
9. 7 行异常坐标 inclusion：这 7 行必须在正式输出中存在，且单独给出 status 分布。缺了会漏掉静默剔除。
10. 不覆盖发布门：若 `data/derived/rq016d_onsite_gate/l1_v1/` 已存在，默认 fail 或写新版本；正式发布前 manifest 标记 `finalized=true`。缺了会污染下游读取路径。

## **Q6.** 这一轮有没有会污染下游、或不可逆的风险？逐条列出并给出规避办法。

有。

1. 错帧或错角色输出会污染下一轮串联判定。规避：开跑前冻结 `measurement_role`、subject、frame contract，并把三表 `product_row_key` 一对一作为 blocker。
2. 写入固定 `data/derived/rq016d_onsite_gate/l1_v1/` 后被下游误读。规避：canary 与验证先写 `work/M1/canary/`，正式产物 atomic publish，已有目录不覆盖。
3. 用旧 285 行 exact HW10 通道会带错目标窗口且无 MSE。证据：`.codex-fleet/rq016b-wod-onsite-feasibility/work/F1/render_rq016b_report.py:104` 写明其 `target_history_window=10` 且无 per-candidate MSE/log weights。规避：脚本路径 denylist，输入只接受 all-valid 67,861 anchor 与 70,317 dense rows。
4. 直接改 `run_b2_rq015b.py` split 护栏会污染 InterHub 保护。规避：OnSite 独立入口，原文件只读。
5. 坐标异常被预过滤会改变分母。规避：7 行照常进入正式输入，工程失败按工程失败记。
6. 读取非必要 feature/outcome/scoring 字段会触碰安全边界。规避：materializer 用列 allowlist，只读 motion/key/source/window 字段。

## **Q7.** 若你只能提**一条**改动意见，是哪一条？为什么是它而不是别的。

只提一条：在 canary 前新增“OnSite measurement contract”小节，把 `product_row_key` 构造、subject/counterpart、求解帧、窗口、参考线生成、与 RQ016C dry-run 的一对一 join 全部写成可失败断言。理由是门判据、schema、并发 canary 都是在正确测量对象之后才有意义；如果测量合同错了，现有检查仍可能全部通过。

## **Q8.** 这一轮该不该开跑？只答 `GO` / `GO_WITH_CHANGES` / `NO_GO` 三者之一，并给出不超过三句话的理由。若答 `GO_WITH_CHANGES`，把必须先改的条目按重要性排序列出。

`GO_WITH_CHANGES`。主求解与门判据复用方向正确，且我实测关键输入分母与既有 K2/RQ016C 数字吻合；但任务书还没有把 OnSite 行映射到哪一方、哪一帧、哪一个 `product_row_key` 写成会 fail 的合同。必须先改：1. 新增 measurement contract 与三表一对一验收；2. 明确 OnSite 独立入口而不调用 B2 split 入口；3. 把负对照改成 synthetic sentinel 保证 fail；4. canary 纳入真实异常坐标行与 parquet 读回。

## 详细审查依据

### 1. 方案定位与复审范围

任务书的定位是正确的：RQ015 已冻结机制一，RQ016C 已完成机制二支持门 dry-run，本轮只补 OnSite 的机制一判据，不做下一轮的区间比较。依据是任务书 `.codex-fleet/rq016d-onsite-materializer/board/RQ016D_M1_kickoff.md:20-29`。

本次复审为只读。只运行了 `.codex-fleet/rq016d-review-a/work/rq016d_ra_readonly_checks.py`，它读取指定 parquet/json 并写出 `.codex-fleet/rq016d-review-a/work/rq016d_ra_readonly_checks.json`；没有运行 IPV 求解器、没有 Slurm/HPC、没有训练、没有写 `data/derived/`。

### 2. 分母与既有数字抽查

抽查 1：OnSite anchor 表为 67,861 行 × 66 列，`av_included == "AV"` 为 67,861/67,861 = 100.0000%，筛选条件为 anchor 全表，来源 `.codex-fleet/rq016d-review-a/work/rq016d_ra_readonly_checks.json:43-46` 与 `197-199`，列为 `av_included`。

抽查 2：OnSite K2 台账 `artifact_id=onsite_dense_timeseries` 为 281,268 行，`gate_applicable` 为 False 的行是 281,268/281,268 = 100.0000%，筛选条件为 K2 OnSite artifact 全表，来源 `.codex-fleet/rq016d-review-a/work/rq016d_ra_readonly_checks.json:18-21`，列为 `gate_applicable`。同一来源显示 `mse_0..6`、`max_w_log`、`mse_spread`、`status`、`reason_code` 非空计数全为 0，见 `22-35`。

抽查 3：OnSite K2 `source_attempt_status` 为 `UNKNOWN` 274,022、`NOT_ATTEMPTED` 4,272、`ATTEMPTED` 2,974，筛选条件为 K2 OnSite artifact 全表，来源 `.codex-fleet/rq016d-review-a/work/rq016d_ra_readonly_checks.json:35-41`，列为 `source_attempt_status`。

抽查 4：InterHub K2 L1 分区共有 5,197,072 行，其中 `status` 非空分母为 4,981,984；在该分母上 `OK` 为 3,502,340/4,981,984 = 70.3001%，`NEAR_UNIFORM` 为 1,457,746/4,981,984 = 29.2604%，`NO_IPV_EFFECT` 为 19,964/4,981,984 = 0.4007%，`SOLVER_FAILURE` 为 1,934/4,981,984 = 0.0388%。筛选条件为 `artifact_id=interhub_sigma01_hw4_timeseries` 且 `status` 非空，来源 `.codex-fleet/rq016d-review-a/work/rq016d_ra_readonly_checks.json:2-17`，列为 `status` 与 `reason_code`。

抽查 5：RQ016C dry-run 为 67,861 行，`mechanism2_gate_ok` 为 True 的行是 21,936/67,861 = 32.3249%，筛选条件为 dry-run 全表，来源 `.codex-fleet/rq016d-review-a/work/rq016d_ra_readonly_checks.json:260-292` 与 `470-473`，列为 `mechanism2_gate_ok`。

抽查 6：OnSite all-valid provenance 记录 `feature_history_window=10`、`target_history_window=4`、`min_observation=4`、`target_final_offset=6`，来源 `.codex-fleet/rq016d-review-a/work/rq016d_ra_readonly_checks.json:208-220`，字段为 `settings.*`。这与 `configs/ipv_sigma01_exact.json:12-15` 一致。

### 3. 求解链路正确性

任务书指认的核心组件大体正确：K2 `ensure_solver_imports` 会把 B work、repo root、`src` 加入 import path，并取 `loads_array`、`solve_anchor_task`、`weights_from_mse`，见 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:161-172`。但这是 InterHub shard 的入口，不是 OnSite 入口。

B2 `solve_anchor_task` 在 `.codex-fleet/rq015b-repair/work/run_b2_rq015b.py:271-272` 先检查 split，OnSite 直接传入会被拒绝；若为了通过而改这里，就是安全退化。OnSite 应只复用 B2 计算 MSE 的结构和 B1 `solve_current_details` 的 exact candidate 细节，而不是复用这个函数入口。

OnSite `MotionSequence` 构造是方案写得最粗的一段。任务书 `.codex-fleet/rq016d-onsite-materializer/board/RQ016D_M1_kickoff.md:94-97` 只说写 OnSite 版 `build_sequences`，但没有列出必须等价的检查。历史 OnSite all-valid builder 显示真实轨迹列是 ego/counterpart `x,y,vx,vy,heading`，并用观测轨迹去重、最多 40 点、平滑 40 点作为参考线，见 `hpc_onsite_ipv/build_onsite_m3_anchors_hpc.py:705-735`。本轮应使用当前 `src` 估计器，而不是历史 builder 的 vendored estimator；历史 builder 在 `714` 导入 vendored estimator。

### 4. 口径一致性

冻结配置一致：`configs/ipv_sigma01_exact.json:5-15` 与 all-valid provenance `.codex-fleet/rq016d-review-a/work/rq016d_ra_readonly_checks.json:208-220` 均为 exact、sigma 0.1、min_observation 4、当前窗口 10、目标窗口 4、target offset 6、参考线 60/40/40。

候选网格一致：K2 常量 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:53-59` 与 current `agent.py:63-68`、`156-161` 都支持 exact 的 7 点网格。验收必须把 K/grid 写到正式产物和回执，否则错用 realtime 5 点路径可能仍产生可计算输出。

门判据一致：`gate_from_mse` 在 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:649-689` 使用 log 域权重、精确 `spread == 0.0`、再判 `max_w < 0.20`，与任务书 `RQ016D_M1_kickoff.md:101-117` 一致。

旧通道不可用：`.codex-fleet/rq016b-wod-onsite-feasibility/work/F1/render_rq016b_report.py:104` 记录 285 行旧通道目标窗口为 10 且没有 per-candidate MSE/log weights。任务书 `RQ016D_M1_kickoff.md:67-68` 的禁用判断正确。

### 5. 安全边界

held-out 边界：任务书 `RQ016D_M1_kickoff.md:160-172` 的方向正确，但需要更具体。OnSite anchor 表没有 `held` 或 `rq007` 字样列，来源 `.codex-fleet/rq016d-review-a/work/rq016d_ra_readonly_checks.json:119-121`；正式脚本仍应在输入列名和输出 `rq007_split` 上做 blocker，而不是只在报告中写断言。

RQ014 相关字段：B2 旧样本入口有 forbidden sample columns 列表，见 `.codex-fleet/rq015b-repair/work/run_b2_rq015b.py:73-84` 与 `205-208`。OnSite 方案只写“不读取”，没有给 materializer 的输入列 allowlist；建议只读 motion/key/source/window 字段，避免读取与求解无关的 context、outcome 或评分字段。

受保护文件：任务书 `RQ016D_M1_kickoff.md:217-235` 对不改源代码、不改 RQ015B/K、不改既有 `data/derived/`、不投 HPC 的边界明确。需要补一条：若 `data/derived/rq016d_onsite_gate/l1_v1/` 已存在，不得覆盖，改写新版本号或停下报告。

### 6. 验收判据逐条判定

1. 行数守恒：有效，会抓住漏行/重复输出；但必须同时检查 `product_row_key` 唯一与三表交集，否则行数相同仍可错配。
2. 状态守恒：有效，会抓住状态分类总和错误；但抓不到错帧、错角色、错窗口。
3. 门判据可复算：有效，会抓住门实现与落盘不一致；但这是内部一致性，不证明 MSE 来源正确。
4. `ipv_log` 与 `k_eff_log` 恒等式：有效，会抓住权重或候选顺序实现错误；但如果候选网格整体错用 5 点或角色错，它不一定暴露，必须另查 K/grid 和 role。
5. 工程失败隔离：有效，必须保留。
6. 护栏断言：方向正确，但目前是报告断言；应变成脚本 blocker，并检查输入列名、输出 source、InterHub 行数 0、held 标记 0。
7. 两条负对照：思路正确，但自然 OnSite 输出不保证存在敏感行。必须加入 synthetic sentinel，否则 `np.isclose(atol=1e-12)` 或 `theta=0.22` 可能不 fail，检查质量无法判断。
8. 数值健康：有效，会抓 NaN/inf、范围越界、负 spread；但抓不到 reference 生成错、时间排序错、坐标单位错。

缺失的验收包括：三表 `product_row_key` 一对一、测量角色/目标帧合同、`MotionSequence` 重建审计、旧通道 denylist、7 行异常坐标正式输出 presence、正式发布不覆盖检查、canary 读回后的 null scalar pattern。

### 7. 失败路径覆盖

任务书 canary 覆盖四类状态、2 worker、写后读，见 `RQ016D_M1_kickoff.md:174-184`，这是必要但不充分。它没有显式要求真实 OnSite 异常坐标行、没有要求三表 join、没有要求 synthetic sentinel 保证负对照失败，也没有要求检查旧 list schema 不被重新引入。

K2 失败路径的真实修复经验可从代码看到：per-process cache 在 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py:175-184`；多 worker 后按 `sample_order` 排序在 `891-898`；scalar schema 与写出在 `780-847`；manifest/output hash 在 `899-918`。OnSite canary 应同时覆盖这些机制。

### 8. 遗漏与必须改动

必须补的第一项是 measurement contract。任务书写了机制二交叉，但没有规定 materializer 输出的 `product_row_key` 必须和 RQ016C dry-run 完全一致。我的只读核查显示 anchor 构造键和 RQ016C dry-run 已经有 67,861/67,861 一对一基础，来源 `.codex-fleet/rq016d-review-a/work/rq016d_ra_readonly_checks.json:261-263`；本轮输出必须继承这条基础。

必须补的第二项是明确目标帧。历史 OnSite builder 对目标值使用 `pos + TARGET_FINAL_OFFSET` 和 `TARGET_HISTORY_WINDOW=4`，见 `hpc_onsite_ipv/build_onsite_m3_anchors_hpc.py:821-829`。任务书同时说 anchor frames 和 future-target 配置，但未要求输出字段记录 `solve_frame_index`、`anchor_frame_index`、`target_window_end_frame_index` 与 `history_window_used`。

必须补的第三项是 canary 的合成 sentinel 与真实异常行。仅靠自然 canary 不能保证覆盖精确 spread 和 theta 边界。

必须补的第四项是 input allowlist。OnSite materializer 不需要读取 M3 context 特征、target value、outcome 或评分字段；只读 motion/key/source/window 字段足够。

### 9. 总体结论

方案可以开跑，但要先改任务书中的验收合同。主方向不是重做科学设计，而是把 OnSite 行如何接入冻结求解器写成机器可失败条件；否则结果可能数值健康、门判据自洽，却对应错的行、错的车或错的时间点。

state: WAITING_ON_COMMANDER
timestamp_utc: 2026-08-04T03:27:14Z
