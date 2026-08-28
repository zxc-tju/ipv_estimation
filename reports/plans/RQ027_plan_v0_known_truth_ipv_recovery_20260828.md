# RQ027 Plan v0：独立仿真下的已知真值 IPV 恢复与弃权验证

状态：`EXECUTED / PILOT_NO_GO`
用户授权日期：`2026-08-28`
`protected_data=NONE` ｜ `RQ007_held_out=DENY_READ` ｜ `RQ014_blinded_fields=DENY_READ` ｜ `paper_edit=DENIED`

## 1. 定位与来源

RQ027 要回答的是：当前冻结的在线 IPV estimator 在一个不共享其 planner、搜索、代价实现或 likelihood 的独立仿真生成器上，能否恢复仿真中已知的 IPV 参数；候选权重集中度在有真值时是否真的携带误差信息；在无互动负对照中是否会产生持续的高集中 reading。

本计划由用户提供的研究草案修订而来。草案是待评审输入，不是仓库指令：

- 来源：`/Users/xiaocong/Downloads/IPV_recovery_agent_research_plan.md`
- SHA-256：`2a35861eecaff4b03fe0bb58724a8bb9a88ae673e467315f8ef1a497c4fac367`
- 行数：`767`

当前仓库 remote 为 `https://github.com/zxc-tju/ipv_estimation.git`，因此草案所指研究仓库与本 checkout 一致。计划冻结时的代码基线为 commit `889b49bea9dabcd75c2d7bbc316ee2118ece2e2d`；执行 receipt 必须记录实际 commit 和 dirty-state，不得假装未跟踪文件不存在。

## 2. 为什么单开 RQ027

- RQ024 已接受的结论是：同模型 synthetic Tier1 的 `288/288` 行工程健康，但旧 Gate A 在相邻严格阈值比较中 `36/42` 失败；候选权重集中度不能直接叫 accuracy QC。
- RQ026 只负责 frozen monitor runtime evidence，不允许跨入 accuracy/recovery 边界。
- 既有 WP2 pilot 的生成端调用 `Agent.solve_optimization(...)`，属于同模型 S0，只能作为工程 sanity，不能作为独立 S1 recovery 证据。

RQ027 是 RQ024 下游指令所要求的“单独批准的新合同”。它不重开、推翻或改写 RQ017/RQ018/RQ019/RQ021/RQ024/RQ025 的已接受 `decision.md`。

## 3. 研究问题

### RQ027-1：已知真值恢复

独立 rollout generator 产生的行为在几何上存在互动机会时，冻结 estimator 的 run-level reading 是否随 `true_ipv_rad` 单调变化，并比不看轨迹的零值基线更接近真值？

### RQ027-2：候选集中度的误差信息

`effective_candidate_fraction = q_eff = 1 / (K * sum(w_k^2))` 是否与 recovery error 同向变化（数值越大表示权重越分散），固定政策门 `max(w) >= 0.20` 是否在同时报告 coverage 后降低误差或方向翻转风险？

### RQ027-3：无互动负对照

在无几何冲突、时间错位、错误 counterpart 和 post-resolution 条件下，候选集中门是否仍持续通过？

### RQ027-4：稳健性升级

只有 feasibility pilot 通过后，才另行冻结 S2 扰动、模型失配、完整 split 和 confirmatory 阈值；本 v0 不运行 sealed confirmatory test。

## 4. 证据层级

| 层级 | 作用 | 本 v0 状态 |
|---|---|---|
| S0 同模型 sanity | 单位、符号、接口回归 | 复用既有 WP2/RQ024 事实；不作主 claim |
| S1 独立生成器 recovery | 已知真值恢复 | 本轮主 pilot |
| S2 扰动与失配 | 退化边界 | pilot GO 后才设计 |
| S3 无互动负对照 | 虚假高集中 reading | 本轮主 pilot |

## 5. 独立生成器合同

生成器必须是 estimator-compatible，而不是 estimator-equivalent。

允许共享：

- 输出列合同 `[x, y, vx, vy, heading]`；
- `dt = 0.1 s` 和 reference polyline 的输入形状；
- IPV 的方向约定和范围 `[-3π/8, 3π/8]`；
- 物理量纲与加速度/转向边界。

禁止共享：

- `Agent`、`solve_optimization()`、`utility_fun()`；
- `cal_individual_cost()`、`cal_group_cost()`、`cal_traj_reliability()`；
- estimator 候选轨迹、候选缓存、SLSQP 搜索结果、likelihood 或权重；
- `Simulator` 和 `lattice_planner`；
- `true_ipv_rad` 进入 estimator 调用或 estimator 输入表。

最小实现使用独立的离散 action-template / rollout 搜索与独立运动学更新，具有 progress、comfort/jerk、lane following 和 interaction-risk 项；counterpart 必须响应 target 的运动。生成模块必须能在不导入 `sociality_estimation` 的环境中单独导入，测试需扫描 import graph。

## 6. Feasibility pilot 设计

### 6.1 互动 runs

```text
4 scenario templates
× 5 target IPV {-2,-1.5,0,1.5,2} × π/8
× 2 counterpart IPV {-2,+2} × π/8
× 2 interaction intensity {weak,strong}
× 3 seeds
= 240 runs
```

四个模板：clear-priority crossing、ambiguous-priority crossing、merge、same-direction negotiation。五个真值同时含 on-grid 与 off-grid，并避免零值基线在 `π/8` 容差下机械覆盖大多数非零真值。

### 6.2 负对照

四类各 `12` runs，共 `48`：no-conflict neighbour、time-shifted counterpart、wrong-run pseudo-pair、post-resolution window。

### 6.3 统计单位与分母

- 主统计单位：`simulation_run`。
- 帧只用于 onset、persistence 与时序诊断，不作为独立样本。
- 负对照 persistent false-accept 分母：`48` 个 negative-control runs；分子为出现至少 `K=3` 个连续 concentration-pass 帧的 run 数。
- 互动 coverage 分母：`240` 个 interactive runs；分子为在 opportunity 期间出现 persistent onset 且有 run-level estimate 的 run 数。
- 所有失败、碰撞、非有限值和无 onset run 均保留 reason code，不静默删除。

## 7. 冻结测量定义

- estimator candidate generator：现有 `estimate_ipv_pair(return_diagnostics=True)`，`solver_mode=exact`；每个 run 的唯一 primary 是 **target-side directed estimate**，counterpart-side 只能作 secondary symmetry check，不能把一条 run 扩成两个主统计单位。
- 候选网格：`{-3,-2,-1,0,1,2,3} × π/8`。
- `history_window=10` 在当前实现中表示最多 `10` 个历史步加当前点，即最多 `11` 个位置样本、`1.0 s` 时间跨度；`min_observation=4` 表示第 `5` 个样本开始尝试。likelihood `sigma=0.1 m`。
- primary weights / reading：从 diagnostics 中的 observed target track 与七条 virtual target tracks 计算每候选 mean squared position error，再用 log-domain `softmax(-mse/(2*sigma^2))` 得到 `w_log` 与 `ipv_log=sum(grid*w_log)`。这是 `max(weight)>=0.20` 政策门所对应的 RQ015/RQ017 数值语义。
- legacy sensitivity：core `cal_traj_reliability()` 的概率域乘积权重和 legacy IPV 只作敏感性；它可能全下溢后回退为均匀权重/零 IPV，不得作为 primary concentration 或 recovery 读数。
- frame attempt：`t >= 4` 且 estimator 输出与权重有限。
- concentration pass：`mse_spread > 0` 且 `max(w_log) >= 0.20`。它是既有政策阈值，不是假定准确度阈值。
- persistent onset：首次出现 `K=3` 个连续 concentration-pass 帧。
- run-level estimate：persistent onset 后所有 target-side concentration-pass 帧的 `median(ipv_log)`。
- sign accuracy：只在 `abs(true_ipv_rad) >= π/8` 的 runs 计算；`true_ipv_rad=0` 不进入 sign 分母。
- opportunity truth：按独立生成器的几何状态，用 `3.0 s` 常速度投影的最小中心距 `< 5.0 m` 定义；它与 concentration pass 分开记录。
- oracle informativeness：同一状态、同一噪声、同一 counterpart 下，对 `true_ipv_rad ± π/16` 做 `1.0 s` 配对 rollout，取 target XY 的 RMS 差除以 `2Δ`。它只做离线分层，不进入 estimator、不删样本、不定义成功。

## 8. Pilot 主要结果

必须同时报告：

- interactive / negative-control run 数与所有 reason-code 分母；
- bias、MAE、median absolute error、RMSE、Spearman；
- half-grid 与 one-grid success；
- 零值 predictor 在相同 run 集上的对应 baseline；
- accepted-only 与 all-run（abstain 计未恢复）两套结果；
- on-grid/off-grid、template、role/intensity 分层；
- `q_eff`、`max_weight` 与 error 的关系；
- risk-coverage，不要求每一对相邻阈值有限样本下严格单调；
- negative-control frame acceptance 与 persistent run false-accept；
- collision、非有限值、重复 key、隐藏排除和边界饱和健康项。

## 9. Feasibility 判定

本轮不是手稿级 confirmatory claim，只决定是否值得扩规模。

`PILOT_GO` 需同时满足：

1. 独立性、行守恒、唯一 key、有限值和确定性复跑测试通过；
2. interactive run-level Spearman 为正，且四个 template 至少三个不出现负向关系；
3. estimator 的 one-grid success 高于同一分析集的零值 predictor，MAE 低于零值 predictor；
4. 较分散的 `q_eff` 不得呈现更低 error 的稳定反向关系；固定 `max(weight)>=0.20` 的 coverage 与 risk 必须成对报告；
5. negative-control persistent concentration-pass rate 低于 interactive persistent coverage，且不高于 `25%`。

任一条失败即 `PILOT_NO_GO`：保存负结果，不运行 3,120/14,040 全量，不在同轮调门后重跑。附件建议的 `5%` false accept、`80%` half-grid、`ρ>=0.90` 等值保留为未来 confirmatory 候选，不在尚无独立 pilot 依据时冒充已验证标准。

## 10. 执行与产物

首次执行根：

`reports/studies/RQ027_known_truth_ipv_recovery/RQ027_1_feasibility_pilot_<UTC>_<git-short-sha>/`

正式包采用：

```text
00_entry/index.html
01_results/report.md
01_results/conclusions.md
01_results/run_level_results.csv
01_results/frame_level_results.parquet
01_results/summary.json
01_results/evidence_summary.csv
02_process/README.md
02_process/artifact_manifest.csv
```

研究代码放在 `pipelines/simulation/`，测试放在 `tests/`；不得改现有 estimator。正式作图属于交付层，不能成为科学运行 blocker。

## 11. 保护边界与停止条件

- 不读取或解析 RQ007 held-out；不读取 RQ014 致盲评分字段。
- 不读真实轨迹；本轮 `protected_data=NONE`。
- 不改 paper repo、accepted `decision.md`、现有数据或 RQ026 work tree。
- 不将 S0、S1 或仿真真值写成真实人类心理 IPV recovery。
- 不将 concentration、`q_eff` 或 `max_weight` 在验证前称为 uncertainty、confidence 或完整 estimability。
- 不使用因果、production-ready、external validity 或真实车辆/人员判断措辞。

## 12. v0 完成定义

- 合理性审查与 inventory 有持久记录；
- 独立生成器、runner 与测试存在且通过；
- `240 + 48` pilot 完整运行或以结构化 blocker 停止；
- 一轮数值健康自查完成；
- 结果以 `PILOT_GO` / `PILOT_NO_GO` / `BLOCKED` 收口；
- `STUDIES.md`、`START_HERE.md`、知识层和 `main_workflow.log` 同步；
- 不生成或修改 `decision.md`，除非后续由 PI 明确接受结果。

## 13. 执行结果（2026-08-28）

正式 `240 + 48` pilot 已执行并完成独立复算，最终为 `PILOT_NO_GO`。工程层 `288/288` runs 完成、工程失败 `0/288`，但三个科学门均失败：accepted-run MAE `0.553907 rad` 未优于零值 predictor `0.553432 rad`；`q_eff` 与帧级绝对误差 Spearman `-0.124207`，方向与选择性风险假设相反；负对照 persistent concentration false accept `35/48 = 72.9167%`。因此按 v0 停止条件，不扩展 S2、不运行 sealed confirmatory、不调门后重跑。

正式包：`reports/studies/RQ027_known_truth_ipv_recovery/RQ027_1_known_truth_recovery_20260828T094332Z_889b49be/`。
