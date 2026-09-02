# RQ029 v0：20×15 人类轨迹合成微观数据

状态：`EXECUTED / PASS`
日期：2026-09-02

## 1. 问题与用途

当前 RQ022 只有经确认的 20 名驾驶员 × 15 个场景聚合统计，仓库内没有真实人类逐帧轨迹。
本任务生成一份**工程代理数据**，用于联调、可视化、数据接口测试和统计复算演示。它回答的是：
能否在不改变 AV 场景与背景交通参与者运动的前提下，构造沿同一路径、速度略有差异的主车轨迹，
并让合成微观表复现现有 RQ022 聚合统计。

这不是缺失人类轨迹的“恢复”。输出必须标为 `SYNTHETIC_NOT_OBSERVED`，不得作为真实人类证据、
RQ022 的独立复算、车辆或驾驶员评价，也不得替换受控归档中的真实随件表。

## 2. 分析单位与来源

- 运行单位：`driver_id × scenario_id`，20 × 15 = 300 次运行。
- 时刻单位：10 Hz 左右的 AV 重放帧。
- 场景：A1–A7、B1–B4、C1–C4。
- 地域约束：用户确认 20×15 人类驾驶全部在上海完成，因此只使用上海场景变体，禁止混入北京
  `caseId` 或北京背景流。
- AV 模板：上海 T11 / session `6923-1766197775` 的完整重放日志，来自
  `data/onsite_competition/all_teams_dataset/teams/shanghai/01_T11_wsd/sessions/6923-1766197775/`。
  该 session 的四项必要日志齐全、15/15 场景均存在，共 4,094 个感知合成轨迹帧；源本身没有可选的
  `vehicle_perception_trajectory.log`，因此合成包也不伪造该可选文件。
- 聚合目标：`reports/plans/RQ029_human_statistical_targets_v1.json`，逐项抄录自
  `.codex-fleet/rq022-matched-scenario/work/T1_target_figure/human_arm_data.json`，并受
  `reports/knowledge/RQ022_matched_scenario_human_arm/decision.md` 约束。
- 明确排除：RQ007 held-out、RQ014 致盲评分字段、官方分数、伤害标签和偏好评分。

## 3. 生成合同

1. 输出目录仿照 AV all-team 包：每个 D01–D20 各有一个 session，保留
   `monitor.log`、`simulation_trajectory.log`、
   `vehicle_perception_simulation_trajectory.log`、`vehicle_trajectory.log`；只有源存在时才保留可选的
   `vehicle_perception_trajectory.log`。
2. `simulation_trajectory.log` 与 AV 模板逐字节相同，并通过硬链接复用；感知轨迹中除主车外的
   交通参与者对象逐字段相同。
3. 主车经确定性、平滑、单调的路径时间重参数化生成；起终点和路径折线不变，速度有小幅个体/场景
   差异，默认局部扰动约束在 ±12% 内。
4. 候选时刻、两道门和参照带标签按已知边际统计约束采样。合成 `ipv_log` 与参照带只用于复现
   已知统计，不声称由真实人类轨迹反演得到。
5. 冻结签名统计同时保存原始运动学派生列和目标校准列。校准列必须带 `_calibrated` 或在 manifest
   中显式列入 `aggregate_constrained_fields`。
6. 原聚合对象只给出 `n_cases=186`，没有真实 case 与时刻的对应关系；输出中的 186 组仅命名为
   `pseudo_case_id` 以复现结构计数，不得用于推断。所有合成 bootstrap 一律按可追溯的
   `run_id = driver_id × scenario_id` 重抽。

## 4. 验收标准

- 结构：20 名合成驾驶员、15 场景、300 个唯一运行；每个 session 的必要日志齐全且 JSONL 可解析。
- 背景：共享 `simulation_trajectory.log` SHA-256 与 T15 源完全相同；感知日志全部非主车对象语义哈希
  与源一致。
- 主车：所有位置/速度有限，时间戳单调；合成点落在源路径折线上，最大横向偏差 ≤ 0.05 m；
  速度确有变化但保持小幅，95% 绝对相对偏差 ≤ 12%。
- 统计精确匹配：候选 78,903；门一通过 40,993；两门同过 15,598；80/90/95 三档上下侧/带内计数、
  15 个场景的 90% 带外计数与目标逐项相等。
- 签名匹配：目标样本数、TTC 四分位数与 `<2 s` 计数、对手速度下降/波动中位数、
  `<−3 m/s²` 分子分母与目标在声明精度内一致；bootstrap CI 另列合成复算值，不把目标 CI 冒充复算值。
- 质量：候选键无重复、必填列无缺失、带宽嵌套、所有计数守恒；测试和独立验证脚本通过。

## 5. 交付边界

读者侧保留生成器、目标合同、manifest、运行/逐时刻/逐单元表、统计复核、质量报告和文件校验清单。
大型 JSONL/parquet 只放 `data/derived/rq029_human_synthetic_microdata/v1/`（被 Git 忽略）；
跟踪的执行报告只记录口径、结果和可复跑命令。
