# RQ029-1：20×15 上海人类轨迹合成微观数据执行报告

## 定位与结果

RQ022 的真实人类参照臂有 20 名驾驶员 × 15 个上海场景的确认聚合统计，但仓库内没有真实逐帧
轨迹。本次是为数据接口、可视化和流水线联调构造一份完整微观代理数据：复用上海 AV 场景和
背景交通流，只让主车沿同一路径产生小幅速度差异，再以约束采样复现已知边际统计。

执行已完成，最终状态为 **PASS / SYNTHETIC_NOT_OBSERVED**。大型数据位于
`data/derived/rq029_human_synthetic_microdata/v1/`；它不是对真实人类轨迹的恢复，也不能作为
RQ022 的独立复算或新手稿证据。

## 来源、单位与范围

- 地域：上海；场景 A1–A7、B1–B4、C1–C4。
- AV 模板：T11/WSD，session `6923-1766197775`，15/15 场景齐全。
- 运行单位：`driver_id × scenario_id`，20 × 15 = 300 个唯一运行。
- 原始帧：每位驾驶员 4,094 帧，合计 81,880 帧。
- 必要日志：每位驾驶员均有 `monitor.log`、`simulation_trajectory.log`、
  `vehicle_perception_simulation_trajectory.log`、`vehicle_trajectory.log`。
- T11 源没有可选的 `vehicle_perception_trajectory.log`，合成包没有伪造该文件。
- RQ007 held-out、RQ014 致盲评分字段、官方得分、伤害和偏好标签均未读取。

## 生成方式

背景 `simulation_trajectory.log` 先复制到输出自有共享目录，再由 20 个 session 以硬链接复用；
这样既保持与 AV 源逐字节一致，又不会让合成目录的后续修改反向影响唯一 AV 源。感知日志中只替换
唯一 `isPerception==0` 主车对象，`mvSimulation`、交通灯、障碍物及其时间戳逐字段不变。

主车使用平滑、单调的路径弧长时间重参数化：起终点不变，所有点位于源折线上，速度扰动由固定
seed `20260902` 决定。聚合统计无法识别真实驾驶员效应、变量协方差或 IPV 到轨迹的映射，因此
`status`、参照带、`ipv_log` 和带 `_calibrated` 的签名列明确属于约束合成值；同表保留 `_raw`
运动学派生列供区分。

## 数值结果

- 候选时刻：78,903；门一通过：40,993；两门同过：15,598。
- 80% 带：下侧/上侧/带内 = 879/676/14,043。
- 90% 带：下侧/上侧/带内 = 435/351/14,812；带外 786/15,598 = 5.039%。
- 95% 带：下侧/上侧/带内 = 189/167/15,242。
- 15 个场景的 90% `n_both` 与 `n_flagged` 均逐项命中目标。
- 主车速度绝对相对偏差 95% 分位为 5.062%；独立点到源折线复算的最大横向偏差为
  `6.99e-10 m`（验收阈值 0.05 m）。
- TTC、对手减速/速度波动、急刹分子分母和中位数均命中目标；合成 bootstrap 的方向与原统计
  判断一致，区间数值没有被强行写成原区间。
- 合成 90% 带外率按 300 个 `run_id` 重抽的 95% CI 为 [4.596%, 5.568%]；原聚合记录为
  [3.4%, 5.5%]。差异来自未知的真实单元内依赖/异质性，已显式保留，不伪造一致。

## 验证

- 最终验证：195/195 检查 PASS；包括 20 份原始感知主车与 parquet 逐帧回连、两份主车日志互校、
  非主车语义哈希、背景字节哈希、场景/键/计数守恒、嵌套参照带和五组 CI 明细重算。
- RQ029 专项测试：7 passed。
- RQ029 + 既有 RQ027 仿真回归：13 passed。
- 独立结果复核：PASS；独立代码审阅在修复两处可假通过路径后 APPROVE，且篡改原始主车或
  `achieved_summary.json` 均会触发 FAIL。
- 全仓库无选择 pytest 会误收历史归档测试并在现行根测试中暴露 3 个既有 RQ014 pin 漂移失败；
  本轮运行到 205 passed、1 skipped 后因旧测试挂起而中断。这些失败不在 RQ029 修改范围内。

## 产物

- 入口与边界：`data/derived/rq029_human_synthetic_microdata/v1/{README.md,manifest.json}`。
- AV 同构原始日志：`.../raw/drivers/D01` 至 `D20`。
- 规范化轨迹与统计表：`.../tables/`。
- 机器验证：`.../validation/{validation_summary.json,checks.csv,raw_file_inventory.csv,REPORT.md}`。
- 生成器、验证器和测试：
  `pipelines/simulation/generate_rq029_human_synthetic.py`、
  `pipelines/simulation/validate_rq029_human_synthetic.py`、
  `tests/test_rq029_human_synthetic.py`。

## 使用边界

本包可以直接支撑结构联调、读取器测试、绘图测试和“给定聚合约束的微观数据”方法演示；只能作为
相关结构/方向的旁证。它不能证明任何真实驾驶员个体轨迹、真实联合分布、真实 IPV 输出、车辆优劣
或 RQ022 的独立复现。受控真实随件表找回后，应以真实表为唯一权威来源。
