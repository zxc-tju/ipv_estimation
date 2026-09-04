# RQ029-3：v2 合成轨迹连续性审计

## 定位与结论

本轮要回答：RQ029 v2 的 300 个上海合成驾驶运行是否帧间连续、路径形状合理，
以及是否存在时间、坐标或动力学跳变。RQ029 v2 已于 2026-09-02 完成 paper-aligned 校准；
本次由独立子代理全量审查 `trajectory_pairs.parquet`，并对 `A4` 路径的视觉疑点做专项复算。
这是执行层数据质量结论，不产生新的科学效应主张。

本轮结论为 **PASS_WITH_CAVEATS / SYNTHETIC_NOT_OBSERVED**。

- 通过项：300/300 个 run、81,880/81,880 帧都能在 `trajectory_pairs.parquet` 中读到连续有序的主车轨迹；
  `frame_gap_count > 0` 的 run 为 `0/300`，`timestamp_nonpositive_count > 0` 的 run 为 `0/300`，
  `path_cross_track_m > 0` 的 run 为 `0/300`。
- 保留边界：这些轨迹是按聚合约束构造的**合成、非观测**人类代理轨迹，不是实测人类逐帧记录；
  不能把这份审计写成对真实驾驶员轨迹连续性的验证。
- 两个 caveat：第一，时间间隔是 `0.101–0.125 s`，说明它接近但不等于严格 10 Hz；
  第二，`A3/A4/A6/C4` 等固定曲率段的 jerk / yaw-rate 峰值偏高，反映的是合成路径几何与速度重建的局部陡变，
  不是 frame gap。
- 范围边界：本轮**只**复核 `data/derived/rq029_human_synthetic_microdata/v2_paper_aligned/tables/trajectory_pairs.parquet`；
  没有新做一次全量 `raw -> parquet` 回连，也没有改原始数据。

## 输入与范围

- 审计对象：`data/derived/rq029_human_synthetic_microdata/v2_paper_aligned/tables/trajectory_pairs.parquet`
- 审计规模：20 名合成驾驶员 × 15 个场景 = `300` runs，总帧数 `81,880`
- 本次持久化来源：`/Volumes/ZHITAI 2T/.codex-tmp/rq029-trajectory-audit-v2-subagent/`
- 本次写入目录：`reports/studies/RQ029_human_synthetic_microdata/RQ029_3_trajectory_continuity_audit_20260904/`

## 连续性结果

分母说明：

- run 级检查分母 = `300` runs
- 相邻帧差分分母 = `81,880 - 300 = 81,580` 个相邻步

| 检查项 | 字段 / 阈值 | 分子 / 分母 | 结果 |
|---|---|---:|---|
| coarse frame gap | `frame_gap_count > 0` | `0 / 300` runs | PASS |
| 非正时间增量 | `timestamp_nonpositive_count > 0` | `0 / 300` runs | PASS |
| 时间步上界超 0.11 s | `dt_max_s > 0.11` | `280 / 300` runs | caveat |
| 时间步上界超 0.12 s | `dt_max_s > 0.12` | `100 / 300` runs | caveat |
| 全局时间步范围 | `dt_global_min_s`, `dt_global_max_s` | `0.101–0.125 s` | caveat |
| 单步位移 > 1.0 m | `step_max_m > 1.0` | `55 / 300` runs | note |
| 单步位移 > 1.05 m | `step_max_m > 1.05` | `1 / 300` runs | note |
| 单步位移 > 3 / 5 / 8 / 10 m | `step_max_m` | `0 / 300` runs for all four thresholds | PASS |
| 最大绝对加速度 > 3 / 4 / 5 / 6 m/s² | `accel_max_abs_mps2` | `161 / 0 / 0 / 0` over `300` runs | note |
| 最大绝对 jerk > 5 / 8 / 10 / 15 m/s³ | `jerk_max_abs_mps3` | `280 / 260 / 233 / 178` over `300` runs | note |
| 最大绝对 yaw-rate > 20 / 30 / 45 / 60 deg/s | `yaw_rate_max_abs_deg_s` | `122 / 20 / 0 / 0` over `300` runs | note |
| 速度重建误差 > 1 / 2 / 3 / 5 km/h | `speed_recon_max_abs_kmh` | `300 / 300 / 240 / 87` over `300` runs | note |
| 轨迹偏离模板 | `cross_track_max_m > 0` | `0 / 300` runs | PASS |
| 弧长回退 | `arc_backtrack_count > 0` | `0 / 300` runs | PASS |

补充说明：

- `summary.json` 记录的全局最大值为：`step_global_max_m = 1.0533536470`，
  `accel_global_max_abs_mps2 = 3.9340594416`，`jerk_global_max_abs_mps3 = 31.2680064830`，
  `yaw_rate_global_max_abs_deg_s = 34.2877832525`，`speed_recon_global_max_abs_kmh = 5.8367988129`，
  `cross_track_global_max_m = 0.0`。
- `anomaly_details.csv` 中只有 3 类 run 级异常标签：`dt_outside_0.09_0.11 = 280`，
  `speed_recon_gt_3kmh = 240`，`jerk_gt_10mps3 = 233`；没有 frame-gap 或 cross-track 漂移类异常。

## v1 / v2 不变项

`summary.json` 给出的 12 个 v1/v2 共用运动学列都保持 `n_changed = 0`，
即 `timestamp_ms`、`time_s`、`ego_latitude`、`ego_longitude`、`ego_speed_kmh`、
`ego_course_deg`、`ego_vx_mps`、`ego_vy_mps`、`source_ego_latitude`、
`source_ego_longitude`、`source_ego_speed_kmh`、`path_cross_track_m` 全部逐行不变。

这意味着本轮连续性审计没有发现“v2 改坏了主车基础轨迹”的证据；本次补档也没有对原始数据包做任何修改。

## A4 专项核查

问题背景：旧版 `paths_overlay_15x20.png` 只画 synthetic 路径，`A4` 面板肉眼看起来像两段。
本轮用 `source_ego_latitude` / `source_ego_longitude` 叠加每场景 AV 源路径（深色虚线）后，
重新生成了 `figures/paths_overlay_15x20.png`，并对 `A4` 做了顺序相邻复算。

`A4` 的核查口径如下：

- 场景规模：`20` runs，`6,080` 帧，`6,060` 个相邻步
- run 级 coarse gap：`frame_gap_count > 0` 为 `0 / 20`
- 逐 run 米制复算的最大相邻步：`0.798643–0.859405 m`，中位 `0.846180 m`
- 逐 run 时间步范围：`0.101–0.119 s`
- 用 `dt > 0.2 s` 或 `step > 5 m` 或 `dt <= 0` 作为断段条件时，
  `segment_count_gap0p2_or_step5m = 1` 的 run 为 `20 / 20`
- 叠加的 AV 源路径同样是 `1` 个连续段，代表性源路径的最大相邻步为 `0.828865 m`，
  `dt = 0.101–0.119 s`，`frame gap = 0`

因此，`A4` 面板里“像有两段”的视觉效果来自该场景本身的回转/折返几何，以及 synthetic 与 source path 的近乎重合，
**不是**未检出的时间跳变、坐标断裂或隐藏的 frame gap。

## Caveats

1. **非严格 10 Hz。**
   本批轨迹的时间步不是固定 `0.100 s`，而是 `0.101–0.125 s`。
   这足以支持连续性审计和工程绘图，但不能把它写成“严格 10 Hz 实测采样”。

2. **固定曲率段的 jerk / yaw-rate 峰值偏高。**
   以较高的 run 级阈值 `jerk_max_abs_mps3 > 20` 统计，超限只集中在
   `A3/A4/A6/C4`，分别为 `14/20`、`20/20`、`19/20`、`11/20` runs；其他 11 个场景
   均为 `0/20`，合计 `64/300` runs。若放宽到 `>15 m/s³`，`A5/B3/C2/C3` 等场景也会命中，因此
   不用该较低阈值声称异常只属于四个场景。

   转向角速度的更高阈值 `yaw_rate_max_abs_deg_s > 30` 只在 `A3` 出现，为
   `20/20` runs，即总体 `20/300`；其他场景均为 `0/20`。在较低的 `>20 deg/s` 阈值下，除
   `A3/A4/A6` 外，`B4/C1/C2/C3` 也有 `6/20`、`20/20`、`19/20`、`19/20` runs 命中。
   这些峰值在同场景的固定帧位复现，且全部 `300/300` runs 仍满足
   `|yaw rate| < 45 deg/s`、`|accel| < 4 m/s²`，因此更符合局部几何/速度重建陡变，
   而不是轨迹断裂。

## 可直接使用与不能证明

- 可直接支撑：这份 v2 `trajectory_pairs.parquet` 在执行层的连续性说明、工程绘图、场景路径展示、
  “A4 视觉疑点不是隐藏跳变”的补充证据。
- 可作旁证：v2 仍保留 v1 的主车基础轨迹与模板对齐关系，连续性风险没有被 v2 校准过程放大。
- 不能证明：真实人类逐帧轨迹连续性、真实采样频率合同、全量 `raw -> parquet` 回连已经重新做过、
  或者任何新的独立科学结论。

## 产物

- `summary.json`
- `trajectory_quality_summary.csv`
- `scenario_quality_summary.csv`
- `anomaly_details.csv`
- `worst3_timeseries.csv`
- `figures/paths_overlay_15x20.png`
- `figures/run_level_rankings.png`
- `figures/worst3_timeseries.png`
- `chart_map.md`
