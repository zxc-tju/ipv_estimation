# RQ029-3 Chart Map

我亲自查看了本目录下 3 张最终 PNG，下面记录每张图回答的问题、字段、编码和 QA 结论。

| 文件 | 分析问题 | 图形 | 主要字段 | 结论 | 颜色 / 非颜色区分 | QA |
|---|---|---|---|---|---|---|
| `figures/paths_overlay_15x20.png` | 15 个场景的 synthetic ego path 是否连续，`A4` 的视觉间隙是不是隐藏跳变 | 3×5 场景路径叠加图 | `scenario_id`, `run_id`, `timestamp_ms`, `ego_latitude`, `ego_longitude`, `source_ego_latitude`, `source_ego_longitude` | 15 个场景都显示为连续轨迹；`A4` 在叠加 AV 源路径后仍是一条连续回转曲线，不存在未检出的断段 | 浅绿色半透明实线 = 20 条 synthetic runs；深色虚线 = source AV path；深色点/红点 = 起点/终点；不依赖颜色的区分还有线型和端点标记 | 已目视检查。标题、图例、A4 注释均可读；A4 面板不再被误读为数据断裂 |
| `figures/run_level_rankings.png` | 300 个 run 按综合严重度排序后，哪些 run 靠近连续性边界 | 2×2 排名条形图 | `combined_rank_score`, `step_max_m`, `accel_max_abs_mps2`, `jerk_max_abs_mps3`, `yaw_rate_max_abs_deg_s` | 连续性风险主要表现为 jerk / yaw / 速度重建误差的局部高峰；没有 run 落到“大步长/大缺口”型断裂 | 单色蓝柱；区分靠坐标轴标题和子图位置，不靠多色编码 | 已目视检查。四个子图标题、量纲和排序轴都清楚，没有截断 |
| `figures/worst3_timeseries.png` | 最差 3 个 run 的问题是断裂还是局部动力学尖峰 | 3×4 时序折线图 | `run_id`, `time_s`, `step_m`, `accel_mps2`, `jerk_mps3`, `yaw_rate_deg_s` | 最差 3 个 run 都来自 `A3`；其问题是局部 jerk / yaw-rate 尖峰，不是 step 或时间索引中断 | 单色蓝线；区分靠列标题（step / accel / jerk / yaw rate）和每行 `run_id`，不靠颜色 | 已目视检查。12 个面板都能读清，趋势连续，无空白断线伪影 |

补充 QA 说明：

- `paths_overlay_15x20.png` 是本轮唯一重生成的 PNG；我确认了新版已经包含 source AV 虚线叠加与 `A4` 注释。
- `run_level_rankings.png` 和 `worst3_timeseries.png` 沿用 analyst 产出的最终 PNG，本轮只做持久化与目视确认，没有改图数据。
