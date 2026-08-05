# RQ019-B1：异常 IPV 是否把代价转嫁给交互对手方（执行任务书 v1）

你是执行 agent。监督方是 Claude。**本轮是探索性/描述性产出：一个 agent、一轮自查、
出报告、结束。不做盲审、不出第二版规格、不建授权闸门。发现自己在写规格 v2 就是跑偏了，停下。**

---

## 1. 这项工作要解决什么问题

最终目标是**在线验证**：判断一辆自动驾驶车（AV）表现出的社会交互倾向像不像人。判定由两道
串联弃权机制构成——**机制一**判断某一帧的 **IPV**（Interaction Preference Value，表示交互
倾向的标量）是否携带七个候选间的判别信息；**机制二**用**人类参照分布（envelope）**判断
当前情境是否有足够人类样本可比。两关都过才可判。

RQ015 冻结机制一，RQ016C 建好纯人-人 envelope，RQ017 在 AV 数据上算出机制一判据，
RQ018 发现：**IPV 低于该情境人类参照下界（即比人类更激进）时，后续最小 TTC 的分布整体
左移（中位 7.51 s 对 8.81 s），但危险阈值以下的帧反而更少。** 即它压缩安全裕度，
不制造极端危险。

**本轮（RQ019）问的是这个压缩的代价由谁承担。**

### 为什么这个问题是机制对齐的（务必理解，它决定了解释口径）

IPV 的代价函数在受保护源码 `src/sociality_estimation/core/agent.py:1193` 是：

```
util = cos(ipv) × 自身代价 + sin(ipv) × 交互代价
```

**`sin(ipv)` 就是对方代价的权重。** 候选网格 `[-3..3]×π/8`。IPV 越大越看重对方代价
（更合作让行）；**IPV 越负则反向压制对方（更竞争激进）**。

所以「IPV 向负偏离 ⇒ 对手方承担更多代价」不是外部借来的代理指标，
而是这个量的定义本身所预测的后果。**本轮检验的正是这条。**

⚠ **下侧越界是「比人类更激进」，不是「更消极」。** 不要写反。

---

## 2. 必须先知道的三件事（不遵守本轮就白跑）

### 2.1 只能用锚点之后的窗口

envelope 的特征合同
（`reports/studies/RQ016C_human_only_envelope/RQ016C_1_human_only_envelope_20260804T005716Z_7480c173/envelope_model/feature_contract.json`）
的 22 项 `numeric_context` **包含** `counterpart_vx_anchor`、`counterpart_vy_anchor`、
`counterpart_heading_anchor`、`relative_*_anchor`、`closing_ttc_anchor`、`apet_online_proxy`。

即**锚点时刻的对手车运动学已被 envelope 条件化**。用它作同期结果变量是循环论证。
**结果变量一律取锚点之后的窗口。**

### 2.2 必须用原始日志，不能用派生表的速度

派生表 `onsite_ipv_timeseries_multi_allvalid.parquet` 的对手车速度是**用位置差分算出来的**，
逐帧差分得到的 |加速度| > 10 m/s² 占 **19.8%**，无法用于测量制动。

**原始竞赛日志直接记录了 `speed` 与 `acceleration`**，|a|>10 仅 **0.019%**，
下界恰为 **−7.5 m/s²**（仿真器制动上限，说明是真实控制量）。**本轮一律用原始日志。**

⚠ **单位陷阱**：原始日志 `speed` 单位是 **km/h**（最大值恰为 60.00），
派生表是 **m/s**。`acceleration` 是 **m/s²**。换算错会让所有数字错 3.6 倍。

### 2.3 结论必须是分布级的，不能只给回归系数

RQ018 的教训：`log(1+TTC)` 的负系数被误读成「更危险」，实际上危险端方向相反，
负系数完全来自安全端压缩。**本轮必须同时给：**

1. **分位数表**（1/5/10/25/50/75/90 分位），按越界侧分组
2. **阈值超越占比**（例如制动强于 −2 / −3 / −4 m/s² 的帧占比）
3. **case 层 bootstrap 的占比差与 95% CI**
4. 回归系数只作为补充，**不得单独作为结论**

**明确区分「分布整体偏移」与「尾部恶化」，两者结论不同。**

---

## 3. 数据与对齐配方（已由监督方端到端验证，照做即可）

### 3.1 输入

| 用途 | 路径 | 关键列 |
|---|---|---|
| 机制一判据 | `data/derived/rq017_onsite_gate/l1_v1/` | `product_row_key`, `status`, `ipv_log` |
| 机制二区间与支持门 | `.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet` | `product_row_key`, `lo_90`, `hi_90`, `width_90`, `mechanism2_gate_ok`, `context_cell` |
| 锚点与连接键 | `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet` | `case_key`, `anchor_frame_index`, `perspective`, `session_id`, `counterpart_key_agent`, `counterpart_selection`, `target_window_end_frame_index`, `unit_composite_key` |
| 派生逐帧（只用于时间戳与对齐核验） | `.../onsite_anchors_multi/onsite_ipv_timeseries_multi_allvalid.parquet` | `case_key`, `session_id`, `frame_index`, `timestamp_ms`, `counterpart_key_agent`, `counterpart_x`, `counterpart_y` |
| **原始逐帧日志（结果变量来源）** | `data/onsite_competition/all_teams_dataset/teams/*/sessions/<session_id>/simulation_trajectory.log` | JSONL，见下 |

原始日志每行：`{"type":"trajectory","value":{"timestamp":..., "globalTimeStamp":"<ms字符串>",
"value":[...], "trafficLight":[...], "obstacles":[...]}}`。
车辆记录（在 `value` 或 `obstacles` 数组内）共 23 字段，本轮需要：
`id`, `name`, `frameId`, `globalTimeStamp`, `x`, `y`, `speed`(km/h), `acceleration`(m/s²), `courseAngle`。

⚠ `globalTimeStamp` 是**字符串**，需 `int()`。约 2.88% 的记录字段为 `None`，需过滤并计数。

### 3.2 对齐配方（监督方已验证）

1. `session_id`（如 `6931-1766206339`）→ 会话目录名，用
   `find data/onsite_competition/all_teams_dataset/teams -type d -name "<session_id>"` 定位
2. 派生表 `counterpart_key_agent`（如 `500002`）→ 原始日志记录的 `id`（**数值相等**）
3. 时间：派生 `timestamp_ms` → 原始 `int(globalTimeStamp)` 取**最近邻**
4. **坐标系不同**：原始与派生的 x/y 相差一个**每 case 恒定的平移**
   （实测样例 x 偏移 84.995 m、y 偏移 −306.095 m）

**硬断言（不通过就停下报告，不要自行放宽）**：
- 最近邻时间戳差的 95 分位 **< 150 ms**
- 去掉 per-case 中位平移后，位置残差中位 **< 0.5 m**
  （监督方样例实测 x 0.0072 m、y 0.0777 m）
- 原始日志中 `|acceleration| > 10 m/s²` 的占比 **< 1%**

---

## 4. 曝露变量（与 RQ018 同口径，便于对照）

分析集：机制一 `status == OK` 且机制二 `mechanism2_gate_ok == True`，**预期 14,099 帧**，
覆盖 231 个 case、19 个 team。

对每帧在 α=90（主）与 80/95（敏感性）计算**非负幅度**形式：

```
upper_a = max(0, ipv_log - hi_a) / width_a      # 比人类更合作让行
lower_a = max(0, lo_a - ipv_log) / width_a      # 比人类更激进
```

**上侧与下侧必须分开保留，不得合并成绝对值。**
参考数字：α=90 时上侧 2,700 帧、下侧 1,998 帧、区间内 9,401 帧。

---

## 5. 结果变量：对手车在锚点后窗口的反应（主线）

窗口两套并报：
1. **合同窗口** `[anchor_frame_index, target_window_end_frame_index]` 对应的时间区间
2. **固定 3 秒窗口**（按时间，不按帧数）

对手车在窗口内（全部取自原始日志）：

- `cp_min_acceleration`：窗口内 `acceleration` 最小值（最强制动，负值越小越强）
- `cp_brake_share_2 / _3 / _4`：`acceleration < -2 / -3 / -4 m/s²` 的帧占比
- `cp_speed_drop_kmh`：窗口内 `max(speed) - min(speed)`，并另报锚点时 speed 减窗口内最小 speed
- `cp_max_abs_yaw_rate`：`courseAngle` 逐帧变化率绝对值的最大值
  （**先按 case 做角度 unwrap**，用 `globalTimeStamp` 的真实时间差，不要用帧数）
- `cp_total_heading_change`：窗口内 `courseAngle` 的总变化量绝对值

⚠ `courseAngle` 的单位与取值范围**先自己确认**（度还是弧度、[0,360) 还是 [−180,180)），
并在报告中写明依据。

**分组隔离**：`counterpart_selection == 'online_first_conflict_nearest_timing_eligible_prefer_scripted_from_vehicle'`
的行（预期约 7,575 行、30 个 case）**单独一组报告**，因为 scripted 对手可能不反应。
主结论用非 scripted 组；两组都要报。

---

## 6. 次要（探索性，不求结论）

在**不超过 5 个会话**上试算：锚点时刻自车周围一定范围内（自己选一个半径并说明理由，
例如 50 m）**所有**车辆的同类反应量，只回答「这条路可不可行、数据够不够干净」，
**不做统计推断，不下结论**。原始日志单会话约 121 辆车。

---

## 7. 分析与推断

- 帧嵌套在 case 内、case 嵌套在 team 内（约 19 个有效 team）。
  **所有推断必须给 case 层（主）与 team 层（次）聚类稳健口径或 case 层 bootstrap CI，
  并同时列出朴素口径以显示差异。以聚类口径为准。**
- 控制 `context_cell`。
- **先给分布对比（§2.3 的四项），再给回归。** 回归不得单独作结论。

### 最小负对照（两条都要做，不做完整 battery）

1. **case 层标签置换** ≥200 次，给经验 p 值。
2. **安慰剂曝露**：每帧从 `[lo_90 - width_90, hi_90 + width_90]` 均匀抽假 IPV，
   按同公式算 exceedance，重跑主分析。

---

## 8. 硬约束（与流程无关，不得放松）

```
1. RQ007 held_out 不得被解析（污染不可恢复）
2. RQ014 致盲相关的评分字段不得读取；遇到 rating/score/preference/human-score
   字段先停下报告。（注：OnSite 的 official_* 竞赛分数不在此列，可读；
   但本轮不需要它们，不要去读 monitor.log）
3. 不得静默覆盖已冻结产物或已接受的 decision.md
4. 描述性结果不得写成因果主张
```

补充：

- **禁用词**：`estimability`、「测出/未测出 IPV」。可辩护表述是
  「权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息」。
- **禁用「导致」等因果表述。**
- **不得改动**：五个受保护文件（`src/sociality_estimation/core/{agent,ipv_estimation,reliability_logdomain}.py`、
  `pipelines/interhub/process_interhub.py`、`configs/ipv_sigma01_exact.json`）、
  `data/derived/` 下已有内容、RQ009/RQ016/RQ016C/RQ017/RQ018 已落盘 run 目录。
- **不得执行任何 git 写操作**（不 commit、不 push、不切分支）。
- **本轮不对任何车辆、任何队伍作出判断**，不得出现「某队更激进/更不安全」之类表述。
- **不要对 `reports/` 做全仓库 `rg`**（会把 RQ003 controlled-access 行拉进上下文）。
- 原始日志有 113 个、合计 4.2 GB，**不要一次性全读进内存**；按 session 流式解析。

---

## 9. 交付物（写到 `.codex-fleet/rq019-counterpart-burden/work/B1/`）

1. `rq019_counterpart_burden.py` —— 可复跑主脚本
2. `alignment_contract.json` —— §3.2 三条硬断言的结果
3. `key_numbers.json` —— 每个数字带分子/分母/筛选条件/来源文件/列名
4. `distribution_results.json` —— 分位数表与阈值占比（主结论载体）
5. `regression_results.json`、`negative_controls.json`、`data_health.json`
6. `surrounding_probe.json` —— §6 探索性试算
7. 报告 `.codex-fleet/rq019-counterpart-burden/board/reports/RQ019_1_counterpart_burden.md`

### 报告写法（硬性）

- **开头先定位**：这项工作解决什么问题、整体走到哪一步、本次是哪一环。
  假设读者没跟进过程，不得直接从增量讲起。
- 不用黑话不用比喻；必须用专有名词时当场一句话解释。
- **结论与待决事项分开成节。**
- **数字自带口径**：分子、分母、筛选条件、来源文件与列名。读者无法自行复算的数字等于没给。
- **明确区分「分布偏移」与「尾部恶化」。**
- 结尾 `state: WAITING_ON_COMMANDER` 与 `timestamp_utc:`
  （用 `date -u +%Y-%m-%dT%H:%M:%SZ`，不要前瞻估计）。

---

## 10. 自查清单（跑完逐条给机器证据）

- [ ] 分析集行数与 14,099 一致；不一致报告差异来源
- [ ] §3.2 三条对齐硬断言全部通过（给实测数值）
- [ ] 原始日志字段缺失率（参考：样例会话 2.88%）
- [ ] 能匹配到原始日志的锚点帧占比；未匹配的原因与计数
- [ ] 上侧/下侧/区间内三组计数之和 = 分析集行数
- [ ] scripted 组与非 scripted 组的行数与 case 数
- [ ] 窗口越出 case 末尾的帧数与占比
- [ ] `acceleration` / `speed` / `courseAngle` 的数值健康（NaN、±inf、越界）
- [ ] case 帧数分布、case 数、team 数
- [ ] 朴素与 case 聚类两种 p 值并列
- [ ] 两条负对照结果
- [ ] 7 行坐标系异常（`relative_distance_anchor` ≈ 570,761 m，全部来自
      `onsite:shanghai:T10:C4:native_case:2311`）是否进入本轮分析集及其影响

---

## 11. 结项

写完报告状态写 `WAITING_ON_COMMANDER`，**不要自行转 DONE**，不要执行 git 操作。
监督方会独立复算后再决定入库。

如果发现本任务书某条设定与数据实际不符（列不存在、行数对不上、断言过不了），
**停下来在报告里写清楚，不要自行改设定继续跑**。
