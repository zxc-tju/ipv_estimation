# RQ018-A1：异常 IPV 与行为劣化的关联性正向探索（执行任务书 v1）

你是执行 agent。监督方是 Claude（通过文件与你异步交互）。**本轮是探索性/描述性产出：
一个 agent、一轮自查、出报告、结束。不做盲审、不出第二版规格、不建授权闸门。**

---

## 1. 这项工作要解决什么问题（先读完再动手）

最终目标是**在线验证**：判断一辆自动驾驶车（AV）表现出的社会交互倾向像不像人。

判定由两道串联的弃权机制构成：

- **机制一**：判断某一帧的 **IPV**（Interaction Preference Value，表示交互倾向的标量）
  数值是否携带七个候选之间的判别信息。不携带就弃权，不进入下一关。
- **机制二**：用**人类参照分布（envelope）**判断当前情境下是否有足够的人类样本可比。

已完成：RQ015 冻结机制一；RQ016C 用 2,442,625 行**纯人-人**样本建好 envelope；
RQ017 首次在 AV 数据上算出机制一判据（67,861 帧，两门都过 14,099 帧）。

**本轮（RQ018）问的是全新的一步**：当 AV 的 IPV 超出人类合理范围时，
它的行为有没有呈现出可测的劣化？

这是**正向探索**，目的是判断"这种线索是否存在"，不是闭环验证。**不留出数据集。**

---

## 2. 必须先知道的两件事（不遵守本轮就白跑）

### 2.1 同期运动学关联是循环论证，禁止作为主结论

envelope 的特征合同（权威文件
`reports/studies/RQ016C_human_only_envelope/RQ016C_1_human_only_envelope_20260804T005716Z_7480c173/envelope_model/feature_contract.json`）
里，`numeric_context` 的 22 项**包含**：

```
relative_distance_anchor, relative_speed_anchor, closing_rate_anchor,
heading_difference_anchor, relative_distance_mean_wx, relative_distance_std_wx,
relative_speed_mean_wx, closing_rate_mean_wx, closing_ttc_anchor, apet_online_proxy
```

`support_gate_distance_numeric` 的 12 项同样包含 `closing_ttc_anchor` 与 `apet_online_proxy`。

**即：锚点时刻的每一个运动学危险变量都是 envelope 的条件化特征。**
拿它们当同期结果变量，等于用条件化过的量去解释条件化的残差，结论无效。

**因此帧级结果变量必须取锚点之后的未来窗口**（envelope 未条件化过的部分）。
锚点时刻的 `closing_ttc_anchor` 等只能作为**协变量或描述性对照**出现，
并在报告中明写"它是 envelope 的 context 特征，不作为结果变量"。

### 2.2 14,099 帧不是 14,099 个独立观测

帧嵌套在 case 内，case 嵌套在队伍内（约 19 个有效队伍簇）。
**所有帧级推断必须给出 case 层（主）与 team 层（次）的聚类稳健标准误或
case 层 block bootstrap 置信区间。** 禁止只报朴素 p 值。
若某个效应在朴素口径下显著、在聚类口径下不显著，**必须两个都报，并以聚类口径为准**。

---

## 3. 数据与路径（全部已验证存在，不要另找）

| 用途 | 路径 | 关键列 |
|---|---|---|
| 机制一判据（RQ017 产物） | `data/derived/rq017_onsite_gate/l1_v1/` | `product_row_key`, `status`, `reason_code`, `ipv_log` |
| 机制二区间与支持门（RQ016C dry-run） | `.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet` | `product_row_key`, `lo_80/hi_80/width_80`, `lo_90/hi_90/width_90`, `lo_95/hi_95/width_95`, `mechanism2_gate_ok`, `context_cell` |
| 锚点表（context 与连接键） | `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet` | 66 列；`case_key`, `anchor_frame_index`, `perspective`, `unit_composite_key`, `target_window_start_frame_index`, `target_window_end_frame_index` |
| 稠密逐帧时序（算未来窗口风险） | `.../stage3plus/onsite_anchors_multi/onsite_ipv_timeseries_multi_allvalid.parquet` | 70,317 行；`case_key`, `frame_index`, `time_s`, `ego_x/y/vx/vy/heading`, `counterpart_x/y/vx/vy/heading`, `distance_m`, `closing_rate_mps`, `relative_speed_mps` |
| unit 级结果变量（已连接好，**不要重建**） | `data/derived/onsite_competition/RQ012B_event_harm/stage4plus/unit_analysis_table.parquet` | 267 行 × 72 列；`unit_composite_key`, `case_key`, `analysis_set`, `official_safety/efficiency/comfort/compliance/coordination/comprehensive`, `collision_intervention_deduction_any`, `safety_intervention`, `E**_primary_count` |

`product_row_key` 形如
`case_key=onsite:beijing:T15:A5:native_case:2340|anchor_frame_index=279|perspective=onsite_av_primary|source_dataset=onsite_competition_clean_285`，
可直接解析出 `case_key` / `anchor_frame_index` / `perspective` 与上述各表连接。

监督方已实测的基线数字（你应复算并核对，不一致立即报告）：
- RQ017：`status == OK` 37,520/67,861；两门都过 14,099/67,861
- RQ016C 支持门通过 21,936/67,861
- unit 表：`analysis_set == True` 245/267；`official_safety < 100` 21/267；
  `collision_intervention_deduction_any != 0` 18/267

---

## 4. 曝露变量：异常 IPV（必须保留方向）

分析集：**机制一 `status == OK` 且机制二 `mechanism2_gate_ok == True`**（预期 14,099 帧）。

对每一帧，在 α ∈ {80, 90, 95} 三层各计算：

```
signed_exceedance_a = ipv_log - hi_a      若 ipv_log > hi_a   （上侧越界）
                    = ipv_log - lo_a      若 ipv_log < lo_a   （下侧越界，为负）
                    = 0                   若在区间内
norm_exceedance_a   = signed_exceedance_a / width_a           （跨 context 可比）
outside_a           = (signed_exceedance_a != 0)
```

**必须分别保留上侧与下侧**，不得合并成绝对值。上侧与下侧对应两种不同的不合理表达
（一侧偏激进、一侧偏消极），合并会把两个相反机制抵消掉。RQ012B 上一轮留下过
"passivity→deadlock 未确认的线索"，本轮要保留检验它的能力。

主口径用 α=90，另两层作敏感性。

---

## 5. 结果变量：两条线

### 线 A（主线，帧级，未来窗口风险）

对每个锚点帧，在**同一 `case_key`** 的稠密时序中取锚点之后的窗口，计算：

- `future_min_distance_m`：窗口内 `distance_m` 最小值
- `future_min_ttc_s`：窗口内逐帧 TTC 的最小值（TTC 由 `distance_m` 与 `closing_rate_mps`
  构造；`closing_rate_mps <= 0`（远离）时该帧 TTC 记为 `+inf`，不参与最小值除非全窗口皆是，
  此时记为缺失并计数）
- `future_max_closing_rate_mps`：窗口内 `closing_rate_mps` 最大值

窗口定义用两套并报：
1. **合同窗口**：`[anchor_frame_index, target_window_end_frame_index]`（与 IPV 的目标窗口一致）
2. **固定时长窗口**：锚点后 3 秒（按 `time_s`，不按帧数，因为采样率敏感——
   RQ012B 的 EH-2 已确认该数据在 5Hz/20Hz 下事件计数变化 +88%/−30%）

窗口越出 case 末尾的帧：如实标记并单独计数，不静默丢弃。

**分析**：`norm_exceedance_90`（及上/下侧分开）对上述三个风险量的关联，
带 case 层聚类稳健推断，并控制 `context_cell`。

### 线 B（unit 级，非安全分数）

分母 245（`analysis_set == True`）。用**新曝露**在 unit 层聚合
（`frac_outside_90`、`mean_signed_exceedance_90`、上/下侧分开、`max_abs_exceedance_90`），
对 `official_efficiency` / `official_comfort` / `official_compliance` / `official_coordination` 做关联。

**注意**：RQ012B 已标注这些官方子分数互相吸收严重（以非安全子分数为基线时 R²=0.969）。
因此**必须同时报告**：(a) 单变量关联；(b) 控制 `official_comprehensive` 后的偏关联。
不要只报其中一个。

### 次要（如实报出，不作主线）

`official_safety`、`collision_intervention_deduction_any`、`safety_intervention`。
**功效已知不足**：全 267 个 unit 中 `official_safety < 100` 仅 21 个、碰撞/接管扣分非零仅 18 个。
照常计算并报出效应量与区间，但在报告中明写功效限制，**不得因不显著而声称"无关联"**，
也不得因显著而忽略功效。

---

## 6. 最小负对照（不做完整 battery，但这两条必须做）

1. **case 层标签置换**：在 case 层打乱结果变量与曝露的对应，重跑主分析 ≥200 次，
   给出置换分布下的经验 p 值。
2. **安慰剂曝露**：对每一帧，从其自身 `[lo_90, hi_90]` 区间内均匀抽一个假 IPV，
   按同样公式算 exceedance（构造上恒为 0，故改为从
   `[lo_90 - width_90, hi_90 + width_90]` 均匀抽样），重跑主分析。
   真实曝露的效应应当明显强于安慰剂，否则如实报告"未能区分"。

这两条很便宜，且正是上一轮 RQ012B 判定 null 的依据（标签置换 p=0.743）。
**跳过它们会让本轮的任何正向发现失去意义。**

---

## 7. 硬约束（与流程无关，不得放松）

```
1. RQ007 held_out 不得被解析（污染不可恢复）
2. RQ014 致盲相关的评分字段不得读取；遇到 rating/score/preference/human-score
   字段先停下报告，不要读内容。（注：OnSite 的 official_* 竞赛分数不在此列，可读）
3. 不得静默覆盖已冻结产物或已接受的 decision.md
4. 描述性结果不得写成因果主张
```

补充：

- **禁用词**：全文禁用 `estimability` 与"测出/未测出 IPV"。
  可辩护的表述是"权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息"。
- **不得改动**：`src/sociality_estimation/core/{agent,ipv_estimation,reliability_logdomain}.py`、
  `pipelines/interhub/process_interhub.py`、`configs/ipv_sigma01_exact.json`、
  `data/derived/` 下已有内容、RQ009/RQ016/RQ016C/RQ017 已落盘 run 目录。
- **不得执行任何 git 写操作**（不 commit、不 push、不切分支）。产物由监督方统一入库。
- **本轮不对任何车辆、任何队伍作出判断**，也不得给出"某队更不安全"之类的表述。
- **不要对 `reports/` 做全仓库 `rg`**（宽泛检索会把 RQ003 controlled-access 行拉进上下文）。

### 与已接受结论的关系（重要）

RQ012B 已有**被接受的冻结结论** `RQ012-KC-HARM-NULL`：用 **RQ009 M3 envelope** 衡量的偏离
与官方 harm **无 IPV 特异的基线增量关联**（n=245，Spearman r≈−0.12，p≈0.06，
标签置换 p=0.743，输给 placebo 与 context-only 基线）。

本轮**不是对它的推翻**，而是**不同的曝露定义与不同的分析单元**：
新 envelope 是纯人-人（旧的含 10.9009% AV 目标值），并新增了机制一过滤，
主线换成帧级未来窗口风险。**报告中必须明确写出这一层关系，不得表述为"推翻了旧结论"。**

---

## 8. 交付物（全部写到 `.codex-fleet/rq018-abnormal-ipv-degradation/work/A1/`）

1. `rq018_association.py` —— 可复跑主脚本
2. `key_numbers.json` —— 全部关键数字，每个都带分子/分母/筛选条件/来源文件/列名
3. `frame_level_results.json`、`unit_level_results.json`
4. `negative_controls.json`
5. `data_health.json` —— 数值健康与覆盖自查（见 §9）
6. 报告 `.codex-fleet/rq018-abnormal-ipv-degradation/board/reports/RQ018_1_association.md`

### 报告写法（硬性）

- **开头必须先定位**：这项工作要解决什么问题、整体走到哪一步、本次是哪一环。
  不得直接从增量讲起。假设读者没有跟进过程。
- **不用黑话不用比喻**；必须用专有名词时当场一句话解释。
- **结论与待决事项分开**，需要监督方拍板的单独成节。
- **数字自带口径**：每个百分比都要有分子、分母、筛选条件、来源文件与列名。
  读者无法自行复算的数字等于没给。
- 结尾写 `state: WAITING_ON_COMMANDER` 与 `timestamp_utc:`（用 `date -u +%Y-%m-%dT%H:%M:%SZ`，不要前瞻估计）。

---

## 9. 自查清单（跑完必须逐条给出机器证据）

- [ ] 分析集行数 = 机制一 OK ∩ 机制二 gate_ok，与预期 14,099 一致；不一致就报告差异来源
- [ ] `product_row_key` 连接命中率；未命中行数与原因
- [ ] `ipv_log` 的 NaN/±inf 计数；`width_a > 0` 恒成立；`lo_a < hi_a` 恒成立
- [ ] 未来窗口越界（窗口超出 case 末尾）的帧数与占比
- [ ] `future_min_ttc_s` 全窗口远离导致缺失的帧数与占比
- [ ] 上侧越界帧数、下侧越界帧数、区间内帧数，三者之和 = 分析集行数
- [ ] 每个 case 的帧数分布（min/p25/中位/p75/max），以及 case 数、team 数
- [ ] 朴素 p 值与 case 层聚类 p 值**并列**给出
- [ ] 两条负对照的结果
- [ ] 7 行坐标系异常（`relative_distance_anchor` ≈ 570,761 米，全部来自
      `onsite:shanghai:T10:C4:native_case:2311`）如何处理——照常参与还是剔除，
      两种口径的主结论是否改变

---

## 10. 结项

写完报告后把状态写成 `WAITING_ON_COMMANDER`，**不要自行转 DONE**，
不要执行 git 操作。监督方会独立复算你的数字后再决定入库。

如果中途发现本任务书里的某条设定与数据实际不符（例如某列不存在、行数对不上），
**停下来在报告里写清楚，不要自行改设定继续跑**。
