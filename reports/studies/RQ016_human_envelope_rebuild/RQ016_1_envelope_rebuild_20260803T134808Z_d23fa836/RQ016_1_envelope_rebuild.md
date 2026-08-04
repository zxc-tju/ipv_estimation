# RQ016-A1 envelope rebuild

本轮要解决的问题是：机制二的人类 envelope 过去建在含伪零的 RQ009 样本上，非 OK 行在旧目标列里可能表现为精确 0，从而把“权重近均匀，IPV 数值不携带候选间判别信息”和“IPV 恰为中性”混在一起。整体研究链路已经有 RQ015 冻结机制一；本次是机制二重建环节，只在 K2 台账覆盖的 `development + guard` 域内做 A/B 两臂描述性计算。

人类 envelope 是 RQ009 已接受的 context-conditioned split-conformal 区间：先用上下文变量给人类目标 IPV 建条件分位数，再用 calibration fold 的 split-conformal 半径扩展区间；测试行落在支持门外时，机制二弃权。

## 结论

**重建后 coverage。** B 臂是只保留 `status == "OK"` 的重建结果。90% 名义层下，coverage = 0.902689，分子分母为 545,159/603,928；筛选条件：K2 `artifact_id == rq009_feature_matrix`、`measurement_role == target_future`、`rq007_split in {development, guard}`、RQ009 `fold == test`、`status == OK`、机制二支持门通过；来源：`data/derived/rq015k_logdomain_gate/l1_v1` 的 `status/rq007_split/measurement_role` 与 `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix` 的 `fold/target_ipv_future`。A 臂同域基线 coverage = 0.898832，分子分母为 758,857/844,270；B-A = +0.003857。

**重建后区间宽度。** B 臂 90% mean width = 1.300967，median width = 1.332781，分母为机制二支持门通过的 603,928 行；来源同上，区间列由脚本按 `q_lo/q_hi/c_alpha` 在内存生成。A 臂 90% mean width = 1.016189，median width = 0.999071；B 相对 A 的 mean width 变化为 1.300967/1.016189 - 1 = +28.02%。

**机制二自身弃权率。** B 臂 90% 层的机制二弃权率与所有 alpha 层相同，因为支持门不随 alpha 变化：4.9857%，分子分母为 31,690/635,618；筛选条件：同上但分母为 `status == OK` 的 test 行，分子为机制二支持门未通过；来源：`data/derived/rq015k_logdomain_gate/l1_v1` 的 `status` 与 `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix` 的 RQ009 M2 上下文字段。A 臂同域基线机制二弃权率为 5.0200%，分子分母为 44,622/888,892。

**两道门串联后的合并弃权率。** 在 K2 覆盖的 RQ009 test 目标行上，合并弃权率 = 32.0583%，分子分母为 284,964/888,892。其中机制一贡献 253,274/888,892 = 28.4932%，筛选条件为 `status != OK`；机制二贡献 31,690/888,892 = 3.5651%，筛选条件为 `status == OK` 且机制二支持门未通过。来源：`data/derived/rq015k_logdomain_gate/l1_v1` 的 `status/measurement_role/rq007_split` 与 `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix` 的 `fold`。

## 两臂结果

A 臂（域内基线，保留未过门行，含伪零）：

| alpha | coverage | covered/through-mechanism2 | mean width | median width | mechanism2 abstention |
|---|---:|---:|---:|---:|---:|
| 80 | 0.812983 | 686,377/844,270 | 0.591135 | 0.502602 | 5.0200% (44,622/888,892) |
| 90 | 0.898832 | 758,857/844,270 | 1.016189 | 0.999071 | 5.0200% (44,622/888,892) |
| 95 | 0.949887 | 801,961/844,270 | 1.507477 | 1.645892 | 5.0200% (44,622/888,892) |

B 臂（处理组，只保留 `status == "OK"`）：

| alpha | coverage | covered/through-mechanism2 | mean width | median width | mechanism2 abstention |
|---|---:|---:|---:|---:|---:|
| 80 | 0.798378 | 482,163/603,928 | 0.789838 | 0.769365 | 4.9857% (31,690/635,618) |
| 90 | 0.902689 | 545,159/603,928 | 1.300967 | 1.332781 | 4.9857% (31,690/635,618) |
| 95 | 0.949835 | 573,632/603,928 | 1.678538 | 1.775906 | 4.9857% (31,690/635,618) |

A/B 差值：

| alpha | coverage B-A | mean width B-A | mean width B vs A formula | mechanism2 abstention B-A |
|---|---:|---:|---:|---:|
| 80 | -0.014605 | +0.198703 | B/A - 1 = +33.61% | -0.0343% |
| 90 | +0.003857 | +0.284778 | B/A - 1 = +28.02% | -0.0343% |
| 95 | -0.000052 | +0.171061 | B/A - 1 = +11.35% | -0.0343% |

RQ009 已发表数只能作外部参照：`reports/knowledge/RQ009_dynamic_counterpart_conditioned_envelope/decision.md` 的 RQ009-KC-R3 写的是原 RQ009 域。原 RQ009 `metrics_summary.csv` 中 M2 90% coverage = 0.898889，分子分母按 `coverage * n` 取整为 1,087,527/1,209,857；M2 90% 弃权率 = 4.7781%，分子分母为 60,709/1,270,566；M2 相对 M0 的 mean_width 变化 = 1.009483/1.748666 - 1 = -42.27%，Winkler 变化 = 1.423146/2.210261 - 1 = -35.61%。筛选条件：`tier in {M2, M0}`、`alpha_label == 90`、原 RQ009 `fold == test`；来源：`reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/05_evaluation/metrics_summary.csv` 的 `coverage/n/abstained_n/total_n/mean_width/winkler` 和 `reports/knowledge/RQ009_dynamic_counterpart_conditioned_envelope/decision.md`。这个 test 域包含 held_out，因此本轮结果不写作复现或未复现 RQ009。

## 自查

**连接健康。** K2 target_future 台账左连接到 RQ009 feature matrix：命中 4,497,368/4,497,368，未命中 0/4,497,368；K2 `product_row_key` 重复 0，K2 `canonical_key` 重复 0，matrix `product_row_key` 重复 0，one-to-zero-or-one 检查结果为 `True`。来源：`data/derived/rq015k_logdomain_gate/l1_v1` 的 `product_row_key/canonical_key` 与 `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix` 的 `case_key/anchor_frame_index/perspective/source_dataset`。

**held_out 断言。** 本轮所有参与计算的连接后行中，`rq007_split` 不在 `{development, guard}` 的实测计数为 0 行；来源：`data/derived/rq015k_logdomain_gate/l1_v1` 的 `rq007_split`。本轮没有打开任何受保护 confirmation 划分文件。

**两臂只差一个变量。** 两臂共用 alpha 层 `['80', '90', '95']`、RQ009 fold 结构 `['train', 'guard_tune', 'calibration', 'test']`、M2 数值上下文字段 `['elapsed_time_s', 'history_row_count', 'ego_vx_anchor', 'ego_vy_anchor', 'ego_heading_anchor', 'counterpart_vx_anchor', 'counterpart_vy_anchor', 'counterpart_heading_anchor', 'relative_dx_anchor', 'relative_dy_anchor', 'relative_distance_anchor', 'relative_dvx_anchor', 'relative_dvy_anchor', 'relative_speed_anchor', 'closing_rate_anchor', 'heading_difference_anchor', 'relative_distance_mean_wx', 'relative_distance_std_wx', 'relative_speed_mean_wx', 'closing_rate_mean_wx', 'closing_ttc_anchor', 'apet_online_proxy']`、M2 类别上下文字段 `['geometry_path_category', 'geometry_path_relation', 'turn_pair_label', 'agent_type_pair', 'vehicle_type_list', 'av_included', 'priority_role']`、支持门联合格字段 `['geometry_path_category', 'priority_role', 'agent_type_pair']`。`source_dataset` 仅作为连接键/报告键，未作为预测变量。B 臂所有 fold 的行键都是 A 臂子集：`True`。唯一变量是行过滤：A 为 K2 覆盖的全部 target_future 行，B 为 `status == OK` 行。

**每格支撑量。** 支持门联合格为 `geometry_path_category + priority_role + agent_type_pair`，不按数据源分格。A 臂全样本 23 格，最小格样本数 194，低于 `MIN_SUPPORT_L1_PER_L2 = 5` 的格数 0；B 臂全样本 23 格，最小格样本数 178，低于 5 的格数 0。完整逐格计数在 `key_numbers.json` 的 `context_support`。

**负对照。** 我故意把 10 行 test 样本中的一行 `rq007_split` 改成 `held_out` 后重跑 held_out 断言，输出为：`FAIL held_out_guard: invalid_split_rows=1`；负对照状态 `EXPECTED_FAIL`。

**数值健康。** B 臂 test 目标列 NaN/inf 计数为 0/0；A 臂 test 目标列 NaN/inf 计数为 0/0。B 臂 90% 区间负宽度行数 0，病态常数宽度标记 `False`；所有 alpha 的 coverage 均落在 [0, 1]：`{'80': True, '90': True, '95': True}`。

## 待监督方拍板

本轮没有请求新的执行授权或阈值选择。唯一需要监督方确认的是是否接受本报告采用的 M2-only 支持门：判断依据是任务书禁止 M3/M4 旧 IPV-conditioning 通道，本脚本因此在训练模型和支持门距离特征中均排除了 `counterpart_ipv_current/counterpart_ipv_error_current/counterpart_ipv_slope_pre_anchor`；不接受的后果是需要另开一轮，明确允许使用 RQ009 原支持门中的旧 counterpart IPV 通道，但那会引入任务书要求避免的第二条污染路径。

state: WAITING_ON_COMMANDER
timestamp_utc: 2026-08-03T13:48:08Z

---

# 监督方附录

以下由监督方追加，**不修改上文 A1 原文**，A1 的原状态行保留在上方。

## A. 独立复算结果

监督方未采信 A1 报告的数字，用独立脚本从原始产物重算。以下全部一致：

| 量 | 监督方独立算得 | 分母与筛选 |
|---|---:|---|
| RQ009 matrix `fold=test` 行 | 1,270,566（键重复 0） | `03_features/matrix`，`fold=test` |
| K2 台账 `measurement_role=target_future` 行 | 4,497,368（键重复 0） | `l1_v1`，`artifact_id=rq009_feature_matrix` |
| A 臂域（test 行落在台账覆盖域内） | 888,892 / 1,270,566 = 69.9603% | 上二者按 `product_row_key` 内连接 |
| 未覆盖 | 381,674 | 与 L1 记录的整案级排除一致 |
| 覆盖行 split 组成 | development 633,924 + guard 254,968 | 列 `rq007_split` |
| **held_out 行** | **0**（实测） | 独立证实 A1 的红线断言 |
| 机制一弃权 | 253,274 / 888,892 = 28.4932% | 列 `status != OK` |
| └ 拆分 | NEAR_UNIFORM 249,726 / NO_IPV_EFFECT 3,324 / SOLVER_FAILURE 224 | 列 `reason_code` |
| B 臂域（机制一通过） | 635,618 | 列 `status == OK` |

A1 报告内所有分子分母的算术自洽性亦逐条核过（过门后行数、coverage 分数、
合并弃权率的两段拆分），全部对得上。

## B. 头条结论的机制证据（A1 未做，监督方补做）

A1 的头条是「B 臂 90% 平均区间宽度比 A 臂宽 28.02%」。为确认这不是其 conformal
实现的产物，监督方做了两项**独立于该实现**的检查：

**1. 直接数零点**（源 `03_features/matrix` 的 `target_ipv_future`）：

| 臂 | n | 恰为 0 | 占比 | 四分位距 IQR | 标准差 |
|---|---:|---:|---:|---:|---:|
| A（含伪零） | 888,892 | 192,221 | 21.6248% | **0.0493** | 0.4278 |
| B（只过门） | 635,618 | 99,908 | 15.7182% | **0.2017** | 0.4895 |

机制一未通过的行中，恰为 0 的占 92,313/253,274 = 36.4479%。
**去掉伪零后目标的四分位距变为约 4.09 倍**——这就是区间变宽的直接机制。

**2. 与模型无关的边际宽度**（不训练任何模型，直接取目标分位数）：

| alpha | A 臂边际宽度 | B 臂边际宽度 | B/A | A1 报的条件宽度 B/A |
|---|---:|---:|---:|---:|
| 80% | 0.907596 | 1.249083 | 1.3763 | 1.336 |
| 90% | 1.782905 | 2.120901 | 1.1896 | 1.280 |
| 95% | 2.320291 | 2.350562 | 1.0130 | 1.113 |

两列同向、同量级、且同样随 alpha 递减。**故 +28% 不是实现产物。**

## C. 必须记入的边界（A1 未写明）

**去掉机制一失败行并没有消除目标的零点聚集，只是把它减半。**
`|y| < 1e-6` 的行占比：A 臂 42.39%（分母 888,892）、B 臂 **29.63%**（分母 635,618）。
重建后的 envelope 仍然建在一个高度零点聚集的目标上。任何基于本轮结果的下游主张
都必须带上这条限定。

## D. 一处未追查的差异（按速度原则不追）

监督方在台账覆盖域上算得精确零点 **192,221**（99,908 OK + 92,313 非 OK）；
RQ015-L1 已发表 **192,271**（99,938 OK + 92,333 非 OK）。差 50 行 = 0.026%。

两者源文件不同：L1 用 RQ009 predictions 的 `y`，监督方用 feature matrix 的
`target_ipv_future`。已排除 endpoint-nudge 解释（`|y| <= 1e-10` 为 375,548，量级完全不同）。
对本轮任何结论无影响。**但两个数不得互相替代引用。**

## E. 对 A1 上报待决项的裁定：接受 M2-only 支持门

A1 在训练模型与支持门距离特征中都排除了 `counterpart_ipv_current` /
`counterpart_ipv_error_current` / `counterpart_ipv_slope_pre_anchor`。
监督方核对 RQ009 `02_process/04_calibration/calibration.py:141-157`：其
`GATE_DISTANCE_NUMERIC` 共 15 项，确实**包含**这 3 个 M3 通道列。故 A1 的支持门用了
12 项而非 15 项——**这是真实偏离，A1 主动披露，属正确行为**。

**裁定：接受。** 判据：(1) 那 3 列由旧估计器算出、本身携带伪零，留在门里等于让本轮
要清除的缺陷从门这条路重新进来；(2) 两臂用同一个门，A/B 对比不受影响，而 A/B 对比是
本轮全部结论的来源；(3) RQ009 自己记录 M3 ≈ M2（配对 90% Winkler 差 −0.0002，
case 聚类 p=0.863），该通道无可测量增益。

顺带核实：A1 的 M2 特征集与 `calibration.py:95-127` 的 `BASE_NUMERIC_CONTEXT`（22 项）
与 `BASE_CATEGORICAL_CONTEXT`（7 项）**逐字相同**，未擅自增删；`source_dataset`
未作预测变量。

## F. 对已接受研究的影响（不处置，只记录）

本轮结果对 RQ009 已接受主张 RQ009-KC-R3 中「宽度较全局基线 −42.3%」这条锐度主张有影响：
现有证据表明其中一部分来自伪零而非真实的条件信息。

PI 于 2026-08-03 已裁定「本轮不处置 RQ009，只记边界」。按同一口径处理：
**不改动** `reports/knowledge/RQ009_dynamic_counterpart_conditioned_envelope/decision.md`。
本条仅作记录，不构成对 RQ009 的处置。

state: COMMANDER_VERIFIED
timestamp_utc: 2026-08-03T15:34:33Z
