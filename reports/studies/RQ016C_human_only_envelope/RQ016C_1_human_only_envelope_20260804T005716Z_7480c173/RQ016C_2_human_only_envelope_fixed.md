# RQ016C-H2 fixed human-only envelope

本轮要解决的问题是：在线验证要判断一辆自动驾驶车表现出的社会交互倾向像不像人。IPV（Interaction Preference Value）是表示交互倾向的标量；判定由两道串联弃权机制构成，机制一先判断这一帧的 IPV 数值能不能进入后续比较，机制二再把通过机制一的数值与人类参照分布（envelope）比较。

整体已经走到：RQ015 冻结机制一，RQ016/RQ016C 在准备机制二的人类参照分布。H1 已在纯人-人样本上拟合 envelope，但 H1 把 `vehicle_type_list` 留在类别 context 中；纯人-人参照池没有 `AV` 取值，真实 OnSite 行全部带 `['AV','HV']`，所以 H1 的持久化产物不能用于它唯一的外部打分用途。本次是 H2 修正重跑，只把类别 context 从 H1 的 5 项改为 4 项：`geometry_path_category`、`geometry_path_relation`、`turn_pair_label`、`priority_role`；其余样本口径、fold、支持门、距离特征、alpha 层和 conformal 计算方式沿用 H1。

## 结论

H2 产物已能在真实 OnSite 67,861 行上完成打分路径 dry-run：只加载 `.codex-fleet/rq016c-human-only-envelope/work/H2/envelope_model/rq016c_h2_envelope.pkl`，不重新拟合，输出 `.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet` 和 `.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun_summary.json`。支持门通过率为 32.3249% (21,936/67,861; filter=all OnSite rows; source=.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet; column=mechanism2_gate_ok)。

这个 dry-run 只证明机制二打分管线在真实 OnSite 行上可运行，不构成对任何一辆自动驾驶车的判定。机制一边界的只读核验是：K2 台账 `data/derived/rq015k_logdomain_gate/l1_v1` 中 `artifact_id == onsite_dense_timeseries` 有 281,268 行，`status` 非空为 0/281,268，`reason_code` 非空为 0/281,268，七个 `mse_0..mse_6` 非空计数分别为 `{'mse_0': 0, 'mse_1': 0, 'mse_2': 0, 'mse_3': 0, 'mse_4': 0, 'mse_5': 0, 'mse_6': 0}`；筛选条件为 `artifact_id == onsite_dense_timeseries`，来源列为 `['artifact_id', 'status', 'reason_code', 'mse_0', 'mse_1', 'mse_2', 'mse_3', 'mse_4', 'mse_5', 'mse_6']`。机制一未通过之前，不进入机制二作车辆层面的范围判断。

纯人-人参照池合计 2,442,625 行；来源为 K2 台账 `data/derived/rq015k_logdomain_gate/l1_v1` 的 `product_row_key/status/rq007_split/measurement_role` 与 RQ009 矩阵 `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix` 的 `case_key/anchor_frame_index/perspective/source_dataset/fold/agent_type_pair/av_included` 精确连接后筛选 `status == OK` 且 `agent_type_pair == HV;HV`。split 组成是 development 1,752,509 + guard 690,116，参与计算行中 `rq007_split` 不在 `{development, guard}` 的实测计数为 0。

90% 名义层的纯人-人 envelope coverage = 0.898038，分子分母为 414,837/461,937；筛选条件为纯人-人 test fold 且机制二支持门通过；来源列为 RQ009 矩阵 `target_ipv_future/fold/agent_type_pair` 与 K2 `status/rq007_split`。mean width = 1.238468，分母为同一批机制二支持门通过的 461,937 行。机制二弃权率 = 5.0801% (24,723/486,660; filter=纯人-人 test fold; source=data/derived/rq015k_logdomain_gate/l1_v1 + data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix; columns=status/rq007_split/fold/agent_type_pair)。

## Alpha 层结果

| alpha | coverage | covered / gate-passing rows | mean width | median width | mechanism-two abstention |
|---|---:|---:|---:|---:|---:|
| 80 | 0.796299 | 367,840/461,937 | 0.782266 | 0.757047 | 5.0801% (24,723/486,660; filter=纯人-人 test fold; source=data/derived/rq015k_logdomain_gate/l1_v1 + data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix; columns=status/rq007_split/fold/agent_type_pair) |
| 90 | 0.898038 | 414,837/461,937 | 1.238468 | 1.265956 | 5.0801% (24,723/486,660; filter=纯人-人 test fold; source=data/derived/rq015k_logdomain_gate/l1_v1 + data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix; columns=status/rq007_split/fold/agent_type_pair) |
| 95 | 0.948623 | 438,204/461,937 | 1.714635 | 1.759300 | 5.0801% (24,723/486,660; filter=纯人-人 test fold; source=data/derived/rq015k_logdomain_gate/l1_v1 + data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix; columns=status/rq007_split/fold/agent_type_pair) |

表内 coverage 用小数表示，不写成百分数；coverage 分母是对应 alpha 下机制二支持门通过的纯人-人 test 行，来源列同上。表内机制二弃权率分母是纯人-人 test 行，分子是支持门未通过行。

## 与 H1 对照

| alpha | H2 coverage / H1 coverage | delta | H2 mean width / H1 mean width | H2 median width / H1 median width | H2 abstention / H1 abstention |
|---|---:|---:|---:|---:|---:|
| 80 | 0.796299 / 0.796022 | +0.000277 | 0.782266 / 0.783479 | 0.757047 / 0.760315 | 0.050801 / 0.050801 |
| 90 | 0.898038 / 0.898272 | -0.000234 | 1.238468 / 1.242394 | 1.265956 / 1.271731 | 0.050801 / 0.050801 |
| 95 | 0.948623 / 0.949064 | -0.000442 | 1.714635 / 1.710243 | 1.759300 / 1.755910 | 0.050801 / 0.050801 |

H1 来源为 `.codex-fleet/rq016c-human-only-envelope/work/H1/key_numbers.json`。H1 与 H2 的逐项对照只解释一次规格修正：H1 类别 context 为 `['geometry_path_category', 'geometry_path_relation', 'turn_pair_label', 'vehicle_type_list', 'priority_role']`，H2 类别 context 为 `['geometry_path_category', 'geometry_path_relation', 'turn_pair_label', 'priority_role']`，移除项为 `['vehicle_type_list']`。两者同用纯人-人行筛选、22 项数值 context、`geometry_path_category + priority_role` 支持门分格键、12 项支持门距离特征、80/90/95 三个 alpha 层、RQ009 fold 结构、同一 conformal 计算方式和同一 random state。

## 样本计数自查

fold 计数逐项相符：

| fold | pure human rows | status OK rows before human filter | pure human share |
|---|---:|---:|---:|
| train | 974,984 | 1,290,663 | 75.5413% (974,984/1,290,663; filter=fold == train and status == OK; source=data/derived/rq015k_logdomain_gate/l1_v1 + data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix; columns=status/fold/agent_type_pair) |
| calibration | 481,088 | 629,593 | 76.4125% (481,088/629,593; filter=fold == calibration and status == OK; source=data/derived/rq015k_logdomain_gate/l1_v1 + data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix; columns=status/fold/agent_type_pair) |
| guard_tune | 499,893 | 646,772 | 77.2905% (499,893/646,772; filter=fold == guard_tune and status == OK; source=data/derived/rq015k_logdomain_gate/l1_v1 + data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix; columns=status/fold/agent_type_pair) |
| test | 486,660 | 635,618 | 76.5649% (486,660/635,618; filter=fold == test and status == OK; source=data/derived/rq015k_logdomain_gate/l1_v1 + data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix; columns=status/fold/agent_type_pair) |

这些比例的筛选条件为 K2 精确连接后 `status == OK` 的各 RQ009 fold，分子再筛 `agent_type_pair == HV;HV`；来源列为 `data/derived/rq015k_logdomain_gate/l1_v1` 的 `status` 与 `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix` 的 `fold/agent_type_pair`。

## 特征集裁定执行

代码断言结果：`agent_type_pair`、`av_included`、`vehicle_type_list` 不在 M2 特征列表；`agent_type_pair`、`av_included` 不在支持门分格键；`source_dataset` 不在预测变量；`counterpart_ipv_current/counterpart_ipv_error_current/counterpart_ipv_slope_pre_anchor` 不在特征或支持门距离特征。理据是：`vehicle_type_list` 编码场景中各车辆类型，它对 OnSite 的判别内容正是“这里有一辆自动驾驶车”，而车辆是否为自动驾驶车是被检验对象，不是它所处的情境；保留它会使外部行落入训练中从未出现的类别。

## 类别词表覆盖

四个类别 context 特征全部通过词表覆盖断言。

| scope | column | hit rows | unmatched OnSite values |
|---|---|---:|---|
| categorical_context | `geometry_path_category` | 100.0000% (67,861/67,861; filter=all OnSite rows; source=data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet; column=geometry_path_category; reference=data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix; reference_filter=K2 status OK and agent_type_pair HV;HV) | `[]` |
| categorical_context | `geometry_path_relation` | 100.0000% (67,861/67,861; filter=all OnSite rows; source=data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet; column=geometry_path_relation; reference=data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix; reference_filter=K2 status OK and agent_type_pair HV;HV) | `[]` |
| categorical_context | `turn_pair_label` | 100.0000% (67,861/67,861; filter=all OnSite rows; source=data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet; column=turn_pair_label; reference=data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix; reference_filter=K2 status OK and agent_type_pair HV;HV) | `[]` |
| categorical_context | `priority_role` | 100.0000% (67,861/67,861; filter=all OnSite rows; source=data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet; column=priority_role; reference=data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix; reference_filter=K2 status OK and agent_type_pair HV;HV) | `[]` |
| support_gate_key | `geometry_path_category` | 100.0000% (67,861/67,861; filter=all OnSite rows; source=data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet; column=geometry_path_category; reference=data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix; reference_filter=K2 status OK and agent_type_pair HV;HV) | `[]` |
| support_gate_key | `priority_role` | 100.0000% (67,861/67,861; filter=all OnSite rows; source=data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet; column=priority_role; reference=data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix; reference_filter=K2 status OK and agent_type_pair HV;HV) | `[]` |

## 数值值域对照

下表对 22 项数值 context 比较纯人-人参照池与 OnSite 全量行的 min/p50/max。参照池筛选条件为 K2 精确连接后 `status == OK` 且 `agent_type_pair == HV;HV`，来源 `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix`，列为 22 项数值 context；OnSite 来源 `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet`，筛选条件为 all rows，列为同名 22 项数值 context。OnSite 完全落在参照池 min/max 之外的特征：`[]`。

| numeric context | human min | human p50 | human max | OnSite min | OnSite p50 | OnSite max | complete outside? |
|---|---:|---:|---:|---:|---:|---:|---|
| `elapsed_time_s` | 0.7 | 4.7 | 24.2 | 0.233 | 16.196 | 108.533 | False |
| `history_row_count` | 4 | 10 | 10 | 4 | 10 | 10 | False |
| `ego_vx_anchor` | -62.3065 | -0.0861361 | 51.0156 | -19.0351 | 0.0285841 | 14.3103 | False |
| `ego_vy_anchor` | -61.1719 | -0.012207 | 71.4893 | -23.2703 | -0.00955046 | 15.732 | False |
| `ego_heading_anchor` | -12.5788 | -0.0485981 | 12.4867 | -4.71221 | -1.47069 | 1.57056 | False |
| `counterpart_vx_anchor` | -68.8861 | -0.0804765 | 51.0156 | -41.9826 | 1.78306e-05 | 17.6706 | False |
| `counterpart_vy_anchor` | -53.418 | -0.0109863 | 71.4893 | -51.5942 | -0.000702084 | 41.4805 | False |
| `counterpart_heading_anchor` | -12.5788 | -0.0426268 | 12.4906 | -4.71225 | -1.48517 | 1.56981 | False |
| `relative_dx_anchor` | -187.805 | 0.0022383 | 187.805 | -570762 | 1.26292 | 134.111 | False |
| `relative_dy_anchor` | -164.723 | 0.03228 | 164.723 | -211.037 | 7.15533 | 218.472 | False |
| `relative_distance_anchor` | 0.0741317 | 14.4895 | 223.083 | 2.53947 | 22.0041 | 570762 | False |
| `relative_dvx_anchor` | -63.0139 | 0 | 63.0139 | -39.7675 | -0.00152489 | 18.955 | False |
| `relative_dvy_anchor` | -68.089 | 0 | 68.089 | -49.7473 | -0.0183054 | 41.4853 | False |
| `relative_speed_anchor` | 0 | 1.65768 | 78.9887 | 0 | 2.17475 | 63.6886 | False |
| `closing_rate_anchor` | -78.2168 | 0.161416 | 68.7526 | -63.5354 | -0.00018396 | 30.3434 | False |
| `heading_difference_anchor` | -3.14159 | -0.000314513 | 3.14159 | -3.14155 | -0.00687566 | 3.14158 | False |
| `relative_distance_mean_wx` | 0.237472 | 14.6128 | 227.13 | 2.75949 | 22.0762 | 570762 | False |
| `relative_distance_std_wx` | 0.00106296 | 0.35594 | 42.1145 | 0 | 0.363099 | 6.12755 | False |
| `relative_speed_mean_wx` | 0 | 1.66209 | 39.8694 | 0 | 2.42312 | 25.9083 | False |
| `closing_rate_mean_wx` | -38.2039 | 0.244294 | 38.5617 | -19.1631 | -0.000298936 | 23.9544 | False |
| `closing_ttc_anchor` | 0.0304676 | 11.02 | 20 | 0.207849 | 19.3173 | 20 | False |
| `apet_online_proxy` | 2.82588e-06 | 1.1787 | 19.9289 | 0.00351394 | 5.07238 | 19.496 | False |

## 逐格支撑量

新分格键为 `geometry_path_category + priority_role`，纯人-人参照池共有 12 格，最小格样本数 2,209。

| context cell | rows | cases |
|---|---:|---:|
| `CP|equal` | 2,209 | 34 |
| `CP|priority` | 57,461 | 1,150 |
| `CP|yield` | 57,373 | 1,152 |
| `F|equal` | 3,029 | 36 |
| `F|priority` | 45,283 | 687 |
| `F|yield` | 46,530 | 690 |
| `HO|equal` | 4,817 | 88 |
| `HO|priority` | 4,674 | 147 |
| `HO|yield` | 4,728 | 147 |
| `MP|equal` | 23,424 | 288 |
| `MP|priority` | 1,044,964 | 17,882 |
| `MP|yield` | 1,148,133 | 17,883 |

## OnSite 落格预演

OnSite 源文件 `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet` 读取 67,861 行，列为 `geometry_path_category/priority_role`；落入 9 格，缺格 0 个。落入 OnSite 的格中，人类支撑最小的是 `CP|equal`，人类支撑 2,209 行，OnSite 该格 116 行。

| OnSite context cell | OnSite rows | human support rows | human support cases |
|---|---:|---:|---:|
| `CP|equal` | 116 | 2,209 | 34 |
| `CP|priority` | 2,336 | 57,461 | 1,150 |
| `CP|yield` | 1,488 | 57,373 | 1,152 |
| `F|equal` | 1,535 | 3,029 | 36 |
| `F|priority` | 29,677 | 45,283 | 687 |
| `F|yield` | 14,537 | 46,530 | 690 |
| `MP|equal` | 291 | 23,424 | 288 |
| `MP|priority` | 10,291 | 1,044,964 | 17,882 |
| `MP|yield` | 7,590 | 1,148,133 | 17,883 |

## 真实 OnSite 全量 dry-run

dry-run 只加载持久化模型，不重新拟合；输入来自 `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet`，筛选条件为 all rows，列为 `['case_key', 'anchor_frame_index', 'perspective', 'source_dataset', 'elapsed_time_s', 'history_row_count', 'ego_vx_anchor', 'ego_vy_anchor', 'ego_heading_anchor', 'counterpart_vx_anchor', 'counterpart_vy_anchor', 'counterpart_heading_anchor', 'relative_dx_anchor', 'relative_dy_anchor', 'relative_distance_anchor', 'relative_dvx_anchor', 'relative_dvy_anchor', 'relative_speed_anchor', 'closing_rate_anchor', 'heading_difference_anchor', 'relative_distance_mean_wx', 'relative_distance_std_wx', 'relative_speed_mean_wx', 'closing_rate_mean_wx', 'closing_ttc_anchor', 'apet_online_proxy', 'geometry_path_category', 'geometry_path_relation', 'turn_pair_label', 'priority_role']`，并且刻意不加载 `target_ipv_future`。输出 `.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet` 每行包含 `lo_80/hi_80/width_80`、`lo_90/hi_90/width_90`、`lo_95/hi_95/width_95`、`mechanism2_gate_ok` 和 `context_cell`。

逐格支持门通过率：

| context cell | pass rows | fail rows | pass rate |
|---|---:|---:|---:|
| `CP|equal` | 0 | 116 | 0.0000% (0/116; filter=context_cell == CP|equal; source=.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet; column=mechanism2_gate_ok) |
| `CP|priority` | 50 | 2,286 | 2.1404% (50/2,336; filter=context_cell == CP|priority; source=.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet; column=mechanism2_gate_ok) |
| `CP|yield` | 275 | 1,213 | 18.4812% (275/1,488; filter=context_cell == CP|yield; source=.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet; column=mechanism2_gate_ok) |
| `F|equal` | 368 | 1,167 | 23.9739% (368/1,535; filter=context_cell == F|equal; source=.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet; column=mechanism2_gate_ok) |
| `F|priority` | 13,958 | 15,719 | 47.0331% (13,958/29,677; filter=context_cell == F|priority; source=.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet; column=mechanism2_gate_ok) |
| `F|yield` | 4,783 | 9,754 | 32.9022% (4,783/14,537; filter=context_cell == F|yield; source=.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet; column=mechanism2_gate_ok) |
| `MP|equal` | 8 | 283 | 2.7491% (8/291; filter=context_cell == MP|equal; source=.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet; column=mechanism2_gate_ok) |
| `MP|priority` | 1,387 | 8,904 | 13.4778% (1,387/10,291; filter=context_cell == MP|priority; source=.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet; column=mechanism2_gate_ok) |
| `MP|yield` | 1,107 | 6,483 | 14.5850% (1,107/7,590; filter=context_cell == MP|yield; source=.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet; column=mechanism2_gate_ok) |

区间宽度分布：

| alpha | rows used | min | p05 | p50 | p95 | mean | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 80 | 67,861 | 0.000001 | 0.058457 | 0.383974 | 1.153775 | 0.455534 | 2.011540 |
| 90 | 67,861 | 0.027035 | 0.122959 | 0.852227 | 1.719279 | 0.898734 | 2.367381 |
| 95 | 67,861 | 0.295434 | 0.460064 | 1.341371 | 2.214666 | 1.408280 | 2.492950 |

## 负对照

1. 把 `vehicle_type_list` 放回类别 context 后，词表覆盖断言状态 `EXPECTED_FAIL`，失败输出：

```text
vocabulary_coverage_failed {"column": "vehicle_type_list", "matched_rows": 0, "scope": "categorical_context", "total_rows": 67861, "unmatched_values": ["['AV', 'HV']"]}
```

2. 把 `agent_type_pair` 放回支持门分格键后，OnSite 落格断言状态 `EXPECTED_FAIL`，失败输出：

```text
onsite_landing_check_failed {"cells": 9, "checks": {"cells": true, "min_cell": false, "min_cell_onsite_rows": false, "min_support": false, "missing_cells": false, "rows": true}, "min_row": {}, "missing_cells": ["CP|equal|AV;HV", "CP|priority|AV;HV", "CP|yield|AV;HV", "F|equal|AV;HV", "F|priority|AV;HV", "F|yield|AV;HV", "MP|equal|AV;HV", "MP|priority|AV;HV", "MP|yield|AV;HV"], "missing_rows": 67861, "rows": 67861, "support_columns": ["geometry_path_category", "priority_role", "agent_type_pair"]}
```

## 持久化模型

已拟合 envelope 保存在 `.codex-fleet/rq016c-human-only-envelope/work/H2/envelope_model`。其中 `rq016c_h2_envelope.pkl` 含条件分位数模型、数值 imputer、类别 encoder、支持门 scaler/encoder/kNN tree、全局 conformal 半径和逐格 calibration 半径；`feature_contract.json` 固化列清单；`support_gate.json` 固化支持门规则与逐格支撑量；`HOWTO_score_external_rows.md` 说明如何给外部行打分。打分接口自测从 test fold 取 256 行，只加载持久化产物、不重新拟合，区间边界逐位一致为 `True`，支持门一致为 `True`，判定一致为 `True`，最大边界差 0.0e+00。

## 自查

held_out 断言：参与计算行中 `rq007_split` 不在 `{development, guard}` 的计数为 0；来源列为 `data/derived/rq015k_logdomain_gate/l1_v1` 的 `rq007_split`。本轮没有打开受保护 confirmation 划分文件。

数值健康：test fold 目标列 NaN/正无穷/负无穷计数为 0/0/0；80/90/95 三层负宽度行数为 {'80': 0, '90': 0, '95': 0}; coverage 均落在 [0,1]：`{'80': True, '90': True, '95': True}`。

## 待监督方拍板

本轮没有新增需要监督方拍板的阈值、授权或样本口径。若监督方不接受“车辆类型不是 context 变量”这一裁定执行方式，后果是 OnSite 行会重新遇到训练时未出现类别或支持门缺格问题，需要另开一轮定义新的外部打分合同。

state: WAITING_ON_COMMANDER
timestamp_utc: 2026-08-04T00:59:00Z
