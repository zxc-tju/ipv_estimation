# RQ016C-H1 human-only envelope

本轮要解决的问题是：在线验证要判断 OnSite 自动驾驶车的 IPV（Interaction Preference Value，表示交互倾向的标量）是否落在人类参照分布范围内；RQ016 已经完成一次机制二 envelope 重建，但其中一部分目标值来自自动驾驶车自身。整体链路已经走到机制一由 RQ015 冻结、机制二需要给外部行提供参照分布的阶段；本次只执行 PI 在 2026-08-04 的裁定：只用纯人-人样本重建一个将来供 OnSite 自动驾驶车打分的人类 envelope。

本轮 envelope 是 context-conditioned split-conformal 区间：用 RQ009/RQ016 的 22 个数值 context 和删去 `agent_type_pair`、`av_included` 后的 5 个类别 context 拟合条件分位数；用 calibration fold 计算 split-conformal 半径；支持门分格键只保留 `geometry_path_category + priority_role`。`source_dataset` 只用于连接与溯源，不作为预测变量。

## 结论

纯人-人参照池合计 2,442,625 行；来源为 K2 台账 `data/derived/rq015k_logdomain_gate/l1_v1` 的 `product_row_key/status/rq007_split/measurement_role` 与 RQ009 矩阵 `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix` 的 `case_key/anchor_frame_index/perspective/source_dataset/fold/agent_type_pair/av_included` 精确连接后筛选 `status == OK` 且 `agent_type_pair == HV;HV`。split 组成是 development 1,752,509 + guard 690,116，参与计算行中 `rq007_split` 不在 `{development, guard}` 的实测计数为 0。

90% 名义层的纯人-人 envelope coverage = 0.898272，分子分母为 414,945/461,937；筛选条件为纯人-人 test fold 且机制二支持门通过；来源列为 RQ009 矩阵 `target_ipv_future/fold/agent_type_pair` 与 K2 `status/rq007_split`。mean width = 1.242394，分母为同一批机制二支持门通过的 461,937 行。机制二弃权率 = 5.0801%，分子分母为 24,723/486,660；筛选条件为纯人-人 test fold，分子为支持门未通过。

与 RQ016 旧 B 臂对照，90% coverage 差为 -0.004417，旧 B 臂分子分母为 545,159/603,928，本轮分子分母为 414,945/461,937；mean width 差为 -0.058574，ratio = 1.242394/1.300967 - 1 = -4.50%；机制二弃权率差为 +0.000944，旧 B 臂分子分母为 31,690/635,618，本轮分子分母为 24,723/486,660。旧 B 臂来源：`reports/studies/RQ016_human_envelope_rebuild/RQ016_1_envelope_rebuild_20260803T134808Z_d23fa836/key_numbers.json` 的 `arms.B_status_ok.metrics`。这只是两个 envelope 的描述性对照；样本口径和特征集都变了，不写成因果主张。

## Alpha 层结果

| alpha | coverage | covered / gate-passing rows | mean width | median width | mechanism-two abstention |
|---|---:|---:|---:|---:|---:|
| 80 | 0.796022 | 367,712/461,937 | 0.783479 | 0.760315 | 5.0801% (24,723/486,660) |
| 90 | 0.898272 | 414,945/461,937 | 1.242394 | 1.271731 | 5.0801% (24,723/486,660) |
| 95 | 0.949064 | 438,408/461,937 | 1.710243 | 1.755910 | 5.0801% (24,723/486,660) |

表内 coverage 用小数表示，不写成百分数；coverage 分母是对应 alpha 下机制二支持门通过的纯人-人 test 行，来源列同上。表内机制二弃权率分母是纯人-人 test 行，分子是支持门未通过行。

## 与 RQ016 旧 B 臂对照

| alpha | coverage delta | mean width delta | mean width ratio | abstention delta |
|---|---:|---:|---:|---:|
| 80 | -0.002356 | -0.006359 | 0.783479/0.789838 - 1 = -0.81% | +0.000944 |
| 90 | -0.004417 | -0.058574 | 1.242394/1.300967 - 1 = -4.50% | +0.000944 |
| 95 | -0.000771 | +0.031705 | 1.710243/1.678538 - 1 = +1.89% | +0.000944 |

旧 B 臂是 `status == OK` 但包含 `agent_type_pair == AV;HV` 与 `HV;HV` 的 envelope；本轮只含 `HV;HV`，并且从特征和支持门分格键中移除了 `agent_type_pair` 与 `av_included`。表中 width ratio 是宽度数值的比值，不是样本占比；其数据来源为本轮 `key_numbers.json` 与 `reports/studies/RQ016_human_envelope_rebuild/RQ016_1_envelope_rebuild_20260803T134808Z_d23fa836/key_numbers.json`。

## 样本计数自查

fold 计数逐项相符：

| fold | pure human rows | status OK rows before human filter | pure human share |
|---|---:|---:|---:|
| train | 974,984 | 1,290,663 | 75.5413% (974,984/1,290,663) |
| calibration | 481,088 | 629,593 | 76.4125% (481,088/629,593) |
| guard_tune | 499,893 | 646,772 | 77.2905% (499,893/646,772) |
| test | 486,660 | 635,618 | 76.5649% (486,660/635,618) |

这些比例的筛选条件为 K2 精确连接后 `status == OK` 的各 RQ009 fold，分子再筛 `agent_type_pair == HV;HV`；来源列为 `data/derived/rq015k_logdomain_gate/l1_v1` 的 `status` 与 `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix` 的 `fold/agent_type_pair`。

## 特征集裁定执行

代码断言结果：`agent_type_pair` 与 `av_included` 既不在 M2 特征列表，也不在支持门分格键；`source_dataset` 不在预测变量；`counterpart_ipv_current/counterpart_ipv_error_current/counterpart_ipv_slope_pre_anchor` 不在特征或支持门距离特征。理据是：ego 自己的车辆类型是被检验对象，不是它所处的情境；若把它作为 context，OnSite 的自动驾驶车行会带来训练时未出现的类别，并且旧支持门会因为 `AV;HV` 分格缺失而全量弃权。

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

OnSite 只做 context 落格，不打分，因为 OnSite 还没有机制一判据。源文件 `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet` 读取 67,861 行，列为 `geometry_path_category/priority_role`；落入 9 格，缺格 0 个。落入 OnSite 的格中，人类支撑最小的是 `CP|equal`，人类支撑 2,209 行，OnSite 该格 116 行。

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

## 持久化模型

已拟合 envelope 保存在 `.codex-fleet/rq016c-human-only-envelope/work/H1/envelope_model`。其中 `rq016c_h1_envelope.pkl` 含条件分位数模型、数值 imputer、类别 encoder、支持门 scaler/encoder/kNN tree、全局 conformal 半径和逐格 calibration 半径；`feature_contract.json` 固化列清单；`support_gate.json` 固化支持门规则与逐格支撑量；`HOWTO_score_external_rows.md` 说明如何给外部行打分。打分接口自测从 test fold 取 256 行，只加载持久化产物、不重新拟合，区间边界逐位一致为 `True`，支持门一致为 `True`，判定一致为 `True`，最大边界差 0.0e+00。

## 自查

held_out 断言：参与计算行中 `rq007_split` 不在 `{development, guard}` 的计数为 0；来源列为 `data/derived/rq015k_logdomain_gate/l1_v1` 的 `rq007_split`。本轮没有打开受保护 confirmation 划分文件。

负对照：故意把 `agent_type_pair` 放回 M2 类别特征后执行特征集断言，输出为 `feature_contract_failed forbidden_in_features=['agent_type_pair']`；负对照状态 `EXPECTED_FAIL`。

数值健康：test fold 目标列 NaN/正无穷/负无穷计数为 0/0/0；80/90/95 三层负宽度行数为 {'80': 0, '90': 0, '95': 0}; coverage 均落在 [0,1]：`{'80': True, '90': True, '95': True}`。

## 待监督方拍板

本轮没有新增需要监督方拍板的阈值、授权或样本口径。若监督方不接受“ego 自己的车辆类型不是 context 变量”这一裁定执行方式，后果是 OnSite 自动驾驶车行会重新遇到训练时未出现类别或支持门缺格问题，需要另开一轮定义新的外部打分合同。

state: WAITING_ON_COMMANDER
timestamp_utc: 2026-08-04T00:43:31Z
