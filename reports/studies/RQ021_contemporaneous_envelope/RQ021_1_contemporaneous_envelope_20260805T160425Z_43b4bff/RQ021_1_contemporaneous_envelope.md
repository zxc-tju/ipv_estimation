# RQ021-E1 同期 IPV 人类参照区间重跑

这项研究要解决的是自动驾驶车在线表现出的社会交互倾向是否落在人类参照范围内。IPV（Interaction Preference Value）是表示交互倾向的标量；判定先由冻结的机制一判断该帧数值是否携带七个候选之间的判别信息，再由机制二把通过机制一的数值与人类参照区间比较。

整体已经走到：RQ016C-H2 建成纯人-人参照区间，RQ017 在 OnSite 的 67,861 个锚点行上落定机制一结果，RQ018/RQ019 使用旧参照区间形成了已接受的描述性关联结果。本次是目标量校正环节，按 PI 2026-08-05 裁定，只把目标列从锚点之后 `[t+3,t+6]` 的 `target_ipv_future` 换为锚点当下 `[t-9,t]` 的 `ipv_log`；特征、fold、支持门、alpha 层和 split-conformal 流程均沿用 H2。

## 结论

D1 与 D2 均未触发事前停止阈值，步骤 4、5 已完成。新 envelope 使 α=90 两门交集中的下侧/区间内/上侧分组从 1,998/9,401/2,700 变为 519/12,711/869。RQ018 的分布中部与四个危险阈值占比方向未反转；RQ019 的两个速度量和强制动帧占比方向也未反转，但部分 case 层证据变弱，须由监督方决定是否更新已接受主张。

新 envelope 已训练并持久化到 `.codex-fleet/rq021-contemporaneous-envelope/work/E1/envelope_model`。训练目标合同为 `ipv_log`，来自 `data/derived/rq015k_logdomain_gate/l1_v1` 的 `ipv_log` 列；上下文、fold 与旧对照目标来自 `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix`。目标行集等价断言通过：K2 全 8,994,736 行中 `status == OK` 为 6,405,292 行、`ipv_log` 非空为 6,405,292 行、`status == OK` 且 `ipv_log` 为空 0 行、`status != OK` 且 `ipv_log` 非空 0 行。

## 步骤 1：同期 `ipv_log` envelope

参照池为 2,442,625 行 = development 1,752,509 + guard 690,116，held_out 计数为 0；筛选条件为 K2 精确连接后 `status == OK`、RQ009 `agent_type_pair == HV;HV`、`rq007_split in {development, guard}`，来源列为 `data/derived/rq015k_logdomain_gate/l1_v1` 的 `product_row_key/status/rq007_split/ipv_log` 与 `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix` 的连接键、`fold/agent_type_pair`。

fold 行数为 train 974,984、calibration 481,088、guard_tune 499,893、test 486,660，与任务书逐项一致。

| nominal layer | coverage | covered / gate-passing test rows | mean width | median width | mechanism-two abstention |
|---|---:|---:|---:|---:|---:|
| 80% | 0.800254 | 369,667/461,937 | 1.451625 | 1.449786 | 5.0801% (24,723/486,660) |
| 90% | 0.902798 | 417,036/461,937 | 1.865128 | 1.896902 | 5.0801% (24,723/486,660) |
| 95% | 0.955654 | 441,452/461,937 | 2.158838 | 2.206113 | 5.0801% (24,723/486,660) |

表内 coverage 的分母是各层纯人-人 test fold 且支持门通过的 461,937 行；来源列为 `data/derived/rq015k_logdomain_gate/l1_v1` 的 `status/rq007_split/ipv_log` 与 `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix` 的 `fold/agent_type_pair`。机制二弃权百分数的分子为支持门未通过行，分母为纯人-人 test fold 的 486,660 行，来源列为 12 项支持门距离特征及 `geometry_path_category/priority_role`。

## 步骤 2：循环性诊断

边际基线使用同一参照池、同一 train/calibration/test fold、同一支持门和同一 split-conformal 计算，只把条件分位数预测替换为 train fold 的全局分位数。

| target | conditional 90% mean width | marginal 90% mean width | D1 width ratio | test-fold q50 R² |
|---|---:|---:|---:|---:|
| `ipv_log` | 1.865128 | 2.345319 | 0.795255 | 0.209355 |
| `target_ipv_future`（D3） | 1.238468 | 2.099517 | 0.589882 | 0.219939 |

D1 的宽度分子与分母均使用纯人-人 test fold 且支持门通过的 461,937 行；来源列为 `data/derived/rq015k_logdomain_gate/l1_v1` 的 `ipv_log/status/rq007_split` 与 `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix` 的 `target_ipv_future/fold/agent_type_pair` 及 H2 context。D2 主口径使用纯人-人 test fold 全部 486,660/486,660 个有限值，定义为 `1 - sum((y-q50)^2)/sum((y-mean(y))^2)`；点预测是条件分位数模型的 q50 头，不使用边际模型。

事前阈值检查：D1 `0.795255 < 0.25` 为 `False`；D2 `0.209355 >= 0.60` 为 `False`。阈值未在结果后改动，也没有新增例外。

## 任务书指定的旧值对照

以下是任务书第 3.3 节指定必须报告的冻结对照，来源为 `reports/studies/RQ016C_human_only_envelope/RQ016C_1_human_only_envelope_20260804T005716Z_7480c173/RQ016C_1_human_only_envelope.md`：80% coverage 0.796022（367,712/461,937），mean width 0.783479；90% coverage 0.898272（414,945/461,937），mean width 1.242394，median width 1.271731；95% coverage 0.949064（438,408/461,937），mean width 1.710243。三个层的机制二弃权均为 5.0801%（24,723/486,660；筛选条件=纯人-人 test fold，分子=支持门未通过；来源列=旧 H2/H1 支持门的 12 项距离特征和 `geometry_path_category/priority_role`）。

D3 为保证“同特征、同 fold、同流程”，直接加载四类别 H2 持久化模型 `.codex-fleet/rq016c-human-only-envelope/work/H2/envelope_model/rq016c_h2_envelope.pkl` 计算，因此与上段冻结报告数字分开记录，不混用。

## 强不变量

- 纯人-人 test fold 机制二弃权率精确为 5.0801%（24,723/486,660；筛选条件=纯人-人 test fold；来源=`data/derived/rq015k_logdomain_gate/l1_v1` + `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix`；列=`status/rq007_split/fold/agent_type_pair` 与支持门特征）。
- OnSite 支持门通过精确为 32.3249%（21,936/67,861；筛选条件=OnSite 全部行；来源=`data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet`；列=12 项支持门距离特征、`geometry_path_category/priority_role`）。该不变量在步骤 3 先以只执行支持门的方式核对，步骤 4 才加载持久化模型生成区间文件。
- OnSite 两门交集精确为 20.7763%（14,099/67,861；筛选条件=OnSite 全部行中 `status == OK and mechanism2_gate_ok`；来源=`data/derived/rq017_onsite_gate/l1_v1` 的 `product_row_key/status` + `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet` 的支持门特征）。
- OnSite 落 9 格，缺格 0；纯人-人池 12 格，最小格 `CP|equal` 2,209 行，OnSite 该格 116 行。
- 四项类别 context 词表命中均为 100.0000%（每项 67,861/67,861；筛选条件=OnSite 全部行；来源=`data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet`；列=`geometry_path_category/geometry_path_relation/turn_pair_label/priority_role`；参照池筛选同步骤 1）。

## 负对照

1. 把目标改回 `target_ipv_future` 后运行“目标必须为 `ipv_log`”合同断言：`EXPECTED_FAIL`。

```text
target_contract_failed target_column=target_ipv_future expected=ipv_log
```

2. 把 `vehicle_type_list` 放回类别 context 后运行 OnSite 词表覆盖断言：`EXPECTED_FAIL`。

```text
vocabulary_coverage_failed {"column": "vehicle_type_list", "matched_rows": 0, "scope": "categorical_context", "total_rows": 67861, "unmatched_values": ["['AV', 'HV']"]}
```

两项均实际失败；第二项失败输出显示命中 0/67,861。

## 自查与边界

- 参与计算行中 `rq007_split` 不在 `{development, guard}` 的计数为 0，来源 `data/derived/rq015k_logdomain_gate/l1_v1` 的 `rq007_split`；没有解析 RQ007 held_out。
- 读取的 K2 列为 `['artifact_id', 'product_row_key', 'canonical_key', 'measurement_role', 'status', 'reason_code', 'rq007_split', 'context_cell_key', 'gate_applicable', 'ipv_log']`；RQ017 只读取 `['product_row_key', 'status', 'ipv_log']`。未读取 RQ014 致盲评分字段。
- test 目标 `ipv_log` 的 NaN/正无穷/负无穷计数为 0/0/0；80/90/95 三层负宽度行数为 {'80': 0, '90': 0, '95': 0}。
- 未修改冻结机制一、受保护源码、`data/derived/`、RQ009/RQ016/RQ016C/RQ017/RQ018/RQ019 已落盘目录；未执行 Git 写操作；本机运行，未投 Slurm/HPC。

## 步骤 4：OnSite 新打分

新持久化模型只加载、不重新拟合，输出 `.codex-fleet/rq021-contemporaneous-envelope/work/E1/onsite_scoring_dryrun.parquet`。schema 含 `lo/hi/width_80/90/95`、`mechanism2_gate_ok`、`context_cell`；67,861 行中支持门通过 21,936 行。与 RQ017 的冻结 `status` 连接后两门交集仍为 14,099 行。

α=90 两门交集分组由旧 `lower/inside/upper = 1,998/9,401/2,700` 变为新 `519/12,711/869`，两组分母均为 14,099；筛选条件为 `status == OK and mechanism2_gate_ok == True`，来源为 RQ017 的 `product_row_key/status/ipv_log` 与本轮 OnSite 打分的 `lo_90/hi_90/width_90/mechanism2_gate_ok`。

## 步骤 5：RQ018 新旧对照

未来最小 TTC 的四分位数和中位数（秒）：

| 组别 | 旧 q25 / q50 / q75 | 新 q25 / q50 / q75 | 有效行旧 / 新 | 方向是否改变 |
|---|---:|---:|---:|---|
| 下侧越界 | 4.105 / 7.505 / 12.748 | 4.090 / 6.667 / 11.490 | 1,819 / 472 | 见组间比较 |
| 区间内 | 4.089 / 8.807 / 22.376 | 4.181 / 8.876 / 21.864 | 8,739 / 11,669 | 见组间比较 |

下侧减区间内的中位数差旧为 -1.301 s，新为 -2.208 s，均为负，整体分布中部方向未改变。

危险阈值帧占比与 case-bootstrap 95% CI（差值=下侧−区间内）：

| TTC 阈值 | 旧下侧 / 区间内 | 旧差值 [CI] | 新下侧 / 区间内 | 新差值 [CI] | 方向改变？ |
|---|---:|---:|---:|---:|---|
| < 1.0 s | 1.3744% (25/1,819) / 2.6204% (229/8,739) | -0.0125 [-0.0231, -0.0025] | 0.8475% (4/472) / 2.2795% (266/11,669) | -0.0143 [-0.0258, -0.0017] | 否 |
| < 1.5 s | 2.8037% (51/1,819) / 6.6827% (584/8,739) | -0.0388 [-0.0578, -0.0219] | 2.1186% (10/472) / 5.8531% (683/11,669) | -0.0373 [-0.0597, -0.0110] | 否 |
| < 2.0 s | 5.2776% (96/1,819) / 9.8524% (861/8,739) | -0.0457 [-0.0696, -0.0227] | 4.6610% (22/472) / 8.8439% (1,032/11,669) | -0.0418 [-0.0728, -0.0103] | 否 |
| < 3.0 s | 12.5344% (228/1,819) / 16.9012% (1,477/8,739) | -0.0437 [-0.0805, -0.0043] | 12.7119% (60/472) / 15.8540% (1,850/11,669) | -0.0314 [-0.0833, +0.0315] | 否 |

四个阈值的帧占比方向均未改变：下侧越界组仍低于区间内。TTC<2 s 的新 CI 仍不含 0；TTC<3 s 的新 CI 跨 0，因此该阈值的证据强度下降，但结论方向没有反转。TTC<1 与 <1.5 的旧 CI 不在已接受 supervisor JSON 中，本表旧 CI 是 RQ021 用同一 1,000 次 case 重采样方法补算，已在机器对照中标为补充值。

来源：旧/新复核 `.codex-fleet/rq021-contemporaneous-envelope/work/E1/rq018_old_extended_verification.json` / `.codex-fleet/rq021-contemporaneous-envelope/work/E1/rq018_rerun/rq018_supervisor_verification.json`；筛选为 α=90 两门交集且未来 TTC 非空，列为 `case_key/future_min_ttc_s/band`。

## 步骤 5：RQ019 新旧对照

非 scripted、固定 3 s、α=90 的分布中部：

| 对手方结果 | 旧 lower / inside | 旧倍数或差值 [case-bootstrap CI] | 新 lower / inside | 新倍数或差值 [case-bootstrap CI] | 方向改变？ |
|---|---:|---:|---:|---:|---|
| 锚点降速 km/h | 2.893 / 1.243 | 2.328× [+1.013, +2.615] | 2.740 / 1.329 | 2.062× [+0.104, +3.421] | 否 |
| 速度极差 km/h | 4.786 / 2.758 | 1.735× [+1.227, +2.707] | 5.527 / 2.931 | 1.886× [+0.991, +3.511] | 否 |
| 总航向变化 度 | 1.667 / 0.759 | 差 +0.908 [+0.041, +1.932] | 1.842 / 0.783 | 差 +1.059 [-0.148, +3.226] | 否 |

强制动原始帧占比与 case 层等权 p 值：

| 阈值 | 旧 lower / inside；p_case；bootstrap CI | 新 lower / inside；p_case；bootstrap CI | 方向改变？ |
|---|---|---|---|
| < -2 m/s² | 5.3375% (2,867/53,714) / 9.1900% (20,643/224,624); p=0.0040; [-0.0605, -0.0140] | 4.1304% (570/13,800) / 8.5584% (26,552/310,246); p=0.0985; [-0.0648, -0.0226] | 否 |
| < -3 m/s² | 2.8317% (1,521/53,714) / 7.0202% (15,769/224,624); p=0.0045; [-0.0627, -0.0219] | 2.9130% (402/13,800) / 6.3118% (19,582/310,246); p=0.4704; [-0.0532, -0.0155] | 否 |
| < -4 m/s² | 2.5450% (1,367/53,714) / 6.3813% (14,334/224,624); p=0.0038; [-0.0571, -0.0220] | 2.6304% (363/13,800) / 5.7625% (17,878/310,246); p=0.5299; [-0.0487, -0.0144] | 否 |

两个速度量仍为下侧越界组约两倍，方向不变；总航向变化差仍为正，但新 case-bootstrap CI 跨 0，继续不作转向主张。三个强制动阈值的帧占比方向均不变，且 pooled case-bootstrap CI 仍低于 0；不过 case 等权 p 值由旧的均小于 0.006 变为新 0.0985/0.4704/0.5299，独立单位层证据明显变弱。

来源：速度/航向旧新复核 `reports/studies/RQ019_counterpart_burden/RQ019_1_counterpart_burden_20260805T014215Z_7b9f47b/rq019_supervisor_verification.json` / `.codex-fleet/rq021-contemporaneous-envelope/work/E1/rq019_rerun/rq019_supervisor_verification.json`；强制动旧新 `reports/studies/RQ019_counterpart_burden/RQ019_1_counterpart_burden_20260805T014215Z_7b9f47b/distribution_results.json` / `.codex-fleet/rq021-contemporaneous-envelope/work/E1/rq019_rerun/distribution_results.json`。强制动筛选为 α=90、非 scripted、固定 3 s；分子列为 `acceleration < threshold` 的原始帧数，分母为同组有效 acceleration 帧。

### RQ019 输入合同修正

复制的原脚本在分析开始前硬编码旧 α=90 分组数 `2700/1998/9401`，新输入实际为 `869/519/12711`，首次按仅改路径运行因此如实失败。为完成同一统计流程，只把这项输入数据合同更新为新实测计数；14,099 行、231 case、19 team、分析逻辑、模型、阈值、随机种子、1,000 次 bootstrap 与 1,000 次置换均未改。首次失败输出保存在 `.codex-fleet/rq021-contemporaneous-envelope/work/E1/rq019_rerun/initial_input_contract_failure.txt`，修正可由脚本 diff 复核。

## 待监督方拍板

本执行轮已完成重训、OnSite 打分和 RQ018/RQ019 原流程重跑，但不自行替换三份已接受 `decision.md`。

- 选项 A：在独立复核前继续保留旧 RQ018/RQ019 主张与旧输入证据链。本轮产物保持 `WAITING_ON_COMMANDER`；后果是手稿暂不引用同期 `ipv_log` envelope 的新数字。
- 选项 B：监督方独立复算本轮新输入，并据新证据更新 RQ018/RQ019 decision。判断依据是主要方向未反转，但 RQ018 的 TTC<3 s CI 改为跨 0，RQ019 三个强制动 case 等权 p 值均不再低于 0.05；后果是主张措辞需按新证据强度收窄。

若不拍板，旧已接受产物保持不变，本轮新结果不会自动进入手稿。

state: WAITING_ON_COMMANDER
timestamp_utc: 2026-08-05T16:04:25Z
