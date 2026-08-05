# RQ018-1：异常 IPV 与后续行为风险的关联性正向探索

## 1. 工作定位与当前进度

这项研究最终要在线验证一辆自动驾驶车表现出的社会交互倾向是否落在人类合理范围内。IPV（Interaction Preference Value）是表示交互倾向的标量；机制一判断某帧 IPV 数值是否携带七个候选之间的判别信息，机制二判断该情境是否有足够人类参照。RQ015 已冻结机制一，RQ016C 已建立纯人-人参照区间，RQ017 已在 OnSite 自动驾驶车数据上完成两门计算。本次 RQ018-A1 是下一环：只做一次探索性、描述性分析，检查越出人类参照区间的 IPV 与锚点后行为风险及 unit 级竞赛结果是否有关联。

本轮已完成指定输入复算、未来窗口构造、case/team 聚类稳健推断、unit 级关联、两项负对照和数据健康自查。结果不作因果解释，也不对任何车辆或队伍作判断。

## 2. 分析合同与避免循环论证

锚点时刻的相对距离、接近率、TTC 与 PET proxy 都已用于人类参照区间的情境条件化，因此本报告不把它们当同期结果。帧级结果全部来自锚点后窗口：合同窗口 `[anchor_frame_index, target_window_end_frame_index]`，以及按 `time_s` 定义的锚点起 3 秒窗口。TTC 逐帧按 `distance_m / closing_rate_mps` 计算；`closing_rate_mps <= 0` 的帧不进入最小值，全窗口均不接近时 TTC 记为缺失。

主曝露为 90% 人类参照区间的有符号、区间宽度归一化越界量。上侧和下侧用两个非负幅度变量联合进入方向模型；80% 与 95% 仅用于敏感性。帧级模型控制 `context_cell` 固定效应，case 聚类为主口径，team 聚类为次口径；同时列出朴素 p 值以显示忽略嵌套会造成什么差异。

## 3. 基线复算与覆盖

| 数字 | 口径、来源与列 |
|---|---|
| 机制一 OK：37,520/67,861 = 55.2895% | 筛选 `status == OK`；来源 `data/derived/rq017_onsite_gate/l1_v1`；列 `status` |
| 机制二通过：21,936/67,861 = 32.3249% | 筛选 `mechanism2_gate_ok == True`；来源 `.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet`；列 `mechanism2_gate_ok` |
| 两门交集：14,099/67,861 = 20.7763% | 筛选 `status == OK AND mechanism2_gate_ok == True`；来源 `data/derived/rq017_onsite_gate/l1_v1` + `.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet`；连接列 `product_row_key` |
| unit 基础集：245/267 = 91.7603% | 筛选 `analysis_set == True`；来源 `data/derived/onsite_competition/RQ012B_event_harm/stage4plus/unit_analysis_table.parquet`；列 `analysis_set` |
| unit 曝露有定义：225/245 = 91.8367% | `analysis_set == True` 且至少一帧两门通过；来源 RQ017 + M2 + 锚点表 + unit 表；列 `status, mechanism2_gate_ok, unit_composite_key, analysis_set`。其余 unit 保持缺失，不赋零值 |

`product_row_key` 从 RQ017 到机制二命中 67,861/67,861 = 100.0000%；两门交集解析到锚点表命中 14,099/14,099 = 100.0000%。两者分别使用 `product_row_key` 和 `case_key + anchor_frame_index + perspective`，来源见上表。

## 4. 帧级主结果：90% 参照区间

系数单位为曝露增加一个参照区间宽度时结果变量的变化；距离和 TTC 下降表示风险增加，最大接近率上升表示风险增加。方括号为 case 聚类 95% 置信区间。

| 窗口 | 结果 | 有符号单斜率 | 上侧幅度（方向联合模型） | 下侧幅度（方向联合模型） |
|---|---|---|---|---|
| contract | future_min_distance_m | 0.3051 [-0.1565, 0.7667]; p_naive=0.0101, p_case=0.1941, p_team=0.2828 | 1.1290 [0.2695, 1.9884]; p_naive=<0.0001, p_case=0.0103, p_team=0.0289 | 0.6525 [-0.0895, 1.3945]; p_naive=0.0002, p_case=0.0845, p_team=0.0373 |
| contract | future_min_ttc_s | 116443.9515 [-130290.3339, 363178.2369]; p_naive=0.8909, p_case=0.3534, p_team=0.3545 | 647801.2561 [-672640.0151, 1968242.5272]; p_naive=0.6101, p_case=0.3347, p_team=0.3087 | 346405.3560 [-406287.8870, 1099098.5991]; p_naive=0.7696, p_case=0.3654, p_team=0.3916 |
| contract | future_max_closing_rate_mps | -0.2707 [-0.3941, -0.1472]; p_naive=<0.0001, p_case=<0.0001, p_team=0.0010 | -0.5589 [-0.7959, -0.3220]; p_naive=<0.0001, p_case=<0.0001, p_team=<0.0001 | -0.0644 [-0.2864, 0.1576]; p_naive=0.3544, p_case=0.5680, p_team=0.6315 |
| fixed_3s | future_min_distance_m | 0.5213 [0.0294, 1.0131]; p_naive=<0.0001, p_case=0.0379, p_team=0.0972 | 1.8642 [0.9584, 2.7700]; p_naive=<0.0001, p_case=<0.0001, p_team=0.0008 | 1.0396 [0.2954, 1.7839]; p_naive=<0.0001, p_case=0.0064, p_team=0.0009 |
| fixed_3s | future_min_ttc_s | -415.8669 [-1319.8411, 488.1073]; p_naive=0.6337, p_case=0.3656, p_team=0.5027 | -3343.1372 [-9787.4982, 3101.2238]; p_naive=0.0065, p_case=0.3078, p_team=0.3214 | -2795.5107 [-8174.4521, 2583.4308]; p_naive=0.0301, p_case=0.3069, p_team=0.3274 |
| fixed_3s | future_max_closing_rate_mps | -0.2037 [-0.3375, -0.0700]; p_naive=0.0002, p_case=0.0030, p_team=0.0121 | -0.4616 [-0.7266, -0.1965]; p_naive=<0.0001, p_case=0.0007, p_team=<0.0001 | -0.0960 [-0.3525, 0.1606]; p_naive=0.2497, p_case=0.4620, p_team=0.4457 |

### 4.1 80% / 95% 敏感性

下表只列有符号单斜率；上、下侧联合模型的完整朴素、case 聚类和 team 聚类结果在 `frame_level_results.json`。

| α | 窗口 | 结果 | case 聚类效应 [95% CI] 与三种 p 值 |
|---:|---|---|---|
| 80 | contract | future_min_distance_m | 0.0690 [-0.0357, 0.1738]; p_naive=0.0096, p_case=0.1955, p_team=0.3242 |
| 80 | contract | future_min_ttc_s | -19517.4537 [-62565.2469, 23530.3394]; p_naive=0.9189, p_case=0.3726, p_team=0.2530 |
| 80 | contract | future_max_closing_rate_mps | -0.0906 [-0.1431, -0.0381]; p_naive=<0.0001, p_case=0.0008, p_team=0.0007 |
| 80 | fixed_3s | future_min_distance_m | 0.1499 [0.0333, 0.2665]; p_naive=<0.0001, p_case=0.0120, p_team=0.0615 |
| 80 | fixed_3s | future_min_ttc_s | 243.7172 [-266.6585, 754.0930]; p_naive=0.2231, p_case=0.3477, p_team=0.3401 |
| 80 | fixed_3s | future_max_closing_rate_mps | -0.0818 [-0.1336, -0.0301]; p_naive=<0.0001, p_case=0.0021, p_team=0.0009 |
| 95 | contract | future_min_distance_m | 0.4717 [-1.1712, 2.1146]; p_naive=0.2719, p_case=0.5721, p_team=0.6322 |
| 95 | contract | future_min_ttc_s | 1177680.8735 [-1233290.1409, 3588651.8878]; p_naive=0.7005, p_case=0.3368, p_team=0.3231 |
| 95 | contract | future_max_closing_rate_mps | -0.6730 [-1.1213, -0.2247]; p_naive=<0.0001, p_case=0.0034, p_team=0.0184 |
| 95 | fixed_3s | future_min_distance_m | 1.0564 [-0.6811, 2.7939]; p_naive=0.0144, p_case=0.2322, p_team=0.2956 |
| 95 | fixed_3s | future_min_ttc_s | -1421.2154 [-4583.8328, 1741.4019]; p_naive=0.6509, p_case=0.3768, p_team=0.5073 |
| 95 | fixed_3s | future_max_closing_rate_mps | -0.6379 [-1.1251, -0.1507]; p_naive=0.0015, p_case=0.0105, p_team=0.0234 |

### 4.2 TTC 的 `log1p` 数值健康敏感性

原始 TTC 在接近率非常接近零但仍为正时可达到极大有限值。任务书原始尺度结果仍在主表；下表额外对 `log(1 + future_min_ttc_s)` 拟合同一模型，不设接近率阈值、不改变帧的纳入。负系数表示越界幅度增加时未来最小 TTC 变短。

| α | 窗口 | 有符号单斜率 | 上侧幅度（联合模型） | 下侧幅度（联合模型） |
|---:|---|---|---|---|
| 80 | contract | 0.0376 [0.0038, 0.0714]; p_naive=<0.0001, p_case=0.0292, p_team=0.0376 | 0.0186 [-0.0108, 0.0480]; p_naive=0.0099, p_case=0.2145, p_team=0.2010 | -0.0922 [-0.1378, -0.0465]; p_naive=<0.0001, p_case=<0.0001, p_team=0.0010 |
| 80 | fixed_3s | 0.0365 [0.0075, 0.0654]; p_naive=<0.0001, p_case=0.0138, p_team=0.0333 | 0.0212 [-0.0050, 0.0475]; p_naive=0.0026, p_case=0.1128, p_team=0.1426 | -0.0837 [-0.1385, -0.0289]; p_naive=<0.0001, p_case=0.0029, p_team=0.0071 |
| 90 | contract | 0.1167 [0.0422, 0.1913]; p_naive=<0.0001, p_case=0.0023, p_team=0.0052 | -0.0146 [-0.1732, 0.1441]; p_naive=0.7194, p_case=0.8564, p_team=0.8525 | -0.2311 [-0.3495, -0.1126]; p_naive=<0.0001, p_case=0.0002, p_team=0.0013 |
| 90 | fixed_3s | 0.0847 [0.0256, 0.1437]; p_naive=0.0013, p_case=0.0052, p_team=0.0089 | -0.0220 [-0.1874, 0.1435]; p_naive=0.5546, p_case=0.7939, p_team=0.7143 | -0.2016 [-0.3443, -0.0589]; p_naive=<0.0001, p_case=0.0058, p_team=0.0097 |
| 95 | contract | 0.2981 [0.0335, 0.5628]; p_naive=0.0023, p_case=0.0274, p_team=0.0548 | -0.0722 [-0.6512, 0.5067]; p_naive=0.6334, p_case=0.8060, p_team=0.8063 | -0.5774 [-0.9099, -0.2450]; p_naive=<0.0001, p_case=0.0007, p_team=0.0006 |
| 95 | fixed_3s | 0.2556 [0.0258, 0.4853]; p_naive=0.0072, p_case=0.0294, p_team=0.0655 | -0.0153 [-0.6409, 0.6103]; p_naive=0.9121, p_case=0.9617, p_team=0.9529 | -0.5098 [-0.8935, -0.1261]; p_naive=0.0001, p_case=0.0094, p_team=0.0074 |

## 5. 负对照

case 标签置换共 200 次：整段打乱 case 的曝露轨迹与结果轨迹的对应；不同帧数按相对锚点位置最近邻对齐，然后原样重拟合含 `context_cell` 固定效应与 case 聚类推断的帧模型。安慰剂曝露按任务书从每帧 `[lo_90-width_90, hi_90+width_90]` 均匀抽取，使用固定种子；安慰剂 p 值比较真实 case 聚类 |t| 与 200 次安慰剂 |t| 分布。

| 窗口 | 结果 | 标签置换 p（有符号） | 安慰剂 p（有符号） | 标签置换 p（下侧幅度） | 安慰剂 p（下侧幅度） |
|---|---|---:|---:|---:|---:|
| contract | future_min_distance_m | 0.2836 | 0.2040 | 0.3383 | 0.0945 |
| contract | future_min_ttc_s | 0.1393 | 0.8756 | 0.5174 | 0.9751 |
| contract | future_max_closing_rate_mps | 0.0050 | 0.0050 | 0.6766 | 0.6070 |
| contract | future_log1p_min_ttc | 0.0199 | 0.0100 | 0.0149 | 0.0050 |
| fixed_3s | future_min_distance_m | 0.0846 | 0.0348 | 0.0896 | 0.0199 |
| fixed_3s | future_min_ttc_s | 0.4229 | 0.8010 | 0.1592 | 0.3184 |
| fixed_3s | future_max_closing_rate_mps | 0.0249 | 0.0100 | 0.6766 | 0.4527 |
| fixed_3s | future_log1p_min_ttc | 0.0249 | 0.0050 | 0.0249 | 0.0100 |

## 6. unit 级非安全子分数

unit 基础分母是 `analysis_set == True` 的 245 个；其中 225 个至少有一帧两门通过，20 个曝露无定义而不进入关联。unit 越界量先除以各帧区间宽度，再在 unit 内聚合。下表以 team 为 block 给出 Spearman 或控制 `official_comprehensive` 后的偏 Spearman 95% bootstrap 区间；`U` 为单变量，`P` 为偏关联。

| 结果 | unit 曝露 | U: rho [team-block 95% CI] | P: partial rho [team-block 95% CI] |
|---|---|---|---|
| official_efficiency | frac_outside_90 | -0.151 [-0.296, -0.000] (n=225, team-block p=0.0500) | -0.210 [-0.305, -0.037] (n=225, team-block p=0.0200) |
| official_efficiency | mean_signed_exceedance_90 | -0.019 [-0.172, 0.127] (n=225, team-block p=0.8212) | 0.005 [-0.116, 0.138] (n=225, team-block p=0.8292) |
| official_efficiency | mean_upper_exceedance_90 | -0.104 [-0.258, 0.081] (n=225, team-block p=0.2597) | -0.223 [-0.332, -0.077] (n=225, team-block p=0.0020) |
| official_efficiency | mean_lower_exceedance_90 | -0.134 [-0.263, 0.037] (n=225, team-block p=0.1139) | -0.253 [-0.337, -0.154] (n=225, team-block p=0.0020) |
| official_efficiency | max_abs_exceedance_90 | -0.152 [-0.318, 0.043] (n=225, team-block p=0.1139) | -0.304 [-0.381, -0.190] (n=225, team-block p=0.0020) |
| official_comfort | frac_outside_90 | -0.000 [-0.187, 0.184] (n=225, team-block p=0.9371) | 0.015 [-0.169, 0.197] (n=225, team-block p=0.8032) |
| official_comfort | mean_signed_exceedance_90 | -0.021 [-0.159, 0.124] (n=225, team-block p=0.7972) | -0.018 [-0.150, 0.123] (n=225, team-block p=0.7752) |
| official_comfort | mean_upper_exceedance_90 | 0.037 [-0.139, 0.208] (n=225, team-block p=0.6973) | 0.045 [-0.122, 0.220] (n=225, team-block p=0.6094) |
| official_comfort | mean_lower_exceedance_90 | 0.027 [-0.076, 0.137] (n=225, team-block p=0.6354) | 0.039 [-0.058, 0.142] (n=225, team-block p=0.4156) |
| official_comfort | max_abs_exceedance_90 | 0.057 [-0.046, 0.164] (n=225, team-block p=0.3157) | 0.069 [-0.050, 0.192] (n=225, team-block p=0.2458) |
| official_compliance | frac_outside_90 | 0.061 [-0.082, 0.227] (n=225, team-block p=0.4336) | 0.098 [-0.030, 0.237] (n=225, team-block p=0.1219) |
| official_compliance | mean_signed_exceedance_90 | 0.029 [-0.106, 0.166] (n=225, team-block p=0.7493) | 0.037 [-0.101, 0.188] (n=225, team-block p=0.6114) |
| official_compliance | mean_upper_exceedance_90 | -0.029 [-0.175, 0.126] (n=225, team-block p=0.7053) | -0.014 [-0.149, 0.144] (n=225, team-block p=0.8971) |
| official_compliance | mean_lower_exceedance_90 | -0.004 [-0.136, 0.124] (n=225, team-block p=0.9031) | 0.020 [-0.098, 0.132] (n=225, team-block p=0.6553) |
| official_compliance | max_abs_exceedance_90 | -0.013 [-0.125, 0.094] (n=225, team-block p=0.8472) | 0.013 [-0.095, 0.125] (n=225, team-block p=0.7433) |
| official_coordination | frac_outside_90 | 0.000 [-0.181, 0.197] (n=225, team-block p=0.9890) | 0.079 [-0.044, 0.214] (n=225, team-block p=0.2198) |
| official_coordination | mean_signed_exceedance_90 | -0.077 [-0.228, 0.046] (n=225, team-block p=0.2777) | -0.081 [-0.242, 0.070] (n=225, team-block p=0.3596) |
| official_coordination | mean_upper_exceedance_90 | 0.031 [-0.120, 0.206] (n=225, team-block p=0.7213) | 0.078 [-0.044, 0.209] (n=225, team-block p=0.2038) |
| official_coordination | mean_lower_exceedance_90 | 0.135 [-0.005, 0.296] (n=225, team-block p=0.0699) | 0.227 [0.122, 0.338] (n=225, team-block p=0.0020) |
| official_coordination | max_abs_exceedance_90 | 0.092 [-0.060, 0.263] (n=225, team-block p=0.2597) | 0.177 [0.061, 0.289] (n=225, team-block p=0.0020) |

## 7. 次要安全结果与功效边界

全 267 个 unit 中，`official_safety < 100` 为 21/267 = 7.8652%（筛选 `official_safety < 100`，来源 `data/derived/onsite_competition/RQ012B_event_harm/stage4plus/unit_analysis_table.parquet`，列 `official_safety`）；`collision_intervention_deduction_any != 0` 为 18/267 = 6.7416%（同一来源，列 `collision_intervention_deduction_any`）；`safety_intervention != 0` 为 8/267 = 2.9963%（同一来源，列 `safety_intervention`）。阳性事件稀少，以下结果只给效应和区间；不显著不能解释成没有关联，显著也必须按低功效探索看待。

| 次要结果 | unit 曝露 | Spearman rho [team-block 95% CI] |
|---|---|---|
| official_safety | frac_outside_90 | -0.133 [-0.247, 0.018] (n=225, team-block p=0.0879) |
| official_safety | mean_signed_exceedance_90 | 0.011 [-0.149, 0.162] (n=225, team-block p=0.8292) |
| official_safety | mean_upper_exceedance_90 | -0.125 [-0.247, 0.034] (n=225, team-block p=0.1379) |
| official_safety | mean_lower_exceedance_90 | -0.140 [-0.231, -0.030] (n=225, team-block p=0.0180) |
| official_safety | max_abs_exceedance_90 | -0.128 [-0.220, 0.004] (n=225, team-block p=0.0659) |
| collision_intervention_deduction_any | frac_outside_90 | 0.123 [-0.043, 0.239] (n=225, team-block p=0.1518) |
| collision_intervention_deduction_any | mean_signed_exceedance_90 | -0.029 [-0.192, 0.147] (n=225, team-block p=0.6893) |
| collision_intervention_deduction_any | mean_upper_exceedance_90 | 0.120 [-0.056, 0.246] (n=225, team-block p=0.1898) |
| collision_intervention_deduction_any | mean_lower_exceedance_90 | 0.156 [0.042, 0.251] (n=225, team-block p=0.0120) |
| collision_intervention_deduction_any | max_abs_exceedance_90 | 0.146 [0.009, 0.239] (n=225, team-block p=0.0380) |
| safety_intervention | frac_outside_90 | 0.122 [0.028, 0.179] (n=225, team-block p=0.0021) |
| safety_intervention | mean_signed_exceedance_90 | -0.014 [-0.174, 0.153] (n=225, team-block p=0.8940) |
| safety_intervention | mean_upper_exceedance_90 | 0.114 [-0.089, 0.235] (n=225, team-block p=0.2990) |
| safety_intervention | mean_lower_exceedance_90 | 0.142 [0.064, 0.233] (n=225, team-block p=0.0021) |
| safety_intervention | max_abs_exceedance_90 | 0.121 [-0.000, 0.209] (n=225, team-block p=0.0546) |

## 8. 数值健康、覆盖与坐标异常

- 90% 层上侧 2,700 帧、下侧 1,998 帧、区间内 9,401 帧，合计 14,099/14,099。筛选为两门交集；来源 RQ017 `ipv_log` 与机制二区间列 `lo_90, hi_90, width_90`。
- 合同窗口越出 case 末尾 0/14,099 = 0.0000%；3 秒窗口越出 case 末尾 734/14,099 = 5.2060%。来源 `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_ipv_timeseries_multi_allvalid.parquet`，列 `case_key, frame_index, time_s`；越界窗口使用 case 末尾前可见部分并保留标记。
- 合同窗口因全窗口 `closing_rate_mps <= 0` 而 TTC 缺失 1,211/14,099 = 8.5893%；3 秒窗口为 461/14,099 = 3.2697%。来源 `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_ipv_timeseries_multi_allvalid.parquet`，列 `distance_m, closing_rate_mps`。
- TTC 数值边界：稠密表最小正接近率为 4.915e-09 m/s；合同窗口 `future_min_ttc_s > 10^6` 的行数为 7，3 秒窗口为 0。来源 `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_ipv_timeseries_multi_allvalid.parquet`，列 `distance_m, closing_rate_mps`。本轮保留任务书公式，不私设接近率下限；极大但有限的 TTC 会使原始尺度 OLS 对这些行敏感。
- case 帧数分布 min/p25/median/p75/max = 1/20.0/50.0/90.0/222；case 数 231，team 数 19。筛选为两门交集；来源 RQ017 + M2 + 锚点表，列 `status, mechanism2_gate_ok, case_key`。
- 全锚点表坐标异常为 7/67,861 = 0.0103%，全部来自 `onsite:shanghai:T10:C4:native_case:2311`；来源 `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet`，列 `case_key, relative_distance_anchor`。照常保留时，这 7 行中进入两门交集的是 0/14,099 = 0.0000%；因此剔除口径与照常参与口径的主模型系数最大绝对差为 0。
- `ipv_log` NaN/+inf/-inf 均为 0；80/90/95 三层均满足全部 14,099 行 `width > 0` 且 `lo < hi`。逐项机器证据见 `data_health.json`。

## 9. 与 RQ012B 已接受结论的关系

RQ012B 的冻结结论 `RQ012-KC-HARM-NULL` 使用含自动驾驶车目标值的旧 RQ009 M3 参照，分析的是 unit 级官方 harm，并判定没有相对 placebo 与 context-only 基线的 IPV 特异增量关联。本轮使用纯人-人参照、增加机制一过滤，并把主分析单元换成帧级未来窗口风险。因此本轮是不同曝露定义和不同分析单元的探索，不构成对旧结论的推翻。

## 10. 结论

1. **下侧越界与更短的未来 TTC 是本轮最直接对应‘劣化’方向的线索。** 在 90% 层，`log(1+TTC)` 的下侧幅度系数在合同窗口为 -0.2311 [-0.3495, -0.1126]（case 聚类 p=0.0002），3 秒窗口为 -0.2016 [-0.3443, -0.0589]（case 聚类 p=0.0058）。80% 与 95% 层同方向。case-block 标签置换 p 分别为 0.0149 与 0.0249，安慰剂 p 分别为 0.0050 与 0.0100；按预先要求的 case 聚类、标签置换、安慰剂三项同时低于 0.05 的规则，两窗口总体判定为通过。这是关联性线索，不是因果结论。
2. **上侧越界没有显示行为劣化，多个结果反而指向更大的未来距离或更低的最大接近率。** 例如 3 秒窗口有符号斜率对最小距离为 0.5213 m [0.0294, 1.0131]（case 聚类 p=0.0379）。这可由场景选择、控制后的剩余结构或行为差异解释，本轮不能把它写成‘上侧异常更安全’。
3. **unit 级结果不形成一致的跨子分数劣化模式。** 控制 `official_comprehensive` 后，效率与部分越界汇总呈负相关，而协调性对下侧幅度呈正相关；舒适与合规多数区间跨零。官方综合分与子分数存在强机械吸收，加上本轮同时查看多种曝露，unit 结果只适合作为后续假设来源。
4. 次要安全结果的阳性 unit 很少，虽然下侧幅度与若干安全结果同向，当前区间与多重比较不足以支持‘有关联’或‘无关联’的稳定判断。监督方复算前，本报告保持探索性证据状态；任何后续手稿表述都需要独立数据复现与单独接受。

## 11. 待监督方决定

1. 是否将通过 case 聚类、标签置换和安慰剂三项检查的方向性线索列为下一数据集的预注册目标。依据是第 4–5 节三种证据同时成立；若不推进，本轮只保留为描述性产物。
2. unit 基础集 245 个中有 20 个没有任何两门通过帧。本轮按缺失处理，避免把“无可分析帧”编码为零越界；监督方如需另一个政策口径，应另立分析而不是覆盖本轮。

## 12. 可复跑产物

- 脚本：`.codex-fleet/rq018-abnormal-ipv-degradation/work/A1/rq018_association.py`
- 关键数字：`.codex-fleet/rq018-abnormal-ipv-degradation/work/A1/key_numbers.json`
- 帧级结果：`.codex-fleet/rq018-abnormal-ipv-degradation/work/A1/frame_level_results.json`
- unit 级结果：`.codex-fleet/rq018-abnormal-ipv-degradation/work/A1/unit_level_results.json`
- 负对照：`.codex-fleet/rq018-abnormal-ipv-degradation/work/A1/negative_controls.json`
- 数据健康：`.codex-fleet/rq018-abnormal-ipv-degradation/work/A1/data_health.json`

state: WAITING_ON_COMMANDER
timestamp_utc: 2026-08-04T16:35:46Z

---

# 监督方附录

以下由监督方追加，**不修改上文执行方原文**，执行方原状态行保留在上方。
复核脚本 `work/A1_supervisor/rq018_supervisor_verification.py`（不使用执行方脚本，
从原始数据重建），机器证据 `work/A1_supervisor/rq018_supervisor_verification.json`。

## A. 独立复算：逐位一致

| 量 | 监督方独立算得 | 与执行方一致 |
|---|---:|---|
| 机制一 `status == OK` | 37,520/67,861 | 是 |
| 机制二 `mechanism2_gate_ok` | 21,936/67,861 | 是 |
| 两门交集 | **14,099** | 是 |
| case 数 | 231 | 是 |
| 上侧/下侧/区间内 | 2,700 / 1,998 / 9,401 | 是 |
| 合同窗口 TTC 缺失 | 1,211 | 是 |
| `log1p(TTC)` 下侧系数 | **−0.2311**，case 聚类 SE **0.0601**，p=**0.0002** | 是 |
| `log1p(TTC)` 上侧系数 | −0.0146，p=0.8564 | 是 |

**执行方的计算没有问题，可复现。以下四条是对其解释与完整性的修正。**

## B. IPV 符号语义（执行方未澄清，下游极易读反）

从受保护源码确认：`src/sociality_estimation/core/agent.py:1193` 为
`util = cos(ipv) × 自身代价 + sin(ipv) × 交互代价`；:859-861 同构
（`weight_deviation/travel = cos(ipv)`，`weight_inter = sin(ipv)`）。
候选网格 `[-3,-2,-1,0,1,2,3] × π/8`，即 −67.5° 至 +67.5°。

因此 **IPV 越大 = 越看重对方代价 = 越合作让行；IPV 越负 = 反向压制对方 = 越竞争激进**。
**下侧越界是「比人类更激进」，不是「更消极」。** 下侧组 IPV 中位 −0.7838 弧度 ≈ −44.9°
（此处 cos=0.709、sin=−0.705，即同时在意自身代价并主动降低对方效用）。

### 下侧越界不等价于「IPV 为负」

- 判据是 `ipv_log < lo_90`，而 `lo_90` 是**随情境变化的**人类区间下界，
  全部 14,099 行均为负（中位 **−0.5427**，5–95 分位 **[−1.1033, −0.0518]**）。
- 实测下侧越界 1,998 行**全部** `ipv_log < 0`（1,998/1,998 = 100%）；
- **但反向不成立**：区间内还有 **3,611/9,401 = 38.41%** 的行 `ipv_log < 0` 却未越界，
  因为人类在那些情境下同样取负值。全部负 IPV 行共 5,609，其中仅 1,998（35.62%）构成下侧越界。

## C. 尾部风险分解：负系数来自安全端压缩，不是危险端增加（**核心修正**）

执行方把 `log1p(TTC)` 的下侧负系数写成「本轮最直接对应劣化方向的线索」。
监督方补做了执行方未做的分位数与阈值分解，**该解释不成立**。

合同窗口 TTC 分位（分母为该组有定义 TTC 的帧）：

| 组 | n | 1% | 5% | 25% | 50% | 75% |
|---|---:|---:|---:|---:|---:|---:|
| 下侧（更激进） | 1,819 | 0.856 | 1.950 | **4.105** | 7.505 | 12.75 |
| 区间内 | 8,739 | 0.724 | 1.297 | **4.089** | 8.807 | 22.38 |

**25% 分位两组几乎相同（4.105 对 4.089）；差异全部在中位与上尾。**

危险阈值以下的帧占比：

| 阈值 | 下侧越界 | 区间内 | 方向 |
|---|---:|---:|---|
| TTC < 1.0 s | 25/1,819 = **1.37%** | 229/8,739 = 2.62% | 更少 |
| TTC < 1.5 s | 51/1,819 = **2.80%** | 584/8,739 = 6.68% | 更少 |
| TTC < 2.0 s | 96/1,819 = **5.28%** | 861/8,739 = 9.85% | 更少 |
| TTC < 3.0 s | 228/1,819 = **12.53%** | 1,477/8,739 = 16.90% | 更少 |

case 层 bootstrap（1,000 次，按 case 重采样）：
- TTC<2.0s 占比差（下侧−区间内）= **−0.0457**，95% CI **[−0.0696, −0.0227]**，不含 0
- TTC<3.0s 占比差 = **−0.0437**，95% CI **[−0.0805, −0.0043]**，不含 0

**即：向下越界的帧在每一个危险阈值上都更少出现短 TTC，且差异稳健。**
`log(1+TTC)` 回归量的是中心位置移动，把「上尾被压缩」与「下尾变危险」记成了同一个负系数。

## D. 执行方未报告的完整性问题

1. **TTC 缺失与曝露相关**。按组：上侧 370/2,700 = **13.70%**、下侧 179/1,998 = **8.96%**、
   区间内 662/9,401 = **7.04%**。执行方在数据健康中只报了总数 1,211，未做分组。
   缺失的是「全窗口都在远离」即最安全的帧，剔除它们会让下侧**看起来更危险**——
   偏倚方向与 §C 结论相反，故 §C 的结论只会被低估。
2. **主口径无结果被弱化**。任务书指定的结果变量是 `future_min_ttc_s`；在该原始尺度上
   TTC 三种曝露、两个窗口**全部不显著**（p_case 0.31–0.37）。`log1p` 是执行方看到数值病态后
   事后加的变换，报告应先声明主口径无结果。
3. **结果变量互相矛盾未解释**。同为下侧越界，最小距离方向相反且显著
   （fixed_3s 系数 +1.0396，p_case=0.0064，即距离更大）。
4. **多重比较**。产物中共 288 个 p 值（帧级 216、负对照 72）。`p_case=0.0002` 大致可承受
   Bonferroni，但用于背书的标签置换 p=0.0149（200 次中 3 次）不能。

## E. 监督方改写后的结论

**本轮未观察到「异常 IPV 对应更高风险」的证据。**

可辩护的表述：**在 OnSite 数据上，自动驾驶车 IPV 低于该情境人类参照下界（即比人类更激进）时，
其后续最小 TTC 的分布整体左移（中位 7.51 s 对 8.81 s），但危险阈值以下的帧反而更少
（TTC<2 s：5.28% 对 9.85%，case bootstrap 95% CI 不含 0）。分布左移来自安全端长 TTC 的减少，
不是危险端的增加。**

对 PI 三个问题的回答：
1. **是否造成危险**——未发现，方向相反（§C）。
2. **是否更容易发生事故**——功效不足，答不了。全 267 unit 中 `official_safety < 100` 21 个、
   `collision_intervention_deduction_any != 0` 18 个。
3. **是否存在各方面劣化**——unit 级四个非安全子分数无一致模式（效率负、协调性正、舒适与合规跨零）。

**不得使用「导致」等因果表述；本轮为描述性关联，设计不支持因果。**
本轮与 RQ012B 冻结结论 `RQ012-KC-HARM-NULL` 是不同曝露定义与不同分析单元的探索，
**不构成推翻**，且方向上与其 null 不矛盾。

state: COMMANDER_VERIFIED
timestamp_utc: 2026-08-04T22:44:27Z
