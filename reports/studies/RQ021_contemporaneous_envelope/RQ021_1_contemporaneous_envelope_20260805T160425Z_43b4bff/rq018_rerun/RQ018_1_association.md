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
| 机制二通过：21,936/67,861 = 32.3249% | 筛选 `mechanism2_gate_ok == True`；来源 `.codex-fleet/rq021-contemporaneous-envelope/work/E1/onsite_scoring_dryrun.parquet`；列 `mechanism2_gate_ok` |
| 两门交集：14,099/67,861 = 20.7763% | 筛选 `status == OK AND mechanism2_gate_ok == True`；来源 `data/derived/rq017_onsite_gate/l1_v1` + `.codex-fleet/rq021-contemporaneous-envelope/work/E1/onsite_scoring_dryrun.parquet`；连接列 `product_row_key` |
| unit 基础集：245/267 = 91.7603% | 筛选 `analysis_set == True`；来源 `data/derived/onsite_competition/RQ012B_event_harm/stage4plus/unit_analysis_table.parquet`；列 `analysis_set` |
| unit 曝露有定义：225/245 = 91.8367% | `analysis_set == True` 且至少一帧两门通过；来源 RQ017 + M2 + 锚点表 + unit 表；列 `status, mechanism2_gate_ok, unit_composite_key, analysis_set`。其余 unit 保持缺失，不赋零值 |

`product_row_key` 从 RQ017 到机制二命中 67,861/67,861 = 100.0000%；两门交集解析到锚点表命中 14,099/14,099 = 100.0000%。两者分别使用 `product_row_key` 和 `case_key + anchor_frame_index + perspective`，来源见上表。

## 4. 帧级主结果：90% 参照区间

系数单位为曝露增加一个参照区间宽度时结果变量的变化；距离和 TTC 下降表示风险增加，最大接近率上升表示风险增加。方括号为 case 聚类 95% 置信区间。

| 窗口 | 结果 | 有符号单斜率 | 上侧幅度（方向联合模型） | 下侧幅度（方向联合模型） |
|---|---|---|---|---|
| contract | future_min_distance_m | -3.0081 [-9.5894, 3.5732]; p_naive=0.0785, p_case=0.3688, p_team=0.4667 | 6.1312 [-6.4355, 18.6979]; p_naive=0.0124, p_case=0.3374, p_team=0.4171 | 12.1382 [3.9410, 20.3353]; p_naive=<0.0001, p_case=0.0039, p_team=0.0003 |
| contract | future_min_ttc_s | 15795778.1772 [-15807342.2113, 47398898.5658]; p_naive=0.1713, p_case=0.3257, p_team=0.2771 | 28830337.8121 [-28972602.9254, 86633278.5497]; p_naive=0.0798, p_case=0.3267, p_team=0.2024 | -2439904.4196 [-8143794.6390, 3263985.7997]; p_naive=0.8836, p_case=0.4002, p_team=0.4217 |
| contract | future_max_closing_rate_mps | 0.0680 [-1.5567, 1.6926]; p_naive=0.9191, p_case=0.9344, p_team=0.9460 | -1.7289 [-4.8312, 1.3734]; p_naive=0.0718, p_case=0.2733, p_team=0.4292 | -1.8631 [-3.9798, 0.2537]; p_naive=0.0523, p_case=0.0842, p_team=0.1877 |
| fixed_3s | future_min_distance_m | -2.1330 [-9.1748, 4.9087]; p_naive=0.2148, p_case=0.5512, p_team=0.5840 | 12.8638 [0.6043, 25.1233]; p_naive=<0.0001, p_case=0.0398, p_team=0.0744 | 17.1148 [9.0526, 25.1771]; p_naive=<0.0001, p_case=<0.0001, p_team=<0.0001 |
| fixed_3s | future_min_ttc_s | -6771.5331 [-22467.9675, 8924.9012]; p_naive=0.5793, p_case=0.3962, p_team=0.5038 | -45798.4926 [-135094.1848, 43497.1996]; p_naive=0.0092, p_case=0.3133, p_team=0.3280 | -31656.0037 [-92621.1431, 29309.1357]; p_naive=0.0696, p_case=0.3073, p_team=0.3162 |
| fixed_3s | future_max_closing_rate_mps | -1.1367 [-3.1263, 0.8530]; p_naive=0.1557, p_case=0.2615, p_team=0.1772 | -2.9161 [-6.6821, 0.8498]; p_naive=0.0112, p_case=0.1285, p_team=0.1811 | -0.6410 [-3.6412, 2.3592]; p_naive=0.5768, p_case=0.6742, p_team=0.6730 |

### 4.1 80% / 95% 敏感性

下表只列有符号单斜率；上、下侧联合模型的完整朴素、case 聚类和 team 聚类结果在 `frame_level_results.json`。

| α | 窗口 | 结果 | case 聚类效应 [95% CI] 与三种 p 值 |
|---:|---|---|---|
| 80 | contract | future_min_distance_m | -0.6687 [-3.1382, 1.8008]; p_naive=0.2715, p_case=0.5942, p_team=0.6632 |
| 80 | contract | future_min_ttc_s | -1846036.1872 [-5416024.4565, 1723952.0821]; p_naive=0.6637, p_case=0.3093, p_team=0.3047 |
| 80 | contract | future_max_closing_rate_mps | -0.2056 [-0.9214, 0.5101]; p_naive=0.3876, p_case=0.5719, p_team=0.6407 |
| 80 | fixed_3s | future_min_distance_m | -0.3646 [-2.9659, 2.2366]; p_naive=0.5511, p_case=0.7826, p_team=0.7928 |
| 80 | fixed_3s | future_min_ttc_s | -926.6484 [-4152.5462, 2299.2494]; p_naive=0.8315, p_case=0.5719, p_team=0.7180 |
| 80 | fixed_3s | future_max_closing_rate_mps | -0.4425 [-1.2877, 0.4028]; p_naive=0.1203, p_case=0.3034, p_team=0.3009 |
| 95 | contract | future_min_distance_m | -6.0917 [-29.5019, 17.3185]; p_naive=0.3821, p_case=0.6086, p_team=0.6409 |
| 95 | contract | future_min_ttc_s | 3762909.1513 [-5603904.4228, 13129722.7255]; p_naive=0.9369, p_case=0.4294, p_team=0.3935 |
| 95 | contract | future_max_closing_rate_mps | -2.5772 [-8.3539, 3.1996]; p_naive=0.3449, p_case=0.3803, p_team=0.4989 |
| 95 | fixed_3s | future_min_distance_m | 0.5695 [-24.7314, 25.8704]; p_naive=0.9353, p_case=0.9647, p_team=0.9650 |
| 95 | fixed_3s | future_min_ttc_s | 8727.8837 [-22115.2986, 39571.0660]; p_naive=0.8626, p_case=0.5777, p_team=0.6439 |
| 95 | fixed_3s | future_max_closing_rate_mps | -6.2574 [-13.4247, 0.9099]; p_naive=0.0553, p_case=0.0867, p_team=0.0246 |

### 4.2 TTC 的 `log1p` 数值健康敏感性

原始 TTC 在接近率非常接近零但仍为正时可达到极大有限值。任务书原始尺度结果仍在主表；下表额外对 `log(1 + future_min_ttc_s)` 拟合同一模型，不设接近率阈值、不改变帧的纳入。负系数表示越界幅度增加时未来最小 TTC 变短。

| α | 窗口 | 有符号单斜率 | 上侧幅度（联合模型） | 下侧幅度（联合模型） |
|---:|---|---|---|---|
| 80 | contract | 0.3518 [-0.0281, 0.7316]; p_naive=0.0096, p_case=0.0693, p_team=0.1604 | -0.0607 [-1.1862, 1.0648]; p_naive=0.7837, p_case=0.9155, p_team=0.9153 | -0.6327 [-1.3173, 0.0520]; p_naive=0.0005, p_case=0.0700, p_team=0.0448 |
| 80 | fixed_3s | 0.4217 [0.0819, 0.7616]; p_naive=0.0014, p_case=0.0152, p_team=0.0424 | -0.1068 [-1.3655, 1.1519]; p_naive=0.6123, p_case=0.8674, p_team=0.8385 | -0.8012 [-1.5794, -0.0230]; p_naive=<0.0001, p_case=0.0437, p_team=0.0284 |
| 90 | contract | 0.1710 [-0.9103, 1.2523]; p_naive=0.6434, p_case=0.7556, p_team=0.7760 | -0.8687 [-3.1132, 1.3757]; p_naive=0.0989, p_case=0.4464, p_team=0.4664 | -1.2364 [-2.6100, 0.1372]; p_naive=0.0204, p_case=0.0775, p_team=0.0643 |
| 90 | fixed_3s | 0.6504 [-0.4217, 1.7225]; p_naive=0.0785, p_case=0.2331, p_team=0.2327 | -0.5516 [-3.2417, 2.1385]; p_naive=0.3000, p_case=0.6866, p_team=0.6463 | -1.8340 [-3.3607, -0.3072]; p_naive=0.0005, p_case=0.0188, p_team=0.0158 |
| 95 | contract | 2.6105 [-1.8037, 7.0246]; p_naive=0.0862, p_case=0.2451, p_team=0.3363 | -0.6048 [-12.2977, 11.0882]; p_naive=0.8214, p_case=0.9189, p_team=0.9315 | -4.1885 [-8.9958, 0.6188]; p_naive=0.0249, p_case=0.0874, p_team=0.1220 |
| 95 | fixed_3s | 4.4226 [0.3384, 8.5068]; p_naive=0.0038, p_case=0.0339, p_team=0.0502 | 1.5916 [-11.4574, 14.6407]; p_naive=0.5554, p_case=0.8103, p_team=0.7925 | -5.7942 [-10.3370, -1.2515]; p_naive=0.0019, p_case=0.0127, p_team=0.0223 |

## 5. 负对照

case 标签置换共 200 次：整段打乱 case 的曝露轨迹与结果轨迹的对应；不同帧数按相对锚点位置最近邻对齐，然后原样重拟合含 `context_cell` 固定效应与 case 聚类推断的帧模型。安慰剂曝露按任务书从每帧 `[lo_90-width_90, hi_90+width_90]` 均匀抽取，使用固定种子；安慰剂 p 值比较真实 case 聚类 |t| 与 200 次安慰剂 |t| 分布。

| 窗口 | 结果 | 标签置换 p（有符号） | 安慰剂 p（有符号） | 标签置换 p（下侧幅度） | 安慰剂 p（下侧幅度） |
|---|---|---:|---:|---:|---:|
| contract | future_min_distance_m | 0.5771 | 0.3930 | 0.1393 | 0.0100 |
| contract | future_min_ttc_s | 0.0149 | 0.7811 | 0.7562 | 0.9851 |
| contract | future_max_closing_rate_mps | 0.9602 | 0.9502 | 0.3134 | 0.1144 |
| contract | future_log1p_min_ttc | 0.8308 | 0.7761 | 0.2289 | 0.0995 |
| fixed_3s | future_min_distance_m | 0.6667 | 0.6269 | 0.0348 | 0.0050 |
| fixed_3s | future_min_ttc_s | 0.5522 | 0.8458 | 0.2040 | 0.3234 |
| fixed_3s | future_max_closing_rate_mps | 0.4776 | 0.2587 | 0.8209 | 0.6468 |
| fixed_3s | future_log1p_min_ttc | 0.3781 | 0.2239 | 0.1493 | 0.0199 |

## 6. unit 级非安全子分数

unit 基础分母是 `analysis_set == True` 的 245 个；其中 225 个至少有一帧两门通过，20 个曝露无定义而不进入关联。unit 越界量先除以各帧区间宽度，再在 unit 内聚合。下表以 team 为 block 给出 Spearman 或控制 `official_comprehensive` 后的偏 Spearman 95% bootstrap 区间；`U` 为单变量，`P` 为偏关联。

| 结果 | unit 曝露 | U: rho [team-block 95% CI] | P: partial rho [team-block 95% CI] |
|---|---|---|---|
| official_efficiency | frac_outside_90 | -0.096 [-0.237, 0.071] (n=225, team-block p=0.2338) | -0.291 [-0.406, -0.166] (n=225, team-block p=0.0020) |
| official_efficiency | mean_signed_exceedance_90 | -0.013 [-0.166, 0.136] (n=225, team-block p=0.8611) | 0.042 [-0.081, 0.157] (n=225, team-block p=0.5295) |
| official_efficiency | mean_upper_exceedance_90 | -0.088 [-0.224, 0.046] (n=225, team-block p=0.1938) | -0.221 [-0.342, -0.089] (n=225, team-block p=0.0020) |
| official_efficiency | mean_lower_exceedance_90 | -0.081 [-0.216, 0.060] (n=225, team-block p=0.2537) | -0.272 [-0.350, -0.182] (n=225, team-block p=0.0020) |
| official_efficiency | max_abs_exceedance_90 | -0.088 [-0.224, 0.059] (n=225, team-block p=0.2318) | -0.313 [-0.415, -0.204] (n=225, team-block p=0.0020) |
| official_comfort | frac_outside_90 | 0.047 [-0.104, 0.202] (n=225, team-block p=0.5514) | 0.052 [-0.095, 0.204] (n=225, team-block p=0.4555) |
| official_comfort | mean_signed_exceedance_90 | -0.058 [-0.190, 0.077] (n=225, team-block p=0.4336) | -0.055 [-0.167, 0.073] (n=225, team-block p=0.3976) |
| official_comfort | mean_upper_exceedance_90 | 0.019 [-0.115, 0.148] (n=225, team-block p=0.7572) | 0.025 [-0.093, 0.155] (n=225, team-block p=0.6933) |
| official_comfort | mean_lower_exceedance_90 | 0.069 [-0.039, 0.175] (n=225, team-block p=0.2218) | 0.071 [-0.034, 0.179] (n=225, team-block p=0.1678) |
| official_comfort | max_abs_exceedance_90 | 0.096 [-0.019, 0.211] (n=225, team-block p=0.1239) | 0.099 [-0.032, 0.225] (n=225, team-block p=0.1319) |
| official_compliance | frac_outside_90 | 0.067 [-0.072, 0.224] (n=225, team-block p=0.3676) | 0.079 [-0.042, 0.223] (n=225, team-block p=0.2358) |
| official_compliance | mean_signed_exceedance_90 | -0.047 [-0.165, 0.079] (n=225, team-block p=0.4735) | -0.041 [-0.165, 0.080] (n=225, team-block p=0.5375) |
| official_compliance | mean_upper_exceedance_90 | 0.030 [-0.064, 0.135] (n=225, team-block p=0.5095) | 0.043 [-0.040, 0.130] (n=225, team-block p=0.3017) |
| official_compliance | mean_lower_exceedance_90 | 0.104 [-0.012, 0.220] (n=225, team-block p=0.0939) | 0.114 [-0.005, 0.230] (n=225, team-block p=0.0659) |
| official_compliance | max_abs_exceedance_90 | 0.080 [-0.052, 0.199] (n=225, team-block p=0.2298) | 0.088 [-0.025, 0.204] (n=225, team-block p=0.1079) |
| official_coordination | frac_outside_90 | 0.126 [-0.026, 0.292] (n=225, team-block p=0.1019) | 0.178 [0.062, 0.288] (n=225, team-block p=0.0060) |
| official_coordination | mean_signed_exceedance_90 | -0.045 [-0.208, 0.100] (n=225, team-block p=0.6374) | -0.039 [-0.185, 0.082] (n=225, team-block p=0.6294) |
| official_coordination | mean_upper_exceedance_90 | 0.100 [-0.030, 0.230] (n=225, team-block p=0.1718) | 0.153 [0.064, 0.251] (n=225, team-block p=0.0040) |
| official_coordination | mean_lower_exceedance_90 | 0.145 [0.015, 0.274] (n=225, team-block p=0.0320) | 0.193 [0.092, 0.296] (n=225, team-block p=0.0040) |
| official_coordination | max_abs_exceedance_90 | 0.178 [0.060, 0.303] (n=225, team-block p=0.0080) | 0.233 [0.131, 0.322] (n=225, team-block p=0.0020) |

## 7. 次要安全结果与功效边界

全 267 个 unit 中，`official_safety < 100` 为 21/267 = 7.8652%（筛选 `official_safety < 100`，来源 `data/derived/onsite_competition/RQ012B_event_harm/stage4plus/unit_analysis_table.parquet`，列 `official_safety`）；`collision_intervention_deduction_any != 0` 为 18/267 = 6.7416%（同一来源，列 `collision_intervention_deduction_any`）；`safety_intervention != 0` 为 8/267 = 2.9963%（同一来源，列 `safety_intervention`）。阳性事件稀少，以下结果只给效应和区间；不显著不能解释成没有关联，显著也必须按低功效探索看待。

| 次要结果 | unit 曝露 | Spearman rho [team-block 95% CI] |
|---|---|---|
| official_safety | frac_outside_90 | -0.131 [-0.218, 0.002] (n=225, team-block p=0.0559) |
| official_safety | mean_signed_exceedance_90 | -0.054 [-0.152, 0.081] (n=225, team-block p=0.4076) |
| official_safety | mean_upper_exceedance_90 | -0.101 [-0.174, -0.008] (n=225, team-block p=0.0440) |
| official_safety | mean_lower_exceedance_90 | -0.063 [-0.144, 0.015] (n=225, team-block p=0.1239) |
| official_safety | max_abs_exceedance_90 | -0.078 [-0.162, 0.030] (n=225, team-block p=0.1319) |
| collision_intervention_deduction_any | frac_outside_90 | 0.116 [-0.027, 0.213] (n=225, team-block p=0.1079) |
| collision_intervention_deduction_any | mean_signed_exceedance_90 | 0.029 [-0.122, 0.141] (n=225, team-block p=0.6893) |
| collision_intervention_deduction_any | mean_upper_exceedance_90 | 0.080 [-0.014, 0.150] (n=225, team-block p=0.1139) |
| collision_intervention_deduction_any | mean_lower_exceedance_90 | 0.083 [0.005, 0.174] (n=225, team-block p=0.0460) |
| collision_intervention_deduction_any | max_abs_exceedance_90 | 0.076 [-0.017, 0.161] (n=225, team-block p=0.1099) |
| safety_intervention | frac_outside_90 | 0.109 [-0.037, 0.201] (n=225, team-block p=0.2093) |
| safety_intervention | mean_signed_exceedance_90 | 0.099 [0.046, 0.167] (n=225, team-block p=0.0021) |
| safety_intervention | mean_upper_exceedance_90 | 0.069 [-0.016, 0.163] (n=225, team-block p=0.1629) |
| safety_intervention | mean_lower_exceedance_90 | 0.006 [-0.040, 0.042] (n=225, team-block p=0.7930) |
| safety_intervention | max_abs_exceedance_90 | 0.029 [-0.052, 0.108] (n=225, team-block p=0.4853) |

## 8. 数值健康、覆盖与坐标异常

- 90% 层上侧 869 帧、下侧 519 帧、区间内 12,711 帧，合计 14,099/14,099。筛选为两门交集；来源 RQ017 `ipv_log` 与机制二区间列 `lo_90, hi_90, width_90`。
- 合同窗口越出 case 末尾 0/14,099 = 0.0000%；3 秒窗口越出 case 末尾 734/14,099 = 5.2060%。来源 `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_ipv_timeseries_multi_allvalid.parquet`，列 `case_key, frame_index, time_s`；越界窗口使用 case 末尾前可见部分并保留标记。
- 合同窗口因全窗口 `closing_rate_mps <= 0` 而 TTC 缺失 1,211/14,099 = 8.5893%；3 秒窗口为 461/14,099 = 3.2697%。来源 `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_ipv_timeseries_multi_allvalid.parquet`，列 `distance_m, closing_rate_mps`。
- TTC 数值边界：稠密表最小正接近率为 4.915e-09 m/s；合同窗口 `future_min_ttc_s > 10^6` 的行数为 7，3 秒窗口为 0。来源 `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_ipv_timeseries_multi_allvalid.parquet`，列 `distance_m, closing_rate_mps`。本轮保留任务书公式，不私设接近率下限；极大但有限的 TTC 会使原始尺度 OLS 对这些行敏感。
- case 帧数分布 min/p25/median/p75/max = 1/20.0/50.0/90.0/222；case 数 231，team 数 19。筛选为两门交集；来源 RQ017 + M2 + 锚点表，列 `status, mechanism2_gate_ok, case_key`。
- 全锚点表坐标异常为 7/67,861 = 0.0103%，全部来自 `onsite:shanghai:T10:C4:native_case:2311`；来源 `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet`，列 `case_key, relative_distance_anchor`。照常保留时，这 7 行中进入两门交集的是 0/14,099 = 0.0000%；因此剔除口径与照常参与口径的主模型系数最大绝对差为 0。
- `ipv_log` NaN/+inf/-inf 均为 0；80/90/95 三层均满足全部 14,099 行 `width > 0` 且 `lo < hi`。逐项机器证据见 `data_health.json`。

## 9. 与 RQ012B 已接受结论的关系

RQ012B 的冻结结论 `RQ012-KC-HARM-NULL` 使用含自动驾驶车目标值的旧 RQ009 M3 参照，分析的是 unit 级官方 harm，并判定没有相对 placebo 与 context-only 基线的 IPV 特异增量关联。本轮使用纯人-人参照、增加机制一过滤，并把主分析单元换成帧级未来窗口风险。因此本轮是不同曝露定义和不同分析单元的探索，不构成对旧结论的推翻。

## 10. 结论

1. **下侧越界与更短的未来 TTC 是本轮最直接对应‘劣化’方向的线索。** 在 90% 层，`log(1+TTC)` 的下侧幅度系数在合同窗口为 -1.2364 [-2.6100, 0.1372]（case 聚类 p=0.0775），3 秒窗口为 -1.8340 [-3.3607, -0.3072]（case 聚类 p=0.0188）。80% 与 95% 层同方向。case-block 标签置换 p 分别为 0.2289 与 0.1493，安慰剂 p 分别为 0.0995 与 0.0199；按预先要求的 case 聚类、标签置换、安慰剂三项同时低于 0.05 的规则，两窗口总体判定为未同时通过。这是关联性线索，不是因果结论。
2. **上侧越界没有显示行为劣化，多个结果反而指向更大的未来距离或更低的最大接近率。** 例如 3 秒窗口有符号斜率对最小距离为 -2.1330 m [-9.1748, 4.9087]（case 聚类 p=0.5512）。这可由场景选择、控制后的剩余结构或行为差异解释，本轮不能把它写成‘上侧异常更安全’。
3. **unit 级结果不形成一致的跨子分数劣化模式。** 控制 `official_comprehensive` 后，效率与部分越界汇总呈负相关，而协调性对下侧幅度呈正相关；舒适与合规多数区间跨零。官方综合分与子分数存在强机械吸收，加上本轮同时查看多种曝露，unit 结果只适合作为后续假设来源。
4. 次要安全结果的阳性 unit 很少，虽然下侧幅度与若干安全结果同向，当前区间与多重比较不足以支持‘有关联’或‘无关联’的稳定判断。监督方复算前，本报告保持探索性证据状态；任何后续手稿表述都需要独立数据复现与单独接受。

## 11. 待监督方决定

1. 是否将通过 case 聚类、标签置换和安慰剂三项检查的方向性线索列为下一数据集的预注册目标。依据是第 4–5 节三种证据同时成立；若不推进，本轮只保留为描述性产物。
2. unit 基础集 245 个中有 20 个没有任何两门通过帧。本轮按缺失处理，避免把“无可分析帧”编码为零越界；监督方如需另一个政策口径，应另立分析而不是覆盖本轮。

## 12. 可复跑产物

- 脚本：`.codex-fleet/rq021-contemporaneous-envelope/work/E1/rq018_rerun/rq018_association.py`
- 关键数字：`.codex-fleet/rq021-contemporaneous-envelope/work/E1/rq018_rerun/key_numbers.json`
- 帧级结果：`.codex-fleet/rq021-contemporaneous-envelope/work/E1/rq018_rerun/frame_level_results.json`
- unit 级结果：`.codex-fleet/rq021-contemporaneous-envelope/work/E1/rq018_rerun/unit_level_results.json`
- 负对照：`.codex-fleet/rq021-contemporaneous-envelope/work/E1/rq018_rerun/negative_controls.json`
- 数据健康：`.codex-fleet/rq021-contemporaneous-envelope/work/E1/rq018_rerun/data_health.json`

state: WAITING_ON_COMMANDER
timestamp_utc: 2026-08-05T15:52:16Z
