# F1 — RQ007 已接受契约在下溢发现之后的重核

## 1. 结论表

判定阈值：扛得住 = 扣除下溢后效应量变化 < 20% 且方向不变；需重述 = 方向不变但数字/边界需要改；需重跑 = 效应量塌缩过半、方向翻转、或分母被破坏。

|主张|判定（三选一）|关键数字|
|---|---|---|
|RQ007-KC-C1|需重述|development raw -0.1497->-0.1571, Δ=-0.0074, rel=4.9%; guard raw -0.1470->-0.1560, Δ=-0.0090, rel=6.1%|
|RQ007-KC-C2|本轮未取证|本轮未重算 C2 的 \|dθ\| / settling 指标|
|RQ007-KC-C3|扛得住|mean\|Δ\| 0.2647->0.2919 (relative change 10.3%); sign 0.234->0.234|

## 2. held_out 合规

`held_out_parsed_rows = 0`。C 文件流式读取时先按 A 的 26,886 个 scene 白名单过滤；非白名单行未访问 IPV/error 字段。

- A: split 行数 {'development': 1788593, 'guard': 702399}，scene 数 {'development': 19258, 'guard': 7628}；1,788,593 + 702,399 = 2,490,992; 19,258 + 7,628 = 26,886。

- D: split measurement 行数 {'development': 3731250, 'guard': 1465822}，attempt_status {'ATTEMPTED': 4981984, 'NOT_ATTEMPTED': 215088}；(2,490,992 valid physical frames + 107,544 warmup physical frames) * 2 = 5,197,072。

- C: total 3,695,981；allowlisted split {'development': 1865625, 'guard': 732911}；A-valid split {'development': 1788593, 'guard': 702399}；warmup/non-A allowlisted never parsed 107,544；dropped_rows_never_parsed 1,097,445。

## 3. §0 靶表复现

|slot|cv_cpa_conflict|n|n_underflow|pct|
|---|---|---|---|---|
|c1|0|2,391,270|285,605|11.944%|
|c1|1|99,722|8,220|8.243%|
|c2|0|2,391,270|380,699|15.920%|
|c2|1|99,722|11,396|11.428%|
|pooled|0|4,782,540|666,304|13.932%|
|pooled|1|199,444|19,616|9.835%|

帧加权 pooled mean_c: conflict=0 0.46487，conflict=1 0.29379，gap=-0.17108。靶 count 匹配：True。

## 4. Q1 — C3 是否为下溢伪影

方法：B 直读复现 C3；C 用 scene 白名单流式读取后，仅对 A 中有效帧解析 IPV/error；active 定义为 A.cv_cpa_conflict==1。
|筛选|case-agent n|mean abs delta|严格变号率|
|---|---|---|---|
|n_active_frames_gt0|13,214|0.264661|23.448%|
|non_nan_pair|13,214|0.264661|23.448%|

Q1b 重建：all_valid within 1e-12 比例 1.000000，max diff 2.220e-16；active within 1e-12 比例 1.000000，max diff 2.220e-16。

Q1c 扣除下溢：mean|Δ| 0.264661 -> 0.291937（相对变化 10.306%），严格变号率 0.234483 -> 0.234483；dropout 628/13,214 (4.753%)。

下溢占比：all_valid 674,941/4,981,984 (13.548%)；interaction_active 19,495/199,444 (9.775%)。替代法（下溢 IPV 替换为该 case-agent 非下溢 all-valid 均值）：mean|Δ| 0.244598，变号率 0.206868。

一句话判读：C3 的 0.26 rad / 22% 基线可复现；剔除下溢后 mean|Δ| 未塌缩，替代法只带来小幅下降。

## 5. Q2 — C1 gap 偏移方向与扣除下溢

方法：方向判断使用 A 的 raw cv_cpa_conflict c1/c2；RQ007 headline 另从既有 Phase-5 输出直读；扣除下溢在 raw conflict case-slot gap 上执行，bootstrap 按 scene/case_id 整簇重抽 1000 次。
|split|frame conf=0|frame conf=1|case conf=0|case conf=1|
|---|---|---|---|---|
|development|13.889%|9.829%|15.165%|12.594%|
|guard|14.041%|9.852%|15.166%|13.347%|
|dev_guard|13.932%|9.835%|15.165%|12.806%|

RQ007 Phase-5 既有 baseline：
|split|headline real_gap|case mean|n_case|n rows|
|---|---|---|---|---|
|development|-0.132074|-0.137267|4,743|159,612|
|guard|-0.128595|-0.135781|1,864|64,300|

Raw conflict 下溢敏感性：
|split|variant|case gap|case bootstrap CI|Δgap|
|---|---|---|---|---|
|development|baseline_raw_conflict|-0.149716|[-0.153510, -0.145904]|0.000000|
|development|exclude_underflow_both_conflict_groups|-0.157084|[-0.160661, -0.153368]|-0.007369|
|development|exclude_underflow_conflict0_only|-0.100231|[-0.104294, -0.095818]|0.049485|
|development|exclude_underflow_conflict1_only|-0.202912|[-0.206706, -0.199248]|-0.053196|
|guard|baseline_raw_conflict|-0.146969|[-0.152427, -0.140790]|0.000000|
|guard|exclude_underflow_both_conflict_groups|-0.155984|[-0.161241, -0.150631]|-0.009015|
|guard|exclude_underflow_conflict0_only|-0.095115|[-0.101515, -0.088529]|0.051854|
|guard|exclude_underflow_conflict1_only|-0.202762|[-0.208439, -0.197176]|-0.055793|

一句话判读：conflict=0 的下溢率高于 conflict=1，但两侧同时剔除下溢时 raw gap 反而更负；单侧分解显示 conflict=0 下溢会把 gap 推浅、conflict=1 下溢会把 gap 推深，后者在本口径下杠杆更大。

## 6. Q3 — c1/c2 与 q_eff 是否同一个量

定义层：D 内部 `ipv_error = 1 - 1/sqrt(k_eff)` 的 max residual 2.290e-16，within 1e-12 比例 1.000000；`q_eff=k_eff/K` max residual 8.882e-16。

数值层：A(c1/c2) 与 D(ipv_error) join 4,981,984 行，exact ratio 0.002185，within 1e-12 ratio 0.043816，max diff 0.622036，Pearson 0.432385，Spearman 0.473345，first material diff frame (>1e-9) 5。frame4 [{'frame_index': 4, 'n': 53772, 'exact_equal_ratio': 0.01528676634679759, 'mean_abs_diff': 2.440928126145099e-13, 'max_abs_diff': 5.000444502911705e-13}]；frame5 [{'frame_index': 5, 'n': 53768, 'exact_equal_ratio': 0.006974408570153251, 'mean_abs_diff': 0.06562900189004892, 'max_abs_diff': 0.622035526991}]。

三选一结论：**同族但不同版本**。

## 7. Q4 — PI 解释的直接检验

Q4a pooled strict 下溢：
|split|conflict|n|denominator|pct|
|---|---|---|---|---|
|development|0|476,784|3,432,734|13.889%|
|development|1|14,198|144,452|9.829%|
|guard|0|189,520|1,349,806|14.041%|
|guard|1|5,418|54,992|9.852%|
|dev_guard|0|666,304|4,782,540|13.932%|
|dev_guard|1|19,616|199,444|9.835%|

Q4b anchor: spread==0 行 400/2,300；join missing 0；2x2 [[spread>0 conf0, conf1],[spread=0 conf0, conf1]] = [[1837, 63], [395, 5]]；P(conflict=0|spread=0)=0.987500，P(spread=0|conflict=0)=0.176971，P(spread=0|conflict=1)=0.073529，Fisher p=0.022843。

source 构成：[{'source': 'nuplan', 'False': 750, 'True': 400}, {'source': 'waymo', 'False': 1150, 'True': 0}]。一句话判读：`spread(mse)==0` 在 E 锚点里是无需阈值的精确判据；join 到 A 后可描述其与 conflict mask 的关联，但不作因果解释。

## 8. Q5 — 分母重报

方法：D 台账按 case/frame/measurement_role join A 的 conflict；warmup/NOT_ATTEMPTED 在 A 中无匹配，单列 `unmatched_warmup_or_no_A`。near-uniform 分子只计 ATTEMPTED 行，避免把 warmup 的 `ipv_error=1.0` 当作集中度结果。
|denominator|conflict|n rows|strict|eps=0.01|
|---|---|---|---|---|
|D_ATTEMPTED_measurement_rows|0|4,782,540|226,239 (4.731%)|2,563,606 (53.603%)|
|D_ATTEMPTED_measurement_rows|1|199,444|3,547 (1.778%)|20,431 (10.244%)|
|D_ATTEMPTED_measurement_rows|unmatched_warmup_or_no_A|0|0 (NA)|0 (NA)|
|D_all_measurement_rows_including_warmup|0|4,782,540|226,239 (4.731%)|2,563,606 (53.603%)|
|D_all_measurement_rows_including_warmup|1|199,444|3,547 (1.778%)|20,431 (10.244%)|
|D_all_measurement_rows_including_warmup|unmatched_warmup_or_no_A|215,088|0 (0.000%)|0 (0.000%)|
|A_valid_measurement_rows_2x_2490992|0|4,782,540|666,304 (13.932%)|1,796,447 (37.563%)|
|A_valid_measurement_rows_2x_2490992|1|199,444|19,616 (9.835%)|22,431 (11.247%)|

attempt_status by conflict: [{'conflict_group': '0', 'attempt_status': 'ATTEMPTED', 'n_rows': 4782540}, {'conflict_group': '1', 'attempt_status': 'ATTEMPTED', 'n_rows': 199444}, {'conflict_group': 'unmatched_warmup_or_no_A', 'attempt_status': 'NOT_ATTEMPTED', 'n_rows': 215088}]。

## 9. 自查

- Q2 Phase-5 event-window residual 未在本轮完全重建；报告中 C1 扣除下溢的 bootstrap 是 raw conflict gap，不等同于 Phase-5 headline。
- Q1 的 valid/active 定义虽然数值重建与 B 对齐，但依赖 A 已经剔除 warmup 的结构；若 RQ007 另有隐藏过滤，影响会体现在 B 对齐之外的解释层。
- bootstrap 已按 scene/case_id 整簇重抽；点估计仍是 case-slot 均值，若 PI 要求先把两个 agent slot 合并到 scene-level，点估计会略变。

## 10. 限制

本轮为描述性/敏感性分析，不含因果推断。描述 RQ015 侧新结果时，本报告只使用“权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息”这一表述；仅在引用 RQ007 正式主张时保留其原文术语。
