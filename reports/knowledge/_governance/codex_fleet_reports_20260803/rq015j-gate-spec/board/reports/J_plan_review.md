# Track J 计划独立复审：门规格与既定数字核查

生成时间（UTC）：2026-08-02T01:48:48Z。

本次复审对象是 `.codex-fleet/rq015j-gate-spec/board/J-leader-kickoff.md`。这份计划要为 online verification 冻结第一道弃权门：当单次运行内的 7 个候选 MSE 与 log 域权重显示当前 IPV 数值不携带候选间判别信息时，返回弃权而不是返回 `ipv=0`。整体工作尚未执行；本次是执行前独立复审，只复算计划里当作既定事实的数字，并审查计划是否能服务 RQ009 的上下文条件 conformal envelope。

复算脚本为 `.codex-fleet/rq015j-gate-spec/work/j_plan_review_compute.py`，证据 JSON 为 `.codex-fleet/rq015j-gate-spec/work/j_plan_review_compute.json`。解释器为 `<local-rq009-venv>/bin/python`。未改 `agent.py` / `ipv_estimation.py` / `process_interhub.py` / `reliability_logdomain.py`，未提交 git commit，未提交 HPC 作业，未重解任何锚点。最终证据不依赖 RQ009 feature matrix 行级 join；当前脚本只查 schema 与 feature contract，因为 feature partitions 不带 `rq007_split`，不能在本计划当前写法下安全证明行级 join 不触及 RQ007 held_out。

## 1. 核查表

| # | 断言值 | 我的复算值 | 判定与原因 |
|---|---:|---:|---|
| 1 | `spread(mse_per_candidate[7]) == 0` 为 400 个（17.4%），全部 nuplan，waymo 0 | 400 / 2,300 = 17.3913%；nuplan 400 / 1,150，waymo 0 / 1,150 | 一致。来源：`anchor_mse.csv` 的 `mse_per_candidate[7]` 和 `source`；逐行 `max(mse)-min(mse)==0.0`。 |
| 2 | `k_eff_log` 在 6.75-7.00 格为 766 个，是最大模态 | 全 2,300 行中为 1,166 个（50.6957%，上界按 `<=7+1e-12`）；排除 #1 的 400 个 `spread==0` 后才是 766 个 | 不一致。监督方算错了，或漏写了“先排除 `spread==0`”这个筛选条件。来源：`anchor_mse.csv` 的 `k_eff_log`、`mse_per_candidate[7]`。最大模态成立，但计数口径不是计划原文口径。 |
| 3 | `k_eff=6.75` 对应最佳候选权重 0.2027 | 在非 `spread==0` 且 `6.75<=k_eff_log<=7+1e-12` 的 766 行里，`max(max(w_log))=0.202664916`；但从 `k_eff=1/sum(w^2)` 不能唯一换算 `max(w)`。若假设 1 个最大权重、其余 6 个相等，则 `k_eff=6.75` 给 `max(w)=0.210200646`；`max(w)=0.2027` 在同一假设下给 `k_eff=6.801094` | 部分一致。0.2027 能作为样本边缘经验值复现，但“对应/换算”这句话错了。来源：`anchor_mse.csv` 的 `w_log[7]`、`k_eff_log`；我从 `w_log[7]` 重算 `k_eff_log`，最大绝对差 8.88e-16。 |
| 4 | 门后留下 1,017 个 = 44.2% | 1,017 / 2,300 = 44.2174%；弃权 1,283 / 2,300 = 55.7826% | 一致，但要标成未加权样本比例。来源：`anchor_mse.csv` 的 `mse_per_candidate[7]`、`w_log[7]`。另：1,283 个弃权全都由 `max(w_log)<0.20` 覆盖，400 个 `spread==0` 完全包含在其中。 |
| 5 | 门后按最佳候选权重分带，nuplan / waymo 的 `|ipv|` 均值为 0.131/0.051、0.178/0.200、0.273/0.356、0.438/0.571、0.755/0.970 | 0.20-0.25：0.130556 (n=81) / 0.050730 (n=86)；0.25-0.35：0.177683 (n=83) / 0.200066 (n=96)；0.35-0.50：0.273417 (n=68) / 0.356040 (n=77)；0.50-0.75：0.438332 (n=41) / 0.570614 (n=94)；0.75-1.01：0.755081 (n=39) / 0.969816 (n=352) | 一致。来源：`anchor_mse.csv` 的 `w_log[7]`、`ipv_log`、`source`；筛选为 `spread!=0 and max(w_log)>=0.20`，分箱为左闭右开 `[0.20,0.25)` 等，最后一箱 `[0.75,1.01)`。 |
| 6 | 最高带样本数 nuplan 39 vs waymo 352 | nuplan 39，waymo 352 | 一致。来源同 #5。 |
| 7 | 门后汇总 `|ipv|` 均值 nuplan/waymo 为 0.293/0.633，边界饱和占比 1%/20% | `|ipv_log|` 均值为 nuplan 0.292740 (n=312)、waymo 0.632620 (n=705)，一致；但 `at_grid_boundary` 为 nuplan 79/312=25.3205%、waymo 398/705=56.4539%，不是 1%/20%。若改用 `abs(|ipv_log|-3*pi/8)<=1e-9`，则 nuplan 3/312=0.9615%、waymo 143/705=20.2837% | 均值一致，边界口径不一致。监督方把“精确端点命中”写成了“边界饱和”。如果按数据列 `at_grid_boundary` 理解，监督方算错了。来源：`anchor_mse.csv` 的 `ipv_log`、`source`、`at_grid_boundary`。 |
| 8 | `zero_postwarm_scope == True` 等价于 `signature in {U,Z}`，会排除 signature N | `N|False` 500，`U|True` 1,200，`Z|True` 600；不一致行 0 | 一致。来源：`mechanism_split.csv` 的 `zero_postwarm_scope`、`signature`。 |
| 9 | 4 份 parquet 合计 14,473,982 行，不含 `mse_per_candidate[7]` 与 `w_log[7]`，其中 `k_eff` 是连乘域派生 | 4 份 parquet 行数：5,197,072 + 281,268 + 8,994,736 + 906 = 14,473,982；schema union 只有 `artifact_id, product_row_key, measurement_role, case_id, rq007_split, ipv_error, K, candidate_grid_id, k_eff, q_eff, attempt_status, reason_code, recoverability, ledger_schema_version, aggregation_perspective, aggregation_configuration`，没有 `mse_per_candidate[7]` / `w_log[7]`。`reports/plans/RQ015A_ledger_schema_v4_20260731.json` 写明 `k_eff = 1.0 / (1.0 - ipv_error) ** 2`，不是从 log 域权重列派生 | 基本一致。schema 直接证明缺 MSE/log 权重，也证明 `k_eff` 从既有 `ipv_error` 派生；因此不能用现有 ledger 直接普查 J 门。来源：4 份 concentration_ledger parquet schema 与 `RQ015A_ledger_schema_v4_20260731.json` 的 `k_eff`/`q_eff` 字段。 |

## 2. 发现的问题

### 2.1 严重：计划把两个错误或缺口径的数字写成已确认事实

问题：`k_eff_log` 的 6.75-7.00 格计数在全样本中是 1,166，不是 766；766 只有在额外排除 `spread==0` 的 400 行后才成立。边界饱和如果按 `at_grid_boundary` 列算是 25.32% / 56.45%，不是 1% / 20%；1% / 20% 对应的是 `|ipv_log|` 精确端点命中（1e-9 容差），不是边界饱和列。

证据：`.codex-fleet/rq015b-repair/work/anchor_mse.csv`；列 `k_eff_log`、`mse_per_candidate[7]`、`w_log[7]`、`ipv_log`、`at_grid_boundary`、`source`。

后果：J1 会把隐藏筛选后的 766 写成总体模态大小，把端点命中写成边界饱和，读者无法复算。更严重的是，边界问题会被低估：`at_grid_boundary` 下 waymo 门后有 398/705=56.45% 在边界。

建议：计划必须改成“全样本 1,166；排除 `spread==0` 后 766”；并把“精确端点命中”和 `at_grid_boundary` 分开命名、分开报告。

### 2.2 严重：按 RQ009 上下文变量分格这一步，锚点样本自身无法执行

问题：RQ009 的 context-only 变量是 22 个数值变量加 7 个类别变量，包括 `relative_distance_anchor`、`relative_speed_anchor`、`closing_rate_anchor`、`geometry_path_category`、`priority_role` 等。`anchor_mse.csv` 与 `mechanism_split.csv` 中这 29 个 context-only 变量一个都没有。

证据：RQ009 `calibration.py` 的 `BASE_NUMERIC_CONTEXT` 与 `BASE_CATEGORICAL_CONTEXT`；`calibration_gate.json` 的 `M3_numeric` / `M3_categorical`；锚点文件列为 `source`、`n_band`、`signature`、`frame_index`、`agent_slot`、`n_obs`、MSE/权重/IPV 结果等，不含 RQ009 context columns。

后果：计划第 3 件要求“按 RQ009 实际用的上下文变量分格”不能直接完成。若 J1 临时去 join RQ009 feature matrix，又必须先有 `rq007_split` 安全过滤方案；feature partitions 自身没有 `rq007_split`，计划未定义这一 join 的安全路径。

建议：执行前二选一：要么把 J1 的交付改成“只报告锚点自带变量上的内部检查，不做 RQ009 context 分格”；要么补一个明确的、按 case split 白名单先过滤的 join 方案，并规定 join 覆盖率和失败处理。当前计划不能直接派发。

### 2.3 中等：两条门判据在样本上完全重叠，贡献占比不能按两条独立机制解释

问题：400 个 `spread==0` 行的 `max(w_log)` 全部等于 1/7=0.142857，全部满足 `max(w_log)<0.20`。因此 `spread==0 OR max(w_log)<0.20` 的弃权集合等于 `max(w_log)<0.20`，样本中判据 1 对 union 没有新增行。

证据：`anchor_mse.csv` 的 `mse_per_candidate[7]`、`w_log[7]`；`spread==0` 400 行，`overlap(spread==0, max(w_log)<0.20)=400`。

后果：如果报告“两条判据各自贡献”而不定义互斥顺序，会重复计数。按 HT 全域权重看，若先记 `spread==0`，它只占 13,482.74 / 2,646,058 = 0.5095%；非 `spread==0` 但 `max(w_log)<0.20` 占 746,744.164 / 2,646,058 = 28.2210%。

建议：保留判据 1 可以，因为它有清楚的机制标签；但线上统计必须用互斥 reason：先判 `spread==0`，否则再判 `max(w_log)<0.20`，并明确判据 1 是语义标签，不是额外筛出能力。

### 2.4 中等：`theta=0.20` 的“近均匀模态边缘”依据不成立，只能作为政策阈值

问题：`max(w_log)` 在 0.20 附近没有明显空隙：`[0.18,0.20)` 有 95 行，`[0.20,0.22)` 有 71 行。把 theta 从 0.18 改到 0.22，门后样本从 1,112/2,300=48.35% 变为 946/2,300=41.13%，差 166 行、7.22 个百分点；门后 `|ipv_log|` 均值从 0.48745 变为 0.56261。

证据：`anchor_mse.csv` 的 `w_log[7]`、`ipv_log`、`at_grid_boundary`。theta 敏感性：0.18 -> 1,112 行；0.19 -> 1,062；0.20 -> 1,017；0.21 -> 971；0.22 -> 946；0.25 -> 850。

后果：计划写“theta=0.20 排掉的就是这个模态，不多不少”过强。0.20 不是数据里自然断点，会影响样本组成和下游分布。

建议：把 0.20 写成“简单、可解释的政策阈值”，不要写成从直方图自然边缘推出。若仍执行，必须带一个短敏感性附注，而不是改阈值。

### 2.5 中等：`max(w_log)` 可用，但不是唯一或最稳的近均匀统计量

问题：`max(w_log)` 只看最佳候选一个分量。`k_eff_log=1/sum(w^2)` 和归一化熵使用全部 7 个权重，更贴近“整体近均匀”。在本样本中 `max(w_log)` 与 `k_eff_log` Spearman rho = -0.9968，归一化熵与 `k_eff_log` rho = 0.9991，三者几乎同序；因此换统计量不会带来本质新信息，但会改变阈值解释。

证据：`anchor_mse.csv` 的 `w_log[7]`、`k_eff_log`；`max(w_log)<0.20` 为 1,283 行，`k_eff_log>6.75` 为 1,166 行，二者重叠 1,164 行。

后果：如果计划把 `k_eff=6.75 -> max(w)=0.2027` 当成数学换算，后续会继续混用统计量。H 轨还显示环境内权重集中度不等于跨环境稳定性，因此不能把任何单一集中度统计写成“可靠”的证明。

建议：门可以继续用 `max(w_log)`，因为它最容易在线解释；但报告必须同时记录 `k_eff_log` 或熵作为诊断，并删掉“唯一换算”说法。

### 2.6 中等：source 内部检查支持“最高带样本量不平衡”，但不支持“各带内基本重合”

问题：分带均值和样本数复现了，但“控制集中度后各带内基本重合”过强。0.20-0.25 带为 0.1306 vs 0.0507，0.50-0.75 带为 0.4383 vs 0.5706，0.75-1.01 带为 0.7551 vs 0.9698；最高带中位数为 nuplan 0.0639 vs waymo 0.8005。边界口径按 `at_grid_boundary` 也在各带内持续更高于 waymo。

证据：`anchor_mse.csv` 的 `w_log[7]`、`ipv_log`、`source`、`at_grid_boundary`；门后筛选同 #5。

后果：若报告写成“差异几乎全部来自选择效应”，会把残余 source 形状差异提前消掉。PI 已定 envelope 不按 source 拆分，但内部检查不能反向证明 source 差异不存在。

建议：改成“最高权重带的样本量不平衡解释了汇总均值差异的相当一部分；分带后仍有残差，不作为下游分源口径，只作为内部风险披露。”

### 2.7 中等：“AV 侧同一把尺子，偏移两边抵消”的论证不成立

问题：同一门作用在人类和 AV 上，只能保证两边都被条件化到“通过门”的子总体；不能保证偏移抵消。如果 AV 的 `max(w_log)` 分布、边界饱和率、或 IPV 分布形状与人类不同，门会改变被比较对象，可能遮蔽或放大偏移。

证据：计划第 119-124 行要求写“AV 侧用同一把尺子，偏移两边抵消”；本复算显示门后 `|ipv_log|` 随 `max(w_log)` 单调上升，样本组成会显著改变分布。

后果：论文若写“抵消”，会把条件分布比较误写成无条件分布比较。

建议：改为“结论只对同一门通过后的条件分布成立；若 AV 的门通过机制与人类不同，须分别报告 AV 通过率与人类通过率，并把未通过作为监控结果的一部分。”

### 2.8 低到中等：设计基估计方法基本可用，但必须使用全域分母且不能引用 44.2% 作为全域影响

问题：抽样设计本身与 `ht_weight` 一致：12 个 `source x n_band x signature` 单元，每个 source/n_band 下 U=300、Z=150、N=125；全域 HT 分母为 2,646,058，N 层 cluster 为 491。按 J 门在 Mac 锚点样本上做 HT ratio，门后保留权重为 1,885,831.096 / 2,646,058 = 71.2695%，B=2000、seed=20260731 的 cluster bootstrap CI 为 [67.1729%, 75.2135%]。

证据：`mechanism_split.csv` 的 `source`、`n_band`、`signature`、`ht_weight`、`zero_postwarm_scope`，以及 `anchor_mse.csv` 的门判据列。

后果：样本未加权 44.2174% 与全域设计基估计 71.2695% 方向差很大。计划若没有反复标注“44.2% 是样本内，不是全域”，读者会误解门的全语料影响。

建议：执行时主文用 HT ratio 和 CI；样本比例只放在核查或方法附注。

### 2.9 低：工程输出契约还不够明确

问题：计划说“不得返回 `ipv=0`”，但没有指定机器可读返回结构。

证据：计划第 72-75 行只要求输入、`w_log` 定义、弃权返回和计数记录；没有状态枚举、空值规则、reason 优先级、日志字段。

后果：下游 RQ009 或 online verifier 可能把 `NaN`、`None`、空字符串、缺列、`ipv=0` 混成不同含义。

建议：输出至少包括 `status in {OK, ABSTAIN}`、`ipv_log`（弃权时为 null）、`reason_codes`（互斥主因加可选副因）、`max_w_log`、`mse_spread`、`k_eff_log`、`candidate_grid_id`、`K`、`frame_id/timestamp`。RQ009 使用时还需要门通过率和 context cell key。

## 3. 明确判定

判定：**需修改后执行**。

理由：

1. 数字层面已有两处必须改：`k_eff_log` 6.75-7.00 格的总体计数不是 766；边界饱和 1%/20% 是端点命中口径，不是 `at_grid_boundary` 口径。
2. 方法层面，`spread==0` 完全被 `max(w_log)<0.20` 包含，计划必须把两条判据写成有序 reason 记录，而不是两个独立贡献。
3. 用途层面，RQ009 context 分格按当前锚点文件无法直接执行；这是交付 blocker，不是细节优化。
4. 设计基估计本身可以做，`ht_weight` 与 U/Z/N x source/n_band 配额一致，B=2000、seed=20260731、N 层 491 cluster 也合理；因此不是“不应执行”。
5. 速度原则上，这一轮不需要多轮复审或方法学扩张；但修正上述事实和接口缺口是必要工作，不属于过度加固。

## 4. 最薄弱的一条

最薄弱的是“按 RQ009 实际上下文变量分格”。

这条直接服务最终 online verification，但锚点样本文件没有任何 RQ009 context-only 变量；RQ009 feature matrix 有这些变量的 schema，却不能在本计划当前写法下安全行级 join，因为 feature partitions 不带 `rq007_split`。如果这条不改，J1 要么无法完成核心交付，要么会临时发明 join 口径并触碰 RQ007 split 边界。相比 theta=0.20 是否最优，这才是真正会阻断执行和下游接入的问题。
