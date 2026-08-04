# RQ015 收官合并报告

## 1. 位置与问题

本报告解决的问题是：在 online verification 中，判断一辆自动驾驶车的 IPV 是否可以进入 RQ009 已接受的 human-envelope 判据。IPV 是 Interaction Preference Value，用一个标量表示交互倾向；envelope 是 RQ009 已接受的人类分布覆盖区间，用来判断一个通过前置门的 IPV 数值是否落在人类分布支持范围内。

整体进度是：RQ009 的 envelope 判据已经 accepted；RQ015B 到 RQ015K 已完成数值修复、跨环境证据、弃权门规格、全语料台账普查；RQ015L 的 L1/L2 又补齐了 RQ009 精确零点原子与 OnSite `UNKNOWN` 的最后两项解释。本轮 L3 是收官成文，只整合已落盘报告，不重算任何数字。

最终判据由两个弃权机制串联。机制一是 RQ015 本线负责的前置门：先判断当前帧的 IPV 数值是否携带七个候选之间的判别信息；若机制一弃权，则该帧直接结束，不进入机制二。机制二是 RQ009 已接受的 envelope 支持度判据，即在人类分布覆盖区间内判支持、区间外判不支持。

canonical 求解单元指去重后的一次 IPV 求解，一个求解单元可以支撑多条下游台账行。台账行指 materializer 写出的行级记录，它带有数据源、角色、键、门状态和来源状态等字段。anchor 指一次抽样或物化时选中的 case-frame-role 锚点，用来代表一个具体帧与角色的求解位置。HT 权重指 Horvitz-Thompson 权重，是按抽样概率倒数加权的设计基估计权重。

## 2. 缺陷的性质

原实现在线性概率域计算候选权重。残差相对 sigma 较大时，七个候选的密度连乘会数值下溢；旧代码在分母归零后退回“七个候选等权”的兜底。候选网格是对称的，所以等权平均必然给出 `ipv` 恰为 `0`，同时 `ipv_error` 恰为 `1 - 1/sqrt(7) = 0.6220355269907728`。

这个行为的后果是：数据里无法只凭数值区分“计算失败后被写成 0”和“该个体的 IPV 点值本身为 0”。这里的 0 是中性取值，不应被下游自动解释为弃权或失败。

七个候选给出逐位相同轨迹的原因是解析性的。无实质交互时，前向目标函数退化为：

```text
cos(ipv) * interior + constant
```

候选网格全部落在 `(-pi/2, pi/2)` 内，因此 `cos(ipv) > 0`。正标量只缩放目标函数，不移动 `argmin`；于是七个候选得到同一个最优轨迹、同一个 MSE。D/G/J 轨均把这件事落到同一口径：2,300 个锚点样本中有 400 行 `spread(mse)==0`，分母为锚点样本 2,300，筛选为 `max(mse_per_candidate[7])-min(mse_per_candidate[7])==0`，来源为 D1/J1/G leader 报告引用的 `anchor_mse.csv` 的 `mse_per_candidate[7]` 列；比例为 17.3913%（400/2,300）。G 轨在 HPC 上复核为同样 400 行，且这 400 行的 `mse_per_candidate[7]` JSON 字符串 400/400 逐位相同，来源为 G leader 判定的 Mac/HPC CSV 对照。

## 3. 修复

修复是把权重计算从线性连乘域改到 log 域：

```text
log_score_i = -MSE_i / (2 * sigma^2)
w_i = softmax(log_score)_i
```

这不是近似。它与线性域公式

```text
w_i = exp(-MSE_i / (2 * sigma^2)) / sum_j exp(-MSE_j / (2 * sigma^2))
```

在数学上精确等价；差别只是用 log-sum-exp 做归一化，避免 `exp(...)` 的中间值先下溢成 0。B2 的平价门证据显示，在未触发兜底且无部分下溢的 1,526 个锚点上，legacy 权重与 log 域权重最大差为 `3.75e-15`；分母为 2,300 锚点样本，筛选为 `legacy_fallback_triggered=False` 且 `partial_underflow=False`，来源为 B2 报告的 `anchor_mse.csv` 列 `w_legacy[7]`、`w_log[7]`、`legacy_fallback_triggered`、`partial_underflow`。G 轨在 HPC 冻结环境下复核同一结论：eligible 1,528 行，最大差 `3.2196468e-15`，来源为 G1/G leader 对 `anchor_mse_hpc.csv` 的复核。

log 域修复解决的是数值下溢造成的等权兜底；它不保证每个通过数值计算的点值都携带候选间判别信息。因此还需要机制一的门。

## 4. 门的规格

机制一的冻结规格如下：

| 判据 | 输出 |
|---|---|
| `mse_spread == 0` | `reason_code=NO_IPV_EFFECT` |
| `max(w_log) < 0.20` | `reason_code=NEAR_UNIFORM` |
| 否则 | `status=OK` |

`mse_spread` 是七个候选 MSE 的最大值减最小值。`w_log` 是 log 域 softmax 后的七个候选权重。`NO_IPV_EFFECT` 表示七候选 MSE 完全相同，IPV 不改变该前向目标；`NEAR_UNIFORM` 表示最大候选权重低于阈值，权重近均匀，所以该 IPV 数值不携带候选间的判别信息。

`theta = 0.20` 是政策阈值，不是数据断点，不得写成由样本中的拐点导出。J1 的敏感性记录只说明样本内门后行数随阈值移动：`theta=0.18/0.20/0.22` 时为 1,112/1,017/946，分母均为 2,300 锚点，筛选均为 `spread(mse)!=0 and max(w_log)>=theta`，来源为 J1 报告附录的 `anchor_mse.csv` 列 `mse_per_candidate[7]` 与 `w_log[7]`。

## 5. 确定性证据

同一 HPC 软件栈下，AMD EPYC 与 Intel 对同一个计算逐位相同。G2 使用 Slurm job `2024766` 在 `fata02` 上重算 `ipv_000001`，与 `cpui158` 参考产物比较：xlsx 数值 348/348 逐位相同，CSV 数值 4/4 逐位相同，`max|Δ|=0.0`。来源为 `G2_crossnode_gate.md`，筛选为同一输入 CSV、同一 PKL、同一 entrypoint、同一 `configs/ipv_sigma01_exact.json`、同一 frozen Python env；比较列为 `ipv_key_agent_1`、`ipv_key_agent_1_error`、`ipv_key_agent_2`、`ipv_key_agent_2_error` 及 CSV 汇总数值列。

Mac 与 HPC 不同。监督方在 K2 期间直接比较 RQ015B 的 Mac `anchor_mse.csv` 与 G 轨 HPC `anchor_mse_hpc.csv`，两份文件各 2,300 个锚点，`anchor_id` 完全重合；`mse_per_candidate[7]` 字符串完全相同的为 433/2,300，不同的为 1,867/2,300，最大逐元素绝对差为 `7.044643e+01`，约 70.4，发生在 `ipv_008361|77|2`。若写成比例，81.1739%（1,867/2,300），分母为 2,300 个重合锚点，筛选为两份 CSV 的 `anchor_id` 交集，来源为 `.codex-fleet/rq015k-fullcorpus-gate/board/commander_notes.md` 的监督方裁定记录，比较列为 `mse_per_candidate[7]`。

因此差异来自软件栈而非 CPU。这个结论是由两条证据合取推出的：同一 HPC 软件栈跨 AMD/Intel 逐位相同，而 Mac 与 HPC 在大量锚点上不同。G leader 的方法学结论是：曲面越平，`argmin` 越不可复现，但由曲面形状定义的量反而更可复现。支撑数字包括 `argmin_candidate` 686/2,300 发生 Mac/HPC 翻转，即 29.8261%（686/2,300；分母为 2,300 个重合锚点，来源 G leader 判定，列 `argmin_candidate`），但机制拆分的最大偏移只有 1.73 个百分点，来源 G leader 对 Mac/HPC `mechanism_split` 的复核。

## 6. 全语料普查与分母纪律

InterHub 全量为 4,981,984 个 canonical 求解单元。K2 普查结果如下，全部来自 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/validation/a_class_manifest_rollup.json`，筛选为 A-class InterHub solve rows，分母字段为 `a_rows`，状态字段为 `a_status_counts`，原因字段为 `a_reason_counts`。

| 结果 | 计数 | 占比 | 来源字段 |
|---|---:|---:|---|
| `OK` | 3,502,340 | 70.3001%（3,502,340/4,981,984） | `a_status_counts.OK / a_rows` |
| `NEAR_UNIFORM` | 1,457,746 | 29.2604%（1,457,746/4,981,984） | `a_reason_counts.NEAR_UNIFORM / a_rows` |
| `NO_IPV_EFFECT` | 19,964 | 0.4007%（19,964/4,981,984） | `a_reason_counts.NO_IPV_EFFECT / a_rows` |
| `SOLVER_FAILURE` | 1,934 | 0.0388%（1,934/4,981,984） | `a_status_counts.SOLVER_FAILURE / a_rows` |

RQ009 台账行域的 `OK` 为 6,405,292/8,994,736，即 71.2116%（分母为 RQ009 ledger rows 8,994,736，筛选为 `artifact_id=rq009_feature_matrix` 且 `status=OK`，来源 K2 报告 §4.1 监督方 addendum，列 `canonical_key`、`status`、`gate_applicable`）。G 锚点的正确 HPC 基线比较结果为 `compared_rows=2300`、`max_abs_diff=0.0`，来源 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/validation/g_anchor_hpc_baseline.json`。RQ009 回填的唯一性实测为 `rows=8,994,736`、`unique_keys=8,994,736`、`duplicates=0`，来源 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/validation/rq009_join_key_uniqueness.json`，列为 `canonical_key`；这是实测，不是硬编码。

总台账为 14,473,982 行，构成为 InterHub 5,197,072 行、RQ009 8,994,736 行、OnSite 281,268 行、WOD 906 行，来源 K2 报告 `final_validation_summary.json` 的 row accounting。这里的 InterHub 5,197,072 是台账行数，包含非求解来源状态；上表的 4,981,984 是去重后的 InterHub canonical 求解单元，不能混用。

J 轨给的是设计基估计，不是普查。J 的 HT 分母为 2,646,058，门后保留权重为 1,885,831.096，因此通过率为 71.2695%（1,885,831.096/2,646,058；筛选为 `spread!=0 and max(w_log)>=0.20`，来源 J1 附录，文件列为 `anchor_mse.csv` 的 `mse_per_candidate[7]`、`w_log[7]` 与 `mechanism_split.csv` 的 `ht_weight`）。K2 是普查；在台账行域上，上一段 RQ009 台账行域普查值与本段 J 设计基估计值相差 0.0579 个百分点，理由是 J 的 HT 估计目标是行加权通过率。求解单元域的 K2 普查值为 70.3001%（3,502,340/4,981,984；来源同本节 InterHub 普查表），与 J 相差 0.9694 个百分点，必须单独列出。

求解单元与台账行的压缩比为 2.804×，来源 K2 报告 §4.1；但是 J 轨 HT 权重分母 2,646,058 与 RQ009 台账行 8,994,736 的关系尚未确立。可辩护表述上限是：在台账行域上，设计基估计与普查相差 0.06 个百分点。不得把它表述为通过式验证，也不得声称两个域已经对齐。

## 7a. L1：RQ009 精确零点原子的拆分

RQ009 自己已经在 accepted 报告的 Limitations 中留下警告：其打分目标存在精确零点原子，计数为 273,819/1,270,566，即 21.5509%（分母为 RQ009 M3 `fold=test` 单个 alpha 层目标行 1,270,566，筛选为 `alpha≈0.10` 且 `y==0.0`，来源 RQ009 `evidence.csv` 的 `C6_ATOM` 行与 RQ009 `90_report/index.html` Limitations，列为 `y` 与 `alpha`）。RQ009 还把 nominal `0.80` coverage 层标注为 boundary-tie / `1e-10` endpoint-nudge 脆弱；这里的 `0.80` 是 RQ009 coverage 层标签，不是 L 轨新计算比例，来源 RQ009 `evidence.csv` 的 `C1/P4/P9` 行与 `90_report/index.html` Limitations。RQ009 当时无法判断这些精确 0 是真实中性点值，还是数值流程把弃权情形写成了 0。

连接可行，但不是全覆盖。连接键是：

```text
case_key + anchor_frame_index + perspective + source_dataset + |role=target_future
```

其中 `role=target_future` 是 K2 台账的 measurement role，表示 RQ009 目标未来窗口对应的测量角色；`alpha` 是 RQ009 的 coverage 层参数。`1,270,566` 是 RQ009 M3 `fold=test` 预测文件在单个 alpha 层上的目标行；`3,811,698 = 1,270,566 x 3` 来自三个 alpha 层。K2 的 `8,994,736` 是 `4,497,368` 个 product 行乘以 2 个 measurement role，即 `target_future` 与 `counterpart_current`。来源为 L1 报告与 L1b leader 自查，列为 RQ009 predictions 的 `case_key`、`anchor_frame_index`、`perspective`、`source_dataset`、`alpha`，以及 K2 ledger 的 `canonical_key`、`product_row_key`、`measurement_role`。

左连接是精确、无重复的 one-to-zero-or-one 连接，但不是全覆盖。全部单 alpha 目标行中，命中 888,892/1,270,566，即 69.9601%（筛选为上述左连接命中，来源 L1b，列 `canonical_key`）；未命中 381,674/1,270,566，即 30.0399%（筛选为上述左连接未命中，来源 L1b，列 `case_key`、`canonical_key`）。未命中定性以 L1b 为准：它是整案级排除，涉及 2,270 个 case；这些 case 中出现在 K2 台账 `case_id` 里的为 0/2,270，即 0.0000%（来源 L1b，列 `case_key` 与 K2 `case_id`），同时既有命中行又有未命中行的 case 为 0/7,576，即 0.0000%（来源 L1b，列 `case_key`）。命中的 5,306 个 case 与 E1 独立记录的 dev+guard case 集吻合；由于本轮未打开受保护的 confirmation 划分文件，2,270 个未命中 case 属 RQ007 held_out 是推断，不是直读标签。

L1 的主口径分母必须是 192,271，即 RQ009 精确零点原子中落在 K2 台账覆盖域内的行。不能把 273,819 全零点原子直接当作机制一可分类分母。

| 类别 | 行数 | 占台账覆盖域 | 占全零点原子 | 来源、筛选与列 |
|---|---:|---:|---:|---|
| 过门的真中性零（`status=OK`） | 99,938 | 51.9777%（99,938/192,271） | 36.4978%（99,938/273,819） | L1b；筛选 `y==0.0`、left join 命中、`status=OK`；列 `y`、`canonical_key`、`status` |
| 弃权而被记成 0（`status!=OK`） | 92,333 | 48.0223%（92,333/192,271） | 33.7205%（92,333/273,819） | L1b；筛选 `y==0.0`、left join 命中、`status!=OK`；列 `y`、`canonical_key`、`status` |
| `NEAR_UNIFORM` | 90,490 | 47.0638%（90,490/192,271） | 33.0474%（90,490/273,819） | L1b；同上且 `reason_code=NEAR_UNIFORM`；列 `reason_code` |
| `NO_IPV_EFFECT` | 1,796 | 0.9341%（1,796/192,271） | 0.6559%（1,796/273,819） | L1b；同上且 `reason_code=NO_IPV_EFFECT`；列 `reason_code` |
| `SOLVER_FAILURE`（工程失败） | 47 | 0.0244%（47/192,271） | 0.0172%（47/273,819） | L1b；同上且 `status=SOLVER_FAILURE`；列 `status` |
| 台账未覆盖（整案级排除） | 81,548 | 不适用 | 29.7817%（81,548/273,819） | L1b；筛选 `y==0.0` 且 left join 未命中；列 `y`、`canonical_key`、`case_key` |

描述性结论是：在台账覆盖域上，RQ009 那个精确零点原子里约一半不是中性的 IPV 点值，而是弃权情形下被写成 0 的数值。精确口径为 48.0223%（92,333/192,271；分母为台账覆盖域零点行 192,271，筛选为 `y==0.0`、left join 命中、`status!=OK`，来源 L1b，列 `y`、`canonical_key`、`status`）。主体是权重近均匀：47.0638%（90,490/192,271；同一分母与来源，筛选为 `reason_code=NEAR_UNIFORM`）。这直接回应 RQ009 自己关于精确零点原子、boundary tie 和 practical null 解释受限的警告。

旧指纹不是好判据。旧实现均匀兜底时 `ipv_error` 恰为 `0.6220355269907728`；L1b 在台账覆盖域零点行里用容差 `1e-12` 检查该指纹，来源为 L1b 的 `L1_fingerprint_crosstab.csv` 与 `ipv_error`、`status`、`reason_code` 列。指纹命中行中，74.8911%（22,358/29,854；分母为指纹命中行 29,854，筛选为 `abs(ipv_error-0.6220355269907728)<=1e-12` 且 `status=OK`）状态是 `OK`。非 OK 行中，只有 8.1184%（7,496/92,333；分母为非 OK 零点行 92,333，筛选为同一指纹命中条件）命中指纹。那 22,358 行占台账覆盖域零点行 11.6284%（22,358/192,271；分母 192,271，来源 L1b），机制解释是：旧实现在线性域下溢后退回等权，而 log 域恢复出非均匀权重，`max(w_log)>=0.20`。因此旧的 `ipv_error` 数值既不充分也不必要；下游必须用 `status` 与 `reason_code` 判别。

## 7b. L2：OnSite 274,022 行 `UNKNOWN` 的定性

OnSite 的 `UNKNOWN` 是显式分支，不是隐式兜底。来源代码位置为 `scripts/rq015a/build_ledger.py:1219-1233`，逻辑是：先按 OnSite 局部序号 `local_position < 4` 返回 `NOT_ATTEMPTED`；否则若 role 对应的 `ipv_error` 为空，返回 `UNKNOWN` 与 `EMPTY_CELL_UNEXPLAINED`；否则若 `q_eff` 为空，返回 `UNKNOWN` 与 `DEGENERATE_IPV_ERROR`；剩余情况才返回 `ATTEMPTED`。本轮不修改该脚本。

K2 OnSite 台账合计 281,268 行。来源为 L2 报告与 K2 OnSite ledger，筛选 `artifact_id=onsite_dense_timeseries`，来源状态列为 `source_attempt_status`，原因列为 `source_reason_code`。

| 来源状态 | 行数 | 占 OnSite 台账 | 来源、筛选与列 |
|---|---:|---:|---|
| `UNKNOWN` | 274,022 | 97.4238%（274,022/281,268） | L2；`source_attempt_status=UNKNOWN`，列 `source_attempt_status` |
| `ATTEMPTED` | 2,974 | 1.0574%（2,974/281,268） | L2；`source_attempt_status=ATTEMPTED`，列 `source_attempt_status` |
| `NOT_ATTEMPTED` | 4,272 | 1.5188%（4,272/281,268） | L2；`source_attempt_status=NOT_ATTEMPTED`，列 `source_attempt_status` |

全部 `UNKNOWN` 行的 `source_reason_code` 都是 `EMPTY_CELL_UNEXPLAINED`：100.0000%（274,022/274,022；分母为 OnSite `UNKNOWN` 行，筛选 `source_attempt_status=UNKNOWN`，来源 L2，列 `source_reason_code`）。这些行不能解释为“数据确实不支持”；更准确的判断是“既有流水线没有走到这些 dense role 行”。依据是：在这些 `UNKNOWN` 行中，`case_key/frame_index/timestamp_ms`、ego/counterpart 位置速度 heading、配对 ID、距离和相对速度字段均为 100.0000%（274,022/274,022；分母为 OnSite `UNKNOWN` 行，来源 L2 dense 源表字段完整性审计，列包括 `case_key`、`frame_index`、`timestamp_ms`、位置、速度、heading、配对 ID、距离、相对速度）。

OnSite stage3plus 生成脚本默认 `--max-anchors-per-unit 1`，只对选中 anchor 的支撑帧与 target 帧填 `ipv_*`，对应代码为 `build_onsite_m3_anchors.py:776-831`。因此 dense 表有轨迹行，但多数 role 行没有被物化到 IPV 字段。保留的输入边界是：dense 源表没有真实地图、车道、route 或 reference-line 字段，0.0000%（0/274,022；分母为 OnSite `UNKNOWN` 行，来源 L2，列为 map/lane/route/reference-line 字段检查）；当时使用 observed trajectory fallback 作为参考线。

与 RQ015A 旧口径一致的是分子分母，不是列名：RQ015A 用 `attempt_status`，K2 用 `source_attempt_status`。RQ015A 的 OnSite `ATTEMPTED` 为 1.0574%（2,974/281,268；筛选 `artifact == onsite_dense_timeseries` 且 `attempt_status == ATTEMPTED`，来源 `concentration_ledger_summary.csv` 的 `artifact`、`attempt_status`、`rows` 列）。K2 的同一来源状态计数为 1.0574%（2,974/281,268；筛选 `artifact_id=onsite_dense_timeseries` 且 `source_attempt_status=ATTEMPTED`，来源 L2/K2，列 `source_attempt_status`）。

本轮不补齐 OnSite。WOD 906 行与 OnSite 2,974 行继续保持门不适用状态，直到另行授权 materializer。

## 8. 交付给下游的接口约束

`ipv_log = 0` 是合法且高频的通过门估计值。

**这里必须先纠正一个被错标了分母的数字（track L leader 于 2026-08-03T03:44:10Z 复算发现）。**
K2 报告 §9 与本轨任务书都写作「门后 23.40% 的通过行取 `ipv_log = 0`」，读起来像是全语料普查值。
它不是。**23.40% 的实际出处是 J 轨的锚点样本：238/1,017**（分母为门后锚点样本 1,017，
筛选 `spread(mse)!=0 and max(w_log)>=0.20` 且 `abs(ipv_log)<=1e-9`，来源 J1 §3.1.1，
文件列为 `anchor_mse.csv` 的 `mse_per_candidate[7]`、`w_log[7]`、`ipv_log`）。

**InterHub 全语料普查值如下**（来源 `.codex-fleet/rq015l-consolidate/work/L1_rq009_zero_atom_split/L3b_ipvlog_zero_census.json`，
脚本见同目录；筛选 `artifact_id=interhub_sigma01_hw4_timeseries` 且 `gate_applicable=true` 且 `status=OK`，
列为 `status`、`ipv_log`、`gate_applicable`）：

| 零值口径 | 计数 | 占门后通过行（分母 3,502,340） |
|---|---:|---:|
| `ipv_log == 0.0` 恰好相等 | 175,458 | 5.0097% |
| `abs(ipv_log) <= 1e-12` | 346,312 | 9.8880% |
| `abs(ipv_log) <= 1e-9` | 348,539 | 9.9516% |

**两个数不矛盾，但不能互相替代。** J 的 1,017 个锚点是按 HT（Horvitz-Thompson，按抽样概率
倒数加权）设计抽出来的，未加权的锚点占比本来就不是行级普查值的估计量；
而 3,502,340 是 InterHub 门后通过的 canonical 求解单元数。
**错的不是 23.40% 这个数本身，而是把它写成「通过行」的全语料属性。**
今后引用时必须写成「J 锚点样本 238/1,017」，或改用上表的普查值。

这个纠正**不削弱接口约束，反而使它更硬**：即便按最严的恰好相等口径，
门后仍有 5.0097%（175,458/3,502,340）的通过行取 `ipv_log = 0`。
结论不变：弃权不再写成 `ipv_log = 0`，但反过来**不能把 `ipv_log = 0` 当成弃权**。

下游接口只能用 `status` 与 `reason_code` 判断机制一结果，不能用数值指纹，
**也不能用「`ipv_log` 是否为 0」这一条**——上表已说明门后本来就有相当比例的合法零值。
第 7a 节的指纹检查是正面证据：指纹命中行中 74.8911%（22,358/29,854；分母与来源见第 7a 节）状态是 `OK`，而非 OK 行中只有 8.1184%（7,496/92,333；分母与来源见第 7a 节）命中旧指纹。旧的 `ipv_error=0.6220355269907728` 既不充分也不必要。

台账覆盖范围之外的行必须保持 `gate_applicable=false`，不得被计入 `NO_IPV_EFFECT` 或 `NEAR_UNIFORM`。K2 的非求解行共 497,262 行，来源 K2 `final_validation_summary.json` 的 `non_solve_rows.rows`；其中包括 InterHub `NOT_ATTEMPTED=215,088`、OnSite `NOT_ATTEMPTED=4,272`、OnSite `UNKNOWN=274,022`、OnSite `ATTEMPTED=2,974` 与 WOD `ATTEMPTED=906`。这些行不应被下游混入机制一的 InterHub canonical 求解单元分母 4,981,984。

## 9. 方法学教训

第一，1 行 canary 测不到两条真实路径：多 worker 并发和工程失败行写盘。K2 实际发生的失败恰好都在这两条路径上：solve 阶段先遇到 Matplotlib font-cache 并发锁，再遇到 PyArrow fixed-size-list parquet writer 无法写 null array rows；finalize 阶段又遇到逐行重算源文件 SHA 过慢。来源为 K2 报告失败史，作业分别为 `2069424`、`2069818`、`2071368`。

第二，每条验收判据都要有一次“故意让它失败”的验证。本轮出现两例“看起来在检查、实际没检查该检查的东西”：RQ009 `duplicates` 曾硬编码为 0，后来 K2-2 才实测 `rows=8,994,736`、`unique_keys=8,994,736`、`duplicates=0`；G 锚点曾把 K2-HPC 结果错比到 RQ015B 的 Mac 基线，正确基线应是 G 轨 HPC `anchor_mse_hpc.csv`。来源分别为 K2 报告 §6 与 §5，列为 `canonical_key` 以及 G anchor 的对齐键/数值列。

第三，L1 报了 29.7817%（81,548/273,819；分母为 RQ009 全零点原子 273,819，筛选为 `y==0.0` 且 K2 left join 未命中，来源 L1/L1b，列 `y`、`canonical_key`、`case_key`）的 join miss 却没有查明它是什么，leader 补查后才定性为整案级排除。未命中不是一个可以留白的类别；若不定性，整张拆分表无法解释。

第四（本轮新增，见第 8 节）：**一个数字被反复转引之后，它的分母会掉。**
「门后 23.40% 取 `ipv_log = 0`」在 K2 报告 §9 与本轨任务书中都读作全语料属性，
实际出处是 J 轨的 238/1,017 锚点样本；InterHub 门后通过行的普查值是
5.0097%（175,458/3,502,340，恰好相等口径）到 9.9516%（348,539/3,502,340，`1e-9` 容差口径）。
这与前两条教训是同一种病：**「看起来在讲 A，实际测的是 B」**，只是这次错位发生在分母而不是判据。
可操作的对策是本轨已经在执行的那条——**每个百分数落笔时必须同时写出分子、分母、
筛选条件、来源文件与列名**；写不出来，就说明引用者自己也不知道它是在什么域上测的。

## 10. 待裁定事项与遗留项

### 待监督方裁定

1. L1 证据文件里已经保留 `81,548` 这个未命中精确零点计数。按 L1b 推断，这些行属于 dev+guard 之外 case；但该归属不是直读标签。本报告采用 192,271 作为主口径分母，并原样保留证据，等待监督方判断这是否构成 RQ007 held_out 污染事件，以及是否需要单独处理 RQ009 已发表原子计数跨界问题。
2. 是否授权打开受保护 confirmation 划分文件，把“2,270 个未命中 case 属 RQ007 held_out”从推断升级为直读确认。未获授权前不得打开。
3. **第 8 节的 23.40% 已由 leader 复算纠正为分母错标**（实际是 J 锚点样本 238/1,017；
   InterHub 门后普查为 5.0097%–9.9516%，分母 3,502,340）。该数字同时出现在
   **K2 报告 §9 与 K2 的 `INTERFACE_NOTE.md`**（下游要读的那份接口说明）。
   需要监督方裁定：是否授权回改 K2 的 `INTERFACE_NOTE.md` 与 §9 措辞。
   **本轨未改 K2 任何文件**，只在本报告内纠正并标注。
   不做的后果是：下游读 `INTERFACE_NOTE.md` 会拿到一个被当作全语料属性的样本值。
4. OnSite 若未来要补齐，需要先决定补齐范围和参考线合同。选项 A 是全 aligned frames；选项 B 是全 RQ009 timing-valid anchor frames；选项 C 是继续每 unit 一个 anchor。参考线选项 A 是沿用 observed trajectory fallback；选项 B 是要求真实地图或车道参考线。不做这些决定的后果是新分母无法定义，结果无法和 InterHub/RQ009 对齐。

### 已知遗留项

1. J 轨 HT 权重分母 2,646,058 与 RQ009 台账行 8,994,736 的关系尚未确立。K2 报告 §4.1 明写 `not yet established`；不得声称两个域已经对齐。
2. OnSite/WOD materializer 未做，PI 已裁定本轮不做。OnSite 2,974 行与 WOD 906 行继续保持门不适用状态。
3. 第 7a 节 held_out 归属是推断，未直读受保护划分文件。

## 状态

state: WAITING_ON_COMMANDER
timestamp_utc: 2026-08-03T03:34:36Z
