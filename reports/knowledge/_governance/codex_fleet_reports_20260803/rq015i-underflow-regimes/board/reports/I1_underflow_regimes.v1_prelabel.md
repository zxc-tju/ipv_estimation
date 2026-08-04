# RQ015I 连乘域失效区间规模报告：区间①实测与区间②/③设计基估计（design-based estimate）

生成时间（UTC）：2026-08-01T12:46:01Z。计算脚本：`work/i1_compute_underflow_regimes.py`；报告脚本：`work/i1_make_report.py`。
Python：`<local-rq009-venv>/bin/python`。常量：`TGT=0.6220355269907728`，`MIN_NORMAL=2.2250738585072014e-308`。

## 结论摘要

- 区间①在 4 份 ledger 上可直接识别：严格口径 929,488 / 14,473,982（6.42%），容差口径 966,227 / 14,473,982（6.68%），容差比严格多 36,739 行。
- 区间②/③不能在现有 ledger 上直接分类；下面给出的是设计基估计（design-based estimate）和 95% percentile cluster bootstrap CI，不是普查计数。
- 区间②设计基估计：Mac 合并 1.417% [0.603%, 2.442%]，HPC 合并 0.953% [0.214%, 1.867%]；waymo 高于 nuplan，两版 CI 明显重叠。
- `nzero==6` 风险格设计基估计：Mac 合并 0.807% [0.179%, 1.607%]，HPC 合并 0.372% [0.000%, 0.954%]；这是由下溢删掉六个候选后形成的事实 hard argmax，现有 legacy 标记不会单独提示。
- 按第0节定义，区间③样本不是 2 行：Mac 为 7 行、HPC 为 9 行。预期表中的 2 行是其中 `max_abs_diff > 1e-12` 的可见权重差异子集；主估计仍按第0节定义计算。
- 用现有 ledger 可得列拟合的判别函数 AUC 约 0.94-0.95，但在高特异度规则下只抓到约 31-33% 阳性；因此不能把它当作区间②/③全量实测识别器。

## 输入、筛选与断言

只读取以下输入；未读取 RQ014 致盲评分字段，未提交 HPC 作业，未重算锚点。

| 输入 | 字节 | mtime UTC |
| --- | --- | --- |
| .codex-fleet/rq015b-repair/work/anchor_mse.csv | 2,472,655 | 2026-07-31T10:31:16Z |
| .codex-fleet/rq015g-hpc-resolve/work/anchor_mse_hpc.csv | 2,463,645 | 2026-08-01T01:53:52Z |
| .codex-fleet/rq015b-repair/work/mechanism_split.csv | 333,990 | 2026-07-31T10:31:16Z |
| .codex-fleet/rq015g-hpc-resolve/work/mechanism_split_hpc.csv | 332,640 | 2026-08-01T01:57:14Z |
| concentration_ledger/interhub_sigma01_hw4_timeseries.parquet | 143,534,467 | 2026-07-31T09:40:12Z |
| concentration_ledger/rq009_feature_matrix.parquet | 247,558,811 | 2026-07-31T09:44:45Z |
| concentration_ledger/onsite_dense_timeseries.parquet | 732,770 | 2026-07-31T09:38:19Z |
| concentration_ledger/wod_rq010b_full479_audited.parquet | 28,875 | 2026-07-31T09:38:19Z |

样本文件的 `split` 和 4 份 ledger 的 `rq007_split` 均有硬断言：若出现 `held_out` 立即报错退出。本次未触发；实际 split 分布如下。

| ledger | rq007_split | 行数 |
| --- | --- | --- |
| interhub_sigma01_hw4_timeseries | development | 3,731,250 |
| interhub_sigma01_hw4_timeseries | guard | 1,465,822 |
| onsite_dense_timeseries | RQ007_SPLIT_NOT_APPLICABLE | 281,268 |
| rq009_feature_matrix | development | 6,459,684 |
| rq009_feature_matrix | guard | 2,535,052 |
| wod_rq010b_full479_audited | RQ007_SPLIT_NOT_APPLICABLE | 906 |

区间重算口径：`legacy_density_product[7]` 用 `ast.literal_eval` 解析；`nzero` 是恰等于 `0.0` 的分量数；区间②为 `1 <= nzero <= 6` 且 `legacy_prod_sum > 0`；区间③为 `nzero == 0` 且最小 product 小于 IEEE double 正常数下界。

## 样本基准复现

| 版本 | 量 | 预期 | 本轮重算 | 说明 |
| --- | --- | --- | --- | --- |
| Mac | 非兜底行数 | 1697 | 1697 | 一致 |
| Mac | 区间②行数 | 164（占非兜底 9.7%） | 164（9.7%） | 一致 |
| Mac | nzero直方图（非兜底） | {0:1533,1:94,2:29,3:8,4:9,5:7,6:17} | {0:1533,1:94,2:29,3:8,4:9,5:7,6:17} | 一致 |
| Mac | 区间②上max_abs_diff | p99 5.78e-2，max 0.324 | lower-p99 5.78e-02，max 3.244e-01 | 一致；lower-p99口径复现预期表 |
| Mac | 区间③行数 | 2 | 按第0节定义 7；可见权重差异子集 2 | 不一致；预期表的2行是可见权重差异子集，不是第0节完整定义 |
| Mac | 区间③预期具体行 | ipv_034135\|21\|2、ipv_034642\|27\|2 | ipv_034135\|21\|2、ipv_034642\|27\|2 | 可见权重差异子集一致；均为 waymo / Z / n_obs=11 |
| HPC | 非兜底行数 | 1687 | 1687 | 一致 |
| HPC | 区间②行数 | 150（8.9%） | 150（8.9%） | 一致 |
| HPC | nzero直方图（非兜底） | {0:1537,1:89,2:30,3:7,4:9,5:6,6:9} | {0:1537,1:89,2:30,3:7,4:9,5:6,6:9} | 一致 |
| HPC | 区间②上max_abs_diff | p99 1.04e-4，max 4.08e-3 | lower-p99 1.04e-04，max 4.078e-03 | 一致；lower-p99口径复现预期表 |
| HPC | 区间③行数 | 2 | 按第0节定义 9；可见权重差异子集 2 | 不一致；预期表的2行是可见权重差异子集，不是第0节完整定义 |
| HPC | 区间③预期具体行 | ipv_034135\|21\|2、ipv_034642\|27\|2 | ipv_034135\|21\|2、ipv_034642\|27\|2 | 可见权重差异子集一致；均为 waymo / Z / n_obs=11 |

既有 `partial_underflow` 列的反推语义如下。它不得当作区间②标志位，因为它还包括第0节区间③。

| 版本 | partial但非区间②行数 | 其中nzero==0 | 第0节区间③ | 可见权重差异子集 | 反推语义 |
| --- | --- | --- | --- | --- | --- |
| HPC | 9 | 9 | 9 | 2 | partial_underflow == 区间② 或 第0节区间③ |
| Mac | 7 | 7 | 7 | 2 | partial_underflow == 区间② 或 第0节区间③ |

## 区间①全语料实测

严格口径：`(ipv_error == TGT) & (q_eff == 1.0)`。容差口径：`abs(ipv_error - TGT) <= 1e-15`、`abs(q_eff - 1.0) <= 1e-12`、`abs(k_eff - K) <= 1e-9`。以下分母包含 `NOT_ATTEMPTED` / `UNKNOWN` 行；这些行若 `q_eff`/`k_eff` 为 NaN，自然不会命中。

| ledger | 总行数 | 严格命中 | 严格占比 | 容差命中 | 容差占比 | 容差-严格 | 仅ipv_error容差且q_eff非严格 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| interhub_sigma01_hw4_timeseries | 5,197,072 | 172,094 | 3.31% | 187,340 | 3.60% | 15,246 | 13,718 |
| rq009_feature_matrix | 8,994,736 | 756,805 | 8.41% | 778,298 | 8.65% | 21,493 | 19,011 |
| onsite_dense_timeseries | 281,268 | 366 | 0.13% | 366 | 0.13% | 0 | 0 |
| wod_rq010b_full479_audited | 906 | 223 | 24.61% | 223 | 24.61% | 0 | 0 |
| 合计 | 14,473,982 | 929,488 | 6.42% | 966,227 | 6.68% | 36,739 | 32,729 |

分层计数：

| ledger | rq007_split | measurement_role | 总行数 | 严格命中 | 容差命中 | 容差-严格 |
| --- | --- | --- | --- | --- | --- | --- |
| interhub_sigma01_hw4_timeseries | development | agent_1 | 1,865,625 | 50,038 | 55,501 | 5,463 |
| interhub_sigma01_hw4_timeseries | development | agent_2 | 1,865,625 | 72,697 | 78,058 | 5,361 |
| interhub_sigma01_hw4_timeseries | guard | agent_1 | 732,911 | 21,195 | 23,415 | 2,220 |
| interhub_sigma01_hw4_timeseries | guard | agent_2 | 732,911 | 28,164 | 30,366 | 2,202 |
| onsite_dense_timeseries | RQ007_SPLIT_NOT_APPLICABLE | counterpart_hw10 | 70,317 | 253 | 253 | 0 |
| onsite_dense_timeseries | RQ007_SPLIT_NOT_APPLICABLE | counterpart_hw4 | 70,317 | 23 | 23 | 0 |
| onsite_dense_timeseries | RQ007_SPLIT_NOT_APPLICABLE | ego_hw10 | 70,317 | 88 | 88 | 0 |
| onsite_dense_timeseries | RQ007_SPLIT_NOT_APPLICABLE | ego_hw4 | 70,317 | 2 | 2 | 0 |
| rq009_feature_matrix | development | counterpart_current | 3,229,842 | 432,329 | 437,568 | 5,239 |
| rq009_feature_matrix | development | target_future | 3,229,842 | 108,686 | 118,582 | 9,896 |
| rq009_feature_matrix | guard | counterpart_current | 1,267,526 | 171,849 | 174,125 | 2,276 |
| rq009_feature_matrix | guard | target_future | 1,267,526 | 43,941 | 48,023 | 4,082 |
| wod_rq010b_full479_audited | RQ007_SPLIT_NOT_APPLICABLE | candidate | 906 | 223 | 223 | 0 |

与 `reason_code` 的交叉验证：

| ledger | reason_code | 总行数 | 严格命中 | 容差命中 | reason内命中比例 |
| --- | --- | --- | --- | --- | --- |
| interhub_sigma01_hw4_timeseries | D0_WARMUP | 215,088 | 0 | 0 | 0.00% |
| interhub_sigma01_hw4_timeseries | 空 reason_code | 4,981,984 | 172,094 | 187,340 | 3.76% |
| onsite_dense_timeseries | D0_WARMUP | 4,272 | 0 | 0 | 0.00% |
| onsite_dense_timeseries | EMPTY_CELL_UNEXPLAINED | 274,022 | 0 | 0 | 0.00% |
| onsite_dense_timeseries | 空 reason_code | 2,974 | 366 | 366 | 12.31% |
| rq009_feature_matrix | 空 reason_code | 8,994,736 | 756,805 | 778,298 | 8.65% |
| wod_rq010b_full479_audited | 空 reason_code | 906 | 223 | 223 | 24.61% |

`attempt_status` 与 NaN 分母说明：

| ledger | attempt_status | 行数 |
| --- | --- | --- |
| interhub_sigma01_hw4_timeseries | ATTEMPTED | 4,981,984 |
| interhub_sigma01_hw4_timeseries | NOT_ATTEMPTED | 215,088 |
| onsite_dense_timeseries | ATTEMPTED | 2,974 |
| onsite_dense_timeseries | NOT_ATTEMPTED | 4,272 |
| onsite_dense_timeseries | UNKNOWN | 274,022 |
| rq009_feature_matrix | ATTEMPTED | 8,994,736 |
| wod_rq010b_full479_audited | ATTEMPTED | 906 |

| ledger | 总行数 | ipv_error NaN | q_eff NaN | k_eff NaN |
| --- | --- | --- | --- | --- |
| interhub_sigma01_hw4_timeseries | 5,197,072 | 0 | 215,088 | 215,088 |
| onsite_dense_timeseries | 281,268 | 278,108 | 278,294 | 278,294 |
| rq009_feature_matrix | 8,994,736 | 0 | 0 | 0 |
| wod_rq010b_full479_audited | 906 | 0 | 0 | 0 |

`ipv_error` 满足 `abs(ipv_error - TGT) <= 1e-15` 但 `q_eff != 1.0` 严格不成立的行，只出现在 interhub 与 RQ009，且均为 `ATTEMPTED`、空 `reason_code`：

| ledger | attempt_status | reason_code | 行数 |
| --- | --- | --- | --- |
| interhub_sigma01_hw4_timeseries | ATTEMPTED | 空 reason_code | 13,718 |
| rq009_feature_matrix | ATTEMPTED | 空 reason_code | 19,011 |

这些行的 `q_eff` 和 `k_eff` 都只是浮点尾差；它们在容差口径下进入区间①签名。

| ledger | 行数 | k_eff范围 | \|k_eff-K\|范围 | q_eff范围 | \|q_eff-1\|范围 |
| --- | --- | --- | --- | --- | --- |
| interhub_sigma01_hw4_timeseries | 13,718 | 6.99999999999996 .. 7 | 2.665e-15 .. 3.642e-14 | 0.9999999999999948 .. 0.9999999999999996 | 3.331e-16 .. 5.218e-15 |
| rq009_feature_matrix | 19,011 | 6.99999999999996 .. 7 | 2.665e-15 .. 3.642e-14 | 0.9999999999999948 .. 0.9999999999999996 | 3.331e-16 .. 5.218e-15 |

严格口径与容差口径差异分解：

| ledger | 差异类型 | 行数 |
| --- | --- | --- |
| interhub_sigma01_hw4_timeseries | both_ipv_error_and_q_eff_taildiff | 13,718 |
| interhub_sigma01_hw4_timeseries | ipv_error_taildiff_only | 1,528 |
| interhub_sigma01_hw4_timeseries | q_eff_taildiff_only | 0 |
| onsite_dense_timeseries | both_ipv_error_and_q_eff_taildiff | 0 |
| onsite_dense_timeseries | ipv_error_taildiff_only | 0 |
| onsite_dense_timeseries | q_eff_taildiff_only | 0 |
| rq009_feature_matrix | both_ipv_error_and_q_eff_taildiff | 19,011 |
| rq009_feature_matrix | ipv_error_taildiff_only | 2,482 |
| rq009_feature_matrix | q_eff_taildiff_only | 0 |
| wod_rq010b_full479_audited | both_ipv_error_and_q_eff_taildiff | 0 |
| wod_rq010b_full479_audited | ipv_error_taildiff_only | 0 |
| wod_rq010b_full479_audited | q_eff_taildiff_only | 0 |

## 区间②/③设计基估计（design-based estimate）

分母固定为 `zero_postwarm_scope == True`；每版 1,800 行，cluster 为 `scene_unique_id`，B=2000，seed=20260731，percentile CI。若样本命中数为 0 或 1，CI 标为不可给出。

| 版本 | 分母行 | cluster数 | HT权重和 | source行数 | source权重和 |
| --- | --- | --- | --- | --- | --- |
| Mac | 1,800 | 1,459 | 534,939 | {"nuplan": 900, "waymo": 900} | {"nuplan": 145821.0, "waymo": 389118.0} |
| HPC | 1,800 | 1,459 | 534,939 | {"nuplan": 900, "waymo": 900} | {"nuplan": 145821.0, "waymo": 389118.0} |

| 指标 | 版本 | source/合并 | 样本行 | 样本命中 | cluster | HT命中权重/HT分母 | HT设计基估计占比（95% CI） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 区间②（设计基估计） | Mac | all | 1800 | 19 | 1459 | 7,579.5 / 534,939 | 1.417% [0.603%, 2.442%] |
| 区间②（设计基估计） | Mac | nuplan | 900 | 8 | 632 | 227.84 / 145,821 | 0.156% [0.039%, 0.304%] |
| 区间②（设计基估计） | Mac | waymo | 900 | 11 | 827 | 7,351.66 / 389,118 | 1.889% [0.779%, 3.355%] |
| 区间②（设计基估计） | HPC | all | 1800 | 7 | 1459 | 5,095.65 / 534,939 | 0.953% [0.214%, 1.867%] |
| 区间②（设计基估计） | HPC | nuplan | 900 | 0 | 632 | 0 / 145,821 | 0.000%；CI 不可给出（insufficient_hits） |
| 区间②（设计基估计） | HPC | waymo | 900 | 7 | 827 | 5,095.65 / 389,118 | 1.310% [0.298%, 2.532%] |
| 区间③，第0节定义（设计基估计） | Mac | all | 1800 | 3 | 1459 | 2,045.55 / 534,939 | 0.382% [0.000%, 0.967%] |
| 区间③，第0节定义（设计基估计） | Mac | nuplan | 900 | 1 | 632 | 55.0333 / 145,821 | 0.038%；CI 不可给出（insufficient_hits） |
| 区间③，第0节定义（设计基估计） | Mac | waymo | 900 | 2 | 827 | 1,990.52 / 389,118 | 0.512% [0.000%, 1.286%] |
| 区间③，第0节定义（设计基估计） | HPC | all | 1800 | 3 | 1459 | 2,045.55 / 534,939 | 0.382% [0.000%, 0.967%] |
| 区间③，第0节定义（设计基估计） | HPC | nuplan | 900 | 1 | 632 | 55.0333 / 145,821 | 0.038%；CI 不可给出（insufficient_hits） |
| 区间③，第0节定义（设计基估计） | HPC | waymo | 900 | 2 | 827 | 1,990.52 / 389,118 | 0.512% [0.000%, 1.286%] |
| 区间③可见差异子集，补充（设计基估计） | Mac | all | 1800 | 2 | 1459 | 1,990.52 / 534,939 | 0.372% [0.000%, 0.952%] |
| 区间③可见差异子集，补充（设计基估计） | Mac | nuplan | 900 | 0 | 632 | 0 / 145,821 | 0.000%；CI 不可给出（insufficient_hits） |
| 区间③可见差异子集，补充（设计基估计） | Mac | waymo | 900 | 2 | 827 | 1,990.52 / 389,118 | 0.512% [0.000%, 1.286%] |
| 区间③可见差异子集，补充（设计基估计） | HPC | all | 1800 | 2 | 1459 | 1,990.52 / 534,939 | 0.372% [0.000%, 0.952%] |
| 区间③可见差异子集，补充（设计基估计） | HPC | nuplan | 900 | 0 | 632 | 0 / 145,821 | 0.000%；CI 不可给出（insufficient_hits） |
| 区间③可见差异子集，补充（设计基估计） | HPC | waymo | 900 | 2 | 827 | 1,990.52 / 389,118 | 0.512% [0.000%, 1.286%] |
| nzero==6，事实hard argmax风险格（设计基估计） | Mac | all | 1800 | 10 | 1459 | 4,317.87 / 534,939 | 0.807% [0.179%, 1.607%] |
| nzero==6，事实hard argmax风险格（设计基估计） | Mac | nuplan | 900 | 4 | 632 | 71.3333 / 145,821 | 0.049% [0.003%, 0.135%] |
| nzero==6，事实hard argmax风险格（设计基估计） | Mac | waymo | 900 | 6 | 827 | 4,246.53 / 389,118 | 1.091% [0.244%, 2.189%] |
| nzero==6，事实hard argmax风险格（设计基估计） | HPC | all | 1800 | 2 | 1459 | 1,990.52 / 534,939 | 0.372% [0.000%, 0.954%] |
| nzero==6，事实hard argmax风险格（设计基估计） | HPC | nuplan | 900 | 0 | 632 | 0 / 145,821 | 0.000%；CI 不可给出（insufficient_hits） |
| nzero==6，事实hard argmax风险格（设计基估计） | HPC | waymo | 900 | 2 | 827 | 1,990.52 / 389,118 | 0.512% [0.000%, 1.325%] |

按 signature 的补充设计基估计（只列区间②与 `nzero==6`）：

| 指标 | 版本 | signature | 样本命中 | HT设计基估计占比（95% CI） |
| --- | --- | --- | --- | --- |
| 区间②（设计基估计） | Mac | U | 10 | 0.979% [0.078%, 2.122%] |
| 区间②（设计基估计） | Mac | Z | 9 | 1.798% [0.418%, 3.542%] |
| 区间②（设计基估计） | HPC | U | 0 | 0.000%；CI 不可给出（insufficient_hits） |
| 区间②（设计基估计） | HPC | Z | 7 | 1.782% [0.401%, 3.529%] |
| nzero==6（设计基估计） | Mac | U | 8 | 0.935% [0.036%, 2.068%] |
| nzero==6（设计基估计） | Mac | Z | 2 | 0.696% [0.000%, 1.808%] |
| nzero==6（设计基估计） | HPC | U | 0 | 0.000%；CI 不可给出（insufficient_hits） |
| nzero==6（设计基估计） | HPC | Z | 2 | 0.696% [0.000%, 1.808%] |

Mac 与 HPC 的区间②设计基估计差距在加权外推后缩小：合并差约 0.464 个百分点，且 CI 重叠；waymo 单元差约 0.580 个百分点，CI 同样重叠。HPC 的 nuplan 单元样本命中为 0，只能给点估计 0，CI 不可给出。

## 可识别性结论

判断依据：ledger 的 `ipv_error` / `k_eff` 与 legacy 连乘域更一致。样本中 legacy 通道的 `TGT` 附近和 `q_eff≈1` 质量明显多于 log 通道，且 ledger 大规模出现 `TGT` + `q_eff≈1` + `k_eff≈K` 的区间①签名。因此监督验证使用样本中的 `ipv_error_legacy`、`k_eff_legacy` 和 `q_eff_legacy=k_eff_legacy/K` 对应 ledger 的 `ipv_error`、`k_eff`、`q_eff`。

| 版本 | legacy兜底行 | legacy ipv_error≈TGT | log ipv_error≈TGT | legacy q_eff≈1 | log q_eff≈1 |
| --- | --- | --- | --- | --- | --- |
| Mac | 603 | 1025 | 422 | 1040 | 437 |
| HPC | 613 | 1036 | 423 | 1050 | 437 |

判别函数只使用全语料可得的数值列：`ipv_error`、`k_eff`、`q_eff`。样本里没有可直接对齐的 `candidate_grid_id`、`measurement_role`、`attempt_status`、`reason_code`、`recoverability` 标签；`n_obs` 没有用于模型，因为 ledger 没有该列。

| 版本 | 目标 | 模型 | 阳性数 | ROC-AUC | 灵敏度 | 特异度 | 混淆摘要 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mac | 区间② | balanced_depth3_tree_5fold_oof | 164 | 0.949 | 0.976 | 0.871 | TP 160 / FP 276 / FN 4 |
| Mac | 区间②∪第0节区间③ | balanced_depth3_tree_5fold_oof | 171 | 0.942 | 0.965 | 0.874 | TP 165 / FP 268 / FN 6 |
| HPC | 区间② | balanced_depth3_tree_5fold_oof | 150 | 0.946 | 0.980 | 0.871 | TP 147 / FP 278 / FN 3 |
| HPC | 区间②∪第0节区间③ | balanced_depth3_tree_5fold_oof | 159 | 0.940 | 0.975 | 0.872 | TP 155 / FP 275 / FN 4 |

简单高特异度规则（要求特异度 >= 0.99）如下。它们能较干净地抓到最极端低 `k_eff/q_eff` 尾部，但灵敏度只有约三成。

| 版本 | 目标 | 规则 | 灵敏度 | 特异度 | 混淆摘要 |
| --- | --- | --- | --- | --- | --- |
| Mac | 区间② | k_eff <= 1.29931 | 0.323 | 0.991 | TP 53 / FP 20 / FN 111 |
| Mac | 区间② | q_eff <= 0.185616 | 0.323 | 0.991 | TP 53 / FP 20 / FN 111 |
| Mac | 区间②∪第0节区间③ | k_eff <= 1.29931 | 0.316 | 0.991 | TP 54 / FP 19 / FN 117 |
| Mac | 区间②∪第0节区间③ | q_eff <= 0.185616 | 0.316 | 0.991 | TP 54 / FP 19 / FN 117 |
| HPC | 区间② | k_eff <= 1.33719 | 0.327 | 0.991 | TP 49 / FP 20 / FN 101 |
| HPC | 区间② | q_eff <= 0.191027 | 0.327 | 0.991 | TP 49 / FP 20 / FN 101 |
| HPC | 区间②∪第0节区间③ | k_eff <= 1.33719 | 0.314 | 0.991 | TP 50 / FP 19 / FN 109 |
| HPC | 区间②∪第0节区间③ | q_eff <= 0.191027 | 0.314 | 0.991 | TP 50 / FP 19 / FN 109 |

结论：现有 ledger 可得列不足以直接识别区间②/③的完整集合。若只需要高置信尾部筛查，可用 `k_eff` 或 `q_eff` 阈值获得高特异度；若需要普查级实测计数，现有 ledger 必须新增连乘 product 或等价诊断列后才能完成。本轮未执行任何全量重算。

## 可复算文件

- `work/i1_compute_underflow_regimes.py`：主计算脚本，包含 `held_out` 断言、row-group ledger 扫描、HT ratio 与 cluster bootstrap、判别函数验证。
- `work/i1_make_report.py`：从 CSV 生成本报告。
- `work/i1_sample_baseline.csv`、`work/i1_regime3_definition_rows.csv`、`work/i1_partial_underflow_semantics_rows.csv`：样本基准与既有列语义。
- `work/i1_ledger_regime1_by_artifact.csv`、`work/i1_ledger_regime1_strata.csv`、`work/i1_ledger_reason_cross.csv`、`work/i1_ledger_*`：区间①全语料实测与尾差解释。
- `work/i1_design_estimates.csv`、`work/i1_design_estimates_by_signature.csv`、`work/i1_design_domain_checks.csv`：区间②/③设计基估计。
- `work/i1_classifier_metrics.csv`、`work/i1_simple_rules.csv`、`work/i1_sample_domain_judgment.csv`：可识别性验证。

复算命令：

```bash
<local-rq009-venv>/bin/python .codex-fleet/rq015i-underflow-regimes/work/i1_compute_underflow_regimes.py
<local-rq009-venv>/bin/python .codex-fleet/rq015i-underflow-regimes/work/i1_make_report.py
```
