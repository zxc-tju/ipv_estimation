# RQ027 独立仿真 known-truth IPV recovery 可行性报告

状态：`PILOT_NO_GO`
执行层：development-only S1 + S3 feasibility pilot
分析单位：simulation run；frame 仅用于 persistence 与误差诊断
正式分母：`240` interactive runs + `48` negative-control runs

## 1. 问题与整体阶段

RQ027 要检验的不是在线/离线实现是否一致，而是：当生成端不共享 estimator 的 planner、搜索、代价实现或 likelihood 时，冻结 estimator 能否恢复仿真中已知的 target IPV；候选权重集中度是否对应更低误差；无互动或配对错误时是否会持续给出高集中 reading。

附件原计划经合理性审查后收窄为一个可证伪 pilot。本次已完成该 pilot，是完整研究中的第一道科学可行性门。结果为 NO_GO，因此不进入 S2 扰动、全量消融或 sealed confirmatory。

## 2. 执行健康

| 项目 | 结果 | 口径与来源 |
|---|---:|---|
| Scheduled runs | 288/288 | `run_level_results.csv: run_id` |
| Interactive / negative | 240 / 48 | `run_kind` |
| Target-side attempted frames | 3,456 = 288×12 | `frame_level_results.csv: run_id,step` |
| Engineering failures | 0/288 | `run_status == ENGINEERING_FAILURE` |
| Duplicate run IDs | 0/288 | `run_id` |
| Non-finite primary frames | 0/3,456 | `ipv_log,max_weight,q_eff,mse_spread` |
| Collision-tagged runs | 2/288 | 两个均为 post-resolution negative control；未删除 |
| Exact elapsed time | 590.815 s | `numerical_health.json: elapsed_seconds` |

独立验证逐项复算了行守恒、有限值和关键指标，`validation_status=PASS`。

## 3. Recovery 结果

interactive runs 中有 persistent opportunity-aware reading 的为 `215/240=89.5833%`。高 coverage 并未转化为有用的点值恢复：

| 指标 | Estimator | 零值 predictor | 判读 |
|---|---:|---:|---|
| Accepted-run MAE | 0.553907 rad | 0.553432 rad | estimator 略差 0.000476 rad |
| Accepted-run Spearman | 0.214511 | 不适用 | 仅弱正相关 |
| Half-grid success | 74/215 = 34.4186% | 42/215 = 19.5349% | 容差命中更高，但不足以补偿 MAE |
| One-grid success | 88/215 = 40.9302% | 42/215 = 19.5349% | 同上 |
| All-run one-grid success | 88/240 = 36.6667% | 42/240 = 17.5000% | abstain 计未恢复 |
| 非零真值 sign accuracy | 106/173 = 61.2717% | — | 距预期稳定方向恢复较远 |

因此，`recovery_vs_zero_baseline=false`。这不是说 estimator 完全没有排序信息；它说明在本独立 generator 与冻结配置下，点值恢复没有达到比“始终报 0”更低的平均绝对误差。

### 分层

| Template | Accepted/total | Spearman | MAE rad | Zero MAE rad |
|---|---:|---:|---:|---:|
| ambiguous-priority crossing | 60/60 | 0.189862 | 0.647345 | 0.549779 |
| clear-priority crossing | 60/60 | 0.277647 | 0.521590 | 0.549779 |
| merge | 60/60 | -0.107182 | 0.662066 | 0.549779 |
| same-direction negotiation | 35/60 | 0.817464 | 0.263714 | 0.572219 |

结果明显依赖 geometry：same-direction 有较强恢复，而 merge 方向反转。完整数值见 `factor_summary.csv`。这正是不能用 pooled 结果宣称通用 recovery 的原因。

## 4. Concentration 与选择性风险

`q_eff` 越大表示七候选权重越分散。若 concentration 真能作选择性质量控制，预期 `q_eff` 与误差正相关，且固定门后的误差低于全体。

实测：

- Spearman(`q_eff`, frame absolute error) = `-0.124207`，方向相反；
- 固定 `mse_spread>0 and max(w_log)>=0.20` 通过 `2,476/2,880=85.9722%` interactive frames；
- 门后 MAE `0.597503 rad`，高于全部 attempted frames 的 `0.586513 rad`。

因此 `q_eff_error_not_reversed=false`。本结果与 RQ024 的旧 Gate A 失败方向一致，但不是同一数据或同一生成器的重复：RQ024 是同模型 S0，本轮是独立 S1。

## 5. 负对照

concentration-only persistent false accept 为 `35/48=72.9167%`：

| Negative control | Persistent false accept | Collision-tagged runs |
|---|---:|---:|
| no-conflict neighbour | 10/12 = 83.3333% | 0/12 |
| time-shifted counterpart | 10/12 = 83.3333% | 0/12 |
| wrong-run pseudo-pair | 7/12 = 58.3333% | 0/12 |
| post-resolution window | 8/12 = 66.6667% | 2/12 |

四类均高，不是单一 control 造成。即使把两个 collision-tagged post-resolution runs 视为生成器瑕疵，其他三类仍为 `27/36=75.0%`，所以 NO_GO 不依赖这两行。

## 6. 结论

三个科学门全部失败：

1. recovery MAE 未优于零值 predictor；
2. concentration-error 关系反向，固定门后误差更高；
3. negative-control persistent false accept 过高。

因此当前冻结 estimator + concentration policy 在本独立仿真可行性域内，**不能被写成已验证的 known-truth recovery 或 accuracy-selective abstention**。按预冻结停止条件，S2、3,120/14,040 规模扩展和 sealed confirmatory 均停止；不调门后重跑。

## 7. 证据状态与边界

- `可直接支撑`：本独立 generator、四模板、冻结 exact/log-domain 口径下的 `PILOT_NO_GO`。
- `可作旁证`：same-direction 局部可恢复、既有 S0/parity 工程健康。
- `待核验`：其他独立 generator、更多模板、重新定义的 measurement/gate 合同。
- `不能证明`：所有 IPV estimator 都不可恢复、真实人类不存在固定 IPV、真实系统外部有效性、因果、生产可用性，或任何 accepted RQ017+ 结论应被推翻。

本报告不创建 `decision.md`；知识层接受与否仍由 PI 后续裁定。
