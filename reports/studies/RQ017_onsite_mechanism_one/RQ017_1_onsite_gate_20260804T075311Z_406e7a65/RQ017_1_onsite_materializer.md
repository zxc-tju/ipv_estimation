# RQ017-M1 OnSite 机制一判据 materializer 报告

## 任务定位

在线验证由两道串联弃权机制构成：机制一判断某一帧的 IPV 数值是否携带七个候选之间的判别信息，机制二用人类参照分布判断当前场景是否有足够支持。本轮只补 OnSite 67,861 个自动驾驶车 anchor 的机制一判据；不做机制二重新打分，也不对任何车辆作出判断。

已完成的前置工作是 RQ015 在同济 HPC 冻结机制一规格并产出 InterHub K2 台账，RQ016C 建好纯人-人 envelope 并在 OnSite dry-run 中产出机制二支持门。本轮是缺口补齐环节：为 `artifact_id == onsite_dense_timeseries` 的 OnSite anchor 生成七候选 MSE、log-domain 权重、状态和 reason。

## 执行结果

- 正式 OnSite 产物行数：67861/67,861，筛选条件为 `artifact_id == onsite_dense_timeseries`，来源为 `data/derived/rq017_onsite_gate/l1_v1` 的 `product_row_key` 列。
- `OK`：37520/67861，筛选条件为 `status == OK`，来源为正式产物 `status` 列。
- `ABSTAIN`：30341/67861，筛选条件为 `status == ABSTAIN`，来源为正式产物 `status` 列。
- `NEAR_UNIFORM`：30341/67861，筛选条件为 `reason_code == NEAR_UNIFORM`，来源为正式产物 `reason_code` 列。
- `NO_IPV_EFFECT`：0/67861，筛选条件为 `reason_code == NO_IPV_EFFECT`，来源为正式产物 `reason_code` 列。
- 工程失败：0/67861，筛选条件为 `status in [NON_FINITE_INPUT, SOLVER_FAILURE]`，来源为正式产物 `status` 列；其中 `NON_FINITE_INPUT` 0，`SOLVER_FAILURE` 0。
- 与 RQ016C 支持门交叉后的最终可判行数：14099/67861，筛选条件为正式产物 `status == OK` 且 `.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet` 中 `mechanism2_gate_ok == True`，连接列为 `product_row_key`，dry-run 只读取 `product_row_key` 与 `mechanism2_gate_ok` 两列。

## 与 InterHub 对照

InterHub K2 全语料冻结口径为 4,981,984 个求解单元：`OK` 3,502,340/4,981,984，`NEAR_UNIFORM` 1,457,746/4,981,984，`NO_IPV_EFFECT` 19,964/4,981,984，`SOLVER_FAILURE` 1,934/4,981,984；来源为 RQ015K K2 台账的 `status` 与 `reason_code` 列。本轮 OnSite 数字只描述同一机制一规则在 OnSite anchor 上的分布，不构成车辆层判断。

## Blocker 证据

- 测量合同 preflight：`.codex-fleet/rq017-onsite-materializer/work/M1/measurement_contract.json`，状态 `PASS`。C1 三方 `product_row_key` 交集 67861/67,861，C4 短历史分布为 {'10': 66289, '4': 267, '5': 265, '6': 264, '7': 261, '8': 258, '9': 257}。
- 环境同源硬断言：`.codex-fleet/rq017-onsite-materializer/work/M1/env_parity.json`，状态 `PASS`。版本为 {'numpy': '1.21.6', 'pandas': '1.4.4', 'pyarrow': '12.0.1', 'python': '3.9.24', 'scipy': '1.7.3'}，G 锚点本轮重算 `max_abs_diff=0.0`。
- 运行回执：`.codex-fleet/rq017-onsite-materializer/work/M1/run_receipt.json`，Slurm 作业号与分区/节点见该 JSON；`sacct` 分区断言不含 `amd`。

## 坐标异常

preflight 复现 7/67,861 行 `relative_distance_anchor > 100000`，来源为本地 anchor parquet 的 `relative_distance_anchor` 列。7 行全部来自 `onsite:shanghai:T10:C4:native_case:2311`，由 `relative_dx_anchor` 约 -570,761 米而 `relative_dy_anchor` 约 -8 米导致；这更符合单侧坐标原点不一致，不符合双轴同时漂移。它们照常进入正式产物；若求解失败，仅按工程失败记录。

## 自查

- 行数守恒、状态守恒、门判据复算、OK 行恒等式、`K == 7` 与 `legacy7_pi_over_8`、工程失败隔离、数值健康、两条负对照均通过；机器证据见 `.codex-fleet/rq017-onsite-materializer/work/M1/key_numbers.json`。
- 远端与本地逐分片行数和 sha256 一致；证据见 `.codex-fleet/rq017-onsite-materializer/work/M1/run_receipt.json`。

## 待监督方拍板

无本轮执行 blocker 需要新增拍板。正式产物只提供机制一判据；是否进入车辆层在线验证或机制二解释属于后续任务。

state: WAITING_ON_COMMANDER
timestamp_utc: 2026-08-04T07:53:11Z

---

# 监督方附录

以下由监督方追加，**不修改上文 M1 原文**，M1 原状态行保留在上方。

## A. 独立复算（未采信 M1 报告，用自己的脚本从产物重算）

| 量 | 监督方独立算得 | 与 M1 一致 |
|---|---:|---|
| 产物行数 / 键去重 | 67,861 / 67,861 | 是 |
| `K` 唯一值 / 网格 | `{7}` / `legacy7_pi_over_8` | 是 |
| `status == OK` | 37,520 = 55.2971% | 是 |
| `status == ABSTAIN` | 30,341 = 44.7029% | 是 |
| `reason_code == NEAR_UNIFORM` | 30,341 | 是 |
| `reason_code == NO_IPV_EFFECT` | **0** | 是 |
| 工程失败（`NON_FINITE_INPUT` + `SOLVER_FAILURE`） | 0 | 是 |
| 与 RQ016C 支持门交叉，连接命中 | 67,861/67,861 | 是 |
| 两门都过 | **14,099 = 20.7763%** | 是 |

**venue 同源已验证**：`env_parity.json` 状态 `PASS`，G 锚点在本轮环境下重算 2,300 个，
**`max_abs_diff = 0.0`**。故本产物与 K2 的 InterHub 台账处于同一软件栈，对照成立。

**负对照真的 FAIL**：`isclose_atol_1e_12` 判 `FAIL`，`theta_0_22` 同在；合成 sentinel
覆盖四种状态（`ABSTAIN` 3 / `OK` 2 / `NON_FINITE_INPUT` 1 / `SOLVER_FAILURE` 1）。

**7 行坐标异常照常入库**：状态全为 `OK`，`max_w_log` 介于 0.2010–1.0000
（其中一行几乎贴着 0.20 阈值），未被静默剔除。

## B. 对预注册预测的检验（预注册时间戳 2026-08-04T06:22:47Z，早于执行）

| 项 | 预注册预测 | 实测 | 判定 |
|---|---|---|---|
| 机制一通过率 | ≈ 80%（区间 65%–85%） | **55.2971%** | **预测错，落在区间外** |
| 最终可判行数 | ≈ 17,600（区间 11,000–18,000） | 14,099 | 落在区间内 |
| 异常阈值 OK 率 >90% 或 <50% | — | 55.30% | 未触发 |
| 异常阈值 `NO_IPV_EFFECT` 显著 >0.4007% | — | 0 | 未触发（方向相反，见 §D） |
| 异常阈值 `SOLVER_FAILURE` >5% | — | 0 | 未触发 |
| 键一对一 | — | 67,861/67,861 | 成立 |

**预测失败的原因正是预注册时写下的那条弱点**：用于校准的 2,974 行来自
`max_anchors_per_unit=1` 年代，锚点是被选出来的、系统性更有交互，把预测抬高。
校准方法本身可用（回代 InterHub 得 70.3% 对实际 70.30%），坏在样本不具代表性。
漏斗预测落在区间内是两处误差部分抵消，**不构成预测成功**。

## C. Case 级可用性（本轮新增分析，帧级数字不能替代）

分母 **267 个 case**（67,861 帧），源见 `work/M1/case_level_availability.json`：

| | 数量 | 占比 |
|---|---:|---:|
| 至少 1 帧两关都过（case 可用） | 231 | **86.5169%** |
| 全程无一帧两关都过 | 36 | 13.4831% |
| └ 因**机制一**全程无解 | **0** | **0.0000%** |
| └ 因**机制二**全程无参照 | 36 | 13.4831% |

**没有任何一个 case 是全程无法估计 IPV 的。** 那 36 个不可用 case 共 2,897 帧，
其机制一通过率 **53.78%**（与全体 55.30% 基本相同），机制二通过率 **0.07%**
（对全体 32.32%）——它们不是更难估，是没有可比的人类样本。

可用的 231 个 case 中，每 case 可判帧数 min=1 / p25=20 / 中位=50 / p75=90 / max=222，
case 总帧数中位 182，case 内可判比例中位 21.8%。可判帧最多的前 25% case 占全部可判帧的
52.7%，前 50% 占 81.5%。

## D. 两条必须写进边界的观察

**D1 — `NO_IPV_EFFECT` 在 OnSite 上实际不可达。** 不只是计数为 0，而是差得很远：

| | 恰为 0 的行 | 最小非零 `mse_spread` |
|---|---:|---:|
| InterHub（分母 4,981,984） | 19,964 = 0.4007% | 4.77e-15 |
| OnSite（分母 67,861） | **0** | **2.32e-08** |

相差七个数量级。这与观测轨迹 fallback 这条参考线合同一致——参考本身随候选变化，
精确逐位相等不会发生。**因此在 OnSite 上机制一的两条科学弃权理由中只有 `NEAR_UNIFORM`
会触发。拿 OnSite 的 reason 构成去与 InterHub 对比不成立，只能比总弃权率。**

**D2 — 机制二的缺口是重叠不是数量。** 各情境格并列（人类支撑量来自 RQ016C
`envelope_model/support_counts_by_cell.csv`）：

| 情境格 | OnSite 帧 | 人类支撑行 | 机制二通过率 |
|---|---:|---:|---:|
| `MP\|yield` | 7,590 | 1,148,133 | 14.58% |
| `MP\|priority` | 10,291 | 1,044,964 | 13.48% |
| `CP\|priority` | 2,336 | 57,461 | 2.14% |
| `CP\|equal` | 116 | 2,209 | **0.00%** |
| `F\|priority` | 29,677 | 45,283 | **47.03%** |
| `F\|yield` | 14,537 | 46,530 | 32.90% |

`MP` 两格有逾百万行人类样本而通过率仅 13–15%；`F|priority` 仅四万余行却达 47%。
**故限制因素不是人类样本数量，而是自动驾驶车的运动学状态是否落在人类样本附近。**

⚠ 机制二比的是**运动学邻域**（12 项距离特征），不是 IPV 数值本身。
**「机制二不通过」只意味着无法判定，不得解读为「该车不像人」。** 为什么两者不重叠，
本轮没有回答。

## E. 未解释的观察（如实记录，不作解释）

短历史行（`history_row_count < 10`，1,572 行）的机制一通过率为 **73.92%**，
而满历史行（66,289 行）为 **54.85%**——**历史越短反而越容易过门**，方向与直觉相反。
监督方未验证其成因，不提供解释。

state: COMMANDER_VERIFIED
timestamp_utc: 2026-08-04T13:09:28Z
