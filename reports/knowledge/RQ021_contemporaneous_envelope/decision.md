# RQ021 Decision：人类参照区间改用同期目标量，取代 RQ016C-H2

Status: **ACCEPTED** — PI 裁定 2026-08-05。

Run ID: `RQ021_1_contemporaneous_envelope_20260805T160425Z_43b4bff`
执行层：`reports/studies/RQ021_contemporaneous_envelope/RQ021_1_contemporaneous_envelope_20260805T160425Z_43b4bff/`
Basis: 监督方**未采信执行方报告**，从原始 parquet 独立重建分组与不变量（复核清单见执行层 README）。

## 术语

- **IPV**（Interaction Preference Value）：表示交互倾向的标量。代价函数
  `util = cos(ipv) × 自身代价 + sin(ipv) × 交互代价`（`agent.py:1193`）。IPV 越负 = 越竞争激进。
- **机制一**：判断某一帧的 IPV 数值在七个候选值之间是否携带判别信息（RQ015 冻结）。
- **机制二**：把通过机制一的数值与**人类参照区间（envelope）**比较。
- **envelope**：情境条件化的 split-conformal 区间，由 22 项数值 context + 4 项类别 context
  拟合条件分位数，再用 calibration fold 求 conformal 半径。
- **同期目标量**：由锚点当下 `[t-9, t]` 求解的 IPV，列名 `ipv_log`。
- **预测型目标量**：由锚点之后 `[t+3, t+6]` 求解的 IPV，列名 `target_ipv_future`。

## 本决定要解决的问题

RQ016C-H2 的 envelope 用**预测型**目标量拟合，而机制二在线时拿去比较的是**同期**数值。
两者在纯人-人 test fold 486,660 行上相关 r = 0.3488、中位绝对差 0.3724，
是两个不同的量。这构成估计量错配：参照分布描述的对象与被判定的对象不是同一件事。

PI 裁定：envelope 不需要预测性，它本质上是一次实时的 IPV 判断。
PI 对循环论证顾虑的裁定理由（原样记录）：context 用的是统计性质的数据（速度均值、车道类型等），
IPV 用的是轨迹层的微观特性，二者信息不同质，可以并列使用。

## Accepted Claims

| Claim ID | Evidence ID | Claim | Required Qualification | Paper Section |
|---|---|---|---|---|
| RQ021-KC-C1 | RQ021-E1 | **机制二的人类参照区间改用同期目标量 `ipv_log` 后，参照分布与被判定量在定义上一致。** 行集完全不变：K2 台账全部 8,994,736 行中 `status == "OK"` 6,405,292 行、`ipv_log` 非空 6,405,292 行、两种不匹配各 0 行，即 `ipv_log` 有定义 ⟺ 机制一通过；参照池仍为 2,442,625 行（development 1,752,509 + guard 690,116，held_out 0）。90% 层 coverage 0.902798（417,036/461,937），mean width 1.865128。 | 必须写明这是**目标量的单变量替换**，特征、fold、支持门、conformal 流程未变。 | Methods |
| RQ021-KC-C2 | RQ021-E2 | **「context 与 IPV 用同一批帧会构成循环」这一顾虑在数据上不成立，且方向与顾虑相反。** 条件区间宽度与无 context 边际区间宽度之比：同期目标量 **0.795**，预测型目标量 0.590；条件模型中位数头对目标的 out-of-fold R²：同期 **0.209**，预测型 0.220。两项停止阈值（比值 < 0.25、R² ≥ 0.60）均为**事前**声明，均未触发。 | 必须写明阈值是事前定死、结果出来后未调整。**不得**把 0.795 解释为「context 无用」，它只说明 context 不足以决定 IPV。 | Methods（稳健性） |
| RQ021-KC-C3 | RQ021-E3 | **机制二的支持门与目标量无关，因此 RQ017 的可用率结论不受本次替换影响。** OnSite 支持门通过 21,936/67,861 = 32.3249%、两门交集 14,099/67,861 = 20.7763%，新旧 envelope 下**精确相同**（监督方从两份打分表独立复算）。 | 引用 RQ017 三条主张时无需附加条件。 | Results（方法可用性） |
| RQ021-KC-C4 | RQ021-E4 | **在同期口径下，OnSite 自动驾驶车落在人类参照区间之外的频率与人类自身没有差别。** α=90、两门交集 14,099 帧：下侧 519（3.681%）、上侧 869（6.164%）、合计 1,388（9.845%）；人类自身在同一 envelope 的纯人-人 test fold 上为下侧 4.798% + 上侧 4.922% = 9.720%。 | **描述性，不得写成因果或评价。** 必须同时给出：该比例的分母是两门都通过的帧，不是全部锚点帧；人类侧 9.720% 是 90% 名义区间的构造性结果，两者相近说明**在这一分辨率下无法区分**，不等于「自动驾驶车与人类相同」。 | Results |

### Evidence IDs

| Evidence ID | 来源 | 口径 |
|---|---|---|
| RQ021-E1 | `key_numbers.json → human_only_envelope.metrics`；`envelope_model/conformal_radii_global.json` | 纯人-人 test fold 且支持门通过的 461,937 行；分母见 Claim |
| RQ021-E2 | `key_numbers.json → human_only_envelope.circularity_diagnostics` | D1 分子分母同为 461,937 行；D2 用全部 486,660 行 |
| RQ021-E3 | 监督方从 `.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet` 与 `.codex-fleet/rq021-contemporaneous-envelope/work/E1/onsite_scoring_dryrun.parquet` 独立复算 | OnSite 全部 67,861 行；连接 `data/derived/rq017_onsite_gate/l1_v1` 的 `product_row_key/status` |
| RQ021-E4 | 同 E3，加 `ipv_log` 与 `lo_90/hi_90`；人类侧取 `key_numbers.json → human_only_envelope.metrics.90.lower_tail/upper_tail` | 分母 14,099（OnSite）与 461,937（人类） |

## Superseded

| 被取代对象 | 处置 |
|---|---|
| `reports/studies/RQ016C_human_only_envelope/RQ016C_1_human_only_envelope_20260804T005716Z_7480c173/` 的 envelope | **不再用于机制二打分**。目录原样保留为证据链，不删除、不改写。 |
| 用旧 envelope 得到的 RQ018/RQ019 数字 | 已在本轮全部重跑，各自 `decision.md` 已按新数字改写。 |

## Rejected Or Deferred Claims

| Claim | Reason |
|---|---|
| 「新 envelope 比旧的更准确」 | 两者目标量不同，coverage 不可直接比优劣。本决定的依据是**定义一致性**，不是精度。 |
| 「自动驾驶车与人类的社会交互倾向相同」 | C4 只说在 90% 区间这一分辨率下越界频率无差别。**禁止**升级为等同性主张。 |
| 「越界频率下降说明自动驾驶车变好了」 | 越界频率变化完全由参照区间变宽引起，与任何车辆行为变化无关。**禁止**这样解读。 |
| 跨数据集复现 | 未做。本决定是方法口径修正，其正确性由定义一致性与不变量核验承担，不由外部复现承担。 |

## Boundaries

- **B1** 本决定是**方法口径修正**，不是新的科学发现。C4 是描述性观察。
- **B2** **单一数据集**（OnSite）。外部效度以 OnSite 为限。
  - **B2 补记（2026-08-07，RQ021-E2）**：人类参照池内部的**留一源迁移已实测，边界维持**。
    事前判读带 α=90 覆盖率 ∈ [0.87, 0.93]，四源实测 waymo 0.7425（143,380/193,096）、
    nuplan 0.9899（149,068/150,587）、lyft 0.7497（68,270/91,069）、av2 0.8997（11,315/12,576），
    仅 av2 带内——且其留出弃权率 44.32%（10,012/22,588），是弃权机制拦截无支撑情境的实例。
    与旧 RQ009 检验同构（旧 0.748–0.992，本轮 0.743–0.990）：迁移边界是语料源间异质性的属性，
    不随目标量口径改变。手稿 Fig 3 caption 的边界句不变。执行记录：
    `reports/studies/RQ021_contemporaneous_envelope/RQ021_2_lodo_transfer_20260807T114305Z_0c4d280/`。
- **B3** 分析集是被选出的子集：14,099/67,861 = 20.7763%，正是「人类有可比运动学邻域」的情境。
  全部结论须带「在人类参照存在的情境下」这一条件。
- **B4** D1 = 0.795 说明 context 只能解释同期 IPV 的一部分。这**支持**了目标量替换的合理性，
  但不构成「context 与 IPV 完全独立」的证明。
- **B5** 禁用 `estimability` 与「测出/未测出 IPV」表述；禁用「过度消极」描述下侧越界
  （IPV 越负是越对抗，不是越消极）。描述性结果不得写成因果主张。

## 监督方复核记录

- 独立复现新旧三组计数、支持门与两门交集，全部逐项一致（见执行层 README 的复核清单）。
- 补算了执行方未做的 case/team 分层计数：下侧 175 → 120 case，上侧 182 → 129 case。
- 执行方一处归因不准确（D3 宽度 0.32% 差异被归因为「口径不同」，实为浮点精度），不影响判断。
- 执行方主动申报了 RQ019 重跑脚本中 `expected_90` 输入计数断言的修改，与 diff 一致。

## Paper Handoff

手稿引用机制二时，必须以本轮 envelope 为准。
C4 可用于 Results，但**必须**带上 Required Qualification 的两条。
RQ017 三条主张无需修改（C3）。RQ018/RQ019 见各自 `decision.md`。
