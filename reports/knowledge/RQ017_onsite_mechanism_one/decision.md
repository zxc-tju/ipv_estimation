# RQ017 Decision：两道弃权机制在真实自动驾驶车上的可用率

Status: **ACCEPTED** — 能力主张冻结（PI 裁定 2026-08-05）。单数据集；未做跨数据集复现（见 Boundaries）。

> **2026-08-05 复核：RQ021 更换机制二的 envelope 目标量，本决定的三条主张不受影响。**
> 机制二的支持门只依赖 context 特征与参照池行集，与 envelope 的目标列无关；本轮行集与特征均未变。
> 监督方从新旧两份 OnSite 打分表独立复算：支持门通过 **21,936/67,861** 与两门交集
> **14,099/67,861** 在新旧 envelope 下**精确相同**。C1/C2/C3 的全部数字无需修改。
> 详见 `reports/knowledge/RQ021_contemporaneous_envelope/decision.md` 的 RQ021-KC-C3。

Run ID: `RQ017_1_onsite_gate_20260804T075311Z_406e7a65`
执行层：`reports/studies/RQ017_onsite_mechanism_one/RQ017_1_onsite_gate_20260804T075311Z_406e7a65/`
Decision commit: 见本文件所在提交
Basis: 监督方**未采信执行方报告**，用独立脚本从产物重算，逐位一致；环境同源
`env_parity` PASS，G 锚点重算 `max_abs_diff = 0.0`；两条负对照真的 FAIL；
`sacct` 断言所有 array task 分区不含 `amd`。

## 术语（首次出现即解释，勿用黑话）

- **IPV**（Interaction Preference Value）：表示交互倾向的标量。代价函数为
  `util = cos(ipv) × 自身代价 + sin(ipv) × 交互代价`（`src/sociality_estimation/core/agent.py:1193`）。
  **IPV 越大越看重对方代价（更合作让行），越负越反向压制对方（更竞争激进）。**
- **机制一**：判断某一帧的 IPV 数值是否携带七个候选之间的判别信息。不携带则弃权。
- **机制二**：用人类参照分布判断当前情境是否有足够人类样本可比。

## Accepted Claims

| Claim ID | Evidence ID | Claim | Required Qualification | Paper Section |
|---|---|---|---|---|
| RQ017-KC-C1 | RQ017-E1 | **这套两道门的在线检查能够在真实自动驾驶车数据上开口。** 分母 67,861 个自动驾驶车锚点帧：机制一通过 37,520 = 55.2971%，两门都过 14,099 = 20.7763%；工程失败 0。 | 必须给出分母 67,861 与两个比例；不得省略「工程失败 0」。 | Results（方法可用性） |
| RQ017-KC-C2 | RQ017-E2 | **可用率在 case 层远高于帧层，且瓶颈不是轨迹能否被估计。** 分母 267 个 case：至少一帧可判的 231 个 = 86.5169%；全程不可判的 36 个 = 13.4831%，**其中因机制一全程无解的 0 个 = 0.0000%**。 | 必须同时给帧级与 case 级两个分母；「0 个 = 0.0000%」是本条的核心，不得省略。 | Results（方法可用性） |
| RQ017-KC-C3 | RQ017-E3 | **限制因素是人类参照的覆盖，而非人类样本的数量。** 36 个全程不可判 case 的机制一通过率 53.78%（对全体 55.30%），机制二通过率 0.07%（对全体 32.32%）。情境格并列：`MP\|yield` 1,148,133 行人类支撑而通过率 14.58%；`F\|priority` 仅 45,283 行却 47.03%。 | 必须写明「机制二比的是运动学邻域（12 项距离特征），不是 IPV 数值」。 | Discussion（方法边界） |

### Evidence IDs

| Evidence ID | 来源 | 口径 |
|---|---|---|
| RQ017-E1 | `data/derived/rq017_onsite_gate/l1_v1/`（未入库，19 MB，67,861 行）列 `status`、`reason_code` | 筛选 `artifact_id == onsite_dense_timeseries` |
| RQ017-E2 | 同上 + `case_level_availability.json` | 分母 267 个 case |
| RQ017-E3 | 同上 + RQ016C `envelope_model/support_counts_by_cell.csv` | 分组为 `geometry_path_category \| priority_role` |

## Rejected Or Deferred Claims

| Claim | Reason |
|---|---|
| 对任何车辆或队伍作出「像不像人」的判断 | 本轮只产出机制一判据，车辆层在线验证属后续任务。 |
| 以弃权理由构成与 InterHub 对比 | `NO_IPV_EFFECT` 在 OnSite 上实际不可达（0/67,861，最小非零 `mse_spread` 2.32e-08 对 InterHub 4.77e-15，差七个数量级）。**只能比总弃权率。** |
| 「自动驾驶车比人类更难估计」的解释 | 观察成立（OnSite `OK` 55.2971% 对 InterHub 70.3001%），但**成因本轮未验证，不提供解释**。 |
| 短历史行通过率更高的解释 | 观察成立（短历史 73.92% 对满历史 54.85%，方向与直觉相反），成因未验证。 |
| 跨数据集迁移 | 无同源迁移证据；RQ009 的 LODO 只含 4 个留出源，OnSite 不在其中。 |

## Boundaries

- **描述性结果，不构成因果主张。** 禁用 `estimability` 与「测出/未测出 IPV」表述；
  可辩护表述是「权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息」。
- **「机制二不通过」只意味着无法判定，不得解读为「该车不像人」。**
- 参考线用观测轨迹 fallback，与 InterHub 不同源；可比性由「同一估计器、同一冻结配置、
  同一软件栈」保证（G 锚点重算 `max_abs_diff = 0.0`），不由参考线来源保证。
  **必要性依据**：Mac 与 HPC 的求解结果在 81.17%（1,867/2,300）的锚点上不同，
  差异来自软件栈而非 CPU。
- 7 行坐标系异常（`relative_distance_anchor` ≈ 570,761 m，全部来自
  `onsite:shanghai:T10:C4:native_case:2311`）照常参与求解并入库，未静默剔除。
- **单数据集。** PI 于 2026-08-05 裁定本条无需跨数据集复现即可升级为主张；
  监督方记录：跨数据集复现仍是提升外部效度的唯一途径，本条主张的适用范围以 OnSite 为限。

## 过程记录：一次公开失败的预测

监督方在执行前**预注册**了预测（时间戳 `2026-08-04T06:22:47Z`，早于派发）：
预测机制一通过率 ≈ 80%（区间 65%–85%）。**实测 55.2971%，落在区间外，预测失败。**
原因正是预注册时同时写下的那条弱点：校准样本来自 `max_anchors_per_unit=1` 年代，
锚点是被选出来的、系统性更有交互。**从被挑选的子集外推到全集时，选择效应会整体平移结果。**

## Paper Handoff

手稿可使用 C1–C3，作为**方法可用性与适用边界**的陈述。
必须带上 Boundaries 全部条目。**不得**写成对任何车辆的判定，
**不得**把 `NO_IPV_EFFECT` 的跨数据集差异写成方法性质差异。
