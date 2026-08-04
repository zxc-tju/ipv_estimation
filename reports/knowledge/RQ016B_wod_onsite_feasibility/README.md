# RQ016B：把重建后的 envelope 用到 WOD 与 OnSite 的可行性（知识层）

状态：**审计完成，结论已由监督方独立复算；尚无已接受的手稿主张（无 `decision.md`）**

执行层：`reports/studies/RQ016B_wod_onsite_feasibility/RQ016B_1_feasibility_20260804T001351Z_7480c173/`

## 问题

在线验证要判断一辆自动驾驶车表现出的社会交互倾向像不像人。判定串联两道弃权机制：
机制一判断某一帧的 IPV（Interaction Preference Value，表示交互倾向的标量）数值能否估出
（RQ015 已冻结）；机制二拿它与人类参照分布（envelope）比。RQ015 与 RQ016 都只用了 InterHub
的人类数据，而真正要判的自动驾驶车在 WOD 与 OnSite 里。

**RQ016B 问的是**：把 envelope 用到这两个数据集上，要付什么代价、哪些部分做不成。

## 结论

**B1 — 直接套用不可行，卡在机制一。** WOD 与 OnSite **一行都没有**七候选 MSE：
`mse_0..mse_6`、`max_w_log`、`mse_spread`、`status`、`reason_code` 在
`wod_rq010b_full479_audited`（906 行）与 `onsite_dense_timeseries`（281,268 行）上的非空计数
全部为 0，`gate_applicable` 全为 False。要做必须先按冻结网格与 σ 重跑估计器（materializer）。

**B2 — WOD 本地不具备做的条件。** 本地包只有 4 列 906 行
（`segment_key`、`candidate_index`、`ego_ipv`、`ego_ipv_error`），M2 的 29 个 context 特征
全部 MISSING。要做必须先从上游重做一次含轨迹与 context 的脱敏投影，而该步触及 RQ014 致盲边界。
PI 已裁定本轮放弃 WOD。

**B3 — OnSite 的输入条件远好于此前假设。**
`onsite_m3_av_anchors_multi_allvalid.parquet` 有 67,861 行、66 列，**29 个 M2 特征一个不缺**
（27 项满填；`closing_ttc_anchor` 31,740/67,861、`apet_online_proxy` 5,364/67,861 为稀疏），
且全部 67,861 行 `av_included == "AV"`。四个类别特征的取值 100% 被 InterHub 覆盖
（67,861/67,861）。缺的只有 materializer——是算力问题，不是数据缺口。

**B4 — RQ016 的 envelope 里混有自动驾驶车自己的 IPV。**
`target_ipv_future` 取自 ego 一侧（`build_features.py:665-674`），而 `ego` 是自动驾驶车的
专属 track id（`AV;HV` 行上 `perspective=key_agent_1` 时 `ego_key_agent=='ego'` 为
829,784/829,784 = 100%；`HV;HV` 行里 `ego` 出现 0 次）。RQ016 B 臂域 635,618 行的构成为：
E1 ego 是自动驾驶车 69,288 = 10.9009%、E2 ego 是人对手是 AV 79,670 = 12.5343%、
E3 纯人-人 486,660 = 76.5649%。拟合 fold 同构（train E1 13.41%、calibration 12.80%）。

**这一条直接触发了 PI 在 2026-08-04 的裁定：改用纯人-人样本重建，见 RQ016C。**

**B5 — 无同源迁移证据。** RQ009 的 LODO 只有 4 个留出源（tier=M2、alpha=90 下 90% coverage
分别为 av2 0.9659、lyft 0.7484、nuplan 0.9921、waymo 0.7902），**OnSite 与 WOD 的
`wod_rq010b_full479_audited` 都不在其中**。

## 效度边界

1. WOD 的判定只针对本地脱敏产物，不是对 Waymo 原始数据能包含什么的判断。
2. 无同源迁移证据；跨源覆盖在 0.7484–0.9921 之间波动。
3. `apet_online_proxy` 填充率 OnSite 7.90% 对 InterHub 40.26%，系统性差异。
4. OnSite 与 InterHub 情境分布重心不同：OnSite 65% 的行在 `F` 类几何，InterHub 人类池最厚的是 `MP`。
5. 描述性结果，不构成因果主张。

## 监督方对执行方的一处纠正

F1 称 OnSite 的几何/转向/优先权标签「是启发式，不是 InterHub 的审计标签」，暗示口径不通。
监督方实测为 **OnSite 的取值是 InterHub 的严格子集，100% 可落格**，不存在未知类别；
只是覆盖窄。原判断保留在执行记录中，此处订正。

## 待决事项（属下一轮）

OnSite materializer 动工前，PI 需先定两件事，否则新分母无法定义：

- **范围**：全 aligned frames（70,317 物理帧）／全 timing-valid anchor frames（67,861）／
  继续每 unit 一个 anchor（267）
- **参考线合同**：沿用 observed trajectory fallback／要求真实地图或车道参考线
  （OnSite dense 源表真实地图/车道/route/reference-line 字段为 0/274,022）
