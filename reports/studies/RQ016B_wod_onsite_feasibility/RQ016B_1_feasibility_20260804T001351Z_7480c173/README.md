# RQ016B-1：把重建后的 envelope 用到 WOD 与 OnSite 上的可行性审计（执行记录）

执行日期：2026-08-03 至 2026-08-04｜基线提交：`7480c173`｜执行方：两个 codex agent（F1、F2）｜监督方：Claude

## 这一轮在整个研究里的位置

最终目标是**在线验证**：判断一辆自动驾驶车表现出的社会交互倾向像不像人。判定由两道
串联的弃权机制构成——机制一判断某一帧的 IPV（Interaction Preference Value，表示交互
倾向的标量）数值能否估出，估不出直接弃权；机制二判断当前场景收集到的人类样本是否足以
判断该车是否偏离，依据是人类参照分布（envelope）。

RQ015 完成并冻结了机制一；RQ016 用只过门的样本重建了机制二的 envelope。但两者都只用了
InterHub 的人类数据，而这项研究真正要判的对象——自动驾驶车——在 **WOD**（Waymo Open
Dataset）与 **OnSite**（含自动驾驶车的竞赛场景库）里。

**本轮回答的是：把重建后的 envelope 用到这两个数据集上，要付出什么代价、哪些部分做不成。**

## 两个 agent 各做了什么

- **F1（可行性审计）**：分数据集回答五问——重跑估计器的输入齐不齐、M2 的 29 个 context
  特征能不能造出来、规模有多大、迁移效度证据是什么、最小可行路径与代价。
- **F2（ego 身份查证）**：监督方在准备下一步时实测发现 RQ016 的 envelope 里有 25.94% 的行
  属于含自动驾驶车的交互对，于是派 F2 查清这些行的目标值到底是自动驾驶车自己的 IPV
  还是人的 IPV。

## 主要结论

### WOD：现在做不了

本地包实测只有 4 列 906 行（`segment_key`、`candidate_index`、`ego_ipv`、`ego_ipv_error`），
没有配对轨迹、时间戳、参考线、context，**M2 的 29 个特征全部 MISSING**。要做必须先从上游
重新做一次含轨迹与 context 的脱敏投影，而 WOD 属 RQ014 的致盲边界，该步需 PI 授权。
PI 已裁定本轮放弃 WOD。

### OnSite：可行，输入条件远好于此前假设

`data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet`
有 67,861 行、66 列，**29 个 M2 context 特征一个不缺**（27 个满填；`closing_ttc_anchor`
31,740/67,861 = 46.77%、`apet_online_proxy` 5,364/67,861 = 7.90% 为稀疏）。全部 67,861 行
`av_included == "AV"`，即整张表都是自动驾驶车的锚点。

缺的只有 materializer——按冻结网格与 σ 重跑估计器以取得七候选 MSE。这是算力问题，不是数据缺口。

**监督方对 F1 的一处纠正**：F1 称 OnSite 的几何/转向/优先权标签是启发式、口径与 InterHub
不通。监督方实测为 **OnSite 的取值是 InterHub 的严格子集，67,861/67,861 = 100% 落在
InterHub 也有的取值上**，不存在未知类别；只是覆盖窄（`geometry_path_relation` 用了 13 个
里的 5 个，`turn_pair_label` 用了 24 个里的 9 个）。

### envelope 里混有自动驾驶车自己的 IPV

F2 从代码定死：`target_ipv_future` 取自该行 `ego_key_agent` 一侧
（`build_features.py:665-674`，`perspective` 决定谁是 ego）。监督方以独立路径复核：
`agent_type_pair` 只有 `HV;HV` 与 `AV;HV` 两个取值；在 `AV;HV` 行上
`perspective == "key_agent_1"` 的 `ego_key_agent` 为 `ego` 的比例是 829,784/829,784 = 100%，
`key_agent_2` 为 0%；而 `HV;HV` 行里 `ego` 这个 id 出现 0 次——故 `ego` 是自动驾驶车的专属
track id，判定规则成立且完备。

RQ016 B 臂域（即重建后的 envelope）635,618 行的构成：

| 类别 | 行数 | 占比 |
|---|---:|---:|
| E1 ego 是自动驾驶车（目标值是 AV 自己的 IPV） | 69,288 | 10.9009% |
| E2 ego 是人、对手是自动驾驶车 | 79,670 | 12.5343% |
| E3 纯人-人 | 486,660 | 76.5649% |

拟合用的 fold 同构：train E1 343,017/2,558,374 = 13.41%、calibration 162,097/1,266,282 = 12.80%。
**即模型本身也是部分在自动驾驶车的行为上拟合的。**

这一发现直接导致 PI 在 2026-08-04 裁定改用纯人-人样本重建，执行记录见
`reports/studies/RQ016C_human_only_envelope/`。

### 迁移效度：无同源证据

RQ009 的 LODO（leave-one-dataset-out）表实测只有 4 个留出源，tier=M2、alpha=90 下 90% coverage
分别为 `av2_motion_forecasting` 0.9659、`lyft_train_full` 0.7484、`nuplan_train` 0.9921、
`waymo_train` 0.7902。**OnSite 不在其中，WOD 的 `wod_rq010b_full479_audited` 也不在其中。**
故无同源迁移证据，跨源覆盖在 0.7484–0.9921 之间波动。

## 目录内容

| 文件 | 内容 |
|---|---|
| `RQ016B_1_feasibility.md` | F1 的可行性审计报告（五问，WOD/OnSite 分开） |
| `RQ016B_2_ego_identity.md` | F2 的 ego 身份查证报告 |
| `feasibility_matrix.json` | 29 个特征的逐项判定（AVAILABLE/DERIVABLE/MISSING），WOD 与 OnSite 各一份 |
| `audit_evidence.json` | F1 的列级证据 |
| `ego_identity.json` | F2 的三类计数（E1/E2/E3），分全矩阵、各 fold、B 臂域 |
| `rq016b_f1_audit.py` / `render_rq016b_report.py` | F1 可复跑脚本 |
| `rq016b_f2_ego_identity.py` | F2 可复跑脚本 |
| `RQ016B_F1_kickoff.md` / `RQ016B_F2_kickoff.md` | 两轮任务书 |

## 效度边界

1. **F1 的 WOD 判定只针对本地脱敏产物**，不是对 Waymo 原始数据能包含什么的判断。
2. **无同源迁移证据**（见上）。
3. **`apet_online_proxy` 填充率 OnSite 7.90% 对 InterHub 40.26%**，是系统性差异。
4. **OnSite 与 InterHub 的情境分布重心不同**：OnSite 65% 的行集中在 `F` 类几何
   （44,214/67,861），InterHub 人类池最厚的是 `MP` 类。
5. 描述性结果，不构成因果主张。

## 合规自证

- 未打开受保护的 confirmation 划分文件；未读取 RQ014 致盲相关评分字段。
- F2 的 Q3 只使用结构性列（`fold`、`agent_type_pair`、`perspective`、`av_included`），
  未按 RQ007 split 筛出 held_out 行做分析；B 臂域本身只含 development + guard。
- 未修改 `data/derived/`、RQ009 原 run 目录、五个受保护的估计器/管线/配置文件。
- 本目录已做绝对路径与密钥扫描，命中的用户机器路径已改为仓库相对路径或 `~`。
