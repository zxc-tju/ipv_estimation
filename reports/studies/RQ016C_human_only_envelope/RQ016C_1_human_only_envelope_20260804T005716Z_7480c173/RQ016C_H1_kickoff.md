# RQ016C-H1 任务书：只用纯人-人样本重建一个供 OnSite 使用的人类 envelope

你是本轮唯一的执行 agent。读完就开工，不写第二版方案，不开子轨。
描述性产出：跑出来 → 一轮自查 → 出报告 → 结束。不做盲审，不加授权闸门。

仓库根即当前工作目录，以下路径都相对仓库根。

---

## 0. 这件事在哪一步（不要跳过）

最终目标是**在线验证**：判断一辆自动驾驶车表现出的社会交互倾向像不像人。
IPV（Interaction Preference Value）是表示交互倾向的标量。判定由两道串联的弃权机制构成：
机制一判断这一帧的 IPV 数值能不能估（RQ015 已冻结）；机制二拿通过机制一的数值与
**人类参照分布（envelope）**比。

RQ016 已经重建过一次 envelope，去掉了机制一未通过的行。但监督方随后查实：
**那个 envelope 里 10.9009% 的目标值是自动驾驶车自己的 IPV**
（69,288/635,618，B 臂域）。证据：`build_features.py:665-674` 显示 `target_ipv_future`
取自 ego 一侧；数据侧 track id `ego` 在 `agent_type_pair == "HV;HV"` 的行里出现 0 次、
在 `"AV;HV"` 行里恒为第一位，故 `ego` 即自动驾驶车。

PI 已裁定（2026-08-04）：**envelope 是建在数据分布之上的查询机制，针对不同目标可以建
不同的 envelope。这项研究要问的是「OnSite 的自动驾驶车的 IPV 是否落在人类的分布范围内」，
参照对象就应当是纯人类的 IPV 分布。故本轮只用纯人-人样本重建。**

**本轮就是执行这条裁定。** 产出的 envelope 将来用于给 OnSite 的自动驾驶车打分。

---

## 1. 要建的是什么

在**只含纯人-人交互**的样本上，用与 RQ016 相同的方法重拟一次
context-conditioned split-conformal envelope，并把拟合结果持久化到可以在将来给外部行打分。

### 1.1 样本口径（监督方已实测，直接用）

参照池定义 = 同时满足以下三条的 RQ009 特征矩阵行：

1. 能连接到 K2 台账（保证只含 `development` + `guard`，**不含 RQ007 held_out**）
2. 机制一通过（`status == "OK"`）
3. **`agent_type_pair == "HV;HV"`**（纯人-人）

连接方式：用 RQ009 矩阵构造
`case_key=<...>|anchor_frame_index=<...>|perspective=<...>|source_dataset=<...>`，
与 K2 台账 `data/derived/rq015k_logdomain_gate/l1_v1/`
（筛选 `artifact_id == "rq009_feature_matrix"` 且 `measurement_role == "target_future"`）的
`product_row_key` 做精确连接。

监督方实测的可用量（**你的数必须与之相符，不符就停下报告**）：

```
参照池合计 2,442,625 行   split 组成：development 1,752,509 + guard 690,116，held_out 0
  fold=train        974,984  （该 fold 过门总数 1,290,663，占 75.54%）
  fold=calibration  481,088  （                629,593，占 76.41%）
  fold=guard_tune   499,893  （                646,772，占 77.29%）
  fold=test         486,660  （                635,618，占 76.56%）
```

### 1.2 特征集的必要修改（**这不是可选项**）

在纯人-人池内，`agent_type_pair` 恒为 `"HV;HV"`、`av_included` 恒为 `"all_HV"`——它们是常数。

**必须把这两列同时从 M2 特征集与支持门分格键中移除。** 两个理由，缺一不可：

1. 它们在拟合池内是常数，作为类别特征不携带任何信息；而将来要打分的 OnSite 行取值为
   `"AV;HV"` / `"AV"`——**一个训练时从未出现过的类别**，编码器在预测时会遇到未知类别。
2. `agent_type_pair` 是 RQ009 支持门三个联合分格键之一
   （`GATE_SUPPORT_CATEGORICAL`，见 `calibration.py:158-162`）。若保留，OnSite 的
   `AV;HV` 行落不进任何一个格，机制二会对 OnSite **全量弃权**，本轮产出直接作废。

**这一改动的理据要写进报告**：ego 自己的车辆类型是被检验的对象，不是它所处的情境，
因此不应作为 context 变量。

### 1.3 其余一律照 RQ016 原样，不得改动

- 数值 context：`calibration.py:95-117` 的 `BASE_NUMERIC_CONTEXT` 全部 22 项
- 类别 context：`calibration.py:119-127` 的 `BASE_CATEGORICAL_CONTEXT` 7 项**减去**
  `agent_type_pair` 与 `av_included` → 剩 5 项
- 支持门分格键：`GATE_SUPPORT_CATEGORICAL` 减去 `agent_type_pair` →
  `geometry_path_category` + `priority_role`
- 支持门距离特征：RQ016 用的 12 项（RQ009 `GATE_DISTANCE_NUMERIC` 15 项减去
  3 个 `counterpart_ipv_*` 列）保持不变
- alpha 层 `[80, 90, 95]`、RQ009 fold 结构、conformal 计算方式、随机种子处理：全部照 RQ016
- **不得使用 M3/M4 通道**（`counterpart_ipv_current` 等由旧估计器算出的列，携带伪零）
- **不得把 `source_dataset` 作为预测变量**

参考实现：`reports/studies/RQ016_human_envelope_rebuild/RQ016_1_envelope_rebuild_20260803T134808Z_d23fa836/run_rq016_a1_envelope_rebuild.py`
——**读它、复用它的结构**，不要从零重写，也不要改动它原文件。

---

## 2. 交付物

- 报告：`.codex-fleet/rq016c-human-only-envelope/board/reports/RQ016C_1_human_only_envelope.md`
- 机器数字：`.codex-fleet/rq016c-human-only-envelope/work/H1/key_numbers.json`
- **可复用的已拟合 envelope**：`.codex-fleet/rq016c-human-only-envelope/work/H1/envelope_model/`
  —— 必须能在**不重新拟合**的前提下给外部行打分，至少包含：逐格 conformal 半径、
  拟合好的条件分位数模型、特征清单与编码器、支持门规则与逐格支撑量。
  同目录给一个 `HOWTO_score_external_rows.md`，写明如何拿一批外部行（列名与本 envelope 一致）
  得到区间与「支持/不支持/弃权」判定。**这是本轮最重要的一件产物**，因为它的用途就是将来给
  OnSite 的自动驾驶车打分。
- 脚本放 `.codex-fleet/rq016c-human-only-envelope/work/H1/`，可复跑

### 报告要回答的

1. 纯人-人 envelope 自身的 **coverage**、**区间宽度**、**机制二弃权率**（三个 alpha 层各一组）
2. 与 RQ016 那个含 AV 的 envelope（B 臂）**逐项对照**：coverage 差、宽度差、弃权率差。
   说明差异，但**不要写成因果主张**；两者样本口径与特征集都变了，是两个 envelope，不是同一
   envelope 的前后。
3. **逐格支撑量**：新分格键（`geometry_path_category` + `priority_role`）下每格的样本数。
   监督方已实测过门 E3 池在这套分格下有 12 格，**你的格数与之不符就停下报告**。
4. **OnSite 的落格预演**（只用 context 列，不需要任何 IPV 数值）：
   把 `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet`
   的 67,861 行按新分格键落格，报告每格落多少行、对应的人类支撑量是多少、有没有落进
   本 envelope 没有的格。监督方已实测：OnSite 落进 9 格、无缺格、最小人类支撑 2,209 行
   （格 `('CP','equal')`，OnSite 116 行）。**与之不符就停下报告。**
   注意：这一步**只做落格预演**，不打分——因为 OnSite 还没有机制一的判据，一行都不能判。

---

## 3. 硬边界

```
不改：src/sociality_estimation/core/{agent,ipv_estimation,reliability_logdomain}.py
      pipelines/interhub/process_interhub.py
      configs/ipv_sigma01_exact.json
不改 RQ009 与 RQ016 已落盘的 run 目录（只读，包括那个参考脚本）
不写 data/derived/
不做 git commit / 不碰 git 的任何写操作
禁止 git checkout -- . / restore . / stash / reset --hard / clean -fd
RQ007 held_out 不得被解析。参照池以 K2 台账驱动，天然只含 development + guard；
   报告里必须有一行实测断言：参与计算的行中 rq007_split 不在 {development, guard} 的计数为 0。
   ⚠ RQ009 的 fold 与 RQ007 的 split 正交，每个 fold 都含约 29% held_out，
   **只按 fold 过滤会解析 1,899,898 行 held_out**。必须先连接台账再按 fold 切。
RQ014 致盲相关的评分字段不得读取
不得静默覆盖已冻结产物或已接受的 decision.md
不要对 reports/ 做全仓库 rg；用定向 ls/grep
git status 一律用 git --no-optional-locks status --porcelain
时间戳一律实取 date -u +%Y-%m-%dT%H:%M:%SZ
不投 Slurm/HPC；本机跑（RQ016 同规模作业在本机约 20 分钟完成）
```

**措辞禁令**：全文禁用 `estimability` 与「测出/未测出 IPV」。可辩护表述是
**权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息**。描述性结果不得写成因果主张。
**不得声称本轮结果复现或未复现 RQ009**——域与特征集都不同。
不用比喻、不用自造简称。

**分母纪律**：每个百分数必须紧跟分子、分母、筛选条件、来源文件与列名。

环境：本机 python3 已有 pyarrow 21.0.0 / pandas 2.3.3。缺依赖直接装上继续。

---

## 4. 自查（一轮，但必须有牙齿）

1. **参照池计数**与第 1.1 节的四个 fold 数字逐一相符。
2. **held_out 断言**：参与计算行中 `rq007_split` 不在 `{development, guard}` 的实测计数为 0。
3. **特征集断言**：`agent_type_pair` 与 `av_included` **既不在特征列表、也不在分格键里**，
   用代码断言而不是口头声明。
4. **分格数**与 OnSite 落格预演结果与第 2 节给的实测值相符。
5. **打分接口自测**：从 test fold 里取一小批行，走一遍「不重新拟合、只加载持久化产物」的
   打分路径，验证得到的区间与主流程逐位一致。**这是持久化产物有效性的唯一证明。**
6. **负对照（强制）**：挑一条你自己的验收判据，故意扰动使它失败，把失败输出贴进报告。
7. **数值健康**：NaN/inf 计数、负宽度行数、coverage 是否落在 [0,1]。

## 5. 报告结尾必须带状态行

```
state: WAITING_ON_COMMANDER
timestamp_utc: <实取>
```

报告开头先定位（问题是什么、整体走到哪、本次是哪一环），写给完全没跟进过程的读者。
需要监督方拍板的事单独成节，写清选项、判断依据、不做的后果。
