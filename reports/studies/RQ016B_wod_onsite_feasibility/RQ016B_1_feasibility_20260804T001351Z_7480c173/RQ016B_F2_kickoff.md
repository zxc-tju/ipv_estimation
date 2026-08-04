# RQ016B-F2 任务书：查清人类 envelope 目标值的 ego 到底是谁

你是本轮唯一的执行 agent。这是一个**只读的事实查证任务**，范围很窄，不要扩大。
不跑估计器、不投 Slurm、不训练模型、不做 materializer、不碰 git 写操作。

仓库根即当前工作目录，以下路径都相对仓库根。

---

## 0. 这件事在哪一步，为什么要查（不要跳过）

最终目标是**在线验证**：判断一辆自动驾驶车表现出的社会交互倾向像不像人。
IPV（Interaction Preference Value）是表示交互倾向的标量。判定由两道串联的弃权机制构成：
机制一判断这一帧的 IPV 数值能不能估（RQ015 已冻结）；机制二拿通过机制一的数值
去跟**人类参照分布（envelope）**比，落在区间内判支持、区间外判不支持、样本不足则弃权。

RQ016 刚用只过门的样本重建了这个 envelope。下一步本来要把它用到 OnSite 的自动驾驶车数据上
（OnSite 有 67,861 个 AV 锚点，29 个 context 特征齐全）。

**但监督方在准备这一步时实测发现一个问题**（源
`data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix`）：

- 整个 RQ009 特征矩阵 6,397,266 行中，`agent_type_pair == "AV;HV"` 且 `av_included == "AV"`
  的有 **1,659,568 行 = 25.9418%**，其余 4,737,698 行是 `HV;HV` / `all_HV`。
- 分 fold：train 686,034/2,558,374 = 26.82%、calibration 324,194/1,266,282 = 25.60%、
  guard_tune 323,510/1,302,044 = 24.85%、test 325,830/1,270,566 = 25.64%。
- **RQ016 重建后的 envelope 里仍有这些行**：B 臂域 635,618 行中 `AV;HV` 为
  **148,958 = 23.44%**。RQ016 的行过滤只有 `status == OK` 一条，从未按 AV/HV 筛选过。
- 这些 `AV;HV` 行的 `perspective` 是**精确五五开**：`key_agent_1` 829,784 行、
  `key_agent_2` 829,784 行。
- `vehicle_type_list` 在这些行上的取值为 `['HV','AV']` 1,321,006 行、`['HV','HV','AV']` 293,030、
  `['HV','HV','HV','AV']` 25,008、`['AV','HV']` 12,724、`['HV','AV','HV']` 6,244 等。

**问题是**：`AV;HV` 只说明这一对里有一辆自动驾驶车，**不等于被测的那个 IPV 就是它的**。
每一对都算了两个视角，所以其中可能有相当一部分行，**被当作「人类参照」的目标值
其实是自动驾驶车自己的 IPV**。若如此，用这个 envelope 去判 OnSite 的自动驾驶车就是循环论证。

**你的任务就是把这件事查清楚，给出可复算的计数。** 不要做别的。

---

## 1. 你要回答的三个问题

### Q1 — `target_ipv_future` 是谁的 IPV？

查清 envelope 的目标列 `target_ipv_future`（以及作为对照的 `counterpart_ipv_current`）
在构造时取的是哪一方的 IPV：是该行 `ego_key_agent` 所指的那个 agent，还是
`counterpart_key_agent` 所指的那个。

**必须从代码定死，不要从列名猜。** 关键文件：

- `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/03_features/build_features.py`
- 同目录 `finalize_features.py`
- 同目录 `feature_dictionary.csv`（`ego_key_agent`、`counterpart_key_agent`、`perspective`、
  `target_ipv_future`、`vehicle_type_list`、`av_included`、`agent_type_pair` 的定义原文）

**给出行号级引用**（`文件:行号`）与关键代码片段。

### Q2 — 每一行的 ego 是 AV 还是人？

查清 `perspective`（`key_agent_1` / `key_agent_2`）、`ego_key_agent`、
`counterpart_key_agent`、`key_agents`、`vehicle_type_list` 这几列之间的对应规则：
**`vehicle_type_list` 的元素顺序是否与 key_agent 的编号对应？**

要点：`vehicle_type_list` 的长度常常大于 2（如 `['HV','HV','AV']`），说明它列的是**场景里所有
agent 的类型**，而不只是这一对的两个。所以不能想当然按位置取。必须查清它的索引口径，
以及如何由它判定某个 `key_agent` 的车辆类型。

**若从现有列 + 代码无法唯一确定某一行的 ego 车辆类型，直说无法确定，并说明缺什么。
不要编一个规则然后按它统计。**

### Q3 — 三个计数

在能确定的前提下，给出这三类的行数，**每个数带分子、分母、筛选条件、来源文件与列名**：

- **E1 = ego 是 AV** 的行（其目标值是自动驾驶车自己的 IPV）
- **E2 = ego 是人、但对手是 AV** 的行（目标值是人的，但交互对象是 AV）
- **E3 = 纯人-人** 的行

对以下每个群体各算一遍：

1. 整个 RQ009 特征矩阵（分母 6,397,266）
2. 每个 fold（train / calibration / guard_tune / test，分母分别为 2,558,374 / 1,266,282 / 1,302,044 / 1,270,566）
3. **RQ016 的 B 臂域**（重建后的 envelope，分母 635,618）——这是最关键的一个。
   该域的定义是：RQ009 `fold == test` 的行，按
   `case_key=<...>|anchor_frame_index=<...>|perspective=<...>|source_dataset=<...>`
   连接到 K2 台账 `data/derived/rq015k_logdomain_gate/l1_v1/`
   （筛选 `artifact_id == rq009_feature_matrix` 且 `measurement_role == target_future`）的
   `product_row_key`，取命中且 `status == "OK"` 的行。
   监督方已实测该域为 635,618 行，其中 `AV;HV` 为 148,958 行——**你的 E1+E2 之和必须等于
   148,958，对不上就说明规则错了，停下来报告，不要强行凑数。**

---

## 2. 硬边界

```
只读。禁止运行估计器、禁止 Slurm/HPC、禁止训练模型、禁止写 data/derived/
不改：src/sociality_estimation/core/{agent,ipv_estimation,reliability_logdomain}.py
      pipelines/interhub/process_interhub.py
      configs/ipv_sigma01_exact.json
不改 RQ009 与 RQ016 已落盘的 run 目录（只读）
不做 git commit / 不碰 git 的任何写操作
禁止 git checkout -- . / restore . / stash / reset --hard / clean -fd
RQ007 held_out 不得被解析。注意：RQ009 的 fold 与 RQ007 的 split 正交，每个 fold 都含约 29%
   held_out，按 fold 过滤会解析 1,899,898 行 held_out。
   **Q3 的第 1、2 项只统计车辆类型与 perspective 这类结构性字段，不涉及 held_out 的结果或标签内容；
   若你需要读 rq007_split，只允许做取值计数，不得据其筛选出 held_out 行做任何分析。**
   第 3 项（B 臂域）本身只含 development + guard，天然安全。
RQ014 致盲相关的评分字段不得读取
不得静默覆盖已冻结产物或已接受的 decision.md
不要对 reports/ 做全仓库 rg；用定向 ls/grep
git status 一律用 git --no-optional-locks status --porcelain
时间戳一律实取 date -u +%Y-%m-%dT%H:%M:%SZ
```

**措辞禁令**：全文禁用 `estimability` 与「测出/未测出 IPV」。描述性结果不得写成因果主张。
不用比喻、不用自造简称。

**分母纪律**：每个百分数必须紧跟分子、分母、筛选条件、来源文件与列名。

环境：本机 python3 已有 pyarrow 21.0.0 / pandas 2.3.3。缺依赖直接装上继续。

---

## 3. 交付物

- 报告：`.codex-fleet/rq016b-wod-onsite-feasibility/board/reports/RQ016B_2_ego_identity.md`
- 机器可读：`.codex-fleet/rq016b-wod-onsite-feasibility/work/F2/ego_identity.json`
- 可复跑脚本放 `.codex-fleet/rq016b-wod-onsite-feasibility/work/F2/`

报告开头先定位（问题是什么、整体走到哪、本次是哪一环），写给完全没跟进过程的读者。
**结论先行**：先直接回答「envelope 的目标值里有没有自动驾驶车自己的 IPV，有多少」，再给证据。

## 4. 自查（一轮，但必须有牙齿）

1. **E1 + E2 必须等于 148,958**（B 臂域的 `AV;HV` 行数）。对不上就停下报告。
2. **E1 + E2 + E3 必须等于该群体的总行数**，逐群体核对。
3. **负对照（强制）**：挑一条你判定 ego 车辆类型的规则，故意扰动使它失败，把失败输出贴进报告。
4. **代码引用可核**：Q1 的结论必须能由你给的 `文件:行号` 直接验证。
5. 若结论是「无法唯一确定」，同样要给出你排除了哪些可能、缺哪一块信息才能定。

## 5. 报告结尾必须带状态行

```
state: WAITING_ON_COMMANDER
timestamp_utc: <实取>
```
