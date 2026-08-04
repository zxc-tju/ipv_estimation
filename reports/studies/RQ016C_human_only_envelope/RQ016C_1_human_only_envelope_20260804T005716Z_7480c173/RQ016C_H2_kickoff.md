# RQ016C-H2 任务书：修正特征集并重拟人类 envelope，且必须在真实 OnSite 行上证明可打分

你是本轮唯一的执行 agent。这是对 H1 产物的一次**修正重跑**，不是新方案。
读完就开工，不写第二版方案，不开子轨。

仓库根即当前工作目录，以下路径都相对仓库根。

---

## 0. 位置与本轮由来（不要跳过）

最终目标是**在线验证**：判断一辆自动驾驶车表现出的社会交互倾向像不像人。
IPV（Interaction Preference Value）是表示交互倾向的标量。判定由两道串联弃权机制构成：
机制一判断这一帧的 IPV 数值能不能估（RQ015 已冻结）；机制二拿通过机制一的数值与
**人类参照分布（envelope）**比。

PI 已裁定：envelope 是建在数据分布之上的查询机制，针对不同目标可建不同 envelope；
本研究要问「OnSite 的自动驾驶车的 IPV 是否落在人类分布范围内」，故参照必须是**纯人类**分布。
上一轮 H1 据此在纯人-人样本上重拟了 envelope，产物在
`.codex-fleet/rq016c-human-only-envelope/work/H1/`。

**但监督方复核发现该产物不能用于它唯一的用途。** 实测：

- 纯人-人训练池 2,442,625 行中，`vehicle_type_list` 含 `AV` 字样的为 **0 行（0.0000%）**，
  取值形如 `['HV','HV']` 2,133,839 行、`['HV','HV','HV']` 285,495 行等。
- OnSite 全部 **67,861/67,861** 行的 `vehicle_type_list` 都是 `['AV','HV']`。
- 因此 OnSite 行落在训练池已见取值上的比例为 **0/67,861 = 0.0000%**——
  用 H1 的持久化产物给 OnSite 打分，**每一行都会撞上训练时未出现的类别**。

`vehicle_type_list` 是第三个携带「双方是不是自动驾驶车」这一信息的列，H1 的特征合同里仍保留了它
（`envelope_model/feature_contract.json` 的 `categorical_context` 含 `vehicle_type_list`）。
这是监督方上一份任务书的规格错误，不是 H1 的执行错误——上一份任务书把类别集写成
「7 项减去 `agent_type_pair` 与 `av_included` = 5 项」。

**本轮就是修正这一点，并且这一次必须用真实 OnSite 行证明产物真的能打分。**

---

## 1. 唯一的规格改动

类别 context 由 5 项改为 **4 项**：

```
geometry_path_category
geometry_path_relation
turn_pair_label
priority_role
```

即在 H1 的 5 项基础上**再移除 `vehicle_type_list`**。理据（写进报告）：
该列编码的是场景中各车辆的类型，其对 OnSite 的判别内容恰好是「这里有一辆自动驾驶车」，
而车辆是否为自动驾驶车正是被检验的对象，不是它所处的情境；保留它会使外部行在打分时
落入训练中从未出现的类别。

**其余一律沿用 H1，不得改动**：22 项数值 context、支持门分格键
`geometry_path_category + priority_role`、12 项支持门距离特征、alpha 层 `[80,90,95]`、
RQ009 fold 结构、conformal 计算方式、样本口径（K2 台账覆盖 + `status == "OK"` +
`agent_type_pair == "HV;HV"`）。不得使用 M3/M4 旧 IPV 通道，不得把 `source_dataset` 作预测变量。

复用 H1 的脚本 `.codex-fleet/rq016c-human-only-envelope/work/H1/run_rq016c_h1_human_only_envelope.py`
与打分脚本 `score_external_rows.py`，改最小必要处即可，不要从零重写。

**已知不变量（对不上就停下报告，不要凑数）**：

```
参照池 2,442,625 行  = development 1,752,509 + guard 690,116，held_out 0
  train 974,984 / calibration 481,088 / guard_tune 499,893 / test 486,660
分格键下纯人-人池 12 格，最小格 2,209 行（CP|equal）
OnSite 落 9 格、无缺格，最小人类支撑 2,209 行（CP|equal，OnSite 116 行）
```

---

## 2. 本轮新增的三条硬性验收（H1 缺的就是这些）

### 2.1 类别词表覆盖断言（**最重要**）

对**每一个**类别 context 特征以及每一个支持门分格键，实测并报告：
**OnSite 的取值有多少落在纯人-人训练池已见取值内**，给出 `命中行数/67,861` 与未命中取值清单。

**四个类别特征必须全部 100.0000% 命中，否则本轮产物仍然不可用，停下报告。**

同时对 22 项数值特征给出 OnSite 值域与训练池值域的对照（min/max/分位数），
标出任何 OnSite 完全落在训练池值域之外的特征——这不构成 FAIL，但必须列出。

### 2.2 真实 OnSite 行的端到端打分演练（**这是本轮产物有效性的唯一证明**）

用持久化产物（**只加载、不重新拟合**）对
`data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet`
的**全部 67,861 行**跑一遍打分路径，产出：

- 每行的区间上下界与宽度（三个 alpha 层）
- 每行的支持门判定（通过/未通过）与所属格
- 汇总：支持门通过率（带分子分母）、逐格通过率、区间宽度分布

落盘到 `.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet`
与其汇总 JSON。

**必须写明的边界，不得省略**：本演练**只证明打分管线在真实 OnSite 行上可运行**，
**不构成对任何一辆自动驾驶车的判定**——因为 OnSite 一行都没有机制一的判据
（七候选 MSE 全部缺失，`status`/`reason_code` 为 0/281,268 非空），机制一未通过之前
不得进入机制二。**报告中禁止出现任何「OnSite 的 AV 落在/不落在人类范围内」这类结论。**

### 2.3 可用的负对照（H1 这条失效了，必须修好）

H1 的负对照报了 `UNEXPECTED_PASS`——扰动之后断言仍然通过，说明那条断言没有在检查它声称检查的东西。

本轮必须做两条负对照，**每条都要真的 FAIL 并把失败输出贴进报告**：

1. 把 `vehicle_type_list` 放回类别 context，然后跑 2.1 的词表覆盖断言——
   **必须 FAIL**（OnSite 命中率会变成 0/67,861）。
2. 把 `agent_type_pair` 放回支持门分格键，然后跑 OnSite 落格检查——
   **必须 FAIL**（OnSite 会缺格）。

**若任一条没有 FAIL，说明断言本身是坏的，必须先修断言再继续，并在报告中说明你怎么修的。**

---

## 3. 交付物

- 报告：`.codex-fleet/rq016c-human-only-envelope/board/reports/RQ016C_2_human_only_envelope_fixed.md`
- 机器数字：`.codex-fleet/rq016c-human-only-envelope/work/H2/key_numbers.json`
- 持久化 envelope：`.codex-fleet/rq016c-human-only-envelope/work/H2/envelope_model/`
  （内容与 H1 同构：逐格 conformal 半径、条件分位数模型、特征清单与编码器、支持门规则、
  逐格支撑量，外加 `HOWTO_score_external_rows.md`）
- OnSite 打分演练产物与汇总（见 2.2）
- 脚本放 `.codex-fleet/rq016c-human-only-envelope/work/H2/`

**不要删除或覆盖 H1 的产物**，它是错误历史，原样保留。

报告要给出：三个 alpha 层的 coverage / 区间宽度 / 机制二弃权率；与 H1 的逐项对照
（说明仅由移除 `vehicle_type_list` 引起）；逐格支撑量；2.1 与 2.2 的完整结果。

---

## 4. 硬边界

```
不改：src/sociality_estimation/core/{agent,ipv_estimation,reliability_logdomain}.py
      pipelines/interhub/process_interhub.py
      configs/ipv_sigma01_exact.json
不改 RQ009 与 RQ016 已落盘的 run 目录（只读）；不改 H1 的产物
不写 data/derived/
不做 git commit / 不碰 git 的任何写操作
禁止 git checkout -- . / restore . / stash / reset --hard / clean -fd
RQ007 held_out 不得被解析。必须先连接 K2 台账再按 fold 切——
   RQ009 的 fold 与 RQ007 的 split 正交，每个 fold 含约 29% held_out，
   只按 fold 过滤会解析 1,899,898 行 held_out。
   报告须含实测断言：参与计算行中 rq007_split 不在 {development, guard} 的计数为 0。
RQ014 致盲相关的评分字段不得读取
不得静默覆盖已冻结产物或已接受的 decision.md
不要对 reports/ 做全仓库 rg；用定向 ls/grep
git status 一律用 git --no-optional-locks status --porcelain
时间戳一律实取 date -u +%Y-%m-%dT%H:%M:%SZ
不投 Slurm/HPC；本机跑（H1 同规模作业约 12 分钟）
```

**措辞禁令**：禁用 `estimability` 与「测出/未测出 IPV」。描述性结果不得写成因果主张。
不得声称复现或未复现 RQ009。不用比喻、不用自造简称。
**分母纪律**：每个百分数必须紧跟分子、分母、筛选条件、来源文件与列名。

环境：本机 python3 已有 pyarrow 21.0.0 / pandas 2.3.3。缺依赖直接装上继续。

## 5. 报告结尾必须带状态行

```
state: WAITING_ON_COMMANDER
timestamp_utc: <实取>
```

报告开头先定位，写给完全没跟进过程的读者。需要监督方拍板的事单独成节。
