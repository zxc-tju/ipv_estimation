# RQ021 任务书 v1：把人类参照区间的目标量从「锚点之后的读数」换成「锚点当下的读数」，并重新给 OnSite 打分

你是本轮唯一的执行 agent。这是对 RQ016C-H2 产物的一次**单变量重跑**，不是新方案。
读完就开工。不写第二版方案，不开子轨，不提替代设计。

仓库根即当前工作目录，以下路径都相对仓库根。

---

## 0. 位置与本轮由来（不要跳过）

**这项研究要解决什么问题。** 最终目标是**在线验证**：判断一辆自动驾驶车（AV）当下表现出的
社会交互倾向像不像人。IPV（Interaction Preference Value）是表示交互倾向的标量，
数值越正表示越把对方的代价计入自己的目标，越负表示越对抗对方。

**整体走到哪一步了。** 判定由两道串联弃权机制构成：

- **机制一**判断这一帧的 IPV 数值在七个候选值之间是否携带判别信息（RQ015 已冻结，本轮不得改动）；
- **机制二**拿通过机制一的数值，与**人类参照区间（envelope）**比，看是否落在人类范围内。

RQ016C-H2 已建成当前在用的纯人-人 envelope；RQ017 已在 OnSite 的 67,861 个锚点行上算完两门；
RQ018 与 RQ019 已基于该 envelope 得到 OnSite 的关联性结果并被接受为主张。

**本次是其中哪一环。** 监督方复核发现现用 envelope 存在一处**目标量错配**，本轮修正它：

| 量 | 由哪些帧求解 | 存放位置 |
|---|---|---|
| envelope 现在的目标列 `target_ipv_future` | 锚点**之后**的 `[t+3, t+6]`（历史窗 4 帧，`TARGET_FINAL_OFFSET = 6`） | RQ009 矩阵 |
| 在线时监控器实际算出的量 `ipv_log` | 锚点**当下**的 `[t-9, t]`（历史窗 10 帧） | K2 台账 |
| context 特征 | 锚点**当下**的 `[t-9, t]` | RQ009 矩阵 |

也就是说，现用 envelope 是一个**预测型**参照：它用当下的情境去预测 0.6 秒之后的 IPV，
但机制二实际拿去比对的是**当下**的 IPV。监督方实测这两个量在纯人-人 test fold 的
486,660 行上相关只有 **r = 0.3488**，中位绝对差 **0.3724**——它们不是同一个量。

**PI 已于 2026-08-05 裁定：envelope 不需要做预测性的，它本质上是一次实时的 IPV 判断。**
PI 对「context 与 IPV 用同一批帧会不会构成循环论证」的裁定理由如下，原样记录，本轮不得重新讨论：

> context 的计算和 IPV 的计算虽然用的时间窗口是一样的，但它们所利用的信息精度是不同质的：
> context 建立用的是统计性质的数据，比如速度均值、车道类型等信息；IPV 计算用的是轨迹层的
> 微观特性。这两者的交叉程度不会很高，可以并列使用，不需要做预测性的。

这条裁定是**可实测的**，第 3 节给出了实测口径与**事前声明的停止阈值**。你的任务是执行并如实报告，
不是论证它对或不对。

---

## 1. 唯一的规格改动

**目标列由 `target_ipv_future` 改为 `ipv_log`。其余一切不变。**

这是一次严格的单变量改动，因为**行集完全不变**。监督方已实测确认（K2 台账
`rq009_feature_matrix` 全部 8,994,736 行）：

```
status == "OK"          6,405,292 行
ipv_log 非空            6,405,292 行
status==OK 且 ipv_log 为空        0 行
status!=OK 且 ipv_log 非空        0 行
```

即 `ipv_log` 有定义 **⟺** 机制一 `status == "OK"`。而 RQ016C-H2 的参照池本来就已经筛了
`status == "OK"`，所以换目标列**一行都不会增减**。

以下**全部沿用 RQ016C-H2，不得改动**：

- 22 项数值 context + 4 项类别 context（`geometry_path_category` / `geometry_path_relation` /
  `turn_pair_label` / `priority_role`）
- 支持门分格键 `geometry_path_category + priority_role`、12 项支持门距离特征
- alpha 层 `[80, 90, 95]`，90% 为主口径
- RQ009 fold 结构、split-conformal 计算方式
- 样本口径：K2 台账精确连接 + `status == "OK"` + `agent_type_pair == "HV;HV"` + `rq007_split ∈ {development, guard}`
- 不得把 `agent_type_pair` / `av_included` / `vehicle_type_list` / `source_dataset` 作预测变量或分格键

**唯一的实现差异**：`target_ipv_future` 取自 RQ009 矩阵，`ipv_log` 取自 K2 台账
（`data/derived/rq015k_logdomain_gate/l1_v1/artifact_id=rq009_feature_matrix` 的 `ipv_log` 列）。
两侧本来就已经在同一次连接里，把目标从矩阵一侧改到台账一侧即可。

复用脚本 `.codex-fleet/rq016c-human-only-envelope/work/H2/run_rq016c_h2_human_only_envelope.py`
与 `score_external_rows.py`，改最小必要处，**不要从零重写**。

**已知不变量（对不上就停下报告，不要凑数）：**

```
参照池 2,442,625 行 = development 1,752,509 + guard 690,116，held_out 0
  train 974,984 / calibration 481,088 / guard_tune 499,893 / test 486,660
分格键下纯人-人池 12 格，最小格 2,209 行（CP|equal）
OnSite 落 9 格、无缺格，最小人类支撑 2,209 行（CP|equal，OnSite 116 行）
```

---

## 2. 五个步骤

### 步骤 1 — 重训 envelope（主要工作量）

按第 1 节的规格重训，产出与 RQ016C-H2 同构的持久化产物。
报告三个 alpha 层的 coverage、mean/median width、机制二弃权率，口径写法照抄 RQ016C-H2 报告。

### 步骤 2 — 循环性诊断（**本轮的关键**，见第 3 节）

### 步骤 3 — 检查点：若第 3 节任一停止阈值触发，**在此停下并报告**，不执行步骤 4、5

### 步骤 4 — 重新给 OnSite 打分

用新产物（**只加载、不重新拟合**）对
`data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet`
的全部 **67,861** 行重跑打分路径，落盘为
`.codex-fleet/rq021-contemporaneous-envelope/work/E1/onsite_scoring_dryrun.parquet`，
schema 与 RQ016C-H2 的同名文件一致（`lo_80/hi_80/width_80`、`lo_90/...`、`lo_95/...`、
`mechanism2_gate_ok`、`context_cell`）。**不得覆盖 RQ016C-H2 的那一份。**

**机制一完全不动。** OnSite 的 `status` / `ipv_log` 直接读
`data/derived/rq017_onsite_gate/l1_v1`，不得重算、不得改动冻结的门规格。

### 步骤 5 — 用新打分重跑 RQ018 与 RQ019

原样复用两份已落盘的分析脚本，**只改机制二输入路径这一个常量**：

| 脚本 | 要改的常量 | 新值 |
|---|---|---|
| `reports/studies/RQ018_abnormal_ipv_degradation/RQ018_1_association_20260804T224427Z_276cf4c/rq018_association.py:36` | `M2_REL` | 步骤 4 的新 parquet |
| `reports/studies/RQ019_counterpart_burden/RQ019_1_counterpart_burden_20260805T014215Z_7b9f47b/rq019_counterpart_burden.py:149` | 同名机制二路径 | 同上 |

**除输入路径与输出目录外，不得改动这两个脚本的任何分析逻辑、模型设定、阈值、bootstrap 设置或报告文字模板。**
把脚本副本放进本轮 work 目录再改，**不要就地修改 `reports/studies/` 下已落盘的运行目录**。
输出写到 `.codex-fleet/rq021-contemporaneous-envelope/work/E1/rq018_rerun/` 与 `.../rq019_rerun/`。

报告须给出**新旧对照表**：对 RQ018 与 RQ019 各自已被接受的关键数字，逐项列出旧值、新值、
以及是否改变结论方向。至少必须覆盖下列已接受数字（旧值来自各自的
`rq0xx_supervisor_verification.json`）：

- RQ018：ego 未来最小 TTC 的 lower/inside 中位数与四分位数、四个危险阈值（TTC < 1 / 1.5 / 2 / 3 s）的帧占比及 case 层 bootstrap CI
- RQ019：对手方锚点降速倍数、速度极差倍数、总航向变化差值、强制动（< −2 / −3 / −4 m/s²）帧占比及 case 层 p 值

---

## 3. 循环性诊断与**事前声明的停止阈值**

阈值现在就定死。**看到结果之后不得调整阈值，也不得新增「但是」条款。**

### 3.1 边际基线（必须先做，它是所有比值的分母）

在**同一批行、同一 fold 结构、同一 conformal 流程**下，另拟一个**无 context** 的边际
envelope（预测全局分位数），对 **`ipv_log` 与 `target_ipv_future` 两个目标各做一次**。
报告两者的 90% 层 mean width。

### 3.2 三个诊断量

| # | 诊断量 | 定义 | 停止阈值 |
|---|---|---|---|
| D1 | 宽度比 | `mean_width_90(条件) / mean_width_90(边际)`，同目标同行集 | **< 0.25 → 停止并报告** |
| D2 | 点预测解释力 | 条件分位数模型的中位数头在 test fold 上对 `ipv_log` 的 out-of-fold R² | **≥ 0.60 → 停止并报告** |
| D3 | 对照参考 | 对 `target_ipv_future` 同样算 D1 与 D2 | 无阈值，只作对照 |

D3 是为了让 D1/D2 可读：如果新目标的宽度比和旧目标差不多，说明「换目标没有额外引入
context 对 IPV 的解释力」；如果新目标的宽度比显著更低，说明确有额外解释力，需要报告幅度。

**触发即停**指：完成步骤 1–3 与本节全部诊断，写完报告，**不执行步骤 4、5**，
在报告的「待监督方拍板」一节写清触发了哪一条、实测值多少、以及不继续的后果。
**不要自行决定放宽阈值继续跑。**

### 3.3 必须报告的对照数字（RQ016C-H2 旧值，来自其报告与 `key_numbers.json`）

```
90% coverage = 0.898272（414,945/461,937）
90% mean width = 1.242394
90% median width = 1.271731
机制二弃权率 = 5.0801%（24,723/486,660）
80% coverage = 0.796022（367,712/461,937），mean width = 0.783479
95% coverage = 0.949064（438,408/461,937），mean width = 1.710243
```

监督方另已实测（纯人-人 test fold 486,660 行）：
`ipv_log` 中位数 +0.1173、标准差 0.6398；`target_ipv_future` 中位数 0.0000、标准差 0.4877。
新 envelope 的宽度**理应比旧的更宽还是更窄，本轮不预设方向**，如实报告即可。

---

## 4. 必须成立的强不变量（对不上一律停下报告）

支持门只依赖 context 特征与行集，**与目标列无关**。本轮行集与特征都没变，因此：

```
纯人-人 test fold 机制二弃权率必须精确等于 5.0801%（24,723/486,660）
OnSite 机制二支持门通过必须精确等于 21,936/67,861
OnSite 两门交集（机制一 OK 且机制二支持门通过）必须精确等于 14,099/67,861
OnSite 落 9 格、无缺格
四项类别特征的 OnSite 词表命中率必须全部 100.0000%（67,861/67,861）
```

任何一条对不上，说明改动溢出了目标列，**停下报告，不要继续**。

## 5. 负对照（每条都必须真的 FAIL，并把失败输出贴进报告）

1. 把目标列改回 `target_ipv_future` 后跑「目标列即 `ipv_log`」的合同断言——**必须 FAIL**。
2. 把 `vehicle_type_list` 放回类别 context 后跑 OnSite 词表覆盖断言——**必须 FAIL**
   （命中率会变成 0/67,861）。

若任一条没有 FAIL，说明断言本身是坏的，先修断言再继续，并在报告中说明怎么修的。

---

## 6. 交付物

- 报告：`.codex-fleet/rq021-contemporaneous-envelope/board/reports/RQ021_1_contemporaneous_envelope.md`
- 机器数字：`.codex-fleet/rq021-contemporaneous-envelope/work/E1/key_numbers.json`
- 持久化 envelope：`.codex-fleet/rq021-contemporaneous-envelope/work/E1/envelope_model/`
  （与 RQ016C-H2 同构，含 `HOWTO_score_external_rows.md`）
- OnSite 打分产物与汇总（步骤 4）
- RQ018/RQ019 重跑产物与新旧对照（步骤 5）
- 脚本放 `.codex-fleet/rq021-contemporaneous-envelope/work/E1/`

**不得删除或覆盖 RQ016C-H2、RQ017、RQ018、RQ019 的任何已落盘产物**，它们是被接受主张的证据链。

---

## 7. 硬边界

```
不改：src/sociality_estimation/core/{agent,ipv_estimation,reliability_logdomain}.py
      pipelines/interhub/process_interhub.py
      configs/ipv_sigma01_exact.json
不改 RQ009 / RQ016 / RQ016C / RQ017 / RQ018 / RQ019 已落盘的 run 目录（只读）
不改机制一的冻结门规格（log_score = -mse/(2σ²), σ=0.1；spread==0 用精确浮点相等，
   禁止 np.isclose；θ=0.20 是策略阈值，禁止调参、禁止阈值扫描）
不写 data/derived/
不做 git commit / 不碰 git 的任何写操作
禁止 git checkout -- . / restore . / stash / reset --hard / clean -fd / force push / rebase / amend
RQ007 held_out 不得被解析。必须先连接 K2 台账再按 fold 切——
   RQ009 的 fold 与 RQ007 的 split 正交，只按 fold 过滤会解析 1,899,898 行 held_out。
   报告须含实测断言：参与计算行中 rq007_split 不在 {development, guard} 的计数为 0。
RQ014 致盲相关的评分字段不得读取；遇到任何 rating/preference/score/human-score 字段，
   停下报告，不要读取内容
不得静默覆盖已冻结产物或已接受的 decision.md
不要对 reports/ 做全仓库 rg；用定向 ls/grep
git status 一律用 git --no-optional-locks status --porcelain
时间戳一律实取 date -u +%Y-%m-%dT%H:%M:%SZ
不投 Slurm/HPC；本机跑（RQ016C-H2 同规模作业约 12 分钟）
若出现任何密码提示，停下报告，不得输入、存储或打印
```

**措辞禁令**：禁用 `estimability` 与「测出/未测出 IPV」；禁用「过度消极」描述 IPV 下侧越界
（IPV 越负表示越对抗对方，不是越消极）。描述性结果不得写成因果主张。不用比喻、不用自造简称。

**分母纪律**：每个百分数必须紧跟分子、分母、筛选条件、来源文件与列名。

环境：本机 python3 已有 pyarrow 21.0.0 / pandas 2.3.3。缺依赖直接装上继续。

---

## 8. 报告要求

报告开头先定位：这项工作要解决什么问题、整体走到哪一步、本次是其中哪一环——
**写给完全没跟进过程的读者**。结论与待监督方拍板的事分开成节，后者要写清选项、
判断依据、以及不做的后果。

报告结尾必须带状态行：

```
state: WAITING_ON_COMMANDER
timestamp_utc: <实取>
```
