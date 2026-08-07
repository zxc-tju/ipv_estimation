# RQ021-E2 任务书：同期 envelope 的留一源迁移检验（leave-one-dataset-out）

你是本轮唯一的执行 agent。读完就开工，不写第二版方案，不开子轨。
仓库根即当前工作目录，以下路径都相对仓库根。

---

## 0. 位置与本轮由来（不要跳过）

最终目标是**在线验证**：判断一辆自动驾驶车表现出的社会交互倾向像不像人。
IPV（Interaction Preference Value）是表示交互倾向的标量。判定由两道串联弃权机制构成：
机制一判断这一帧的 IPV 数值能不能估（已冻结）；机制二拿通过机制一的数值与
**人类参照区间（envelope）**比。

RQ021-E1（2026-08-05，产物在 `.codex-fleet/rq021-contemporaneous-envelope/work/E1/`）
已按 PI 裁定把 envelope 的目标量换成锚点当下 `[t-9,t]` 的 `ipv_log`，
并已被接受、进入手稿。手稿 Figure 3 的 caption 现在写着一条边界：
「Transfer to a data source not seen during fitting is not established and is
reported as a boundary of the present monitor.」

**本轮就是去实测这条边界**：把参照池中四个数据源逐一留出，用其余三个源拟合完整
envelope，在留出源的 test 行上量覆盖率。旧 envelope（RQ009，预测型目标量）做过同样的
检验，留出覆盖率范围 0.748–0.992，四个折没有一个落在名义值 3 个百分点以内。
新（同期）envelope 从未做过。结果好就解除手稿边界，不好就维持现状——两种结果都可交付。

---

## 1. 事前判读标准（先于结果固定，不得事后调整）

主口径 α=90。**四个留出源的覆盖率若全部落在 [0.87, 0.93]（名义 0.90 ± 3 个百分点，
与旧 RQ009 检验同一把尺），判「迁移获得支持」；任何一个源落在带外，判「边界维持」。**

看到结果之后不得调整这条带、不得新增例外条款、不得改用其他 α 层重新判读。
80/95 层照常报告，但只作描述。本轮是描述性诊断，一轮做完出报告，不设盲审。

---

## 2. 数据与配方（全部沿用 E1，只加一个留出维度）

参照池构建与 E1 逐字节相同：

- K2 台账 `data/derived/rq015k_logdomain_gate/l1_v1`（取
  `product_row_key/status/rq007_split/ipv_log`）与 RQ009 矩阵
  `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix`
  精确连接，筛 `status == "OK"`、`agent_type_pair == "HV;HV"`、
  `rq007_split ∈ {development, guard}`。
- 目标列 `ipv_log`；22 项数值 + 4 项类别 context；支持门分格键
  `geometry_path_category + priority_role`、12 项距离特征；alpha 层 [80,90,95]；
  RQ009 fold 结构；split-conformal 计算方式。全部照抄 E1。

**留出维度**：`source_dataset` 列来自 RQ009 矩阵，取值恰为
`{waymo_train, nuplan_train, lyft_train_full, av2_motion_forecasting}`。
对每个源 S 各做一折：

- 条件分位数模型：只用 train fold 中 `source_dataset != S` 的行拟合
- **支持门也必须只用 train fold 中非 S 的行重拟**——沿用全源支持门会把 S 泄进
  它自己的迁移检验，这条做错整轮作废
- conformal 半径：只用 calibration fold 中非 S 的行
- 评估：test fold 中 `source_dataset == S` 的行

每折每 α 报告：留出源 test 行数、支持门通过数与弃权率、覆盖率（分子/分母）、
平均与中位区间宽度。

**次要对照表（便宜，必做）**：加载 E1 的持久化模型
`.codex-fleet/rq021-contemporaneous-envelope/work/E1/envelope_model/rq016c_h2_envelope.pkl`
（只加载、不重拟合），对全部 486,660 test 行打分一次，按 `source_dataset` 分组报告
**全源拟合下的逐源覆盖率**。这张表用来区分「S 本来就难覆盖」和「S 只在未见时难覆盖」。

## 3. 已知不变量（对不上就停下报告，不要凑数）

留出过滤之前的完整参照池必须与 E1 精确一致：

```
参照池 2,442,625 行 = development 1,752,509 + guard 690,116，held_out 0
  train 974,984 / calibration 481,088 / guard_tune 499,893 / test 486,660
无留出时机制二弃权 24,723/486,660（如做全源自检）
```

纯人-人各源行数（**status 过滤前**，来自矩阵 `fold × agent_type_pair × source_dataset`，
监督方 2026-08-07 实测，用作量级核对；你要报告 status=="OK" 过滤后的精确值）：

```
fold:                    calibration  guard_tune    test   train
av2_motion_forecasting        45,038      45,778   45,808   91,460
lyft_train_full              179,690     199,630  181,368  364,028
nuplan_train                 305,132     300,504  301,572  589,754
waymo_train                  412,228     432,622  415,988  827,098
```

## 4. 复用与产物

复用 E1 的脚本（`.codex-fleet/rq021-contemporaneous-envelope/work/E1/run_rq016c_h2_human_only_envelope.py`），
复制到本轮 work 目录后加留出参数，改最小必要处，不要从零重写。
E1 单次全量拟合约 12 分钟；本轮 4 折约 50 分钟，本机跑，不投 Slurm/HPC。

- 脚本与中间产物：`.codex-fleet/rq021-contemporaneous-envelope/work/E2/`
- 机器数字：`.codex-fleet/rq021-contemporaneous-envelope/work/E2/key_numbers_e2.json`
  （含每折每 α 的全部计数、事前判读带、判读结果、次要对照表）
- 报告：`.codex-fleet/rq021-contemporaneous-envelope/board/reports/RQ021_2_lodo_transfer.md`

报告开头先定位（这项工作解决什么问题、整体走到哪、本次是哪一环），写给完全没跟进
过程的读者。每个百分数紧跟分子、分母、筛选条件、来源文件与列名。
判「迁移获得支持」或「边界维持」必须单独成节，并列出四个源的实测值与判读带的关系。

## 5. 硬边界

```
不改：src/sociality_estimation/core/{agent,ipv_estimation,reliability_logdomain}.py
      pipelines/interhub/process_interhub.py
      configs/ipv_sigma01_exact.json
不改 E1 的产物与已落盘的 reports/studies/ 运行目录（只读）
不写 data/derived/
不做 git commit / 不碰 git 的任何写操作
禁止 git checkout -- . / restore . / stash / reset --hard / clean -fd
RQ007 held_out 不得被解析。必须先连接 K2 台账再按 fold 切——RQ009 的 fold 与 RQ007
   的 split 正交，只按 fold 过滤会解析约 29% 的 held_out 行。
   报告须含实测断言：参与计算行中 rq007_split 不在 {development, guard} 的计数为 0。
RQ014 致盲相关的评分字段不得读取
不要对 reports/ 做全仓库 rg；用定向 ls/grep
git status 一律用 git --no-optional-locks status --porcelain
时间戳一律实取 date -u +%Y-%m-%dT%H:%M:%SZ
```

**措辞禁令**：禁用 `estimability` 与「测出/未测出 IPV」。描述性结果不得写成因果主张。
不得把覆盖率不达标写成「失败/输给」，写实测值与判读带的关系即可。

环境：本机 python3 已有 pyarrow 21.0.0 / pandas 2.3.3。缺依赖直接装上继续。

## 6. 报告结尾必须带状态行

```
state: WAITING_ON_COMMANDER
timestamp_utc: <实取>
```
