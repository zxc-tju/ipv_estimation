# RQ015A 执行 fleet — 把审计真正跑出来

Fleet dir：`./.codex-fleet/rq015a-run`
指挥者：Claude（PI 角色代理）｜执行：codex CLI（`gpt-5.5`, `xhigh`）
适用理念：`AGENTS.md` → **Research Velocity Principle（PI 裁定 2026-07-31）**

## 本轮的定位

上一轮交付的是"可运行审计的实现包"，**审计一次没跑过**，画像为零。
本轮只有一个目标：**把数字跑出来，写成报告。**

**这是描述性 / 诊断性产出，按速度原则只需一轮自查。**
不做盲审、不做三路复审、不出新的计划版本、不加新的授权闸门。
如果你发现自己在写第二版规格——停下，那说明跑偏了。

## 编制：两个 agent，一轮，中间我查一次

```
A1 wire & run  ──►  指挥者自查（守恒 / held_out / 评分列 / 抽样看行）  ──►  A2 report
```

## 现状（已核实）

- 授权已开：`execution_authorized=true`、`allowed_operations=[rq015a_concentration_audit]`、
  `authorized_package_commit=e11ef71d`（语义为"HEAD 祖先或等于"，**新提交不会破坏绑定**）
- 组件齐备：`build_ledger.py`（含 `open_measurement_reader` / `build_l1_for_artifact` /
  `sort_l1_rows` / `assert_l1_conservation`）、`factor_analysis.py`、`receipt.py`、
  `validate_only.py`、`rq015a_contracts.py`
- **唯一缺口**：`scripts/rq015a/run_rq015a.py:206-209` 在许可签发成功后无条件抛出
  `"execution permit unexpectedly succeeded in BUILD_WHILE_DENY"`——
  没有任何代码把各组件接成一次真实审计
- 测试 269 passed；validate-only exit 0 / machine_verdict=PASS
- 绑定 run spec v7 / schema v4

## 审计范围（4 个 ledger-bearing 产物）

| artifact | 规模 | measurement 行 |
|---|---|---|
| `onsite_dense_timeseries` | 70,317 物理行，csv | 281,268 |
| `wod_rq010b_full479_audited` | 906 行，4 列 | 906 |
| `interhub_sigma01_hw4_timeseries` | 2.2 GB csv，dev+guard 2,598,536 | 5,197,072 |
| `rq009_feature_matrix` | 138 parquet parts，dev+guard 4,497,368 | 8,994,736 |

合计约 1,447 万 measurement 行。纯本地 CPU，不需要 HPC。

不在范围：`wod_phase1_phase1b_10hz_schemeb`、`rq014_g2r_anchor_scores`
（PI 2026-07-31 裁定不取回，探测已确认其表头无 error 列）。

## 只有四条硬约束（其余按速度原则从简）

```
1. RQ007 held_out 集不得被解析；receipt 的 held_out_parsed_rows 必须为 0
2. 不得读取任何 rating / preference / human-score 字段
3. 不得静默覆盖已冻结产物或任何 decision.md
4. 描述性结果不得写成因果主张
```

这四条是效度边界，不是流程。**除此之外遇到"要不要再加一道保险"，默认不加。**

## A1 — wire & run（implementer，`--worktree`）

见 `prompts/A1-wire-and-run.md`。要点：

- 删掉 `run_rq015a.py:206-209` 的硬拒绝，把 `build_ledger → 聚合 → factor_analysis →
  receipt → 报告骨架` 接成 `_run_execute` 的真实路径
- **先在 `onsite_dense_timeseries`（最小，70,317 行）端到端打通**，再跑其余三个
  —— 理由：这条路径从未被执行过，从未执行过的路径会藏东西
- 改了 `run_rq015a.py` 会使校验清单失配 → 必须**重签清单**并重跑 validate-only
- 产出按 run spec v7：`concentration_ledger*`、`portraits.json`、`c0_routing.json`、
  `run_receipt.json`；`bounded_report.md` 由 A2 写
- **允许的偏离**：台账逐行数据用 parquet 落盘（1,447 万行 CSV 不合理），
  另出人读的 `concentration_ledger_summary.csv`；在 receipt 里记明该偏离

## 指挥者自查（我做，一轮）

1. 三条守恒恒等式逐产物成立，且实测行数与冻结事实吻合
   （sigma01 measurement 5,197,072 / NOT_ATTEMPTED 215,088；feature matrix 8,994,736；
   onsite 281,268；wod 906）
2. `held_out_parsed_rows == 0`
3. 台账列名无 rating/preference 命中
4. 随机抽若干行人工看一眼 `ipv_error → k_eff → q_eff → attempt_status` 的推导对不对
5. warm-up（`error=1.0`）落在 `NOT_ATTEMPTED` 而非 `UNKNOWN`

通过即放 A2；不通过就把具体问题打回 A1，不启动新一轮审计编排。

## A2 — findings report（experimenter）

见 `prompts/A2-findings-report.md`。这一步产出的才是**科学交付物**：

1. 逐产物、逐层的连续 `q_eff` 分布（这是主量）
2. 分层：按数据源 / 配置（hw4 vs hw10）/ 视角 / case / episode
3. **可用子集清单**——哪些 case / episode 的 IPV 携带判别信息（最有实操价值的一项）
4. 协变因素：Spearman + case-cluster bootstrap CI，**只描述不下因果**
5. C0 路由：逐下游消费者（M3 训练标签 / RQ009 包络 / RQ014 筛选）给出四态之一
6. 覆盖披露：4/6 产物，WOD 一支为**部分覆盖**，禁止表述为"全语料"

## 完成定义

`bounded_report.md` 存在、数字经指挥者自查、`STUDIES.md` 与 `START_HERE.md` 同步、
`main_workflow.log` 记一条。**到此结束，不再加轮次。**

## 停止条件（只剩三条）

1. 任何 held_out 行被解析
2. 需要读取评分字段
3. 实测数字与冻结事实出现无法解释的冲突（说明前序核验有误，找我）
