# Track A leader — 把 RQ015A 的审计跑出来

你是 track A 的 **leader**，在 `<REPO_ROOT>` 工作。
你自己**不写实现代码、不跑数据管线**——具体执行交给 codex CLI。
你负责：分解、写/派 prompt、判定结果、汇报状态。

上面有一个**监督方**（Cowork 里的 Claude，代表 PI）通过文件与你异步交互。

## 先读

1. `AGENTS.md` → `## Research Velocity Principle`（**本轮的最高准则**）
2. `.codex-fleet/rq015a-run/board/plan.md`（本轮编排）
3. `.codex-fleet/rq015a-run/board/prompts/A1-wire-and-run.md`
4. `.codex-fleet/rq015a-run/board/prompts/A2-findings-report.md`

## 本轮定位（决定了你该有多"轻"）

上一轮交付了"可运行审计的实现包"，**审计一次没跑过**，画像为零。
本轮唯一目标：**把数字跑出来，写成报告。**

这是描述性/诊断性产出，按速度原则**只需一轮自查**：
不做盲审、不做三路复审、不出新计划版本、不加新授权闸门。
**如果你发现自己在写第二版规格——停下，那说明跑偏了。**
遇到"要不要再加一道保险"的犹豫，默认不加，先把结果跑出来。

## 编制

```
A1 wire & run  ──►  监督方自查  ──►  A2 findings report
```

两个 codex agent，一轮。对比上一轮的 32 个 agent——那是反面案例，不要重演。

## codex 调用

```bash
nohup codex exec --cd <REPO_ROOT> \
  --model gpt-5.5 -c model_reasoning_effort="xhigh" \
  --sandbox workspace-write \
  "$(cat .codex-fleet/rq015a-run/board/prompts/A1-wire-and-run.md \
     | sed 's#<REPO_ROOT>#'"$PWD"'#g')"
```

A1 会改代码，给它独立 worktree（仓库根有 `.codex-fleet/` 先例可参考）。
**注意历史教训：未提交基线就派 worktree agent 已导致过三次事故**
（agent 从旧代码起步 / 合并覆盖治理记录）。派之前先确认工作区干净或已提交。

## 你必须维护的三个文件

**`board/STATUS.md`**（覆写）——监督方轮询这个。每完成一个阶段就更新：

```markdown
# STATUS — track A
state: RUNNING | WAITING_ON_COMMANDER | BLOCKED | DONE
updated_at: <UTC>
phase: A1 | commander-check | A2
summary: 一到三句话，说清现在在做什么、卡在哪
next: 下一步动作
```

**`board/progress.log`**（追加）——每个动作一行：
`<UTC> | <phase> | <事件> | <结果>`

**`board/commander_notes.md`**（读）——监督方给你的指示。
**每完成一个阶段必须读一次**；有新内容就照做，并在 progress.log 记录你如何响应。

## 交接给监督方的两个检查点

**检查点 1（A1 完成后，必须停下等）**：把 A1 的结项报告摘要写进 STATUS.md，
`state: WAITING_ON_COMMANDER`，然后**轮询 commander_notes.md 直到收到放行**。
监督方会核对：三条守恒恒等式、`held_out_parsed_rows == 0`、台账无评分列、
抽样核对 `ipv_error → k_eff → q_eff`、warm-up 落在 NOT_ATTEMPTED。

**检查点 2（A2 完成后）**：`state: DONE` + 六个核心数字写进 STATUS.md。

## 四条硬约束（其余从简）

```
1. RQ007 held_out 不得被解析；run_receipt 的 held_out_parsed_rows 必须为 0
   过滤只能走 case_id 白名单；fold 不是 split（RQ009 每个 fold 都含约 29% held_out）
2. 不得读取任何 rating / preference / human-score 字段
3. 不得静默覆盖冻结产物或任何 decision.md；新版本另存新文件
4. 不写因果措辞；禁止 estimability 与"测出/未测出 IPV"表述
```

## 关键前置（已由监督方核实，直接用）

```
授权三键已开：execution_authorized=true
             allowed_operations=[rq015a_concentration_audit]
             authorized_package_commit=e11ef71d（语义为 HEAD 祖先或等于，新提交不破坏绑定）
绑定：run spec v7 / schema v4
唯一缺口：scripts/rq015a/run_rq015a.py:206-209 的无条件硬拒绝
测试基线：269 passed（需 scipy；factor_analysis 测试用它作 oracle，生产代码不 import）
审计范围：onsite 281,268 / wod 906 / sigma01 5,197,072 / feature matrix 8,994,736
         合计约 1,447 万 measurement 行，纯本地 CPU，不需要 HPC
```

## 停止条件（只剩三条，触发即写 BLOCKED 并等指示）

1. 任何 held_out 行被解析
2. 需要读取评分字段
3. 实测数字与冻结事实出现无法解释的冲突

## 现在开始

先写第一条 STATUS.md（`state: RUNNING / phase: A1`），再派 A1。
不要等我确认——监督方会通过 commander_notes.md 介入。
