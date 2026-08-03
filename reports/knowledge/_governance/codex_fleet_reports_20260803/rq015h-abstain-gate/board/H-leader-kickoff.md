# Track H leader — 定义弃权门（可估计性契约），并用跨环境塌缩测试验收它

仓库根：`<REPO_ROOT>`
你的看板：`<REPO_ROOT>/.codex-fleet/rq015h-abstain-gate/board/`
你的工作区：`<REPO_ROOT>/.codex-fleet/rq015h-abstain-gate/work/`

## 为什么有这一轮

PI 已经作出研究设计决策：论文**同时**主张群体率与个体 IPV，但个体量**只在通过弃权门的
子集上报告，且强制带不确定性区间**。这条决策把「弃权门」推上了关键路径——
没有它，个体层面的任何一句话都写不了。

背景事实（均已由监督方从原始数据独立复算，可直接引用，不必重验）：

- 缺陷根源：权重来自密度**连乘**，`agent.py:1136-1139` 在 `sum(var)==0` 时回退均匀权重；
  网格对称 ⇒ 均匀权重的加权平均**恰好** 0.0、误差**恰好** `1-1/sqrt(7) = 0.6220355269907728`。
  于是「算不出来」与「完全自利」不可区分。
- PI 的科学论点（已被接受为前提）：**IPV 可估计性本来就依赖交互的存在性。**
  无实质交互的帧估不出来是**正常且合理**的，不同数据源比例不同也正常。
  因此弃权门的目的**不是**把失败降到零，而是**把「不可估」与「估出来是 0」分开记账**。
- G 轨（HPC 冻结环境重解 2,300 锚点）已确立：
  - 群体率跨环境稳定（机制拆分最大偏移 1.73pp）
  - **单锚点点估计跨环境不稳定**：全样本 `|Δipv_log|` 中位数 `2.776e-17`、
    p90 `0.0574`、p99 `0.6166`、最大 `1.8375`；网格全幅仅 `2.3562` rad（±3π/8）
  - 差异归因于**软件栈**而非 CPU 微架构（AMD 与 Intel 逐位一致）

## 本轮的核心洞察，先读懂再动手

**下溢是连乘域的数值 bug，log 域已经消除了它。所以弃权门不是用来检测下溢的。**

弃权门要处理的是**实质性不可辨识**：MSE 曲面是真的平（⇒ 无交互 ⇒ 合理地不可估），
还是有峰（⇒ 可估）。这两件事必须分开。由此得到本轮的头号问题：

> **切到 log 域之后，原本被判为 D1（全下溢）的 43.17% 里，有多少变成了可估计的？**

如果 D1 主要是数值 bug，它应当大部分被 log 域「救回来」；
如果 D1 救不回来，说明 D1 与 D2 本来就是同一现象——曲面本来就平，
七个候选的似然一起小到下溢，下溢只是症状不是病因。

**这个「救回率」是 H 轨能产出的最有价值的单个数字**，因为它直接量化了 log 域重写
到底买到了什么。先回答它，再谈门。

## 门的设计要求

1. **必须建立在 log 域量上。** 任何建立在连乘域量（`legacy_var`、`legacy_prod_sum`、
   `legacy_fallback_triggered`）之上的门，都继承了它试图检测的病理。
   现成实现：`src/sociality_estimation/core/reliability_logdomain.py`（RQ015B 新增）。
2. **首选统计量：log 域 `k_eff`。** `q_eff = 1/((1-e)^2 * K)`、`k_eff = q_eff * K`，
   范围严格 `[1, 7]`。`k_eff = 1` 表示质量全在一个候选（判别力最强）；
   `k_eff = 7` 表示均匀（无判别力）。门的形式即 `k_eff <= tau`。
3. **单一连续统计量优于清单式规则。** 如果你认为必须加第二个条件（例如 `n_obs` 下限），
   要给出它不能被 `k_eff` 吸收的证据，不要因为「看起来更稳妥」就叠加条件。
4. **阈值 tau 不得靠肉眼选。** 用下面的验收测试反解：**取能通过验收的最宽松 tau**
   （即在满足验收判据的前提下让通过率最大），并报出 tau 对通过率的敏感性曲线。

## 验收测试（决定性，数据已在盘上，不需要任何 HPC 作业）

我们手里有同一批 2,300 锚点的两个环境版本：

```
Mac : <REPO_ROOT>/.codex-fleet/rq015b-repair/work/anchor_mse.csv
HPC : <REPO_ROOT>/.codex-fleet/rq015g-hpc-resolve/work/anchor_mse_hpc.csv
```
两份 2,300 行、36 列同名同序，`anchor_id` 全集相同。含 `w_log[7]`、`ipv_log`、
`ipv_error_log`、`mse_per_candidate[7]`、`n_obs`、`source`、`signature`、`n_band`。

**测试：把跨环境 `|Δipv_log|` 的分布限制在「通过门」的锚点上重算。**

- 报 **通过者** 与 **未通过者** 各自的 p50 / p90 / p99 / max，**并按 source 分列
  （waymo / nuplan 必须分开，不得只报合并值）**
- **通过判据（我设定，不得放宽）：通过者的 p99 `|Δipv_log|` 必须小于一个网格间距
  `pi/8 = 0.3926991`。** 理由：跨环境差异超过一个网格步长，意味着 argmin 本身在跳，
  此时报告出来的那个数值没有意义
- 若最宽松的 tau 都过不了这条判据 ⇒ **`k_eff` 门无效，如实写下来并说明原因**，
  不要为了让它通过而修改判据。这是一个可以否决自身设计的测试，这正是它的价值

## 还必须交付的两件

**（1）选择偏倚刻画。** 带门意味着个体层面的样本量会掉，而且掉得不随机。
把**通过者 vs 未通过者**的协变量分布摊开对比，至少含：
`source`、`n_obs`（给分布不是只给均值）、`signature`、`n_band`，
以及若样本中可得的「是否存在交互机会」指示。
**通过率必须按 source 分别报告。** 预期 waymo 与 nuplan 差距很大（waymo 的 D1 率 58.93%，
nuplan 仅 1.10%），若通过率不差很多，反而说明门有问题，要解释。

**（2）区间宽度的下界校准。** 论文要求个体量强制带区间。
用现成的跨环境差异给区间做一次外部校准：
**在通过者上，你构造的区间半宽必须 >= 观测到的 p99 `|Δipv_log|`。**
若你提出的区间比它窄，那个区间就是在说谎，须如实指出并说明差多少。
区间构造方式你自己定（log 域权重是网格上的正规化后验，可直接取可信区间；
或对帧做 bootstrap），但**必须说明选了哪种、为什么**。

## 明确不做的事（越界即为跑偏）

- **不重解任何锚点**，不提交任何 HPC 作业。本轮是纯分析，输入已全部在盘上
- **不跑全语料**（14M 行）。全语料是 I 轨的事，两轨并行，不要抢
- 不改 `agent.py` / `ipv_estimation.py` / `process_interhub.py` / `reliability_logdomain.py`
- 不写「规格 v2」、不做盲审、不做多路复审
- 不提交 git commit

## 产出

`board/reports/H1_abstain_gate.md`，须含：

1. **救回率**：切到 log 域后，原 D1 行中变为可估的比例（分 source）
2. **门的定义**：统计量、阈值 tau、tau 的敏感性曲线、以及为什么不需要第二个条件
3. **验收测试结果**：通过者/未通过者的 `|Δipv_log|` 四分位数，分 source，
   与 `pi/8 = 0.3926991` 判据的对照，明确写「通过 / 不通过」
4. **通过率**：合并与分 source
5. **选择偏倚表**：通过者 vs 未通过者的协变量分布
6. **区间**：构造方式、半宽、与 p99 `|Δipv_log|` 的对照
7. 每个数字须可复算：写出用了哪个文件的哪一列、什么筛选条件

## 编制

**一个 codex agent（H1），一轮 leader 自查，出报告，结束。** 不要开第二个 agent。
派发用 `.codex-fleet/rq015a-run/board/detach_launch.py`（见通用规则第 3 条）。
本轮无外部依赖、无排队，预计单个 agent 一小时内可完成——如果超过两小时还没出数字，
说明 prompt 给得不够具体，回来收紧范围而不是加人。

## 舰队通用规则（A/B 轮踩出来的，逐条都是事故换来的）

**你的角色**：leader。你自己**不写实现代码、不跑数据管线**——交给 codex CLI。
你负责：分解、写/派 prompt、判定结果、汇报。上面有监督方（Cowork 的 Claude）通过文件与你异步交互。

**1. 速度原则是最高准则**（见 `AGENTS.md` → Research Velocity Principle）。
本轮是诊断性/描述性产出：**一个 codex agent，一轮自查，出报告，结束**。
不做盲审、不做多路复审、不出第二版规格、不加授权闸门。
**发现自己在写规格 v2 就是跑偏了，停下。**
反面案例：上一轮一个描述性审计走了 8 个计划版本、7 轮盲审、32 个 agent，科学结论产出为零。

**2. `claude -p` 是单回合的——派完 codex 不要结束回合。**
A/B 两轨的 leader 都栽在这：派出 codex、写完汇报、进程即退出，子进程随之被杀，
`STATUS.md` 还停在 `RUNNING`，**没有人在收结果**。
正确做法：派完在**本回合内**阻塞轮询（`sleep 60` 看日志增长，每 5 分钟写一行 progress.log），
等 codex 真正结项 → 做那一轮自查 → 写 `WAITING_ON_COMMANDER` → 才结束回合。

**3. 派发必须脱离进程组，macOS 没有 `setsid`。**
用现成的：`.codex-fleet/rq015a-run/board/detach_launch.py`（双 fork + `os.setsid`）
```bash
python3 .codex-fleet/rq015a-run/board/detach_launch.py \
  --log <你的 board>/reports/<AGENT>.log --pidfile <你的 board>/<AGENT>.pid \
  -- codex exec --cd "$PWD" --model gpt-5.5 -c model_reasoning_effort="xhigh" \
     --sandbox workspace-write "$(cat <你的 prompt 文件>)"
```
派完立刻自检：`ps -o pid,ppid,pgid -p <新pid>` → **PPID 必须是 1**，PGID ≠ 你的 PGID。
⚠ `codex exec` **没有** `--ask-for-approval` 参数（0.144.1 起），带上直接报错退出。

**4. 两条 track（H/I）并发在同一个工作区。铁律，不可协商：**
```
禁止 git checkout -- . / git restore . / git stash / git reset --hard / git clean -fd
禁止 git checkout 任何历史提交到主工作区（要看旧代码用 git worktree add）
禁止 git commit（本轮产物由 PI 统一提交）
工作区非空是【预期状态】——另一条 track 的 agent 正在同一仓库工作。
你只对自己创建/修改的文件负责；清洁性检查只查自己的文件清单，不看全仓库 git status。
```

**5. 四条硬约束（与流程无关，不得放松）**
```
1. RQ007 held_out 不得被解析（污染不可恢复）
2. RQ014 致盲相关的评分字段不得读取
3. 不得静默覆盖冻结产物或已接受的 decision.md
4. 描述性结果不得写成因果主张
```
另：全文禁用 `estimability` 与"测出/未测出 IPV"。
可辩护的表述是：**权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息**。

**6. 杂项，都是踩过的**
- 解释器钉死 `<local-rq009-venv>/bin/python`（系统 python3 缺 pytest，会把基线判错）
- 时间戳一律 `date -u +%Y-%m-%dT%H:%M:%SZ`，**不要前瞻估计**（上一轮 progress.log 里出现过比墙钟早 12 分钟的行）
- 不要对 `reports/` 做全仓库 `rg`——宽泛检索会把 RQ003 `12_blind_annotation/controlled_identity_map.csv`
  的 controlled-access 行整行拉进上下文
- 给 codex 的 prompt 里**直接列出要读的文件路径**并限定检索范围；
  A1 第一次跑把 31 次 exec 全花在摸索仓库上，一行计算都没跑就被杀了
- `launch_leader.sh` 会**覆写 STATUS.md**；真正要留存的交接信息写进 `commander_notes.md`（追加式，不会丢）

**7. 你必须维护的三个文件**
- `board/STATUS.md`（覆写）：`state: RUNNING|WAITING_ON_COMMANDER|BLOCKED|DONE` / `updated_at` / `phase` / `summary` / `next`
- `board/progress.log`（追加）：`<UTC> | <阶段> | 做了什么 | 结论`
- `board/commander_notes.md`：监督方写给你的，**每完成一个阶段读一次**

结项后写 `state: WAITING_ON_COMMANDER` 并轮询 `commander_notes.md` 等放行，**不要自行转 DONE**。
