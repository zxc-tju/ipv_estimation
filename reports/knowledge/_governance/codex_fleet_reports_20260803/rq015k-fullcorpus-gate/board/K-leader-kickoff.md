# Track K leader — 全语料重算：物化 log 域门判据台账


> ## ⚠ 本任务书已按双盲独立复审修订（2026-08-02T08:30Z）
>
> 两位互不可见的独立审查方（KR-A 从计划书出发、KR-B 从数据反推）**判定一致：需修改后执行**。
> 两方都独立确认了本轮的前提成立（现有 4 份 ledger 无逐候选量，`k_eff` 由旧 `ipv_error` 派生
> 不可替代），并各自列出了查过而没有的全部路径。监督方已复核其关键数字。
>
> **下列 11 条为强制条款，优先级高于本文其余部分。与正文冲突时以本节为准。**
>
> **1【最重要】门适用性契约。** 14,473,982 行中只有 **13,980,600 行是 `ATTEMPTED`**；
> 另有 **219,360 行 `NOT_ATTEMPTED`**、**274,022 行 `UNKNOWN`**（三者相加恰等于总数，监督方已验）。
> OnSite 的 274,022 行 UNKNOWN 是 `ipv_error` 为 NULL 的空单元、`reason_code=EMPTY_CELL_UNEXPLAINED`
> （`bounded_report.md` 第 8、28 行）。
> **只对 `attempt_status == ATTEMPTED` 的行计算本门。** 其余行必须写
> `gate_applicable = false`、门字段全部 null，并保留 `source_attempt_status`。
> **绝不允许把"上游没有有效输入"写成 `NO_IPV_EFFECT` 或 `NEAR_UNIFORM`** ——
> 那会把工程缺失混进科学弃权原因，直接污染 RQ009 的弃权分布与 context cell 通过率。
>
> **2 工程失败必须有独立状态。** 现规格只有 `OK`/`ABSTAIN`，不足。
> `reliability_logdomain.py` 第 36-49 行已有 `NON_FINITE_INPUT`、`SOLVER_FAILURE` 等互斥状态，复用它们。
> 这类行 `ipv_log = null`、`w_log[7]` 全 null，**不得归入两个科学 reason**。
>
> **3 σ 必须在批处理规格里钉死。** `log_score_i = -mse_i / (2 * sigma^2)`，**`sigma = 0.1`**
> （B1 代码 `SIGMA=0.1`、七点网格 `legacy7_pi_over_8`）。
> 若不同 artifact 误用不同 σ，`max_w_log` 与 `k_eff_log` 会整体偏移、门判定整体失真。
> 输出必须持久化 `log_score[7]`，或记录 σ 与公式，使读者能复算 `w_log[7]`。
>
> **4 `mse_spread == 0` 是精确浮点相等，不是容差。** 先断言 7 个 MSE 均为有限 float64、
> 长度为 7，再判 `max - min == 0.0`。**不得改成 `np.isclose`** ——
> 这是精确退化标签（Mac/HPC 两环境对这 400 行 MSE 逐位相同已证），不是阈值。
> 非有限的行走第 2 条的状态，不进这两个科学 reason。
>
> **5 softmax 复用现成实现。** `reliability_logdomain.py` 第 172-188 行 `weights_from_mse()`
> 已是稳定实现（减最大值后 exp 再归一，并检查分母有限）。不要另写。
>
> **6 互斥 reason 必须有序赋值。** 样本内 400 个 `mse_spread==0` 的行**全部**也满足
> `max_w_log < 0.20`。向量化实现必须用有序 `np.select`，或先写 `NO_IPV_EFFECT`、
> 再只在 `reason is null` 的行写 `NEAR_UNIFORM`。**两个布尔掩码顺序覆盖会写反。**
>
> **7 求解单元必须收敛到规范键，派生行不得重复求解。** RQ015A schema 写明
> `rq009_feature_matrix`（8,994,736 行）是 `interhub_sigma01_hw4_timeseries`（5,197,072 行）的
> **派生**产物，且禁止跨 artifact 池化。K1 必须把求解单元定义到
> `(artifact/source case, frame, agent/role, grid)` 一层，算完 join 回派生行。
> **把 14,473,982 直接当求解量是错的。**
>
> **8 pilot 必须分层。** 原文只说"小批 2,000–5,000 单元"，不够。
> 必须按 **artifact、source、measurement_role、轨迹长度、PKL 分组** 分层抽样，
> 并覆盖 OnSite/WOD。每层分别报墙钟、worker 峰值 RSS、PKL 复用情况。
> **必须给出 worker 数、PKL 分片、RSS 峰值、Slurm `--mem` 的硬指标，不能只报平均秒/单元。**
> 这是两位审查方共同点名的"最可能造成实际损失的一处"。
>
> **9 分片、断点续算、失败重投、产物校验要从原则变成验收项。**
> 分片：固定输入范围、PKL 清单、行键范围、行数、预期输出行数。
> 续算：先写临时文件、校验后原子 rename；已完成由 manifest 的
> 输入 SHA + 代码 SHA + 命令 + 行数 + 输出 SHA 判定，**不能只看文件存不存在**。
> 重投：失败至少分 `OOM / TIMEOUT / SOLVER_FAILURE / NON_FINITE_INPUT / SCHEMA_MISMATCH`；
> 超阈值必须停下上报，**不得扩大资源硬跑**。
> 校验：主键唯一、无缺失或重复分片、`K=7`、数组长度 7、有限性规则、reason 互斥顺序、
> `ipv_log` null 规则、状态计数、抽样重跑一致、输入清单 SHA、`held_out_parsed_rows = 0`。
>
> **10 增加 RQ009 join 干跑。** 抽样解析 `rq009_feature_matrix` 的 `product_row_key`，
> 证明 `context_cell_key`（或其生成输入）能一对一取到。
> **若取不到，K2 的产物接不上下游，这是 go/no-go 条件之一。**
>
> **11 两处措辞更正。**
> (a) **"全语料"必须限定为「RQ015A 当前 4 份本地可审计 L1 parquet ledger 的全行」**，
> 不得写成"全 WOD"或"全项目全语料"——本轮覆盖 4/6 产物，WOD 只含 full479 的 906 行。
> (b) 正文第八节写"只增加持久化输出列、不改任何计算"**容易被误读为没有实现工作**。
> 事实是：`process_interhub.py` 第 1168-1175 行未启用 diagnostics、第 854-885 行只写 legacy IPV/error；
> `ipv_estimation.py` 第 340-367 行的 diagnostics 也不含 MSE/log-score。
> **K2 需要新增 materializer 或 writer。四个受保护文件仍然不许改，但不要假设零实现工作。**
>
> **另**：正文第二节第 5 条（查清 RQ009 三个 IPV 输入列的填充规则）**降级为 K1 报告附录，
> 不作为 K2 投产的阻断条件**。两位审查方一致认为它是下游解释问题。
> KR-A 已查明规则并可直接引用：`build_features.py` 第 774-776 行直接沿用上游 legacy IPV/error 数值，
> slope 由 `theil_sen_slope()` 计算、有效历史点少于 2 个时返回 NaN（第 583-597 行），
> 之后被 calibration 的中位数填补吸收（`Preprocessor` 第 211-235 行、gate 第 704-715 行）。
> **K1 只需照此说明，不要量化污染规模，不要重算 RQ009 的 4.78% 弃权率。**


仓库根：`<REPO_ROOT>`
看板：`<REPO_ROOT>/.codex-fleet/rq015k-fullcorpus-gate/board/`
工作区：`<REPO_ROOT>/.codex-fleet/rq015k-fullcorpus-gate/work/`

## 一、这一轮在整个研究里的位置

最终用途是 online verification：判断自动驾驶车辆当前的 IPV 是否符合人类的分布。
**PI 已裁定：两个弃权机制串联执行。**

```
机制一（RQ015，本轮要物化的）：这一帧的 IPV 能不能估
    ├─ ABSTAIN → 直接结束，不进机制二
    └─ OK      → 进入机制二
机制二（RQ009 已 accepted）：当前场景的人类数据够不够判定 AV 是否偏离
```

机制一的规格已由 track J 定稿并经监督方复算，见
`.codex-fleet/rq015j-gate-spec/board/reports/J1_gate_spec_and_impact.md` §1.2 与 §1.3。
本轮**不改这个规格**，只是把它落到全语料上。

**为什么必须重算**：现有台账（4 份 parquet，合计 `14,473,982` 行）**没有**
`mse_per_candidate[7]` 与 `w_log[7]`；其中已有的 `k_eff` 由连乘域的 `ipv_error` 派生
（`reports/plans/RQ015A_ledger_schema_v4_20260731.json`：`k_eff = 1.0 / (1.0 - ipv_error) ** 2`），
**不能替代 log 域权重**。所以机制一无法在现有产物上逐行判定。

## 二、本轮分两段，K1 结束后**必须停下报监督方**，不得直接进 K2

这是一个千万行级的 HPC 作业。**不允许基于估算直接投全量。**

### K1（勘察与资源规划，不跑全量）—— 本轮先只做这一段

1. **确定计算单元与真实数量。** 14,473,982 是台账行数，不等于需要做 7 候选求解的单元数。
   查清一个「需判门的单元」到底对应什么（case × measurement_role × frame？），
   给出确切数量与分 artifact 的拆分。
2. **【优先做，可能省掉整轮】找有没有现成产物已经保存了逐候选量。**
   若某个中间产物已存 `mse_per_candidate[7]`、逐候选似然或 log 分数，
   则本轮只需重算 softmax，成本从"重跑估计器"降到"扫一遍现有文件"。
   **这一条若成立，是本轮最大的节省，务必先查。** 查过的路径与结论都要列出来，
   包括查了但没有的，便于监督方复核你没漏。
3. **实测单位成本与内存。** 在 HPC 冻结环境里跑一个小批（建议 2,000–5,000 单元），
   测出每单元耗时、每 worker 常驻内存、以及 PKL 载入的内存放大倍数。
   **G 轨有三个不同的时间口径，外推时三个都要报，不得只用其中一个**（监督方已逐一核过来源）：
   - 求解循环 **500.6s**（`.codex-fleet/rq015g-hpc-resolve/board/progress.log` 第 28 行
     `completed=2300/2300 elapsed=500.6s`）
   - driver 完整耗时 **702.9936774782836s**（`work/g1_hpc_summary.json` 的 `t5.elapsed_seconds`；
     同处 `seconds_per_anchor_wall=0.3056494249905581`）
   - Slurm 作业墙钟 **00:14:22 = 862s**（`G_leader_adjudication.md` 第 5 行）
   单节点 6 worker 外推到 13,980,600 个 ATTEMPTED 行，三口径分别约 **35 / 49 / 61 天**，不含排队与重投。
   24 worker 曾因每进程各载一份 PKL 而 OOM（TRES `mem=160992M` 仍不够）。
4. **给出资源方案**：分区、节点数、每节点 worker 数、每作业单元数、预计墙钟、
   以及失败重投与断点续算的设计（千万行级作业不允许"失败就从头再来"）。
5. **查清一件与机制二耦合的事**（PI 已知悉，属顺手查清，不要扩大）：
   RQ009 的分布外判据用了 `counterpart_ipv_current`、`counterpart_ipv_error_current`、
   `counterpart_ipv_slope_pre_anchor` 三列
   （`reports/studies/RQ009_.../02_process/04_calibration/calibration.py` 第 153–155 行）。
   **当对手车的 IPV 不可估时，这三列被填成什么？**
   如果填的是污染值（`ipv=0`、`ipv_error=0.6220355269907728`），
   那么大量"测不出对手车 IPV"的帧会在特征空间里塌到同一点。
   只需查清填充规则并如实记录，**不要去改 RQ009，也不要重算它的 4.78% 弃权率**。

**K1 完成后写 `WAITING_ON_COMMANDER` 停下。K2 需监督方另行放行。**

### K2（执行，本轮尚未授权）

按 K1 的资源方案重算并物化台账。**未获放行不得开始。**

## 三、门的规格（照抄，不得改动、不得"优化"）

```text
输入（单帧内可得）: frame_id, candidate_grid_id, K=7, candidate_ipv[7],
                    mse_per_candidate[7], log_score[7], context_cell_key

w_log      = softmax(log_score)  用 log-sum-exp
mse_spread = max(mse_per_candidate) - min(mse_per_candidate)
max_w_log  = max(w_log)
k_eff_log  = 1 / sum(w_log_i^2)

if mse_spread == 0:      status=ABSTAIN, reason_code=NO_IPV_EFFECT,  ipv_log=null
elif max_w_log < 0.20:   status=ABSTAIN, reason_code=NEAR_UNIFORM,   ipv_log=null
else:                    status=OK,      reason_code=null,
                         ipv_log = sum(candidate_ipv_i * w_log_i)
```

- **必须在 log 域算。** 连乘域下溢会把可估的行错判为不可估
- `theta = 0.20` 是政策阈值，**不是数据自然断点**，不得据此调参
- 两条判据**互斥且有序**：先 `NO_IPV_EFFECT`，否则再 `NEAR_UNIFORM`。
  样本上 400 个 `mse_spread==0` 的行其 `max(w_log)` 全部恰为 `1/7`，
  判据 1 的额外筛出量是 0；它保留为语义标签，**不得与判据 2 相加当作两份贡献**
- 弃权时 `ipv_log` 必须为 **null**，不得为 0、NaN 或缺列

## 四、必须随台账一起交付的一条下游警告

track J 已测出：**门后（`status=OK`）的行里有 23.40% 的 `ipv_log` 恰好为零**
（`|ipv_log| <= 1e-9`；分 signature 为 N 12/363、U 91/511、Z 135/143；
占门后 HT 权重 10.2788%）。

这是本门要消除的那处混淆的**镜像情形**：本门保证「弃权」不再被写成 `ipv=0`，
但反过来**不成立**——`ipv_log = 0` 仍是合法且高频的**通过门**的估计值（中性社会倾向）。
**判别字段只能是 `status` 与 `reason_code`，不是 `ipv_log` 的数值。**
这条必须写在交付给 RQ009 的接口说明里，不能只留在报告正文。

旁证：RQ009 自己的 `decision.md` Boundaries 一节写着
`Target exact-zero atom ~21.6%`，与本轮的 23.40% 量级一致，两者来源独立。

## 五、台账要物化的列

至少包含：`mse_per_candidate[7]`、`w_log[7]`、`max_w_log`、`mse_spread`、`k_eff_log`、
`status`、`reason_code`、`ipv_log`（弃权为 null）、`candidate_grid_id`、`K`、
帧/行键，以及与 RQ009 对接所需的 `context_cell_key` 与门通过率聚合字段。
完整契约见 J1 报告 §1.3，照它执行。

## 六、可对照的既有结果（K2 完成后用来验收，K1 阶段不必算）

- 设计基估计：全域可估率 **71.2695%**，CI `[67.1729%, 75.2135%]`
  （HT 分母 2,646,058；保留权重 1,885,831.096；B=2000、seed=20260731、cluster 1,909）
- 两条原因的全域权重占比：`NO_IPV_EFFECT` 0.5095%、`NEAR_UNIFORM` 28.2210%
- **K2 的普查结果若落在上述 CI 之外，须重点解释**，不得直接改口径

## 七、HPC 侧的既有通道（复用，不要另造）

G 轨已跑通并留下可用配置：

- 冻结环境：`/share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/bin/python`
  （Python 3.9.24，numpy 1.21.6 / scipy 1.7.3，OpenBLAS）
- 冻结 PKL 快照：`.../data/interhub/snapshots/interhub_legacy_20260711_v1/full_datasets/pkl`
  （**禁用** `subsets_for_yiru/pkl`，那是更小的 legacy 子集）
- 线程钉死：`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`（不设会破坏确定性）
- managed checkout `6bdcc2e6`
- 已验证：fata02（AMD EPYC 9654）与 cpui158（Intel）对本计算**逐位相同**
  （Slurm 2024766，348/348 float64 bitwise equal），故分区可按队列可得性选择
- 每次投递用**新建** work_dir，不得覆盖任何既有目录

## 八、明确不做的事

- **K1 阶段不投全量作业**，只投小批测成本
- 不改 `agent.py` / `ipv_estimation.py` / `process_interhub.py` / `reliability_logdomain.py`
  ——本轮只是**增加持久化的输出列**，不改任何计算
- 不改 `theta`、不做阈值扫描、不提替代判据。**规格已定稿**
- 不改 RQ009 的任何东西，不重算它的弃权率
- 不解析 RQ007 held_out，不读 RQ014 致盲字段
- 不写"规格 v2"、不做盲审、不做多路复审、不提交 git commit
- 不对 `reports/` 做全仓库 `rg`

## 九、编制

**K1：一个 codex agent，一轮 leader 自查，出报告，停在 WAITING_ON_COMMANDER。**
派发用 `.codex-fleet/rq015a-run/board/detach_launch.py`（见通用规则第 3 条）。
产出 `board/reports/K1_preflight_and_plan.md`。
**若第二节第 2 条查出现成产物已含逐候选量，立刻在 progress.log 写明并优先报监督方——
那会改变整轮的成本结构。**

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

**4. 本轮只有 K 一条 track 在跑。铁律，不可协商：**
```
禁止 git checkout -- . / git restore . / git stash / git reset --hard / git clean -fd
禁止 git checkout 任何历史提交到主工作区（要看旧代码用 git worktree add）
禁止 git commit（本轮产物由 PI 统一提交）
工作区非空是【预期状态】——此前轨道留下的文件仍在，工作区非空是预期状态。
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
