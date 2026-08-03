# Track J leader — 冻结弃权闸规格，并估计它在全语料上的影响

仓库根：`<REPO_ROOT>`
你的看板：`<REPO_ROOT>/.codex-fleet/rq015j-gate-spec/board/`
你的工作区：`<REPO_ROOT>/.codex-fleet/rq015j-gate-spec/work/`

## 一、这一轮在整个研究里的位置

最终用途是 **online verification：判断当前自动驾驶车辆的 IPV 是否符合人类的分布**。
这个判断有两个弃权机制，本轮只做第一个：

1. **IPV 没估出来** —— 就是本轮要冻结的这个闸
2. 当前场景收集到的人类 IPV 不足以判定 AV 是否偏离 —— 属 RQ009 envelope 的样本量条件，**本轮不做**

下游是 RQ009 的上下文条件 conformal envelope（已 accepted）。
PI 明确要求：**envelope 按场景上下文分格，但不按数据源拆分**——
社会倾向 IPV 应当是人类群体的固有属性，按录制来源拆分等于预设不同来源的人有不同倾向，不合理。

PI 同时明确：**这两个弃权机制在论文里只要有即可，不做重点，设计上不必苛求细节。**
本轮按这个尺度执行，**不要做成一个方法学专题**。

## 二、门的规格（已由 PI 与监督方定稿，不得改动、不得"优化"）

```
判据 1：spread(mse_per_candidate[7]) == 0   → 弃权
判据 2：max(w_log[7]) < 0.20                → 弃权
其余                                        → 可估，报 ipv_log
```

**必须在 log 域算。** 连乘域会下溢，下溢时代码回退均匀权重，会把本来可估的行错判为不可估——
门自己就被污染了。现成实现：`src/sociality_estimation/core/reliability_logdomain.py`。

**⚠ 独立复审（2026-08-02）已证实：判据 1 在样本上被判据 2 完全包含。**
400 个 `spread(mse)==0` 的行，其 `max(w_log)` 全部**恰好等于 1/7 = 0.142857143**，
一律满足 `max(w_log) < 0.20`。判据 1 的额外筛出量是 **0 行**。

因此判据 1 **保留为语义标签，不是额外的筛选条件**。线上统计必须用**互斥 reason**：
先判 `spread(mse)==0`（reason = `NO_IPV_EFFECT`），否则再判 `max(w_log)<0.20`
（reason = `NEAR_UNIFORM`）。**不得把两条写成"各自贡献"并相加，那会重复计数。**

两条判据的依据（监督方已复算，复审已核，**直接引用，不要重新论证**）：

- 判据 1 精确、无参数。七个候选给出逐位相同的 MSE，说明 IPV 对前向模型没有影响
  （无实质交互时目标退化为 `cos(ipv)·interior + 常数`，正标量不改极小点）。
  2,300 锚点样本中 **400 个（17.4%），全部来自 nuplan，waymo 零行**。
- 判据 2 的 `θ = 0.20` **是一个简单、可解释的政策阈值，不是从直方图的自然断点推出的。**
  监督方此前写"取在近均匀模态边缘"，**该依据已被复审推翻**：`max(w_log)` 在 0.20 附近没有空隙
  （`[0.18,0.20)` 95 行、`[0.20,0.22)` 71 行、`[0.22,0.24)` 74 行），
  且 `k_eff → max(w)` 不是唯一换算（同"一大六等"假设下 k_eff=6.75 给 max(w)=0.2102，
  而 max(w)=0.2027 给 k_eff=6.801）。
  报告中 θ 必须写成政策阈值，并附一行敏感性：
  θ=0.18/0.20/0.22 → 样本内门后 1,112 / 1,017 / 946 行。**不要改 θ，只要如实标注。**

## 三、一个已查明的硬约束，决定了本轮怎么做

监督方已核查全语料台账 schema：

```
reports/studies/RQ015A_ipv_estimability_labelling/
  RQ015A_1_concentration_audit_20260731T093746Z_e82091ce/concentration_ledger/
  共 4 份 parquet，合计 14,473,982 行
  列：artifact_id, product_row_key, measurement_role, case_id, rq007_split,
      ipv_error, K, candidate_grid_id, k_eff, q_eff, attempt_status,
      reason_code, recoverability, ledger_schema_version,
      aggregation_perspective, aggregation_configuration
```

**台账没有 `mse_per_candidate[7]`，也没有 `w_log[7]`。** 而且台账里的 `k_eff` 是
**连乘域**的，不是 log 域的——下溢行在那里一律显示为 `k_eff = 7`，
正是本门要救回的那批（样本中原 D1 行有 35.8% 在 log 域下是可估的）。

**所以这个门无法在现有台账上直接普查。** 真要普查必须重算全语料，
而**全量重跑未获授权，本轮不做，也不要提议做**。

因此本轮的产出是**规格 + 设计基估计**，不是普查。这一点必须在报告标题与每处引用中写明。

## 四、要做的三件事

### 第 1 件 —— 把门写成可执行、可移植的规格

产出一段独立的、不依赖本仓库上下文的判据说明（伪代码 + 输入契约 + 输出契约），
使它既能用于离线建 envelope，也能在 online 推理时逐帧调用。要点：

- 输入只允许**单次运行内可得的量**（在线场景没有第二次运行可比）
- 明确 `w_log` 的定义与归一化方式，以及与 `k_eff` 的换算关系
- 明确弃权时返回什么（不得返回 `ipv = 0`——那正是本项目要消除的混淆）
- 写明两条判据各自的触发计数如何记录，便于线上统计弃权率

### 第 2 件 —— 用设计权重估计全语料的可估率

2,300 锚点是有已知抽取概率的分层概率样本，直接复用现成机制，**不要另造**：

```
锚点与权重：.codex-fleet/rq015b-repair/work/anchor_mse.csv（36 列，含 w_log[7]、
            k_eff_log、mse_per_candidate[7]、legacy_fallback_triggered 等）
            .codex-fleet/rq015b-repair/work/mechanism_split.csv（含 ht_weight）
bootstrap： B=2000、seed=20260731（与 D0–D4 一致，不得改）
```

**⚠ 分母陷阱（I 轨已踩过，不要重犯）**：`zero_postwarm_scope == True` 等价于
`signature ∈ {U, Z}`，**会把 signature N 整层排除**。而 N 层恰恰是关键层。
本轮必须用**全域分母**（I 轨已确立：分子分母口径见其报告，全域分母 2,646,058），
N 层用其自身的 491 个 cluster。**不得沿用 D0–D4 的 `zero_postwarm_scope` 分母。**

**复审已算出这个数，你的任务是复现并解释它，不是重新发明：**
全域分母 `2,646,058`，门后保留权重 `1,885,831.096`，
**全域可估率 = 71.2695%**，cluster bootstrap CI `[67.1729%, 75.2135%]`。

**这个数与样本内的 44.2174% 差距很大，必须解释清楚差在哪。** 监督方已复算出原因：

| signature | 样本内可估率 | 该层占全域权重 | 层内加权可估率 |
|---|---|---|---|
| N | 363/500 = 72.6% | **79.8%** | 76.3% |
| U | 511/1200 = 42.6% | 9.4% | 76.0% |
| Z | 143/600 = 23.8% | 10.8% | 29.9% |

样本按配额把 U 抽成了 1,200 行（占样本 52%），而 U 在全域只占 **9.4%** 的权重；
全域实际由 N 主导（79.8%），而 N 是高可估的。**所以样本比例严重低估了全域可估率。**

交付要求：**主文一律用 HT 比率与 CI，样本未加权比例只能出现在方法附注里，并注明它不是全域影响。**
两条判据的计数按互斥 reason 报：全域权重下 `NO_IPV_EFFECT` 占 0.5095%、
`NEAR_UNIFORM`（非 spread==0 部分）占 28.2210%。

### 第 3 件 —— 门后分布的形状（**已按复审收窄**）

**⚠ 原计划要求"按 RQ009 实际用的上下文变量分格"，独立复审已证实这一条无法执行，现取消。**
理由：RQ009 的 context-only 变量共 22 个数值 + 7 个类别
（`relative_distance_anchor`、`relative_speed_anchor`、`closing_rate_anchor`、
`geometry_path_category`、`priority_role` 等，见 RQ009 `calibration.py` 的
`BASE_NUMERIC_CONTEXT` / `BASE_CATEGORICAL_CONTEXT`），
而 `anchor_mse.csv` 与 `mechanism_split.csv` 中**这 29 个变量一个都没有**。
临时去 join RQ009 feature matrix 又会撞上 RQ007 边界——feature partitions 不带 `rq007_split`，
本计划没有定义安全的 join 路径。**不要临时发明 join 口径。**

改为：**只用锚点自带变量报门后分布形状**，即 `signature`、`n_band`、`n_obs` 分箱。
每格报：样本数、HT 加权可估率、门后 `ipv_log` 的 p5 / 中位 / p95、边界占比（口径见下）。

**边界口径必须分开报两个数，不得混用**（复审发现监督方此前混了）：
- `at_grid_boundary` 列：门后 nuplan **25.32%**、waymo **56.45%**
- `|ipv_log|` 精确命中 `3π/8`（1e-9 容差）：nuplan **0.96%**、waymo **20.28%**

监督方此前对外只报了后者（说成"边界饱和 1%/20%"），**低估了边界问题**。
两个口径都要报，并写明各自定义。

**不得按 source 拆分输出**。source 只作为内部检查（见下）。

**内部检查（一小节，不展开）**：监督方复算的分带表如下，复审已逐格核对一致：

| 权重带 | nuplan \|ipv\| 均值 | waymo \|ipv\| 均值 |
|---|---|---|
| 0.20–0.25 | 0.1306 (n=81) | 0.0507 (n=86) |
| 0.25–0.35 | 0.1777 (n=83) | 0.2001 (n=96) |
| 0.35–0.50 | 0.2734 (n=68) | 0.3560 (n=77) |
| 0.50–0.75 | 0.4383 (n=41) | 0.5706 (n=94) |
| 0.75–1.01 | 0.7551 (n=39) | 0.9698 (n=352) |

**但结论措辞已按复审改弱。** 不得写"各带内基本重合、差异几乎全部来自选择效应"——
带内仍有可见残差（最高带中位数 nuplan 0.0639 vs waymo 0.8005），
且按 `at_grid_boundary` 口径 waymo 在各带内持续更高。正确写法：

> 最高权重带的样本量不平衡（waymo 352 vs nuplan 39）解释了汇总均值差异的相当一部分；
> 分带后仍存在残余差异。该残差不作为下游分源口径的依据，仅作内部风险披露。

### 另需写进报告的一句限定（**已按复审改弱**）

门后保留的样本，`|ipv|` 随集中度单调上升，因此
**envelope 是"可估交互条件下人类的 IPV 分布"，不是"人类的 IPV 分布"**，论文措辞必须带这个限定。

**但不得写成"AV 侧用同一把尺子，偏移两边抵消"——复审已指出该论证不成立。**
同一道门只保证两边都被条件化到"通过门"的子总体，不保证偏移抵消；
若 AV 的 `max(w_log)` 分布、边界占比或 IPV 形状与人类不同，门会改变被比较的对象。
正确写法：

> 结论只对"同一道门通过后的条件分布"成立。若 AV 的通过机制与人类不同，
> 须分别报告 AV 与人类的门通过率，并把"未通过"本身作为监控结果的一部分。

## 五、明确不做的事

- **不重算任何锚点，不提交任何 HPC 作业，不提议全量重跑**
- **不做跨环境（Mac vs HPC）可复现性分析**——PI 已明示这条不再细究
- 不改 `agent.py` / `ipv_estimation.py` / `process_interhub.py` / `reliability_logdomain.py`
- 不调整 θ、不做阈值敏感性扫描、不提出替代判据。**门已定稿**
- 不设计第二个弃权机制（envelope 样本量条件），那是 RQ009 的事
- 不写「规格 v2」、不做盲审、不做多路复审、不提交 git commit
- 不对 `reports/` 做全仓库 `rg`（会把 RQ003 controlled-access 行拉进上下文）

## 六、产出

`board/reports/J1_gate_spec_and_impact.md`，须含：

1. **门的可执行规格**。输出契约必须机器可读，至少含：
   `status ∈ {OK, ABSTAIN}`、`ipv_log`（弃权时为 null，**不得为 0**）、
   `reason_code`（互斥主因：`NO_IPV_EFFECT` / `NEAR_UNIFORM`）、
   `max_w_log`、`mse_spread`、`k_eff_log`、`candidate_grid_id`、`K`、`frame_id`。
   RQ009 接入还需门通过率与 context cell key 两个字段
2. **全域可估率的设计基估计**（点估计 + CI，**标题与每处引用必须写明是 design-based estimate，不是普查**），
   两条判据各自贡献
3. **按锚点自带变量（signature / n_band / n_obs）分格的表**（不分源），
   每格含 HT 加权可估率与门后分布分位数；边界占比两个口径都报
4. **source 内部检查一小节**（复现上表，确认或否定"差异是选择效应"）
5. **「可估交互条件下」这一限定的一段说明**
6. **明确写出：要把这个门真正应用到全语料，需要什么**（台账缺 log 域权重，须重算；给出量级估计供 PI 决策）
7. 每个数字须可复算：写明文件、列名、筛选条件、权重

## 七、编制

**一个 codex agent（J1），一轮 leader 自查，出报告，结束。** 不要开第二个 agent。
派发用 `.codex-fleet/rq015a-run/board/detach_launch.py`（见通用规则第 3 条）。
本轮无外部依赖、无排队、无重算，是三件事里最轻的一轮——
**若超过 90 分钟还没出数字，说明范围失控，回来收紧而不是加人。**

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

**4. 本轮只有 J 一条 track 在跑。铁律，不可协商：**
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
