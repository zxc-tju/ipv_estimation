# L1 — 把 RQ009 打分目标里的「精确零点」拆成两类

你是 RQ015 收官轮（track L）的第一个执行 agent。仓库根就是你的 `--cd`，
即 `.`。

---

## 一、位置（先读完，不要跳到任务）

这项工作的最终用途是 **online verification**：判断一辆自动驾驶车的 IPV
（Interaction Preference Value，一个刻画社会交互倾向的标量参数）是否落在人类分布内。

整条判据由**两个弃权机制串联**：
- **机制一**：判「这一帧的 IPV 数值到底携不携带信息」。弃权则直接结束，不进机制二。
- **机制二**：RQ009 已 accepted 的 envelope 支持度判据。

**RQ015 整条线做的是机制一。** 它的起点是一个具体缺陷：
原实现在数值下溢时退回「七个候选等权」的兜底；而候选网格对称，
于是必然算出 `ipv` **恰为 0**、`ipv_error` **恰为 `1 - 1/√7 = 0.6220355269907728`**。
后果是：**「算失败了」与「该个体完全自利（IPV=0）」在数据里不可区分。**

RQ015 A–K 十一条轨道已全部结项。K2 交付了全语料台账（14,473,982 行），
其中逐行记录了新的 log 域门判据结果，因此**现在有能力把上面这两件事分开**。

**本轮（track L）三件事：L1（你）、L2（并行的另一个 agent）、L3（成文）。你只做 L1。**

---

## 二、L1 要解决的问题

RQ009 自己的报告里有一条限制声明。原文位置：

```
reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/90_report/index.html
```
第 127 行（`<h2>Limitations</h2>` 之后第二段）：

> A separate target-distribution caveat is atom/boundary mass: the scored IPV target has a
> ~21.5% exact-zero atom (273,819/1,270,566) plus many exact-repeat and boundary values.
> This creates the 80% boundary-tie / 1e-10 endpoint-nudge coverage fragility and attenuates
> correlations, so it qualifies both interval-tie behavior and the interpretation of the practical null.

也就是说：**RQ009 的打分目标里有 273,819 / 1,270,566 = ~21.5% 的行，目标值恰好等于 0。**
RQ009 当时无法判断这些 0 是什么，只能把它当作分布上的一个原子（atom）来警告。

**按上面第一节的缺陷描述，这个零点原子里混着两种完全不同的东西：**
1. **真中性零**：交互确实被评估了，argmin 落在候选 0 上，这是一个合法的 IPV 估计值；
2. **伪零**：均匀兜底触发，代码写下 0，它根本不是估计结果。

**你的任务：用 K2 台账把这 273,819 行拆开，给出两类各自的行数与占比。**

### PI 裁定（硬约束）
**只拆分，不动 RQ009。** 不得修改 RQ009 目录下任何文件、不得重算它的 envelope、
不得重算它的 4.78% 弃权率、不得改它已 accepted 的任何结论。你只做只读分析。

---

## 三、**第一步是判定 join 是否成立，不是直接 join**

这一条是监督方点名要核的，**比拆分结果本身更重要**。本轮此前已经出过两次
「看起来在检查、实际没检查该检查的东西」（RQ009 `duplicates` 硬编码为 0、G 锚点比错基线），
不要出第三次。

你必须**先**把下面三件事查清并写进机器可读证据，**然后**才允许做 join：

1. **RQ009 打分目标的行键到底是什么？** 那 1,270,566 行的唯一键由哪些字段构成？
   它是哪个文件、哪一列？（打分目标 = 被 RQ009 用作预测目标 `y` 的那个 IPV 数值列）
2. **K2 台账 `artifact_id=rq009_feature_matrix` 分区里两个键各是什么口径？**
   - `canonical_key`（据 K2 报告为 `product_row_key` + `|role=` + `measurement_role`）
   - `interhub_canonical_key`（样例形如 `ipv_035969|7|2`）
   分别在什么粒度上唯一？
3. **1,270,566 与 8,994,736 是什么关系？** 子集？不同粒度？不同过滤条件？
   必须给出可复算的答案，不能只说「大概是」。

**如果无法建立精确的一对一 join，就如实写「不可精确 join」并说明卡在哪一步、
差在哪个字段。绝对不许用近似匹配、不许放宽键、不许写"大致对得上"。**
如实报告不可 join 是一个**合格的交付**，胡乱 join 出一个数字不是。

### 已知线索（省你时间，但必须自己复核，不许直接引用）

- K2 台账本地路径：`data/derived/rq015k_logdomain_gate/l1_v1/`
  按 `artifact_id=` / `shard_id=` 两级 hive 分区，`artifact_id=rq009_feature_matrix` 下有 45 个 shard。
  列名与一行样例见 K2 报告与你自己的 `pyarrow` 读取。
- K2 报告：`.codex-fleet/rq015k-fullcorpus-gate/board/reports/K2_fullcorpus_gate_ledger.md`
  （§3 行数核算、§6 join 检查、§9 接口警告，必读）
- RQ015E 轨（做过 RQ009 依赖分析，知道 RQ009 冻结产物在哪）：
  `.codex-fleet/rq015e-rq009-dependency/board/reports/E1_dependency_report.md`
  其中记有 `ledger split counts: {'development': 6459684, 'guard': 2535052}`，
  两者之和恰为 8,994,736；held_out 在台账中为 0 行。
  也记有 `M0/M2/M3 join identities: 2666676 + 1145022 = 3811698`。
  **注意 3,811,698 = 1,270,566 × 3。这个整除关系是不是巧合，由你查证，不许假定。**
- RQ009 报告里记录零点原子的机器文件，优先看这几个（已确认含 `1270566` 或 `273819`）：
  - `.../RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/03_features/matrix_audit.json`
  - `.../02_process/03_features/feature_health.json`
  - `.../02_process/03_features/matrix_audit.md`
  - `.../01_results/figures/c6_limitations.source.csv`
  - `.../evidence.csv`、`.../execution_status.json`
- RQ009 的评估代码（只读，别改）：`.../02_process/05_evaluation/evaluate.py`

---

## 四、join 成立时必须给出的结果

对那 273,819 行（分母恒为 273,819，占比也可另给以 1,270,566 为分母的版本，
但**每个百分数后面必须紧跟它的分母**）：

| 类别 | 判据 | 要给 |
|---|---|---|
| 过门的真中性零 | K2 台账 `status = OK` | 行数 + 占比 |
| 弃权而被记成 0 | `status` 非 OK | 行数 + 占比 |
| ├ `reason_code = NEAR_UNIFORM` | | 行数 + 占比 |
| ├ `reason_code = NO_IPV_EFFECT` | | 行数 + 占比 |
| └ 工程失败 `SOLVER_FAILURE` | | 行数 + 占比 |
| join 未命中 | 台账里查无此行 | 行数 + 占比（**必须单列，不许悄悄丢掉**） |

**这直接回答 RQ009 自己那条警告：那 21.5% 里有多少是真中性、有多少是伪零。**

### 两个必须做的一致性检查

1. **零值判定要写清口径。** 台账里 `ipv_log` 的样例值出现过 `-5.551115123125783e-17`，
   即**不是恰好 0**。所以你必须分清三件事并分别报数：
   (a) RQ009 打分目标列里**恰好 == 0.0** 的行（这是那 273,819 行的定义）；
   (b) 台账 `ipv_log` **恰好 == 0.0** 的行；
   (c) 台账 `ipv_log` 在浮点容差内为 0 的行（容差自己选并写明，例如 `abs < 1e-12`）。
   **不要把 (a) 和 (b) 混为一谈**，它们来自不同代码路径（旧实现 vs log 域实现）。
2. **兜底指纹交叉验证。** 旧实现触发均匀兜底时 `ipv_error` 恰为 `0.6220355269907728`。
   台账 `artifact_id=rq009_feature_matrix` 分区里**有 `ipv_error` 列**。
   请核对：`ipv_error` 命中该指纹的行，是否与 `status` 非 OK 的行高度重合？
   给出交叉表（两两组合的行数），并说明不重合的部分是什么。
   **这是本节最有价值的独立佐证——如果两条独立线索指向同一批行，结论就硬。**

---

## 五、交付物

全部放在 `.codex-fleet/rq015l-consolidate/work/L1_rq009_zero_atom_split/`：

1. `L1_join_feasibility.json` — 第三节的三个问题的答案 + `join_feasible: true/false` + 卡点说明
2. `L1_zero_atom_split.json` 与 `L1_zero_atom_split.csv` — 第四节的表，每个计数带分子/分母/筛选条件/来源文件与列名
3. `L1_fingerprint_crosstab.csv` — 第四节的兜底指纹交叉表
4. 你用的脚本（`.py`），可重跑
5. **`L1_report_section.md`** — 给 L3 直接引用的报告一节。要求见下。

### `L1_report_section.md` 的写法要求（会被逐条核）

- **开头先定位再讲结果**：这项工作要解决什么问题、RQ015 整体走到哪一步、L1 是其中哪一环。
  不得直接从增量讲起。读者是一个没跟进过程的人，一次就要读懂。
- **不用黑话、不用比喻。** 必须用项目专有名词时，当场一句话说明它是什么。
- **每个百分数后面紧跟分母。** 一个读者无法自行复算的数字等于没给。
- **结论与待决事项分开成节。**
- 若 join 不成立，本节主体就是「为什么不成立」，不要硬凑结果。

---

## 六、硬边界（违反即退回）

- **不改** `src/sociality_estimation/core/agent.py`、`ipv_estimation.py`、`reliability_logdomain.py`、
  `pipelines/interhub/process_interhub.py`、`configs/ipv_sigma01_exact.json`
- **不改 RQ009 目录下任何文件**（只读打开）
- 不投 Slurm，不重解任何锚点，不重跑 K2 的 join，不做任何重算
- 不 `git commit`；**禁止** `git checkout -- .` / `git restore` / `git stash` / `git reset --hard` / `git clean -fd`
- **不解析 RQ007 held_out 集**（`rq007_split == 'held_out'`）。台账里 held_out 为 0 行，
  你的筛选也不得以任何方式把 held_out 行读进统计。不读 RQ014 致盲相关评分字段。
- **不要对 `reports/` 做全仓库 `rg`/`grep -r`**（会把 RQ003 controlled-access 内容拉进上下文）。
  只在 RQ009 那一个报告目录内做定向查找。
- **全文禁用 `estimability` 一词，禁用「测出 / 未测出 IPV」的说法。**
  可辩护的表述是：**权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息**。
- 描述性结果**不得**写成因果主张。
- 时间戳一律用 `date -u +%Y-%m-%dT%H:%M:%SZ` 实取，**不要前瞻估计、不要编**。

## 七、过程纪律

**一轮做完 + 一轮自查，就结束。** 不做盲审、不出第二版规格、不写治理文书。
本项目此前出现过严重过程膨胀（一个描述性审计走了 8 个计划版本、7 轮盲审、32 个 agent，
科学结论产出为零），那是反面案例不是标准。

自查只查三件事：(1) 每个数字能否从你落盘的文件复算出来；(2) 是否有无分母的百分数；
(3) join 可行性结论是否与你实际做的检查一致。

完成后在 `L1_report_section.md` 末尾写一行
`state: WAITING_ON_LEADER` 和实取的 UTC 时间戳，然后结束。**不要自称 DONE。**
