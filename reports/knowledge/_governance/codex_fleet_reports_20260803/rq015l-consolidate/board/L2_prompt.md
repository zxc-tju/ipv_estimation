# L2 — 查清 OnSite 那 274,022 行为什么是 `UNKNOWN`

你是 RQ015 收官轮（track L）的第二个执行 agent。仓库根就是你的 `--cd`，
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

RQ015 A–K 十一条轨道已全部结项。K2 交付了全语料台账（14,473,982 行）。
**本轮（track L）三件事：L1（并行的另一个 agent）、L2（你）、L3（成文）。你只做 L2。**

---

## 二、L2 要解决的问题

K2 台账里，OnSite 这个数据源（`artifact_id=onsite_dense_timeseries`，共 281,268 行）
按 `source_attempt_status` 的构成是：

| `source_attempt_status` | 行数 |
|---|---:|
| `ATTEMPTED` | 2,974 |
| `NOT_ATTEMPTED` | 4,272 |
| **`UNKNOWN`** | **274,022** |

**未知态比真正尝试过的行多两个量级（274,022 vs 2,974，约 92 倍）。这个比例本身可疑，
所以要查它。** 注意：这里说的是 K2 台账中记录的**来源侧**状态字段，
不是 K2 新算的门判据结果——这些行在台账里都是 `gate_applicable=false`。

**本轮 PI 裁定：只查证，不重算，不补数据。** WOD 的 906 行与 OnSite 的 2,974 行
本轮**不处理**，继续保持不适用状态。你的产出是**判断与证据**，不是修复。

---

## 三、必须回答的四个问题

### Q1. `UNKNOWN` 是在哪一段代码、依据什么条件写出来的？
**给文件路径与行号。** 不许只说「大概在 build_ledger 里」。
要写出触发 `UNKNOWN` 的**判断条件原文**，以及它与 `ATTEMPTED` / `NOT_ATTEMPTED`
两个分支的判定关系（是 else 兜底？是显式条件？是字段缺失导致的默认值？）。

起点线索（自己复核，不要直接引用）：
- `scripts/rq015a/build_ledger.py`
- `scripts/rq015a/rq015a_types.py`
- `scripts/rq015a/rq015a_contracts.py`
- `scripts/rq015a/run_rq015a.py`
- `scripts/rq015a/factor_analysis.py`
- 对应测试：`tests/test_rq015a_build_ledger.py`、`tests/test_rq015a_contracts.py`

### Q2. 这 274,022 行**有没有**做 IPV 估计所需的输入？
这是本任务的核心问题，要把两种可能分开：
- **(a) 数据确实不支持**：观测长度不够、没有可配对的交互对手、缺参考线/地图、
  轨迹缺失等——即便流水线走到这里也估不出来；
- **(b) 流水线没走到**：输入其实齐备，只是这批行从未被送进求解器。

**判据要落在数据上，不能只靠读代码猜。** 请实际打开 OnSite 的来源产物核查
上述输入字段的存在性与取值分布，给出可复算的计数。
台账侧路径：`data/derived/rq015k_logdomain_gate/l1_v1/artifact_id=onsite_dense_timeseries/`
（2 个 shard）。OnSite 上游数据在 `data/derived/onsite_competition/` 下，
相关 RQ 目录有 `RQ011_onsite_full_universe_readiness`、`RQ011B_matched_scenario`、
`RQ012_onsite_event_annotation_readiness`、`RQ012B_event_harm`。
另可参考 `.codex-fleet/rq011-onsite-readiness/` 与 `.codex-fleet/RQ012_event_annotation_readiness/` 的既有结论。

**如果证据不足以在 (a)/(b) 之间下判断，就明确写「证据不足以区分」，
并列出还缺哪一项具体证据。不许含糊表述。**

### Q3. 与 RQ015A 审计当时的口径是否一致？
RQ015A 的审计记录：OnSite 仅 **1.06%** 的行携带 IPV 数值。
请核对这个 1.06% 的**分子、分母、筛选条件、来源文件与列名**，
并说明它与本轮 `2,974 / 281,268` 的关系（`2,974 / 281,268 = 1.0574%`，看着接近，
**但你必须查证是否同一口径**，不许因为数值接近就断言一致）。
若口径不同，写清差在哪。RQ015A 的执行记录在
`reports/studies/RQ015A_ipv_estimability_labelling/` 与 `.codex-fleet/rq015a-run/board/`。

### Q4. 若属于「流水线没走到」，补齐需要什么？
**只给判断与清单，不要动手补。** 要写：需要哪些输入、大致什么量级的计算、
有没有已知阻碍（如缺参考线、缺配对逻辑）。不要给推荐、不要给排期。

---

## 四、交付物

全部放在 `.codex-fleet/rq015l-consolidate/work/L2_onsite_unknown/`：

1. `L2_unknown_provenance.json` — Q1 的答案：文件、行号、条件原文、分支关系
2. `L2_input_availability.csv` 与 `.json` — Q2 的逐字段可用性计数
   （每个计数带分子/分母/筛选条件/来源文件与列名）
3. `L2_rq015a_consistency.json` — Q3 的口径比对
4. 你用的脚本（`.py`），可重跑
5. **`L2_report_section.md`** — 给 L3 直接引用的报告一节

### `L2_report_section.md` 的写法要求（会被逐条核）

- **开头先定位再讲结果**：这项工作要解决什么问题、RQ015 整体走到哪一步、L2 是其中哪一环。
  不得直接从增量讲起。读者是一个没跟进过程的人，一次就要读懂。
- **不用黑话、不用比喻。** 必须用项目专有名词时，当场一句话说明它是什么。
- **每个百分数后面紧跟分母。**
- **结论与待决事项分开成节。** 需要上级拍板的事单独成节，写清选项、判断依据、不做的后果。
- Q2 若判不出来，就把「证据不足以区分」写成明确结论，不要模糊过去。

---

## 五、硬边界（违反即退回）

- **不改** `src/sociality_estimation/core/agent.py`、`ipv_estimation.py`、`reliability_logdomain.py`、
  `pipelines/interhub/process_interhub.py`、`configs/ipv_sigma01_exact.json`
- **不改 RQ009 目录下任何文件**
- **不修改 `scripts/rq015a/` 下任何文件**（只读阅读，这是查证不是重构）
- 不投 Slurm，不重解任何锚点，不重跑任何求解，不补任何数据
- 不 `git commit`；**禁止** `git checkout -- .` / `git restore` / `git stash` / `git reset --hard` / `git clean -fd`
- **不解析 RQ007 held_out 集**，不读 RQ014 致盲相关评分字段
- **不要对 `reports/` 做全仓库 `rg`/`grep -r`**（会把 RQ003 controlled-access 内容拉进上下文）。
  只在 RQ015A / RQ011 / RQ012 各自的报告目录内做定向查找。
- **全文禁用 `estimability` 一词，禁用「测出 / 未测出 IPV」的说法。**
  可辩护的表述是：**权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息**。
- 描述性结果**不得**写成因果主张。
- 时间戳一律用 `date -u +%Y-%m-%dT%H:%M:%SZ` 实取，**不要前瞻估计、不要编**。

## 六、过程纪律

**一轮做完 + 一轮自查，就结束。** 不做盲审、不出第二版规格、不写治理文书。
本项目此前出现过严重过程膨胀（一个描述性审计走了 8 个计划版本、7 轮盲审、32 个 agent，
科学结论产出为零），那是反面案例不是标准。

自查只查三件事：(1) 每个数字能否从你落盘的文件复算出来；(2) 是否有无分母的百分数；
(3) Q1 给的行号是否真的指向写出 `UNKNOWN` 的那段代码（**打开确认，别凭记忆**）。

完成后在 `L2_report_section.md` 末尾写一行
`state: WAITING_ON_LEADER` 和实取的 UTC 时间戳，然后结束。**不要自称 DONE。**
