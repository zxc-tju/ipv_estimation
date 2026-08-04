# RQ016B-F1 任务书：把重建后的 envelope 用到 WOD 与 OnSite 上，可行性审计

你是本轮唯一的执行 agent。读完就开工，不要写第二版方案，不要开子轨。
本任务是**只读的可行性审计**：查清楚 → 一轮自查 → 出报告 → 结束。
**不跑任何估计器、不投 Slurm、不做 materializer、不训练任何模型。**

仓库根即当前工作目录，以下路径都相对仓库根。

---

## 0. 这件事在整个研究里的位置（不要跳过）

最终目标是**在线验证**：一辆自动驾驶车在路上跑，判断它表现出的社会交互倾向像不像人。
IPV（Interaction Preference Value）是一个标量，表示交互倾向。判定由**两道串联的弃权机制**构成：

- **机制一**：这一帧的 IPV 数值能不能估？若七个候选的权重近均匀，则该数值不携带候选间的
  判别信息，**直接弃权，不进机制二**。RQ015 已完成并冻结这道门。
- **机制二**：当前场景收集到的人类样本，够不够判断这辆车是否偏离？依据是人类参照分布
  （envelope，RQ009 已接受的 context-conditioned split-conformal 区间）。

**RQ016 刚刚完成的事**：机制二依赖的人类 envelope 此前建在含伪零的样本上（旧估计器数值
下溢时退回七候选等权，写出 IPV 恰为 0，使「没估出来」与「恰为中性」不可区分）。用只过门的
样本重建后，90% 区间平均宽度增加 28.02%（1.016189 → 1.300967），覆盖基本不变
（0.898832 → 0.902689）。结论、边界与可复跑脚本见
`reports/studies/RQ016_human_envelope_rebuild/RQ016_1_envelope_rebuild_20260803T134808Z_d23fa836/`。

**但 RQ016 全程只用了 InterHub 的人类数据。** 而这项研究真正要判的对象——自动驾驶车——
在 **WOD**（Waymo Open Dataset，含 AV 轨迹）与 **OnSite**（含 AV 的竞赛场景库）里。

**所以本轮要回答的问题是**：把重建后的 envelope 用到 WOD 与 OnSite 上，
**要付出什么代价才能做成，以及哪些部分根本做不成**。

---

## 1. 监督方已实测的事实（直接引用，不要重算后报一个略不同的数）

源：`data/derived/rq015k_logdomain_gate/l1_v1/`（510 个 parquet 分片，14,473,982 行）。
监督方于 2026-08-04 在本机实测：

**机制一的门判据输入在两个数据集上完全不存在。**

| 列 | `wod_rq010b_full479_audited` (n=906) | `onsite_dense_timeseries` (n=281,268) | 对照 `interhub_sigma01_hw4_timeseries` (n=5,197,072) |
|---|---:|---:|---:|
| `mse_0` … `mse_6` | 0/906 非空 | 0/281,268 非空 | 4,980,050 非空 |
| `w_log_0` / `max_w_log` / `mse_spread` / `k_eff_log` | 0/906 | 0/281,268 | 4,980,050 |
| `ipv_log` | 0/906 | 0/281,268 | 3,502,340 |
| `status` / `reason_code` | 0/906 | 0/281,268 | 4,981,984 / 1,479,644 |
| `gate_applicable` | False×906 | False×281,268 | True×4,981,984 |
| `ipv_error`（旧标量） | 906/906 | 3,160/281,268 | 5,197,072 |
| `q_eff`（旧标量） | 906/906 | 2,974/281,268 | 4,981,984 |
| `candidate_grid_id` / `K` | 906/906 | — | 4,981,984 |
| `context_cell_key` | 0/906 | 0/281,268 | 4,981,984 |

`out_of_scope_reason`：WOD 为 `NO_MATERIALIZER_IN_SCOPE`×906；
OnSite 为 `SOURCE_UNKNOWN`×274,022、`SOURCE_NOT_ATTEMPTED`×4,272、`NO_MATERIALIZER_IN_SCOPE`×2,974。

OnSite 的 `source_attempt_status`：`UNKNOWN` 274,022（97.4238%）、`NOT_ATTEMPTED` 4,272（1.5188%）、
`ATTEMPTED` 2,974（1.0574%），分母 281,268。全部 `UNKNOWN` 行的 `source_reason_code`
都是 `EMPTY_CELL_UNEXPLAINED`（274,022/274,022）。

**因此结论已定，不必再证**：现有产物不足以对 WOD/OnSite 评估机制一，
新 envelope 无法直接套用。**本轮不是去论证这一点，而是查清楚要做成得付什么代价。**

监督方另已确认一个正面信号：OnSite 侧存在带 `geometry_path_category` / `turn_pair_label` /
`priority_role` 的文件，位于
`data/derived/onsite_competition/RQ011B_matched_scenario/RQ011B_1_matched_scenario_20260625T202454_8331bd49/`
下的 `onsite_ipv_channel_target*.csv` 等。**这条线索必须查实并给出列级证据。**

---

## 2. 你要回答的五个问题（这就是交付物的骨架）

对 **WOD** 与 **OnSite** 各回答一遍，分开写，不要合并成一套说法。

### Q1 — 重跑估计器的输入齐不齐？

七候选 MSE 要靠按冻结配置重跑估计器才能得到。查清每个数据集是否具备：
配对轨迹、采样率与时间步长、可用历史窗口长度、参考线/地图、以及冻结配置
（`configs/ipv_sigma01_exact.json`，网格 `legacy7_pi_over_8`、`K=7`、`sigma=0.1`）
所要求的一切输入。**逐项给出文件路径与列名**，不要只说「有」或「没有」。

已知的相关约束：L2 查明 OnSite dense 源表的轨迹、配对 ID、位置、速度、heading、距离、
相对速度字段为 274,022/274,022 非空，但**真实地图/车道/reference-line 字段为 0/274,022**，
当时用 observed trajectory fallback 作参考线。请核实这一条，并说明它对重跑的影响。

### Q2 — M2 的 29 个 context 特征能不能造出来？

envelope 的分箱用的是 RQ009 的 M2 特征集：22 个数值 + 7 个类别，定义在
`reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/04_calibration/calibration.py:95-127`
（`BASE_NUMERIC_CONTEXT` 与 `BASE_CATEGORICAL_CONTEXT`）。支持门另用
`GATE_DISTANCE_NUMERIC`（同文件 141-157 行，本轮口径为其中不含 `counterpart_ipv_*` 的 12 项）
与 `GATE_SUPPORT_CATEGORICAL`（158-162 行）。

**逐个特征给出三选一的判定**，做成一张 29 行的表：

- `AVAILABLE` —— 该数据集已有，给出文件路径与列名
- `DERIVABLE` —— 可由已有字段算出，给出算法与所需输入列
- `MISSING` —— 造不出来，说明缺什么、以及补齐需要什么

**这是本轮最重要的一张表。** 若某个特征在 InterHub 上是「audited」标签
（如 `geometry_path_category`、`geometry_path_relation`、`turn_pair_label`、`priority_role`），
必须查清它在 WOD/OnSite 上是否有同义物、口径是否一致、还是需要重新标注。

### Q3 — 规模与范围

给出各数据集在不同范围口径下的**行数/锚点数**，每个数都带分母与筛选条件：

- WOD：906 台账行对应多少 case/scene/帧？`rq015a_full479_projected` 里有什么？
- OnSite：已知构建时设了 `max_anchors_per_unit = 1`，
  `valid_anchor_candidate_total = 67,861`、`anchors_excluded_by_cap = 67,594`、
  `total_av_anchors = 267`。**核实这三个数**并给出来源文件与字段。
  再给出 PI 此前提过的三个范围选项各自的规模：
  (A) 全 aligned frames，(B) 全 RQ009 timing-valid anchor frames，(C) 继续每 unit 一个 anchor。

### Q4 — 迁移效度

envelope 是在 InterHub 人类数据上拟合的，用到 WOD/OnSite 属跨数据集迁移。
查清 RQ009 已有的 LODO（leave-one-dataset-out）证据说了什么：
已知 90% coverage 区间为 0.749–0.991，来源 `reports/knowledge/RQ009_dynamic_counterpart_conditioned_envelope/decision.md`
的 Boundaries 节与 `02_process/05_evaluation/lodo_results.csv`。
**查明 LODO 里包含哪些 source_dataset，WOD 与 OnSite 是否在其中。**
若不在，直说「无同源迁移证据」，不要用别的数据集的数字替代。

### Q5 — 最小可行路径与代价

给出**能产出一个真实结果的最小路径**，分数据集写，包含：

- 需要哪几步、每步的输入与产出
- 计算量估计（多少个求解单元、单元耗时依据、能否本机跑完还是必须上 HPC）
- 哪些步骤需要 PI 先拍板才能定义分母（例如 OnSite 的范围选项与参考线合同）
- **明确指出哪条路是「做不成」的**，并说明卡在哪一步

**若 WOD 与 OnSite 的最小路径代价差很多，明确说哪个更值得先做，并给判据。**

---

## 3. 硬边界

```
只读审计。禁止运行估计器、禁止投 Slurm/HPC、禁止训练模型、禁止写 data/derived/
不改：src/sociality_estimation/core/{agent,ipv_estimation,reliability_logdomain}.py
      pipelines/interhub/process_interhub.py
      configs/ipv_sigma01_exact.json
不改 RQ009 与 RQ016 已落盘的 run 目录（只读）
不做 git commit / 不碰 git 的任何写操作
禁止 git checkout -- . / restore . / stash / reset --hard / clean -fd
RQ007 held_out 不得被解析
RQ014 致盲相关的评分字段不得读取 —— 本轮尤其注意：WOD 是 RQ014 的地盘，
   那 906 行的 61 个含 rating 的列在 HPC 侧就已投影掉、本地从未出现。
   若你在任何 WOD 相关文件里遇到 rating / score / preference / human-score 一类字段，
   立即停下写进报告，不要读取内容。
不得静默覆盖已冻结产物或已接受的 decision.md
不要对 reports/ 做全仓库 rg（会把 RQ003 controlled-access 行拉进上下文）；用定向 ls/grep
git status 一律用 git --no-optional-locks status --porcelain
时间戳一律实取 date -u +%Y-%m-%dT%H:%M:%SZ，不要前瞻估计
```

**措辞禁令**：全文禁用 `estimability` 与「测出/未测出 IPV」。可辩护的表述是
**权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息**。描述性结果不得写成因果主张。
不用比喻、不用自造简称；必须用项目专有名词时当场用一句话说明它是什么。

**分母纪律**：每个百分数必须紧跟分子、分母、筛选条件、来源文件与列名。
写不出来就不要写这个数。不许在不同分母之间搬运比率。

环境：本机 python3 已有 pyarrow 21.0.0 / pandas 2.3.3。缺依赖直接装上继续，不要停下来问。

---

## 4. 交付物

- 报告：`.codex-fleet/rq016b-wod-onsite-feasibility/board/reports/RQ016B_1_feasibility.md`
- 机器可读：`.codex-fleet/rq016b-wod-onsite-feasibility/work/F1/feasibility_matrix.json`
  （至少含 Q2 那张 29 行特征表的结构化版本，WOD 与 OnSite 各一份）
- 你写的探查脚本放 `.codex-fleet/rq016b-wod-onsite-feasibility/work/F1/`，可复跑

报告结构照第 2 节的五问，WOD 与 OnSite 分开写。**开头必须先定位**：这项工作要解决什么问题、
整体走到哪一步、本次是其中哪一环——写给一个完全没跟进过程的读者。
**结论与待决事项分开**：需要监督方或 PI 拍板的事单独成节，写清选项、判断依据、不做的后果。

## 5. 自查（一轮，但必须有牙齿）

1. **列级证据**：Q2 表里每一行标 `AVAILABLE` 的，都要能给出「文件路径 + 列名 + 非空计数/总行数」。
   给不出就不能标 `AVAILABLE`。
2. **负对照（强制）**：挑一条你自己的判定规则，故意扰动使它失败，把失败输出贴进报告。
   一条永远不会 FAIL 的检查不算检查。
3. **数字可复算**：报告里每个计数都要能由你留下的脚本重跑得到。
4. **边界事件**：若遇到疑似 RQ014 评分字段或 RQ007 held_out，写进报告的单独一节，
   注明你没有读取内容。

## 6. 报告结尾必须带状态行

```
state: WAITING_ON_COMMANDER
timestamp_utc: <实取>
```
