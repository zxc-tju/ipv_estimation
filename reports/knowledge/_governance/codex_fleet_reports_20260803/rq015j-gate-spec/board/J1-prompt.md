# J1 — 冻结弃权闸规格，并用设计权重估计它在全语料上的影响

你是 track J 的唯一执行 agent（J1）。本轮只有你一个 agent，出一份报告就结束。
仓库根：`.`
Python 解释器**钉死**：`<local-rq009-venv>/bin/python`（系统 python3 缺 pytest/pandas，会把基线判错）

---

## 零、先读这一节：这项工作在整个研究里的位置

最终用途是 **online verification**：判断当前自动驾驶车辆（AV）的 IPV
（Interaction Preference Value，社会互动倾向的标量参数）是否符合人类的分布。
这个判断有两个弃权（abstain）机制，**本轮只做第一个**：

1. **IPV 没估出来** —— 就是本轮要冻结的这道门
2. 当前场景收集到的人类 IPV 不足以判定 AV 是否偏离 —— 属 RQ009 envelope 的样本量条件，**本轮不做，不要设计**

下游是 RQ009 的上下文条件 conformal envelope（已 accepted）。
PI 已定：**envelope 按场景上下文分格，但不按数据源（source）拆分**——
社会倾向 IPV 应当是人类群体的固有属性，按录制来源拆分等于预设不同来源的人有不同倾向，不合理。

PI 同时明确：**这两个弃权机制在论文里只要有即可，不做重点，设计上不必苛求细节。**
按这个尺度执行，**不要做成一个方法学专题**。本轮是诊断性/描述性产出，一轮出报告即结束。

---

## 一、门的规格（已由 PI 与监督方定稿，**不得改动、不得"优化"**）

```
判据 1：spread(mse_per_candidate[7]) == 0   → 弃权
判据 2：max(w_log[7]) < 0.20                → 弃权
其余                                        → 可估，报 ipv_log
```

**必须在 log 域算。** 连乘域会下溢；下溢时既有代码回退均匀权重，
会把本来可估的行错判为不可估——门自己就被污染了。
现成实现：`src/sociality_estimation/core/reliability_logdomain.py`
（可读，**不得修改**；相关符号：`candidate_mse`、`weights_from_mse`、`estimate_reliability`、
`STATUS_OK`、`STATUS_FLAT_LIKELIHOOD`、`K_EFF_FLAT_RATIO`、`SCHEMA_VERSION`）。

### 已被独立复审证实的两条事实，**直接引用，不要重新论证、不要试图推翻**

**(a) 判据 1 在样本上被判据 2 完全包含。**
400 个 `spread(mse)==0` 的行，其 `max(w_log)` 全部**恰好等于 1/7 = 0.142857143`**，
一律满足 `max(w_log) < 0.20`。判据 1 的额外筛出量是 **0 行**。

因此判据 1 **保留为语义标签，不是额外的筛选条件**。线上统计必须用**互斥 reason**：
先判 `spread(mse)==0`（`reason_code = NO_IPV_EFFECT`），
否则再判 `max(w_log)<0.20`（`reason_code = NEAR_UNIFORM`）。
**不得把两条写成"各自贡献"并相加，那会重复计数。**

**(b) θ = 0.20 是一个简单、可解释的政策阈值，不是从直方图的自然断点推出的。**
`max(w_log)` 在 0.20 附近**没有空隙**：`[0.18,0.20)` 95 行、`[0.20,0.22)` 71 行、`[0.22,0.24)` 74 行。
且 `k_eff → max(w)` **不是唯一换算**：同"一大六等"假设下 k_eff=6.75 给 max(w)=0.2102，
而 max(w)=0.2027 给 k_eff=6.801。
报告中 θ 必须写成**政策阈值**，并附一行敏感性：
θ=0.18/0.20/0.22 → 样本内门后 1,112 / 1,017 / 946 行。
**不要改 θ，不要做阈值扫描，不要提替代判据。门已定稿。**

判据 1 的机制依据（复审已核，直接引用）：判据 1 精确、无参数。七个候选给出逐位相同的 MSE，
说明 IPV 对前向模型没有影响（无实质交互时目标退化为 `cos(ipv)·interior + 常数`，正标量不改极小点）。

---

## 二、一个已查明的硬约束，决定了本轮**不做普查**

监督方已核查全语料台账 schema（复审已复核一致）：

```
reports/studies/RQ015A_ipv_estimability_labelling/
  RQ015A_1_concentration_audit_20260731T093746Z_e82091ce/concentration_ledger/
  共 4 份 parquet，合计 14,473,982 行
  列：artifact_id, product_row_key, measurement_role, case_id, rq007_split,
      ipv_error, K, candidate_grid_id, k_eff, q_eff, attempt_status,
      reason_code, recoverability, ledger_schema_version,
      aggregation_perspective, aggregation_configuration
```

**台账没有 `mse_per_candidate[7]`，也没有 `w_log[7]`。** 且台账里的 `k_eff` 是**连乘域**派生的
（`reports/plans/RQ015A_ledger_schema_v4_20260731.json` 写明 `k_eff = 1.0 / (1.0 - ipv_error) ** 2`），
不是从 log 域权重列派生；下溢行在那里一律显示为 `k_eff = 7`，正是本门要救回的那批。

**所以这道门无法在现有台账上直接普查。** 真要普查必须重算全语料，
而**全量重跑未获授权，本轮不做，也不要提议做**。

因此本轮产出是**规格 + 设计基估计（design-based estimate）**，不是普查。
**这一点必须写在报告标题里，并在每一处引用全域数字时重复写明。**

---

## 三、你要做的三件事

### 第 1 件 —— 把门写成可执行、可移植的规格

产出一段**独立的、不依赖本仓库上下文**的判据说明（伪代码 + 输入契约 + 输出契约），
使它既能用于离线建 envelope，也能在 online 推理时逐帧调用。要点：

- 输入只允许**单次运行内可得的量**（在线场景没有第二次运行可比，不得引用任何跨运行/跨环境量）
- 明确 `w_log` 的定义与归一化方式（log 域 softmax / log-sum-exp 稳定化），以及与 `k_eff = 1/sum(w²)` 的换算关系
  ——并明确写出**这个换算不是一一对应的**（见 §一(b)）
- 明确弃权时返回什么：**不得返回 `ipv = 0`**——那正是本项目要消除的混淆
  （`ipv=0` 是一个合法的社会倾向取值，不能用来表示"没估出来"）
- 写明两条判据各自的触发计数如何记录（互斥 reason），便于线上统计弃权率

**输出契约必须机器可读**，至少含：
`status ∈ {OK, ABSTAIN}`、`ipv_log`（弃权时为 **null**，不得为 0、不得为 NaN、不得缺列）、
`reason_code`（互斥主因：`NO_IPV_EFFECT` / `NEAR_UNIFORM`；`status=OK` 时为 null）、
`max_w_log`、`mse_spread`、`k_eff_log`、`candidate_grid_id`、`K`、`frame_id`。
RQ009 接入还需两个字段：**门通过率**与 **context cell key**。
请给出字段级的类型与空值规则表，以及一个 OK 样例和一个 ABSTAIN 样例（JSON）。

### 第 2 件 —— 用设计权重估计全语料的可估率

2,300 锚点是**有已知抽取概率的分层概率样本**，直接复用现成机制，**不要另造**：

```
锚点与门判据列：.codex-fleet/rq015b-repair/work/anchor_mse.csv（36 列）
    关键列：mse_per_candidate[7]、w_log[7]、k_eff_log、ipv_log、at_grid_boundary、
            source、n_band、signature、n_obs、scene_unique_id、anchor_id、
            legacy_fallback_triggered
设计权重：  .codex-fleet/rq015b-repair/work/mechanism_split.csv（13 列）
    关键列：anchor_id、scene_unique_id、source、n_band、signature、ht_weight、zero_postwarm_scope
    按 anchor_id join
bootstrap： B=2000、seed=20260731（与 D0–D4 及 I 轨一致，**不得改**）
```

**⚠ 分母陷阱（I 轨已踩过，不要重犯）**：`zero_postwarm_scope == True` 等价于
`signature ∈ {U, Z}`（复审已核：`N|False` 500、`U|True` 1,200、`Z|True` 600，不一致行 0），
**会把 signature N 整层排除**。而 N 层恰恰是关键层。
本轮必须用 **I 轨已确立的全域分母口径**：覆盖全部 2,300 个锚点，
HT 分母 **2,646,058**，cluster 为全部 **1,909** 个 `scene_unique_id`，B=2000、seed=20260731。
（491 是 signature N **层内**的 cluster 数，只在报 N 层单层估计时使用；
全域 bootstrap 重抽的是 1,909 个 cluster。）
**不得沿用 D0–D4 的 `zero_postwarm_scope` 分母。**

**复审已算出下列数，你的任务是复现并解释，不是重新发明：**

```
全域分母             2,646,058
门后保留权重         1,885,831.096
全域可估率           71.2695%
cluster bootstrap CI [67.1729%, 75.2135%]
互斥 reason 的全域权重占比：NO_IPV_EFFECT 0.5095%；NEAR_UNIFORM（非 spread==0 部分）28.2210%
样本内未加权门后行数 1,017 / 2,300 = 44.2174%
```

**若你的复算与上述任一数字不一致，不要静默改写，也不要静默采纳：**
在报告里单列一小节写明"复审值 / 我的复算值 / 差异原因"，并把两者都给出。

**这个数与样本内的 44.2174% 差距很大，必须解释清楚差在哪。** 监督方已复算出原因：

| signature | 样本内可估率 | 该层占全域权重 | 层内加权可估率 |
|---|---|---|---|
| N | 363/500 = 72.6% | **79.8%** | 76.3% |
| U | 511/1200 = 42.6% | 9.4% | 76.0% |
| Z | 143/600 = 23.8% | 10.8% | 29.9% |

样本按配额把 U 抽成了 1,200 行（占样本 52%），而 U 在全域只占 **9.4%** 的权重；
全域实际由 N 主导（79.8%），而 N 是高可估的。**所以样本比例严重低估了全域可估率。**
请复现这张表（含每层的 HT 分母与保留权重绝对值），并用它把 44.2174% → 71.2695% 的落差讲清楚。

**交付要求：主文一律用 HT 比率与 CI；样本未加权比例只能出现在方法附注里，
并当场注明"这是样本内比例，不是全域影响"。**

### 第 3 件 —— 门后分布的形状

**⚠ 原计划的"按 RQ009 实际用的上下文变量分格"已被独立复审证实无法执行，现已取消，不要做。**
理由（直接引用，不要重新调查）：RQ009 的 context-only 变量共 22 个数值 + 7 个类别
（`relative_distance_anchor`、`relative_speed_anchor`、`closing_rate_anchor`、
`geometry_path_category`、`priority_role` 等，见 RQ009 `calibration.py` 的
`BASE_NUMERIC_CONTEXT` / `BASE_CATEGORICAL_CONTEXT`），
而 `anchor_mse.csv` 与 `mechanism_split.csv` 中**这 29 个变量一个都没有**。
临时去 join RQ009 feature matrix 会撞上 RQ007 边界——feature partitions 不带 `rq007_split`，
本计划没有定义安全的 join 路径。**不要临时发明 join 口径，不要去读 RQ009 feature matrix。**

改为：**只用锚点自带变量报门后分布形状**，即 `signature`、`n_band`、`n_obs` 分箱。
每格报：样本数、HT 加权可估率、门后 `ipv_log` 的 p5 / 中位 / p95、边界占比。
`n_obs` 分箱口径由你选定，但必须在表下写明箱边界与每箱行数。

**边界口径必须分开报两个数，不得混用**（复审发现监督方此前混了）：
- 口径 A —— `at_grid_boundary` 列：门后 nuplan **25.32%**（79/312）、waymo **56.45%**（398/705）
- 口径 B —— `|ipv_log|` 精确命中 `3π/8`（1e-9 容差）：nuplan **0.96%**（3/312）、waymo **20.28%**（143/705）

监督方此前对外只报了口径 B（说成"边界饱和 1%/20%"），**低估了边界问题**。
两个口径都要报，并写明各自定义。

**不得按 source 拆分输出**（PI 已定 envelope 不分源）。source 只作为内部检查，见下。

### 第 3.5 件 —— source 内部检查（一小节，不展开）

监督方复算的分带表如下，复审已逐格核对一致。请复现确认：

| 权重带（`max(w_log)`） | nuplan \|ipv\| 均值 | waymo \|ipv\| 均值 |
|---|---|---|
| 0.20–0.25 | 0.1306 (n=81) | 0.0507 (n=86) |
| 0.25–0.35 | 0.1777 (n=83) | 0.2001 (n=96) |
| 0.35–0.50 | 0.2734 (n=68) | 0.3560 (n=77) |
| 0.50–0.75 | 0.4383 (n=41) | 0.5706 (n=94) |
| 0.75–1.01 | 0.7551 (n=39) | 0.9698 (n=352) |

筛选：`spread(mse)!=0 and max(w_log)>=0.20`；分箱左闭右开，末箱 `[0.75,1.01)`。

**结论措辞已按复审改弱，必须照此写：**
不得写"各带内基本重合、差异几乎全部来自选择效应"——带内仍有可见残差
（最高带中位数 nuplan 0.0639 vs waymo 0.8005），且按 `at_grid_boundary` 口径 waymo 在各带内持续更高。
正确写法：

> 最高权重带的样本量不平衡（waymo 352 vs nuplan 39）解释了汇总均值差异的相当一部分；
> 分带后仍存在残余差异。该残差不作为下游分源口径的依据，仅作内部风险披露。

### 第 3.6 件 —— "可估交互条件下"这一限定（**已按复审改弱**）

门后保留的样本，`|ipv|` 随集中度单调上升，因此
**envelope 是"可估交互条件下人类的 IPV 分布"，不是"人类的 IPV 分布"**，论文措辞必须带这个限定。

**但不得写成"AV 侧用同一把尺子，偏移两边抵消"——复审已指出该论证不成立。**
同一道门只保证两边都被条件化到"通过门"的子总体，不保证偏移抵消；
若 AV 的 `max(w_log)` 分布、边界占比或 IPV 形状与人类不同，门会改变被比较的对象。
正确写法：

> 结论只对"同一道门通过后的条件分布"成立。若 AV 的通过机制与人类不同，
> 须分别报告 AV 与人类的门通过率，并把"未通过"本身作为监控结果的一部分。

---

## 四、明确不做的事（违反即为跑偏）

- **不重算任何锚点，不提交任何 HPC 作业，不提议全量重跑**
- **不做跨环境（Mac vs HPC）可复现性分析**——PI 已明示这条不再细究
- 不改 `agent.py` / `ipv_estimation.py` / `process_interhub.py` / `reliability_logdomain.py`
- 不调整 θ、不做阈值敏感性扫描（§一(b) 那一行三点敏感性除外，且只是如实标注）、不提替代判据
- 不设计第二个弃权机制（envelope 样本量条件），那是 RQ009 的事
- 不写「规格 v2」、不做盲审、不做多路复审、**不提交 git commit**
- **禁止** `git checkout -- .` / `git restore .` / `git stash` / `git reset --hard` / `git clean -fd`；
  禁止 checkout 历史提交到主工作区。工作区非空是**预期状态**（此前轨道留下的文件仍在），
  你只对自己创建的文件负责，不要清理别人的文件，不要看全仓库 `git status` 来判断清洁性
- **不要对 `reports/` 做全仓库 `rg`**——宽泛检索会把 RQ003
  `12_blind_annotation/controlled_identity_map.csv` 的 controlled-access 行整行拉进上下文。
  要读 `reports/` 下的文件请用精确路径

### 四条硬约束（与流程无关，不得放松）

1. **RQ007 `held_out` 不得被解析**（污染不可恢复）——本轮所有输入都不含 held_out，不要去找
2. **RQ014 致盲相关的评分字段不得读取**
3. 不得静默覆盖冻结产物或已接受的 `decision.md`
4. **描述性结果不得写成因果主张**

### 术语禁令

全文**禁用** `estimability` 一词与"测出 / 未测出 IPV"这类说法
（引用文件路径时原样保留路径字符串即可，不算违规）。
可辩护的表述是：**权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息**。
统一用"可估 / 弃权（abstain）"。

---

## 五、产出

**报告路径**：`.codex-fleet/rq015j-gate-spec/board/reports/J1_gate_spec_and_impact.md`
**标题必须写明这是 design-based estimate，不是普查。**

计算脚本与证据 JSON 放 `.codex-fleet/rq015j-gate-spec/work/`（自行命名，建议 `j1_*.py` / `j1_*.json`）。

报告须含以下 7 节：

1. **门的可执行规格**（伪代码 + 输入契约 + 机器可读输出契约 + JSON 样例）
2. **全域可估率的设计基估计**（点估计 + CI，**标题与每处引用都要写明 design-based，不是普查**），
   两条判据按互斥 reason 各自贡献
3. **按锚点自带变量（signature / n_band / n_obs）分格的表**（**不分源**），
   每格含 HT 加权可估率与门后 `ipv_log` 分位数；边界占比**两个口径都报**
4. **source 内部检查一小节**（复现分带表，按改弱后的措辞写结论）
5. **"可估交互条件下"这一限定的一段说明**（按改弱后的措辞写）
6. **明确写出：要把这道门真正应用到全语料，需要什么**
   （台账缺 log 域权重与逐候选 MSE，须重算；给出**量级估计**供 PI 决策——
   例如需要重算多少行、哪些产物、大致算力量级。只给量级，不写作业脚本，不提交作业）
7. **可复算性附录**：每个数字写明来源文件、列名、筛选条件、权重、分母

**每个数字必须可被读者自行复算**：给出分子、分母、筛选条件、来源文件与列名。
一个读者无法自行复算的数字，等于没给。

## 六、汇报口径（硬性要求）

报告开头必须先交代三件事：这项工作要解决什么问题、整体已经走到哪一步、本次是其中哪一环。
不得直接从增量讲起。不用黑话、不用比喻；必须使用项目专有名词时，当场用一句话说明它是什么。
需要 PI 拍板的事单独成节（写清选项、判断依据、不做的后果），不得混在叙述里当陈述句带过。

## 七、完成标准

报告写完后，自己过一遍这个清单，并在报告末尾附上自查结果：

- [ ] 7 节齐全
- [ ] 标题与每处全域数字都标注了 design-based estimate（不是普查）
- [ ] 两条判据按互斥 reason 报，没有相加重复计数
- [ ] 全域分母用的是 2,646,058 / 1,909 cluster，不是 `zero_postwarm_scope` 分母
- [ ] 边界占比两个口径都报且各自写明定义
- [ ] 未按 source 拆分输出（source 只在内部检查小节出现）
- [ ] θ 写成政策阈值，不是自然断点
- [ ] 弃权时 `ipv_log` 为 null，不是 0
- [ ] 全文无 `estimability`、无"测出/未测出 IPV"
- [ ] 没有 git commit、没有改受保护文件、没有提交 HPC 作业
- [ ] 复审已给的数字都复现了；不一致处已单列说明

完成后在报告末尾写一行 `J1_DONE <UTC 时间戳>`（时间戳用 `date -u +%Y-%m-%dT%H:%M:%SZ` 实取，不要估计）。
