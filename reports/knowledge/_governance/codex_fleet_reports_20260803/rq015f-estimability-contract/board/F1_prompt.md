# F1 — RQ007 已接受契约在下溢发现之后的重核（join + 统计，只读）

你是 track F 的唯一执行 agent。本任务是**诊断性/描述性**的：一轮做完，出报告，结束。
**不重解任何轨迹**，不训练，不改任何已有产物。全部是读文件 + join + 统计。

工作目录 = 仓库根 `.`
解释器**必须**用 `<local-rq009-venv>/bin/python`（系统 python3 缺包）。
中间产物写到 `.codex-fleet/rq015f-estimability-contract/work/`，
报告写到 `.codex-fleet/rq015f-estimability-contract/board/reports/F1_contract_recheck.md`，
机器可读数字同时写到 `.codex-fleet/rq015f-estimability-contract/work/f1_results.json`。
你写的脚本放 `.codex-fleet/rq015f-estimability-contract/work/`，不要放仓库其他位置。

---

## 0. Leader 已完成的侦察 —— 这些是既定事实，**核验一次即可，不要重新发现**

侦察已花掉本轮相当一部分预算。以下四条都已用实数据确认过，你**只需各用一两行代码复核**，
然后把预算全部投到 Q1–Q5 的统计上。**不要再花 exec 去摸索仓库结构。**

**事实 1 —— RQ007 的集中度指数 c1/c2 就是估计器的 `ipv_key_agent_N_error` 字段。**
在 `ipv_000001` 的 frame 4/5/20 上，两个 agent 共 6 个值与下面"事实 2"的文件 12 位有效数字完全相同。
含义：RQ007 的因变量不是新造的指标，就是估计器权重弥散字段本身。
→ 复核方式：随机抽 200 个 (scene, frame)，比对 `c1 == ipv_key_agent_1_error`、`c2 == ipv_key_agent_2_error`，
报出最大绝对偏差与完全相等的比例。

**事实 2 —— RQ007 的主输入是 20260612 那份，不是 RQ009 的 hw4 重建。**
```
RQ007 主输入（c1/c2 与 ipv 都对得上）：
  data/derived/interhub/20260612_sigma_0_1_full_rerun/00_hpc_outputs/sigma01_ipv_timeseries.csv   (2.21 GB)
RQ009/RQ015A 用的是另一份（20260626 hw4 重建，frame 5 起就与 RQ007 对不上）：
  data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/
  RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/target_hw4/sigma01_hw4_ipv_timeseries.csv
```
两份的 `scene_unique_id|frame_index` 键相同但**数值不同**（frame 4 恰好相同，frame 5/20 不同）。
**Q1/Q2 一律用 20260612 那份**（与 RQ007 同源，才能复现它的已接受数字）。
**Q3 必须把这个版本差当作首要候选解释**，不要预设 c 与 q_eff 的差异来自定义不同。

**事实 3 —— held_out 在两个主要文件里根本不存在，这是结构性的，不是承诺。**
- `replication_frame_metrics.csv`：2,490,992 行，split 只有 `development`(1,788,593 行 / 19,258 scene)
  与 `guard`(702,399 行 / 7,628 scene)，**没有第三个取值**。19,258 + 7,628 = 26,886 = RQ007 封条里的
  Development + guard，sealed 11,342 不在此文件内。
- RQ015A ledger `interhub_sigma01_hw4_timeseries.parquet`：5,197,072 行
  = (2,490,992 有效帧 + 107,544 warmup 帧) × 2 agent。**同样不含 held_out。**
  → 你要用 `rq007_split` 的 value_counts 把这条独立验一遍。
- **唯一含 held_out 的文件是那两个 2.2 GB 的 `sigma01*_ipv_timeseries.csv`**（含全部 3,695,981 primary 行）。

**事实 4 —— 下溢在 RQ007 因变量里的占比已经量出来了（这是你的复现靶子）。**
下溢常数 `U = 0.6220355269907728`（= 1 − 1/√7）。在 `replication_frame_metrics.csv` 上按 `|c − U| ≤ 1e-9`：

| slot | cv_cpa_conflict | n | n_underflow | pct |
|---|---|---|---|---|
| c1 | 0 | 2,391,270 | 285,605 | 11.944% |
| c1 | 1 | 99,722 | 8,220 | 8.243% |
| c2 | 0 | 2,391,270 | 380,699 | 15.920% |
| c2 | 1 | 99,722 | 11,396 | 11.428% |
| **pooled** | **0** | **4,782,540** | **666,304** | **13.932%** |
| **pooled** | **1** | **199,444** | **19,616** | **9.835%** |

帧加权原始均值：`mean_c(conf=0) = 0.46487`，`mean_c(conf=1) = 0.29379`，**帧加权 gap = −0.17108**。
（RQ007 C1 的头条 −0.132/−0.129 是 **case 加权**，与这个帧加权数不同是预期的 —— 见 Q2。）

**你的第一个动作就是复现上表。** 对不上就停下来先查为什么，别往下做。

---

## 1. 硬边界（违反即本轮作废）

```
□ held_out 绝不解析。对两个 2.2 GB 的 timeseries csv，必须【先建 26,886 个 scene 的白名单】
  （白名单来自 replication_frame_metrics.csv 的 scene_unique_id 去重），
  分块读时【在触碰任何 ipv/error 列之前】先按白名单丢弃非白名单行，
  并累计报出 dropped_rows_never_parsed。结项必须给出 held_out_parsed_rows = 0
  加上【结构佐证】：三个文件各自的 split 计数与求和恒等式。
□ 不得修改 RQ007 任何冻结产物、decision.md，不得写入 RQ015A/RQ015B/RQ015D 的 run 目录。
  你只写 .codex-fleet/rq015f-estimability-contract/ 下面的东西。
□ 不得重解任何轨迹。需要修正权重就用 anchor_mse.csv 里现成的 w_log[7]。
□ 不读取 RQ014 致盲相关评分字段。不要对 reports/ 做全仓库 rg（会拉进 controlled-access 行）。
□ 禁止 git commit / git checkout -- . / git restore / git stash / git reset --hard / git clean。
  另有两条 track 的 agent 在同一仓库工作，工作区非空是预期状态。
□ 描述性结果不得写成因果主张。
□ 用词：描述 RQ015 侧的新结果时【禁用】"estimability"，【禁用】"测出/未测出 IPV"。
  引用 RQ007 已接受主张的原文时可以保留 estimability（那是它的正式术语），需加引号标明是引用。
  可辩护表述：【权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息】。
□ 【只给证据，不给建议】。每条主张只输出三选一判定 + 支撑数字。
  "要不要改手稿 / 要不要撤回"是 PI 的决定，报告里不得出现任何"建议撤回/建议改写"式措辞。
```

---

## 2. 数据清单（直接用，不要再找）

```
A  RQ007 逐帧指标（2,490,992 行，dev+guard，无 held_out）
   data/derived/interhub/RQ007_interaction_conditioned_ipv_estimability/
   RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/01_intermediate/replication_frame_metrics.csv
   列：scene_unique_id, split, frame_index, cv_cpa_conflict, c_pair, c1, c2, low1, low2, g1, g2
   （low1/low2/g1/g2 语义未知，用到前先在 RQ007 报告里确认，确认不了就不要用）

B  RQ007 的 C3 原始产物（53,772 行 = case × agent_role）—— C3 的 0.26 rad / 22% 就出自这里
   data/derived/interhub/RQ007_interaction_conditioned_ipv_estimability/
   RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_outputs/summary_sensitivity/
   per_case_summary_sensitivity.csv
   列：case_id, source_row, split, agent_role, agent_id, n_valid_frames, n_active_frames,
       active_frame_share, all_valid_mean, interaction_active_mean, estimability_weighted_mean,
       mean_concentration_index, estimability_weight_sum, mean_abs_frame_to_frame_ipv_change,
       max_abs_pairwise_delta, most_sensitive_rule_pair, opportunity_mask_row_count,
       opportunity_mask_active_count

C  RQ007 主输入 IPV 时序（2.21 GB，【含 held_out，必须先白名单过滤】）
   data/derived/interhub/20260612_sigma_0_1_full_rerun/00_hpc_outputs/sigma01_ipv_timeseries.csv
   需要的列只有：scene_unique_id, frame_index,
                 ipv_key_agent_1, ipv_key_agent_1_error, ipv_key_agent_2, ipv_key_agent_2_error
   （usecols 只取这 6 列，别整表载入）

D  RQ015A 台账（5,197,072 行，dev+guard，无 held_out）
   reports/studies/RQ015A_ipv_estimability_labelling/RQ015A_1_concentration_audit_20260731T093746Z_e82091ce/concentration_ledger/interhub_sigma01_hw4_timeseries.parquet
   列：product_row_key ("scene_unique_id=ipv_000001|frame_index=0"), measurement_role ("agent_1"/"agent_2"),
       case_id, rq007_split, ipv_error, K, k_eff, q_eff, attempt_status, reason_code, recoverability

E  RQ015B 抽样重解（2,300 锚点）
   .codex-fleet/rq015b-repair/work/anchor_mse.csv
   列含 scene_unique_id, split, frame_index, agent_slot, mse_per_candidate[7](JSON 串),
        w_legacy[7], w_log[7], legacy_fallback_triggered, partial_underflow, min_mse, min_rms,
        k_eff_legacy, k_eff_log, at_grid_boundary

F  RQ015D 报告（400 行退化锚点的既有结论，读来对齐口径，不要重算它的结论）
   .codex-fleet/rq015d-sigma-rederive/board/reports/D1_sigma_report.md
   .codex-fleet/rq015d-sigma-rederive/work/d1_sigma_stats.json

G  RQ007 决策原文（引用 C1/C2/C3 措辞用）
   reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/decision.md
   reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/synthesis.md

H  RQ007 自己做过的 low-information 红队（先读，避免重复它）
   .../01_intermediate/red_team_low_information_case_probe_summary.csv
```

join 键：`scene_unique_id + frame_index + agent_slot`，其中 c1↔agent_slot 1↔`measurement_role=agent_1`
↔`ipv_key_agent_1*`；c2 同理。

**两类特殊行必须分开计数，不要混为一谈：**
- 下溢行：`|ipv_error − 0.6220355269907728| ≤ 1e-9` 且 `ipv == 0.0`
- warmup / NOT_ATTEMPTED 行：`ipv_error == 1.0` 且 `ipv == 0.0`（台账 D 里 `attempt_status=NOT_ATTEMPTED`,
  `reason_code=D0_WARMUP`）。这批**不在**文件 A 的 2,490,992 行里（A 已剔除 warmup），
  但**在**台账 D 的 5,197,072 行里。任何比例都要写清分母含不含它。

---

## 3. 五问（按序做；Q1、Q2 最重，做不完也要保证这两问完整）

### Q1 —— C3（≈0.26 rad / ≈22% 变号）是不是下溢伪影？【最高优先级】

**必须三步走，不能跳过 Q1b。** 跳过 Q1b 就无法区分"结论塌了"和"你的重建口径不对"。

**Q1a 基线直读**：直接读文件 B，复现 RQ007 的 0.26 rad 与 22%。
写清楚你用的筛选（哪些行参与、`interaction_active_mean` 为 nan 的怎么处理 ——
注意 `n_active_frames == 0` 的 case 该列是 nan，RQ007 大概率只在 `n_active_frames > 0` 上算）。
报出：参与 case-agent 数、`mean |all_valid_mean − interaction_active_mean|`、严格变号率
（`sign(a) != sign(b)`，两者都非零且非 nan）。**必须落在 0.26 / 22% 附近**，
偏差 > 15% 就说明筛选口径没找对，换一种筛选再试，把试过的口径都列出来。

**Q1b 重建校验**：从文件 C（白名单过滤后）+ 文件 A 的 `cv_cpa_conflict` 掩码，
**不做任何排除**地重算 `all_valid_mean` 与 `interaction_active_mean`，
与文件 B 的对应列逐 case-agent 比对。报出：完全匹配比例、最大绝对偏差、p99 偏差。
**若重建对不上基线**（例如"valid"的定义不是 A 里的行集），先把定义找对再往下 ——
在报告里写清最终采用的 valid / active 定义。

**Q1c 扣除下溢后重算**：在 Q1b 已校准的口径上，把下溢帧从 `all_valid` 与 `interaction_active`
两个集合里都剔除，重算两个均值，再算：
- `mean |Δ|` 塌到多少（原 ≈0.26）
- 严格变号率塌到多少（原 ≈22%）
- 各自的相对塌缩幅度（%）
- 剔除后 `n_active_frames` 或 `n_valid_frames` 归零而**整个 case-agent 掉出统计**的数量与占比
  （这本身就是一个结果：如果大量 case 是靠下溢帧才有"有效帧"的，说明 C3 的分母有问题）

**再补两个分解数字**（决定"伪影"这个词站不站得住）：
- `all_valid` 帧里下溢占比 vs `interaction_active` 帧里下溢占比
- 只把下溢帧的 ipv 从"精确 0.0"换成 NaN 剔除后，均值差里有多少是纯粹由"0 在网格中央"造成的：
  给出对照 —— 把下溢帧的 ipv 用该 case-agent 的非下溢帧均值替代（而非剔除），再看 mean|Δ| 与变号率。
  两种处理都报，说明它们的差异。

### Q2 —— C1 的 −0.13 gap 偏了多少、往哪偏？

**Q2a 先定方向（已有初值，你要复核并补 case 加权版）**：
下溢率 conflict=0 为 13.932%、conflict=1 为 9.835%（帧加权，事实 4）。
下溢把指数钉在 `U = 0.622`，这是**最不集中**的一端，即抬高该组均值。
下溢在**窗外更常见** ⇒ 抬高 `mean_c(conf=0)` ⇒ 使 `gap = mean_c(1) − mean_c(0)` **比真值更负**。
→ 即 RQ007 的 −0.13 在幅度上**被高估**。**这个方向判断你必须用数据独立复核一遍**，
并同时给出 **case 加权**下的下溢率差（RQ007 头条是 case 加权，见文件 H 的 `case_weighted_mean` 列）。

**Q2b 复现基线**：用文件 A 复现 RQ007 的 case 加权 gap，dev 与 guard 分开报。
靶子：dev ≈ −0.132、guard ≈ −0.129（文件 H 的 `all_rebuilt` 行给的是
dev case_weighted −0.13418 / frame_weighted −0.13138，guard −0.13387 / −0.13031，
它是 red-team 重建口径，与头条略有出入，两个都可以当靶子，**报告里写明你对齐的是哪一个**）。

**Q2c 扣除下溢后的 gap**：剔除下溢帧后重算同一个 case 加权 gap，
给 **case-clustered bootstrap CI**（按 `case_id` 整簇重抽，≥1000 次，报 2.5%/97.5% 分位）。
dev / guard 分开报，并报 `Δgap = gap_excl − gap_baseline` 及其符号是否与 Q2a 的方向预测一致。

**Q2d 敏感性**：再报一个"只剔除 conflict=0 侧下溢"和"只剔除 conflict=1 侧下溢"的假想对照，
用来量化两侧各自贡献了多少偏差。这是描述性分解，不要写成因果。

### Q3 —— c1/c2 与 q_eff 是不是同一个量？

**注意事实 2：两者来自不同数据版本（20260612 vs 20260626 hw4）。**
所以要把"定义是否同族"和"数值是否同源"**拆开**回答：

- **定义层**：在**同一份数据**上比。台账 D 的 `ipv_error` 与它自己的 `q_eff`/`k_eff` 是同一版本，
  先在 D 内部求出 `q_eff = f(ipv_error)` 的解析关系（D1 已确认 `k_eff` 是 inverse-Simpson；
  试 `ipv_error = 1 − 1/√k_eff` 之类的闭式，报拟合残差）。若能写成闭式，
  则 c（= RQ007 的 ipv_error）与 q_eff 就是同一个量的单调重参数化，**RQ007 与 RQ015A 互证**。
- **数值层**：把 A 的 c1/c2 与 D 的 `ipv_error` 按键 join，报完全相等比例、
  最大偏差、相关系数（Pearson + Spearman）。若相等比例低而相关高 ⇒ 差异是**数据版本**，不是定义。
  给出两版本差异随 frame_index 的分布（frame 4 相同、frame 5 起分叉，说明分叉从哪开始）。
- 明确结论：**同一个量 / 同族但不同版本 / 不同量**，三选一，给支撑数字。

### Q4 —— PI 那个解释的直接检验

- **Q4a**：近均匀占比按 `cv_cpa_conflict` 分组。"近均匀"给两个口径都报：
  (i) 严格下溢 `|c − U| ≤ 1e-9`；(ii) 阈值式 `c ≥ U − ε`，ε 取 0.01 与 0.05。
  分 dev/guard、分 slot、也给 pooled。
- **Q4b**：从文件 E 解析 `mse_per_candidate[7]`，算 `spread = max − min`，取 `spread == 0`
  （精确逐位相同）的锚点集合，核对是否为 400 行（D1 的既有结论）。
  然后按 `scene_unique_id + frame_index + agent_slot` join 到文件 A 取 `cv_cpa_conflict`，
  出 **2×2 列联表**（spread==0 与否 × conflict 0/1），报：
  - 这些退化锚点落在 `cv_cpa_conflict = 0` 的比例
  - 反向条件概率：`P(spread==0 | conflict=0)` 与 `P(spread==0 | conflict=1)`
  - join 不上的锚点数（可能因为版本/warmup），单独列出，**不要静默丢弃**
  - Fisher 精确检验或卡方（描述性关联，**不得写成因果**）
  若高度吻合，可陈述：`spread(mse)==0` 是一个**无需阈值的精确判据**，
  用来标记"该帧的 7 个候选给出逐位相同的 MSE，因而该 IPV 数值不携带候选间的判别信息"。
  注意 D1 已报 waymo 0 行、nuplan 34.78% —— 报告里要说明这 400 行的 source 构成，
  以免把数据集差异误读成 conflict 效应。

### Q5 —— 分母重报

RQ015A 现有比例是对**全部 ATTEMPTED 行**报的。按 `cv_cpa_conflict` 分组重报，
分母写清楚（ATTEMPTED / 全部含 warmup / A 的 2,490,992 有效帧，三种分母都列一列）：
- **在交互机会窗内（conflict=1），近均匀占比是多少**；窗外是多少；
- 同时给 `attempt_status` 各取值在两组里的分布。
台账 D 无 `cv_cpa_conflict`，需按键 join 文件 A 才能分组；join 不上的行（warmup 等）单列一行说明。

---

## 4. 报告要求

`board/reports/F1_contract_recheck.md`，中文，结构：

1. **结论表（放最前面）** —— 三行，每行三选一 + 一句话 + 关键数字：

   | 主张 | 判定（扛得住 / 需重述 / 需重跑） | 关键数字 |
   |---|---|---|
   | RQ007-KC-C1 | | |
   | RQ007-KC-C2 | | |
   | RQ007-KC-C3 | | |

   C2 本轮没有专门实验，若证据不足以判定就写"本轮未取证"，**不要硬凑**。
   判定标准（自己在报告里写明你采用的阈值）建议：
   - 扛得住 = 扣除下溢后效应量变化 < 20% 且方向不变、结论文字无需改动
   - 需重述 = 方向不变但数字/边界需要改（给出改后的数字）
   - 需重跑 = 效应量塌缩过半、或方向翻转、或分母被破坏

2. **held_out 合规节** —— `held_out_parsed_rows = 0` + 三个文件的 split 计数 + 求和恒等式
   + 2.2 GB 文件的 `dropped_rows_never_parsed` 计数。**给结构佐证，不要只写声明。**

3. Q1 / Q2 / Q3 / Q4 / Q5 各一节，每节：方法一段（含分母定义）→ 数字表 → 一句话判读。
   每个数字都标注**分母来自哪个文件**（A/B/C/D/E）。

4. **自查节**：你自己列出本轮结果里最可能被推翻的三点，各写一句为什么。
   包括但不限于：valid/active 定义是否真的复现了 RQ007、版本差是否污染了某个对比、
   bootstrap 是否真的按 case 整簇重抽。

5. **限制节**：明确写"本轮为描述性/敏感性分析，不含因果推断"。

同时把所有关键数字写进 `work/f1_results.json`，顶层键用 `q1`/`q2`/`q3`/`q4`/`q5`/`heldout`/`verdicts`。

## 5. 执行纪律

- 先跑 §0 事实 4 的复现（几分钟），**对上了再往下**。对不上立刻在报告开头写明并停止后续统计。
- 2.2 GB 文件只读 6 列、分块、先白名单过滤。不要整表载入。
- 每完成一问就把结果落盘到 `work/`（json 或 csv），避免最后一步失败丢掉全部工作。
- 遇到定义不明（如 low1/g1）就明说"未确认，未使用"，**不要猜**。
- 不要写规格文档、不要开第二轮、不要征求授权 —— 直接做完出报告。
