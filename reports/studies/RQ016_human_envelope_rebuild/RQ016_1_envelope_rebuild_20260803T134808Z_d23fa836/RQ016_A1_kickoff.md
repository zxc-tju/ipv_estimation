# RQ016-A1 任务书：用只过门的样本重建人类 envelope

你是本轮唯一的执行 agent。读完这份文件就开工，不要写第二版方案，不要开子轨。
本任务属**描述性产出**：跑出来 → 一轮自查 → 出报告 → 结束。不做盲审，不加授权闸门。

仓库根即当前工作目录。以下所有路径都相对仓库根。

---

## 0. 你要在哪个问题上工作（不要跳过这一节）

最终研究目标是**在线验证**：一辆自动驾驶车在路上跑，判断它表现出的社会交互倾向像不像人。
IPV（Interaction Preference Value）是一个标量，表示交互倾向。判定由**两道串联的弃权机制**构成：

- **机制一**：这一帧的 IPV 数值能不能估？若权重近均匀，则该数值不携带候选间的判别信息，
  **直接弃权，不进机制二**。这道门已由 RQ015 完成并冻结。
- **机制二**：当前场景收集到的人类样本，够不够判断这辆车是否偏离？这就是**人类 envelope**
  （RQ009 已 accepted 的 context-conditioned split-conformal 区间），落在区间内判支持，
  区间外判不支持，样本不足则弃权。

**本轮要解决的问题**：机制二当前依赖的人类 envelope，是建在一个**含伪零**的样本上的。
原估计器在数值下溢时退回"七个候选等权"，而候选网格对称，于是必然写出 `ipv` 恰为 `0`——
**"没估出来"与"该个体 IPV 恰为中性"在数据里长得一模一样**。RQ015 的 L1 已证实：
RQ009 打分目标的精确零点原子里，48.0223%（92,333/192,271，分母为零点原子中落在 K2 台账
覆盖域内的行）不是中性点值，而是弃权被写成了 0。

**所以本轮的唯一任务是**：用只过门的样本重建 envelope，然后报告机制二的覆盖行为、
区间宽度、以及两道门串联后的合并弃权率。

---

## 1. 交付物（就这一份，不要多产）

一份报告：`.codex-fleet/rq016-envelope-rebuild/board/reports/RQ016_1_envelope_rebuild.md`
一份机器可读数字：`.codex-fleet/rq016-envelope-rebuild/work/A1/key_numbers.json`
你写的脚本放 `.codex-fleet/rq016-envelope-rebuild/work/A1/`，可复跑。

报告要回答且**只**回答这四问：

1. 重建后，envelope 的 **coverage** 变成什么样？
2. 重建后，**区间宽度**变成什么样？
3. **机制二自身的弃权率**（样本不足/超出支持范围）变成什么样？
4. **两道门串联后的合并弃权率**是多少？——这是本轮最重要的一个数。
   定义：合并弃权 = 机制一弃权的行 + （过了机制一但被机制二判为超出支持而弃权的行）。
   必须给出分子、分母、以及两段各自的贡献。

---

## 2. 实验设计：必须做两臂，不能只做一臂

**这是本任务书最重要的一条，做错了整份报告作废。**

不能直接拿"重建后的数"去和 RQ009 已发表的数对比，因为那样会把**两个变化**混在一起：
(a) 去掉伪零；(b) 把域限制到 development+guard（原因见第 4 节的 held_out 红线）。
混在一起就无法归因，结论不成立。

所以在**同一个域**（台账覆盖域 = development + guard）上，用**同一套方法**跑两次：

| 臂 | 样本口径 | 作用 |
|---|---|---|
| **A 臂（域内基线）** | 台账覆盖域全部行，**保留**未过门行（即现状，含伪零） | 本轮 A/B 对比的基线 |
| **B 臂（处理组）** | 台账覆盖域中**只保留 `status == "OK"`** 的行 | 重建后的 envelope |

**唯一变量是样本口径。** context 分箱、alpha 层、calibration/test 结构、
conformal 计算方式在两臂之间必须完全相同。

RQ009 已发表的数（90% 名义水平下 coverage ≈ 0.899、宽度较全局基线 −42.3%、
Winkler −35.6%、弃权 4.78%，来源 `reports/knowledge/RQ009_dynamic_counterpart_conditioned_envelope/decision.md`
的 RQ009-KC-R3）**只能作为外部参照引用，并且必须同时写明它是在一个包含 held_out 的域上算的**。
**禁止**把本轮结果写成"复现了 RQ009"或"未能复现 RQ009"——域不同，不构成复现关系。

---

## 3. 用什么定义 context 分箱（已裁定，不要重新讨论）

**沿用 RQ009 的 context 定义，不要另造。** PI 已裁定：envelope 要分格，但**不应该分源**
（社交倾向的表达在人类群体中应有固有分布，按驾驶情境分箱，不按数据集分）。

监督方已查证并确认这与 RQ009 一致，你**不需要重查**，直接用：

- 特征字典：`reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/03_features/feature_dictionary.csv`
- 其中 `role == "feature"` 的 **38 个变量**就是 context 预测变量
- `source_dataset` 的 role 是 `key`，字典定义原文为
  "Dataset source label; reporting/LOSO/M5 only, excluded from main M3 predictors"
  ——**它本来就不在预测变量里**，保持这样，不要把它加进去

用的是 **M2（context-only）**，即 RQ009 已接受claim 里那一组数所对应的模型。
**不要用 M3/M4**：M3 额外加了 `counterpart_ipv_current`、`counterpart_ipv_error_current`、
`counterpart_ipv_slope_pre_anchor` 等由**旧估计器**算出来的输入，那些输入本身就带伪零，
用它们会引入第二条污染路径。RQ009 自己也记录 M3 ≈ M2（配对 90% Winkler 差 −0.0002，
case 聚类 p=0.863），所以用 M2 没有损失。

参考实现（读，不要改动原文件）：
- `.../02_process/04_calibration/calibration.py` 与 `conformal_radii.json`
- `.../02_process/05_evaluation/evaluate.py` 与 `metrics_summary.csv`

---

## 4. 红线：held_out 绝对不能被解析

这是全项目最硬的一条，污染不可恢复。

**陷阱在这里**：RQ009 自己的 fold 标签 `{train, guard_tune, calibration, test}` 与
RQ007 的三个 split `{development, guard, held_out}` **是正交的**——
**每个 fold 都含约 29% 的 held_out 行**。
**只按 fold 过滤会解析 1,899,898 行 held_out。**

**所以：**

1. 本轮**必须以 K2 台账为驱动**：`data/derived/rq015k_logdomain_gate/l1_v1/`。
   监督方已实测该台账**只含 development 与 guard**：RQ009 分区
   `development=6,459,684` + `guard=2,535,052` = `8,994,736` 行，held_out 行数为 0。
2. 若你需要用 RQ009 的 fold 结构来复刻 split-conformal 的 calibration/test 划分，
   **必须取 fold 与 `rq007_split ∈ {development, guard}` 的交集**，不得只按 fold 取。
3. 报告里必须有一行硬断言的实测结果：本轮所有参与计算的行中，
   `rq007_split` 取值不在 `{development, guard}` 的行数为 **0 行**（给出实测计数，不是声称）。
4. **不得打开任何受保护的 confirmation 划分文件**去确认 case 归属。前任监督方已裁定不授权，
   本轮沿用。

另外：**不得读取 RQ014 致盲相关的评分字段**。

---

## 5. 已核实的事实（直接用，不要重算后报一个略不同的数）

以下数字由监督方于 2026-08-03 在本机独立复算，源为
`data/derived/rq015k_logdomain_gate/l1_v1/`（510 个 parquet 分片，14,473,982 行）：

```
台账四个分区：interhub_sigma01_hw4_timeseries 5,197,072 / rq009_feature_matrix 8,994,736
             / onsite_dense_timeseries 281,268 / wod_rq010b_full479_audited 906

InterHub 求解单元 4,981,984（= source_attempt_status=ATTEMPTED = gate_applicable=true，
  canonical_key 去重后仍为 4,981,984）
  OK 3,502,340 (70.3001%) / NEAR_UNIFORM 1,457,746 (29.2604%)
  / NO_IPV_EFFECT 19,964 (0.4007%) / SOLVER_FAILURE 1,934 (0.0388%)

RQ009 台账行域 8,994,736 行：status=OK 6,405,292 = 71.2116%
  canonical_key 去重 8,994,736，重复 0，gate_applicable=false 行数 0
  机制一在该域丢掉 2,589,444/8,994,736 = 28.7884%
  measurement_role: target_future 4,497,368 + counterpart_current 4,497,368
```

门的冻结规格（**一个字不许改，不得调参，不得做阈值扫描**）：

```
log_score_i = -mse_i / (2 * sigma^2)      sigma = 0.1
w_log       = softmax(log_score)          用 log-sum-exp
mse_spread  = max(mse_per_candidate) - min(mse_per_candidate)

if 输入非有限 / 缺列 / 求解失败:  ENGINEERING_FAILURE (NON_FINITE_INPUT | SOLVER_FAILURE)
elif mse_spread == 0:             ABSTAIN, NO_IPV_EFFECT      # 精确浮点相等，不得用 np.isclose
elif max(w_log) < 0.20:           ABSTAIN, NEAR_UNIFORM       # theta=0.20 是政策阈值，不是数据断点
else:                             OK, ipv_log = sum(candidate_ipv_i * w_log_i)
```

台账已经把这些判据的结果写在 `status` 与 `reason_code` 两列里，监督方已验证其与冻结规格
逐行一致（4,980,050 个科学可判行零处不一致）。**你直接用 `status` / `reason_code`，
不要自己重跑门。**

**判别只能看 `status` 与 `reason_code`，绝对不能看 `ipv_log` 的数值。**
`ipv_log = 0` 是合法且常见的通过门估计值：门后通过行里恰为 0 的占 5.0097%（175,458/3,502,340）。
（早期文档流传过一个 23.40%，那是 J 轨锚点样本 238/1,017 被错当成全语料属性，已订正，
不要引用。）

连接键（L1 已验证为精确的 one-to-zero-or-one 左连接）：

```
case_key + anchor_frame_index + perspective + source_dataset + measurement_role=target_future
```

---

## 6. 分母纪律（违反即退回重做）

至少有四个分母在流通，**不许在它们之间搬运比率**：

| 分母 | 含义 |
|---|---|
| 2,646,058 | J 轨 HT 权重的全域分母 |
| 4,981,984 | InterHub canonical 求解单元 |
| 8,994,736 | RQ009 台账行 |
| 1,270,566 | RQ009 打分目标行（单 alpha 层） |

- **每个百分数必须紧跟分子、分母、筛选条件、来源文件与列名**。写不出来就说明你自己也不知道
  它是在什么域上测的，那就不要写这个数。
- `2,646,058` 与 `8,994,736` 的关系**仍未确立**，不得称"域一致"。
- 求解单元与台账行的压缩比 `2.804×` 已知。

---

## 7. 措辞禁令

- 全文**禁用** `estimability` 一词，**禁用**"测出/未测出 IPV"这类说法。
  可辩护的表述是：**权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息**。
- **描述性结果不得写成因果主张。**
- 不用比喻、不用自造简称。必须用项目专有名词时，当场用一句话说明它是什么。

---

## 8. 自查（一轮，但必须有牙齿）

出报告前做完这些，把结果写进报告的自查节：

1. **连接健康**：左连接命中/未命中各多少（带分母）；确认是 one-to-zero-or-one，重复键 0。
2. **held_out 断言**：参与计算的行中 `rq007_split` 不在 `{development, guard}` 的**实测计数为 0**。
3. **两臂只差一个变量**：明确列出两臂共用的 context 特征集、alpha 层、calibration/test 划分，
   并证明它们逐项相同。
4. **每格支撑量**：重建前后每个 context 格还剩多少样本；有多少格跌破 `MIN_SUPPORT_L1_PER_L2 = 5`。
5. **负对照（强制）**：至少挑一条你自己的验收判据，**故意扰动使它失败**，并把失败输出贴进报告。
   一条永远不会 FAIL 的检查不算检查。本项目已经出过两例"看起来在检查、实际没检查"。
6. **数值健康**：NaN/inf 计数、区间宽度是否出现负值或病态常数、coverage 是否落在 [0,1]。

---

## 9. 硬边界

```
不改：src/sociality_estimation/core/agent.py
      src/sociality_estimation/core/ipv_estimation.py
      src/sociality_estimation/core/reliability_logdomain.py
      pipelines/interhub/process_interhub.py
      configs/ipv_sigma01_exact.json
不改 RQ009 run 目录下的任何原文件（只读）
不做 git commit（产物由 PI 统一提交）
禁止 git checkout -- . / restore . / stash / reset --hard / clean -fd
禁止改动或删除 data/derived/ 下任何东西（只读）
RQ007 held_out 不得被解析
RQ014 致盲相关的评分字段不得读取
不得静默覆盖已冻结产物或已接受的 decision.md
git status 一律用 git --no-optional-locks status --porcelain
时间戳一律实取 date -u +%Y-%m-%dT%H:%M:%SZ，不要前瞻估计
注意：仓库根目录当前有另一个已授权的 git 清理任务在跑，你不要碰 git 的任何写操作
```

环境：本机 python3 已有 pyarrow 21.0.0 / pandas 2.3.3。缺依赖就直接在当前环境装上继续，
不要停下来问。

---

## 10. 报告结尾必须带状态行

```
state: WAITING_ON_COMMANDER
timestamp_utc: <实取>
```

报告开头必须先定位：这项工作要解决什么问题、整体走到哪一步、本次是其中哪一环。
不要直接从增量讲起。需要监督方拍板的事**单独成节**，写清选项、判断依据、不做的后果。
