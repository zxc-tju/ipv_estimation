# J1 IPV 弃权门规格与全语料影响的 design-based estimate（不是普查）

这项工作解决的问题是：在 online verification 中，先判断当前帧的 IPV（Interaction Preference Value，表示社会互动倾向的一个标量参数）是否可估；若该数值不携带候选间判别信息，则本帧先弃权，不把 `ipv=0` 当成失败占位。整体研究已经走到 RQ009 envelope 已接受、第一道弃权门规则已定稿的阶段；本次 J1 是把这道门冻结成可执行规格，并用 2,300 个有设计权重的锚点样本估计它对全语料的影响。本文所有“全域”数字均为 design-based estimate（不是普查），因为现有全语料台账缺少逐候选 MSE 与 log 域权重，不能直接逐行套用这道门。

## 1. 门的可执行规格

### 1.1 输入契约

本门只使用单次运行内可得的量；online 场景逐帧调用时不需要、也不得引用跨运行或跨环境比较量。

| 字段 | 类型 | 空值规则 | 说明 |
|---|---:|---|---|
| `frame_id` | string | 不可为空 | 当前 online 帧或离线行的稳定标识。 |
| `candidate_grid_id` | string | 不可为空 | 7 个候选 IPV 网格的版本标识；下游用它确认候选点一致。 |
| `K` | integer | 不可为空；本门要求 `K=7` | 候选 IPV 数量。 |
| `candidate_ipv[7]` | array[number] | 不可为空；长度为 7 | 与 MSE / 权重同序的 IPV 候选值。 |
| `mse_per_candidate[7]` | array[number] | 不可为空；长度为 7；必须有限 | 当前帧单次前向模型对 7 个候选的 MSE。 |
| `log_score[7]` 或可由 MSE 得到的等价量 | array[number] | 不可为空；长度为 7；必须有限 | 推荐定义为 `log_score_i = -mse_i / (2 * sigma^2)`；如果实现内部已有等价 log likelihood，可直接使用。 |
| `context_cell_key` | string 或 null | online 若已分格则不可为空；单帧裸调用可为 null | RQ009 的上下文分格键；本门不设计第二道样本量条件。 |

`w_log` 定义为 log 域归一化权重：

```text
log_z = logsumexp(log_score[0:7])
w_log_i = exp(log_score_i - log_z)
sum_i w_log_i = 1
```

`k_eff_log = 1 / sum_i(w_log_i^2)`，范围为 `[1, 7]`。这个换算不是一一对应的：例如在“一大六等”的假设下，`k_eff=6.75` 给 `max(w)=0.2102`，而 `max(w)=0.2027` 给 `k_eff=6.801`。因此本门按 `max(w_log)` 判断，不用 `k_eff_log` 反推。

### 1.2 伪代码

```text
input: frame_id, candidate_grid_id, K=7, candidate_ipv[7], mse_per_candidate[7], log_score[7], context_cell_key

w_log = softmax(log_score) using log-sum-exp
mse_spread = max(mse_per_candidate) - min(mse_per_candidate)
max_w_log = max(w_log)
k_eff_log = 1 / sum(w_log_i * w_log_i)

if mse_spread == 0:
    status = "ABSTAIN"
    reason_code = "NO_IPV_EFFECT"
    ipv_log = null
elif max_w_log < 0.20:
    status = "ABSTAIN"
    reason_code = "NEAR_UNIFORM"
    ipv_log = null
else:
    status = "OK"
    reason_code = null
    ipv_log = sum(candidate_ipv_i * w_log_i)

return machine-readable record
```

判据 1 的机制依据：7 个候选给出逐位相同的 MSE，说明 IPV 对当前前向目标没有影响；无实质交互时目标退化为 `cos(ipv) * interior + constant`，正标量不改变极小点。判据 1 保留为语义标签，不作为额外筛选量与判据 2 相加。样本内复核：`spread(mse)==0` 的 400 / 2,300 行，其 `max(w_log)` 全部为 `1/7 = 0.142857143`，一律满足 `max(w_log)<0.20`。

`theta=0.20` 是政策阈值，不是自然断点。仅按固定说明记录一行样本内敏感性：`theta=0.18/0.20/0.22` 时，样本内门后行数为 `1,112 / 1,017 / 946`，分母均为 `2,300`，筛选均为 `spread(mse)!=0 and max(w_log)>=theta`。

### 1.3 机器可读输出契约

| 字段 | 类型 | 空值规则 | 说明 |
|---|---:|---|---|
| `status` | enum string | 不可为空；`OK` 或 `ABSTAIN` | 本帧是否通过第一道门。 |
| `ipv_log` | number 或 null | `status=OK` 时必须为有限数；`ABSTAIN` 时必须为 null，不得为 0、NaN 或缺列 | log 域权重下的 IPV 点值。 |
| `reason_code` | enum string 或 null | `status=OK` 时为 null；`ABSTAIN` 时必须为 `NO_IPV_EFFECT` 或 `NEAR_UNIFORM` | 互斥主因；先判 `NO_IPV_EFFECT`，否则再判 `NEAR_UNIFORM`。 |
| `max_w_log` | number | 不可为空；范围 `[1/K, 1]` | 7 个 log 域归一权重的最大值。 |
| `mse_spread` | number | 不可为空；`>=0` | `max(mse_per_candidate)-min(mse_per_candidate)`。 |
| `k_eff_log` | number | 不可为空；范围 `[1, K]` | `1/sum(w_log^2)`，用于诊断，不替代 `max_w_log`。 |
| `candidate_grid_id` | string | 不可为空 | 候选网格版本。 |
| `K` | integer | 不可为空；本门为 7 | 候选数量。 |
| `frame_id` | string | 不可为空 | 帧或行标识。 |
| `gate_pass_rate` | number 或 null | 单帧原始记录可为 null；按批次或 RQ009 cell 聚合后为 `[0,1]` | RQ009 接入时报告同一门的通过率。 |
| `context_cell_key` | string 或 null | RQ009 聚合记录不可为空；单帧裸调用可为 null | RQ009 上下文分格键。 |

OK 样例：

```json
{
  "status": "OK",
  "ipv_log": 0.5805016,
  "reason_code": null,
  "max_w_log": 0.2726758,
  "mse_spread": 0.2752463,
  "k_eff_log": 3.976017,
  "candidate_grid_id": "ipv_grid_7_default",
  "K": 7,
  "frame_id": "ipv_007308|15|1",
  "gate_pass_rate": null,
  "context_cell_key": null
}
```

ABSTAIN 样例：

```json
{
  "status": "ABSTAIN",
  "ipv_log": null,
  "reason_code": "NO_IPV_EFFECT",
  "max_w_log": 0.142857143,
  "mse_spread": 0.0,
  "k_eff_log": 7.0,
  "candidate_grid_id": "ipv_grid_7_default",
  "K": 7,
  "frame_id": "example_flat_mse_frame",
  "gate_pass_rate": null,
  "context_cell_key": null
}
```

## 2. 全域可估率的 design-based estimate（不是普查）

主结果：全域 design-based estimate（不是普查）分母为 `2,646,058` HT 权重，门后保留权重为 `1,885,831.096`，全域 design-based estimate（不是普查）可估率为 `71.2695%`。不确定性按 `scene_unique_id` cluster bootstrap 估计，全域 design-based estimate（不是普查）95% CI 为 `[67.1729%, 75.2135%]`；cluster 数 `1,909`，`B=2,000`，seed `20260731`。

下表全部为全域 design-based estimate，不是普查。

| 项 | 分子 | 分母 | 比例 |
|---|---:|---:|---:|
| 门后保留 | HT 保留权重 `1,885,831.096` | HT 分母 `2,646,058` | `71.2695%` |
| `NO_IPV_EFFECT` | HT 权重 `13,482.740` | HT 分母 `2,646,058` | `0.5095%` |
| `NEAR_UNIFORM`，排除 `spread==0` 后 | HT 权重 `746,744.164` | HT 分母 `2,646,058` | `28.2210%` |

两条弃权判据按互斥 reason 统计：先判 `spread(mse)==0` 记为 `NO_IPV_EFFECT`，否则再判 `max(w_log)<0.20` 记为 `NEAR_UNIFORM`。样本内 400 行 `spread(mse)==0` 全部已经落入 `max(w_log)<0.20`，所以判据 1 的额外筛出量为 0 行，不能把两条“各自贡献”相加。

样本内未加权门后比例为 `1,017 / 2,300 = 44.2174%`。这是样本内比例，不是全域影响；它只用于解释配额样本与设计权重结果为什么不同。

下表 HT 各列全部为全域 design-based estimate，不是普查。

| signature | 样本内可估率 | HT 分母 | 门后保留权重 | 该层权重占比 | 层内 HT 加权可估率 |
|---|---:|---:|---:|---:|---:|
| N | `363 / 500 = 72.6000%` | `2,111,119.000` | `1,611,149.536` | `2,111,119 / 2,646,058 = 79.7835%` | `76.3173%` |
| U | `511 / 1,200 = 42.5833%` | `249,018.000` | `189,192.167` | `249,018 / 2,646,058 = 9.4109%` | `75.9753%` |
| Z | `143 / 600 = 23.8333%` | `285,921.000` | `85,489.393` | `285,921 / 2,646,058 = 10.8055%` | `29.8997%` |

落差来自抽样设计：U 在样本中为 `1,200 / 2,300 = 52.1739%`，但全域 HT 权重占比只有 `249,018 / 2,646,058 = 9.4109%`；N 在样本中为 `500 / 2,300 = 21.7391%`，但全域 HT 权重占比为 `2,111,119 / 2,646,058 = 79.7835%`，且 N 的层内 HT 加权可估率为 `1,611,149.536 / 2,111,119 = 76.3173%`。因此样本内 `44.2174%` 会显著低估全域 design-based estimate（不是普查）的 `71.2695%`。

分母口径复核：`zero_postwarm_scope == True` 与 `signature in {U, Z}` 完全一致，样本计数为 `N|False 500`、`U|True 1,200`、`Z|True 600`，不一致行 `0`。本轮全域 design-based estimate（不是普查）使用全部 `2,300` 锚点、HT 分母 `2,646,058`、全部 `1,909` 个 cluster；没有沿用排除 N 层的分母。

## 3. 按锚点自带变量分格的门后形状

本节不使用 RQ009 的上下文变量，因为本轮输入文件不含那 29 个变量，且没有定义安全 join 路径。以下仅按锚点自带变量 `signature`、`n_band`、`n_obs` 分箱。表中可估率为 HT 加权比率；`ipv_log` 分位数为门后样本按 `ht_weight` 加权的经验分位数；边界占比是门后样本计数比例，口径 A 与口径 B 分开报告。

边界口径 A：`at_grid_boundary == True`。边界口径 B：`abs(abs(ipv_log) - 3*pi/8) <= 1e-9`。

### 3.1 signature

| signature | 样本 n | 门后样本 n | HT 分母 | HT 可估率 | `ipv_log` p5 | `ipv_log` 中位 | `ipv_log` p95 | 口径 A 边界占比 | 口径 B 边界占比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N | 500 | 363 | 2,111,119.000 | 76.3173% | -0.736858 | 0.112087 | 1.152608 | 118 / 363 = 32.5069% | 13 / 363 = 3.5813% |
| U | 1,200 | 511 | 249,018.000 | 75.9753% | -1.178097 | 0.366293 | 1.178097 | 298 / 511 = 58.3170% | 133 / 511 = 26.0274% |
| Z | 600 | 143 | 285,921.000 | 29.8997% | 0.000000 | 0.000000 | 0.000000 | 61 / 143 = 42.6573% | 0 / 143 = 0.0000% |

### 3.1.1 门后仍有一批 `ipv_log` 恰好为 0（leader 复核补记）

这一条由 leader 在自查复核时发现并补入，不在 J1 原报告中；数字由 leader 用同一对输入文件复算。

门后（`status=OK`）的 1,017 行里，有 **238 行（`238 / 1,017 = 23.40%`；占门后 HT 保留权重 `10.2788%`）
的 `ipv_log` 绝对值 `<= 1e-9`，即恰好为 0**。分 signature 为 N `12 / 363`、U `91 / 511`、Z `135 / 143`。
这解释了 §3.1 中 signature Z 的 p5 / 中位 / p95 三个分位数都是 `0.000000`：
Z 门后 143 行里只有 8 行非零，这 8 行只占 Z 门后权重的 `5.1198%`，
因此加权 p95 仍落在 0；Z 的加权 p99 才是 `0.143087`。**该分位数不是常数化缺陷，是真实的零点质量。**

这一点必须向下游明确披露，因为它正是本门要消除的那处混淆的镜像情形：
本门保证「弃权」不再被写成 `ipv=0`，但**反过来并不成立**——
`ipv_log = 0` 仍然是一个合法且高频的**通过门**的估计值（社会倾向取 0 表示中性）。
RQ009 接入时不得把 `ipv_log == 0` 反推为弃权或缺失；
两者的唯一判别字段是 `status` 与 `reason_code`，不是 `ipv_log` 的数值。

复算口径：`anchor_mse.csv` 的 `ipv_log`，筛选 `spread(mse)!=0 and max(w_log)>=0.20`，
零判定为 `abs(ipv_log)<=1e-9`；权重占比分母为门后 `sum(ht_weight)=1,885,831.096`。

### 3.2 n_band

| n_band | 样本 n | 门后样本 n | HT 分母 | HT 可估率 | `ipv_log` p5 | `ipv_log` 中位 | `ipv_log` p95 | 口径 A 边界占比 | 口径 B 边界占比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FULL | 1,150 | 554 | 2,484,454.000 | 71.6437% | -0.747895 | 0.078640 | 1.175337 | 244 / 554 = 44.0433% | 53 / 554 = 9.5668% |
| RAMP | 1,150 | 463 | 161,604.000 | 65.5157% | -0.969033 | 0.191105 | 1.178097 | 233 / 463 = 50.3240% | 93 / 463 = 20.0864% |

### 3.3 n_obs 分箱

箱边界：`5-6` 包含 `n_obs in {5,6}`，样本 `444` 行；`7-8` 包含 `{7,8}`，样本 `350` 行；`9-10` 包含 `{9,10}`，样本 `356` 行；`11` 包含 `{11}`，样本 `1,150` 行。

**口径提示（leader 复核补记）**：`n_obs` 与 `n_band` 不是两个独立的切法。
`n_band=FULL` 的 1,150 行 `n_obs` 全部为 `11`，`n_band=RAMP` 的 1,150 行 `n_obs` 全部在 `5..10`，
两者一一对应。因此下表 `11` 箱与 §3.2 的 `FULL` 行是**同一批行**（样本 1,150、门后 554、
HT 分母 2,484,454.000、可估率 71.6437% 逐格相同），不是彼此独立的证据；
`5-6 / 7-8 / 9-10` 三箱合计即 `RAMP`（HT 分母 `61,000.567 + 52,648.010 + 47,955.423 = 161,604.000`）。
读者不应把这两张表当作两次独立验证。

| n_obs 箱 | 样本 n | 门后样本 n | HT 分母 | HT 可估率 | `ipv_log` p5 | `ipv_log` 中位 | `ipv_log` p95 | 口径 A 边界占比 | 口径 B 边界占比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5-6 | 444 | 109 | 61,000.567 | 51.3194% | -1.171246 | 0.259908 | 1.178097 | 63 / 109 = 57.7982% | 26 / 109 = 23.8532% |
| 7-8 | 350 | 158 | 52,648.010 | 72.3735% | -0.778352 | 0.007994 | 1.178097 | 75 / 158 = 47.4684% | 33 / 158 = 20.8861% |
| 9-10 | 356 | 196 | 47,955.423 | 76.0451% | -0.531013 | 0.224237 | 1.178097 | 95 / 196 = 48.4694% | 34 / 196 = 17.3469% |
| 11 | 1,150 | 554 | 2,484,454.000 | 71.6437% | -0.747895 | 0.078640 | 1.175337 | 244 / 554 = 44.0433% | 53 / 554 = 9.5668% |

## 4. source 内部检查

本节只作内部风险披露，不作为 RQ009 下游分格依据。

| 权重带 `max(w_log)` | nuplan `|ipv_log|` 均值 | waymo `|ipv_log|` 均值 |
|---|---:|---:|
| 0.20-0.25 | 0.1306 (n=81) | 0.0507 (n=86) |
| 0.25-0.35 | 0.1777 (n=83) | 0.2001 (n=96) |
| 0.35-0.50 | 0.2734 (n=68) | 0.3560 (n=77) |
| 0.50-0.75 | 0.4383 (n=41) | 0.5706 (n=94) |
| 0.75-1.01 | 0.7551 (n=39) | 0.9698 (n=352) |

筛选条件为 `spread(mse)!=0 and max(w_log)>=0.20`；分箱左闭右开，末箱为 `[0.75,1.01)`。

边界口径 A（`at_grid_boundary` 列）下，门后 nuplan 为 `79 / 312 = 25.3205%`，waymo 为 `398 / 705 = 56.4539%`。边界口径 B（`|ipv_log|` 精确命中 `3*pi/8`，容差 `1e-9`）下，门后 nuplan 为 `3 / 312 = 0.9615%`，waymo 为 `143 / 705 = 20.2837%`。口径 B 是精确端点命中，不能替代口径 A 的边界诊断。

最高权重带的样本量不平衡（waymo 352 vs nuplan 39）解释了汇总均值差异的相当一部分；分带后仍存在残余差异。该残差不作为下游分源口径的依据，仅作内部风险披露。最高权重带 signed `ipv_log` 中位数为 nuplan `0.0639`、waymo `0.8005`；按口径 A，waymo 在每个权重带内的边界占比也更高。

## 5. “可估交互条件下”的限定

门后保留的样本中，`|ipv_log|` 随权重集中度上升而上升。因此 envelope 应表述为“可估交互条件下人类的 IPV 分布”，不是无条件的“人类 IPV 分布”。

结论只对“同一道门通过后的条件分布”成立。若 AV 的通过机制与人类不同，须分别报告 AV 与人类的门通过率，并把“未通过”本身作为监控结果的一部分。

## 6. 真正应用到全语料需要什么

现有全语料台账为 4 份 parquet、合计 `14,473,982` 行，已有列包括 `artifact_id`、`product_row_key`、`measurement_role`、`case_id`、`rq007_split`、`ipv_error`、`K`、`candidate_grid_id`、`k_eff`、`q_eff`、`attempt_status`、`reason_code`、`recoverability`、`ledger_schema_version`、`aggregation_perspective`、`aggregation_configuration`。它缺少本门必需的 `mse_per_candidate[7]` 和 `w_log[7]`；已有 `k_eff` 来自连乘域派生，不能用来替代 log 域权重。

所以要把这道门真正应用到全语料，需要重新为全语料中需要判门的行物化以下列：`mse_per_candidate[7]`、log 域归一权重 `w_log[7]`、`max_w_log`、`mse_spread`、`k_eff_log`、`status`、互斥 `reason_code`、`ipv_log`（弃权为 null）、`candidate_grid_id`、`K`、帧/行键，以及 RQ009 所需的 `context_cell_key` 与门通过率聚合字段。

量级估计：重算规模是千万行级，以上限说约为当前台账的 `14,473,982` 行；每行需评价 7 个候选并保留 7 维 MSE 与 7 维 log 域权重，因此写出量级约为数千万个浮点值外加状态列。需要的产物不是新的阈值或新判据，而是一个带 log 域权重和逐候选 MSE 的全语料门判据台账。本轮没有授权全量重算、没有作业脚本、没有提交 HPC 作业。

需要 PI 拍板的事项只在将来需要普查数字时出现：

| 选项 | 判断依据 | 不做的后果 |
|---|---|---|
| 维持本报告作为 design-based estimate（不是普查） | 2,300 锚点是有已知抽取概率的分层概率样本，已能估计全域影响 | 不能声称已有逐行全语料门判据结果，只能报告设计基估计和 online 前瞻接入规格。 |
| 另行授权全语料重算并物化新台账 | 只有逐行 `mse_per_candidate[7]` 与 `w_log[7]` 才能无歧义套用本门 | 若不授权，就不能得到按任意后续分格展开的普查式弃权统计。 |

## 7. 可复算性附录

复算入口：`<local-rq009-venv>/bin/python .codex-fleet/rq015j-gate-spec/work/j1_gate_spec_compute.py`。证据 JSON：`.codex-fleet/rq015j-gate-spec/work/j1_gate_spec_evidence.json`。输入文件：`.codex-fleet/rq015b-repair/work/anchor_mse.csv` 与 `.codex-fleet/rq015b-repair/work/mechanism_split.csv`，按 `anchor_id` 一对一 join。

| 数字 | 数据文件与列 | 筛选条件 | 权重与分母 |
|---|---|---|---|
| `spread(mse)==0` 样本 `400 / 2,300`，且 `max(w_log)=1/7` | `anchor_mse.csv`: `mse_per_candidate[7]`, `w_log[7]` | `max(mse)-min(mse)==0` | 样本计数；分母为锚点 `2,300`。 |
| `theta=0.18/0.20/0.22` 样本内门后 `1,112 / 1,017 / 946` | `anchor_mse.csv`: `mse_per_candidate[7]`, `w_log[7]` | `spread!=0 and max(w_log)>=theta` | 样本计数；分母均为 `2,300`。 |
| 全域 design-based estimate（不是普查）HT 分母 `2,646,058` | `mechanism_split.csv`: `ht_weight` | 全部 2,300 锚点 | `sum(ht_weight)`。 |
| 全域 design-based estimate（不是普查）门后保留权重 `1,885,831.096` 与可估率 `71.2695%` | `anchor_mse.csv`: 判据列；`mechanism_split.csv`: `ht_weight` | `spread!=0 and max(w_log)>=0.20` | 分子 `sum(ht_weight * gate_ok)`；分母 `2,646,058`。 |
| 全域 design-based estimate（不是普查）CI `[67.1729%, 75.2135%]` | 同上，并用 `scene_unique_id` | 对 `scene_unique_id` 聚合后 bootstrap | 全部 `1,909` cluster；`B=2,000`；seed `20260731`；`numpy.random.default_rng`。 |
| 全域 design-based estimate（不是普查）互斥 reason `NO_IPV_EFFECT 0.5095%` / `NEAR_UNIFORM 28.2210%` | `anchor_mse.csv`: `mse_per_candidate[7]`, `w_log[7]`; `mechanism_split.csv`: `ht_weight` | 先 `spread==0`，否则 `max(w_log)<0.20` | 分母 `2,646,058`；分子分别为 HT 权重 `13,482.740` 与 `746,744.164`。 |
| signature 解释表 | `anchor_mse.csv`: `signature`, 判据列；`mechanism_split.csv`: `ht_weight` | 各 signature 内分别套门 | 每层 HT 分母为 `sum(ht_weight)`，层内可估率为 `sum(ht_weight*gate_ok)/sum(ht_weight)`。 |
| `zero_postwarm_scope` 复核 | `mechanism_split.csv`: `signature`, `zero_postwarm_scope` | 交叉计数 | 样本计数：`N|False 500`、`U|True 1,200`、`Z|True 600`，不一致行 `0`。 |
| 分格形状表 | `anchor_mse.csv`: `signature`, `n_band`, `n_obs`, `ipv_log`, `at_grid_boundary`; `mechanism_split.csv`: `ht_weight` | 各格内套门；`n_obs` 箱为 `5-6`, `7-8`, `9-10`, `11` | 可估率用 HT 比率；`ipv_log` 分位数为门后 `ht_weight` 加权经验分位数；边界为门后样本计数比例。 |
| 第 4 节内部检查 | `anchor_mse.csv`: 第 4 节分组列、`w_log[7]`, `ipv_log`, `at_grid_boundary` | `spread!=0 and max(w_log)>=0.20`；权重带左闭右开，末箱 `[0.75,1.01)` | 样本计数与样本均值，仅作内部风险披露。 |
| 边界口径 A / B | `anchor_mse.csv`: `at_grid_boundary`, `ipv_log` | 仅门后样本；A 为 `at_grid_boundary==True`，B 为 `abs(abs(ipv_log)-3*pi/8)<=1e-9` | 样本计数比例；例如内部检查中 nuplan A `79/312`、B `3/312`，waymo A `398/705`、B `143/705`。 |

自查结果：

- [x] 7 节齐全。
- [x] 标题与每处全域数字都标注了 design-based estimate（不是普查）。
- [x] 两条判据按互斥 reason 报，没有相加重复计数。
- [x] 全域分母用的是 `2,646,058` / `1,909` cluster，不是 `zero_postwarm_scope` 分母。
- [x] 边界占比两个口径都报且各自写明定义。
- [x] 未按数据来源拆分主输出；数据来源列只用于第 4 节内部检查与附录复算说明。
- [x] `theta=0.20` 写成政策阈值，不是自然断点。
- [x] 弃权时 `ipv_log` 为 null，不是 0。
- [x] 已执行术语禁令：无禁用英文术语、无禁用中文表述。
- [x] 没有 git commit、没有改受保护估计器文件、没有提交 HPC 作业。
- [x] 复审已给的数字都复现了；无不一致处。

J1_DONE 2026-08-02T04:11:51Z

---

## 附：leader 自查记录（2026-08-02T04:26Z）

leader 用钉死解释器 `<local-rq009-venv>/bin/python` 独立复算了本报告的头部数字，
未复用 J1 的脚本：门后样本 `1,017 / 2,300`、HT 分母 `2,646,058`、门后保留权重 `1,885,831.096`、
可估率 `71.2695%`、cluster 数 `1,909` —— **逐项与 J1 一致**。
另核对内部一致性：三个 signature 层的 HT 分母合计与保留权重合计分别等于全域值；
`NO_IPV_EFFECT 13,482.740 + NEAR_UNIFORM 746,744.164 = 760,226.904 = 2,646,058 - 1,885,831.096`，
互斥 reason 无重复计数。

合规检查：报告全文对两条术语禁令均无命中（禁用的英文术语与「已/未测得 IPV」式中文表述各 0 处，
扫描时排除本节这句说明本身）；
`src/`、`pipelines/` 无改动（含 `reliability_logdomain.py`）；无新增 git commit；无 HPC 作业提交。

leader 在 J1 结项后所做的编辑，共三处，均不改动任何数字：

1. **新增 §3.1.1**（实质性补记）。J1 未披露门后仍有 238 行 `ipv_log` 恰好为 0
   （占门后 HT 权重 10.2788%）。这是本门混淆问题的镜像情形，必须让 RQ009 知道
   「`ipv_log==0` 不得反推为弃权」。同时解释了 §3.1 中 Z 层三个分位数全为 0 的成因，
   并确认那**不是**常数化缺陷。
2. **新增 §3.3 口径提示**。`n_obs` 的 `11` 箱与 `n_band` 的 `FULL` 行是同一批行，
   两张表不构成两次独立验证，原文未说明。
3. **§2 两处表格去冗余**。原文把「全域 design-based estimate（不是普查）」逐字插入表格列名与名词短语内部，
   已改为在表前统一声明一次。**design-based、不是普查的标注本身一处未删**，
   仅去掉了表格列名内部的重复。
