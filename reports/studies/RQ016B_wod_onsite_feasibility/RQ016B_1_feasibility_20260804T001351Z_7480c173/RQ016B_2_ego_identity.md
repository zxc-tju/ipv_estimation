# RQ016B-F2 ego identity audit

## 定位

最终目标是在线验证：把一辆自动驾驶车表现出的社会交互倾向与人类参照分布比较。RQ016 已经重建了人类参照分布，下一步要把它用于 OnSite 自动驾驶车数据。本次 F2 是进入下一步之前的只读事实查证：确认 RQ009/RQ016 envelope 里的目标列到底属于哪一方，以及其中是否混入自动驾驶车自己的目标值。

## 结论先行

有。RQ016 B 臂域共 635,618 行，来源是 RQ009 `fold == test` feature rows 精确连接 K2 台账中 `artifact_id == "rq009_feature_matrix"`、`measurement_role == "target_future"`、`status == "OK"` 的 `product_row_key`。其中 E1，即 ego 是 AV、目标值是自动驾驶车自己的 IPV，为 69,288/635,618 = 10.9009%；E2，即 ego 是人、对手是 AV，为 79,670/635,618 = 12.5343%；E3，即纯人-人，为 486,660/635,618 = 76.5649%。

关键核对通过：B 臂域的 E1+E2 = 148,958/635,618 = 23.4351%，正好等于监督方给出的 B 臂域 `AV;HV` 行数 148,958/635,618 = 23.4351%。

## Q1: `target_ipv_future` 是谁的 IPV

结论：`target_ipv_future` 是该行 `ego_key_agent` 的未来 IPV；`counterpart_ipv_current` 是该行 `counterpart_key_agent` 的当前 IPV。

代码证据：

```python
# reports/.../build_features.py:665-674
if perspective == "key_agent_1":
    ego_id, cp_id = key_agent_1, key_agent_2
    prefix_ego, prefix_cp = "key_agent_1", "key_agent_2"
    target_ipv = targets["target_ipv_key_agent_1_by_row"][target_final_pos]
elif perspective == "key_agent_2":
    ego_id, cp_id = key_agent_2, key_agent_1
    prefix_ego, prefix_cp = "key_agent_2", "key_agent_1"
    target_ipv = targets["target_ipv_key_agent_2_by_row"][target_final_pos]
```

同一函数随后把 `ego_id` 写入 `ego_key_agent`，把 `cp_id` 写入 `counterpart_key_agent`，并把 `target_ipv` 写入 `target_ipv_future`：`build_features.py:724-726` 与 `build_features.py:779`。`counterpart_ipv_current` 明确来自 `row[f"ipv_{prefix_cp}"]`，即另一方当前列：`build_features.py:774`。目标 lookup 来源列是 `TARGET_HW4 ipv_key_agent_1/2 at frame t*+6`：`build_features.py:619-623`、`feature_dictionary.csv:57`。`finalize_features.py` 只把 `target_ipv_future` 读入分布审计，不改变它的身份口径：`finalize_features.py:153`。

字典原文也一致：`perspective` 是 “key_agent_1 or key_agent_2 as ego”，`ego_key_agent` 是从 `key_agents` 解析的 ego id，`counterpart_ipv_current` 是 counterpart alias，`target_ipv_future` 是 ego hw=4 IPV：`feature_dictionary.csv:7-9`、`feature_dictionary.csv:52`、`feature_dictionary.csv:57`。

## Q2: 每一行的 ego 是 AV 还是人

可确定，但不能用 `vehicle_type_list` 的位置直接判定。

确定 ego 与对手身份的代码规则是：`key_agents` 被分号切成两个 agent，第一位是 `key_agent_1`，第二位是 `key_agent_2`；`perspective == "key_agent_1"` 时 ego 是第一位，`perspective == "key_agent_2"` 时 ego 是第二位。对应代码在 `build_features.py:492-496`、`build_features.py:658`、`build_features.py:665-674`、`build_features.py:724-726`。`data_health.json:443-444` 也记录了同一合同：`key_agents` 分号切分，列位置 `key_agent_1/key_agent_2` 对应第一/第二 agent。

车辆类型的可用口径是 `agent_type_pair`，它来自源列 `key_agents_type`：`build_features.py:741` 与 `feature_dictionary.csv:24`。非报告代码中对 key-agent 车辆类型的生成说明是：先把 pkl `metadata.vehicle_type` 与 `metadata.track_ids` 对齐，再按 CSV `key_agents` 顺序抽取 key agent 1/2 的类型；见 `pipelines/interhub/tools/update_ipv_distribution_report.py:291-297`、`pipelines/interhub/tools/update_ipv_distribution_report.py:665-685`。

本次脚本只读源结构列复核了这个顺序：`data/derived/interhub/20260612_sigma_0_1_full_rerun/00_hpc_outputs/sigma01_ipv_timeseries.csv` 的 `key_agents_type == "AV;HV"` 源行共有 951,217/3,695,981 = 25.7365%，筛选条件为源列 `key_agents_type == "AV;HV"`，来源列为 `key_agents`, `key_agents_type`, `vehicle_type`, `AV_included`, `key_agent_1`, `key_agent_2`；这些行中 `key_agents` 第一位为 `ego` 的有 951,217/951,217 = 100.0000%，第一位不是 `ego` 的为 0/951,217 = 0.0000%。因此在 RQ009 feature matrix 中使用以下规则：

- `agent_type_pair == "AV;HV"` 且 `perspective == "key_agent_1"`：E1，ego 是 AV。
- `agent_type_pair == "AV;HV"` 且 `perspective == "key_agent_2"`：E2，ego 是人、对手是 AV。
- `agent_type_pair == "HV;HV"`：E3，纯人-人。

`vehicle_type_list` 不能独立映射到 key-agent 编号。构造代码只是把源列 `vehicle_type` 原样写入 `vehicle_type_list`：`build_features.py:742`；字典也只说它是 “Vehicle type metadata list”：`feature_dictionary.csv:25`。本次 B 臂域中 `vehicle_type_list` 的主要取值包括：

```json
{
  "['HV', 'HV']": 430669,
  "['HV', 'AV']": 117138,
  "['HV', 'HV', 'HV']": 52892,
  "['HV', 'HV', 'AV']": 26807,
  "['HV', 'HV', 'HV', 'HV']": 2973,
  "['AV', 'HV']": 2120,
  "['HV', 'HV', 'HV', 'AV']": 2581,
  "['HV', 'AV', 'HV']": 312,
  "['HV', 'HV', 'HV', 'HV', 'HV']": 126
}
```

这些列表常含 3 个或更多元素，且元素没有 agent id；所以不能从 `vehicle_type_list[0]` 或 `vehicle_type_list[1]` 直接推出 ego 类型。

## Q3: 三类计数

### 整个 RQ009 feature matrix 与各 fold

| 群体 | 类别 | 行数与百分比 | 筛选条件 | 来源文件与列 |
|---|---:|---:|---|---|
| 整个 RQ009 feature matrix | E1: ego 是 AV | 829,784/6,397,266 = 12.9709% | all rows in RQ009 feature matrix; `agent_type_pair == "AV;HV"` 且 `perspective == "key_agent_1"` | `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix`; columns `fold`, `agent_type_pair`, `perspective`, `av_included` |
| 整个 RQ009 feature matrix | E2: ego 是人、对手是 AV | 829,784/6,397,266 = 12.9709% | all rows in RQ009 feature matrix; `agent_type_pair == "AV;HV"` 且 `perspective == "key_agent_2"` | `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix`; columns `fold`, `agent_type_pair`, `perspective`, `av_included` |
| 整个 RQ009 feature matrix | E3: 纯人-人 | 4,737,698/6,397,266 = 74.0582% | all rows in RQ009 feature matrix; `agent_type_pair == "HV;HV"` | `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix`; columns `fold`, `agent_type_pair`, `perspective`, `av_included` |
| train | E1: ego 是 AV | 343,017/2,558,374 = 13.4076% | fold == train; `agent_type_pair == "AV;HV"` 且 `perspective == "key_agent_1"` | `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix/fold=train`; columns `fold`, `agent_type_pair`, `perspective`, `av_included` |
| train | E2: ego 是人、对手是 AV | 343,017/2,558,374 = 13.4076% | fold == train; `agent_type_pair == "AV;HV"` 且 `perspective == "key_agent_2"` | `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix/fold=train`; columns `fold`, `agent_type_pair`, `perspective`, `av_included` |
| train | E3: 纯人-人 | 1,872,340/2,558,374 = 73.1848% | fold == train; `agent_type_pair == "HV;HV"` | `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix/fold=train`; columns `fold`, `agent_type_pair`, `perspective`, `av_included` |
| calibration | E1: ego 是 AV | 162,097/1,266,282 = 12.8010% | fold == calibration; `agent_type_pair == "AV;HV"` 且 `perspective == "key_agent_1"` | `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix/fold=calibration`; columns `fold`, `agent_type_pair`, `perspective`, `av_included` |
| calibration | E2: ego 是人、对手是 AV | 162,097/1,266,282 = 12.8010% | fold == calibration; `agent_type_pair == "AV;HV"` 且 `perspective == "key_agent_2"` | `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix/fold=calibration`; columns `fold`, `agent_type_pair`, `perspective`, `av_included` |
| calibration | E3: 纯人-人 | 942,088/1,266,282 = 74.3980% | fold == calibration; `agent_type_pair == "HV;HV"` | `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix/fold=calibration`; columns `fold`, `agent_type_pair`, `perspective`, `av_included` |
| guard_tune | E1: ego 是 AV | 161,755/1,302,044 = 12.4232% | fold == guard_tune; `agent_type_pair == "AV;HV"` 且 `perspective == "key_agent_1"` | `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix/fold=guard_tune`; columns `fold`, `agent_type_pair`, `perspective`, `av_included` |
| guard_tune | E2: ego 是人、对手是 AV | 161,755/1,302,044 = 12.4232% | fold == guard_tune; `agent_type_pair == "AV;HV"` 且 `perspective == "key_agent_2"` | `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix/fold=guard_tune`; columns `fold`, `agent_type_pair`, `perspective`, `av_included` |
| guard_tune | E3: 纯人-人 | 978,534/1,302,044 = 75.1537% | fold == guard_tune; `agent_type_pair == "HV;HV"` | `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix/fold=guard_tune`; columns `fold`, `agent_type_pair`, `perspective`, `av_included` |
| test | E1: ego 是 AV | 162,915/1,270,566 = 12.8222% | fold == test; `agent_type_pair == "AV;HV"` 且 `perspective == "key_agent_1"` | `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix/fold=test`; columns `fold`, `agent_type_pair`, `perspective`, `av_included` |
| test | E2: ego 是人、对手是 AV | 162,915/1,270,566 = 12.8222% | fold == test; `agent_type_pair == "AV;HV"` 且 `perspective == "key_agent_2"` | `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix/fold=test`; columns `fold`, `agent_type_pair`, `perspective`, `av_included` |
| test | E3: 纯人-人 | 944,736/1,270,566 = 74.3555% | fold == test; `agent_type_pair == "HV;HV"` | `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix/fold=test`; columns `fold`, `agent_type_pair`, `perspective`, `av_included` |

全矩阵核对：`AV;HV` 为 1,659,568/6,397,266 = 25.9418%，筛选条件为 `agent_type_pair == "AV;HV"`，来源文件 `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix`，来源列 `agent_type_pair`。四个 fold 的 E1+E2+E3 都等于各自分母，`unexpected == 0`。

### RQ016 B 臂域

B 臂域定义：RQ009 `fold == test` 行，构造 `case_key=<...>|anchor_frame_index=<...>|perspective=<...>|source_dataset=<...>`，精确连接 K2 台账 `data/derived/rq015k_logdomain_gate/l1_v1/artifact_id=rq009_feature_matrix` 中 `artifact_id == "rq009_feature_matrix"`、`measurement_role == "target_future"`、`status == "OK"` 的 `product_row_key`。

| 类别 | 行数与百分比 | 筛选条件 | 来源文件与列 |
|---|---:|---|---|
| E1: ego 是 AV | 69,288/635,618 = 10.9009% | B 域条件; `agent_type_pair == "AV;HV"` 且 `perspective == "key_agent_1"` | `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix/fold=test joined to data/derived/rq015k_logdomain_gate/l1_v1/artifact_id=rq009_feature_matrix`; columns `case_key`, `anchor_frame_index`, `perspective`, `source_dataset`, `agent_type_pair`, `vehicle_type_list`, K2 `product_row_key`, `measurement_role`, `status` |
| E2: ego 是人、对手是 AV | 79,670/635,618 = 12.5343% | B 域条件; `agent_type_pair == "AV;HV"` 且 `perspective == "key_agent_2"` | `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix/fold=test joined to data/derived/rq015k_logdomain_gate/l1_v1/artifact_id=rq009_feature_matrix`; columns `case_key`, `anchor_frame_index`, `perspective`, `source_dataset`, `agent_type_pair`, `vehicle_type_list`, K2 `product_row_key`, `measurement_role`, `status` |
| E3: 纯人-人 | 486,660/635,618 = 76.5649% | B 域条件; `agent_type_pair == "HV;HV"` | `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix/fold=test joined to data/derived/rq015k_logdomain_gate/l1_v1/artifact_id=rq009_feature_matrix`; columns `case_key`, `anchor_frame_index`, `perspective`, `source_dataset`, `agent_type_pair`, `vehicle_type_list`, K2 `product_row_key`, `measurement_role`, `status` |

## 自查

1. B 臂域 `E1 + E2`：69,288 + 79,670 = 148,958，等于监督方给出的 148,958。
2. 各群体 `E1 + E2 + E3` 都等于分母；脚本对 `all`、四个 fold、B 臂域逐项断言，且 `unexpected == 0`。
3. 负对照：故意错误地把 `vehicle_type_list[0]` / `vehicle_type_list[1]` 当作 `key_agent_1` / `key_agent_2` 类型。该规则在 B 臂域失败，输出如下：

```json
{
  "rule": "Incorrectly treat vehicle_type_list[0]/[1] as key_agent_1/key_agent_2 types.",
  "counts": {
    "E1": 64101,
    "E2": 55469,
    "E3": 516048,
    "unclassified": 0,
    "total": 635618
  },
  "observed_E1_plus_E2": 119570,
  "expected_E1_plus_E2_from_agent_type_pair_AV_HV": 148958,
  "status": "FAIL_EXPECTED"
}
```

4. Q1 的结论可由 `build_features.py:665-674`、`build_features.py:724-779`、`feature_dictionary.csv:52`、`feature_dictionary.csv:57` 直接验证。
5. 本次没有读取 RQ014 致盲评分字段；没有读取 `target_ipv_future` 数值列来做 Q3 计数；没有按 RQ007 split 筛选 held-out 行。Q3 第 1、2 项只用 RQ009 结构列 `fold`, `agent_type_pair`, `perspective`, `av_included`；B 臂域使用 K2 的 `product_row_key`, `measurement_role`, `status`。

## 复跑

```bash
python3 .codex-fleet/rq016b-wod-onsite-feasibility/work/F2/rq016b_f2_ego_identity.py
```

输出：

- JSON: `.codex-fleet/rq016b-wod-onsite-feasibility/work/F2/ego_identity.json`
- Report: `.codex-fleet/rq016b-wod-onsite-feasibility/board/reports/RQ016B_2_ego_identity.md`

state: WAITING_ON_COMMANDER
timestamp_utc: 2026-08-04T00:13:51Z
