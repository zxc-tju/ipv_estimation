## L2 OnSite UNKNOWN source-status audit

RQ015 的目标是给 online verification 增加机制一：先判断一个 IPV（Interaction Preference Value，用一个标量刻画交互倾向）数值是否携带候选间的判别信息；通过机制一后，才进入 RQ009 已接受的人类分布 envelope 支持度判据，这个判据用于判断数值是否落在人类分布覆盖范围内。K2 已生成 14,473,982 行全语料台账，但 OnSite 数据源 `onsite_dense_timeseries` 出现 274,022 个 `UNKNOWN` 来源状态。L2 本轮只查这些来源状态为什么出现、这些源行是否有输入，不重算、不补数据。

### 结论

1. `UNKNOWN` 是 RQ015A 台账构建代码的显式分支，不是隐式 else 兜底。`scripts/rq015a/build_ledger.py:1219-1233` 的关系是：先按 OnSite 局部序号 `local_position < 4` 返回 `NOT_ATTEMPTED`；否则如果 role 对应的 `ipv_error` 为空，返回 `UNKNOWN` 和 `EMPTY_CELL_UNEXPLAINED`；否则如果 `q_eff` 为 None，返回 `UNKNOWN` 和 `DEGENERATE_IPV_ERROR`；剩余情况才返回 `ATTEMPTED`。OnSite 的局部序号规则在 `scripts/rq015a/build_ledger.py:1126-1148`，schema 明文是 `reports/plans/RQ015A_ledger_schema_v4_20260731.json:439-449`。
2. K2 OnSite 台账合计 281,268 行：`UNKNOWN` 97.4238%（274,022/281,268），`ATTEMPTED` 1.0574%（2,974/281,268），`NOT_ATTEMPTED` 1.5188%（4,272/281,268）。所有 `UNKNOWN` 行中，100.0000%（274,022/274,022） 是 `source_reason_code=EMPTY_CELL_UNEXPLAINED`，且 `ipv_error/k_eff/q_eff` 为空。
3. 数据证据支持“流水线没走到大多数 dense 行”，不支持把这 274,022 行解释为轨迹或配对输入普遍缺失。OnSite dense 源表有 70,317 个物理行；其中每 case 前 4 个局部观测产生 1.5188%（4,272/281,268） 的 `NOT_ATTEMPTED` role 行。剩余局部序号大于等于 4 的源 role 行里，`UNKNOWN` 全部是对应 role 的误差列为空：100.0000%（274,022/274,022）。在这些 `UNKNOWN` role 行中，`case_key/frame_index/timestamp_ms`、ego/counterpart 的位置、速度、heading、配对 ID、距离与相对速度字段均为 100.0000%（274,022/274,022） 非空。
4. OnSite stage3plus 的生成脚本默认 `--max-anchors-per-unit 1`，非 `--all-valid-anchors` 时只给选中 anchor 的支撑帧、target 帧和少量斜率帧填 `ipv_*`。脚本在 `build_onsite_m3_anchors.py:776-831` 先把四个 `ipv/err` 数组初始化为 NaN，只对 `needed_h10` 与 `needed_h4` 写值；`build_onsite_m3_anchors.py:1212` 也把该表说明为“all aligned frames with IPV populated only for selected-anchor support/target frames”。因此 dense 表有 70,317 个轨迹行，但只有 2,974 个 role 行携带 IPV 数值。
5. 进一步的规模证据同向：默认 build 只产出 267 个 AV anchor；同目录 all-valid artifact 有 67,861 个 anchor 行。用 all-valid 行数减默认 bounded 行数，未物化的 anchor 候选为 99.6065%（67,594/67,861）。对包含至少一个 `UNKNOWN` role 的物理行，97.6317%（67,609/69,249） 满足该脚本的 RQ009 timing anchor 条件但没有在默认 bounded run 中填满 role 值。
6. 需要保留一个输入边界：dense 源表没有真实地图、车道、route 或 reference-line 字段，计数是 0.0000%（0/274,022）。当时脚本用 observed trajectory fallback 作为参考线，且成功 dense cases 的 ego/counterpart 轨迹都能构造这个 fallback：100.0000%（274,022/274,022）。所以本结论只说明在既有 OnSite 生成合同下多数行未被送入求值；若未来要求真实地图或车道参考线，现有 dense 表不能证明该输入已齐备。

### 与 RQ015A 口径

RQ015A 的 1.0574%（2,974/281,268） 来自 `concentration_ledger_summary.csv`，筛选条件是 `artifact == onsite_dense_timeseries` 且 `attempt_status == ATTEMPTED`，列名是 `artifact/attempt_status/rows`；分子 2,974，分母 281,268。K2 本轮的分子和分母是同一组来源状态计数，只是列名改成 `source_attempt_status`，因为 K2 另有 `gate_applicable/status/reason_code` 表示新门判据字段。

### 补齐清单（只列需求，不执行）

若要把 OnSite 从默认 bounded-anchor 产物补成更完整的 dense role 表，需要先确定补齐范围：全 aligned frames、全 RQ009 timing-valid anchor frames，或继续保留每 unit 一个 anchor。不同范围会改变分母；不先定范围，后续行数不能互相比较。

补齐需要的输入包括：每 case 的 ego 与 counterpart 连续轨迹、同一配对逻辑、局部序号和 history window 规则、候选 IPV 网格与 sigma、参考线来源、以及输出 schema 如何区分空源单元、warm-up 与数值退化。已知阻碍是：18/285 个 unit 在原 build 中失败，原因包括无合格 counterpart、无 timing-eligible counterpart、无 RQ009 timing-valid anchor、observed reference 少于两个唯一点；此外 dense 表本身没有真实地图或车道参考线字段。

### 待决事项

1. 需要确定补齐范围。选项 A 是全 aligned frames，证据链最完整但计算量接近 70,317 个物理行 × 4 个 role；选项 B 是全 RQ009 timing-valid anchor frames，规模约 67,861 个 anchor 行再展开 role；选项 C 是继续每 unit 一个 anchor，只能解释当前 2,974 个 role 行。不做该决定的后果是无法定义新的分母。

2. 需要确定参考线合同。选项 A 是沿用 observed trajectory fallback，与当前 OnSite stage3plus 产物一致；选项 B 是要求真实地图或车道参考线，这会暴露当前 dense 表 0 个 map/refline 字段的缺口。不做该决定的后果是后续补齐结果无法和 RQ009/InterHub 的输入合同对齐。

state: WAITING_ON_LEADER 2026-08-03T03:21:28Z
