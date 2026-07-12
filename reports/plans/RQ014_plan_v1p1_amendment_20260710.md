# RQ014 Plan v1.1 Amendment — B1–B5 修复

状态：`DRAFT_FOR_INDEPENDENT_REVIEW`
日期：2026-07-10（在任何评分 join / 新结果产生之前提出，符合 v1 §3 的版本化 amendment 规则）
基底：`RQ014_plan_v1_wod_e2e_rating_ipv_deviation_recovery_20260710.md`（保留不动）
评审依据：`reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/claude_plan_review_v1_20260710.md`
+ PI 对 specificity 设计的口头异议（B5，2026-07-10 会话）。

配套文件（本 amendment 生效后取代 v1 同名 registry）：

- `RQ014_recovery_extension_registry_v1p1.yaml`（新增，第三臂）
- `RQ014_forensic_registry_v1p1.yaml`（v1 + FL05）
- `RQ014_config_space_v1p1.yaml`（v1 + §B5 specificity 块替换；12 configs 不变）
- `RQ014_plan_v1p1_checksums_20260710.sha256`

未在本文修改的 v1 条款一律原样有效。

---

## B1 新增第三个 analysis family：Recovery-extension（不可晋级）

v1 的 valid family 把 envelope 形态与 endpoint 焊死为"matched time-indexed path-type HV +
rolling exceedance"。G1 取证为 NOT_FOUND，无证据支持旧实验就是该形态。为使
`NOT_RECOVERED_WITHIN_FROZEN_VALID_FAMILY` 与"family 太窄"可辨，新增第三臂：

- registry：`RQ014_recovery_extension_registry_v1p1.yaml`；
- 地位与 forensic family 相同：**不可晋级、不进 confirmation、不进任何 alpha family、
  不得触发 RQ010B amendment**；产出仅为 specification-recovery 信号；
- 全部为 V04 类比基线（4 Hz, W2.5, future-only, HMAX5）上的单因素变体（X01–X06，
  见 registry）：terminal-only endpoint、sigma01 静态状态条件包络、candidate-internal
  包络、80/95 band、pooled 统计单位；
- 与 forensic family 的分工：forensic = 缺陷与历史指纹；extension = **无缺陷但被 valid
  family 排除的合理旧设置候选**；
- 多重性：臂内 6 个 cell 的 within-scene permutation maxT，仅作 flagging；
- 允许的最高结论：`SPEC_RECOVERY_CANDIDATE`（某 cell 独占强负相关 ⇒ 旧设置候选），
  措辞不得高于 discovery 级；
- 运行时点：与 valid discovery 同批（同一 rating access 事件），结果进入同一 ledger 的
  独立分区。

§12 verdict 表补充一行：

| 条件 | 唯一 verdict |
|---|---|
| valid family 未晋级/未确认，但 extension 臂某 cell 满足 `rho_WS<=-0.30` 且臂内 maxT p<0.05 | `SPEC_RECOVERY_CANDIDATE_OUTSIDE_VALID_FAMILY`（不改变 scientific verdict） |

## B2 Forensic registry 新增 FL05：RQ010B 中间统计全索引

G1 后记忆来源最高概率宿主是 RQ010B 执行期在 HPC 产生的中间相关性表。新增：

```yaml
- forensic_id: FL05
  paths:
    - /share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis
    - /share/home/u25310231/ZXC/RQ010B_wod_e2e/results
  action: read_only_recursive_index_of_all_recorded_correlation_statistics
  deliverable: historical_stats_index.csv
  fields: [statistic_name, value, n, unit, config_fields_as_recorded, source_file,
           source_file_sha256, mtime]
  fingerprint_rule: any_recorded_rho_le_minus_0.30_becomes_historical_fingerprint_candidate
  compute: login-node_safe_grep_and_parse_only_no_recomputation
```

FL05 在 G0 关闭前完成；命中的 fingerprint 只更新 forensic 指纹与优先级，不得修改 valid registry（v1 §4 规则不变）。

## B3 §6.2 规则 8 替换文本（required-cell 盲定义）

原文"任一 config 所需的任一 path-type×tau cell 未过 gate ⇒ 整个 config 失效"替换为：

> **Required-cell 集合**在 ratings-blind feature build 期间物化：对每个 config，required
> cells = 冻结分析 manifest 中 ≥1 个 primary-eligible 场景实际查询的
> `(path_type, tau)` 组合，写入 `required_envelope_cells.csv` 并登记 SHA-256。
> required cell 未过 blind envelope gate ⇒ 该 config 在 rating join 前整体标为
> `INELIGIBLE_ENVELOPE_*`；**非 required** cell 未过 gate 仅记录，不影响 config 资格。
> required-cell 集合物化后冻结，不得因结果调整。

## B4 G0 闭环执行令

F05–F08 用既备脚本关闭：`RQ014_forensics_hpc_pass3_20260710.sh`（重试）与
`RQ014_forensics_mac_pass3b_20260710.sh`。FL05 并入同一次 HPC 会话执行。全部 surface 达到
三种合法终态后 G0 方可关闭；F09 维持 `INACCESSIBLE` + residual-risk 声明。

## B5 §11 Specificity 重构：正交性 → 三层主张

**动机（PI 异议，成立）**：IPV 由轨迹计算而来，与运动学指标相关是机制本身；若运动学是
IPV→评分的中介，"控制运动学后要求增量"会把信号本身扣除，检验失败在中介/冗余之间不可辨。
v1 的 12 项同过门（逐个赢过 7 个单项运动学 + 增量）检验的是一个过强且概念错位的命题。
IPV 的核心主张应为**简约性**：一个自由度的社会性指标达到多特征运动学组合的解释力。

§11 与 §12 相应条款替换如下。

### Tier C — 构造有效性（必须通过）

- family：4 个 negative controls（`shuffled_ipv`、`counterpart_swap`、`role_flip`、
  `sign_flip`），统计量 `A_m = S_IPV - S_control_m`，`S = max(0, -rho_WS)`；
- 9,999 次 paired scene bootstrap，studentized maxT one-sided simultaneous lower bounds，
  FWER 0.025（构造同 v1 §11.6–7，family size 4）；
- 全部 lower bound > 0 ⇒ PASS；
- `role_flip` 因 role envelope 未过 blind gate 而不可用时：family 缩为 3、记录原因，
  不再触发 v1 的"cap at association"（可用性问题不是证据问题）；可用 controls < 3 ⇒
  Tier C `UNAVAILABLE`，最高标签停在 ASSOCIATION；
- Tier C FAIL ⇒ association 结论加注 `CONSTRUCTION_SUSPECT`，不得进入 Tier P/I。

### Tier P — 简约性（核心主张，非劣性）

- 对比对象：**七特征合成** `kinematics_combined_cost` 的 within-scene rank 效应
  `S_K`（单项运动学指标只并排报告效应量，不设门）；
- 冻结非劣性边际 `delta_NI = 0.10`（= promotion 阈值 |rho_WS| 的量级；理由：容许
  IPV 比组合模型弱不超过发现门槛一个单位）；
- 统计量 `A_P = S_IPV - S_K + delta_NI`；9,999 次 paired scene bootstrap 的 one-sided
  95% lower bound > 0 ⇒ PASS；
- PASS 语义：单一 IPV 偏离指标达到（不明显劣于）7 特征运动学组合的解释力 ——
  即"一个指标表达综合倾向"的直接证据；
- `kinematics_combined_cost` 精确公式/代码 SHA 无法从 RQ010B 恢复时，Tier P 记
  `UNAVAILABLE`（沿用 v1 §11.1 的保守规则），不得以临时公式替代。

### Tier I — 增量性（可选加冕）

- 保留 v1 的 nested rank model 与 `T_inc = -beta_D`（含 `SSE_full<SSE_baseline` sanity
  check），作为**单一统计量**检验（不再并入 12 项 family）；bootstrap one-sided lower
  bound > 0 ⇒ PASS；
- FAIL 的登记措辞固定为 `AMBIGUOUS_MEDIATION_OR_REDUNDANCY`：不降级 Tier P 结论，
  不得解释为"IPV 冗余"。

### G2 校准（沿用并缩减）

`S` 在 0 处非光滑 ⇒ G2 global-null simulation（≥10,000 次）对 Tier C（family=4）与
Tier P（单统计量）分别验证 empirical FWER/size ≤ 0.03；数值失效政策同 v1
（`SPECIFICITY_NUMERICAL_INVALID` ⇒ 最高标签停在 ASSOCIATION）。

### §12 科学分支标签替换

原 `VALID_INTERNAL_CONFIRMATION_ASSOCIATION` / `..._IPV_SPECIFIC` 两档替换为层级标签：

| 条件 | 标签 |
|---|---|
| G4 association 通过 | `VALID_INTERNAL_CONFIRMATION_ASSOCIATION` |
| + Tier C 全过 | `..._ASSOCIATION_ROBUST` |
| + Tier P 非劣性过 | `..._PARSIMONIOUS_INDEX` |
| + Tier I 增量过 | `..._INCREMENTAL_BEYOND_KINEMATICS` |

`IPV_SPECIFIC` 一词废止。手稿措辞映射：PARSIMONIOUS_INDEX 及以上方可支持"单指标表达
综合社会倾向"类句式；INCREMENTAL 方可支持"超越运动学的额外信息"类句式。

---

## 生效条件

1. 本 amendment + 三个 v1p1 registry 通过独立复审（G1 重审）；
2. B4 的 G0 闭环完成、无 OPEN surface；
3. checksums 文件重生成并登记。

在此之前 registry 维持 `PROPOSED_NOT_EXECUTABLE`，不提交任何计算作业。
