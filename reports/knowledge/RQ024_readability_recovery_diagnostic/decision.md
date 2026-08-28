# RQ024 Decision

Date: 2026-08-24
Status: ACCEPTED
PI acceptance source: user message on 2026-08-24
Verdict: MIXED_DIAGNOSTIC
Tier2 status: denied

## Problem定位

RQ024 要解决的问题是：在 WP2 Tier1 已封存 synthetic 结果上，解释 readability-style gate 为什么没有起到“误差质量控制”作用，反而在更严格时抬高 `risk_mae_rad`。整体阶段上，WP2 Gate A 已先被合同裁定失败；本次是失败后的 bounded diagnostic 知识层接受，不是新实验批准。

本 decision 只接受 sealed synthetic Tier1 诊断结论；不改 comparator，不改阈值，不重估，不新生成 synthetic，不重开任何已接受的 `RQ017/RQ018/RQ019/RQ021`。

## Accepted Claims

| Claim ID | Accepted claim | Evidence | Boundary |
| --- | --- | --- | --- |
| `RQ024-KC-C1` | Sealed synthetic Tier1 engineering health 在 `288/288` 行上完成，但 Gate A 合同失败；相邻更严格阈值比较共 `42` 组，其中 `36/42` 组 `risk_mae_rad` 上升，因此 Tier2 继续阻断。 | `diagnostic_summary.json`；`VERDICT.json`；`ADJUDICATION.md` | 这只是 synthetic Tier1 合同诊断，不是对真实数据或真实系统性能的判断。 |
| `RQ024-KC-C2` | `q_eff`、`k_eff`、`ipv_error` 是近乎同一单调“浓度”指标，不是彼此独立的精度证据。接受的精确事实：Spearman(`q_eff`,`ipv_error`) = `0.999990952350554`，Spearman(`q_eff`,`k_eff`) = `1.0`，Spearman(`ipv_error`,`k_eff`) = `0.9999909523505539`。 | `diagnostic_summary.json`；`metric_association.csv` | 本 claim 只说明指标同构，不说明哪一个可单独代表真实精度。 |
| `RQ024-KC-C3` | 噪声成对失配与严格门边界/高误差富集同时成立：在 `144` 个噪声成对单元里，`115/144` 出现“误差更差但至少一个 gate 指标朝更安全方向移动”；严格 `q_eff < 0.4` 通过 `69/288`，其中边界 `5/69`、近边界 `11/69`、高误差 `27/69`；严格 `ipv_error <= 0.2` 通过 `29/288`，其中边界 `5/29`、近边界 `9/29`、高误差 `13/29`。 | `paired_noise_deltas.csv`；`boundary_summary.json`；`evidence_summary.csv` | 接受的是“严格门富集边界/近边界高误差行”这一描述性机制，不是因果主张。 |
| `RQ024-KC-C4` | 综合判定接受为 `MIXED_DIAGNOSTIC`：当前 readability gate 不能被描述为 accuracy QC；它在本 sealed synthetic grid 上混合了浓度指标、噪声混杂与边界效应。已接受的 `RQ017+` 结论不因此被重开、推翻或降级。 | `report.md`；`conclusions.md`；`synthesis.md` | 这是 synthetic/descriptive 边界内的接受，不外推到真实 AV / human 结论。 |

## Evidence And Boundary Table

| Item | Numerator | Denominator | Filter / meaning | Source |
| --- | ---: | ---: | --- | --- |
| Tier1 rows completed | 288 | 288 | all rows in sealed `tier1_results.csv` | `diagnostic_summary.json` |
| Gate A adjacent stricter failures | 36 | 42 | adjacent stricter comparisons under `q_eff_lt` or `ipv_error_le` | `VERDICT.json` |
| Noise-paired mismatch | 115 | 144 | paired cells with error up and at least one metric moving safer | `diagnostic_summary.json` |
| Argmax unresolved | 286 | 288 | `INCONCLUSIVE_ARGMAX`; no raw `weights` in `tier1_results.csv` | `boundary_summary.json` |

## Accepted Boundaries

- 只接受 boundary-distance / boundary-saturation mechanism；**不**接受“所有失败都已精确定位为 argmax 选边界”，因为 `286/288` 行仍是 `INCONCLUSIVE_ARGMAX`。
- 只接受 synthetic、描述性、工程合同层结论；**不**接受因果、泛化到真实系统、或对真实 AV / human 行为优劣的表述。
- 本 decision 不推翻、不修订、不重审任何已接受的 `RQ017/RQ018/RQ019/RQ021`。

## Forbidden Phrasings

- 不得写“当前 readability gate 能做 accuracy QC”。
- 不得写“已经证明 argmax 机制导致全部失败”。
- 不得写“真实 AV / human 数据也会如此”。
- 不得写“RQ017+ 已被本结果推翻、动摇或重开”。

## Downstream Directive

Tier2 继续 denied。除非未来有**单独批准的新合同**明确替换或修复 Gate A，否则不得把当前 readability gate 当作 accuracy quality-control 证据，不得据此推进 Tier2。

## Official Links

- Run entry: `reports/studies/RQ024_readability_recovery_diagnostic/RQ024_1_bounded_diagnostic_20260824T065142Z_3698873/00_entry/index.html`
- Official report: `reports/studies/RQ024_readability_recovery_diagnostic/RQ024_1_bounded_diagnostic_20260824T065142Z_3698873/01_results/report.md`
- Evidence summary: `reports/studies/RQ024_readability_recovery_diagnostic/RQ024_1_bounded_diagnostic_20260824T065142Z_3698873/01_results/evidence_summary.csv`
- Process archive: `archived/report_process/RQ024_readability_recovery_diagnostic_RQ024_1_bounded_diagnostic_20260824T065142Z_3698873`
- Plan / contract: `.codex-fleet/nmi-revision-research-lead/work/WP2_recovery/design/ACCEPTANCE.md`
- Knowledge synthesis: `reports/knowledge/RQ024_readability_recovery_diagnostic/reviews/synthesis.md`
