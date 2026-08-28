# RQ025 Decision

- Date: `2026-08-24`
- Status: `ACCEPTED`
- Verdict: `BOUNDED_MIXED`
- PI acceptance source: `user message on 2026-08-24`

## Context

RQ025评估的是 AV-only、episode-onset matched consequence estimand。到本决定为止，冻结匹配、accepted-contract outcome regeneration、pair assembly、point estimates、treated-case clustered bootstrap、numeric review 与 method review 均已完成。当前决定接受的是这一整套结果作为知识层结论；不新增 `paper` 结论，也不改写其他 RQ 的已接受结论。

官方 study run：
`reports/studies/RQ025_av_episode_consequence/RQ025_1_matched_episode_20260824T064959Z_3698873/`

官方 process archive：
`archived/report_process/RQ025_1_matched_episode_20260824T064959Z_3698873/`

关键 review：
- `reports/knowledge/RQ025_av_episode_consequence/reviews/numeric_review.md`
- `reports/knowledge/RQ025_av_episode_consequence/reviews/method_review.md`
- `reports/knowledge/RQ025_av_episode_consequence/reviews/synthesis.md`

计划依据：
- `reports/plans/RQ025_plan_v0_av_episode_consequence_20260824.md`
- `reports/plans/RQ025_plan_v0p1_ego_contract_correction_20260824.md`

## Accepted Claims

| Claim ID | Accepted claim | Evidence anchor |
|---|---|---|
| `RQ025-KC-C1` | Frozen any-side `gap=10` matched design is accepted as outcome-blind and deterministic: primary is `461/475` same-scenario `q50`; sensitivity is `361/475` same-run `q50`. | official run `01_results/effect_results.csv`; archive `frozen_estimand/` |
| `RQ025-KC-C2` | Accepted-contract outcome regeneration is accepted, including complete denominators and the full seven-outcome family over all `24` branch/subset interval rows. | official run `01_results/effect_results.csv`; `01_results/evidence_summary.csv`; archive `ego_accepted_contract/`, `counterpart_regeneration/`, `pair_assembly/` |
| `RQ025-KC-C3` | All `24/24` reported 95% intervals include zero. Ego `future_min_ttc_lt_2s` risk differences are primary `-0.010667`, 95% CI `[-0.041899, 0.019289]`, `n=375`; sensitivity `-0.016835`, 95% CI `[-0.0511, 0.012545]`, `n=297`. Therefore current evidence does not support a stable positive episode-level consequence contrast. | official run `01_results/effect_results.csv`; `01_results/statistics_review.md` |
| `RQ025-KC-C4` | Other routine outcomes are mixed or branch-flipping, and ego TTC continuous summaries show long-tail mean/median divergence; overall classification is `BOUNDED_MIXED`. | official run `01_results/report.md`; `01_results/conclusions.md`; reviews `synthesis.md` |

## Stable Readout

本轮接受的最稳妥表述是：

“在冻结匹配账本与 treated-case clustered bootstrap 下，七个预先列出的结局整体呈 bounded/mixed 描述性对比，当前结果不支持稳定正向 episode-level consequence contrast。”

## Boundaries

1. `CI` 跨零不等于等效，也不等于“没有效应”。
2. matched pair row 是描述单位；treated case 是不确定性聚类单位。
3. frozen ledger 允许 with-replacement control reuse；区间是在这一限制下的 descriptive treated-case clustered intervals。
4. 禁止因果、非劣、incremental-value、safety improvement、vehicle/team judgment 表述。
5. 本结论仅适用于 AV-only RQ025 estimand。
6. RQ018 / RQ019 的已接受 frame-level claims 在这里既未被复制，也未被推翻，因为 estimand 不同。

## Allowed Wording

- `BOUNDED_MIXED`
- `descriptive contrast`
- `does not support a stable positive episode-level consequence contrast`
- `mixed / branch-flipping / long-tail-sensitive`

## Forbidden Wording

- `证明更安全`
- `稳定改善`
- `等效`
- `非劣`
- `没有效应`
- `因而导致`
- `优于人类`

## Verification

- Official result table path exists and is the accepted reader table: `reports/studies/RQ025_av_episode_consequence/RQ025_1_matched_episode_20260824T064959Z_3698873/01_results/effect_results.csv`
- Accepted evidence row count is `24`
- No placeholder text is accepted into this decision
