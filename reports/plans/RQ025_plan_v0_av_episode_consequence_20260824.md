# RQ025 Plan v0 (WP7 AV-only existing-data consequence analysis) — approved 2026-08-24

Status: `APPROVED / READY`
Scope: `RQ025_wp7_consequence`
User approval date: `2026-08-24`
protected_data=NONE
human_collection=denied
causal_claim=denied

This plan freezes the user-approved WP7 episode-onset and matching designs and then regenerates row-level outcomes from existing local data only. It does not reopen the episode contract, the matching contract, or the caliper contract. It does not edit paper files.

## 1. Context

RQ025 is a consequence-analysis follow-on to the approved WP7 AV-only existing-data design. The frozen pair tables already exist and are outcome-blind. The only remaining work is to regenerate the approved outcome ledgers from accepted raw sources and then analyze matched consequence contrasts at the episode-onset row.

The frozen design facts that must remain fixed are:

- Primary design: `any_side_combined`, `gap_frames=10`, `same_scenario`, `with_replacement`, `q50`
- Sensitivity design: `any_side_combined`, `gap_frames=10`, `same_run`, `with_replacement`, `q50`
- Primary pair count: `461/475` treated episodes matched
- Sensitivity pair count: `361/475` treated episodes matched
- Clustering unit: `case_id`

These pair counts are pre-outcome facts. They are denominators for the outcome work and must not change during regeneration or analysis.

## 2. Research Questions

### RQ025-1

What descriptive consequence contrasts are observable on the frozen AV-only episode-onset pairs after row-level outcomes are regenerated from accepted existing data?

### RQ025-2

Does the same-run sensitivity branch preserve the same descriptive direction without reopening the episode, matching, or caliper contracts?

## 3. Unit and hierarchy

- Analysis unit: `episode_onset_row`
- Nesting: `frames -> episode onsets -> case_id -> run_id`
- Clustering: `case_id`
- Pair table status: outcome-blind only
- Exposure backbone: two-gate accepted AV-only analysis rows
- Protected data: none

## 4. Exact existing-data sources

Use only the following existing sources and their accepted logic:

### 4.1 Frozen pair tables

- `.codex-fleet/nmi-revision-research-lead/work/RQ025_wp7_consequence/frozen_design/primary_pairs.parquet`
- `.codex-fleet/nmi-revision-research-lead/work/RQ025_wp7_consequence/frozen_design/primary_pairs.csv`
- `.codex-fleet/nmi-revision-research-lead/work/RQ025_wp7_consequence/frozen_design/sensitivity_pairs.parquet`
- `.codex-fleet/nmi-revision-research-lead/work/RQ025_wp7_consequence/frozen_design/sensitivity_pairs.csv`

### 4.2 Two-gate analysis backbone

- `data/derived/rq017_onsite_gate/l1_v1`
- `.codex-fleet/rq021-contemporaneous-envelope/work/E1/onsite_scoring_dryrun.parquet`

Accepted join keys:

- preferred: `product_row_key`
- fallback: `case_key + anchor_frame_index + perspective`

Accepted two-gate analysis rows:

- `status == OK`
- `mechanism2_gate_ok == true`

### 4.3 Ego future-window outcome sources

- `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet`
- `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_ipv_timeseries_multi_allvalid.parquet`
- accepted row-level reconstruction logic from `reports/studies/RQ018_abnormal_ipv_degradation/RQ018_1_association_20260804T224427Z_276cf4c/rq018_supervisor_verification.py`

### 4.4 Counterpart future-window outcome sources

- `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet`
- `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_ipv_timeseries_multi_allvalid.parquet`
- `data/onsite_competition/all_teams_dataset/teams/*/*/sessions/*/simulation_trajectory.log`
- accepted row-level reconstruction logic from `reports/studies/RQ019_counterpart_burden/RQ019_1_counterpart_burden_20260805T014215Z_7b9f47b/rq019_supervisor_verification.py`

### 4.5 Provenance and missingness boundaries

- `reports/studies/RQ019_counterpart_burden/RQ019_1_counterpart_burden_20260805T014215Z_7b9f47b/alignment_contract.json`
- `reports/studies/RQ019_counterpart_burden/RQ019_1_counterpart_burden_20260805T014215Z_7b9f47b/data_health.json`
- `reports/studies/RQ018_abnormal_ipv_degradation/RQ018_1_association_20260804T224427Z_276cf4c/data_health.json`

## 5. Outcome regeneration tasks

### 5.1 Ego ledger

Regenerate the ego future-window ledger from the accepted RQ018 reconstruction logic and keep row-level keys intact.

Required row-level fields:

- `product_row_key`
- `case_key`
- `anchor_frame_index`
- `perspective`
- `future_min_ttc_s`
- `ttc_lt_2s`
- `future_min_ttc_missing`

The row-level output must be derived from the raw anchor/time-series sources, not from figure caches or summary JSON.

### 5.2 Counterpart ledger

Regenerate the counterpart 3 s future-window ledger from the accepted RQ019 reconstruction logic and keep row-level provenance intact.

Required row-level fields:

- `product_row_key`
- `case_key`
- `anchor_frame_index`
- `perspective`
- `session_id`
- `counterpart_key_agent`
- `anchor_ts_ms`
- `is_scripted`
- `speed_range_kmh`
- `anchor_speed_drop_kmh`
- `min_accel`
- `brake_share_2`
- `brake_share_3`
- `counterpart_outcome_missing`

If the accepted reconstruction also yields `brake_share_4`, retain it only as an explicit row-level field, not as a substitute for the raw ledger.

### 5.3 Join discipline

- Join the frozen pair table to the analysis backbone by `product_row_key` when available.
- Use the natural fallback key only when `product_row_key` is absent.
- Never use the figure caches as a pair source; they lose the keys needed for unambiguous episode-onset joins.

## 6. Effect analysis requirements

The analysis is descriptive consequence analysis only.

Required analysis behavior:

- Report matched treated/control consequence contrasts at the episode-onset row.
- Cluster uncertainty by `case_id`.
- Keep primary and sensitivity branches separate.
- Report the full predeclared outcome family together; do not cherry-pick outcomes by the best p-value.
- Treat any inferential summary as descriptive support, not as a claim of causality, non-inferiority, equivalence, or positive incremental value.

Outcome family to carry through if regenerated from the accepted raw sources:

- `future_min_ttc_s`
- `ttc_lt_2s`
- `min_accel`
- `speed_range_kmh`
- `anchor_speed_drop_kmh`
- `brake_share_2`
- `brake_share_3`

Missingness requirements:

- Report row counts before and after regeneration.
- Report missingness by outcome and by branch.
- Preserve provenance flags for missing log or timing-alignment rows.
- Keep row-level denominators visible in every table.

Multiplicity requirements:

- No post-hoc outcome selection.
- No re-ranking of outcomes after seeing the values.
- No collapsing multiple outcomes into a single winner narrative.
- If any p-values are reported, they must appear alongside the full outcome family and the exact denominators.

## 7. Expected outputs

Owned output targets:

- `.codex-fleet/nmi-revision-research-lead/work/RQ025_wp7_consequence/outcome_schema/ego_contract_outcome_ledger.parquet`
- `.codex-fleet/nmi-revision-research-lead/work/RQ025_wp7_consequence/outcome_schema/counterpart_fixed3_outcome_ledger.parquet`

Supporting artifacts already delivered or to be refreshed in place:

- `.codex-fleet/nmi-revision-research-lead/work/RQ025_wp7_consequence/outcome_schema/OUTCOME_SOURCE_MAP.csv`
- `.codex-fleet/nmi-revision-research-lead/work/RQ025_wp7_consequence/outcome_schema/JOIN_CONTRACT.json`
- `.codex-fleet/nmi-revision-research-lead/work/RQ025_wp7_consequence/outcome_schema/REGENERATION_PLAN.md`
- `.codex-fleet/nmi-revision-research-lead/work/RQ025_wp7_consequence/outcome_schema/EXECUTION_BRIEF.md`

## 8. One-pass checks

Run one verification pass only.

Checks:

1. Confirm the frozen pair counts remain `461/475` and `361/475`.
2. Confirm the join contract still works with `product_row_key`, and only falls back to `case_key + anchor_frame_index + perspective` when necessary.
3. Confirm the regenerated ledgers keep row-level keys and provenance flags.
4. Confirm the ego ledger is future-only and the counterpart ledger is strict future-window only.
5. Confirm `case_id` is still present for clustering and is not dropped during output shaping.
6. Confirm the file contains no placeholders or deferred text.

## 9. Stop gates

Stop immediately if any of the following appears:

- A request to reopen episode definition, matching pool, caliper, or replacement rule
- A request to use human collection, protected data, or any human-arm raw source
- A request to use a cache that lacks the row keys needed for pair-level joins
- A request to add causal, NI, equivalence, or positive incremental-value language
- A request to turn this into a paper edit
- A request to change the frozen pair counts or denominators
- A request to expand beyond the approved existing-data sources listed above

## 10. Claim boundaries

Allowed:

- AV-only existing-data consequence contrasts
- Row-level future-window outcomes regenerated from accepted raw sources
- Primary and same-run sensitivity comparisons on the frozen matched design

Not allowed:

- Human comparison
- Causal language
- Non-inferiority or equivalence claims
- Positive incremental-value claims
- Official-score or harm-as-positive-outcome claims
- Paper edits
- Vehicle, driver, or team judgments

This plan is self-contained and approved for execution as written.
