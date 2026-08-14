# E6 run-ledger findings

Date of read-only audit: 2026-08-11  
Research repository audited: `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation`

Labels used below: **PROVEN** means directly traceable to stored rows/fields or executable selection logic; **INFERRED** means consistent with recorded data but not explicitly recorded as the cause; **UNKNOWN** means the necessary artefact is absent.

## 1. Bottom line

### Gap 1: 285 expected -> 267 represented

**PROVEN:** The 18 are not unattempted or unscored competition runs: all 285 clean replay-eligible `algorithm × scenario` cells were submitted to the anchor builder, and the 18 omitted cells all have official score/PDF records—15 scored completions and 3 collision failures. **PROVEN:** Their recorded proximate exclusion reasons at the replay-analysis stage are 11 degenerate observed references, 3 cases with no timing-valid anchors, 3 with no eligible counterpart, and 1 with no timing-eligible counterpart; the loss is strongly system-concentrated in T8 (13/18), not caused by a dropped scenario.

### Gap 2: the asserted 240 -> 175

**PROVEN:** `240` is not a coherent stored cohort: the full 267-case anchor universe contains 30 scripted-labelled cases (hence 237 non-scripted), whereas `27` is the scripted count only after 36 cases have already failed the two monitor gates. **PROVEN:** The actual funnel is `267 -> 231 gate-eligible -> 204 non-scripted -> 175 with RQ019 counterpart-motion outcomes`; the apparent 65 combines 36 earlier gate exclusions with 29 cases whose selected counterpart ID has no raw `simulation_trajectory.log` series, while the correct full-universe non-scripted attrition is 62 = 33 non-scripted no-gate cases + 29 raw-series-missing cases.

This is therefore an accounting-definition problem as well as a missing-data problem. In particular, “no recorded outcome” does **not** mean “no competition result”: all 29 raw-series-missing cases have official scores, with 27/29 recorded as scored completions and 2/29 as collision failures.

## 2. Evidence trail for Gap 1

### 2.1 The denominator is a matched-cell universe, not an identifiable repeated-run ledger

The official outcome table has 300/300 score rows: 20 teams × 15 scenarios, with all six official score columns present. The replay universe then retains 285/300 cells—19 teams × 15 scenarios—because all 15 T19 cells are replay-only exclusions; the accepted RQ011 decision defines the primary unit as `algorithm × scenario` and explicitly says repeated-run, seed-level and run-level claims are not identifiable.

| Finding | Numerator / denominator and filter | Absolute source | Columns / keys |
|---|---|---|---|
| Official outcome universe | 300/300 rows; no filter; 20 distinct `team_code`, each with 15 `scenario` rows | `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/onsite_competition/all_teams_dataset/tables/all_scenario_scores.csv` | `team_code`, `scenario`, `safety`, `efficiency`, `comfort`, `compliance`, `coordination`, `comprehensive` |
| Replay-clean universe | 285/300 rows where `corrected_clean == true`; 19 teams × 15 scenarios; excluded 15/300 are all T19 | `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ011_onsite_full_universe_readiness/RQ011_2_onsite_readiness_20260623T201415+0800_efdd75a5/02_process/04_mapping/corrected_clean_mask.csv` | `unit_composite_key`, `team_id`, `scenario_id`, `run_id`, `session_id`, `corrected_clean` |
| Unit and identifiability contract | Accepted claims define `algorithm × scenario`, `full_300`, `clean_285`, and prohibit run-level claims | `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/knowledge/RQ011_onsite_full_universe_readiness/decision.md` | `RQ011-KC-UNIT`, `RQ011-KC-OUTCOME-300`, `RQ011-KC-REPLAY-285`, `RQ011-KC-IDENTIFIABILITY` |

Thus, the repository does not support describing the 285 as 285 independent physical sessions. It supports 285 replay-eligible system–scenario cells drawn from 19 selected sessions.

### 2.2 The decisive 18-row failure ledger

The anchor-builder coverage file requests 285 units, records 267 units with anchors, 67,861 anchors, and 18 failures. The current RQ017 gate shards independently contain the same 67,861 rows and 267 unique `case_id`; their 18-cell complement against `corrected_clean == true` exactly matches `coverage.json.failures[]`.

| Finding | Numerator / denominator and filter | Absolute source | Columns / keys |
|---|---|---|---|
| Builder accounting | `units_requested=285`; `units_with_anchors=267`; `ipv_cases_failed=18`; `total_av_anchors=67861`; no row filter | `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/hpc_onsite_ipv/process_allvalid_processpool_amd/coverage.json` | top-level counts; `failures[].unit_composite_key`, `.scenario_id`, `.native_replay_case_id`, `.stage`, `.error` |
| Independent current materialization | 67,861/67,861 rows across 136 parquet shards; 267 distinct `case_id`; filter `artifact_id == onsite_dense_timeseries` | `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/rq017_onsite_gate/l1_v1/artifact_id=onsite_dense_timeseries/shard_id=*/part-0.parquet` | `artifact_id`, `case_id`, `product_row_key` |
| Current case denominator | `denominator_cases=267`, `denominator_frames=67861` | `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ017_onsite_mechanism_one/RQ017_1_onsite_gate_20260804T075311Z_406e7a65/case_level_availability.json` | `denominator_cases`, `denominator_frames` |

The recorded builder mechanisms are:

| Recorded mechanism (`coverage.json.failures[].error`) | Count / 18 | Stage |
|---|---:|---|
| `ValueError: observed reference has fewer than two unique points` | 11/18 | `ipv_or_anchor_build` |
| `ValueError: no valid anchors after RQ009 timing filters` | 3/18 | `ipv_or_anchor_build` |
| `no_eligible_counterpart` | 3/18 | `parse_or_counterpart` |
| `no_timing_eligible_counterpart` | 1/18 | `parse_or_counterpart` |

The implementation makes these labels auditable: counterpart timing eligibility requires at least 10 aligned frames; the reference path removes consecutive duplicate coordinates and fails below two unique points; anchor validity requires sufficient history and a future target offset. Source: `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/hpc_onsite_ipv/build_onsite_m3_anchors_hpc.py`, functions `eligible_counterpart`, `select_counterpart`, `prepared_reference`, `valid_anchor_positions`, and `process_case_for_run`.

### 2.3 Concentration by system and scenario

System counts use the 18 `coverage.json.failures[]` rows grouped by `team_id` parsed from `unit_composite_key`; the denominator is 15 clean cells per included system from `corrected_clean_mask.csv`.

| System | Missing / expected | Present with anchors | Failure composition |
|---|---:|---:|---|
| T8 | 13/15 | 2/15 | 11 degenerate references; 2 no timing-valid anchors |
| T12 | 2/15 | 13/15 | 2 no eligible counterpart |
| T10 | 1/15 | 14/15 | 1 no timing-eligible counterpart |
| T20 | 1/15 | 14/15 | 1 no timing-valid anchor |
| T21 | 1/15 | 14/15 | 1 no eligible counterpart |
| Other 14 systems | 0/210 | 210/210 | none |

**PROVEN:** T8 contributes 13/18 = 72.22% of the gap. This is system/session concentration, but it is concentration of an analysis failure—not evidence that the T8 team failed to attempt 13 scenarios.

Scenario counts use `failures[].scenario_id`, with 19 expected replay-clean systems per scenario:

| Scenario | Missing / 19 | Scenario | Missing / 19 | Scenario | Missing / 19 |
|---|---:|---|---:|---|---:|
| A1 | 0/19 | A2 | 1/19 | A3 | 1/19 |
| A4 | 0/19 | A5 | 1/19 | A6 | 1/19 |
| A7 | 1/19 | B1 | 1/19 | B2 | 1/19 |
| B3 | 2/19 | B4 | 1/19 | C1 | 1/19 |
| C2 | 1/19 | C3 | 3/19 | C4 | 3/19 |

**PROVEN:** Every scenario remains represented by 16–19 systems; no scenario was globally dropped.

### 2.4 The 18 were attempted and scored

Joining the 18 failure keys to the official outcome recoding ledger gives `S2_full_scored == true` and `pdf_summary_match_status == PASS` for 18/18. Fifteen have `success_failure == success` and `mission_status == scored_completion`; three—T21/C4, T12/C3 and T12/C4—have `collision_flag_score0 == true`, `mission_status == collision_failure_score0`, and `official_comprehensive == 0`.

| System/scenario | Official result | Builder failure |
|---|---|---|
| T20/B3 | scored completion, comprehensive 96.26 | no valid anchors after timing filters |
| T21/C4 | collision failure, 0.00 | no eligible counterpart |
| T8/A2 | scored completion, 88.09 | no valid anchors after timing filters |
| T8/A3 | scored completion, 79.05 | degenerate reference |
| T8/A5 | scored completion, 88.17 | degenerate reference |
| T8/A6 | scored completion, 85.46 | no valid anchors after timing filters |
| T8/A7 | scored completion, 96.10 | degenerate reference |
| T8/B1 | scored completion, 69.99 | degenerate reference |
| T8/B2 | scored completion, 75.98 | degenerate reference |
| T8/B3 | scored completion, 80.83 | degenerate reference |
| T8/B4 | scored completion, 67.52 | degenerate reference |
| T8/C1 | scored completion, 83.71 | degenerate reference |
| T8/C2 | scored completion, 93.89 | degenerate reference |
| T8/C3 | scored completion, 60.76 | degenerate reference |
| T8/C4 | scored completion, 72.77 | degenerate reference |
| T10/C3 | scored completion, 96.78 | no timing-eligible counterpart |
| T12/C3 | collision failure, 0.00 | no eligible counterpart |
| T12/C4 | collision failure, 0.00 | no eligible counterpart |

Provenance for all 18 results: `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/onsite_competition/RQ011_onsite_full_universe_readiness/RQ011_2_onsite_readiness_20260623T201415+0800_efdd75a5/05_fields/outcome_recoding_by_unit.csv`; filter `unit_composite_key` in `coverage.json.failures[]`; columns `S2_full_scored`, `pdf_summary_match_status`, `success_failure`, `mission_status`, `collision_flag_score0`, `official_comprehensive`, `pdf_collision_deduction`, `pdf_task_completion_score`, `source_locator`.

All five affected sessions also have the full required log set: 5/5 rows have `required_logs_present_current == true` and an empty `missing_required_logs_current`. Source: `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/onsite_competition/all_teams_dataset/tables/all_session_manifest.csv`; filter `session_id` in `{6932-1766206403, 6948-1766217297, 6953-1766219816, 6941-1766212682, 6934-1766207726}`; columns `required_logs_present_current`, `missing_required_logs_current`, `present_logs`.

No per-attempt raw manifest supporting repeated-run or seed-level claims was found in the audited repository. What does exist is sufficient for the present question: a 285-cell replay manifest, a distinct 18-row builder failure ledger, and official result/PDF locators proving all 18 were attempted and scored.

### 2.5 A nearby five-failure ledger is not the explanation

`/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/onsite_competition/RQ012B_event_harm/scenario_extraction_status.csv` has 280 `status == success` rows and 5 failures (`error_class`: two `MissingEgoRows`, three `MissingCaseWindow`). It is a different extraction stage and explains only three overlapping collision cells; it must not be cited as the source of the full 18-cell gap.

## 3. Evidence trail for Gap 2

### 3.1 The documented `240` denominator mixes two stages

The full anchor table has 267 unique `case_key`: 30 are labelled scripted and 237 are labelled non-scripted by exact `counterpart_selection` string. After joining RQ017 and RQ021 by `product_row_key` and filtering `status == "OK" AND mechanism2_gate_ok == true`, only 231 cases remain; within that selected cohort the scripted count is 27 and the non-scripted count is 204.

| Funnel stage | Numerator / denominator and filter | Absolute source | Columns / keys |
|---|---|---|---|
| Full anchor universe | 267 unique `case_key` / 67,861 rows; no filter | `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet` | `case_key`, `anchor_frame_index`, `perspective`, `source_dataset`, `counterpart_selection` |
| Scripted at full-universe stage | 30/267 unique cases and 7,305/67,861 rows where `counterpart_selection == "online_first_conflict_nearest_timing_eligible_prefer_scripted_from_vehicle"` | Same anchor parquet; count also stored in `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/.codex-fleet/rq021-contemporaneous-envelope/work/E1/rq019_rerun/data_health.json` | `case_key`, `counterpart_selection`; JSON `load_contract.all_anchor_counterpart_selection_counts` |
| Two-gate analysis set | 14,099/67,861 frames and 231/267 cases after `status == "OK" AND mechanism2_gate_ok == true` | `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/rq017_onsite_gate/l1_v1/artifact_id=onsite_dense_timeseries/shard_id=*/part-0.parquet` joined to `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/.codex-fleet/rq021-contemporaneous-envelope/work/E1/onsite_scoring_dryrun.parquet` | `product_row_key`, `status`, `ipv_log`, `mechanism2_gate_ok`, `lo_90`, `hi_90` |
| Script split after gates | Scripted 1,075/14,099 frames and 27/231 cases; non-scripted 13,024/14,099 frames and 204/231 cases | Anchor join above; frozen counts in `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/.codex-fleet/rq021-contemporaneous-envelope/work/E1/rq019_rerun/data_health.json` and `key_numbers.json` | `product_row_key`, `case_key`, `counterpart_selection`; JSON `analysis_set.strata` / registered key records |
| Stored non-scripted outcomes | 11,671 rows and 175 unique `case_key`; rows by `band`: 10,483 inside, 469 lower, 719 upper | `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ021_contemporaneous_envelope/RQ021_1_contemporaneous_envelope_20260805T160425Z_43b4bff/fig5_counterpart_outcomes.parquet` | `case_key`, `band`, `is_scripted`, `speed_range_kmh`, `anchor_speed_drop_kmh`, `max_abs_yaw_rate_dps`, `total_heading_change_deg`, `min_accel`, `brake_share_2`, `brake_share_3` |

The arithmetically tempting `267 - 27 = 240` subtracts a **post-gate** scripted count from a **pre-gate** case universe. At the full-universe stage the correct calculation is `267 - 30 = 237`; at the post-gate stage it is `231 - 27 = 204`.

### 3.2 What the apparent 65 actually contains

The apparent `65 = (267 - 27) - 175` decomposes as follows:

| Component | Count | Status and mechanism |
|---|---:|---|
| Cases with no frame passing both monitor gates | 36/267 | **PROVEN earlier gate exclusion.** RQ017 records 231/267 cases with at least one jointly judgeable frame and 36/267 with none. Of these 36, 3 are scripted and 33 non-scripted. |
| Post-gate, non-scripted cases lacking a raw counterpart-motion outcome | 29/204 | **PROVEN raw-series absence.** Each has one or more two-gate frames, but its exact `session_id::counterpart_key_agent` key is absent from the raw-series diagnostics built from `simulation_trajectory.log`. |
| Apparent total | 65 | 36 + 29; this is not a homogeneous “outcome missing” class. |
| Correct full non-scripted attrition | 62/237 = 26.16% | 33 non-scripted no-gate cases + 29 raw-series-missing cases. |

The three scripted-labelled/no-gate cases that are accidentally pulled into the artificial 65 are T11/A7 (`onsite:shanghai:T11:A7:native_case:2319`), T2/A1 (`onsite:shanghai:T2:A1:native_case:2325`), and T2/A7 (`onsite:shanghai:T2:A7:native_case:2319`). They are not members of the full-universe non-scripted denominator.

The 36-case gate result is frozen in `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/knowledge/RQ017_onsite_mechanism_one/decision.md`: 231/267 cases have at least one jointly judgeable frame and 36/267 do not; the limitation is human-reference support, not mechanism-one solvability (0/36 are wholly unsolvable by mechanism one). The columns used to construct this are `case_id`, `status`, `product_row_key` from the RQ017 gate shards plus `mechanism2_gate_ok` from the RQ021 scoring parquet.

For the 29, a reproducible case-level set difference is:

1. Join the RQ017 shards to `onsite_scoring_dryrun.parquet` on `product_row_key`; filter `status == "OK" AND mechanism2_gate_ok == true`.
2. Parse `case_key`, `anchor_frame_index`, and `perspective` from `product_row_key`, then join the full anchor parquet to add `session_id`, `counterpart_key_agent`, and `counterpart_selection`.
3. Remove the exact scripted string and compare the resulting 204 unique cases with the 175 unique `case_key` in `fig5_counterpart_outcomes.parquet`.
4. Form `session_id + "::" + counterpart_key_agent` and test membership in `data_health.json -> alignment_summary.series_diagnostics`.

This produces 1,330/13,024 non-scripted two-gate frames in 29/204 cases with `series_found == false`, versus 11,694/13,024 frames in 175/204 cases with `series_found == true`; there are no cross-cells. The same JSON records `counterpart_series_not_found_rows=1330`, `counterpart_series_found_rows=12769`, `matched_anchor_rows=12750`, and `nearest_time_diff_ge_150ms_rows=19` for the full 14,099-frame two-gate cohort.

Absolute source for the raw-series diagnostics: `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/.codex-fleet/rq021-contemporaneous-envelope/work/E1/rq019_rerun/data_health.json`; keys `alignment_summary.counterpart_series_not_found_rows`, `.counterpart_series_found_rows`, `.matched_anchor_rows`, `.nearest_time_diff_ge_150ms_rows`, `.series_diagnostics`. The implementation loads `simulation_trajectory.log` by exact vehicle ID and skips rows when that ID has no series: `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/.codex-fleet/rq021-contemporaneous-envelope/work/E1/rq019_rerun/rq019_supervisor_verification.py`, functions `load_session_series` and `compute_outcomes`.

The 29 exact cases are:

- T1: A1, A3, B4, C1; T3: A1, B4; T4: A1, B4; T5: A1, A3, B4, C1; T7: A1, B4; T9: A1, A3, B4.
- T12: A1, B4; T14: A6; T15: B1; T17: C1; T18: A6; T20: C4; T21: A6, B1, C1, C2, C3.

By scenario, these are A1 7/29, B4 7/29, C1 4/29, A3 3/29, A6 3/29, B1 2/29, and C2/C3/C4 1/29 each. These counts use the 29-case set above grouped by the anchor-table `scenario_id` column.

### 3.3 Candidate mechanisms tested

| Candidate explanation | Verdict | Evidence |
|---|---|---|
| No counterpart-interaction event occurred | **Not supported as the 29-case mechanism.** | All 29 have selected `counterpart_key_agent` values and 1+ frames passing both gates in the anchor/gate join. What is missing is the same ID in the separate raw simulator-control log. |
| Early termination, collision, off-course, timeout or disqualification | **Not the general mechanism.** | Joining the 29 `unit_composite_key` values to `outcome_recoding_by_unit.csv` gives `S2_full_scored == true` and `pdf_summary_match_status == PASS` for 29/29; 27/29 are `mission_status == scored_completion`, while 2/29—T21/C1 and T3/B4—are `collision_failure_score0`. No stored field links either collision to raw-series absence. |
| Whole-session logging loss | **Ruled out for Gap 1; not demonstrated for the 29.** | The five sessions behind Gap 1 have all required logs. For the 29, the RQ019 loader finds other raw vehicle series in the sessions but not the selected exact IDs; this is ID/series-level, not evidence that the whole session log is absent. |
| Raw logging or cross-log identity/coverage mismatch | **PROVEN at the observable boundary.** | The counterpart is selected from `vehicle_perception_simulation_trajectory.log`; the RQ019 outcome is sought by exact ID in `simulation_trajectory.log`; the 29 IDs are absent from the latter diagnostics, accounting for exactly 1,330 two-gate frames. Whether this is true dropout, namespace mismatch, or recorder filtering is **UNKNOWN**. |
| Pair never closed / undefined TTC-like margin | **Not the same mechanism.** | The 29 lack a raw counterpart series before a post-window outcome can be computed. The separate 47/519 count is created only after a dense trajectory exists and is a frame-level TTC definition issue. |
| Earlier analytical gate | **PROVEN for 36/65.** | The 36 cases have zero frames satisfying both `status == OK` and `mechanism2_gate_ok == true`; 33 are non-scripted and 3 scripted. They never enter the 204-case non-scripted RQ019 cohort. |

Official-result provenance for the 29: `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/onsite_competition/RQ011_onsite_full_universe_readiness/RQ011_2_onsite_readiness_20260623T201415+0800_efdd75a5/05_fields/outcome_recoding_by_unit.csv`; filter `unit_composite_key` in the 29-case set; columns `S2_full_scored`, `pdf_summary_match_status`, `success_failure`, `mission_status`, `collision_flag_score0`, `official_comprehensive`.

The same join shows why collision/failed completion cannot be the selection rule:

| Non-scripted cohort | Scored completions | Collision failures | Filter |
|---|---:|---:|---|
| 175 analysed cases | 163/175 | 12/175 | `case_key` occurs in `fig5_counterpart_outcomes.parquet` |
| 33 no-gate cases | 29/33 | 4/33 | non-scripted full-anchor case with no frame passing both gates |
| 29 raw-series-missing cases | 27/29 | 2/29 | post-gate non-scripted case absent from `fig5_counterpart_outcomes.parquet` |
| True non-scripted residual | 56/62 | 6/62 | union of the preceding 33 and 29 cases |

### 3.4 The 47/519 undefined-TTC count is separate

**PROVEN:** 47/519 lower-band two-gate frames have missing contract-window future TTC, alongside 1,042/12,711 inside and 122/869 upper. The TTC code first selects a dense post-anchor window, retains only frames where `closing_rate_mps > 0`, and returns missing when the entire window is non-closing; this is a frame-level kinematic definition after a trajectory exists.

Sources:

- `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/.codex-fleet/rq021-contemporaneous-envelope/work/E1/rq018_rerun/rq018_supervisor_verification.json`; keys `ttc_missing_total`, `ttc_missing_by_band.lower/inside/upper`.
- `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/.codex-fleet/rq021-contemporaneous-envelope/work/E1/rq018_rerun/rq018_supervisor_verification.py`; filter `closing_rate_mps > 0`, columns `case_key`, `frame_index`, `distance_m`, `closing_rate_mps`.
- `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/.codex-fleet/rq021-contemporaneous-envelope/work/E1/rq018_rerun/data_health.json`; `window_coverage.contract.ttc_all_nonclosing_rows=1211/14099`.

Therefore the 47/519 mechanism cannot explain the 29 case-level missing counterpart outcomes, much less the heterogeneous apparent 65.

## 4. Missingness versus monitor verdicts

A valid comparison is possible only inside the 204 post-gate non-scripted cases. The 36 no-gate cases have no monitor verdict by definition, and 3/36 are scripted, so comparing an undifferentiated “65” with the 175 analysed cases would be invalid.

| Cohort | Cases with any lower or upper verdict | Lower-verdict cases | Upper-verdict cases | Flagged frames | Provenance |
|---|---:|---:|---:|---:|---|
| 175 cases represented in the counterpart-outcome parquet | 132/175 = 75.43% | 101/175 = 57.71% | 111/175 = 63.43% | 1,188/11,694 = 10.16% (469 lower + 719 upper) | Two-gate non-scripted rows whose `case_key` occurs in `fig5_counterpart_outcomes.parquet`; bands computed from `ipv_log < lo_90`, `ipv_log > hi_90`, otherwise inside |
| 29 cases absent from the counterpart-outcome parquet | 7/29 = 24.14% | 5/29 = 17.24% | 3/29 = 10.34% | 13/1,330 = 0.98% (10 lower + 3 upper) | Two-gate non-scripted rows whose `case_key` is absent from `fig5_counterpart_outcomes.parquet`; same band definition |

**PROVEN descriptive result:** the raw-series-missing cases are much less likely to contain any flagged moment than the retained cases (24.14% versus 75.43%), and their flagged-frame share is also lower (0.98% versus 10.16%). This is an accounting comparison, not a new inferential claim: no significance test was run, and no causal interpretation is warranted.

The verdict sources and filters are the RQ017 gate shards plus `onsite_scoring_dryrun.parquet`, joined on `product_row_key`, restricted to `status == "OK"`, `mechanism2_gate_ok == true`, and the non-scripted selection string. Case membership comes from `fig5_counterpart_outcomes.parquet`; `case_key` is the case denominator, and the original joined rows are the frame denominator. The output parquet has 11,671 rather than 11,694 rows because 23 individual frames in retained cases do not yield a valid final output: 19 have nearest timestamp gaps at or above 150 ms and 4 fail subsequent frame-level validity; those 23 do not remove another case and are not part of the 29-case absence.

## 5. Counterpart control

### 5.1 What is proven in the automated-systems arm

**PROVEN:** Retained platform logs show a `ScenarioMachine` with a 15-scene list connected to a TESSNG simulation service through `TESSNG001Control` and `TESSNG001Result`; the participating AV and TESSNG have distinct control channels. Source: `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/onsite_competition/all_teams_dataset/teams/beijing/05_T20_mm/support_materials/onsite_2025-12-20_14-22-54.log`, lines 917–944; fields/literals `ScenarioMachine`, `scene`, `simulateType`, `svControl`, `commandChannel`, `dataChannel`.

Six of seven available Beijing files matching the absolute glob `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/onsite_competition/all_teams_dataset/teams/beijing/*/support_materials/onsite*.log` contain all four literals `simulateType=5`, `TESSNG001Control`, `inject_start`, and `ScenarioMachine`; the seventh contains only startup content. This proves a simulator-backed injection architecture, but the repository has no enum codebook for `simulateType=5` or `svControl=0`.

**PROVEN:** The analytical counterpart is selected from replay vehicle records, not from a controller manifest. The builder excludes ineligible actors, requires timing support, prefers any candidate whose `name` contains the literal `从车`, and emits `online_first_conflict_nearest_timing_eligible_prefer_scripted_from_vehicle`; otherwise it emits `online_first_conflict_nearest_timing_eligible_vehicle`. Source: `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/hpc_onsite_ipv/build_onsite_m3_anchors_hpc.py`, lines 382–415 and 463–491; inputs `participantTrajectories`, `role`, `id`, `name`, `vehicleType`, `length`.

Consequently, “scripted” is a name-based analytical stratum, not a proven fixed-controller class. A concrete inconsistency confirms the limitation: T17/A1 counterpart ID `500002` has numeric `name=500002` in `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/onsite_competition/all_teams_dataset/teams/beijing/01_T17_panda/sessions/6931-1766206339/vehicle_perception_simulation_trajectory.log` (line 102) and is labelled non-scripted, but the same ID is named `2344_从车1` with raw `driveType=2`, `controlType=0` in `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/onsite_competition/all_teams_dataset/teams/beijing/01_T17_panda/sessions/6931-1766206339/simulation_trajectory.log` (line 12); no repository artefact decodes those integers.

**INFERRED:** The accepted RQ019 knowledge decision calls the non-scripted background vehicles reactive based on reported cross-team trajectory differences for repeated counterpart identities. Source: `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/knowledge/RQ019_counterpart_burden/decision.md`, boundary B4. The underlying row-level comparison artefact and columns were not found in this audit, so that reactivity statement is not independently traceable here; in any event, it would not identify the controller, policy inputs, or configuration.

### 5.2 Automated arm versus human-driver arm

**UNKNOWN:** The repository cannot establish that counterpart control is identical between arms. The current human-arm package is explicitly synthetic/target-only, with blank `generated_by`, `source_machine`, `n_drivers`, `n_runs`, `per_unit_table_file`, and `analysis_script`; the operating brief says the real data had not yet been transferred and independently verified.

Sources:

- `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/.codex-fleet/rq022-matched-scenario/work/T1_target_figure/human_arm_data.json`; fields above plus synthetic status.
- `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/.codex-fleet/rq022-matched-scenario/work/T1_target_figure/DATA_INTERFACE.md`; defines `driver × scenario` returns but contains no counterpart controller/configuration identifier.
- `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/START_HERE.md`, lines 11–18; human-arm figures/numbers are synthetic placeholders pending real transfer and verification.
- `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/knowledge/PAPER001_online_sociality_verification_manuscript/review_rounds/round5/AGGREGATION.md`, line 49; cross-arm identity is explicitly described as asserted but undocumented.
- `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/knowledge/PAPER001_online_sociality_verification_manuscript/review_rounds/round5/ESCALATIONS.md`, lines 56–64; records the organizer-material request needed to close the controller-equivalence question.

## 6. What remains UNKNOWN, and the artefact that would close it

| Unknown | Exact artefact needed |
|---|---|
| Why the 11 T8 references collapse below two unique points, and whether ego or counterpart coordinates are responsible | A per-failure anchor-builder diagnostic keyed by `unit_composite_key`, with `selected_counterpart_id`, eligible-candidate count, aligned-frame count, ego and counterpart unique-XY counts, valid-anchor candidate count, and raw line/timestamp locators in `vehicle_perception_simulation_trajectory.log` |
| Whether collision directly caused the parser/counterpart failures in the 3 collision cells among the 18 | A competition event/attempt timeline keyed by scenario and timestamp that links the collision/termination event to the first missing replay records; the present outcome and builder ledgers do not encode that causal link |
| Why the selected counterpart IDs for the 29 cases are absent from `simulation_trajectory.log` | A cross-log actor-identity map keyed by `session_id`, `caseId`, perception-log actor ID and simulator-log actor ID, plus recorder diagnostics for dropped/filtered actors; this would distinguish true logging dropout from ID namespace mismatch |
| Actual controller of each counterpart and whether “scripted” truly means fixed/non-reactive | Organizer simulator protocol/configuration keyed by `caseId` or scenario, including controller class/version, policy/config hash, input channels, and codebooks for `simulateType`, `svControl`, `driveType`, and `controlType` |
| Whether counterpart control is identical in automated and human-driver arms | The real human-arm run manifest and replay logs containing the same controller/configuration identifiers, plus a scenario-keyed cross-arm equivalence table with configuration hashes |
| Whether the counterpart was a virtual injected object, simulator body, physical second vehicle, or teleoperated vehicle | A benchmark apparatus/protocol note from the organizer; this physical/control distinction is not documented in the repository |

## Final accounting statement

The record rejects the hypothesis that the 18 were never attempted: all were officially scored, including 15 completions and 3 collision failures. Their proximate exclusion from the 267-case anchor set is recorded as an anchor/replay processing failure; whether collision caused the three corresponding parser failures remains unknown. The manuscript’s “65 runs without recorded outcomes” description is also not supported as written: the repository shows 36 earlier monitor-gate exclusions plus 29 exact counterpart-series absences, and it does not contain the controller/identity artefacts needed to explain why those 29 IDs disappear or to prove cross-arm counterpart-control equivalence.

No research-repository or manuscript-repository file was modified during this audit; this report is the sole investigation output.
