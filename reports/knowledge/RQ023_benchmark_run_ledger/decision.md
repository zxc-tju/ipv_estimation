# RQ023 Decision: Benchmark Run-Ledger Audit

Status: **accepted (PI ratified 2026-08-12)**

Decision commit: TBD

Basis: completed read-only forensic audit copied to `reports/run_ledger_audit_20260811.md`. The audit defines **PROVEN** as directly traceable to stored rows, fields, or executable selection logic; **INFERRED** as consistent with recorded data but not explicitly recorded as the cause; and **UNKNOWN** as requiring an artefact that is absent. This registration carries those labels over without upgrading them.

## Claims Pending PI Ratification

| Claim ID | One-line claim | Audit confidence | Registration status |
|---|---|---|---|
| RQ023-KC-1 | The 18 absent system-scenario cells were attempted and officially scored; they were omitted by the anchor/replay analysis pipeline, not by competing teams failing to attempt or obtain an official result. | **PROVEN** for the recorded outcomes and proximate pipeline exclusions; collision causality remains **UNKNOWN**. | **accepted (PI ratified 2026-08-12)** |
| RQ023-KC-2 | There is no coherent set of “65 runs without recorded outcomes”; that number mixes pre-gate and post-gate stages, while the valid frozen-number funnel is 267 -> 231 -> 204 -> 175. | **PROVEN** | **accepted (PI ratified 2026-08-12)** |
| RQ023-KC-3 | The 29 post-gate non-scripted cases excluded from the consequence analysis have selected counterpart IDs but no matching exact-ID series in `simulation_trajectory.log`; all nevertheless have official outcomes. | **PROVEN** at the observable cross-log boundary; the reason for the ID/series absence is **UNKNOWN**. | **accepted (PI ratified 2026-08-12)** |
| RQ023-KC-4 | Within the valid 204-case comparison cohort, the 29 excluded cases carry fewer monitor flags than the 175 analysed cases at both case and frame level; the separately reported 47/519 undefined-margin count is a different mechanism. | **PROVEN** descriptive result; no inferential or causal claim was made. | **accepted (PI ratified 2026-08-12)** |
| RQ023-KC-5 | “Scripted” is assigned from a name-based replay-selection string rather than a controller manifest, and T17/A1 counterpart ID `500002` proves that the two retained log sources can disagree on that label. | **PROVEN** for the implemented rule and the concrete inconsistency; the actual controller class remains **UNKNOWN**. | **accepted (PI ratified 2026-08-12)** |

## RQ023-KC-1 — The 18 absent system-scenario cells

**Claim.** The 18 absent cells were attempted and officially scored; their recorded proximate loss occurs in the anchor/replay analysis pipeline rather than at team participation or official scoring.

**Audit confidence:** **PROVEN** for the attempted/scored status, failure ledger, and concentration. Whether a collision caused any corresponding parser failure is **UNKNOWN**.

**Status:** **accepted (PI ratified 2026-08-12)**.

### Evidence and full provenance

- **Anchor-builder accounting:** 285 units requested, 267 units with anchors, 18 failed units, and 67,861 anchors. Absolute source: `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/hpc_onsite_ipv/process_allvalid_processpool_amd/coverage.json`; no row filter; keys `units_requested`, `units_with_anchors`, `ipv_cases_failed`, `total_av_anchors`, `failures[].unit_composite_key`, `failures[].scenario_id`, `failures[].native_replay_case_id`, `failures[].stage`, and `failures[].error`.
- **Four recorded exclusion mechanisms:** 11/18 `ValueError: observed reference has fewer than two unique points`, 3/18 `ValueError: no valid anchors after RQ009 timing filters`, 3/18 `no_eligible_counterpart`, and 1/18 `no_timing_eligible_counterpart`. Absolute source and filter: the same `coverage.json`, all rows in `failures[]`; fields `failures[].error` and `failures[].stage`.
- **Attempted and officially scored:** 18/18 have `S2_full_scored == true` and `pdf_summary_match_status == PASS`; 15/18 have `success_failure == success` and `mission_status == scored_completion`; 3/18 have `collision_flag_score0 == true`, `mission_status == collision_failure_score0`, and `official_comprehensive == 0`. Absolute source: `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/onsite_competition/RQ011_onsite_full_universe_readiness/RQ011_2_onsite_readiness_20260623T201415+0800_efdd75a5/05_fields/outcome_recoding_by_unit.csv`; filter `unit_composite_key` in `coverage.json.failures[]`; columns `S2_full_scored`, `pdf_summary_match_status`, `success_failure`, `mission_status`, `collision_flag_score0`, `official_comprehensive`, `pdf_collision_deduction`, `pdf_task_completion_score`, and `source_locator`.
- **System concentration:** T8 accounts for 13/18 exclusions, equal to 72.22% of the gap, and has 2/15 cells present with anchors; its exclusions comprise 11 degenerate references and 2 cases with no timing-valid anchors. The other 5/18 exclusions are T12 at 2/15 and T10, T20, and T21 at 1/15 each. Numerators come from the 18 `coverage.json.failures[]` rows grouped by `team_id` parsed from `unit_composite_key`; each included system's denominator of 15 clean cells comes from `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ011_onsite_full_universe_readiness/RQ011_2_onsite_readiness_20260623T201415+0800_efdd75a5/02_process/04_mapping/corrected_clean_mask.csv`, filtered to `corrected_clean == true`; columns `unit_composite_key`, `team_id`, `scenario_id`, `run_id`, `session_id`, and `corrected_clean`.
- **No globally dropped scenario:** every scenario remains represented by 16–19 systems. Numerators are the complement of `coverage.json.failures[]` grouped by `failures[].scenario_id`; the denominator is 19 replay-clean systems per scenario from the same `corrected_clean_mask.csv`, filtered to `corrected_clean == true`, using `scenario_id` and `team_id`.

## RQ023-KC-2 — No coherent “65 runs without recorded outcomes” cohort

**Claim.** The apparent 65 is produced by subtracting a post-gate scripted count from a pre-gate universe. The valid funnel is 267 -> 231 -> 204 -> 175; 36 is the frozen gate difference, while 204 and 29 are arithmetic on frozen numbers rather than new measurements.

**Audit confidence:** **PROVEN**.

**Status:** **accepted (PI ratified 2026-08-12)**.

### Frozen-number cross-check and provenance

| Number | Frozen source and meaning | Underlying provenance recorded by the audit |
|---:|---|---|
| 267 | Frozen in RQ017-KC-C2 in `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/knowledge/RQ017_onsite_mechanism_one/decision.md`: case denominator with anchors. | `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/rq017_onsite_gate/l1_v1/artifact_id=onsite_dense_timeseries/shard_id=*/part-0.parquet`, filtered to `artifact_id == onsite_dense_timeseries`; columns `case_id`, `product_row_key`, and `status`; plus `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ017_onsite_mechanism_one/RQ017_1_onsite_gate_20260804T075311Z_406e7a65/case_level_availability.json`, key `denominator_cases`. |
| 231 | Frozen in RQ017-KC-C2 in the same RQ017 decision: cases with at least one jointly judgeable frame. | `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/rq017_onsite_gate/l1_v1/artifact_id=onsite_dense_timeseries/shard_id=*/part-0.parquet` joined on `product_row_key` to `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/.codex-fleet/rq021-contemporaneous-envelope/work/E1/onsite_scoring_dryrun.parquet`; filter `status == "OK" AND mechanism2_gate_ok == true`; columns `case_id`, `status`, `product_row_key`, `mechanism2_gate_ok`, `lo_90`, and `hi_90`. |
| 36 | Frozen in RQ017-KC-C2 in the same RQ017 decision: cases with no jointly judgeable frame, the direct complement of 231 within 267. | The same absolute RQ017 shard and RQ021 scoring paths, join, filter, and columns as the 231 row; `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ017_onsite_mechanism_one/RQ017_1_onsite_gate_20260804T075311Z_406e7a65/case_level_availability.json` records `denominator_cases` and the case-level availability counts. |
| 27 | Frozen in `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/knowledge/RQ019_counterpart_burden/decision.md`: scripted cases in the post-gate analysis set. | Anchor join above; frozen in `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/.codex-fleet/rq021-contemporaneous-envelope/work/E1/rq019_rerun/data_health.json` and `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/.codex-fleet/rq021-contemporaneous-envelope/work/E1/rq019_rerun/key_numbers.json`; filter `counterpart_selection == "online_first_conflict_nearest_timing_eligible_prefer_scripted_from_vehicle"` after both gates; keys `analysis_set.strata` and registered key records; columns `product_row_key`, `case_key`, and `counterpart_selection`. |
| 175 | Frozen in RQ019-KC-C1 and boundary B4 in the same RQ019 decision: non-scripted cases represented in the counterpart-outcome analysis. | `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ021_contemporaneous_envelope/RQ021_1_contemporaneous_envelope_20260805T160425Z_43b4bff/fig5_counterpart_outcomes.parquet`; unique `case_key`, with columns `case_key`, `band`, `is_scripted`, `speed_range_kmh`, `anchor_speed_drop_kmh`, `max_abs_yaw_rate_dps`, `total_heading_change_deg`, `min_accel`, `brake_share_2`, and `brake_share_3`. |

### Corrected arithmetic

- 267 -> 231 removes the frozen 36 cases with no frame passing both monitor gates.
- 231 -> 204 removes the frozen post-gate scripted count of 27. The value 204 is arithmetic on frozen numbers: 231 - 27; it is not registered as a new measurement.
- 204 -> 175 leaves 29. The value 29 is arithmetic on frozen numbers: 204 - 175; it is not registered as a new measurement, although the audit also identifies and diagnoses the exact 29-case set in RQ023-KC-3.
- The apparent 65 equals `(267 - 27) - 175` and mixes stages. The full anchor universe instead contains 30/267 scripted cases and 237/267 non-scripted cases, based on unique `case_key` and exact `counterpart_selection` string in `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet`; no filter for the 267-case denominator, and filter `counterpart_selection == "online_first_conflict_nearest_timing_eligible_prefer_scripted_from_vehicle"` for the 30 scripted cases; columns `case_key` and `counterpart_selection`. The heterogeneous apparent 65 consists of 36 earlier gate exclusions plus 29 later exact-ID series absences, not one “no recorded outcome” class.

## RQ023-KC-3 — What the 29 cases are

**Claim.** The 29 are post-gate non-scripted cases with selected replay counterparts whose exact `session_id::counterpart_key_agent` key is absent from the raw-series diagnostics built from `simulation_trajectory.log`; they are not cases without official competition outcomes.

**Audit confidence:** **PROVEN** at the observable raw-series boundary. Whether the absence is true logging dropout, an ID namespace mismatch, or recorder filtering is **UNKNOWN**.

**Status:** **accepted (PI ratified 2026-08-12)**.

### Evidence and full provenance

- **Cross-log diagnosis:** 1,330/13,024 non-scripted two-gate frames in 29/204 cases have `series_found == false`; 11,694/13,024 frames in 175/204 cases have `series_found == true`, with no cross-cells. Absolute source for the raw-series diagnostics: `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/.codex-fleet/rq021-contemporaneous-envelope/work/E1/rq019_rerun/data_health.json`; keys `alignment_summary.counterpart_series_not_found_rows`, `alignment_summary.counterpart_series_found_rows`, `alignment_summary.matched_anchor_rows`, `alignment_summary.nearest_time_diff_ge_150ms_rows`, and `alignment_summary.series_diagnostics`. Cohort construction joins `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/rq017_onsite_gate/l1_v1/artifact_id=onsite_dense_timeseries/shard_id=*/part-0.parquet` to `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/.codex-fleet/rq021-contemporaneous-envelope/work/E1/onsite_scoring_dryrun.parquet` on `product_row_key`, filters `status == "OK" AND mechanism2_gate_ok == true`, joins `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet` on parsed `case_key`, `anchor_frame_index`, and `perspective`, removes the exact scripted `counterpart_selection` string, and tests `session_id + "::" + counterpart_key_agent` membership in `alignment_summary.series_diagnostics`.
- **Implementation boundary:** `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/.codex-fleet/rq021-contemporaneous-envelope/work/E1/rq019_rerun/rq019_supervisor_verification.py`, functions `load_session_series` and `compute_outcomes`, loads `simulation_trajectory.log` by exact vehicle ID and skips rows when that exact ID has no series.
- **Official-record split:** 29/29 have `S2_full_scored == true` and `pdf_summary_match_status == PASS`; 27/29 have `mission_status == scored_completion`; 2/29—T21/C1 and T3/B4—have `mission_status == collision_failure_score0`. Absolute source: `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/onsite_competition/RQ011_onsite_full_universe_readiness/RQ011_2_onsite_readiness_20260623T201415+0800_efdd75a5/05_fields/outcome_recoding_by_unit.csv`; filter `unit_composite_key` in the exact 29-case set; columns `S2_full_scored`, `pdf_summary_match_status`, `success_failure`, `mission_status`, `collision_flag_score0`, and `official_comprehensive`. No stored field links either collision to the raw-series absence.

## RQ023-KC-4 — Missingness versus monitor verdicts

**Claim.** In the valid post-gate non-scripted comparison, the excluded 29 cases carry fewer monitor flags than the analysed 175 cases at both case and frame level. The separately reported undefined future-margin count occurs after a trajectory exists and is not the same mechanism as exact-ID series absence.

**Audit confidence:** **PROVEN** descriptive result. No significance test was run, and no causal interpretation is warranted.

**Status:** **accepted (PI ratified 2026-08-12)**.

### Evidence and full provenance

| Cohort | Case-level comparison | Frame-level comparison |
|---|---:|---:|
| Analysed cases | 132/175 = 75.43% contain at least one lower or upper verdict | 1,188/11,694 = 10.16% flagged frames, comprising 469 lower and 719 upper |
| Excluded exact-ID-series cases | 7/29 = 24.14% contain at least one lower or upper verdict | 13/1,330 = 0.98% flagged frames, comprising 10 lower and 3 upper |

The direction is therefore plain: the excluded cases carry fewer flags than the analysed cases, both by cases with any flag and by flagged-frame share.

Verdict provenance: `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/rq017_onsite_gate/l1_v1/artifact_id=onsite_dense_timeseries/shard_id=*/part-0.parquet` joined on `product_row_key` to `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/.codex-fleet/rq021-contemporaneous-envelope/work/E1/onsite_scoring_dryrun.parquet`, then joined to `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet` to add `case_key` and `counterpart_selection`; filters `status == "OK"`, `mechanism2_gate_ok == true`, and non-scripted `counterpart_selection`; band definition `ipv_log < lo_90`, `ipv_log > hi_90`, otherwise inside. Membership source: `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ021_contemporaneous_envelope/RQ021_1_contemporaneous_envelope_20260805T160425Z_43b4bff/fig5_counterpart_outcomes.parquet`; `case_key` supplies the case denominator, and original joined rows supply the frame denominator. Columns/keys: `product_row_key`, `case_id`, `case_key`, `status`, `mechanism2_gate_ok`, `ipv_log`, `lo_90`, `hi_90`, `counterpart_selection`, and `band`.

**Separate undefined-margin mechanism:** 47/519 lower-band two-gate frames have missing contract-window future TTC. Absolute sources: `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/.codex-fleet/rq021-contemporaneous-envelope/work/E1/rq018_rerun/rq018_supervisor_verification.json`, keys `ttc_missing_total` and `ttc_missing_by_band.lower`; and `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/.codex-fleet/rq021-contemporaneous-envelope/work/E1/rq018_rerun/rq018_supervisor_verification.py`, filter `closing_rate_mps > 0`, columns `case_key`, `frame_index`, `distance_m`, and `closing_rate_mps`. This frame-level TTC definition is evaluated only after a dense trajectory exists; the 29-case mechanism occurs earlier because an exact counterpart series is absent.

## RQ023-KC-5 — How “scripted” is determined

**Claim.** The analysis labels a counterpart “scripted” through a replay-record name heuristic and an emitted `counterpart_selection` string, not through a controller manifest; a concrete same-ID disagreement between the two log sources proves that the label is not a reliable controller-class identifier.

**Audit confidence:** **PROVEN** for the implemented selection rule and the concrete inconsistency. The controller class and integer-code meanings remain **UNKNOWN**.

**Status:** **accepted (PI ratified 2026-08-12)**.

### Evidence and full provenance

- **Implemented rule:** the builder excludes ineligible actors, requires timing support, prefers a candidate whose replay-record `name` contains the literal `从车`, and emits `online_first_conflict_nearest_timing_eligible_prefer_scripted_from_vehicle`; otherwise it emits `online_first_conflict_nearest_timing_eligible_vehicle`. Absolute source: `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/hpc_onsite_ipv/build_onsite_m3_anchors_hpc.py`, lines 382–415 and 463–491; functions `eligible_counterpart` and `select_counterpart`; input fields `participantTrajectories`, `role`, `id`, `name`, `vehicleType`, and `length`.
- **Concrete counterexample:** T17/A1 counterpart ID `500002` has numeric `name=500002` in `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/onsite_competition/all_teams_dataset/teams/beijing/01_T17_panda/sessions/6931-1766206339/vehicle_perception_simulation_trajectory.log`, line 102, and is labelled non-scripted; the same ID is named `2344_从车1` with raw `driveType=2` and `controlType=0` in `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/onsite_competition/all_teams_dataset/teams/beijing/01_T17_panda/sessions/6931-1766206339/simulation_trajectory.log`, line 12. Fields/literals: `id`, `name`, `driveType`, and `controlType`. The audited repository contains no enum codebook for those integer values.

## Required Qualifications

| Claim ID | Required Qualification | Status |
|---|---|---|
| RQ023-KC-1 | Any manuscript use must state that all 18 excluded cells were attempted and officially scored—15 scored completions and 3 collision failures—and that the recorded exclusion is an analysis-pipeline exclusion, not evidence that teams failed to finish. The 18/18, 15/18, and 3/18 counts use the `outcome_recoding_by_unit.csv` filter and columns recorded under RQ023-KC-1. | **accepted (PI ratified 2026-08-12)** |
| RQ023-KC-3 | Any manuscript use must state that all 29 excluded cases were attempted and officially scored—27/29 scored completions and 2/29 collision failures—and that “missing counterpart outcome” refers to an exact-ID raw-series absence, not a missing competition result. The 29/29, 27/29, and 2/29 counts use the `outcome_recoding_by_unit.csv` filter and columns recorded under RQ023-KC-3. | **accepted (PI ratified 2026-08-12)** |

## Open / UNKNOWN

Every item below retains the audit label **UNKNOWN** and remains outside any paper-safe claim.

| Open question | Current provenance boundary | Exact artefact that would close it |
|---|---|---|
| Why the 11 T8 references collapse below two unique points, and whether ego or counterpart coordinates are responsible | The 11/18 count is from `coverage.json`, filtered to `failures[].error == "ValueError: observed reference has fewer than two unique points"`, grouped by `unit_composite_key`; fields `failures[].error`, `failures[].unit_composite_key`, and `failures[].stage`. | A per-failure anchor-builder diagnostic keyed by `unit_composite_key`, with `selected_counterpart_id`, eligible-candidate count, aligned-frame count, ego and counterpart unique-XY counts, valid-anchor candidate count, and raw line/timestamp locators in `vehicle_perception_simulation_trajectory.log`. |
| Whether collision directly caused the parser/counterpart failures in the 3 collision cells among the 18 | The 3/18 collision count comes from the RQ023-KC-1 join of `coverage.json.failures[]` to `outcome_recoding_by_unit.csv`, using `collision_flag_score0`, `mission_status`, and `official_comprehensive`; neither ledger encodes causal timing. | A competition event/attempt timeline keyed by scenario and timestamp that links the collision/termination event to the first missing replay records. |
| Why the selected counterpart IDs for the 29 cases are absent from `simulation_trajectory.log` | The 29/204 exact-ID absence is from the RQ023-KC-3 post-gate non-scripted join and `data_health.json -> alignment_summary.series_diagnostics`, using `session_id` and `counterpart_key_agent`. | A cross-log actor-identity map keyed by `session_id`, `caseId`, perception-log actor ID, and simulator-log actor ID, plus recorder diagnostics for dropped or filtered actors. |
| The actual controller of each counterpart and whether “scripted” means fixed/non-reactive | The name heuristic is proven by `build_onsite_m3_anchors_hpc.py`; the T17/A1 inconsistency is proven by the two logs in RQ023-KC-5; no enum codebook was found for `simulateType`, `svControl`, `driveType`, or `controlType`. | Organizer simulator protocol/configuration keyed by `caseId` or scenario, including controller class/version, policy/config hash, input channels, and codebooks for `simulateType`, `svControl`, `driveType`, and `controlType`. |
| Whether counterpart control is identical between the automated-systems and human-driver arms | `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/.codex-fleet/rq022-matched-scenario/work/T1_target_figure/human_arm_data.json` has blank `generated_by`, `source_machine`, `n_drivers`, `n_runs`, `per_unit_table_file`, and `analysis_script`; `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/.codex-fleet/rq022-matched-scenario/work/T1_target_figure/DATA_INTERFACE.md` has no controller/configuration identifier; `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/START_HERE.md`, lines 11–18, states that human-arm figures/numbers are synthetic placeholders pending real transfer and verification. | The real human-arm run manifest and replay logs containing the same controller/configuration identifiers, plus a scenario-keyed cross-arm equivalence table with configuration hashes. |
| Whether the counterpart was a virtual injected object, simulator body, physical second vehicle, or teleoperated vehicle | The retained platform logs establish a simulator-backed injection architecture but do not document the physical/control distinction; see the automated-arm provenance in section 5.1 of `reports/run_ledger_audit_20260811.md`. | A benchmark apparatus/protocol note from the organizer. |

## Paper Handoff

**RATIFIED by the PI on 2026-08-12** ("RQ023 五条全批"). RQ023-KC-1 through RQ023-KC-5 and both Required Qualifications are accepted and paper-safe. Manuscript use began the same day (see "Manuscript uptake" below).

---

## PI Statement 2026-08-11 — counterpart control (resolves two UNKNOWN items)

The PI states directly: **all background vehicles in the OnSite benchmark are controlled by TESS NG
traffic-simulation software and respond to the ego vehicle's actions, and the background vehicles in
the human-driving arm are controlled the same way.**

This is a PI-supplied apparatus fact, recorded verbatim in substance and not derived from repository
artefacts. It is registered here because it changes the status of two items in `Open / UNKNOWN` and
because it contradicts a rationale that had been carried in the frozen research record.

| Previously UNKNOWN | New status |
|---|---|
| The actual controller of each counterpart, and whether "scripted" means fixed/non-reactive | **Controller answered by PI statement** (TESS NG, reactive). The enum codebooks for `simulateType`, `svControl`, `driveType` and `controlType` are still absent, so the meaning of the log designation behind the 27-case stratum remains open — see below. |
| Whether counterpart control is identical between the automated-systems and human-driver arms | **Answered by PI statement**: the same control applies in both arms. The configuration-hash-level cross-arm equivalence table remains unavailable, so this rests on the PI statement rather than on a repository artefact. |

### Consequence for RQ023-KC-5 and for the frozen RQ019 hold-out

RQ023-KC-5 (now ratified) established that "scripted" is assigned from a
name-based replay-selection string, not from a controller manifest, and that the two source logs can
disagree on that label. The PI statement is consistent with that finding and sharpens it: because
every counterpart is TESS NG driven and reactive, **the scripted/non-scripted split cannot be a
reactive/non-reactive split**.

This voids the stated reason recorded at `RQ019_counterpart_burden/decision.md` boundary line 60,
which held the 27 cases out on the ground that a scripted counterpart "may not react". That reason
no longer holds. The hold-out itself remains frozen and the 175-case analysis set is unchanged; only
its justification is void.

**Open question for the PI, blocking nothing but affecting wording:** the selector prefers candidates
whose logged `name` contains the literal `从车`. What that designation denotes in the scenario
definitions is not recorded anywhere in the repository. If it denotes the scenario's designated
secondary vehicle rather than a control mode, the 27 held-out cases may be the ones in which the
scenario's intended interaction partner was correctly identified, which would invert the reason for
holding them out. The manuscript currently describes the hold-out neutrally, asserting no mechanism.

### Manuscript coupling

Manuscript commit `195749c` (2026-08-11) states the TESS NG fact in the real-vehicle protocol so
both arms inherit it, and rewrites the 27-case description to assert no mechanism. The clause
removed earlier that day from the human reference arm — that counterpart control matched the
automated-systems arm — is now covered by the PI statement in the protocol paragraph, which is where
it belongs because it governs both arms.

### PI clarification 2026-08-11 (second) — 从车 means "background vehicle"

The PI states that 从车 is simply the term for a background vehicle. Combined with the earlier PI
statement that every background vehicle is TESS NG driven and reactive, this closes RQ023-KC-5's
remaining ambiguity and settles the meaning of the 27-case stratum:

- The selector prefers candidates whose logged `name` contains 从车 — that is, candidates explicitly
  named as background vehicles. It is not selecting a control mode, a scenario role, or a designated
  interaction partner.
- The scripted / non-scripted split therefore records **which naming convention a given run's log
  used**, and nothing about the vehicle. Both sides of the split are the same kind of
  simulator-driven, ego-reactive background vehicle.
- This fully explains the inconsistency registered under RQ023-KC-5: counterpart ID `500002` carries
  the bare numeric `name=500002` in `vehicle_perception_simulation_trajectory.log` and the name
  `2344_从车1` in `simulation_trajectory.log`. The same vehicle falls on either side of the split
  depending on which log the selector read. That is now a naming-convention difference between two
  logs, not a data defect and not a controller difference.
- The label "scripted" is a misnomer inherited from the selection-string name
  `…prefer_scripted_from_vehicle`. It should not be read as "fixed" or "non-reactive" anywhere.

**Effect on the frozen RQ019 hold-out.** The hold-out now has no substantive basis: it separates
runs by a log naming convention. It remains in force because the analysis was frozen before this was
known, the 27 cases are reported in isolation rather than discarded, and dissolving it would require
recomputing frozen numbers. Manuscript commit `40e9a78` states the split as one of record-keeping
rather than apparatus, so no reader can take it for a difference in what the ego faced.

**Option for the PI, not executed and not recommended without a cost check.** The hold-out could be
dissolved by a newly registered analysis pooling the 27 with the 204, which would raise the
counterpart-response case count above the present 175. This would change frozen numbers throughout
the consequence battery — case denominators, bootstrap resampling units, and the flagged-case counts
— and it first requires checking whether the 27 carry the counterpart records the outcome battery
needs (the `is_scripted` column exists in `fig5_counterpart_outcomes.parquet`, but that file holds
only the 175). It is recorded here as an available move, not as a pending action.

---

## Manuscript uptake after ratification (2026-08-12)

The PI ratified all five claims and both Required Qualifications on 2026-08-12. What entered the
manuscript the same day, and what deliberately did not:

| Claim | In the manuscript | Where / how |
|---|---|---|
| RQ023-KC-1 | **Yes**, with its Required Qualification discharged in the same sentence | Methods run accounting: 20 systems fielded, one set aside for unclean replay records, 19 x 15 = 285 cells, 267 with anchors; the 18 stated as attempted and officially scored (15 scored completions, 3 collision failures), with all four exclusion mechanisms, the single-system concentration (13/18) and "no scenario dropped, each retains 16-19 systems". |
| RQ023-KC-2 | **Already in** since 2026-08-11 | The funnel 267 -> 231 -> 204 -> 175 is printed; the invalid 240/65 subtraction is now impossible to form from the text. |
| RQ023-KC-3 | **Partly.** The 29 appear as the funnel's last step. The Required Qualification (all 29 officially scored, 27 completions / 2 collision failures) is **not** printed, because the manuscript never characterises the 29 as lacking a competition result — it says only that the retained 175 have the counterpart logs the outcome battery requires. Nothing in print needs the qualification to be read correctly. It stays available for the response letter, where the qualification must accompany any use. |
| RQ023-KC-4 | **Yes** | Methods, immediately after the funnel: "7 of those 29 contain any flagged moment, against 132 of the 175 retained", framed as evidence the analysed set is not a flag-rich remainder. |
| RQ023-KC-5 | **Yes**, in the form the later PI clarifications settled | The 27-case split is described as a naming convention in the run logs, with an explicit statement that it is record-keeping rather than apparatus. The word "scripted" no longer appears in the manuscript. |

Frame-level KC-4 figures (13/1,330 versus 1,188/11,694) are held for the response letter; the
case-level comparison carries the point in print without a second denominator.
