# Automatic Event Extractor Pilot Report

Run ID: RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37
Worker: RQ012-W17a-extractor-robustness

## Firewall Statement

No IPV, deviation, official coordination score, rank, team identity, human label, agreement result, or event-IPV/outcome association was read or computed. RQ003 manifests were used only for item-ID exclusion, and OnSite trajectory logs were used only for kinematics, geometry, timestamps, and actor IDs.

## Sample

Recorded seed: 202606230510
Selected pilot sessions: 5
RQ003-excluded sessions: 0

| pilot_session_id | session_key | rq003_exclusion_matches |
|---|---|---|
| pilot_24d1c625fa54 | 6933-1766207438 |  |
| pilot_71716c4d8821 | 6937-1766209673 |  |
| pilot_7959bf6bbc48 | 6927-1766200590 |  |
| pilot_d4ffe2386932 | 6922-1766197115 |  |
| pilot_e280c37fc4a0 | 6935-1766208365 |  |

## Health Metrics

Computable fraction summary: E01=0.000000, E02=1.000000, E03=1.000000, E06=1.000000, E09=0.538462, E15=0.538462, E16=1.000000, E18=1.000000, E19=1.000000
Duplicate merge summary: before=4641; after=4198; merged_duplicates=443; post-merge overlaps are reported per event in the CSV.

| event_id | computable_fraction | raw_count | primary_count | suppressed_by_precedence | duplicate_rate_before_after | impossible_values | actor_attribution_failures | missing_data_failures | guard |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| E01 | 0.000000 | 0 | 0 | 0 | before=0; after=0; merged_duplicate_rate=0.000000; post_merge_overlaps=0 | 0 | 1703 | 1703 | not emitted; frozen counterpart relation unavailable |
| E02 | 1.000000 | 1770 | 1770 | 0 | before=2060; after=1770; merged_duplicate_rate=0.140777; post_merge_overlaps=0 | 2 | 0 | 5 | automatic kinematic/geometric pilot output |
| E03 | 1.000000 | 1615 | 1615 | 0 | before=1691; after=1615; merged_duplicate_rate=0.044944; post_merge_overlaps=0 | 2 | 0 | 5 | automatic kinematic/geometric pilot output |
| E06 | 1.000000 | 186 | 186 | 0 | before=199; after=186; merged_duplicate_rate=0.065327; post_merge_overlaps=0 | 2 | 0 | 4 | automatic kinematic/geometric pilot output |
| E09 | 0.538462 | 463 | 350 | 113 | before=481; after=463; merged_duplicate_rate=0.037422; post_merge_overlaps=0 | 4 | 0 | 802 | automatic kinematic/geometric pilot output |
| E15 | 0.538462 | 114 | 114 | 0 | before=115; after=114; merged_duplicate_rate=0.008696; post_merge_overlaps=0 | 4 | 0 | 802 | geometric contact candidate only; not sensor-confirmed collision |
| E16 | 1.000000 | 48 | 48 | 0 | before=93; after=48; merged_duplicate_rate=0.483871; post_merge_overlaps=0 | 2 | 0 | 4 | no-progress only; off-route guarded off because route/lane/goal geometry is unavailable |
| E18 | 1.000000 | 0 | 0 | 0 | before=0; after=0; merged_duplicate_rate=0.000000; post_merge_overlaps=0 | 2 | 0 | 4 | kinematic emergency-stop candidate only; no explicit e-stop command flag |
| E19 | 1.000000 | 2 | 2 | 0 | before=2; after=2; merged_duplicate_rate=0.000000; post_merge_overlaps=0 | 2 | 0 | 4 | automatic kinematic/geometric pilot output |

## Cross-Event Duplicate And Precedence Audit

Primary endpoint counts below apply the frozen hierarchy without deleting raw diagnostic intervals. Full overlap details are written to `cross_event_audit.csv`.

| relation | overlap_rows | suppressed_intervals | event_a_raw | event_a_primary | event_b_raw | event_b_primary | precedence_rule |
|---|---:|---:|---:|---:|---:|---:|---|
| E01/E02 | 0 | 0 | 0 | 0 | 1770 | 1770 | E01 subset of E02; E01 remains deferred without frozen counterpart relation |
| E02/E18 | 0 | 0 | 1770 | 1770 | 0 | 0 | E18 takes primary precedence for overlapping ego hard-stop windows |
| E09/E15 | 114 | 113 | 463 | 350 | 114 | 114 | E15 takes primary precedence for same-pair/time contact windows |

## Phase 5 Fix Audit

W12 found that rows already counted as impossible could still enter event flag construction. The extractor now applies a shared emission-quality guard before event flags or intervals are built; invalid rows split segments and remain counted in health metrics.

W17a adds pair-event timestamp prevalidation, actor identity stability guards, and cross-event primary-endpoint precedence for E01/E02, E02/E18, and E09/E15. Raw counts remain available for diagnostics; primary counts are de-overlapped.

W12 E02 repro closure: pre-fix speed=-1.0 and accel=-3.5 for 0.3 s emitted 1 interval with raw_hits=3 and impossible_values=3; post-fix it emits 0 intervals with raw_hits=0 and impossible_values=3.

Same-seed pilot comparison against the pre-fix central rerun. `post_primary_count` is the de-overlapped endpoint count used for primary reporting.

| event_id | event_count_pre | post_raw_count | post_primary_count | primary_delta_vs_pre | raw_hits_pre | raw_hits_post | raw_hits_delta | impossible_pre | impossible_post | missing_pre | missing_post |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E01 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1703 | 1703 |
| E02 | 1770 | 1770 | 1770 | 0 | 108552 | 108552 | 0 | 2 | 2 | 5 | 5 |
| E03 | 1615 | 1615 | 1615 | 0 | 30099 | 30098 | -1 | 2 | 2 | 5 | 5 |
| E06 | 186 | 186 | 186 | 0 | 108439 | 108437 | -2 | 2 | 2 | 4 | 4 |
| E09 | 950 | 463 | 350 | -600 | 28026 | 16523 | -11503 | 0 | 4 | 974 | 802 |
| E15 | 152 | 114 | 114 | -38 | 4562 | 4118 | -444 | 0 | 4 | 974 | 802 |
| E16 | 48 | 48 | 48 | 0 | 2194 | 2194 | 0 | 2 | 2 | 4 | 4 |
| E18 | 0 | 0 | 0 | 0 | 7 | 7 | 0 | 2 | 2 | 4 | 4 |
| E19 | 2 | 2 | 2 | 0 | 67 | 67 | 0 | 2 | 2 | 4 | 4 |

## Threshold Sensitivity

Band labels follow the recorded sensitivity bands exactly. Counts are not necessarily monotone where a band changes categorical logic, such as E18 brake-mode support.

| event_id | low_count | central_count | high_count |
|---|---:|---:|---:|
| E01 | 0 | 0 | 0 |
| E02 | 2568 | 1770 | 1283 |
| E03 | 8818 | 1615 | 287 |
| E06 | 212 | 186 | 32 |
| E09 | 228 | 350 | 423 |
| E15 | 114 | 114 | 110 |
| E16 | 66 | 48 | 17 |
| E18 | 0 | 0 | 202 |
| E19 | 32 | 2 | 2 |

## Sampling-Rate Sensitivity

| event_id | original_count | decimate2_count | delta | relative_delta |
|---|---:|---:|---:|---:|
| E01 | 0 | 0 | 0 |  |
| E02 | 1770 | 1964 | 194 | 0.109605 |
| E03 | 1615 | 6612 | 4997 | 3.094118 |
| E06 | 186 | 194 | 8 | 0.043011 |
| E09 | 350 | 860 | 510 | 1.457143 |
| E15 | 114 | 161 | 47 | 0.412281 |
| E16 | 48 | 39 | -9 | -0.187500 |
| E18 | 0 | 1 | 1 |  |
| E19 | 2 | 3 | 1 | 0.500000 |

## Proxy And Candidate Guards

- E15 is emitted only as a geometric contact candidate, not a sensor-confirmed collision.
- E16 emits only no-progress timeout; off-route extraction is guarded off.
- E18 is emitted only as a kinematic emergency-stop candidate, not an explicit e-stop command.
- E01 is attempted but not computable because the frozen counterpart relation is unavailable in the pilot inputs.

## Interpretation Boundary

These are data-health and extractor-stability metrics only. Counts are not interpreted as outcomes, labels, agreement, or IPV-related evidence, and no threshold was tuned to the pilot results.
