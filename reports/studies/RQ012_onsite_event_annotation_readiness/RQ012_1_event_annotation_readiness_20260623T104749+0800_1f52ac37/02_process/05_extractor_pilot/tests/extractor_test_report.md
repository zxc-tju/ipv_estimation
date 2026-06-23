# RQ012A Independent Extractor Test Report

Run ID: RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37
Worker ID: RQ012-W12-extractor-test
Extractor under test: `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37/02_process/05_extractor_pilot/extractor_pilot.py`
Config under test: `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37/02_process/05_extractor_pilot/extractor_config.json`

## Scope Firewall

This test worker imported the extractor and constructed synthetic kinematic fixtures only. It did not read media, labels, IPV, scores, ranks, team identity, agreement results, or event-outcome associations. The paper repository was not read.

## Coverage Summary

- Automatic events covered: E01, E02, E03, E06, E09, E15, E16, E18, E19.
- E01 covered as not-computable because frozen counterpart relation is unavailable.
- Computable event paths covered with should-trigger and should-not-trigger synthetic series: E02, E03, E06, E09, E15, E16, E18, E19.
- Boundary coverage includes value thresholds, minimum duration, merge gap, zero-length input, and single-sample input.
- Time-alignment coverage includes within-gap alignment, outside-gap alignment, duplicate timestamps, out-of-order timestamps, and multi-actor frame alignment.
- Impossible-value coverage includes NaN, inf, duplicate/non-increasing timestamps, and negative speed.

## Per-Category Results

| category | tests | failed |
|---|---:|---:|
| boundary | 4 | 0 |
| determinism | 1 | 0 |
| impossible_values | 3 | 0 |
| time_alignment | 5 | 0 |
| unit | 17 | 0 |

## Determinism

Determinism result: PASS. The test ran `extract_all_events` twice on the same synthetic fixture and compared canonical sorted JSON payloads.

## Failures

No failures.

## Manual Spot-Check Protocol

Do not execute this protocol on labels or outcome data. It is a human visual QA protocol for trajectory-only event plausibility.

1. Sample detected intervals from `event_intervals_central.csv` using a fixed seed recorded on the sign-off sheet. Recommended N: 5 intervals per emitted event type, or all intervals when an event has fewer than 5. Include E01 as a not-computable audit row rather than a media check.
2. For each sampled interval, load only the corresponding ego/world trajectory rows, actor IDs, timestamps, geometry, speed, acceleration, braking, lateral acceleration, and steering fields within a window from 2 s before onset to 2 s after offset.
3. Render a trajectory plot with ego and counterpart footprints, event onset/offset markers, and threshold overlays relevant to the event. For E02/E03/E18/E19 also render the relevant time-series threshold trace. For E06 render stop/go states. For E16 render displacement and speed over the 10 s causal window.
4. The reviewer checks whether the plotted kinematics satisfy the frozen ontology rule and whether the proxy guard text is correct. The reviewer must not inspect labels, IPV, score, rank, team identity, agreement, or downstream outcomes.
5. Record accept/reject plus reason. A reject is a candidate extractor defect and should include the event ID, session key, unit ID, interval index, plotted threshold trace, and exact field values used for the decision.

### Sign-Off Sheet Template

| sampled_by | review_date | seed | event_id | pilot_session_id | unit_id | interval_index | plot_path | threshold_trace_checked | accept_reject | reason | forbidden_fields_confirmed_unread |
|---|---|---:|---|---|---|---:|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  | yes/no |

## Acceptance Criteria Results

| criterion | result |
|---|---|
| Unit tests for all automatic events or documented non-computable status | pass |
| Boundary tests for thresholds, durations, merge gaps, zero/single inputs | pass |
| Time-alignment tests | pass |
| Determinism/repro test | pass |
| Impossible-value guards exercised | pass |
| Manual spot-check protocol documented | pass |
| No IPV/outcome read; no event-IPV association; no paper repo read | pass |

## Overall Verdict

PASS
