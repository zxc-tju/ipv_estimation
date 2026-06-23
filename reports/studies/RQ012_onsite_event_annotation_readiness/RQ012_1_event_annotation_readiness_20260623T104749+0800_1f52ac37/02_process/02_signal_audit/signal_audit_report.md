# RQ012A W0 Signal Availability Audit - Gate 012-0

Worker: RQ012-W04-signal-audit  
Run ID: RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37  
Gate: 012-0 signal feasibility  
Status: PASS after removing or demoting automatic candidates without credible signal paths.

## Scope And Guardrails

This audit covers schema and data-health inspection only. It does not define thresholds, labels, ontology rules, event frequencies, event-IPV associations, or outcome-dependent selections.

Inspected scope: Inspected all materialized OnSite replay log roles discovered under ONSITE_DATA: 23 monitor.log, 23 vehicle_trajectory.log, 23 simulation_trajectory.log, 23 vehicle_perception_simulation_trajectory.log, and 19 optional vehicle_perception_trajectory.log files. Missingness figures below come from a bounded schema scan of up to 2,000 JSON-lines per file, plus manifest/header checks; no event frequencies or outcome associations were computed.

Denylist handling: no IPV values, deviation labels, official coordination outcomes, score/rank values, or team identity values were used to classify events or define signal paths. Some source README/manifest text exposes protected selection/team metadata; that material was treated as non-usable leakage context and is not reproduced in this report or the deliverable tables. Area/scenario/run IDs, filenames, paths, ordering, thumbnails, and manifest-derived strata were not used as event-selection or event-definition signals. Raw media content was not opened or interpreted; only media index metadata columns and row counts were checked.

## Schema Inventory Summary

The channel-level inventory is saved at `02_process/02_signal_audit/signal_schema_inventory.csv`. Key findings:

- Required replay logs are materialized for 23 sessions: `monitor.log`, `simulation_trajectory.log`, `vehicle_perception_simulation_trajectory.log`, and `vehicle_trajectory.log` were all discovered 23 times. Optional `vehicle_perception_trajectory.log` was discovered 19 times.
- Ego vehicle state in `vehicle_trajectory.log` provides timestamp, neutral actor ID, latitude/longitude, speed, acceleration, braking, heading/course angle, lateral/longitudinal acceleration, dimensions, steering/wheel angle, and generic status fields. Bounded scan cadence was median 100 ms.
- World actor state in `simulation_trajectory.log` provides timestamp, neutral actor IDs (`id`, `originId`), x/y, latitude/longitude, speed, acceleration, course angle, dimensions, and generic control fields. Bounded scan cadence was median 100 ms; core world actor kinematic fields had 0 missing values in inspected rows.
- Perception logs provide actor roles (`av`, `mvSimulation`, `trafficLight`, `obstacles`), IDs, relative positions, global positions, speed, heading, dimensions, and status fields. One optional perception log role is absent at file level for 4 of 23 sessions; the required perception-simulation log has sparse empty participant value arrays.
- Monitor data provides coarse 1 Hz numeric status channels (`avMonitor.status`, `avMonitor.algorithmStatus`, `avMonitor.driveStatus`, `tessngMonitor.status`) but no local dictionary mapping codes to intervention, takeover, fallback, or failure semantics.
- No explicit lane, route, road, map, off-route, fallback, trajectory-rejection, takeover, emergency-stop, collision/contact, or comfort-event flag was found in the nested-key scan. `futurePlanList` appeared only in a small subset of ego rows and was empty in all inspected occurrences.
- Media metadata exists for 10 video rows with byte counts and paths, but no fps/duration/time-alignment metadata. No media content was inspected.

## Gate 012-0 Event Decisions

| Event | Class | Gate action | Signal-path reason |
|---|---:|---:|---|
| counterpart hard braking | `derivable` | `keep_automatic` | Retain because actor-level speed/acceleration/time/ID channels exist at about 10 Hz; threshold and counterpart rule remain later gated work. |
| high deceleration | `direct` | `keep_automatic` | Retain because deceleration is directly recorded for ego and world actors; W2 must define threshold and spelling precedence. |
| high jerk | `derivable` | `keep_automatic` | Retain because acceleration and timestamp channels exist; derivative method must be frozen before extraction. |
| forced yielding | `partially_observable` | `demote_human_only` | Demote because kinematics alone cannot establish forcedness or right-of-way without route/lane/context signals. |
| yield-role reversal | `partially_observable` | `demote_human_only` | Demote because no credible automatic signal path exists for role reversal semantics. |
| repeated stop-go | `derivable` | `keep_automatic` | Retain because speed/time/ID channels exist at about 10 Hz for ego and world actors. |
| unnecessary stop | `partially_observable` | `demote_human_only` | Demote because automatic stop detection is possible but unnecessary-stop interpretation lacks route/context signals. |
| conflict escalation | `partially_observable` | `demote_human_only` | Demote for now because no complete automatic definition of escalation is supported by direct schema fields. |
| near miss | `derivable` | `keep_automatic` | Retain because positions, dimensions, headings, speeds, timestamps, and actor IDs exist for geometric near-miss derivation. |
| safety-controller intervention | `unavailable` | `remove` | Remove from automatic extraction until an explicit intervention flag or authoritative status-code dictionary is supplied. |
| planner fallback | `unavailable` | `remove` | Remove because no credible fallback signal path exists in available schema. |
| repeated replanning | `unavailable` | `remove` | Remove because repeated replanning cannot be inferred from empty/absent planner plan channels. |
| trajectory rejection | `unavailable` | `remove` | Remove because trajectory rejection is not represented by the available replay schema. |
| mission failure | `unavailable` | `remove` | Remove because mission failure lacks a documented automatic signal path in current schema. |
| collision/contact | `derivable` | `keep_automatic` | Retain only as a geometric overlap/contact-candidate event; do not claim sensor-confirmed collision without a contact flag. |
| off-route or no-progress timeout | `partially_observable` | `keep_automatic` | Retain automatic extraction only for no-progress timeout; remove/offline-hold the off-route subcase until route/lane/goal geometry is supplied. |
| human/operator takeover | `unavailable` | `remove` | Remove because takeover cannot be identified without an explicit flag or authoritative status-code dictionary. |
| emergency stop | `derivable` | `keep_automatic` | Retain as kinematic emergency-stop candidate; do not claim explicit e-stop command without a flag/dictionary. |
| abrupt lateral comfort events | `derivable` | `keep_automatic` | Retain because lateral acceleration/heading/steering/time channels exist for ego trajectory. |
| yielding to a non-counterpart actor | `partially_observable` | `demote_human_only` | Demote because actor kinematics exist but non-counterpart yielding semantics lack route/context and frozen counterpart rules. |

## Counts And Retained Automatic Set

Signal class counts:

- `direct`: 1
- `derivable`: 7
- `partially_observable`: 6
- `human_only`: 0
- `unavailable`: 6

Gate action counts:

- `keep_automatic`: 9
- `demote_human_only`: 5
- `remove`: 6

Automatic candidates retained after Gate 012-0:

- counterpart hard braking
- high deceleration
- high jerk
- repeated stop-go
- near miss
- collision/contact
- off-route or no-progress timeout
- emergency stop
- abrupt lateral comfort events

Demoted to human-only:

- forced yielding
- yield-role reversal
- unnecessary stop
- conflict escalation
- yielding to a non-counterpart actor

Removed from automatic ontology until new signals/dictionaries exist:

- safety-controller intervention
- planner fallback
- repeated replanning
- trajectory rejection
- mission failure
- human/operator takeover

Important qualification: `off-route or no-progress timeout` is retained only for the no-progress timeout subcase. The off-route subcase has no automatic signal path until route/lane/goal geometry is supplied. `collision/contact` is retained only as a geometric overlap/contact-candidate proxy, not as sensor-confirmed physical contact. `emergency stop` is retained only as a kinematic emergency-stop candidate, not as an explicit e-stop command.

## Top Signal-Availability Risks

1. Route/lane/road/map/goal geometry is absent. This blocks automatic forced-yielding, yield-role reversal, unnecessary-stop, off-route, and right-of-way semantics.
2. Planner/system status semantics are opaque. Generic numeric status fields exist, but no authoritative code dictionary maps them to safety-controller intervention, planner fallback, replanning, trajectory rejection, mission failure, or takeover.
3. Optional perception trajectory logs are not complete across sessions. Extraction should prefer required simulation/world actor channels and use optional perception logs only as supplementary evidence.
4. Derivative events need pre-frozen smoothing and missing-data rules. High jerk and lateral comfort events are sensitive to timestamp jitter, acceleration spelling variants, and interpolation choices.
5. Some retained automatic events are proxies. Collision/contact, emergency stop, near miss, and no-progress timeout require careful naming in W1/W2 so they do not overclaim unavailable direct signals.
6. Media metadata is insufficient for automatic extraction. Video rows lack fps/duration/alignment metadata, and content inspection is outside this W0 audit.

## Gate 012-0 Verdict

PASS. Every event retained for automatic extraction has a documented signal path grounded in observed OnSite schema channels. Events without a credible automatic path were either demoted to human-only interpretation or removed from automatic extraction.
