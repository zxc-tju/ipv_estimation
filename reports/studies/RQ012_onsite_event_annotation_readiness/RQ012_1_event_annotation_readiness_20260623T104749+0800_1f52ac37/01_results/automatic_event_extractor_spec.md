# Automatic Event Extractor Spec

Run ID: RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37
Worker: RQ012-W17a-extractor-robustness
Scope: small outcome-blind extractor pilot for the nine retained automatic events.

## Firewall

This extractor reads only OnSite trajectory, geometry, time, and actor-id fields,
plus RQ003 annotation item IDs used solely to exclude already-used annotation
items from the pilot sample. It does not read IPV, deviation, official
coordination scores, ranks, team identities, human labels, agreement files, or
event-outcome associations. No event-IPV/outcome association is computed.

## Shared Implementation Rules

- Parameters are loaded from `extractor_config.json`, copied from the frozen
  confirmatory thresholds and sensitivity bands in `event_threshold_rationale.md`.
- Frame-level detections are converted to runs, filtered by the frozen
  minimum-duration parameter, then merged by the frozen same-actor or same-pair
  merge-gap parameter.
- Sampling-rate sensitivity is measured by rerunning central thresholds after
  deterministic per-actor decimation by a factor of two.
- Threshold sensitivity is measured by rerunning low, central, and high bands.
- Dimensions above 30 source units are treated as centimeters and divided by 100
  before geometric footprint calculations.
- Before any event flags or intervals are emitted, each actor or pair row passes
  a shared emission-quality guard. Rows with missing required fields, NaN/inf
  kinematics, negative speed magnitudes, duplicate/non-increasing timestamps
  after per-series timestamp ordering, or impossible geometry dimensions when
  geometry is required are excluded from emission and split the current segment.
  Excluded rows still increment the missing-data or impossible-value diagnostics.
- Geometry uses current-frame oriented rectangle footprints and nearest ego/world
  alignment within 100 ms. Before pair alignment, ego and world time series are
  prevalidated: missing timestamps are dropped, non-monotonic rows are rejected,
  duplicate timestamps are treated as ambiguous and removed, impossible-value
  diagnostics are recorded, and tied nearest-neighbor matches choose the lower
  timestamp deterministically.
- Pair events require stable world actor identity over the emitted interval.
  Rows are split when the same actor ID carries changing originId/name evidence;
  affected windows increment actor-attribution failures and any resulting pair
  intervals are diagnostic-only rather than primary endpoints.
- Cross-event primary endpoint precedence is applied after raw interval
  extraction: E01 is a counterpart-attributed subset of E02 and remains deferred,
  E18 takes precedence over overlapping ego E02 hard-stop windows, and E15
  contact takes precedence over same-pair/time E09 near-miss candidates under
  the active band. The audit table preserves the overlapping raw intervals.

## Event Rules

| event_id | computed rule | unit of analysis | mode | limitations |
|---|---|---|---|---|
| E01 | Counterpart hard braking would require a frozen counterpart relation plus deceleration persistence. | counterpart actor-window | online | Not computable in this pilot because no frozen counterpart identity relation is available. Non-ego actors are not promoted to counterpart status. |
| E02 | Actor acceleration <= -T_decel for D_min, merged by G_merge. | ego/world actor-window | online | Uses direct acceleration field only; no noncausal interpolation. |
| E03 | Causal rolling-median acceleration, backward-difference jerk, abs(jerk) >= T_jerk for D_min. | ego/world actor-window | online | Derivative is sensitive to gaps and timestamp jitter. |
| E06 | Stop/go states from speed thresholds; alternating qualifying stop/go runs meet N_cycles and D_min. | ego/world actor-window | online | Does not infer traffic-control or route context. |
| E09 | Ego-other oriented-footprint clearance <= T_distance or constant-velocity TTC <= T_time_to_conflict; same-pair/time E15 contact windows suppress E09 only in primary endpoint counts and are retained in the cross-event audit. | ego-other pair-window | online | Near-miss proxy only; no future trajectory or outcome is used. |
| E15 | Ego-other oriented-footprint signed distance <= T_overlap_tolerance for D_min. | ego-other pair-window | online | Geometric contact candidate only; not sensor-confirmed collision. |
| E16 | Ego no-progress if rolling causal window displacement and speed remain below frozen thresholds for D_no_progress. | ego session-window | online | Off-route subcase is guarded off until route/lane/goal geometry is frozen. |
| E18 | Ego hard deceleration plus braking/stop-state rule per band, D_min, and merge-gap. | ego session-window | online | Kinematic emergency-stop candidate only; no explicit e-stop flag/dictionary. |
| E19 | Ego lateral acceleration, lateral jerk, or steering-rate exceeds frozen comfort thresholds after causal smoothing. | ego session-window | online | Does not infer lane-change intent or route curvature. |

## Known Parameters Needing RQ011/Future Freezes

- E01 remains blocked by missing frozen counterpart relation.
- E16 off-route remains blocked by missing route/lane/goal geometry.
- E15 and E09 are geometric proxies; sensor contact flags or authoritative
  collision dictionaries would be a future signal freeze, not a pilot tuning item.
- E18 remains a kinematic candidate until an explicit emergency-stop command flag
  or status dictionary exists.
