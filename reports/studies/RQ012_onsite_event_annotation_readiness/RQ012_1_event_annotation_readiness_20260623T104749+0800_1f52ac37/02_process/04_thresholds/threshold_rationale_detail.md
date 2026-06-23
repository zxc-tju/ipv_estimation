# RQ012A Threshold Rationale Detail

Run ID: RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37  
Worker: RQ012-W09-thresholds  
Phase: phase4  
Gate: 012-1  

## Scope

This file documents the outcome-blind threshold rationale for automatic events retained by the phase 3 ontology:

- E01 counterpart hard braking
- E02 high deceleration
- E03 high jerk
- E06 repeated stop-go
- E09 near miss
- E15 collision/contact geometric proxy
- E16 off-route or no-progress timeout, no-progress subcase only
- E18 emergency stop kinematic proxy
- E19 abrupt lateral comfort events

Human-only and removed events are not assigned automatic threshold values in this phase.

## Inputs Read

- `01_results/event_ontology.yaml`
- `01_results/tables/event_signal_availability.csv`
- `02_process/02_signal_audit/signal_schema_inventory.csv`
- `02_process/01_plan_review/plan_resolution_v0_1.md`
- `START_HERE.md`

The paper repository was not read or written.

## Outcome-Blind Source Hierarchy

The rationale uses only permitted W2 source classes:

1. Engineering/safety standards and primary literature:
   - AASHTO Green Book / NCHRP stopping sight distance convention: design deceleration 3.4 m/s^2, equivalent to 11.2 ft/s^2.
   - Traffic-conflict and surrogate-safety literature using time-to-collision as a conflict-severity measure. The central TTC screen is 1.5 s, with 1.0 to 2.0 s sensitivity.
   - ISO 2631-1 ride-comfort framing and automated-vehicle comfort work using thresholded acceleration and jerk comfort screens.
2. Existing platform thresholds:
   - SUMO vehicle-type defaults provide platform references for normal deceleration ability, 4.5 m/s^2, and physically possible emergency deceleration, 9.0 m/s^2. The RQ012 emergency-stop proxy uses 4.5 m/s^2 centrally and keeps 9.0 m/s^2 only as an extreme reference.
3. Measurement resolution and schema handling:
   - The RQ012 signal audit reports median 100 ms replay cadence and timestamp units in milliseconds.
   - Durations, merge gaps, smoothing windows, and missing-data limits are therefore expressed as multiples of 100 ms.
4. Outcome-blind engineering geometry:
   - Near-miss and contact-candidate rules use current x/y, length, width, heading/courseAngle, timestamp, and neutral actor IDs only.
   - Geometry never uses area, scenario, run ID, filename, path, ordering, thumbnail, manifest-derived stratum, team identity, rank, score, IPV, deviation, labels, or agreement.

## Unit And Sampling Rules

- Time: source timestamps are milliseconds and must be divided by 1000 for seconds.
- Acceleration: source acceleration and lateral acceleration are treated as m/s^2 only after implementation sanity checks. If source documentation later contradicts this, extractor implementation must convert or mark the rule not computable.
- Deceleration: thresholds are reported as positive magnitudes. Implementation applies them to negative longitudinal acceleration or an equivalent longitudinal deceleration magnitude.
- Jerk: jerk is acceleration delta divided by elapsed seconds, producing m/s^3.
- Heading: courseAngle is in degrees; geometry converts degrees to radians for trigonometric operations.
- Speed: m/s is assumed after schema sanity check; stop-speed thresholds are also shown in km/h for interpretability.
- Smoothing: online automatic rules use causal past/current smoothing only. Centered or future-looking smoothers are forbidden for the primary online rule.
- Missingness: a consecutive missing gap beyond 0.3 s makes the actor-window or pair-window not computable unless a parameter-specific rule states a longer event merge gap. Missing-data gaps are not bridged using protected metadata.

## B02 Dual-Track Decision

No data-derived confirmatory threshold value was used. Therefore no frozen development-subset manifest was required for value selection.

If a later worker needs a data-derived threshold, the confirmatory protocol must be frozen before any value inspection:

- Sample frame: RQ011 frozen universe or the then-frozen RQ012 neutral item universe only.
- Selection method: random or explicitly approved outcome-blind stratification that does not use area ID, scenario ID, run ID, filenames, paths, item ordering, thumbnails, manifest-derived strata, team identity, rank, score, IPV, deviation, labels, agreement, or event outcomes.
- Seed/hash: recorded before value inspection, with manifest hash stored beside the manifest.
- No-overlap rule: no overlap with formal annotation or validation items unless an explicit written exception is approved and recorded before threshold derivation.
- Authorized builder: a threshold-development worker with no access to IPV, scores, ranks, team identity, labels, agreement, or event-IPV association.
- Permitted inspection after freeze: outcome-blind kinematic data-health summaries and distributions only.
- Promotion firewall: exploratory-derived values can become confirmatory only if re-derived or confirmed on the frozen confirmatory subset with an explicit promotion record.

Exploratory track was not used. No exploratory threshold candidates are adopted here and no quarantine file was created.

## Parameter Rationale Summary

Deceleration thresholds:

- E01 and E02 use 3.4 m/s^2 centrally because it is a published design deceleration convention, not an outcome-fitted value. SUMO's 4.5 m/s^2 deceleration ability provides the high sensitivity level.
- E18 uses 4.5 m/s^2 centrally because emergency-stop candidates should be stricter than generic high-deceleration episodes. SUMO's 9.0 m/s^2 emergencyDecel is retained only as an extreme reference and not as the central rule.

Derivative thresholds:

- E03 longitudinal jerk and E19 lateral jerk use 5 m/s^3 centrally with 3 and 7 m/s^3 sensitivity. Jerk is computed only after causal smoothing to reduce 100 ms timestamp and acceleration jitter.
- Smoothing windows are 0.3 s centrally, equivalent to three 100 ms frames, with 0.2 and 0.5 s sensitivity.

Stop and no-progress thresholds:

- Stop is <=0.3 m/s, about 1.08 km/h, chosen as a low-speed engineering stop screen above likely simulation quantization but below meaningful movement.
- Go is >=1.0 m/s, about 3.6 km/h, to separate creep from resumed movement.
- E06 requires at least two complete stop-go cycles and a total 4 s alternating sequence.
- E16 no-progress uses <=1 m displacement and <=0.3 m/s speed over a 10 s causal window. Off-route remains unavailable because route, lane, and goal geometry are absent.

Geometry thresholds:

- E09 near miss uses <=0.5 m oriented-footprint clearance or <=1.5 s predicted time to conflict. Contact/overlap takes precedence over near-miss for the same pair/time.
- E15 collision/contact is explicitly a geometric contact candidate, not a sensor-confirmed collision. The central overlap tolerance is 0.0 m, with -0.1 m and +0.1 m sensitivity for strict and broad footprint conventions.

Lateral-comfort thresholds:

- E19 uses 2.5 m/s^2 absolute lateral acceleration centrally, approximately 0.25 g, with 2.0 and 3.0 m/s^2 sensitivity.
- Steering-rate thresholding keeps steeringWheelAngle and wheelAngle separate because their physical meanings differ unless a steering ratio is documented.

Categorical and dependency parameters:

- Counterpart identity must come from a frozen neutral relation or neutral id/originId rule. If that relation is not available, E01 is not computable as counterpart-specific and must not be guessed from protected metadata.
- E16 off-route is fixed as unavailable until route, lane, and goal geometry are supplied and frozen.
- E15 and E18 carry mandatory proxy labels so later reports cannot silently upgrade a geometric or kinematic proxy into a confirmed system event.

## Firewall Statement

The threshold table was selected without IPV, deviations, official coordination scores, ranks, team identity, labels, agreement, event frequencies, or event-IPV associations. It does not use area/scenario/run IDs, filenames, paths, ordering, thumbnails, or manifest-derived strata to choose values. It does not use the paper repository. It does not make causal claims.

Audit note: a broad repository text lookup was abandoned after noisy unrelated prior-report hits. Those hits were not RQ012 OnSite official outcomes, ranks, or team identities and did not inform any threshold. The retained threshold sources are the explicit outcome-blind sources listed above.

## Gate 012-1 Verdict

PASS.

Rationale:

- Every threshold parameter of every retained automatic event has a source, recommended value or categorical rule, units or unit status, unit-conversion note, sampling-rate handling, sensitivity band, confirmatory track label, and outcome-blind flag in `01_results/event_threshold_rationale.md`.
- No confirmatory threshold is data-derived.
- No exploratory threshold is promoted or adopted.
- Parameters that cannot be responsibly inferred from current schema are marked unavailable or not computable rather than guessed.
- No retained threshold depends on an outcome, label, agreement result, IPV value, score, rank, team identity, or later association.
