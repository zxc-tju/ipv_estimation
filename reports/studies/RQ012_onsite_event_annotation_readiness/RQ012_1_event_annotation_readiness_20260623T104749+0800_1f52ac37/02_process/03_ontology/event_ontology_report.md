# RQ012A Phase 3 Event Ontology Report

Worker: RQ012-W08-ontology  
Role: designer (event ontology)  
Run ID: RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37  
Status: complete

## Scope

This phase defines event ontology entries only. It uses the Gate 012-0 signal audit, the W1 SPEC, and binding addendum v0.1. It does not set numeric threshold values, implement extractors, create labels, inspect IPV/outcome data, or run event-IPV analysis.

Candidate directions are a priori behavioural hypotheses only. They were not derived from IPV values, official scores, ranks, labels, event frequencies, or outcomes.

## Accounting Summary

| Set | Count | Events |
|---|---:|---|
| Automatic retained | 9 | E01, E02, E03, E06, E09, E15, E16, E18, E19 |
| Human-only retained | 5 | E04, E05, E07, E08, E20 |
| Removed / documented unavailable | 6 | E10, E11, E12, E13, E14, E17 |
| Total accounted | 20 | E01-E20 |

Machine-readable ontology: `01_results/event_ontology.yaml`.

## Endpoint-Eligibility Summary

Only `independent_consequence_endpoint` events are eligible as possible primary RQ012B event-IPV validation endpoints, subject to later gates and explicit authorization. The following construct-proximal descriptors are explicitly ineligible as primary endpoints and may be used only as secondary descriptive context:

| Event | Extraction mode | Endpoint tier | Primary endpoint status |
|---|---|---|---|
| E04 forced yielding | human_only | construct_proximal_descriptor | NOT eligible |
| E05 yield-role reversal | human_only | construct_proximal_descriptor | NOT eligible |
| E07 unnecessary stop | human_only | construct_proximal_descriptor | NOT eligible |
| E20 yielding to a non-counterpart actor | human_only | construct_proximal_descriptor | NOT eligible |

Planner/system events are also not primary human-observable consequence endpoints in this ontology. They are documented unavailable because the required flags or authoritative status-code dictionaries were not available.

## Class Coverage

Retained events cover `physical_safety`, `interaction_quality`, and `human_motif`.

Planner/system candidates are present in the audit but were removed at Gate 012-0. This ontology therefore records `planner_system` as documented unavailable instead of inventing unsupported automatic planner events.

### physical_safety

| Event | Mode | Actor | Rule structure | Candidate direction |
|---|---|---|---|---|
| E01 counterpart hard braking | automatic | counterpart | Sustained hard-brake condition P_hard_brake_E01 after frozen counterpart identity and context rules. | competitive |
| E02 high deceleration | automatic | ego / counterpart / other | Sustained high-deceleration condition P_high_decel_E02 using frozen acceleration-field precedence. | non_specific |
| E03 high jerk | automatic | ego / counterpart / other | Causal longitudinal jerk J_long_E03 satisfies P_high_jerk_E03 after frozen smoothing. | non_specific |
| E09 near miss | automatic | ego / counterpart_or_other | Frozen pair geometry and motion-risk proxy satisfy P_near_miss_E09. | competitive |
| E15 collision/contact | automatic | ego / counterpart_or_other | Oriented actor footprints satisfy P_overlap_E15; output remains a geometric proxy only. | competitive |
| E18 emergency stop | automatic | ego | Kinematic emergency-stop candidate P_emergency_stop_E18 combines deceleration, braking, and stop-state conditions. | non_specific |
| E19 abrupt lateral comfort events | automatic | ego | Causal lateral dynamics satisfy P_lateral_comfort_E19 after frozen smoothing. | non_specific |

### interaction_quality

| Event | Mode | Actor | Rule structure | Candidate direction |
|---|---|---|---|---|
| E06 repeated stop-go | automatic | ego / counterpart / other | Alternating stop and go states satisfy N_cycles_E06 and duration rules. | non_specific |
| E08 conflict escalation | human_only | ego / counterpart / involved_other | Human-only judgment that progression, interaction linkage, and severity increase support escalation. | competitive |
| E16 off-route or no-progress timeout | automatic | ego | No-progress subcase only: ego progress satisfies P_no_progress_E16 for D_no_progress_E16; off-route has no active rule. | over_yielding |

### human_motif

| Event | Mode | Actor | Rule structure | Candidate direction | Primary endpoint status |
|---|---|---|---|---|---|
| E04 forced yielding | human_only | forcing_actor / yielding_actor | Human-only forcedness, priority-context, and actor-role decision. | competitive | NOT eligible |
| E05 yield-role reversal | human_only | expected_priority_actor / expected_yielding_actor | Human-only expected-priority versus observed-yielding-role decision. | over_yielding | NOT eligible |
| E07 unnecessary stop | human_only | stopped_actor | Human-only stop-visible and necessity-context decision. | over_yielding | NOT eligible |
| E20 yielding to a non-counterpart actor | human_only | yielding_actor / non_counterpart_actor | Human-only yielding-visible, counterpart-identity, and non-counterpart-role decision. | over_yielding | NOT eligible |

### planner_system

No retained planner/system event has a credible automatic signal path in Gate 012-0. The removed events below remain documented unavailable.

## Online / Offline Summary

| Status | Count | Events | Notes |
|---|---:|---|---|
| online | 9 | E01, E02, E03, E06, E09, E15, E16, E18, E19 | Automatic rules use current and past states only; derivative events require causal smoothing. |
| offline | 5 | E04, E05, E07, E08, E20 | Human-only judgments require full-window semantic context. |

## Retained Event Details

All retained event details, including required signals, threshold parameter names, minimum-duration parameter names, merge-gap parameter names, onset/end definitions, missing-data rules, known confounds, endpoint eligibility, and extraction mode, are recorded in `01_results/event_ontology.yaml`.

No numeric threshold values are assigned in this phase. Every automatic event uses parameter names to be set outcome-blind in phase4.

## Removed Events

| Event | Class | Endpoint tier | Removal reason |
|---|---|---|---|
| E10 safety-controller intervention | planner_system | planner_system_event | Removed from automatic extraction until an explicit intervention flag or authoritative status-code dictionary is supplied. |
| E11 planner fallback | planner_system | planner_system_event | Removed because no credible fallback signal path exists in the available schema. |
| E12 repeated replanning | planner_system | planner_system_event | Removed because repeated replanning cannot be inferred from empty or absent planner plan channels. |
| E13 trajectory rejection | planner_system | planner_system_event | Removed because trajectory rejection is not represented by the available replay schema. |
| E14 mission failure | planner_system | planner_system_event | Removed because mission failure lacks a documented automatic signal path in the current schema. |
| E17 human/operator takeover | planner_system | planner_system_event | Removed because takeover cannot be identified without an explicit flag or authoritative status-code dictionary. |

## B01 Compliance

The ontology carries `endpoint_eligibility` for every retained event and every removed event. Construct-proximal descriptors E04, E05, E07, and E20 are explicitly marked `primary_endpoint_eligible: false` in the YAML and are reported above as NOT eligible for primary RQ012B endpoint use.

## Handoff Notes

Phase4 should set threshold values only through the outcome-blind threshold rationale and confirmatory subset protocol. Phase5 should translate the human-only motif rules into a codebook without changing endpoint eligibility tiers.
