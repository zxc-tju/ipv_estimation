# RQ012A Annotation Codebook v2

Run ID: `RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37`

Status: ready for human annotation preparation. This document defines behavior labels and annotation rules only. It does not contain real annotations, agreement results, event-IPV analysis, or model-generated labels.

Source alignment: this v2 refines the RQ003 blind behavioral codebook, follows the RQ012A event ontology, and applies the binding RQ012A addendum. Construct-proximal labels are retained only as secondary descriptive context and are not eligible as primary event-IPV endpoints in any later RQ012B work.

## Release And Blinding Rules

Annotators must use only neutral item IDs and the provided visual/trajectory material. The annotation decision must be based only on behavior visible in the assigned item. Do not use outside information, hidden provenance, public results, or any other contextual lookup.

The annotator-facing package must contain neutral item IDs, the blank annotation template, this label guidance, and training-only fictional practice items if training is being conducted. Training material must remain separate from formal validation material.

Administrative correction to prior source language, not for annotator packet release: the superseded RQ003 codebook sentence allowed scenario labels and viewing order. That instruction is removed. This v2 authorizes neutral item IDs only and must not expose scenario labels, scenario family, viewing order, strata, paths, team or area identifiers, official scores, ranks, IPV outputs, filenames, thumbnails, manifest-derived strata, or prior/borrowed annotation files to annotators.

## Coding Format

For each neutral item, mark each behavior field as `1` only when the behavior is present and materially relevant in the provided item. Mark it as `0` when the behavior is absent, not materially relevant, or not supported by enough visible evidence.

Multiple behavior labels may be marked `1` when they describe genuinely different aspects of the same item. For example, an item may show oscillation followed by deadlock. Do not force a single-label choice unless the formal template later requires it.

Use `event_start_sec` and `event_end_sec` only when the labeled behavior is temporally localizable from the provided material. Leave both blank when the behavior is global, not localizable, or evidence is insufficient.

Use `confidence_1_to_5` for confidence in the behavioral judgment. It is not a rating of item quality, system quality, or sociality.

If the provided formal template has no `insufficient_evidence` column, record insufficient evidence by setting all substantive behavior fields to `0`, leaving event times blank, setting confidence to `1`, and writing `INSUFFICIENT_EVIDENCE:` followed by a short visible-evidence reason in `free_text_notes`.

## Endpoint Eligibility

| Label | endpoint_eligibility | Construct-proximal? | Primary event-IPV endpoint eligibility |
|---|---|---:|---|
| `aggressive_intrusion` | `construct_proximal_descriptor` | yes | Not eligible. Secondary descriptive context only. |
| `appropriate_assertiveness` | `construct_proximal_descriptor` | yes | Not eligible. Secondary descriptive context only. |
| `over_yielding_freeze` | `construct_proximal_descriptor` | yes | Not eligible. Secondary descriptive context only. |
| `oscillation` | `independent_consequence_endpoint` | no | May be eligible later only if RQ012B is separately authorized and all gates remain satisfied. |
| `deadlock` | `independent_consequence_endpoint` | no | May be eligible later only if RQ012B is separately authorized and all gates remain satisfied. |
| `smooth_reciprocal_negotiation` | `independent_consequence_endpoint` | no | May be eligible later only if RQ012B is separately authorized and all gates remain satisfied. |
| `unrelated_failure` | `annotation_quality_label` | no | Not eligible. Use for exclusion or interpretability notes only. |
| `insufficient_evidence` | `annotation_quality_label` | no | Not eligible. Use for exclusion or interpretability notes only. |

## Common Confidence Scale

Use this scale for every label:

| Value | Meaning |
|---:|---|
| 1 | Insufficient visible evidence, unusable item, or only a guess. |
| 2 | Weak evidence; some cues support the label but key context is missing or ambiguous. |
| 3 | Adequate evidence; the label is more likely than not, with some ambiguity. |
| 4 | Strong evidence; visible cues clearly support the label with minor uncertainty. |
| 5 | Unambiguous evidence; visible behavior strongly and directly matches the label. |

When a label is marked `0`, confidence still reflects confidence in that judgment if the template requires a single confidence value. If the item is insufficient, use confidence `1`.

## Label Definitions

All worked examples and counterexamples below are fictional training-style examples. They are not derived from real outcomes, official results, IPV values, prior annotations, or formal validation items.

### `aggressive_intrusion`

Endpoint eligibility: `construct_proximal_descriptor`. This label is construct-proximal and is forbidden as a primary RQ012B event-IPV validation endpoint. Use only as secondary descriptive context.

Definition: Ego enters, occupies, or continues into another road user's conflict space in a way that forces abrupt avoidance, late braking, sudden steering, or unsafe compression of available space.

Inclusion criteria:

- Ego visibly commits into a conflict space while another actor has an apparent ongoing path or priority.
- The other actor must abruptly slow, stop, steer, or accept a visibly compressed gap.
- The conflict is materially tied to ego's motion, not merely coincident timing.
- The behavior is stronger than ordinary assertive progress through an available gap.

Exclusion criteria:

- Ego proceeds through a clearly available gap with legible timing and no abrupt response by another actor.
- Another actor yields early and smoothly before ego commits.
- The apparent conflict is caused by an unrelated third party, display artifact, or missing context.
- There is not enough visible evidence to judge priority, spacing, or actor response.

Onset rule: onset is the first visible moment ego commits into the conflict space after the other actor's relevant path is visible. If the intrusion is only recognized after the other actor reacts, use the earliest visible ego commitment that directly precedes that reaction.

Worked example: In a fictional practice item, ego moves into a crossing gap while another actor is already approaching; the other actor brakes sharply and stops short to avoid overlap. Mark `aggressive_intrusion=1`.

Counterexample: In a fictional practice item, ego enters after the other actor has already slowed smoothly with ample distance, and both actors continue without abrupt response. Do not mark `aggressive_intrusion`.

Confidence guidance: use `5` when ego commitment and the forced response are clear; `3` when the intrusion is plausible but priority or spacing is partly ambiguous; `1` or the insufficient-evidence protocol when the apparent conflict cannot be reliably attributed to ego.

### `appropriate_assertiveness`

Endpoint eligibility: `construct_proximal_descriptor`. This label is construct-proximal and is forbidden as a primary RQ012B event-IPV validation endpoint. Use only as secondary descriptive context.

Definition: Ego claims space or proceeds decisively when doing so is contextually reasonable, legible to others, and does not create avoidable conflict or unsafe compression.

Inclusion criteria:

- Ego proceeds with a visible safe opportunity instead of hesitating unnecessarily.
- Motion is smooth, predictable, and compatible with other actors' behavior.
- Any yielding by others is early, stable, and not forced by a late ego intrusion.
- Ego's progress helps resolve the interaction rather than escalating it.

Exclusion criteria:

- Ego's action forces abrupt avoidance, late braking, or unsafe compression.
- Ego hesitates, freezes, or alternates intent before committing.
- Priority or gap availability is not visible enough to judge the action as appropriate.
- The item contains no meaningful interaction requiring assertiveness.

Onset rule: onset is the first visible moment ego commits to the reasonable proceed action, such as beginning continuous forward motion into the available gap.

Worked example: In a fictional practice item, ego waits briefly, a clear safe gap opens, and ego proceeds smoothly while the other actor maintains a stable path. Mark `appropriate_assertiveness=1`.

Counterexample: In a fictional practice item, ego accelerates late into a narrowing gap and the other actor brakes abruptly. Do not mark `appropriate_assertiveness`; consider `aggressive_intrusion`.

Confidence guidance: use `5` when the safe opportunity, ego commitment, and stable response by others are clear; use `3` when the action appears reasonable but context is incomplete; use `1` if the safe opportunity cannot be judged.

### `over_yielding_freeze`

Endpoint eligibility: `construct_proximal_descriptor`. This label is construct-proximal and is forbidden as a primary RQ012B event-IPV validation endpoint. Use only as secondary descriptive context.

Definition: Ego stops, creeps, or yields longer than visible conditions appear to require, despite a feasible safe opportunity to proceed, causing unnecessary delay or unresolved negotiation.

Inclusion criteria:

- Ego remains stopped or creeping after a visible safe opportunity has opened.
- Ego yields to an actor that is not visibly constraining ego's safe progress.
- Ego repeatedly declines reasonable gaps without a visible safety reason.
- The behavior materially affects interaction resolution.

Exclusion criteria:

- Ego stops for a visible obstacle, traffic-control state, queue, blocked path, or actor with clear priority.
- The possible proceed opportunity is uncertain due to occlusion, missing context, or unclear route intent.
- Ego briefly pauses but then proceeds smoothly without material delay.
- The issue is a display, logging, or material failure rather than behavior.

Onset rule: onset is the first visible moment after a feasible safe opportunity is available but ego remains stopped, creeping, or yielding without a visible need.

Worked example: In a fictional practice item, all nearby actors have cleared, the path is visibly open, and ego continues creeping without entering for several seconds. Mark `over_yielding_freeze=1`.

Counterexample: In a fictional practice item, ego waits while another actor with clear priority passes, then proceeds smoothly. Do not mark `over_yielding_freeze`.

Confidence guidance: use `5` when the open opportunity and unnecessary delay are clear; use `3` when ego likely could proceed but some context is uncertain; use `1` when the reason for waiting cannot be assessed.

### `oscillation`

Endpoint eligibility: `independent_consequence_endpoint`. This interaction-quality motif may be considered as a primary endpoint only in later authorized RQ012B work.

Definition: One or more involved actors repeatedly alternate between yielding and proceeding, or repeatedly change speed/heading intent, producing unclear negotiation.

Inclusion criteria:

- The item shows at least two visible reversals of proceed/yield intent or stop/go state.
- The alternation affects interaction resolution.
- The pattern is not simply normal smooth deceleration, queue movement, or a single cautious pause.
- Oscillation can involve ego, another actor, or both, as long as the interaction meaning is visible.

Exclusion criteria:

- A single pause followed by decisive movement.
- Smooth reciprocal negotiation where each actor's intent remains legible.
- Static deadlock without repeated alternation.
- Stop-and-go motion caused by visible queue movement or traffic-control cycling.

Onset rule: onset is the first reversal that begins the repeated alternation pattern. If the first reversal is unclear, use the earliest visible point where the oscillatory sequence becomes evident.

Worked example: In a fictional practice item, ego starts forward, stops, starts again, then stops as the other actor mirrors hesitation. Mark `oscillation=1`.

Counterexample: In a fictional practice item, ego slows once, the other actor passes, and ego continues. Do not mark `oscillation`.

Confidence guidance: use `5` when repeated reversals are visible and linked to negotiation; use `3` when alternation is visible but actor intent is partly ambiguous; use `1` if movement changes may be playback or measurement artifacts.

### `deadlock`

Endpoint eligibility: `independent_consequence_endpoint`. This interaction-quality motif may be considered as a primary endpoint only in later authorized RQ012B work.

Definition: Two or more involved actors reach a stalled negotiation state in which no actor makes effective progress because each waits, blocks, or defers in relation to the others.

Inclusion criteria:

- There is a visible multi-actor interaction.
- Progress stalls for a material interval beyond a normal brief yield.
- The stalled state is relational: at least two actors' choices contribute to lack of resolution.
- The item does not show a clear external reason that fully explains the stop.

Exclusion criteria:

- Ego alone freezes while other actors are not engaged; consider `over_yielding_freeze`.
- Actors are stopped for a visible queue, blocked path, or traffic-control state.
- The stall is too brief to distinguish from ordinary yielding.
- Evidence is insufficient to identify mutual waiting or blocking.

Onset rule: onset is the first visible moment the actors settle into mutual waiting, blocking, or stalled priority negotiation.

Worked example: In a fictional practice item, ego and another actor both stop at a shared conflict point, each waits for the other, and neither clears the conflict for a material interval. Mark `deadlock=1`.

Counterexample: In a fictional practice item, ego pauses while the other actor passes, then ego proceeds. Do not mark `deadlock`.

Confidence guidance: use `5` when mutual stalling is clear; use `3` when stalling is visible but one actor's reason is uncertain; use `1` when the stall could be fully explained by unseen constraints.

### `smooth_reciprocal_negotiation`

Endpoint eligibility: `independent_consequence_endpoint`. This interaction-quality motif may be considered as a primary endpoint only in later authorized RQ012B work.

Definition: Ego and other involved actors show legible, mutually adaptive behavior with continuous or efficiently resumed progress and no abrupt conflict escalation.

Inclusion criteria:

- Both ego and the relevant actor adapt in a way that makes priority resolution visible.
- Movement remains smooth or becomes smooth after a brief ordinary yield.
- Progress continues or resumes efficiently.
- There is no forced abrupt avoidance, repeated oscillation, deadlock, or unnecessary freeze.

Exclusion criteria:

- There is no meaningful interaction to negotiate.
- Ego dominates the interaction through unsafe intrusion.
- Ego remains stopped or creeping despite a clear opportunity.
- The interaction is unresolved, oscillatory, or evidence is insufficient.

Onset rule: onset is the first visible moment of mutual adaptation or priority resolution, such as one actor smoothly yielding while the other proceeds and then both resume normal motion.

Worked example: In a fictional practice item, ego slows slightly, another actor clears the shared space, and ego proceeds without abrupt braking or renewed hesitation. Mark `smooth_reciprocal_negotiation=1`.

Counterexample: In a fictional practice item, ego and another actor repeatedly start and stop before either proceeds. Do not mark `smooth_reciprocal_negotiation`; consider `oscillation`.

Confidence guidance: use `5` when reciprocal adaptation and smooth resolution are clear; use `3` when the interaction appears smooth but some actor intent is unclear; use `1` when there is not enough interaction evidence.

### `unrelated_failure`

Endpoint eligibility: `annotation_quality_label`. This label is not eligible as a primary event-IPV endpoint.

Definition: The item is dominated by a non-negotiation failure that prevents the behavioral interaction labels from being interpreted as interaction outcomes.

Inclusion criteria:

- The material has a visible display, playback, logging, or rendering failure that dominates interpretation.
- The item is dominated by an obvious task or control failure not tied to social negotiation.
- Actor trajectories or visual layers are so inconsistent that behavior cannot be interpreted as road-user interaction.
- The failure explains the apparent event better than any interaction-quality label.

Exclusion criteria:

- The item is simply difficult, ambiguous, or low-confidence but still behaviorally interpretable.
- The interaction contains poor negotiation but no separate material or non-negotiation failure.
- Evidence is missing or occluded without a visible failure mechanism; use insufficient-evidence protocol.

Onset rule: onset is the first visible moment the unrelated failure begins to dominate interpretation. If it affects the entire item, leave event times blank and describe the issue briefly.

Worked example: In a fictional practice item, the visual layer freezes while trajectories continue jumping, making actor behavior uninterpretable. Mark `unrelated_failure=1`.

Counterexample: In a fictional practice item, ego and another actor clearly hesitate in a visible negotiation. Do not mark `unrelated_failure` just because the behavior is poor.

Confidence guidance: use `5` when the material failure is obvious and dominant; use `3` when a non-negotiation failure is plausible but behavior is partly interpretable; use `1` and insufficient-evidence protocol when the failure itself is uncertain.

### `insufficient_evidence`

Endpoint eligibility: `annotation_quality_label`. This label is not eligible as a primary event-IPV endpoint.

Definition: The provided item does not contain enough visible evidence to make a reliable behavioral judgment for the relevant labels.

Inclusion criteria:

- Key actors, conflict space, timing, or priority context cannot be seen or inferred from the provided material.
- The material is too short, occluded, incomplete, or ambiguous to support a label.
- A possible behavior is visible but cannot be distinguished from a plausible benign explanation.
- The annotator would need outside information to decide.

Exclusion criteria:

- The annotator can make a low-confidence but evidence-based judgment; use the relevant label with low confidence instead.
- The item is dominated by an unrelated material or non-negotiation failure; use `unrelated_failure`.
- The label is simply absent with sufficient visible evidence; mark the behavior field `0`.

Onset rule: if evidence loss is localizable, onset is when the item first becomes insufficient for the behavior judgment. If insufficiency applies to the whole item, leave event times blank.

Worked example: In a fictional practice item, a possible crossing conflict occurs outside the visible frame, and only the aftermath is shown. Use the insufficient-evidence protocol.

Counterexample: In a fictional practice item, the full interaction is visible and ego simply yields to a clearly passing actor. Do not use insufficient evidence; mark relevant labels based on the visible behavior.

Confidence guidance: insufficient-evidence entries should normally use confidence `1`. Use `2` only when the template requires a confidence value and a small amount of partial evidence exists but is still below the threshold for a substantive label.

## Decision Tree And Quick Reference

1. Is the item viewable and behaviorally interpretable from the provided material alone?
   - No: use the insufficient-evidence protocol, unless a visible unrelated failure dominates.
   - Yes: continue.
2. Is interpretation dominated by a non-negotiation material, playback, logging, or task/control failure?
   - Yes: mark `unrelated_failure=1`; add concise notes.
   - No: continue.
3. Is there a visible multi-actor negotiation?
   - No: mark behavior labels `0` unless ego behavior still clearly meets `over_yielding_freeze` or `appropriate_assertiveness`.
   - Yes: continue.
4. Did ego force another actor into abrupt avoidance, late braking, sudden steering, or unsafe compression?
   - Yes: mark `aggressive_intrusion=1`.
5. Did ego proceed decisively through a reasonable visible opportunity without avoidable conflict?
   - Yes: mark `appropriate_assertiveness=1`.
6. Did ego stop, creep, or yield beyond visible need despite a feasible safe opportunity?
   - Yes: mark `over_yielding_freeze=1`.
7. Did the interaction show repeated proceed/yield or stop/go reversals?
   - Yes: mark `oscillation=1`.
8. Did two or more actors settle into mutual stalled negotiation?
   - Yes: mark `deadlock=1`.
9. Did the interaction resolve through legible mutual adaptation without abrupt conflict, oscillation, deadlock, or unnecessary freeze?
   - Yes: mark `smooth_reciprocal_negotiation=1`.
10. Assign confidence using the common scale and add concise notes only when they clarify evidence, timing, or uncertainty.

Conflict rule: labels are not mutually exclusive unless their definitions logically conflict over the same interval. If two labels occur in sequence, mark both when materially relevant and use timing fields only if the template can localize the primary interval. If timing fields cannot capture multiple intervals, use notes to identify the sequence without adding outside information.

Insufficient-evidence rule: do not infer from protected context. If a behavioral decision requires information outside the provided item, use the insufficient-evidence protocol instead of guessing.

## Training And Formal Validation Separation

Training items are fictional or neutral-ID stubs created for practice. Training keys, if used, are illustrative only and must not be copied into formal annotation templates.

Formal validation items are the only items that receive real human labels. The codebook author has not filled and must not fill formal labels. Annotators must not use training examples as evidence for any formal item.

