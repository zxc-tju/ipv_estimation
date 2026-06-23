# RQ012A Codebook v2 Label Detail

Worker: `RQ012-W11-codebook-v2`
Run ID: `RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37`
Phase: phase7
Deliverable: codebook_v2

This file is the long-form protocol record behind `01_results/annotation_codebook_v2.md`. It is internal protocol material for RQ012A readiness. It contains no real labels, no agreement results, no event-IPV analysis, and no model-generated annotations.

## Binding Inputs Applied

- RQ003 source codebook was used only as a starting vocabulary.
- RQ012A binding addendum B01 was applied by assigning `endpoint_eligibility` to every label.
- RQ012A binding addendum A02 was applied by making the annotation rule neutral-ID-only.
- RQ012A event ontology was used to align interaction-quality motifs and annotation-quality labels.
- Corrected issuance notes were used as the annotator-release baseline.

## Corrected Exposure Language

The prior source codebook allowed annotators to receive scenario labels and viewing order. That language is superseded. Annotators must receive neutral item IDs only, and annotator-facing materials must not expose scenario labels, scenario family, viewing order, strata, paths, team or area identifiers, official scores, ranks, IPV outputs, filenames, thumbnails, manifest-derived strata, or prior/borrowed annotation files.

The allowed annotator-facing instruction is: use only the neutral item ID and the visible behavior in the provided item.

## Endpoint Eligibility Detail

`construct_proximal_descriptor` labels semantically restate assertive, competitive, or over-yielding behavior. They are useful as descriptive context but circular as primary event-IPV validation endpoints.

`independent_consequence_endpoint` labels describe interaction-quality outcomes that are conceptually more separable from the IPV construct. They may be eligible for later RQ012B primary endpoint use only after separate authorization and all required gates.

`annotation_quality_label` labels describe interpretability or exclusion state. They are never primary event-IPV endpoints.

| Label | endpoint_eligibility | primary_endpoint_eligible | Reason |
|---|---|---:|---|
| `aggressive_intrusion` | `construct_proximal_descriptor` | false | Semantically close to competitive or intrusive behavior. |
| `appropriate_assertiveness` | `construct_proximal_descriptor` | false | Semantically close to assertive IPV interpretation. |
| `over_yielding_freeze` | `construct_proximal_descriptor` | false | Semantically close to over-yielding IPV interpretation. |
| `oscillation` | `independent_consequence_endpoint` | true | Interaction-quality motif separable from a single IPV direction. |
| `deadlock` | `independent_consequence_endpoint` | true | Stalled negotiation consequence separable from a single IPV direction. |
| `smooth_reciprocal_negotiation` | `independent_consequence_endpoint` | true | Resolution-quality motif separable from a single IPV direction. |
| `unrelated_failure` | `annotation_quality_label` | false | Interpretability/exclusion label. |
| `insufficient_evidence` | `annotation_quality_label` | false | Evidence-quality label. |

## Long-Form Label Rules

All examples in this section are fictional and training-only. They are not derived from real outcomes, official results, IPV values, prior annotations, formal validation items, or model output.

### `aggressive_intrusion`

- endpoint_eligibility: `construct_proximal_descriptor`
- primary_endpoint_eligible: false
- permitted use: secondary descriptive context only

Definition: Ego enters, occupies, or continues into another road user's conflict space in a way that visibly pressures that road user into abrupt avoidance, late braking, sudden steering, or unsafe compression.

Inclusion criteria:

- Ego visibly commits into a shared conflict space.
- Another actor's relevant path or priority is visible before or during ego commitment.
- The other actor responds abruptly or is left with visibly unsafe compression.
- Ego's motion is a material contributor to the response.

Exclusion criteria:

- The gap is visibly adequate and other actors remain stable.
- Other actors yield early, smoothly, and predictably.
- The conflict is better explained by a third actor, material failure, or missing context.
- The annotator would need protected or outside information to decide.

Onset rule: first visible ego commitment into the conflict space that directly leads to the pressured response.

Worked example: a fictional item shows ego entering a narrow crossing gap and another actor braking hard to avoid overlap. This supports `aggressive_intrusion=1`.

Counterexample: a fictional item shows ego proceeding after another actor has already slowed with ample spacing. This does not support `aggressive_intrusion`.

Confidence scale: 5 for clear ego commitment plus clear pressured response; 4 for strong evidence with minor uncertainty; 3 for plausible but partly ambiguous priority or spacing; 2 for weak cues; 1 for insufficient evidence.

### `appropriate_assertiveness`

- endpoint_eligibility: `construct_proximal_descriptor`
- primary_endpoint_eligible: false
- permitted use: secondary descriptive context only

Definition: Ego proceeds decisively through a reasonable visible opportunity in a legible way that supports interaction resolution without avoidable conflict.

Inclusion criteria:

- A safe visible opportunity exists.
- Ego's proceed action is smooth and predictable.
- Other actors do not need abrupt avoidance.
- The action resolves or advances the interaction.

Exclusion criteria:

- Ego creates unsafe compression or a forced response.
- Ego alternates intent before committing.
- Ego remains stopped or creeping despite an open opportunity.
- The item lacks enough context to judge appropriateness.

Onset rule: first visible ego movement that commits to the reasonable proceed action.

Worked example: a fictional item shows ego waiting briefly, then entering a clear gap smoothly while the other actor maintains stable motion. This supports `appropriate_assertiveness=1`.

Counterexample: a fictional item shows ego accelerating into a narrowing gap and forcing a hard brake. This does not support `appropriate_assertiveness`.

Confidence scale: 5 for clear opportunity and smooth resolution; 4 for strong but not perfect context; 3 for adequate evidence with some ambiguity; 2 for weak evidence; 1 for insufficient evidence.

### `over_yielding_freeze`

- endpoint_eligibility: `construct_proximal_descriptor`
- primary_endpoint_eligible: false
- permitted use: secondary descriptive context only

Definition: Ego stops, creeps, or yields beyond visible need despite an apparent feasible safe opportunity to proceed.

Inclusion criteria:

- Ego has a visible opportunity to proceed safely.
- Ego remains stopped, creeping, or yielding without a visible reason.
- The delay materially affects interaction resolution.
- The behavior is not fully explained by visible external constraints.

Exclusion criteria:

- Ego waits for a visible actor with priority, an obstacle, a blocked way, or a traffic-control state.
- The opportunity is not visible enough to judge.
- The pause is brief and followed by smooth progress.
- The issue is a material failure rather than behavior.

Onset rule: first visible moment ego remains stopped, creeping, or yielding after the feasible opportunity becomes visible.

Worked example: a fictional item shows nearby actors clearing while ego continues creeping without entering. This supports `over_yielding_freeze=1`.

Counterexample: a fictional item shows ego waiting for a clearly passing actor and then proceeding. This does not support `over_yielding_freeze`.

Confidence scale: 5 for clear open opportunity plus unnecessary delay; 4 for strong visible evidence; 3 for adequate but partly uncertain opportunity; 2 for weak cues; 1 for insufficient evidence.

### `oscillation`

- endpoint_eligibility: `independent_consequence_endpoint`
- primary_endpoint_eligible: true, subject to later RQ012B authorization
- permitted use: interaction-quality endpoint candidate

Definition: Involved actors show repeated alternation between proceed and yield intent, or repeated stop/go or heading-intent reversals, causing unclear negotiation.

Inclusion criteria:

- At least two visible reversals are present.
- The reversals are linked to the interaction.
- The pattern affects progress or priority resolution.
- The pattern is more than a single cautious pause.

Exclusion criteria:

- One smooth yield followed by progress.
- Static mutual waiting without repeated reversals; consider `deadlock`.
- Queue creep or visible traffic-control cycling.
- Movement changes that appear to be display artifacts.

Onset rule: first reversal that starts the repeated alternation sequence.

Worked example: a fictional item shows ego start, stop, start, and stop while another actor mirrors the hesitation. This supports `oscillation=1`.

Counterexample: a fictional item shows ego slowing once so another actor can pass. This does not support `oscillation`.

Confidence scale: 5 for clear repeated linked reversals; 4 for strong evidence with minor uncertainty; 3 for adequate alternation with some ambiguity; 2 for weak cues; 1 for insufficient evidence.

### `deadlock`

- endpoint_eligibility: `independent_consequence_endpoint`
- primary_endpoint_eligible: true, subject to later RQ012B authorization
- permitted use: interaction-quality endpoint candidate

Definition: A multi-actor negotiation stalls because actors mutually wait, block, or defer such that no effective progress occurs for a material interval.

Inclusion criteria:

- At least two involved actors are visible.
- Progress stalls beyond a normal brief yield.
- The stalled state is relational rather than fully explained by one actor.
- No visible external constraint fully accounts for the stall.

Exclusion criteria:

- Ego alone freezes; consider `over_yielding_freeze`.
- Actors are stopped for a visible external reason.
- The pause is too brief to distinguish from normal yielding.
- Mutual relation cannot be judged from the material.

Onset rule: first visible moment actors settle into mutual waiting, blocking, or stalled priority negotiation.

Worked example: a fictional item shows ego and another actor each stopping at the conflict point while neither proceeds. This supports `deadlock=1`.

Counterexample: a fictional item shows ego briefly waiting while another actor passes. This does not support `deadlock`.

Confidence scale: 5 for clear mutual stall; 4 for strong evidence; 3 for adequate but partly ambiguous actor intent; 2 for weak cues; 1 for insufficient evidence.

### `smooth_reciprocal_negotiation`

- endpoint_eligibility: `independent_consequence_endpoint`
- primary_endpoint_eligible: true, subject to later RQ012B authorization
- permitted use: interaction-quality endpoint candidate

Definition: Involved actors adapt legibly to each other, resolve priority smoothly, and maintain or efficiently resume progress without abrupt escalation.

Inclusion criteria:

- A meaningful interaction is visible.
- Actors adapt in a mutually legible way.
- Progress continues or resumes efficiently.
- No aggressive intrusion, unnecessary freeze, oscillation, or deadlock dominates the interval.

Exclusion criteria:

- No real interaction is visible.
- One actor forces an abrupt response.
- The interaction remains unresolved or repeatedly reverses.
- Evidence is too limited to judge reciprocal adaptation.

Onset rule: first visible moment of mutual adaptation or priority resolution.

Worked example: a fictional item shows ego slowing slightly while another actor clears, then ego proceeds smoothly. This supports `smooth_reciprocal_negotiation=1`.

Counterexample: a fictional item shows repeated start-stop hesitation before either actor proceeds. This does not support `smooth_reciprocal_negotiation`.

Confidence scale: 5 for clear mutual adaptation and smooth resolution; 4 for strong evidence; 3 for adequate but partly unclear actor intent; 2 for weak cues; 1 for insufficient evidence.

### `unrelated_failure`

- endpoint_eligibility: `annotation_quality_label`
- primary_endpoint_eligible: false
- permitted use: exclusion or interpretability context

Definition: A visible material, playback, logging, rendering, task, or control failure dominates interpretation and prevents the item from serving as a clean behavioral interaction label.

Inclusion criteria:

- Material display or trajectory rendering failure dominates interpretation.
- A non-negotiation task/control failure explains apparent behavior better than interaction.
- Actor traces are inconsistent enough that behavior cannot be read.
- The failure is visible in the item itself.

Exclusion criteria:

- The item is difficult but still behaviorally interpretable.
- The interaction itself is poor or unsafe without a separate unrelated failure.
- Key evidence is missing without a visible failure; use insufficient-evidence protocol.

Onset rule: first visible moment the unrelated failure begins to dominate. Leave times blank when it applies to the whole item.

Worked example: a fictional item shows the visual frame freezing while trajectory traces jump discontinuously. This supports `unrelated_failure=1`.

Counterexample: a fictional item shows visible hesitation and conflict but no material failure. This does not support `unrelated_failure`.

Confidence scale: 5 for obvious dominant failure; 4 for strong failure evidence; 3 for plausible failure with partial interpretability; 2 for weak cues; 1 for insufficient evidence.

### `insufficient_evidence`

- endpoint_eligibility: `annotation_quality_label`
- primary_endpoint_eligible: false
- permitted use: evidence-quality and exclusion context

Definition: The provided material does not contain enough visible evidence to make a reliable behavior judgment.

Inclusion criteria:

- Relevant actors or conflict point are outside view.
- Timing, priority, or actor relation cannot be judged.
- The material is incomplete, occluded, too short, or too ambiguous.
- Deciding would require outside information.

Exclusion criteria:

- Evidence is adequate for a low-confidence substantive label.
- A visible unrelated material failure dominates; use `unrelated_failure`.
- The label is simply absent with sufficient evidence; mark it `0`.

Onset rule: if evidence loss is localizable, use the first moment evidence becomes insufficient. Otherwise leave times blank.

Worked example: a fictional item starts after the possible conflict has already occurred, so the annotator cannot tell whether intrusion, yielding, or smooth negotiation happened. Use the insufficient-evidence protocol.

Counterexample: a fictional item clearly shows ego waiting for a passing actor and then proceeding. Do not use insufficient evidence.

Confidence scale: use `1` for whole-item insufficiency. Use `2` only for partial evidence below the threshold for a substantive label.

## Formal Template Compatibility

The current formal templates include seven behavior columns and no separate `insufficient_evidence` column. Because W11 must not edit W06 formal templates, this codebook defines a compatible entry rule: set all substantive behavior fields to `0`, leave event times blank, use confidence `1`, and write `INSUFFICIENT_EVIDENCE:` with a concise reason in `free_text_notes`.

This compatibility rule is a protocol instruction only. It is not a real label and does not fill any formal item.

## Disagreement Interpretability Protocol

Later agreement review should distinguish four disagreement types:

- Presence disagreement: one annotator marks a label present and the other marks it absent.
- Boundary disagreement: both identify a behavior but disagree on timing.
- Motif disagreement: annotators choose different labels for the same visible behavior, such as oscillation versus deadlock.
- Evidence-quality disagreement: one annotator uses insufficient-evidence or unrelated-failure protocol while the other applies a substantive label.

Annotators should use concise notes to explain visible cues, not protected context. Notes should be factual and short, such as "repeated stop-go before clearing" or "actor relation not visible."

