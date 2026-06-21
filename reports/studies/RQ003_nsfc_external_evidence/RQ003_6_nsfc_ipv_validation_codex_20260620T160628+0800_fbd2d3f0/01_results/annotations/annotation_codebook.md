# Blind Behavioral Annotation Codebook

This package prepares a blind behavioral criterion reference independent of the official score and IPV outputs. It is not a completed annotation result. Do not consult team names, official scores, ranks, IPV values, model outputs, or the controlled identity map while coding.

## Unit Of Annotation

Code one blinded replay item at a time. Use only the anonymized behavior material supplied by the coordinator for that blinded item. Record the primary label that best describes the focal interaction behavior, plus brief behavioral evidence notes.

## Required Primary Labels

Use exactly one primary label from this list.

| CSV value | Definition | Decision rule |
|---|---|---|
| aggressive intrusion | The focal vehicle enters, cuts across, or presses into a contested gap in a way that forces the counterpart to brake, evade, or abandon its expected motion. | Use when the behavior is assertive beyond normal negotiation and the counterpart's feasible space is visibly compressed by the focal action. |
| appropriate assertiveness | The focal vehicle takes initiative or claims a gap while preserving enough spacing and timing for reciprocal adjustment. | Use when the focal vehicle progresses decisively but the counterpart is not trapped into an abrupt avoidance response. |
| over-yielding-freeze | The focal vehicle delays, stops, or yields for an extended period despite an available safe opportunity to proceed. | Use when excessive caution or hesitation is the main interaction feature rather than external blockage. |
| oscillation | The focal vehicle repeatedly alternates between advance and yield, or shows repeated start-stop reversals that destabilize negotiation. | Use when the central feature is unstable intent signaling across multiple cycles. |
| deadlock | Both agents remain unable or unwilling to resolve precedence for a sustained period. | Use when the interaction stalls because neither side completes a clear yield or pass decision. |
| smooth reciprocal negotiation | The agents mutually adjust speed or path with clear, stable precedence and no visible forced avoidance. | Use when the interaction resolves with coordinated give-and-take. |
| unrelated failure | The replay item cannot be interpreted as the target social interaction because another failure dominates the scene. | Use for perception/logging failures, map or route problems, noninteraction crashes, missing counterpart behavior, or other defects unrelated to behavioral negotiation. |

## Secondary Fields

- `secondary_label_optional`: Optional. Use only when a second label is clearly present and materially qualifies the primary label.
- `confidence`: Required for completed human coding. Use `high`, `medium`, or `low`.
- `evidence_notes`: Required for completed human coding. Describe observable behavior only, without team identity, score, rank, IPV, or model-output language.
- `cannot_code_reason`: Use only when the item cannot be coded. If used, the primary label should normally be `unrelated failure`.

## Blinding Rules

- Do not infer or record team names, official scores, ranks, or IPV values.
- Do not open `controlled_identity_map.csv` during annotation.
- Do not use selection-rationale fields from mechanism manifests as behavioral evidence.
- Do not add outcome, model-output, or ranking language to annotator templates.

## Adjudication Guidance

If two labels seem plausible, choose the one that explains the dominant interaction failure or success. Prefer behavior visible in the interaction itself over inferred intent. When evidence is ambiguous, use `confidence=low` and explain the ambiguity in `evidence_notes`.
