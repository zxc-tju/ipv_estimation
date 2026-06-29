# ChatGPT Review — RQ009 Dynamic Envelope

Date: 2026-06-29  
Reviewer: ChatGPT  
Status: `review-complete`  
Scope: `RQ009_1_dynamic_envelope_20260625T121905Z_98c433de` and the knowledge-layer decision.

## Verdict

Concur with the accepted RQ009 decision, with one important manuscript-boundary clarification: the supported result is **an estimability-aware, context-conditioned split-conformal envelope**, not a distinct counterpart-IPV mechanism and not a self-anchor verifier.

The result is strong enough to be the manuscript R3 method result: context conditioning produces a materially sharper empirical envelope than a global floor while remaining near-nominal marginally and preserving an abstention interface.

## Accepted manuscript-facing result

Paper-safe reading:

> An estimability-aware, context-conditioned split-conformal envelope provides a sharp, near-nominal empirical runtime monitor for current IPV, with explicit out-of-support abstention.

Key values from the accepted run:

- 90% coverage approximately `0.899`;
- 90% width approximately `1.016`;
- 90% Winkler approximately `1.423`;
- context model versus global floor: width `−42.3%`, Winkler `−35.6%`;
- abstention approximately `4.78%`.

This should be written as a **context-conditioned conformal envelope** result.

## Internal nulls / ablations

The IPV-conditioning channels are internal ablations, not manuscript claims:

- M3 / context + counterpart IPV is effectively the same as context-only / IPV-removed.
- M4 / ego self-history adds only marginal sharpness.
- M3-vs-IPV-removed 90% Winkler effect is about `−0.0002`, CI spans zero, case sign p about `0.863`.

Therefore the paper must not claim that counterpart current IPV is the active mechanism, nor that ego self-history defines the norm.

## Boundaries

- InterHub sigma=0.1 only.
- Empirical/probabilistic monitor, not a formal proof.
- Conditional subgroup coverage is uneven: `126/264` supported subgroup rows outside ±3 pp.
- LODO 90% coverage ranges roughly `0.749–0.991`, so transfer is not unconditional.
- Exact-zero target atom around `21.6%` qualifies endpoint/tie interpretation.
- No planner benefit or realised-harm validity is established by RQ009 alone.

## Recommended manuscript use

Use RQ009 for:

```text
R3 — context-conditioned conformal runtime envelope
```

Do not use RQ009 for:

```text
counterpart-conditioned IPV mechanism
self-anchor verifier
IPV-specific new information beyond kinematics
external validation
planner performance
```

## Downstream implications

RQ009 clears a frozen scorer for downstream RQ010B, RQ011B, RQ012B, and any later RQ013 work, but downstream analyses should treat that scorer as **context-conditioned**. They should not expect or advertise a distinct counterpart-IPV channel.

## Final recommendation

Accept RQ009 as the paper's core method result, but revise all title, abstract, figure, and Results language to remove counterpart-IPV and self-anchor mechanism claims. The safe headline is:

> A context-conditioned, estimability-aware conformal envelope is sharp and near-nominal as an empirical runtime monitor.
