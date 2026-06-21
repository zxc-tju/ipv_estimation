# Codex Review: RQ004 IPV State Space

Status: review-complete, not yet frozen in `decision.md`.

Review date: 2026-06-21.

## Scope

Reviewed study packages:

- `reports/studies/RQ004_ipv_state_space/RQ004_1_state_space_law_nature_20260618/`
- `reports/studies/RQ004_ipv_state_space/RQ004_2_nature_conclusions_multiagent_20260618/`
- `reports/studies/RQ004_ipv_state_space/RQ004_3_generalizable_conclusions_20260618/`

The third package is the cleanest claim-indexed reader artifact and should be
treated as the canonical RQ004 review baseline unless a later report supersedes
it.

## Overall Verdict

RQ004 supports the manuscript claim that social compliance/IPV is organized by
state context rather than by a single global score. It does not support a strong
"generalizable state-space law" or cross-dataset predictive-law claim.

Paper-safe phrasing:

> IPV-based social behavior should be interpreted as a state-conditioned
> response surface over risk, geometry, role and time. Coarse state components
> provide useful normative structure, but the current evidence does not prove a
> dataset-transferable predictive law.

## Claims That Can Be Carried Forward

1. **Priority is risk-modulated, not a static social label.**
   The strongest report-level numbers are priority-minus-nonpriority IPV
   `+0.058` at `PET<=1.0 s`, near zero in the middle range, and `-0.034` at
   `PET>2.0 s`. This can support a conditional negotiation claim, not a claim
   that priority agents are always more prosocial.

2. **Coarse road geometry is a stable behavioral prior.**
   MP - non-MP and S-S - non-S-S contrasts are positive in all four datasets.
   This is strong enough for planner-facing stratification. Fine topology cells
   such as HO/U-turn/small relation-turn cells are too sparse for headline
   generalization.

3. **Social signal often appears before the annotated conflict window.**
   First non-zero IPV appears before interaction start in a majority of cases
   across datasets (`AV2=51.1%`, `Lyft=67.6%`, `Waymo=68.1%`,
   `nuPlan=75.1%`). Treat this as descriptive/replay evidence for
   pre-conflict negotiation, not as causal online early-warning validation.

4. **AV/HV sociality is not a fixed scalar trait.**
   AV-HV differences change sign by dataset, PET risk, path state and priority
   boundary. A single "AVs are more/less social" score should be rejected.

5. **The best transferable manuscript line is the interaction-state response
   surface.**
   RQ004_3's P5 framing is the most useful synthesis: the stable contribution
   is state-space modeling for AV interaction assessment, not a pooled
   vehicle-type comparison.

## Claims To Reject Or Defer

- **Reject:** "Full state-space generalizes across datasets as a predictive
  law." RQ004_1 explicitly marks this unsupported because leave-one-dataset-out
  performance is negative and source imbalance remains strong.
- **Reject:** "AV/HV scalar sociality is a valid headline result." The result
  is context-dependent and source-sensitive.
- **Defer:** OnSite held-out validation. RQ004_1 only provides a protocol; it
  does not provide OnSite validation results.
- **Defer:** causal claims about social behavior. The current evidence is
  observational and relies on annotated/replay windows.

## Knowledge-Layer Action

Update `synthesis.md` to separate:

- accepted RQ004 framing: state-conditioned IPV response surface;
- qualified evidence: risk-modulated priority, coarse geometry priors,
  pre-conflict replay signal;
- rejected overclaim: generalizable state-space law.

Do not mark RQ004 as accepted until `decision.md` freezes the exact paper-safe
wording and cites the RQ004_3 evidence JSON plus the RQ004_1 falsification table.
