# Codex Review: RQ008 InterHub Temporal IPV Discovery

Status: review-complete; study accepts the negative discovery result; knowledge-layer synthesis/decision files are missing.
Review date: 2026-06-24.

## Scope

Reviewed study package:

- `reports/studies/RQ008_interhub_temporal_ipv_discovery/RQ008_1_temporal_ipv_discovery_20260622T234914+0800_3e3e776a/`

Reviewed knowledge-layer files:

- `reports/knowledge/RQ008_interhub_temporal_ipv_discovery/report_index.md`

Primary study evidence read:

- `00_entry/index.html`
- `final_decision.md`
- `final_review.md`
- `freeze_synthesis.md`
- `execution_status.json`
- `reviews/independent_review/discovery_review.md`
- `reviews/red_team/red_team_report.md`
- `reviews/replication/replication_compare.md`

## Overall Verdict

RQ008A should be carried forward as a high-quality negative discovery result and
as a guardrail against overclaiming temporal IPV laws. The strongest supported
claim is simple: after the planned discovery controls, zero of twenty-four
candidate temporal structures survived. The confirmation split remains unopened,
and Wave B must not begin without explicit authorization and a pre-registered
direction-sensitive amendment.

Paper-safe phrasing:

> A broad InterHub temporal-discovery sweep found no robust directional IPV
> motif, alignment, role, or uncertainty structure after matched controls. This
> negative result constrains temporal-law claims but does not rule out all
> source-local, reverse-invariant, or differently parameterized dynamics.

## Claims That Can Be Carried Forward

1. The RQ008A discovery result is negative: 0 of 24 candidate structures
   survived the Phase-7 control stack.
2. The tested candidates covered motif structures, alignment structures, role
   effects, and uncertainty or estimability effects.
3. The confirmation split was protected and not opened. The discovery split
   contains 22,937 cases and 2,218,410 frames; the confirmation split remains a
   future test surface.
4. The frozen primary confirmation endpoint is
   `n_surviving_structures = 0`, with a primary null candidate plus two explicit
   falsification targets.
5. Oracle or circular alignments are disallowed for confirmatory claims. The
   package correctly treats `estimability_onset_proxy` as circular and
   `offline_oracle_phase` as future/full-window information.
6. The result is useful as a manuscript boundary: it supports saying that the
   current evidence does not reveal a robust directional temporal IPV law in the
   tested InterHub design.

## Claims To Reject Or Defer

- Do not claim any positive temporal law or stable motif from RQ008A. The best
  apparent candidates were killed by controls.
- Do not claim confirmation. The protected confirmation split has not been
  opened.
- Do not claim the null covers all possible temporal dynamics. The red-team
  review identified source-composition imbalance, cluster instability, and
  possible over-killing of reverse-invariant structure.
- Do not start Wave B, tune the protocol, or open protected confirmation data
  without explicit authorization.
- Do not use source-mixed global null results to reject source-local behavior
  unless a source-stratified design is run.

## Quality And Compliance Notes

The independent review ratifies the discovery package, the final review accepts
the negative headline, and the study final decision records RQ008A as complete.
The red-team review passes with fixes rather than blockers, but it raises a real
scope limitation: the reversed-time hard gate can destroy reverse-invariant
motifs, so Wave B needs a direction-sensitive companion statistic or an explicit
scope statement that the primary null is directional only.

The replication review is partial because the change-point implementation
diverged, but that divergence does not alter the headline result that 0 of 24
candidate structures survived.

## Knowledge-Layer Action

The knowledge folder currently has an index but no synthesis or decision file.
Recommended decision state: accept RQ008A as a negative discovery-only result,
freeze the confirmation split, require explicit authorization before Wave B, and
add the red-team direction-sensitive amendment before any confirmatory opening.
