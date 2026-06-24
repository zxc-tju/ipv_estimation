# RQ012 Decision: OnSite Event Annotation Readiness

Status: BLOCKED_FOR_HUMAN_LABELS — Wave-A annotation readiness accepted as a protocol/scaffold; substantive event-annotation evidence DEFERRED (knowledge-layer freeze of the deferral, human-directed 2026-06-24).

Run ID: `RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37`
Plan SHA-256: `921f6bb3b850126189895dcca52f054a1c6f5e95a16b05159bec0a13c26ad77e`
Basis for freeze: final review PASS (offline report, links/figure provenance, no simulated labels, no event-IPV association, explicit blocked boundary); `reviews/claude_review.md` and `reviews/codex_review.md` both concur. Frozen at PI direction.

## Accepted Claims (readiness only)

| ID | Claim |
|---|---|
| RQ012-KC-READINESS | Wave-A annotation readiness: Gates 012-0/012-1 pass; 012-2 text/surface-cleared; 012-3 ready-pending-humans; **012B blocked**. A blinded, outcome-free event-labeling design and extractor-readiness checks exist. |
| RQ012-KC-NO-LABELS | No simulated/real labels were introduced (`human_labels_present=false`; blank templates). No event-IPV, event-score, event-rank, or team-identity association is computed or claimed. |
| RQ012-KC-CODEBOOK | The codebook separates automatic, human-only, and removed events; construct-proximal labels are secondary only. The automatic pilot is extractor/data-health evidence only (computability, counts, precedence suppression, sampling-rate sensitivity). |

## Deferred / Blocked Claims

| Claim | Reason |
|---|---|
| Behavioural validation; event-IPV / event-outcome association; event-rate generalization | No human labels; blocked by design. |
| Human annotation complete | Two accepted independent labels + kappa + AC1 still required. |
| Automatic event counts as scientific outcomes | Pilot shows extractor stability/weaknesses only. |
| Construct-proximal labels (aggressive intrusion, appropriate assertiveness, over-yielding freeze) as primary endpoints | Secondary-only by design. |

## Gate 012B Dependencies (all required before opening)

Real two-human blinded labels + kappa/AC1 agreement; final neutral media/card issuance; auditor sign-off; upstream freezes RQ007 (now frozen), RQ011 (now frozen), and RQ009 (still pending); explicit Gate 012B authorization.

## Paper Handoff

Usable only as a protocol/scaffold and a readiness statement; NOT as evidence of event/harm annotation or interaction-consequence measurement. Prepares the behaviour/consequence reference for RQ012B and RQ013; cannot advance without the dependencies above. Note: `evidence.csv` is header-only — populate when 012B runs.
