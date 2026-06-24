# RQ008 Decision: InterHub Temporal IPV Discovery

Status: ACCEPTED — negative, discovery-only boundary (knowledge-layer freeze, human-directed 2026-06-24). Confirmation hold-out never opened; Wave B frozen.

Run ID: `RQ008_1_temporal_ipv_discovery_20260622T234914+0800_3e3e776a`
Plan SHA-256: `3a94986587c3c30afa18b7a7c2151e272ac1382b24c43f8cbbf9a06b015f77db`
Basis for freeze: split gate PASS; discovery review PASS_RATIFIED; red team PASS_WITH_FIXES; replication PARTIAL; `reviews/claude_review.md` and `reviews/codex_review.md` both concur. Frozen at PI direction.

## Accepted Claims

| ID | Claim | Scope | Strength |
|---|---|---|---|
| RQ008-KC-NEGATIVE-DISCOVERY-BOUNDARY | 0/24 candidate directional temporal IPV structures (18 motifs, 2 alignments, 4 role/uncertainty effects) survived the Phase-7 mechanical/compositional control gate; apparent temporal "dynamics" are explained by time-reversal invariance, source composition, duration, and estimation uncertainty rather than a directional temporal IPV law. | InterHub sigma=0.1 discovery split (22,937 cases); directional temporal structure only. | Discovery-side robust (ratified, red-team PASS_WITH_FIXES, partial replication; change-point divergence does not affect 0/24). |

## Rejected Claims (unsupported by RQ008A)

Any positive motif, alignment, role/phase, reciprocity, complementarity, lead-lag, risk-IPV coupling, hysteresis, continuous feedback-adjustment, or IPV-mean temporal-law claim; any claim that uncertainty/estimability dynamics rescue an IPV-mean temporal claim. Reason: every candidate failed at least one matched control (most motifs by `reversed_time`).

## Pending / Conditional (Wave B only; frozen)

| ID | Status | Conditional claim |
|---|---|---|
| RQ008-KC-PENDING-PRIMARY-NULL | `FROZEN_RQ008A` | Held-out Wave B should again yield 0 survivors among the fixed 24-row catalogue after all controls + Holm. |
| RQ008-KC-PENDING-MOTIF-008-FALSIFICATION | `FROZEN_RQ008A` | MOTIF-008 joint-upward endpoint may falsify the null only if it passes every frozen control on held-out data (falsification-only). |
| RQ008-KC-PENDING-UNCERTAINTY-FALSIFICATION | `FROZEN_RQ008A` | Early-minus-late uncertainty decline may falsify only under the frozen controls (falsification-only). |

## Wave-B Governance

The protected confirmation split was never opened. Wave B requires explicit PI authorization AND the pre-Wave-B Attack-10 amendment (add a direction-sensitive companion statistic, or scope the null to directional structure exceeding reverse-invariant shape) before any confirmation data are read. Only Wave B can elevate or overturn this negative boundary.

## Boundaries

Single dataset/sigma (InterHub sigma=0.1); source imbalance (`waymo_train` 60.7%, `MP` geometry 90.3%); estimability is a proxy/error-based quantity (no frozen RQ007 contract was available to this run); reversed-time/SNR can over-kill reverse-invariant shape.

## Paper Handoff

Usable only as an exploratory negative discovery boundary / limitation / program-planning result, scoped to directional temporal IPV structure on InterHub sigma=0.1 discovery. Not a confirmed law; not evidence for a positive temporal IPV process.
