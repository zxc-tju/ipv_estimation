# RQ007 Decision: Interaction-Conditioned IPV Estimability

Status: ACCEPTED — development/guard estimability boundary (knowledge-layer freeze, human-directed 2026-06-24). Held-out sealed; not yet held-out confirmed.

Run ID: `RQ007_1_ipv_estimability_20260622T155229Z_289d9a99`
Plan SHA-256: `7c33f7be76cce64fe2e5d17e4cd6be72435c51216c7a779c2e29f7626912f3b8`
Basis for freeze: study final no-blocker review PASS (all gates); independent review PASS (0 blocking, 0 major, 2 minor); red team PASS; replication PASS/MIXED; `reviews/claude_review.md` and `reviews/codex_review.md` both concur. Frozen at PI direction.
Source of truth: `02_process/11_final_review/conclusions.md`, `claim_evidence_matrix.csv`; `synthesis.md`.

## Accepted Claims

| ID | Claim | Strength | Paper-safe wording |
|---|---|---|---|
| RQ007-KC-C1 | Estimability is interaction-conditioned but mostly proximity-driven | MODERATE, strong boundaries | Within causal interaction-opportunity windows the per-frame IPV concentration index is lower (total ≈ -0.13; dev -0.132 / guard -0.129; replicated -0.134 / -0.133), time-locked (time-shift +0.006) and counterpart-specific (permutation +0.021; re-est. switch +0.122). Most of the gap (≈ -0.096) is spatial proximity; the conflict-geometry-specific residual is small but nonzero (-0.032 to -0.036, case-clustered CIs exclude zero). |
| RQ007-KC-C2 | Estimability ≠ behavioural settling | SUPPORTED | Under estimable (low-index) frames the current IPV estimate still moves (\|dθ\| ≈ 0.30 ego / 0.31 counterpart); high index ≠ IPV 0. Concentration index, current estimate, and episode summary are distinct constructs. |
| RQ007-KC-C3 | Episode IPV summary is definition-dependent | SUPPORTED | All-valid vs interaction-active means differ ≈0.26 rad and flip strict sign in ≈22% of cases; estimability-weighting reduces but does not remove this (~7% flips, Spearman ≈0.91). An episode IPV must state its summary rule. |

## Rejected Or Deferred Claims

| Claim | Reason |
|---|---|
| The full -0.13 gap is conflict-caused | Proximity/history explains the majority; only the small residual is conflict-specific. |
| Temporal precedence / causal onset | Lifecycle is descriptive; ~44–46% of onsets precede the opportunity frame. |
| Held-out confirmation | Sealed split untouched; all claims are development/guard only. |
| Latent IPV truth, planner performance, normative-behaviour validation | RQ007 is an estimator-validity package, not a behavioural-truth package. |
| Estimator-input reruns as robustness proof | Sanity check only (recompute mismatch mean ≈0.11, p95 ≈0.45). |

## Boundaries

Development (19,258) / guard (7,628) only; sealed (11,342) untouched. Opportunity = `cv_cpa_conflict`. "Estimability" = estimator concentration index (identifiability proxy), not a standard deviation. No map/lane, observed PET, intensity, order, priority, or outcome fields used.

## Paper Handoff

Manuscript may use C1–C3 only with the proximity-bounded caveat, the "interaction-conditioned estimability" framing (never "conflict causes IPV identifiability" or "IPV stabilizes during negotiation"), and provisional wording until held-out confirmation under the frozen contract. Underwrites the v4.1 estimability contract and feeds RQ009.
