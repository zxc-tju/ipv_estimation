# RQ001 Decision: Online IPV Interval Deployability

Status: ACCEPTED — bounded legacy result; engineering prior / M4 self-history ablation only (knowledge-layer freeze, human-directed 2026-06-24).

Primary run: `RQ001_3_online_interval_lock_20260619` (RQ001_1/RQ001_2 context only).
Basis: ChatGPT review "PASS WITH REQUIRED METHOD AND CLAIM BOUNDARIES" (`reviews/chatgpt_review.md`); frozen at PI direction.

## Accepted Claims (bounded to the locked balanced lane-referenced InterHub slice, 5k cases / 10k rows)

| ID | Claim |
|---|---|
| RQ001-KC-RISK | Risk/PET is not the main lever for interval sharpness: even oracle PET narrows the interval only ~3% vs global (≈0.833 vs 0.857). |
| RQ001-KC-SIGNAL | Strict-prefix map-lane causal rolling-IPV + split-conformal is the sharpest evaluated online IPV interval and the only compared method reaching ~0.90 Leave-Waymo-Out coverage (TEST 0.899/0.591; LWO 0.902/0.628; vs oracle PET 0.889/0.867, no-roll 0.896/0.738). |
| RQ001-KC-REF | The map-lane reference is online-admissible / route-conditioned (corr 0.993, MAE 0.027 vs observed-prefix 0.281) — non-anticipating, not a broad causal-effect claim. |
| RQ001-KC-CONFORMAL | Split-conformal is necessary (raw quantiles under-cover ~0.86); standard tools isolate the signal's contribution. |
| RQ001-KC-SUPPORT | ~74% of cases have lane/route support; ~26% fall back to a self-anchor-free causal-kinematics CQR interval. |

## Rejected / Deferred

| Claim | Reason |
|---|---|
| Unconditional cross-source robustness | Holds only on the balanced lane-referenced locked slice. |
| Hard-constraint deployment under source shift | Integrated LWO still < nominal; needs target-domain recalibration. |
| Universal lane-independent deployability | ~26% lack lane reference. |
| Strict whole-task no-leakage | Requires a non-overlapping post-anchor target rebuild (RQ002 E1). |
| Self-anchor alone = valid population norm | Outside RQ001; unsafe under RQ002. |

## Paper Handoff

Legacy engineering prior and **M4 self-history ablation only — not the new M3 result.** Do not reuse the 42%/0.902 A/B headline as new M3 numbers; keep the locked-interval table and the integrated-verifier A/B as separate protocols. Use two-tailed D_comp/D_yield downstream.
