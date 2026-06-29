# Claude Code Review — RQ009 (Dynamic Envelope / M3)

Status: filed (2026-06-29)

Reviewer role: research reviewer / repository integrator.

Run reviewed: `RQ009_1_dynamic_envelope_20260625T121905Z_98c433de` (overall COMPLETE; all gates PASS; formal M3-vs-M4 gate `ESCALATE_TO_PI=false`).
Reader entry: `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/90_report/index.html`.

## Verdict

Concur. The accepted, manuscript-facing result is **R3: an estimability-aware, context-conditioned,
conformally calibrated dynamic envelope** — sharp, near-nominal, and auditable. This is the result the
PI accepted (2026-06-29) and it is sound and common-sense.

The conditioning **ablations** (counterpart-IPV = M3-over-M2, and ego self-IPV = M4) were part of internal
trial-and-error and added no measurable value over the context-only envelope. Per PI decision they are
**internal-only** and are **not a manuscript claim**; this review records them for traceability, not for
external argument.

## Key findings (manuscript-relevant)

| Check | Result | Reading |
|---|---|---|
| Context envelope vs floor (M2 vs M0, 90%) | width −42.3%, Winkler −35.6%, coverage ≈0.899 | Context supplies the sharpness; near-nominal coverage. |
| Calibration | split-conformal on final interval; abstention 4.78%; directional tails pass | Calibrated, auditable runtime monitor. |
| Conditioning ablations (internal) | M3≈M2≈ipv_removed (paired 90% Winkler diff −0.0002, p=0.863); M4 ≈ −2% width | No added value from IPV-conditioning channels; kept internal. |

## Boundaries

InterHub σ=0.1; empirical/probabilistic monitor (not a formal proof). Subgroup conditional validity is
uneven (126/264 supported rows outside ±3 pp) and LODO 90% coverage ranges 0.749–0.991 → transfer is not
unconditional. Target has a material exact-zero atom (~21.6%), qualifying endpoint/tie interpretation.

## Reproducibility / process

Identity gates verified; plan SHA pinned; case/scenario 4-way split; finite-sample conformal radius;
non-crossing quantiles; IPV-specificity controls (ipv_removed, kinematics_only) run; final review PASS.
The frozen Phase-5 metrics table and `m3_vs_m4_numbers.json` are auditable.

## Manuscript role

Supports **R3** as a context-conditioned, estimability-aware, conformally calibrated runtime envelope
(empirical monitor; risk excluded from online inputs; abstains out of support). Do **not** feature
counterpart/self IPV conditioning as a mechanism; phrase the envelope as context-conditioned. Keep all
wording bounded to the evaluated setting and to "monitor", not "proof" or "planner benefit".

## Recommendation

Accept R3 (context-conditioned envelope). Freeze an RQ009 `decision.md` to this effect; record the
IPV-conditioning ablations as internal null/ablation only. M3 (the frozen scorer) is usable downstream as
the context-conditioned envelope; downstream lines should not expect a distinct counterpart channel.

## Source pointers

- `02_process/06_m3_vs_m4/m3_vs_m4_verdict.md`, `02_process/05_evaluation/m3_vs_m4_numbers.json`, `metrics_summary.csv`
- `02_process/12_final_review/final_review.md`; `execution_status.json`
