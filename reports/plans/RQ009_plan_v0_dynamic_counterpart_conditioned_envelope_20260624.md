# RQ009 Plan v0 — Dynamic Counterpart-Conditioned Human Envelope (M3)

Status: `approved` (PI greenlit 2026-06-24) · Wave: B · Work group: Group 3 · Date: 2026-06-24

## 1. Research question

> Does an estimability-aware, conformally calibrated dynamic envelope conditioned on the current
> interaction state and the counterpart's current IPV (M3) produce a sharp, calibrated and
> source-transferable human reasonable interval — and how does it compare against the
> demoted ego self-anchor (M4)?

This is the program's central "working verifier" result (manuscript R3). It is the **pivot gate**:
the whole v4.1 framing assumes M3 is good enough without ego self-anchor as the primary signal.

## 2. Intended downstream role

Produces R3 and the **frozen M3 model + per-case predictions** consumed by RQ010B (WOD preference),
RQ011B (OnSite matched-scenario), RQ012B (event harm) and RQ013 (beyond-safety). No downstream line
may start its M3 step until M3 is frozen here.

## 3. Frozen inputs / contracts (must honour)

- RQ007 estimability/valid-window contract: separate interaction opportunity / estimability /
  human-reference support / deviation; abstain when not estimable; never read high uncertainty as neutral.
- RQ008 (negative): **no temporal motifs.** Use context + counterpart current IPV. Any
  estimable-window temporal structure is pre-registered and `\evidencepending` only.
- RQ004: R1 state-conditioned surface (risk × geometry × role × time) is the contextual norm.
- RQ002: ego self-anchor is **M4 ablation only**; the norm is the human population conditional distribution.
- RQ005: leakage contract.

## 4. Models (M0–M5)

| ID | Definition | Role |
|---|---|---|
| M0 | global scalar conformal interval | floor baseline |
| M1 | oracle PET-bin empirical envelope | offline ceiling/baseline (not deployable) |
| M2 | context-only conditional quantile + conformal | ablation |
| **M3** | **context + counterpart current IPV + conformal** | **PRIMARY** |
| M4 | context + ego self-anchor + conformal | sharpness ablation (RQ002 boundary) |
| M5 | source-aware / OOD-gated variants | robustness |

Target = same-window current rolling IPV on a post-anchor, non-overlapping window (E1 fix). Enforce
non-crossing quantiles; conformal radius `c_α = s_(⌈(n+1)(1−α)⌉)` on gate-passing calibration cases.

## 5. Denylisted (leakage) variables

Observed PET; realized passing order; closest-approach frame; post-hoc phase; full-window IPV; ego
early/self-anchor IPV in M3; target-proximal concurrent ego accel/braking; estimator-internal reward
components. (Online causal risk proxies — APET/closing-TTC — are allowed as context.)

## 6. Splits / gates

Case + scenario 4-way split: train / guard-tuning (freeze support/OOD + guard params) / calibration
(conformal radius only) / test (used once). No frame-level random split.

## 7. Endpoints + PASS/FAIL

Coverage @80/90/95 (marginal), mean width, pinball, Winkler; coverage by source/geometry/role/risk/
progress; abstention rate + post-abstention coverage; competitive vs over-yielding FPR; leave-one-
dataset-out transfer; kinematics-only and IPV-removed controls. PASS = M3 reaches near-nominal coverage
with materially smaller width than M0/M1/M2 and non-degenerate abstention.

**Decision gate (escalate to PI):** report the **M3-vs-M4** sharpness/coverage gap explicitly. If M3 is
materially worse than M4, the M3-primary framing must be revisited before downstream lines consume it.

## 8. Deliverables, stop conditions, claim boundaries

Deliverables: frozen M3 model + predictions; M0–M5 metric tables; offline HTML report; `decision.md`.
Stop: if M3 cannot reach acceptable coverage/width, freeze as a bounded/negative result, do not force it.
Claims: empirical/probabilistic monitor, not formal proof; provisional until external validation; no
planner-benefit claim.

## 9. Dependencies

Upstream: all frozen (RQ007/008/004/002/005). Downstream: gates RQ010B-B2, RQ011B, RQ012B, RQ013.
