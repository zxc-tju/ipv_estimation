# Real Optimizer Sign Reconfirmation

Worker: `RQ003_phase4_prep_001`
Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`
Status: `PASS`
Generated: `2026-06-20T18:52:54+08:00`

## What Was Run

The real estimator path was executed through `sociality_estimation.core.ipv_estimation.estimate_ipv_pair`, which calls `Agent.estimate_self_ipv` and `Agent.solve_optimization`; `solve_optimization` invokes `scipy.optimize.minimize(..., method="SLSQP")`. No stubbed theta values were used.

Input trajectories came from allowed InterHub raw pkl samples joined through the repository pipeline helpers. No NSFC coordination, efficiency, comprehensive score, rank, or predictor-outcome result was read.

## Real Theta Evidence

- Sample row 0: last theta ego `1.177265243067`, last theta npc `-1.138023752490`.
- Sample row 1: last theta ego `0.323560689417`, last theta npc `-0.785398163397`.
- Trace file: `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/08_directional_ipv/real_theta_trace.csv`.

## Sign Contract

Positive real theta is interpreted as more prosocial (`sign=+1`); negative real theta is interpreted as more competitive (`sign=-1`). The real trace includes positive theta `1.178096277009` and negative theta `-1.138023752490` after the minimum observation gate.

## Directional Deviation Reconfirmation

Using Gate 0 frozen `w_min_rad=0.196349540849` and a canonical neutral envelope (`Q_low=0`, `Q_high=0`) for orientation-only tests:

- Canonical cut-in / competitive shortfall on real theta `-1.138023752490`: `D_comp=5.795907`, `D_yield=0.000000`.
- Canonical over-yield / prosocial excess on real theta `1.178096277009`: `D_comp=0.000000`, `D_yield=5.999995`.

This confirms `D_comp=max(0,(Q_low-theta_ego)/w)` fires below the lower envelope and `D_yield=max(0,(theta_ego-Q_high)/w)` fires above the upper envelope. These are orientation tests only, not confirmatory NSFC outcome analyses.
