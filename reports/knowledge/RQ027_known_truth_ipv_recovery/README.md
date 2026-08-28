# RQ027 Knowledge Layer

Status: `pilot executed / PILOT_NO_GO / no accepted claims`

Execution layer: `reports/studies/RQ027_known_truth_ipv_recovery/`

Plan: `reports/plans/RQ027_plan_v0_known_truth_ipv_recovery_20260828.md`

## Research Question

Can the frozen online IPV estimator recover a simulation-controlled IPV parameter when the trajectory generator does not share its planner, search, cost implementation or likelihood, and can candidate concentration distinguish lower-error interactive windows from uninformative or mismatched controls?

## Current Interpretation

The bounded `240 interactive + 48 negative-control` feasibility pilot completed with `PILOT_NO_GO`. Engineering execution was complete (`288/288` runs; `0/288` engineering failures), but recovery did not improve MAE over the zero predictor, concentration-error association reversed, and negative-control persistent false acceptance was high (`35/48`). Existing same-model WP2 results remain S0 engineering evidence only. No RQ027 claim is accepted yet; S2 and sealed confirmatory expansion are stopped.

## Evidence Boundary

- `可直接支撑`：the need for an independent known-truth pilot and the revised pilot contract.
- `可作旁证`：existing estimator interface/parity tests and sealed WP2 engineering health.
- `待核验`：independent recovery, concentration-error validity and negative-control false acceptance.
- `不能证明`：human latent preference, causal effects, external validity, production readiness or changes to accepted RQ017+ claims.

## Reviews

- `reviews/plan_rationality_review.md`
- `synthesis.md`
