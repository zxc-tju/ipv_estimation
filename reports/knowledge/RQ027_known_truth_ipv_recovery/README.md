# RQ027 Knowledge Layer

Status: `PI-ACCEPTED / CLOSED_BY_PI_SCOPE_DECISION`

Decision: `reports/knowledge/RQ027_known_truth_ipv_recovery/decision.md`

Execution layer: `reports/studies/RQ027_known_truth_ipv_recovery/`

Plan: `reports/plans/RQ027_plan_v0_known_truth_ipv_recovery_20260828.md`

## Research question

Can the frozen online IPV estimator recover a simulation-controlled IPV parameter when the trajectory generator does not share its planner, search, cost implementation or likelihood, and can candidate concentration distinguish lower-error interactive windows from uninformative or mismatched controls?

## Final interpretation

The bounded independent-generator pilot remains a `PILOT_NO_GO` for cross-model numerical recovery and concentration-based accuracy selection. The PI has closed this RQ and chosen not to pursue further recovery experiments for the current manuscript. The paper will cite the previously published T-ITS validation of IPV identification in controlled VGIM interactions and will use the estimator as a fixed input to the downstream human-reference monitor.

RQ027 is preserved as a research boundary. It does not establish universal IPV failure and does not revise accepted downstream monitor decisions.

## Reviews

- `reviews/plan_rationality_review.md`
- `synthesis.md`
