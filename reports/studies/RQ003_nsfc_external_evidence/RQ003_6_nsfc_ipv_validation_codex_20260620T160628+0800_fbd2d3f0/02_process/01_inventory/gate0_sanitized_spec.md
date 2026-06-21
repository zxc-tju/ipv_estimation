# Gate 0 Outcome-Free Sanitized Measurement Spec

Source: `reports/studies/RQ003_nsfc_external_evidence/plans/RQ003_plan_v2_nsfc_ipv_validation_20260620.md` at SHA-256 `98900786a4819e58e91f16eb8da84724c928ca5a6d2c40b2252cf106b56625e1`. This sanitized version contains only measurement definitions and audit criteria needed by the Gate 0 measurement auditor. It intentionally omits prior exploratory outcomes, official score values, ranks, and predictor-outcome results.

## Scope

Gate 0 audits whether NSFC dynamic IPV measurements are comparable with the InterHub human conditional norm before any outcome association is attempted. The auditor must use this file and `gate0_outcome_denylist.txt`, not raw outcome tables.

## Dynamic Same-Mouth Measurement Contract

- NSFC and InterHub must use the same IPV estimator, the same rolling window definition, the same sampling-rate handling, and the same progress definition.
- Deployed NSFC measurements must be rolling-to-rolling: rolling IPV may not be compared against a full-window mean envelope as the primary deployed metric.
- Short, medium, and long window sensitivities may be audited, but the primary window must be frozen before any outcome access.
- Report rolling/full-window bias, convergence, and valid-frame proportion only as measurement-health diagnostics, not outcome associations.

## Sign Contract

- `theta > 0` means prosocial.
- Competitive shortfall and over-yielding excess are separate one-sided deviations.
- Role exchange, mirror transforms, and time truncation must not flip signs or introduce future leakage.

## Human Conditional Norm

The empirical verifier uses a human conditional norm. It is not self-anchored to the tested ego vehicle and it is not guarded by safety-policy floors for the external-validation measurement.

```text
m(t) = Q_0.5(theta_ego | theta_npc, s, tau)
w(t) = max((Q_high - Q_low) / 2, w_min)
D_comp(t) = max(0, (Q_low(theta_npc, s, tau) - theta_ego) / w(t))
D_yield(t) = max(0, (theta_ego - Q_high(theta_npc, s, tau)) / w(t))
```

Definitions:

- `s` is the state condition.
- `tau` is progress.
- `theta_npc` is the counterpart/NPC dynamic IPV condition.
- `theta_ego` is the tested ego dynamic IPV.
- `Q_low`, `Q_0.5`, `Q_high`, and `w_min` must be frozen before outcome access.

## Self-Anchor, Empirical Verifier, Safety Guard

- Self-anchor means an ego trailing/early history IPV computed only from information before the decision time. In the deployment verifier it can narrow intervals.
- External validation must not use the tested ego vehicle's self-anchor as the expectation value, because that would absorb between-team behavioral differences.
- The empirical verifier output is `D_comp` and `D_yield` from human data.
- The safety-policy guard is a separate high-risk conservative floor. Gate 0 first audits the empirical verifier without adding the guard.

## Conformal Boundary

- Conformal thresholds must be frozen on an InterHub calibration split.
- InterHub human trajectories and NSFC algorithm trajectories are not exchangeable; NSFC measurement may report empirical coverage, OOD status, and abstention, but must not claim nominal conformal coverage.
- NSFC official coordination, comprehensive score, ranks, or any other outcome field must not tune thresholds.
- The distribution of participating algorithms must not redefine normal behavior.

## Support and OOD Gate

Each frame must record:

- risk-proxy confidence,
- geometry/role source,
- progress confidence,
- InterHub cell support,
- estimator uncertainty,
- fallback level,
- abstention or monitor-only status.

Primary measurement traces may label only frames meeting the frozen support requirement as high-support. Low-support frames are monitor-only.

## Counterpart/NPC Conditioning

The auditor must verify dynamic IPV for the counterpart/NPC as well as ego. The main trace should support counterpart-conditioned human norms, simultaneous-competition rate, reciprocity mismatch, violation onset, persistence, and AUC summaries. These are measurement features only at Gate 0.

## Three Recomputable Result Views

Gate 0 must leave enough trace information to recompute:

- marginal view,
- conditional view,
- scalar view.

No view may use outcome values or outcome-tuned thresholds.

## Leakage Rules

Forbidden in online/deployed measurements:

- future frames,
- full-window IPV as the deployed value,
- observed PET as an online feature,
- realized order after the decision point,
- post-hoc phase labels,
- official scores/ranks/outcomes,
- outcome-tuned window, support, or threshold choices.

## Gate 0 Acceptance Criteria

Gate 0 passes only if:

- sign unit tests pass completely for competitive shortfall, over-yielding excess, role/mirror/time-truncation cases, and leakage guards;
- no future information enters online metrics;
- rolling-to-rolling measurement is implemented;
- the primary analysis frames meet the frozen high-support rule;
- marginal, conditional, and scalar views are reproducible;
- required outputs are produced: `ipv_measurement_audit.md`, `ipv_trace.csv`, and `unit_test_results.csv`.

Gate 0 must stop before any IPV-outcome association if any sign, leakage, support, or rolling-to-rolling blocker remains.
