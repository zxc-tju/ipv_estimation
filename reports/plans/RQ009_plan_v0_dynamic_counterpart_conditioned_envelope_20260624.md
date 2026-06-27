# RQ009 Plan v0 — Estimability-Aware Dynamic Counterpart-Conditioned Human Envelope

Status: `PI-authorized for launch; independent plan review required before analysis`  
Wave: B  
Work group: Group 3  
Date: 2026-06-24

## 1. Research question

> Among windows in which an interaction is active and the current IPV is sufficiently estimable,
can a calibrated human reference interval conditioned on causal context and the counterpart's
current IPV provide a reliable and selective test of conditional social atypicality?

The primary model is M3 (`context + counterpart current IPV`). Ego self-history is M4, an
ablation only, and must not define the group norm.

## 2. PI decisions governing this plan

- RQ009 is authorized to start.
- RQ008B will not be run at present; no RQ008 exploratory motif may enter the primary RQ009 model.
- Two-human OnSite annotation is deferred and is not an RQ009 dependency.
- OnSite is the first external-validation priority after RQ009 freezes its inference package.
- WOD-E2E access/pilot is authorized in parallel, but WOD engineering must not block RQ009 or the
  first OnSite analysis.
- The RQ007 sealed split remains unopened during model design. Opening it is a separate,
  irreversible decision after the full RQ009 protocol is frozen.

## 3. Evidence and input hierarchy

### Binding inputs

- `reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/decision.md`
- `reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/synthesis.md`
- RQ007 execution artifacts defining the opportunity mask, concentration-index estimability
  proxy, case split, and accepted C1–C3 boundaries.
- `data/derived/interhub/20260612_sigma_0_1_full_rerun/00_hpc_outputs/sigma01_ipv_timeseries.csv`
- causal InterHub metadata required for state, geometry, role, map/reference quality, and source.
- the current paper-side v4.1 method contract as a specification reference, not as evidence.

### Boundary input

RQ008A may be used only as a negative governance result: no positive temporal motif, lead–lag,
role-phase law, reciprocity, hysteresis, or discovery-derived temporal feature may enter the
primary model. Causal progress may be used only if defined independently of RQ008 discoveries.

### Denylist during model development

- RQ007 sealed/held-out outcomes before the sealed-opening gate;
- RQ008 confirmation data;
- WOD-E2E ratings;
- OnSite scores, ranks, collisions, event labels, or algorithm identities;
- concurrent target-window ego actions in the primary normative feature set;
- observed PET, realised passing order, closest-approach frame, post-hoc phase, full-window IPV,
  or any future-derived feature;
- manuscript headline preferences or desired external-result directions.

## 4. Target and analysis unit

The target is the ego **current rolling IPV** at time `t`, computed with the same estimator,
window, sampling convention, reference geometry, and causal information budget used at deployment.
The counterpart predictor is the counterpart current rolling IPV under the same convention.

Primary unit for fitting and inference is the case/scene, not the frame. Frames/windows may form
training rows, but:

- all rows from a case/scene remain in one partition;
- case weights are normalized so long interactions do not dominate;
- uncertainty intervals and resampling are clustered by case/scene;
- one calibration score per case per prespecified causal-progress anchor is used for the primary
  pointwise conformal estimand.

## 5. Estimability contract

RQ009 must consume the RQ007 contract without outcome-based retuning:

1. `NO_ACTIVE_INTERACTION` means not applicable, not neutral.
2. The concentration index is an estimability/identifiability proxy, not a standard deviation.
3. Most of the gross estimability contrast is proximity-driven; the small conflict-geometry
   residual must not be rewritten as a causal conflict effect.
4. Estimability does not mean behavioural settling; IPV may continue changing in accepted windows.
5. Episode-summary definitions are distinct from the current-IPV target.

The primary M3 interval is evaluated only when:

```text
active interaction opportunity
valid ego current-IPV estimate
valid counterpart current-IPV estimate
frozen ego/counterpart estimability gate
stable counterpart identity
sufficient map/role/progress quality
sufficient human-reference support
```

If the counterpart fails estimability, the guard phase must freeze one policy before test:

```text
ABSTAIN: COUNTERPART_IPV_UNESTIMABLE
or
M2 context-only degraded reference, still reported as UNVERIFIED / degraded
```

## 6. Causal feature contract

### Primary M3 may use

- road/path geometry and route relationship available by time `t`;
- priority/role variables available causally;
- causal interaction progress;
- relative position and relative velocity available before the scored target instant;
- causal TTC/APET-like risk proxies, but not observed PET;
- counterpart current rolling IPV, its audited uncertainty/concentration, and identity stability;
- map/reference confidence, valid-frame fraction, source/support indicators.

### Primary M3 must exclude

- ego self-anchor or early ego IPV;
- target-window concurrent ego acceleration, braking, jerk, or reward components that reconstruct
  the target IPV;
- observed PET, actual order, closest frame, final interaction duration, post-hoc phase;
- full-window ego or counterpart IPV;
- any variable selected because it improves held-out or external outcomes.

Concurrent ego kinematics may enter a prespecified sensitivity model only.

## 7. Model ladder

```text
M0  global current-IPV interval + conformal
M1  oracle PET/phase envelope, offline ceiling only
M2  causal context-only dynamic interval
M3  causal context + counterpart current IPV             PRIMARY
M4  causal context + ego self-history                    ABLATION ONLY
M5  source-aware / estimability-aware / OOD-gated M3
```

All capacity comparisons must use the same training cases, preprocessing, tuning budget,
calibration protocol, and test cases. M1 is not deployable and cannot be a headline verifier.

## 8. Data partitions and sealed-data governance

Use complete case/scene groups and create or reuse four disjoint roles:

```text
training
guard-tuning
conformal calibration
test
```

The RQ007 sealed split must not be opened during plan review, feature selection, model selection,
threshold selection, or code debugging.

Before any sealed data are read, create and independently review:

```text
analysis_freeze.yaml
feature_contract.yaml
estimability_gate_contract.yaml
split_manifest.csv
model_capacity_contract.md
conformal_protocol.yaml
primary_endpoints.md
success_failure_criteria.md
sealed_opening_manifest.json
```

Opening the sealed split requires a separate explicit PI authorization. Once opened, no gate,
window, feature, hyperparameter budget, model family, endpoint, or success threshold may be changed
without downgrading the changed analysis to exploratory.

## 9. Conformal calibration and coverage

Primary levels: P80, P90, and P95. The primary estimand is case-balanced pointwise marginal coverage
at prespecified causal-progress anchors among windows passing the identical frozen gate in
calibration and test.

Requirements:

- train fits the conditional quantiles;
- guard freezes support/OOD/estimability rules and any degraded-reference policy;
- calibration computes finite-sample conformal radii on accepted calibration cases only;
- test is evaluated once after freeze;
- lower/median/upper quantiles must be non-crossing through model constraints or a frozen
  rearrangement/isotonic rule;
- report accepted-window coverage and unconditional case-balanced performance together;
- do not claim conditional or source-shift nominal coverage.

Report at minimum:

```text
coverage and case-clustered CI
mean/median width
pinball loss
Winkler/interval score
active-interaction rate
ego estimability rate
pair estimability rate
human-support acceptance rate
total abstention rate and reason distribution
```

## 10. Primary comparison and success criteria

Primary scientific comparison: **M3 versus M2** on the frozen held-out test.

The independent plan reviewer must freeze numeric non-inferiority and meaningful-improvement
thresholds before any held-out opening. The intended logic is:

- M3 must not materially degrade P90 accepted-window coverage relative to M2;
- at comparable coverage, M3 must improve a prespecified sharpness/interval-score endpoint;
- the advantage must disappear or materially weaken under shuffled/cross-pair counterpart IPV;
- gains cannot be driven solely by source identification, case duration, or target-proximal ego
  kinematics.

Interpretation:

- M3 clearly beats M2: supports a dyadic counterpart-conditioned norm;
- M3 and M2 are equivalent: prefer the simpler M2 and reject counterpart necessity;
- only M4 is sharper: self-history predicts style but does not gain normative authority;
- all models fail calibration/support: report a negative/domain-boundary result.

## 11. Negative controls and sensitivity analyses

Required controls:

- shuffled counterpart IPV within source/state strata;
- random cross-pair counterpart IPV;
- wrong-counterpart identity;
- M2 without counterpart IPV;
- M4 self-history disagreement and norm-laundering stress cases;
- kinematics-only and concurrent-ego-kinematics sensitivity;
- source-label-only and source-shuffle controls;
- wrong-state / wrong-envelope-cell;
- future-leaky oracle/full-window result only as an explicitly labelled optimistic ceiling;
- gate-off and fallback-inclusive sensitivity, never as the primary result.

## 12. Abstention contract

Primary verdicts remain:

```text
WITHIN_NORM
COMPETITIVE_DEVIATION
OVER_YIELDING
ABSTAIN
```

Required abstention reason codes:

```text
NO_ACTIVE_INTERACTION
EGO_IPV_UNESTIMABLE
COUNTERPART_IPV_UNESTIMABLE
LOW_HUMAN_SUPPORT
OOD_SOURCE
MAP_OR_ROLE_UNCERTAIN
COUNTERPART_UNSTABLE
INVALID_INPUT
```

Planner fallback is a control action, not a verifier verdict. A degraded M2 interval cannot be
silently reported as `WITHIN_NORM` under the M3 claim.

## 13. Gates

### Gate 009-0 — Independent plan review

Freeze endpoints, thresholds, splits, feature contract, capacity matching, negative controls,
conformal protocol, and sealed-opening rules.

### Gate 009-1 — Measurement and same-window audit

Verify target/predictor estimator identity, rolling-window equality, causal inputs, estimability
mask, counterpart identity, and absence of future leakage.

### Gate 009-2 — Split and feature isolation

Verify case/scene grouping, no frame leakage, no RQ008 confirmation access, and no external-outcome
access.

### Gate 009-3 — Model/calibration readiness

All M0–M5 implementations, tests, non-crossing correction, conformal code, and abstention logic must
pass on training/guard/calibration data before sealed opening.

### Gate 009-4 — Falsification readiness

Negative controls and self-anchor norm-laundering probes must be fully specified and executable.

### Gate 009-5 — Sealed-opening authorization

Independent reviewer confirms the frozen package; PI explicitly authorizes the irreversible opening.
Until then, report `READY_FOR_SEALED_TEST`, not a held-out result.

### Gate 009-6 — Independent review, red team, and replication

A separate implementation must reproduce masks, quantiles, conformal radii, predictions, coverage,
and the primary M3–M2 comparison within frozen tolerances.

## 14. Deliverables

```text
analysis_freeze.yaml
feature_contract.yaml
estimability_gate_contract.yaml
split_manifest.csv
model_capacity_contract.md
conformal_protocol.yaml
primary_endpoints.md
success_failure_criteria.md
sealed_opening_manifest.json
M0_M5_metrics.csv
conditional_coverage.csv
abstention_coverage_curve.csv
negative_controls.csv
self_anchor_disagreement.csv
cv_predictions.parquet
frozen_inference_package/
prediction_interface.md
tried.md
evidence.csv
90_report/index.html
```

Every reader-facing figure must be created by a Codex worker using Codex's own Nature skill. Claude
must not draw figures. Save SVG, PDF, PNG, source-data CSV, metadata, and a figure manifest.

## 15. External handoff order

After RQ009 freezes and passes review:

1. **OnSite first:** hand the frozen inference package to RQ011B for matched
   `algorithm × scenario` analysis using `full_300` outcomes and `clean_285` replay/IPV cases, with
   the T19 replay-selection caveat.
2. **WOD in parallel:** continue the authorized signed-in, ratings-blind tracking pilot; formal
   preference validity waits for the tracking gate and frozen RQ009 interface.
3. RQ012 human annotation remains deferred; it is not required for RQ009 or the initial RQ011B
   analysis.

## 16. Non-goals

- Do not run RQ008B or use an RQ008 motif as a primary feature.
- Do not open the RQ007 sealed split without a new explicit authorization.
- Do not run WOD/OnSite outcome analyses inside RQ009.
- Do not claim socially inappropriate behaviour from distributional atypicality alone.
- Do not restore self-anchor as the normative target.
- Do not claim information absent from kinematics; use baseline-relative incremental utility.
- Do not edit the paper repository during the research execution.
