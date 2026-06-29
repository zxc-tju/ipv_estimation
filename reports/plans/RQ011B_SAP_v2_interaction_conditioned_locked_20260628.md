> Criterion superseded 2026-06-28 by RQ011B_SAP_v3_broad_interaction_failure_locked_20260628.md: broad interaction-failure composite + per-category + directional (collision-only v2 primary was under-powered at 19 events, found at the pre-model support check). Do NOT alter existing content.

# RQ011B Locked SAP v2 Interaction-Conditioned Parsimony Amendment

Status: **LOCKED — PI-approved 2026-06-28; amends SAP v1 locked 2026-06-25**  
RQ: `RQ011B`  
Run: `RQ011B_1_matched_scenario_20260625T202454_8331bd49`  
Worker: `RQ011B-P3-prereg-lock`  
Role: registrar / pre-registration lock  
Base SAP: `reports/plans/RQ011B_SAP_v1_locked_20260625.md`  
Source packet: `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/04_criterion_consequence/prereg_v4_draft/`  
Re-review packet: `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/04_criterion_consequence/prereg_v4_rereview/`

This locked amendment is the binding interaction-conditioned criterion-consequence
pre-registration for RQ011B. It supersedes the single-anchor criterion sections
of SAP v1, while leaving the SAP v1 provenance, denylist, scope boundary, and
non-outcome firewall in force unless explicitly amended here.

No outcome-consuming criterion execution, scorer run, model fit, plot, registry
update, knowledge decision edit, or paper-repository edit is authorized by this
lock. The required next step is an independent clean-room recompute ledger; W1
is the first outcome-touching step.

## 1. Locked Thesis And Claim Boundary

The v4 PI-accepted thesis is parsimony/sufficiency:

```text
IPV norm-deviation is a parsimonious, socially meaningful monitor.
One standardized IPV-derived indicator should explain or flag official
negative interaction harm approximately as well as the full scheme-matched
physical-parameter ensemble, should depend on real IPV/cell/counterpart
structure, and should generalize across held-out scenarios.
```

The thesis is not incremental-over-physics. IPV is derived from interaction
physics, so the confirmatory question is whether one interpretable IPV-derived
indicator captures a predeclared fraction of the full physics ensemble signal.

Allowed positive interpretation is limited to an interpretable official-harm
monitor at the locked SAP unit, `algorithm_id x scenario_id`. This amendment
does not support causal, planner-benefit, human-normative, independence-from-
physics, or closing-rate-mediation-ruled-out claims.

## 2. Unit, Firewall, And Outcome Boundary

Unit of analysis remains one replay-clean `algorithm_id x scenario_id` cell,
with audited `algorithm_id == team_id` and `case_id == scenario_id`.

All thresholds, schemes, endpoint hierarchy, support labels, model families,
comparison statistics, bootstrap/permutation rules, and decision labels in this
amendment are locked before W1. No endpoint values, endpoint prevalence, model
results, or IPV-outcome associations may be used to choose support rules,
thresholds, transformations, weights, model variants, endpoint pairings, or
exclusions.

The locked SAP denylist carries forward. Banned from scheme conditioning,
feature construction, support logic, imputation, thresholds, weights, controls
except where explicitly named as a falsification transform, and model selection:
observed PET; realized passing order; post-hoc phase labels; closest-approach
or critical-approach frame; realized future minimum distance; full-window IPV;
overlapping, full-window, self-anchor, or post-hoc target IPV construction;
ego early/self-anchor IPV in the M3 primary model; target-proximal ego behavior
in the same window as the scored rolling IPV target; estimator-internal reward
components; `log_opportunity_duration_z` or realized opportunity duration; and
any official score, rank, collision, deduction, success/failure, mission status,
or other outcome-derived variable.

## 3. Endpoint Hierarchy

### 3.1 Primary Endpoint

The confirmatory primary endpoint is official-harm-only:

```text
official_harm_flag =
  collision_flag_score0
  OR collision_pdf_deduction > 0
  OR safety_intervention > 0
```

`collision_flag_score0` is the official score-0 collision recode from the RQ011
official-field provenance. `collision_pdf_deduction` and
`safety_intervention` are parsed diagnostic-PDF official fields or their
audited equivalents.

No RQ012B E02 high-deceleration or E09 near-miss automatic event enters the
primary endpoint.

Expected direction: higher `m3_norm_deviation_S` means greater deviation from
the human M3 envelope and should predict higher probability of
`official_harm_flag == 1`.

Primary endpoint support rule:

- If events and non-events are both `>= 20`, use the locked binary model.
- If events are 10-19 and non-events are `>= 20`, use M3-only exact or blocked
  permutation inference for descriptive reporting only; no positive v2 primary
  claim is allowed.
- If events are `< 10` or non-events are `< 20`, label the primary endpoint
  `UNDER_IDENTIFIED`.

### 3.2 Ordered Secondaries

Secondaries are hierarchical and cannot rescue a failed primary Scheme A
decision.

1. `official_comprehensive`: official per-scenario comprehensive score,
   continuous secondary only. Expected direction is negative.
2. `interaction_tied_deduction_burden`:
   `log1p(collision_pdf_deduction + safety_intervention)`, if numeric
   deduction magnitudes are available under the official-field audit. Expected
   direction is positive.
3. `E02_high_deceleration_rate`: RQ012B `E02_primary_count`, modeled as
   count/rate with exposure/offset only where an allowed scheme pairing exists.
   Expected direction is positive.
4. `E09_near_miss_rate`: RQ012B `E09_primary_count`, modeled as count/rate
   with exposure/offset only for descriptive summaries because v2 hard rules
   conflict with E09 use for all confirmatory schemes. Expected direction is
   positive.
5. `E02_E09_conflict_burden`: count or `log1p(E02_primary_count +
   E09_primary_count)`, descriptive only because every scheme pairing contains
   at least one circular component. Expected direction is positive.

## 4. Schemes A-E

All selected anchors must be support-valid under the frozen expanded M3 anchor
grid. Missing selected-anchor targets, missing intervals, failed
support/estimability, or failed scheme conditioning are not imputed. Anchor
weights are equal within every scheme.

| Scheme | Role | Locked default | Support status |
| --- | --- | --- | --- |
| A | Primary | Closing anchors: `closing_rate_anchor > 0`; `K=5` | PASS: 247 cells, 19 teams, 15 scenarios, min 12 cells/scenario, 31,724 anchors |
| B | Robustness | Proximity anchors: `relative_distance_anchor <= 10.0 m`; `K=5` | UNDER_IDENTIFIED: 98 cells |
| C | Robustness | Causal E02/E09 active anchors; `K=1` | UNDER_IDENTIFIED: 165 cells |
| D | Robustness | Closing plus counterpart deceleration: `closing_rate_anchor > 0` and `counterpart_accel_anchor_mps2 <= -0.5`; `K=3` | PASS: 234 cells, 19 teams, 15 scenarios, min 11 cells/scenario, 7,440 anchors |
| E | Robustness | One peak causal interaction anchor; `K=1` | PASS: 266 cells, 19 teams, 15 scenarios, min 16 cells/scenario, 266 anchors |

Scheme A is the only confirmatory primary scheme. Schemes B-E must be reported
in full but cannot rescue Scheme A. At lock time, B and C are under-identified
and cannot support robustness. "Robust across operationalizations" is allowed
only if Scheme A passes and at least two support-passing robustness schemes
among B-E are constructible, directionally coherent, consistent on the locked
parsimony metrics, and evaluated only on allowed scheme x endpoint pairings.

### 4.1 Scheme Definitions

```text
selected_A(a) = support_valid(a) and closing_rate_anchor(a) > 0
K_A = 5
m3_norm_deviation_A(cell) = mean(exceedance_a over selected_A anchors)
```

```text
selected_B(a) = support_valid(a) and relative_distance_anchor(a) <= 10.0
K_B = 5
m3_norm_deviation_B(cell) = mean(exceedance_a over selected_B anchors)
```

Scheme C uses causal current/past active-state rules:

```text
counterpart_speed_t = sqrt(counterpart_vx_t^2 + counterpart_vy_t^2)
ego_speed_t         = sqrt(ego_vx_t^2 + ego_vy_t^2)
counterpart_accel_t = (counterpart_speed_t - counterpart_speed_{t-1}) / dt
ego_accel_t         = (ego_speed_t - ego_speed_{t-1}) / dt

E02_active_at_anchor =
  at least 3 consecutive current/past rows through the anchor have
  ego_accel <= -3.4 or counterpart_accel <= -3.4

ttc_t = distance_m_t / closing_rate_mps_t  if closing_rate_mps_t > 0
        infinity                           otherwise

E09_active_at_anchor =
  at least 2 consecutive current/past rows through the anchor have
  ttc_t <= 1.5

selected_C(a) =
  support_valid(a) and (E02_active_at_anchor(a) or E09_active_at_anchor(a))
K_C = 1
m3_norm_deviation_C(cell) = mean(exceedance_a over selected_C anchors)
```

Rows with missing or nonpositive `dt` have unavailable acceleration and break
the causal run. Scheme C ledgers must attest that observed PET, realized passing
order, closest/critical approach frame, realized future minimum distance, and
outcome fields were absent from construction.

```text
selected_D(a) =
  support_valid(a)
  and closing_rate_anchor(a) > 0
  and counterpart_accel_anchor_mps2(a) <= -0.5
K_D = 3
m3_norm_deviation_D(cell) = mean(exceedance_a over selected_D anchors)
```

For Scheme E:

```text
candidate_E(a) =
  support_valid(a)
  and (
    closing_rate_anchor(a) > 0
    or relative_distance_anchor(a) <= 10.0
    or counterpart_accel_anchor_mps2(a) <= -0.5
  )
```

Select exactly one peak anchor per cell by this deterministic order:

1. Maximize `closing_pressure_anchor =
   max(closing_rate_anchor, 0) / max(relative_distance_anchor, 0.1)`.
2. Maximize `decel_pressure_anchor =
   max(-counterpart_accel_anchor_mps2, 0)`.
3. Minimize `anchor_timestamp`.
4. Minimize lexicographic `str(counterpart_key_agent)`.

```text
K_E = 1
m3_norm_deviation_E(cell) = exceedance of the selected peak anchor
```

Scheme E is a noisy one-anchor sensitivity and cannot by itself justify a
robustness claim.

## 5. M3 Deviation And Gate 0 Support

For every selected anchor `a`, using the frozen RQ009 M3 interval:

```text
scale_a = max((M3_upper_a - M3_lower_a) / 2, deviation_scale_floor)

exceedance_a =
  0                                      if target_ipv_a in [M3_lower_a, M3_upper_a]
  (target_ipv_a - M3_upper_a) / scale_a  if target_ipv_a > M3_upper_a
  (M3_lower_a - target_ipv_a) / scale_a  if target_ipv_a < M3_lower_a
```

The deviation scale floor remains the locked SAP conditional floor: `0.05`
radians only if RQ009 documents target-IPV noise/scale support. If that
documentation is absent, the phase pauses before prediction consumption.

Gate 0 is the support and estimability gate:

- The locked v2 support manifest classification is binding for A-E at lock
  time.
- A confirmatory scheme must retain all 15 scenarios, at least 16 teams, and
  at least 10 cells per scenario after scheme construction and required model
  fields.
- Required scheme deviation and physics components must be finite on the model
  row set.
- Any zero or nonfinite standard deviation in `m3_norm_deviation_S` or a
  required physics component makes the affected scheme/model
  `UNDER_IDENTIFIED_OR_DEGENERATE`; components are not dropped.
- Missing selected-anchor targets, missing intervals, failed support,
  non-estimability, or failed scheme conditioning are not imputed.

## 6. Scheme x Endpoint Disjointness Rule

A scheme's conditioning events, markers, or deterministic implications must be
disjoint from the endpoint used for that scheme. Any non-disjoint scheme x
endpoint pairing is forbidden for Gate 1, Gate 2, Gate 3, robustness, or
support language. Forbidden pairings may be shown only as descriptive-only
context and must be excluded from confirmatory or robustness claims.

| Scheme | Official primary `official_harm_flag` | Official secondaries | E02 secondary | E09 secondary | E02+E09 burden |
| --- | --- | --- | --- | --- | --- |
| A closing | Allowed | Allowed | Allowed | Descriptive-only: E09 is TTC/closing-implied | Descriptive-only: contains E09 |
| B distance | Allowed | Allowed | Allowed | Descriptive-only: E09 uses distance/TTC proximity | Descriptive-only: contains E09 |
| C E02/E09 active | Allowed | Allowed | Descriptive-only: direct E02 conditioning | Descriptive-only: direct E09 conditioning | Descriptive-only: direct E02/E09 conditioning |
| D closing-plus-deceleration | Allowed | Allowed | Descriptive-only: deceleration conditioning overlaps E02 | Descriptive-only: hard rule forbids D with E02/E09 automatic endpoints | Descriptive-only: contains E02/E09 |
| E peak interaction | Allowed | Allowed | Descriptive-only: peak rule can use deceleration | Descriptive-only: peak rule can use closing/distance | Descriptive-only: contains E02/E09 |

Primary `official_harm_flag`, official score, and official deduction
secondaries are allowed for schemes A-E. Automatic-event endpoints are
descriptive-only wherever scheme conditioning overlaps. In particular, C x
E02/E09, D x E02/E09, A/B x E09, and all E02+E09 burden pairings are
descriptive-only.

## 7. Full Physics Ensemble

For Gate 1, `physics_full_S` uses closed component lists as separate
standardized covariates. No component can be added, dropped, reweighted, or
replaced after this PI lock.

Primitive variables:

```text
d_a                 = max(relative_distance_anchor_a, 0.1)
log_distance_a      = log(d_a)
neg_log_distance_a  = -log_distance_a
ego_speed_a         = sqrt(ego_vx_anchor_a^2 + ego_vy_anchor_a^2)
counterpart_speed_a = sqrt(counterpart_vx_anchor_a^2 + counterpart_vy_anchor_a^2)
speed_pressure_a    = (ego_speed_a + counterpart_speed_a) / 2
relative_speed_a    = relative_speed_anchor_a
positive_closing_a  = max(closing_rate_anchor_a, 0)
closing_pressure_a  = positive_closing_a / d_a
ttc_a               = d_a / positive_closing_a if positive_closing_a > 0 else infinity
ttc_risk_a          = max(0, (1.5 - min(ttc_a, 1.5)) / 1.5)
counterpart_decel_pressure_a = max(-counterpart_accel_anchor_mps2_a, 0)
ego_decel_pressure_a         = max(-ego_accel_anchor_mps2_a, 0)
pair_decel_pressure_a        = max(counterpart_decel_pressure_a, ego_decel_pressure_a)
E02_flag_a = 1 if E02_active_at_anchor_a else 0
E09_flag_a = 1 if E09_active_at_anchor_a else 0
E_closing_flag_a  = 1 if closing_rate_anchor_a > 0 else 0
E_distance_flag_a = 1 if relative_distance_anchor_a <= 10.0 else 0
E_decel_flag_a    = 1 if counterpart_accel_anchor_mps2_a <= -0.5 else 0
```

Cell-level component lists:

```text
physics_full_A and physics_full_B:
  mean(speed_pressure)
  mean(relative_speed)
  mean(positive_closing)
  mean(neg_log_distance)
  mean(closing_pressure)
  mean(ttc_risk)

physics_full_C:
  mean(speed_pressure)
  mean(relative_speed)
  mean(positive_closing)
  mean(neg_log_distance)
  mean(closing_pressure)
  mean(ttc_risk)
  mean(pair_decel_pressure)
  mean(E02_flag)
  mean(E09_flag)

physics_full_D:
  mean(speed_pressure)
  mean(relative_speed)
  mean(positive_closing)
  mean(neg_log_distance)
  mean(closing_pressure)
  mean(ttc_risk)
  mean(counterpart_decel_pressure)

physics_full_E:
  selected-anchor speed_pressure
  selected-anchor relative_speed
  selected-anchor positive_closing
  selected-anchor neg_log_distance
  selected-anchor closing_pressure
  selected-anchor ttc_risk
  selected-anchor counterpart_decel_pressure
  selected-anchor E_closing_flag
  selected-anchor E_distance_flag
  selected-anchor E_decel_flag
```

For each scheme-specific fit, compute raw cell-level components, learn means
and sample standard deviations on the fit row set, and standardize both the
physics components and `m3_norm_deviation_S` on that same row set. In LOSO,
scaling is learned on the 14 training scenarios and applied unchanged to the
held-out scenario.

## 8. Gate 1: Parsimony/Sufficiency

For each allowed scheme `S` and endpoint:

```text
null:
  endpoint = scenario fixed effects

ipv_only_S:
  endpoint = scenario fixed effects
             + standardized m3_norm_deviation_S

physics_full_S:
  endpoint = scenario fixed effects
             + all standardized physics_full_S components
```

If scenario fixed effects are not estimable under the endpoint family, the
predeclared fallback is a scenario random intercept for all three models in the
same scheme x endpoint comparison. Mixing FE and RE families across `null`,
`ipv_only_S`, and `physics_full_S` is forbidden.

### 8.1 Binary Primary Model

The primary binary endpoint uses logistic GLM with scenario fixed effects:

```text
logit Pr(official_harm_flag_i = 1) =
  alpha_{scenario[i]} + beta * z_m3_norm_deviation_{S,i}
```

for `ipv_only_S`, and the analogous model with all standardized
`physics_full_S` covariates for `physics_full_S`.

If fixed-effect logistic fitting has complete or quasi separation,
nonconvergence, singular Hessian, infinite/nonfinite coefficients, or a
scenario-FE stratum that makes finite likelihood estimates unavailable, use a
logistic random-intercept model:

```text
logit Pr(official_harm_flag_i = 1) =
  alpha + u_{scenario[i]} + beta * z_m3_norm_deviation_{S,i}

u_s ~ Normal(0, sigma_s^2)
```

The same random-intercept family is then used for `null`, `ipv_only_S`, and
`physics_full_S`. If the random-intercept model also separates, fails to
converge, has nonfinite fitted probabilities, or has non-estimable variance
components, the scheme x endpoint comparison is
`UNDER_IDENTIFIED_OR_DEGENERATE`. No Firth, ridge, lasso, dropped-scenario, or
post-outcome penalized-likelihood substitute may be introduced after PI
acceptance.

The single decision pseudo-R2 for the binary primary is Tjur's coefficient of
discrimination:

```text
R2_Tjur(model) =
  mean(fitted_probability_i | official_harm_flag_i = 1)
  - mean(fitted_probability_i | official_harm_flag_i = 0)
```

Tjur pseudo-R2 is computed for `null`, `ipv_only_S`, and `physics_full_S` on
the identical row set. McFadden pseudo-R2, Brier score, log-loss, and AUC may
be reported as transparency metrics only.

```text
DeltaR2_ipv_S =
  R2_Tjur(ipv_only_S) - R2_Tjur(null)

DeltaR2_physics_S =
  R2_Tjur(physics_full_S) - R2_Tjur(null)

eff_S =
  DeltaR2_ipv_S / DeltaR2_physics_S
```

### 8.2 Continuous And Count Secondaries

For continuous official secondaries, use OLS with scenario fixed effects on the
identical row set. The decision analogue is OLS partial-R2 over the
scenario/null model:

```text
DeltaR2_ipv_S     = 1 - SSE(ipv_only_S) / SSE(null)
DeltaR2_physics_S = 1 - SSE(physics_full_S) / SSE(null)
eff_S             = DeltaR2_ipv_S / DeltaR2_physics_S
```

If `SSE(null) <= 0`, any model rank is deficient, a component standard
deviation is zero/nonfinite, or a fitted value/residual is nonfinite, the
continuous comparison is `UNDER_IDENTIFIED_OR_DEGENERATE`. Negative partial-R2
values are not clamped.

Allowed count/rate secondaries use Poisson GLM with a predeclared exposure
offset. If overdispersion handling is needed, it must be specified before
outcome readout and cannot be chosen from model results. Count/rate secondaries
cannot rescue the primary decision.

### 8.3 Locked Numeric Defaults And Ratio Rules

```text
tau_physics = 0.03
tau_ratio   = 0.60
```

`tau_ratio=0.60` is the outcome-blind sufficiency tolerance: one socially
meaningful IPV-derived indicator must recover at least 60% of the signal
captured by the full predeclared physics ensemble, conditional on the physics
ensemble itself having a meaningful endpoint relation.

`eff_S` is uncapped. Values above 1 are allowed. Negative numerators and
negative ratios are reported but cannot pass. If the point-estimate denominator
`DeltaR2_physics_S` is negative, zero, nonfinite, or `< tau_physics`, the
comparison is `UNDER_IDENTIFIED_MOOT_PHYSICS` and no point-estimate `eff_S`
ratio is reported. There is no clamping, winsorization, pseudo-R2 replacement,
physics-feature dropping, or endpoint switching to repair denominator failure.

### 8.4 Bootstrap

All Gate 1 confidence intervals use `B=10000` bootstrap replicates and seed
`20260625`. The same replicate multipliers are reused jointly for `null`,
`ipv_only_S`, `physics_full_S`, `union_S`, paired differences, and ratio
estimates.

For observation `i` with team `t(i)` and scenario `s(i)`, draw independent
Rademacher multipliers:

```text
a_t in {-1, +1} for each team
b_s in {-1, +1} for each scenario

omega_i = (a_{t(i)} + b_{s(i)} - a_{t(i)} * b_{s(i)}) / sqrt(3)
```

This is the locked two-way team x scenario blocked wild multiplier. For every
reported scalar other than ratios, use percentile 95% CIs over valid
replicates. If fewer than 80% of `B` replicates are valid for a required Gate 1
scalar, the scalar is `UNDER_IDENTIFIED_BOOTSTRAP` and no positive claim is
allowed.

For ratios, compute `DeltaR2_ipv_b` and `DeltaR2_physics_b` jointly in each
replicate. Replicate denominators that are `< tau_physics`, `<= 0`, or
nonfinite are invalid. Valid replicate ratios are not clamped or winsorized;
negative ratios and ratios above 1 are retained. The 95% ratio CI is the
percentile interval over valid replicate ratios. If fewer than 80% of `B`
ratio replicates are valid, the ratio CI is `UNDER_IDENTIFIED_RATIO_CI` and no
positive Gate 1 claim is allowed.

### 8.5 Gate 1 Pass Rule

Gate 1 passes for scheme `S` only if all are true:

1. The scheme x endpoint pairing is allowed.
2. Gate 0 support passes.
3. Point-estimate `DeltaR2_physics_S >= tau_physics` and the 95% bootstrap CI
   for `DeltaR2_physics_S` has lower bound `> 0`; otherwise the comparison is
   `UNDER_IDENTIFIED_MOOT_PHYSICS`.
4. `DeltaR2_ipv_S` has 95% bootstrap CI lower bound `> 0`.
5. The standardized `m3_norm_deviation_S` coefficient has the expected social
   direction with 95% bootstrap CI excluding zero on the expected side:
   positive for binary/count harm endpoints and negative for official-score
   continuous secondaries.
6. `eff_S >= tau_ratio` by point estimate. The 95% bootstrap CI for `eff_S` is
   mandatory reporting; a wide interval or lower bound below `tau_ratio` must
   be described as imprecise, but v2 does not require the 95% CI lower bound
   for `eff_S` to exceed `tau_ratio`.

Required transparency reports, not gates: `R2(null)`, `R2(ipv_only_S)`,
`R2(physics_full_S)`, `R2(union_S)`, number of physics features versus IPV's
one feature, and the demoted old incremental-over-kinematics diagnostic.

## 9. Gate 2: Structure-Preserving Specificity

Gate 2 tests whether the parsimony signal depends on real IPV, cell,
counterpart, sign, and envelope structure. It does not require IPV to be
independent of the physics from which it is derived.

For every required control `c`, fit on the common row set and only for allowed
scheme x endpoint pairings:

```text
true_ipv_model:
  endpoint = scenario fixed effects
             + standardized true m3_norm_deviation_S

control_model_c:
  endpoint = scenario fixed effects
             + standardized control_deviation_{S,c}
```

Required controls:

1. `shuffled_ipv`: block-permute true scheme-specific M3 deviation within exact
   `scenario_id` and outcome-blind `team_difficulty_tertile`, deranged within
   stratum.
2. `counterpart_swap`: keep ego trajectory, keys, time grid, endpoint, and
   scenario structure fixed; replace the target counterpart with the nearest
   eligible non-target actor in the same cell, tie-broken lexicographically;
   recompute support/deviation through the frozen RQ009 interface.
3. `wrong-cell/wrong-envelope`: assign another eligible cell's frozen M3
   envelope/deviation while preserving the marginal scheme deviation
   distribution; default derangement is within exact `scenario_id`.
4. `sign_flip`: reverse signed IPV quantities before exceedance construction;
   preserve magnitudes, keys, time grid, support flags, and endpoint rows.
5. `IPV_removed`: use the frozen RQ009 M2/context-only envelope that excludes
   counterpart current IPV; same selected anchors, same K, same target
   construction, same scale floor, same equal weights.

`kinematics_only` is reported-only and is not a Gate 2 beat-gate. `role_flip`
may be reported as an additional diagnostic if constructible, but it is not
required.

Locked Gate 2 margins:

```text
delta_beta = 0.05
delta_R2   = 0.01
```

For deterministic controls, true IPV must beat each required control by at
least `delta_beta` in harm-direction coefficient advantage with 95% paired
two-way blocked bootstrap CI lower bound `> 0`, and by at least `delta_R2` in
`DeltaR2_true - DeltaR2_control` with 95% paired bootstrap CI lower bound
`> 0`.

For stochastic controls, use max-statistic permutation with `B_perm=10000`,
base seed `20260625`, and require max-statistic permutation `p < 0.05`.

If a required control is unavailable for more than 10% of otherwise eligible
primary rows, loses all 15-scenario / 16-team / 10-cells-per-scenario support,
or is key-incompatible/non-outcome-blind, no positive primary claim is allowed.

## 10. Gate 3: LOSO Generalization

Gate 3 uses leave-one-scenario-out transfer. For each held-out scenario, train
on the other 14 scenarios. Z-scaling is learned on the 14 training scenarios
only and applied unchanged to the held-out scenario. The held-out scenario
receives no fitted scenario intercept or scenario fixed effect.

Transfer models:

```text
null_transfer:
  endpoint = intercept

ipv_transfer_S:
  endpoint = intercept
             + standardized m3_norm_deviation_S

physics_transfer_S:
  endpoint = intercept
             + all standardized physics_full_S components
```

For continuous endpoints use held-out SSE skill against the held-out null. For
the binary primary use held-out log-loss skill against the held-out intercept
prediction. Define aggregate held-out skill:

```text
DeltaQ2_ipv_all     = aggregate held-out skill for ipv_transfer_S
DeltaQ2_physics_all = aggregate held-out skill for physics_transfer_S
eff_LOSO_S          = DeltaQ2_ipv_all / DeltaQ2_physics_all
```

The same denominator rules as Gate 1 apply. If `DeltaQ2_physics_all <
tau_physics`, the LOSO ratio is `UNDER_IDENTIFIED_MOOT_PHYSICS` and no
`eff_LOSO_S` ratio is reported.

Gate 3 passes for Scheme A only if all are true:

- training-fold IPV coefficient is in the expected direction in at least 12 of
  15 folds;
- held-out IPV skill is positive in at least 12 of 15 folds;
- aggregate `DeltaQ2_ipv_all > 0` with two-way blocked bootstrap CI lower bound
  above 0;
- LOSO physics precondition is met: `DeltaQ2_physics_all >= tau_physics` and
  its CI lower bound is `> 0`;
- `eff_LOSO_S >= tau_ratio` by point estimate, with CI reported;
- no single held-out scenario contributes more than 40% of aggregate positive
  IPV skill;
- removing the largest positive held-out scenario leaves positive aggregate
  IPV skill.

If more than three folds are invalid because of all-event/all-none training,
complete or quasi separation, nonconvergence, nonfinite coefficients,
zero-variance predictors, rank deficiency, or nonfinite held-out predictions,
Gate 3 is `UNDER_IDENTIFIED_OR_DEGENERATE`. Failure of the valid LOSO criteria
gives `NON_GENERALIZING`.

## 11. Decision Labels

Report all schemes A-E, regardless of direction, support, endpoint pairing, or
control outcome. Scheme A against `official_harm_flag` controls the primary
decision. B-E are robustness/sensitivity only. Secondary endpoints cannot
upgrade a failed primary decision.

- `SUPPORTED_PARSIMONIOUS_MONITOR`: Scheme A passes Gate 0 support, Gate 1
  parsimony/sufficiency, Gate 2 structure-preserving specificity, Gate 3 LOSO
  generalization, and the expected-direction interpretability contract for the
  official-harm-only primary endpoint.
- `NON_PARSIMONIOUS`: Scheme A support/precondition is adequate but `eff_A <
  tau_ratio`, or IPV-only increment is not positive/directional, while
  `physics_full` has meaningful criterion signal.
- `NON_SPECIFIC`: Scheme A Gate 1 passes but one or more structure-preserving
  specificity controls is not beaten by the locked margins.
- `NON_GENERALIZING`: Scheme A Gates 1-2 pass but LOSO transfer fails.
- `UNDER_IDENTIFIED`: scheme support, primary endpoint sparsity, physics
  precondition, component degeneracy, missingness, required-control
  availability, model separation/nonconvergence, bootstrap invalidity, or
  forbidden endpoint pairing prevents a valid decision.

## 12. Interpretation Ceiling

A supported v2 claim may say only:

```text
Cells whose behavior deviates more from the frozen human IPV envelope are more
likely to show official harm, and the single IPV deviation indicator captures a
predeclared fraction of the full physical-parameter ensemble's criterion
signal.
```

Required interpretability outputs: coefficient sign and uncertainty; predicted
probability/rate/score contrast per +1 SD of `m3_norm_deviation_S`; Gate 1
efficiency ratio; Gate 2 specificity table; LOSO fold table; decomposition of
`official_harm_flag` into score-0 collision, collision deduction, and safety
intervention components; and descriptive-only E02/E09 summaries separated from
the official primary.

Forbidden interpretations: "IPV predicts harm beyond physics" as the primary
thesis; causal harm language; planner or safety-benefit language; human norm
authority beyond the frozen empirical envelope; counterpart-IPV adds unique
information unless `IPV_removed` is beaten by the locked margins; "not
closing-rate-in-disguise"; or "rules out closing-rate mediation."

## 13. Required Clean-Room Recompute Before W1

Before any outcome-consuming criterion execution, an independent clean-room
worker must recompute from frozen inputs:

- scheme membership A-E;
- selected-anchor counts and K eligibility;
- allowed/descriptive-only scheme x endpoint pairing flags;
- M3 exceedance summaries;
- `physics_full_S` component summaries;
- `kinematics_only_S` compressed index for reported-only legacy diagnostics;
- all required control deviations;
- endpoint-definition joins and component flags.

The clean-room ledger must attest that no outcome values or IPV-outcome
associations were used to choose support rules, thresholds, transformations,
weights, model variants, endpoint pairings, or exclusions. The ledger must
preserve hashes/paths for the M3 and M2 prediction packages, RQ012B
event-definition inputs, and RQ011 official-field provenance inputs.

## Acceptance Record

PI acceptance date: **2026-06-28**.  
Accepted configuration: preregistration v4 with the defaults locked in this
amendment.

Audit chain:

1. SAP v1 lock and review: `reports/plans/RQ011B_SAP_v1_locked_20260625.md`;
   source/review packets under `02_process/01_plan_review/`.
2. Multi-scheme prereg v1 -> independent review -> v2:
   `02_process/04_criterion_consequence/prereg_multischeme_draft/`,
   `prereg_multischeme_review/`, and `prereg_multischeme_draft_v2/`.
3. PI reframed the criterion question from incremental-over-physics to
   parsimony/sufficiency.
4. Parsimony v3 -> v3 review:
   `02_process/04_criterion_consequence/prereg_v3_parsimony_draft/` and
   `prereg_v3_review/`.
5. v4 draft:
   `02_process/04_criterion_consequence/prereg_v4_draft/prereg_v4.md`,
   `scheme_endpoint_pairings.csv`, `changelog_v3_to_v4.csv`, and
   `pi_decision_brief_v4.md`.
6. v4 independent re-review PASS:
   `02_process/04_criterion_consequence/prereg_v4_rereview/prereg_v4_rereview.md`
   and `pi_acceptance_brief.md`.
7. PI accepted v4 on 2026-06-28 with the defaults embedded above.

Diagnostic chain headline recorded for run state: the single-anchor 95.5% zero
result was a sampling artifact; interaction-conditioned signal recovered; the
criterion was reframed as parsimony/sufficiency; `|IPV|`-closing correlation
`r=0.535` will be handled by the structure-preserving controls.
