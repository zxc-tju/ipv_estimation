Superseded 2026-06-29 by `RQ011B_SAP_v4_moment_monitor_locked_20260629.md`: the analysis moves from cell-level criterion to MOMENT/EVENT-level monitor validation (cell-level binary was ill-posed: collision too sparse, any-failure saturated 285/285). Do NOT alter existing content.

# RQ011B Locked SAP v3 Broad Interaction-Failure Amendment

Status: **LOCKED - PI-approved 2026-06-28; supersedes the criterion of SAP v2**  
RQ: `RQ011B`  
Run: `RQ011B_1_matched_scenario_20260625T202454_8331bd49`  
Worker: `RQ011B-P4-relock-v5_1`  
Role: registrar / pre-registration re-lock  
Prior criterion superseded: `reports/plans/RQ011B_SAP_v2_interaction_conditioned_locked_20260628.md`  
Accepted source packet: `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/04_criterion_consequence/prereg_v5_1_draft/`  
Independent re-review packet: `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/04_criterion_consequence/prereg_v5_1_rereview/`

This locked amendment is the binding RQ011B broad-interaction-failure,
per-category, and directional pre-registration. It supersedes only the
criterion/endpoint layer of SAP v2, whose collision-centric primary criterion
was under-powered at 19 collision events at the pre-model support check. SAP v1
and SAP v2 provenance, denylist, scope boundary, non-outcome firewall, and
non-criterion rules remain in force unless explicitly amended here.

No analysis, scorer run, endpoint model, outcome read, plot, knowledge decision
edit, registry/dashboard edit, paper-repository edit, or claim acceptance is
authorized by this lock. The required next step is clean-room recompute; W1 is
the first outcome-touching run on the broad endpoint.

## 1. Ultimate RQ And Claim Boundary

Ultimate RQ:

```text
Can IPV, one socially meaningful and interpretable indicator of deviation from
the human cooperative norm, serve as a parsimonious runtime monitor of failed AV
interaction behavior, such that:

(a) one interpretable number flags interaction failures about as well as the
    full physical-parameter ensemble;
(b) its sign/direction distinguishes aggressive versus passive failure types;
(c) it depends on real IPV structure and generalizes under LOSO?
```

The thesis is parsimony/sufficiency, not incremental-over-physics. IPV is
derived from interaction physics, so the confirmatory question is whether one
interpretable IPV-derived signal recovers a predeclared fraction of the full
scheme-matched physical-parameter ensemble's failure-monitoring signal.

Positive interpretation is bounded to an associational, exposure-conditioned,
runtime-monitor claim at the locked unit, `algorithm_id x scenario_id`. This
does not support causal, planner-benefit, safety-benefit, human-normative,
fault-assignment, or independence-from-closing/proximity-physics claims.

## 2. Unit, Firewall, And Inputs

Unit of analysis remains one replay-clean `algorithm_id x scenario_id` cell,
with audited `algorithm_id == team_id` and `case_id == scenario_id`.

This lock uses field definitions, event definitions, signal availability,
threshold rationales, clean-room artifact filenames, frozen M3 envelope
semantics, and prior frozen design decisions. It does not use endpoint values,
endpoint counts, endpoint prevalence, model results, W1 outputs, scorer
outputs, or IPV-outcome associations to choose thresholds, endpoint categories,
transformations, support rules, weights, exclusions, model variants, or scheme
pairings.

Definition sources:

- RQ011 official field contract: diagnostic PDFs and official score tables for
  comprehensive score, component/sub-item scores, score-0 collision status,
  task-completion failure, safety intervention, and official deduction/rule
  violation categories.
- RQ012 automatic-event contract: `E02`, `E03`, `E06`, `E09`, `E15`, `E16`,
  `E18`, and `E19` with outcome-blind thresholds, causal timing, merge gaps,
  missing-data rules, and precedence rules from the frozen event ontology.
- v5.1 kinematic detector appendix: causal current/past trajectory, geometry,
  timestamp, and neutral actor-ID detectors. Non-executable or discretionary
  detector rows are descriptive-only and excluded from the confirmatory
  composite.
- Clean-room pre-outcome artifacts under
  `data/derived/onsite_competition/RQ011B_matched_scenario/RQ011B_1_matched_scenario_20260625T202454_8331bd49/cleanroom/`.

The v4/v5 denylist carries forward. Banned for endpoint construction decisions,
scheme thresholds, support logic, model selection, or controls except when
explicitly named as a falsification transform: IPV-outcome associations,
official score/rank/team identity for tuning, observed PET, realized passing
order, post-hoc phase labels, closest/critical approach frame selected using
the failure, realized future minimum distance, full-window IPV,
target-proximal ego behavior in the scored rolling IPV target window, estimator
reward components, realized opportunity duration, W1 outputs, and any model fit
against an endpoint.

## 3. M3 Envelope-Anchored Deviation Variables

The frozen RQ009 M3 interval is the human cooperative-norm envelope. IPV sign
semantics are fixed as:

- higher IPV = more cooperative/yielding;
- lower or negative IPV = more selfish/competitive.

For every selected anchor `a`:

```text
scale_a = max((M3_upper_a - M3_lower_a) / 2, deviation_scale_floor)
```

The deviation scale floor remains `0.05` radians only if RQ009 documents
target-IPV noise/scale support. If that documentation is absent, execution
pauses before prediction consumption.

Unsigned aggregate deviation:

```text
unsigned_norm_deviation_a =
  0                                      if target_ipv_a in [M3_lower_a, M3_upper_a]
  (target_ipv_a - M3_upper_a) / scale_a  if target_ipv_a > M3_upper_a
  (M3_lower_a - target_ipv_a) / scale_a  if target_ipv_a < M3_lower_a
```

Signed direction deviation is anchored to the M3 envelope, not to a sample
mean:

```text
signed_norm_deviation_a =
  0                                      if target_ipv_a in [M3_lower_a, M3_upper_a]
 +(target_ipv_a - M3_upper_a) / scale_a  if target_ipv_a > M3_upper_a
 -(M3_lower_a - target_ipv_a) / scale_a  if target_ipv_a < M3_lower_a
```

Interpretation:

- `signed_norm_deviation_a > 0`: above the human norm envelope,
  over-cooperative/passive direction.
- `signed_norm_deviation_a < 0`: below the human norm envelope,
  over-competitive/aggressive direction.
- `signed_norm_deviation_a == 0`: inside the human norm envelope.

For each scheme `S`:

```text
m3_norm_deviation_S        = mean(unsigned_norm_deviation_a over selected anchors)
m3_signed_deviation_S      = mean(signed_norm_deviation_a over selected anchors)
m3_aggressive_tail_raw_S   = max(-m3_signed_deviation_S, 0)
m3_passive_tail_raw_S      = max( m3_signed_deviation_S, 0)
```

Directional tails are split before scaling. No transformation may move the
signed zero away from the M3 envelope. Directional variables may be scaled only
by training-row or LOSO-fold standard deviation:

```text
m3_aggressive_tail_scaled_S = m3_aggressive_tail_raw_S / sd_train(m3_aggressive_tail_raw_S)
m3_passive_tail_scaled_S    = m3_passive_tail_raw_S    / sd_train(m3_passive_tail_raw_S)
m3_signed_scaled0_S         = m3_signed_deviation_S    / sd_train(m3_signed_deviation_S)
```

No mean-centering is allowed for signed-direction variables. Zero or nonfinite
training standard deviation makes the affected directional model
`UNDER_IDENTIFIED_DIRECTIONAL`.

## 4. Confirmatory Interaction-Failure Catalog

The machine-readable catalog is
`prereg_v5_1_draft/interaction_failure_catalog_v5_1.csv`. All endpoint flags
are cell-level unless otherwise stated: an endpoint is positive if at least one
qualifying interval, official field, or deduction category occurs in that
`algorithm_id x scenario_id` cell.

Direction classes:

- `AGGRESSIVE`: expected to associate with below-envelope signed deviation,
  interpreted as too selfish/assertive relative to the human cooperative
  envelope.
- `PASSIVE`: expected to associate with above-envelope signed deviation,
  interpreted as over-yielding/freezing relative to the human cooperative
  envelope.
- `SEVERE_UNDIRECTED`: severe, safety, comfort, or official failures whose
  mechanism may arise from either tail or from non-social context.

Confirmatory categories included in the primary composite:

| Category | Source | Direction class | Locked definition summary |
| --- | --- | --- | --- |
| `OFF_collision_score0` | RQ011 official | `SEVERE_UNDIRECTED` | Official comprehensive score equals 0 using RQ011 score-0 collision semantics. |
| `OFF_collision_or_rule_deduction` | RQ011 official | `SEVERE_UNDIRECTED` | Nonzero official collision, safety, or compliance rule-violation deduction, including parsed collision PDF deduction where available. |
| `OFF_safety_intervention` | RQ011 official | `SEVERE_UNDIRECTED` | Parsed official safety-intervention / safety-officer intervention evidence is nonzero. |
| `OFF_noncollision_task_noncompletion` | RQ011 official | `PASSIVE` | Parsed task-completion failure or mission non-completion with score-0 collision flag equal to 0. |
| `OFF_efficiency_progress_deduction` | RQ011 official | `PASSIVE` | Official efficiency, progress, time, or task-completion deduction is nonzero, excluding score-0 collision. |
| `OFF_comfort_dynamics_deduction` | RQ011 official | `SEVERE_UNDIRECTED` | Official comfort, smoothness, dynamics, or passenger-comfort deduction is nonzero. |
| `E02_high_deceleration` | RQ012 automatic | `SEVERE_UNDIRECTED` | Same-actor acceleration `<= -3.4 m/s^2` for at least 0.3 s, merged by `<=0.3 s` gaps. |
| `E03_high_jerk` | RQ012 automatic | `SEVERE_UNDIRECTED` | Causal 0.3 s rolling-median acceleration and backward-difference longitudinal jerk with `abs(jerk) >= 5 m/s^3` for at least 0.3 s. |
| `E06_repeated_stop_go` | RQ012 automatic | `SEVERE_UNDIRECTED` | At least two complete stop-go cycles using stop `<=0.3 m/s`, go `>=1.0 m/s`, stop/go run `>=0.5 s`, alternating sequence `>=4.0 s`. |
| `E09_near_miss` | RQ012 automatic | `SEVERE_UNDIRECTED` | Ego-other oriented-footprint clearance `<=0.5 m` without contact, or current-state constant-velocity TTC `<=1.5 s`, for at least 0.2 s. |
| `E15_geometric_contact` | RQ012 automatic | `SEVERE_UNDIRECTED` | Ego-other oriented-footprint overlap/contact candidate for at least 0.2 s. |
| `E16_no_progress_deadlock` | RQ012 automatic | `PASSIVE` | Ego displacement `<=1.0 m` and ego speed `<=0.3 m/s` over a causal 10 s rolling window. |
| `E18_kinematic_emergency_stop` | RQ012 automatic | `SEVERE_UNDIRECTED` | Ego deceleration magnitude `>=4.5 m/s^2` plus stop/brake evidence, stop speed `<=0.3 m/s`, for at least 0.3 s. |
| `E19_lateral_comfort` | RQ012 automatic | `SEVERE_UNDIRECTED` | Ego lateral acceleration `>=2.5 m/s^2`, lateral jerk `>=5 m/s^3`, or steering rate `>=100 deg/s` after causal smoothing for at least 0.3 s. |
| `KIN_force_other_hard_brake` | v5.1 detector | `AGGRESSIVE` | Non-ego E02 response within 1.0 s after same-pair ego encroachment pressure, with no qualifying prior non-ego E02 run. |
| `KIN_force_other_evasion` | v5.1 detector | `AGGRESSIVE` | Non-ego lateral acceleration/jerk/heading-rate evasion for at least 0.3 s within 1.0 s after same-pair ego encroachment pressure, with prior-run exclusion. |
| `KIN_cutin_fail_to_yield_proxy` | v5.1 detector | `AGGRESSIVE` | Ego enters a non-ego current-state 2.0 s velocity corridor under closing/proximity pressure, followed within 1.0 s by same-pair hard-brake, evasion, E09, or E15 evidence. |
| `KIN_excessive_hesitation_over_yield` | v5.1 detector | `PASSIVE` | Past 5.0 s ego stop/no-progress window, no active E09/E15, and at least one continuous 3.0 s open-gap subwindow. |

Excluded descriptive/unavailable rows remain excluded from `any_interaction_failure`,
Gate 1, Gate 2, Gate 3, directional confirmatory families, and positive support
language. In particular, `KIN_encroachment_contact_proxy` is descriptive-only:
E15 geometric contact remains confirmatory, but aggressive contact attribution
is not frozen in v5.1.

## 5. Endpoint Hierarchy And Sparse-Event Tiering

Primary composite endpoint:

```text
any_interaction_failure =
  OR over every category in interaction_failure_catalog_v5_1.csv
  where confirmatory_role == "confirmatory"
  and composite_inclusion == "included"
```

Expected aggregate direction: larger unsigned normalized M3 deviation should
increase the probability of `any_interaction_failure`.

Composite sparse-event tiering:

- If composite events and composite non-events are both `>=20`, run full
  binary Gate 1, Gate 2, and Gate 3.
- If the smaller of composite events and non-events is `10-19`, report only
  M3-only exact or blocked-permutation inference and descriptive parsimony
  summaries; no positive aggregate monitor, Gate 2 support, or Gate 3
  generalization claim is allowed.
- If composite events or non-events are `<10`, or the endpoint is all-event or
  all-none, label `UNDER_IDENTIFIED_COMPOSITE_SPARSE` and make no inferential
  aggregate claim.

Secondary per-category endpoints:

- Each of the 18 confirmatory categories is a separate secondary dependent
  variable.
- If events and non-events are both `>=20`, run full Gate 1, Gate 2, and Gate 3
  when the scheme pairing is allowed.
- If the smaller class is `10-19`, use exact or blocked-permutation inference
  for M3-only direction and descriptive parsimony summaries; no positive
  per-category claim.
- If events or non-events are `<10`, or the endpoint is all-event or all-none,
  label `UNDER_IDENTIFIED` and make no inferential claim.
- Per-category results are secondary and cannot rescue or upgrade a failed
  Scheme A aggregate decision.

Directional endpoint families:

```text
aggressive_failure =
  OR over all confirmatory AGGRESSIVE categories

passive_failure =
  OR over all confirmatory PASSIVE categories

undirected_failure =
  OR over all confirmatory SEVERE_UNDIRECTED categories
```

Cells with both aggressive and passive categories remain in both binary
directional models. Multinomial sensitivity assigns cells to `none`,
`aggressive_only`, `passive_only`, or `mixed`; the primary three-class
multinomial excludes `mixed` and `undirected_only` rows from the directional
contrast but reports their support tier before modeling.

## 6. Schemes A-E

| Scheme | Role | Frozen default | Status and claim use |
| --- | --- | --- | --- |
| A | Primary | Closing anchors: `closing_rate_anchor > 0`; `K=5`; equal weights | Primary confirmatory scheme. |
| B | Robustness | Proximity anchors: `relative_distance_anchor <= 10.0 m`; `K=5`; equal weights | Robustness only if support passes; otherwise `UNDER_IDENTIFIED`. |
| C | Robustness/descriptive | Causal E02/E09 active anchors; `K=1`; equal weights | Descriptive-only where endpoint overlaps E02/E09 conditioning; otherwise robustness only if support passes. |
| D | Robustness/descriptive | Closing plus `counterpart_accel_anchor_mps2 <= -0.5`; `K=3`; equal weights | Descriptive-only where endpoint overlaps deceleration/hard-brake/emergency-stop conditioning; otherwise robustness only if support passes. |
| E | Sensitivity/descriptive as allowed | One peak causal interaction anchor from closing, distance, or decel pressure; `K=1` | Descriptive-only for overlapping endpoints/groups/composites. |

Scheme A is the only confirmatory primary scheme. B-E are robustness or
sensitivity only and cannot rescue a failed Scheme A aggregate result. B and C
remain under-identified at v4/v5 defaults unless the clean-room ledger
documents otherwise before W1.

## 7. Corrected Non-Circularity And Scheme x Endpoint Pairings

This lock distinguishes exposure conditioning from event/outcome conditioning
while forbidding every scheme x endpoint overlap. The machine-readable pairing
table is `prereg_v5_1_draft/scheme_endpoint_pairings_v5_1.csv`.

Locked pairing rule:

- Scheme A closing exposure may be paired with broad failure, per-category, and
  directional endpoints. Claims must remain exposure-conditioned, and
  closing/proximity-heavy endpoints require overlap diagnostics.
- Scheme B proximity exposure may be paired analogously if support passes.
  Claims must remain proximity-exposure conditioned, and proximity-heavy
  endpoints require overlap diagnostics.
- Scheme C is event-conditioned on E02/E09. It is descriptive-only against
  `any_interaction_failure`, `E02_high_deceleration`, `E09_near_miss`, any
  E02/E09-containing grouping, and any new kinematic category whose definition
  uses E02, E09, TTC, or E09/E15 follow-up as a required component.
- Scheme D is deceleration-conditioned. It is descriptive-only against
  `any_interaction_failure`, E02/deceleration/hard-brake/emergency-stop
  endpoints, and new kinematic categories whose definition requires hard
  braking or deceleration.
- Scheme E is descriptive-only against `any_interaction_failure`, E02, E09,
  E15, E18, all new confirmatory kinematic detectors, both directional groups,
  and any grouping/composite containing closing, distance, TTC/proximity,
  deceleration, hard-brake, or derived encroachment markers.
- Official-only endpoints remain allowed for all schemes because official
  fields are not scheme-conditioning variables.
- Descriptive-only pairings may be shown for transparency but cannot enter
  Gate 1, Gate 2, Gate 3, robustness, support, or positive claim language.

## 8. Physics Feature Ensemble

The v4/v5 `physics_full_S` component lists carry forward as closed lists. No
component may be added, dropped, reweighted, or replaced after PI acceptance.

Primitive variables include distance, log/negative-log distance,
ego/counterpart speed, speed pressure, relative speed, positive closing,
closing pressure, TTC risk, ego/counterpart/pair deceleration pressure,
E02/E09 active flags, and scheme membership flags. Cell-level components for
A/B, C, D, and E are exactly those in locked SAP v2/v4.

For every scheme-specific fit, compute raw cell-level components, learn means
and sample standard deviations on the fit row set, and standardize physics
components and unsigned IPV deviation variables on that row set. In LOSO,
scaling is learned on the 14 training scenarios and applied unchanged to the
held-out scenario. Zero or nonfinite standard deviation in a required component
makes the affected comparison `UNDER_IDENTIFIED_OR_DEGENERATE`; components are
not dropped.

`kinematics_only` remains reported-only for physics-overlap transparency and is
not a Gate 2 beat-gate.

## 9. Three-Layer Confirmatory Analysis

Layer 1, primary composite parsimony:

- Primary test is Scheme A x `any_interaction_failure`.
- Binary endpoints use logistic GLM with scenario fixed effects and Tjur
  coefficient of discrimination.
- Continuous secondary endpoints use OLS partial-R2 over the scenario/null
  model.
- Decision depends on Gate 1 parsimony/sufficiency, Gate 2
  structure-preserving specificity, and Gate 3 LOSO generalization.

Layer 2, per-category secondary analysis:

- Each of the 18 confirmatory categories is tested as a secondary endpoint when
  sparse-event support and scheme pairing permit.
- Per-category BH-FDR uses the fixed pre-registered denominator `m=18`.
- The denominator does not shrink after W1, support filtering, degeneracy,
  model failure, or forbidden pairings.
- Sparse, under-identified, forbidden-pairing, or degenerate rows count as
  non-rejections and cannot make positive claims.

Layer 3, directional envelope-anchored inverted-U:

- Signed deviation is anchored to the M3 envelope, split into aggressive and
  passive raw tails before scaling, and never mean-centered.
- `DIRECTIONALLY_INTERPRETABLE` requires both aggressive and passive expected
  tails to pass their dominance, bootstrap, Holm, and sign-flip rules.
- The inverted-U interpretation is permitted only when aggregate unsigned
  deviation flags failures and signed tails map to aggressive versus passive
  failure types under the locked per-tail rule.

## 10. Gate 1: Aggregate Parsimony/Sufficiency

For Scheme A and `any_interaction_failure`, fit on the identical row set if the
composite sparse-event rule is in the full tier:

```text
null:
  endpoint = scenario fixed effects

ipv_only_A:
  endpoint = scenario fixed effects
             + standardized m3_norm_deviation_A

physics_full_A:
  endpoint = scenario fixed effects
             + all standardized physics_full_A components
```

If fixed-effect fitting separates, does not converge, has singular/nonfinite
estimates, or has impossible scenario strata, use the predeclared scenario
random-intercept fallback for all three models. If the fallback also fails, the
comparison is `UNDER_IDENTIFIED_OR_DEGENERATE`. No post-outcome Firth, ridge,
lasso, dropped-scenario substitute, endpoint switch, or pseudo-R2 replacement
may be introduced.

Binary decision statistic:

```text
DeltaR2_ipv_A     = R2_Tjur(ipv_only_A) - R2_Tjur(null)
DeltaR2_physics_A = R2_Tjur(physics_full_A) - R2_Tjur(null)
eff_A             = DeltaR2_ipv_A / DeltaR2_physics_A
```

Continuous secondary statistic:

```text
DeltaR2_ipv_S     = 1 - SSE(ipv_only_S) / SSE(null)
DeltaR2_physics_S = 1 - SSE(physics_full_S) / SSE(null)
eff_S             = DeltaR2_ipv_S / DeltaR2_physics_S
```

Locked defaults:

```text
tau_physics    = 0.03
tau_ratio      = 0.60
B_bootstrap    = 10000
bootstrap_seed = 20260625
```

If `DeltaR2_physics_S` is negative, zero, nonfinite, or `< tau_physics`, the
comparison is `UNDER_IDENTIFIED_MOOT_PHYSICS` and no point-estimate ratio is
reported. Ratios are uncapped; values above 1 are allowed; negative IPV
numerators/ratios are reported but cannot pass. No clamping or winsorization is
allowed.

Gate 1 passes for the aggregate only if:

1. Scheme A x `any_interaction_failure` is allowed.
2. Gate 0 support passes.
3. The composite sparse-event rule is in the full tier.
4. `DeltaR2_physics_A >= tau_physics` and its 95% bootstrap CI lower bound is
   greater than 0.
5. `DeltaR2_ipv_A` has 95% bootstrap CI lower bound greater than 0.
6. The coefficient for standardized unsigned `m3_norm_deviation_A` is positive
   with 95% bootstrap CI excluding zero on the positive side.
7. `eff_A >= tau_ratio` by point estimate; its 95% bootstrap CI is mandatory
   reporting but not a strict lower-bound gate.

## 11. Gate 2: Structure-Preserving Specificity

Gate 2 tests whether the monitor depends on real IPV/cell/counterpart/sign/
envelope structure rather than arbitrary IPV-like transforms.

Required controls:

1. `shuffled_ipv`: block-permute true scheme-specific M3 deviation within exact
   `scenario_id` and outcome-blind `team_difficulty_tertile`, deranged within
   stratum.
2. `counterpart_swap`: keep ego trajectory, keys, time grid, endpoint, and
   scenario structure fixed; replace the target counterpart with the nearest
   eligible non-target actor in the same cell, tie-broken lexicographically;
   recompute support/deviation through the frozen RQ009 interface.
3. `wrong-cell/wrong-envelope`: assign another eligible cell's frozen M3
   envelope/deviation while preserving the marginal scheme-deviation
   distribution; default derangement is within exact `scenario_id`.
4. `sign_flip`: reverse signed IPV quantities before exceedance construction;
   preserve magnitudes, keys, time grid, support flags, and endpoint rows.
5. `IPV_removed`: use the frozen RQ009 M2/context-only envelope that excludes
   counterpart current IPV; same selected anchors, same K, same target
   construction, same scale floor, same equal weights.

Locked defaults:

```text
delta_beta = 0.05
delta_R2   = 0.01
B_perm     = 10000
perm_seed  = 20260625
```

For deterministic controls, true IPV must beat each required control by at
least `delta_beta` in harm-direction coefficient advantage with paired two-way
blocked bootstrap CI lower bound greater than 0, and by at least `delta_R2` in
`DeltaR2_true - DeltaR2_control` with paired CI lower bound greater than 0. For
stochastic controls, use max-statistic permutation with `B_perm=10000` and
require max-statistic `p < 0.05`.

If a required control is unavailable for more than 10% of otherwise eligible
primary rows, loses all 15-scenario / 16-team / 10-cells-per-scenario support,
or is key-incompatible/non-outcome-blind, no positive aggregate claim is
allowed.

## 12. Gate 3: LOSO Generalization

For each held-out scenario, train on the other 14 scenarios. Scaling for
unsigned Gate 3 predictors is learned on training scenarios only and applied
unchanged to the held-out scenario. The held-out scenario receives no fitted
scenario intercept.

Transfer models:

```text
null_transfer:    endpoint = intercept
ipv_transfer_S:   endpoint = intercept + standardized m3_norm_deviation_S
physics_transfer: endpoint = intercept + all standardized physics_full_S components
```

For binary endpoints use held-out log-loss skill against the held-out intercept
prediction. For continuous endpoints use held-out SSE skill.

Gate 3 passes for Scheme A only if:

- training-fold IPV coefficient is in the expected direction in at least 12 of
  15 folds;
- held-out IPV skill is positive in at least 12 of 15 folds;
- aggregate `DeltaQ2_ipv_all > 0` with two-way blocked bootstrap CI lower
  bound greater than 0;
- LOSO physics precondition is met: `DeltaQ2_physics_all >= tau_physics` and
  its CI lower bound greater than 0;
- `eff_LOSO_A >= tau_ratio` by point estimate, with CI reported;
- no single held-out scenario contributes more than 40% of aggregate positive
  IPV skill;
- removing the largest positive held-out scenario leaves positive aggregate
  IPV skill.

If more than three folds are invalid because of all-event/all-none training,
separation, nonconvergence, nonfinite coefficients, zero-variance predictors,
rank deficiency, or nonfinite predictions, Gate 3 is
`UNDER_IDENTIFIED_OR_DEGENERATE`.

## 13. Directional / Inverted-U Test

Primary directional models:

```text
logit Pr(aggressive_failure_i = 1) =
  alpha_scenario[i]
  + beta_aggr * m3_aggressive_tail_scaled_A_i
  + beta_pass * m3_passive_tail_scaled_A_i

logit Pr(passive_failure_i = 1) =
  alpha_scenario[i]
  + gamma_aggr * m3_aggressive_tail_scaled_A_i
  + gamma_pass * m3_passive_tail_scaled_A_i
```

Expected per-tail rules:

- Aggressive failures are expected in the below-norm/aggressive tail:
  `beta_aggr > 0` and `beta_aggr > beta_pass`.
- Passive failures are expected in the above-norm/passive tail:
  `gamma_pass > 0` and `gamma_pass > gamma_aggr`.

Directional support requires all of the following:

1. Both `aggressive_failure` and `passive_failure` satisfy the full
   sparse-event rule.
2. `beta_aggr` has 95% bootstrap CI lower bound greater than 0.
3. `gamma_pass` has 95% bootstrap CI lower bound greater than 0.
4. The paired bootstrap CI lower bound for `beta_aggr - beta_pass` is greater
   than 0 and the point contrast is at least `delta_beta = 0.05`.
5. The paired bootstrap CI lower bound for `gamma_pass - gamma_aggr` is greater
   than 0 and the point contrast is at least `delta_beta = 0.05`.
6. Holm correction across the two tail-direction primary contrasts does not
   overturn either contrast at alpha 0.05.
7. The sign-flip Gate 2 control fails to reproduce the directional pattern.

Sensitivity directional model:

```text
multinomial outcome in {none, aggressive_only, passive_only}
  ~ scenario fixed effects + m3_signed_scaled0_A
```

Expected multinomial signs are negative for `aggressive_only` versus `none` and
positive for `passive_only` versus `none`. This sensitivity is not required for
`DIRECTIONALLY_INTERPRETABLE`, but disagreement must be discussed.

Directional labels:

- `DIRECTIONALLY_INTERPRETABLE`
- `NON_DIRECTIONAL`
- `UNDER_IDENTIFIED_DIRECTIONAL`

## 14. Multiplicity, Labels, And Interpretation Ceiling

Primary aggregate decision is controlled by Scheme A on
`any_interaction_failure`. B-E are robustness/sensitivity only, and
descriptive-only pairings cannot support robustness language. Per-category
endpoints and official continuous endpoints are secondary. Directional
interpretability is a separate required interpretability label for the ultimate
RQ.

Aggregate labels:

- `SUPPORTED_PARSIMONIOUS_MONITOR`
- `NON_PARSIMONIOUS`
- `NON_SPECIFIC`
- `NON_GENERALIZING`
- `UNDER_IDENTIFIED`

The strongest v5.1 positive conclusion requires
`SUPPORTED_PARSIMONIOUS_MONITOR` and `DIRECTIONALLY_INTERPRETABLE`. If aggregate
support passes but direction fails, the result can support a parsimonious
undirected failure monitor but not the interpretable social-value claim that
sign distinguishes aggressive versus passive failure modes.

Interpretation ceiling:

- Allowed: one interpretable IPV-derived deviation monitor flags broad
  interaction failure about as well as the full scheme-matched physics ensemble,
  is specific to real IPV/cell/counterpart/sign/envelope structure, generalizes
  under LOSO, and, if the directional layer passes, separates aggressive from
  passive failure modes.
- Not allowed: causal harm, planner benefit, safety benefit, legal fault,
  human-normative authority beyond the frozen empirical envelope,
  independence-from-physics, or broad deployment claims.
- Collision-only SAP v2 under-power is not evidence for or against the broad
  endpoint; W1 has not yet run on this locked broad endpoint.

## 15. Required Outputs At Execution

Required outputs, not gates unless explicitly stated:

- component decomposition of `any_interaction_failure` by category and
  direction class;
- composite and per-category event support tier before modeling;
- Gate 1 table with `R2(null)`, `R2(ipv_only)`, `R2(physics_full)`,
  `R2(union)`, `DeltaR2`, `eff`, coefficient signs, and bootstrap CIs;
- Gate 2 specificity table for all required controls;
- LOSO fold table;
- directional half-wave table and multinomial sensitivity;
- `kinematics_only` overlap diagnostics as transparency only;
- closing/proximity-overlap diagnostics for A/B exposure-conditioned pairings
  with E09, E15, and new kinematic encroachment categories;
- exact list of descriptive-only scheme x endpoint pairings excluded from
  support language;
- fixed-denominator per-category BH-FDR table with `m=18`, including
  under-identified or forbidden rows as non-claim entries.

## 16. Clean-Room Recompute Requirement

Before any W1 or outcome-consuming criterion execution, an independent
clean-room worker must recompute from frozen inputs:

- scheme membership A-E;
- selected-anchor counts and K eligibility;
- corrected scheme x endpoint pairing flags;
- unsigned and envelope-anchored signed M3 exceedance summaries;
- raw signed tails and scale-only directional predictors;
- `physics_full_S` component summaries;
- `kinematics_only_S` reported-only diagnostics;
- all required control deviations;
- endpoint-definition joins and category flags;
- v5.1 detector implementation status, including confirmation that
  descriptive-only categories are excluded from the primary composite and
  directional confirmatory families;
- fixed per-category BH-FDR denominator `m=18`.

The clean-room ledger must attest that no outcome values, endpoint counts before
the support-tier step, W1 outputs, model fits, or IPV-outcome associations were
used to choose thresholds, transformations, support rules, weights, model
variants, endpoint pairings, or exclusions. The ledger must preserve
hashes/paths for M3/M2 packages, RQ012 event definitions, RQ011 official-field
provenance, and the v5.1 catalog.

## 17. Acceptance Record

PI acceptance date: 2026-06-28.

Accepted defaults:

```text
primary scheme/endpoint: Scheme A x any_interaction_failure
scheme K defaults: A=5, B=5, C=1, D=3, E=1
binary pseudo-R2: Tjur coefficient of discrimination
continuous R2: OLS partial-R2 over scenario/null
tau_physics: 0.03
tau_ratio: 0.60
bootstrap: B=10000, seed=20260625
Gate 2 controls: shuffled_ipv, counterpart_swap, wrong-cell/wrong-envelope,
  sign_flip, IPV_removed
Gate 2 margins: delta_beta=0.05, delta_R2=0.01
permutation: B_perm=10000, seed=20260625
LOSO: 12/15 folds and largest positive scenario contribution <=40%
composite/per-category support: >=20 full; 10-19 descriptive exact/permutation;
  <10 no inferential claim
per-category BH-FDR denominator: m=18
```

Audit chain:

```text
v4 interaction-conditioned prereg
-> W1 pre-model support check found collision-only v2 primary under-powered
   at 19 collision events
-> v5 broad interaction-failure + per-category + directional revision
-> v5 independent review
-> v5.1 blocker-closing revision
-> v5.1 independent re-review PASS
-> PI accepted v5.1 on 2026-06-28
-> this locked SAP v3 amendment
```

Run-root pointers:

- accepted draft:
  `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/04_criterion_consequence/prereg_v5_1_draft/prereg_v5_1.md`
- catalog:
  `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/04_criterion_consequence/prereg_v5_1_draft/interaction_failure_catalog_v5_1.csv`
- scheme pairings:
  `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/04_criterion_consequence/prereg_v5_1_draft/scheme_endpoint_pairings_v5_1.csv`
- PI decision brief:
  `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/04_criterion_consequence/prereg_v5_1_draft/pi_decision_brief_v5_1.md`
- independent re-review:
  `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/04_criterion_consequence/prereg_v5_1_rereview/prereg_v5_1_rereview.md`
- PI acceptance brief:
  `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/04_criterion_consequence/prereg_v5_1_rereview/pi_acceptance_brief_v5_1.md`

## Amendment A1 (PI-approved 2026-06-29, pre-W1, outcome-blind)

This amendment changes only the confirmatory status of the Gate-2
`counterpart_swap` control. `counterpart_swap` is demoted from CONFIRMATORY to
DESCRIPTIVE/robustness and will be reported descriptively on its available
coverage.

The confirmatory Gate-2 control set is now the four retained
structure-preserving controls:

1. `shuffled_ipv`
2. `wrong_cell` / `wrong_envelope`
3. `sign_flip`
4. `IPV_removed`

The Gate-2 margins remain `delta_beta=0.05` and `delta_R2=0.01`. The stochastic
Gate-2 control inference remains the max-statistic permutation with
`B_perm=10000` and seed `20260625`.

Justification: the full `counterpart_swap` build is compute-prohibitive in the
current local workflow, with a measured partial run of approximately
`709.5 s` for `100` selected anchors out of `35,754` unique selected anchors,
implying approximately `40-42 h` serial runtime and likely requiring HPC for
completion. It is also partially redundant with the four retained controls,
which jointly test whether the result depends on the real IPV-to-cell
structure through shuffled IPV assignment, wrong-cell/wrong-envelope structure,
sign inversion, and IPV removal.

Honesty note: this amendment was decided and recorded before any W1 outcome
relationship was estimated for the broad interaction-failure endpoint. Nothing
else in the locked SAP v3 changes.
