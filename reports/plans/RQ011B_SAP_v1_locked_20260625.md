# RQ011B Locked Statistical Analysis Plan v1

Status: **LOCKED — PI-approved 2026-06-25; supersedes the underspecified parts of plan v0**  
RQ: `RQ011B`  
Run: `RQ011B_1_matched_scenario_20260625T202454_8331bd49`  
Source packet: `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/01_plan_review/SAP_v2_draft/`  
Review packet: `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/01_plan_review/SAP_v2_review/`

This locked SAP is the binding pre-readout analysis specification for RQ011B phase 3+. It consolidates SAP v2, the expanded denylist, negative-control specification, decision rules, phase-3 entry gate, and PI summary into one self-contained addendum. It resolves Phase-1 blockers `B001` through `B005` by specification only. Phase 3 remains paused until Gate `G1` is cleared by RQ009.

No M3 deviation, outcome association, criterion modeling, plotting, registry update, decision file edit, or paper-repository edit is authorized by this SAP lock.

## 1. Frozen Scope

- Unit of analysis: `algorithm×scenario`, with audited keys `algorithm_id == team_id` and `case_id == scenario_id`.
- Outcome universe: `full_300` only for outcome-only context and denominator statements.
- Confirmatory replay/IPV universe: `clean_285` only. `T19` is excluded for replay/IPV because no unique T19-owned replay/session is identifiable.
- Confirmatory endpoint rows: `clean_285` cells with frozen RQ009 M3 support/estimability, all prespecified baseline covariates, and all required primary-model fields available.
- Cells failing M3 support/estimability are `ABSTAINED`, never imputed.
- Claim ceiling: criterion/consequence validity at `algorithm×scenario` only. No run-level, repeated-run, seed-level, independent-case, algorithm-superiority, planner-benefit, causal, or full_300 replay/IPV claim is authorized.
- Phase-2 provenance status: universe/key/log audit only. No M3 was read and no IPV/outcome association was used. RQ011B phase 3 may not reinterpret replay eligibility using outcome values.

Audited data shape:

- 20 teams x 15 shared scenarios = 300 outcome cells.
- `clean_285` replay/IPV cells after frozen T19 replay-only exclusion.
- Collision is sparse: 33/300 full outcome cells and 24/285 replay-clean cells.
- Team-level ranks are complete but repeated across 15 scenario rows per team; they are not treated as 285 independent cell outcomes.

## 2. Locked PI Configuration

The PI accepted SAP v2 with the following locked values:

- Primary coefficient floor: `|beta| >= 0.15`, expected direction `beta < 0`.
- Primary partial-R2 floor: `tau = 0.03`, where `partial_R2 = 1 - SSE_M3 / SSE_baseline`.
- Negative-control margins: `delta_beta = 0.05`, `delta_R2 = 0.01`, both with paired blocked confidence intervals.
- Uncertainty: two-way team x scenario blocked wild cluster bootstrap, `B = 10000`, seed `20260625`.
- Permutations: `B_perm = 10000`, base seed `20260625`.
- Support floor: `n >= 210`, `n/slope >= 10`, all 15 scenarios, at least 16 teams, and at least 10 cells per scenario.
- LOSO: separate no-scenario-FE transfer model, `k = 12/15`, single-scenario influence cap `<= 40%`.
- Collision secondary: two predictors only if events `>= 20` and non-events `>= 20`; events 10-19 use M3-only exact/permutation inference or descriptive reporting; events `< 10` receive no inferential claim.
- `log_opportunity_duration_z`: dropped from baseline.
- Deviation scale floor `0.05` rad: retained only if RQ009 documents target-IPV noise/scale support. Otherwise the PI must freeze a documented replacement before prediction consumption.
- Phase-3 Gate `G1`: unchanged and still pending. RQ009 must freeze M3 and explicitly clear downstream consumption after the M3-vs-M4 pivot.

## 3. Endpoint Hierarchy

Primary endpoint:

- Official per-scenario comprehensive score (`official_comprehensive` or audited equivalent `score_comprehensive_from_primary_table`) on supported `clean_285`, modeled at the `algorithm×scenario` cell level.
- Positive validity direction: higher M3 norm-deviation predicts lower official comprehensive score.

Ordered secondary endpoints:

1. Deduction burden: `log1p(total_pdf_deduction_v2)`, where `total_pdf_deduction_v2 = pdf_collision_deduction + pdf_safety_intervention_deduction`. `pdf_task_completion_score` and `deduction_reason_categories` are audited fields but not numeric components of this endpoint; they may be reported descriptively only.
2. Collision: `collision_flag_score0`, a binary indicator audited as equivalent to `official_comprehensive == 0`.
3. Team-level rank: `area_rank` and `global_rank`, reported descriptively and with team-blocked summaries only. Rank may not be used as a 285-row independent inferential endpoint.

The primary endpoint gates interpretation of secondaries. Secondary results cannot rescue a failed primary endpoint.

## 4. M3 Norm-Deviation And Guard Outputs

Phase 3 must consume a frozen RQ009 prediction package and produce two separate outputs for every eligible cell before criterion modeling:

1. Empirical norm-deviation output: primary `m3_norm_deviation`.
2. Safety/policy guard output: guard flags, guard margins, or abstention reasons.

Primary norm-validity evidence uses only empirical norm-deviation. Guard-induced flags are excluded from the primary validity model and may appear only in a separate guard diagnostics table.

Primary deviation definition:

- RQ009 must define the target rolling IPV as same-window current rolling IPV on a post-anchor, non-overlapping target window.
- Reject the M3 package if the target uses full-window IPV, overlapping self-anchor windows, ego self-anchor input in M3, post-hoc target construction, target-proximal concurrent ego acceleration/braking, or any official outcome/rank/deduction/collision field.
- For each support-valid frame or segment, compute positive standardized exceedance outside the frozen M3 interval:
  - `0` when observed rolling IPV target is inside `[M3_lower, M3_upper]`.
  - Above interval: `(target_ipv - M3_upper) / scale`.
  - Below interval: `(M3_lower - target_ipv) / scale`.
- `scale = max((M3_upper - M3_lower) / 2, deviation_scale_floor)`.
- `deviation_scale_floor = 0.05 radians` only if RQ009 documents it from target-IPV noise/scale. If not documented, phase 3 must pause until the PI freezes a documented replacement before predictions are consumed.
- Cell-level `m3_norm_deviation` is the estimability-weighted mean positive exceedance over support-valid frames or segments.
- Cells with no support-valid frames are `ABSTAINED_NON_ESTIMABLE`, not imputed.
- Signed lower-tail and upper-tail exceedance summaries may be reported as diagnostics, but they do not replace the primary term after results are seen.

## 5. Primary Score Model

Primary model family: nested linear score model fit on eligible `clean_285` cells after support and missingness gates.

Baseline model:

```text
score_cell = scenario fixed effects
             + ego_speed_at_opportunity_start_z
             + counterpart_speed_at_opportunity_start_z
             + log_initial_gap_at_opportunity_start_z
             + closing_rate_at_opportunity_start_z
             + replay_support_fraction_z
             + error
```

M3 model:

```text
score_cell = scenario fixed effects
             + ego_speed_at_opportunity_start_z
             + counterpart_speed_at_opportunity_start_z
             + log_initial_gap_at_opportunity_start_z
             + closing_rate_at_opportunity_start_z
             + replay_support_fraction_z
             + standardized m3_norm_deviation
             + error
```

`log_opportunity_duration_z` is dropped from the baseline. It may not be reintroduced unless a later PI-approved SAP amendment, made before M3 or outcome association, defines it as a fixed exposure horizon known at opportunity start with detector and timestamp provenance proving no realized-future information.

Primary increment statistic:

```text
partial_R2 = 1 - SSE_M3 / SSE_baseline
```

Both SSE values must be computed on the identical eligible row set. Adjusted R2 may be reported descriptively only and is not a decision statistic.

## 6. Baseline Covariates And Transformations

The compact baseline covariate list is frozen and contains exactly these five non-M3 covariates:

1. `ego_speed_at_opportunity_start_z`.
2. `counterpart_speed_at_opportunity_start_z`.
3. `log_initial_gap_at_opportunity_start_z`.
4. `closing_rate_at_opportunity_start_z`.
5. `replay_support_fraction_z`.

Raw formulas before z-standardization:

- Opportunity start is the frozen RQ009 opportunity-start timestamp for the cell. It must be produced by an outcome-blind opportunity detector and must not depend on realized opportunity end, observed PET, closest/critical frame, realized order, official outcome, or M3-outcome association.
- `ego_speed_at_opportunity_start = sqrt(vx_ego^2 + vy_ego^2)` in m/s at opportunity start.
- `counterpart_speed_at_opportunity_start = sqrt(vx_counterpart^2 + vy_counterpart^2)` in m/s at opportunity start.
- `initial_gap_at_opportunity_start = max(euclidean_distance(ego_xy, counterpart_xy), 0.1 meters)` at opportunity start.
- `log_initial_gap_at_opportunity_start = natural_log(initial_gap_at_opportunity_start)`. Larger values mean larger starting separation and lower generic proximity risk.
- Let `r = counterpart_xy - ego_xy`, `v_rel = counterpart_velocity_xy - ego_velocity_xy`, and `d = max(norm(r), 0.1 meters)` at opportunity start. `closing_rate_at_opportunity_start = - dot(r, v_rel) / d`, so positive values mean the pair is closing and negative values mean opening.
- `replay_support_fraction = support_valid_frames_or_segments / frames_or_segments_considered`, using frozen RQ009 opportunity/support fields only. It must not use outcomes or association readout.

Transformations:

- Primary score stays on the 0-100 official scale for model fitting and is standardized only for standardized effect reporting.
- Continuous baseline predictors and `m3_norm_deviation` are z-standardized using eligible analysis rows for the specific fit before outcome association.
- For LOSO transfer scoring, z-scaling is learned on training scenarios only and applied unchanged to held-out scenarios.
- No outcome-driven winsorization is allowed.
- Deduction burden uses `log1p(total_pdf_deduction_v2)`.
- Collision is binary and is not transformed.

Missingness rule:

- If any required raw covariate cannot be computed from frozen allowed sources, the cell is feature-unavailable for the affected analysis.
- No replacement covariate, imputation, or outcome-tuned missingness rule may be introduced after outcome association.

## 7. Parameter Budget And Support Floor

Primary score model default hard cap:

- 14 scenario fixed-effect parameters.
- 5 non-M3 baseline covariate parameters.
- 1 M3-deviation parameter.
- Total: 20 non-intercept slopes, plus intercept.

The previous 21-slope cap remains the maximum allowed only if a PI-approved pre-readout amendment reintroduces one fixed, causal exposure covariate. No unapproved model may exceed 21 slopes.

A positive primary claim is allowed only if the post-abstention primary eligible row set passes all support floors:

- `n_eligible >= 210`.
- `n_eligible / p_slopes >= 10`, where `p_slopes` is the number of non-intercept slopes in the fitted primary M3 model.
- All 15 shared scenarios represented.
- At least 16 teams represented.
- Per-scenario eligible support `n_s >= 10` for every scenario.

If any support floor is unmet, the result label is `UNDER_IDENTIFIED_OR_DEGENERATE`; report bounded/descriptive results only and make no positive primary claim.

No interaction terms, spline bases, data-adaptive feature selection, team fixed effects, or post-association covariate additions are allowed in the confirmatory model.

## 8. Primary Uncertainty And Permutation Procedures

Use a two-way team x scenario blocked wild cluster bootstrap:

- Bootstrap replicates: `B = 10000`.
- Seed: `20260625`.
- For replicate `b`, draw independent Rademacher weights for every represented team and every represented scenario using NumPy `PCG64(seed=20260625 + b)`.
- Cell residual weight: `w_i = w_team[team_i] * w_scenario[scenario_i]`.
- Refit baseline and M3 models on each bootstrap pseudo-outcome.
- Compute standardized M3 coefficient and `partial_R2` in each replicate.
- Report two-sided percentile 95% intervals.
- Decision rules use the adverse-side coefficient interval and the lower bound for incremental gain.
- Reuse the same bootstrap weights for paired M3-vs-control differences on the common row set for that control.

Permutation controls:

- Stochastic controls use `B_perm = 10000` permutations.
- Base seed: `20260625`.
- Permutation `b` uses seed `20260625 + b`.
- Monte Carlo p-value: `(1 + number of permuted max-statistics >= observed statistic) / (1 + B_perm)`.

## 9. Expanded Denylist

The following variables are banned from the online/feature path, baseline covariates, M3 deviation construction, negative-control construction except where explicitly used as a named falsification transform, imputation, exclusions, and weights:

- Observed PET.
- Realized passing order.
- Post-hoc phase labels.
- Closest-approach or critical-approach frame.
- Realized-future minimum distance.
- Full-window IPV.
- Full-window, overlapping, self-anchor, or post-hoc target IPV construction.
- Ego early/self-anchor IPV in the M3 primary model.
- Target-proximal ego behavior, including ego acceleration or braking in the same window as the scored rolling IPV target.
- Estimator-internal reward components.
- `log_opportunity_duration_z` or any realized opportunity duration in the baseline model, except under a later PI-approved pre-readout SAP amendment defining a fixed exposure horizon known at opportunity start with detector and timestamp provenance proving no realized-future information.
- Any outcome-derived or post-outcome variable, including official score, rank, collision, deduction, success/failure, or mission status, when deciding replay eligibility, feature availability, model covariates, exclusions, transformations, weights, thresholds, support rules, or imputation.

Offline-only descriptive variables may be named in caveat ledgers or audit reports, but they may not enter the confirmatory model, controls, support logic, or feature path.

The following must be frozen before any outcome/IPV/IPV-outcome association readout:

- Cell inclusion and exclusion rules.
- M3 support, estimability, opportunity, and abstention rules.
- M3 target construction and non-overlap rules.
- All feature families and baseline covariates.
- All transformations and scaling rules.
- All weights, if any; default is unweighted cell-level analysis.
- All missingness and abstention handling.
- All negative-control definitions, permutation strata, common-row refit rules, and seeds.
- All model families, parameter caps, support floors, inference methods, and decision thresholds.
- All endpoint hierarchy and multiplicity rules.

Forbidden after association readout:

- Adding, dropping, or reweighting cells because of score, rank, collision, deduction, M3 deviation, or their association.
- Choosing a different primary endpoint because the prespecified endpoint fails.
- Adding covariates, interactions, splines, transformations, winsorization, or imputation rules because they improve association.
- Re-defining M3 deviation, support, estimability, opportunity detection, or abstention to improve the primary or secondary result.
- Re-labeling guard-triggered failures as norm-deviation evidence.
- Selecting a favorable negative-control subset or omitting unfavorable controls.
- Reporting full_300 replay/IPV coverage or using T19 replay-excluded cells as if they had M3 deviation.

RQ011B may not claim:

- Run-level effects.
- Repeated-run effects.
- Seed-level effects.
- Independent-case effects.
- Algorithm superiority.
- Planner benefit, closed-loop benefit, or deployed safety benefit.
- Causal realized harm.
- Formal safety or social-compliance proof.
- Full_300 replay coverage.
- Full_300 IPV coverage.
- Human normative authority from self-anchor-only evidence.
- Directional temporal IPV motif validation.

The strongest possible positive claim is bounded criterion/consequence validity for frozen M3 norm-deviation at the `algorithm×scenario` level on replay-clean supported cells, conditional on passing all decision gates.

## 10. Negative Controls

All decision-gate controls are run at the same primary unit and through the same endpoint, support, baseline, transformation, inference, and nested-comparison pipeline as the primary score model. The frozen M3 model is never retrained inside RQ011B.

Shared gate rules:

- For every control, refit the true M3 model and the control model on the same control-specific eligible row set `E_c`.
- The baseline endpoint, covariates, transformations, and rows are identical for true M3 and control within `E_c`.
- The baseline is never permuted or redefined inside a decision-gate control.
- If a control cannot be constructed for more than 10% of otherwise eligible primary rows, the positive primary claim is blocked. Degeneracy tolerance is `<= 10%`.
- Deterministic controls record `seed = not_applicable`.
- Stochastic controls use NumPy `PCG64`, base seed `20260625`, and `10000` permutations.
- The blocked team/scenario label-permutation is a separate global diagnostic outside the all-control increment gate.

Shared comparison statistics:

- Standardized coefficient direction and paired uncertainty.
- `partial_R2 = 1 - SSE_candidate / SSE_baseline` against the compact baseline on the common row set.
- Paired M3-minus-control differences for adverse coefficient strength and partial R2.
- Common-row support and missingness counts.

Decision-gate controls:

1. `role_flip`: recompute M3 deviation after swapping ego and counterpart roles wherever RQ009 supports both role directions. Preserve cell keys, time grid, scenario fixed effects, baseline covariates, outcomes, and support/abstention logic. Refit true M3 and role-flip models on rows with both terms available.
2. `sign_flip`: reverse the sign of signed IPV quantities before constructing the frozen M3 exceedance summary. Preserve magnitudes, keys, time grid, support flags, structure, baseline covariates, and outcome. Refit true M3 and sign-flip models on rows with both terms available.
3. `counterpart_swap`: keep ego trajectory, keys, time grid, baseline covariates, and outcome fixed; replace the target counterpart at opportunity start with the nearest eligible non-target actor in the same cell, tie-broken by lexicographic actor ID. Recompute support and abstention through the frozen RQ009 interface. Refit on common rows.
4. `kinematics_only`: replace M3 deviation with the frozen kinematics-only index below. Preserve keys, scenario/team structure, support filters, and outcome. The compact baseline remains unchanged.
5. `IPV_removed`: consume the frozen RQ009-supplied M2/context-only prediction package that excludes counterpart current IPV. Compute the same positive standardized exceedance against the frozen M2/context-only envelope. Local RQ011B recomputation is not allowed.
6. `shuffled_ipv`: block-permute true `m3_norm_deviation` within strata preserving exact `scenario_id` and outcome-blind `team_difficulty_tertile`; derange within stratum so no cell receives its own deviation. Sparse strata merge adjacent tertiles within scenario in fixed order; >10% scenario-wide fallback fails Gate 2.
7. `wrong-cell / wrong-envelope`: assign each eligible cell another eligible cell's frozen M3 envelope/deviation while preserving the marginal deviation distribution. Default derangement is within exact `scenario_id`; sparse scenarios merge according to canonical order `A1,A2,A3,A4,A5,A6,A7,B1,B2,B3,B4,C1,C2,C3,C4`.
8. Blocked team/scenario label-permutation diagnostic: preserve marginal distributions of outcomes, M3 deviations, baseline covariates, scenarios, and team-difficulty strata; within each scenario, permute team labels for the full predictor row bundle inside outcome-blind team-difficulty strata while holding outcome rows fixed. This diagnostic cannot satisfy or fail Gate 2.

Frozen kinematics-only index:

```text
speed_pressure = (ego_speed_at_opportunity_start + counterpart_speed_at_opportunity_start) / 2
closing_pressure = max(closing_rate_at_opportunity_start, 0) / initial_gap_at_opportunity_start
gap_protection = log_initial_gap_at_opportunity_start

kinematics_only_index =
  mean_z(speed_pressure,
         closing_pressure,
         -gap_protection)
```

Higher values mean faster, more closing, and smaller-gap generic risk. Missing any component makes the control term unavailable for that cell.

Frozen outcome-blind team-difficulty tertiles:

```text
cell_replay_difficulty =
  mean_z(speed_pressure,
         closing_pressure,
         -log_initial_gap_at_opportunity_start,
         1 - replay_support_fraction)

team_replay_difficulty = mean(cell_replay_difficulty over eligible clean_285 cells for the team)
```

Cutpoints are the type-7 empirical 33.333rd and 66.667th percentiles among represented teams. If ties at a cutpoint create an empty tertile, break ties by lexicographic `team_id` to create the most balanced possible tertiles. Numeric cutpoints and assignments must be written to a pre-readout manifest.

Required M2/context-only package fields for `IPV_removed`:

- `prediction_package_id`.
- `prediction_version_hash` or SHA-256 hash over model manifest and prediction files.
- RQ009 run ID and decision path.
- M2/context-only model ID and version.
- Model/config hash.
- Prediction-generation code hash or git commit.
- Exact same cell keys, frame/segment keys, timestamp grid, opportunity IDs, support fields, and abstention fields as the accepted M3 package.
- Outcome-blind generation attestation matching the phase-3 gate.

If the frozen M2 package is absent, key-incompatible, or not outcome-blind, the `IPV_removed` control is unavailable and no positive RQ011B primary claim is allowed.

## 11. Falsifiable PASS/FAIL Decision Rules

A positive RQ011B primary result requires all four gates jointly:

0. Post-abstention support floor passes.
1. Primary score increment passes effect and uncertainty gates.
2. True M3 increment beats every decision-gate negative control on common rows.
3. LOSO scenario-transfer generalization passes fold, aggregate, and influence gates.

Failure of any gate forces a bounded/null decision with all evidence reported. A secondary endpoint cannot rescue a failed primary endpoint.

Gate 0: Support floor passes only if all Section 7 floors pass. Failure label: `UNDER_IDENTIFIED_OR_DEGENERATE`.

Gate 1: Effect and uncertainty passes only if all are true:

- Standardized M3 coefficient `beta_M3 <= -0.15`.
- The 95% two-way blocked wild-bootstrap CI for `beta_M3` excludes 0 on the expected adverse side: upper bound `< 0`.
- `partial_R2 >= tau = 0.03`.
- The 95% two-way blocked wild-bootstrap CI lower bound for `partial_R2` is `> 0`.

Gate 2: Beats decision-gate negative controls:

- Decision-gate controls are `role_flip`, `sign_flip`, `counterpart_swap`, `kinematics_only`, `IPV_removed`, `shuffled_ipv`, and `wrong-cell/wrong-envelope`.
- For each control `c`, define `E_c` as rows where true M3 term, control term, endpoint, and frozen baseline covariates are available.
- Refit true M3 and control models on identical `E_c`.
- If a control is unavailable for more than 10% of otherwise eligible primary rows, Gate 2 fails.
- `E_c` must retain all 15 scenarios, at least 16 teams, and at least 10 cells per scenario.
- For deterministic or single-realization controls, define `adversity = -beta`; larger values are more adverse in the expected M3 direction.
- Beta practical margin: `adversity_M3 - adversity_control >= delta_beta = 0.05`.
- Beta paired uncertainty: 95% paired two-way blocked bootstrap CI lower bound for `adversity_M3 - adversity_control` is `> 0`.
- Partial-R2 practical margin: `partial_R2_M3 - partial_R2_control >= delta_R2 = 0.01`.
- Partial-R2 paired uncertainty: 95% paired two-way blocked bootstrap CI lower bound for `partial_R2_M3 - partial_R2_control` is `> 0`.
- For stochastic controls, use the max-statistic permutation rule with `B_perm = 10000`, base seed `20260625`; Gate 2 passes only if max-statistic permutation `p < 0.05`.

No required decision-gate control may be silently omitted. A failed, degenerate, or missing control blocks a positive primary claim.

Gate 3: LOSO scenario-transfer generalization:

- LOSO folds: 15 folds, each holding out one shared scenario (`A1` to `C4`) and fitting on the other 14 scenarios.
- The transfer scoring model is separate from the primary scenario-FE model and contains no scenario fixed effects.
- Z-scaling parameters are learned on the 14 training scenarios only and applied unchanged to the held-out scenario.
- Held-out predictions use the training intercept and coefficients; no held-out scenario intercept or scenario fixed-effect coefficient is estimated or imputed.

Baseline transfer model:

```text
score_cell = intercept
             + ego_speed_at_opportunity_start_z
             + counterpart_speed_at_opportunity_start_z
             + log_initial_gap_at_opportunity_start_z
             + closing_rate_at_opportunity_start_z
             + replay_support_fraction_z
             + error
```

M3 transfer model adds `standardized m3_norm_deviation`.

Held-out statistic:

```text
DeltaQ2_s = 1 - SSE_M3_heldout_s / SSE_baseline_heldout_s
DeltaQ2_all = 1 - sum_s(SSE_M3_heldout_s) / sum_s(SSE_baseline_heldout_s)
```

Gate 3 passes only if all are true:

- Training-fold M3 coefficient is adverse (`beta_M3 < 0`) in at least `k = 12` of 15 folds.
- Held-out gain `DeltaQ2_s > 0` in at least `k = 12` of 15 folds.
- Aggregate held-out gain `DeltaQ2_all > 0`, with a 95% two-way team x scenario blocked wild-bootstrap CI lower bound above 0.
- No single held-out scenario contributes more than 40% of aggregate positive SSE reduction.
- After removing the held-out scenario with the largest positive SSE reduction, remaining aggregate held-out SSE reduction remains positive.

Decision labels:

- `SUPPORTED_PRIMARY`: all four primary gates pass; secondary results are reported hierarchically.
- `BOUNDED_NULL_PRIMARY`: one or more primary gates fail; report all evidence and do not claim robust IPV-specific incremental validity.
- `NON_SPECIFIC_SIGNAL`: Gate 1 passes but Gate 2 fails.
- `NON_GENERALIZING_SIGNAL`: Gate 1 passes but Gate 3 fails.
- `UNDER_IDENTIFIED_OR_DEGENERATE`: support, missingness, sparse endpoints, or control degeneracy prevents a valid confirmatory decision.

## 12. Secondary Endpoint Rules

Secondary endpoints are interpreted only after the primary positive decision passes all four gates.

- Deduction burden expected direction: positive; higher M3 deviation predicts higher `log1p(total_pdf_deduction_v2)`.
- Collision expected direction: positive; higher M3 deviation predicts higher collision odds.
- Apply Holm-Bonferroni across deduction and collision.
- Collision may use two predictors only if post-abstention collision events `>= 20` and non-events `>= 20`: M3 deviation plus `collision_baseline_risk_index`.
- If post-abstention events are 10-19 and non-events `>= 20`, use M3-only exact or blocked permutation inference, or report descriptively if infeasible.
- If post-abstention events are `< 10`, make no inferential collision claim.
- Rank is descriptive/team-blocked only; no 285-row inferential claim and no rescue of primary failure.

Collision baseline risk index:

```text
collision_baseline_risk_index =
  mean_z(ego_speed_at_opportunity_start,
         counterpart_speed_at_opportunity_start,
         closing_rate_at_opportunity_start,
         -log_initial_gap_at_opportunity_start)
```

Higher values mean faster, more closing, and smaller-gap baseline risk. Missing any component makes the index unavailable for that cell.

## 13. Phase-3 Entry Gate G1

Phase 3 may start only when all are true:

1. RQ009 has a frozen knowledge/report decision or equivalent PI-accepted freeze record.
2. The record identifies M3 as the frozen downstream model/package.
3. The record reports the M3-vs-M4 pivot comparison.
4. The record explicitly states downstream consumption is cleared for RQ011B.
5. A frozen M3 prediction package exists and passes interface checks.
6. A frozen RQ009 M2/context-only prediction package exists for the `IPV_removed` negative control and passes interface checks.
7. Non-estimable cells are flagged/abstained and are not forcibly imputed.
8. The package contains outcome-blind generation attestation for both M3 and M2/context-only predictions.
9. The package documents M3 target construction as post-anchor, non-overlapping, non-full-window, and not self-anchor based.
10. The package either documents the 0.05-radian deviation scale floor from target-IPV noise/scale or declares that the floor remains a PI-to-freeze item before prediction consumption.

If any item is absent, status is `PAUSED_PHASE3_GATE_NOT_CLEARED`. RQ011B must not reinterpret the M3-vs-M4 materiality decision locally. If RQ009 does not explicitly clear downstream consumption, the gate is not passed.

## 14. Prediction-Interface Requirements

Required M3 package-level fields:

- `prediction_package_id`.
- `prediction_version_hash` or SHA-256 hash over model manifest and prediction files.
- RQ009 run ID and decision path.
- M3 model ID and version.
- Model/config hash.
- Prediction-generation code hash or git commit.
- Generation timestamp.
- `m3_frozen = true`.
- `downstream_consumption_cleared = true`.
- M3-vs-M4 pivot status and PI clearance text/path.
- `outcome_blind_generation_attestation`, stating that RQ011B official outcomes, ranks, deductions, collisions, and success/failure labels were unavailable to model fitting, prediction, filtering, support rules, abstention rules, and package generation.
- `target_ipv_construction_attestation`, stating post-anchor, non-overlapping, non-full-window target construction.
- `self_anchor_exclusion_attestation`, stating ego self-anchor IPV was not used as an M3 input or target-overlapping feature.
- `deviation_scale_floor_documentation`, documenting the target-IPV noise/scale basis for any 0.05-radian floor, or declaring the floor unresolved for PI freeze before consumption.

Required cell keys:

- `cell_key`.
- `unit_composite_key`.
- `area_id`.
- `team_id`.
- `algorithm_id`.
- `scenario_id`.
- `case_id`.
- `run_id` or audited replay/session ID.
- `session_id` where applicable.

Keys must reconcile exactly to audited `cells_clean_285.csv`; `algorithm_id == team_id` and `case_id == scenario_id` must hold.

Required frame or segment keys, if predictions are frame-level:

- `frame_id` or normalized frame index.
- `timestamp_ms`.
- `ego_id`.
- `counterpart_id`.
- Opportunity/window identifier.
- `opportunity_start_timestamp_ms`.
- `anchor_timestamp_ms`.
- `target_window_start_timestamp_ms`.
- `target_window_end_timestamp_ms`.

Required prediction fields:

- M3 interval lower bound.
- M3 interval upper bound.
- M3 center or median, if available.
- Conformal radius or interval construction metadata.
- Target rolling IPV value used for deviation.
- Counterpart current IPV input used by M3.
- Support-valid flag.
- Opportunity flag.
- Estimability flag.
- Human-reference support flag.
- Abstention flag.
- Abstention reason.
- Guard output fields stored separately from norm-deviation fields.

Required cell-level summary fields:

- Number of frames/segments considered.
- Number of support-valid frames/segments.
- Estimability/support fraction.
- `m3_norm_deviation` or sufficient frame-level fields to compute it exactly under this SAP.
- Non-estimable/support/abstention status.

Required M2/context-only fields:

- `prediction_package_id`.
- `prediction_version_hash` or SHA-256 hash over model manifest and prediction files.
- RQ009 run ID and decision path.
- M2/context-only model ID and version.
- Model/config hash.
- Prediction-generation code hash or git commit.
- Generation timestamp.
- `m2_frozen = true`.
- Same cell keys, frame/segment keys, timestamp grid, opportunity IDs, support fields, and abstention fields as the accepted M3 package.
- Context-only interval lower bound.
- Context-only interval upper bound.
- Target rolling IPV value used for deviation, with the same target-construction attestation as M3.
- `outcome_blind_generation_attestation`.

Reject the M3 or M2 package for confirmatory use if target IPV is full-window, post-hoc, self-anchor based, or overlaps the anchor/input window; if M3 uses ego self-anchor IPV as input; if M3 uses denylisted variables; if opportunity/support/estimability/filtering/abstention rules were tuned with outcomes or M3-outcome association; if non-estimable/out-of-support cells are forcibly imputed; or if required keys, hashes, support flags, or abstention fields are absent.

## 15. RQ009-Supplied Dependency List

RQ011B phase 3 remains paused until RQ009 supplies:

1. Frozen M3 package and explicit downstream clearance for RQ011B after the M3-vs-M4 pivot.
2. Frozen M2/context-only package for `IPV_removed`, with matching keys, timestamps, support fields, abstention fields, and version hashes.
3. M3 target-construction attestation: post-anchor, non-overlapping, non-full-window, and not self-anchor based.
4. Outcome-blind generation attestation for M3 and M2, covering outcomes, ranks, deductions, collisions, fitting, prediction, filtering, support, abstention, and package generation.
5. Required interface fields: package IDs, hashes, model/config/code hashes, cell keys, frame/segment keys if applicable, opportunity IDs, support/estimability flags, abstention flags/reasons, and separate guard outputs.
6. Documentation supporting the 0.05-radian deviation scale floor, or explicit declaration that the floor is unresolved and must be frozen by the PI before predictions are consumed.

## 16. Required Phase 3+ Artifacts

Future phase 3+ execution must generate the following before any final claim package:

- Locked SAP and expanded denylist.
- Provenance audit and caveat ledger.
- RQ009 M3 prediction-package manifest and deviation table with support/abstention flags.
- RQ009 M2/context-only prediction package for `IPV_removed`, with matching keys and version hashes.
- Primary baseline and M3 model table.
- Support-floor and missingness ledger.
- Secondary endpoint table with multiplicity handling.
- Negative-control table covering decision-gate controls and the separate global label-permutation diagnostic.
- LOSO and LOTO robustness tables.
- Norm-vs-guard separation table.
- Independent review, red-team review, and independent replication records.
- Evidence rows for every final claim.
- Reader-facing report package.

## Acceptance Record

- Independent adversarial re-review status: **PASS**.
- Blocker closure: **7/7 closed**; no new blocking hole found.
- Review evidence:
  - `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/01_plan_review/SAP_v2_review/sap_v2_review.md`
  - `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/01_plan_review/SAP_v2_review/pi_final_brief.md`
- PI accepted recommended configuration on 2026-06-25:
  - beta floor `0.15`;
  - `delta_R2 = 0.01`;
  - bootstrap `B = 10000`, seed `20260625`;
  - permutations `B_perm = 10000`, base seed `20260625`;
  - conditional 0.05-radian deviation floor rule.
- Gate `G2` is resolved by this locked SAP.
- Gate `G1` remains pending on RQ009 downstream clearance and frozen M3/M2 packages.
