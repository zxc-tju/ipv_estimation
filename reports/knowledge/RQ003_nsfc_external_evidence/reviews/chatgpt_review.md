# ChatGPT Review — RQ003_6 NSFC IPV External Evidence

- Reviewer: ChatGPT (GPT-5.5 Pro)
- Review date: 2026-06-21
- Execution reviewed: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`
- Review status: **PASS WITH TIER B BOUNDARY**
- Recommended paper role: **external feasibility, diagnostic alignment, transfer-boundary and reproducibility evidence; not formal verifier validation**

## 1. Overall verdict

RQ003_6 is a well-audited external stress test of the directional IPV pipeline, but it does not establish robust IPV-specific criterion validity. The strongest paper-relevant contribution is not predictive superiority. It is the demonstration that an InterHub-calibrated, state- and counterpart-conditioned IPV diagnostic can be transferred to an independent real-vehicle challenge workflow with explicit provenance checks, online-measurement contracts, support/OOD abstention, red-team review and independent replication.

The settled Tier B interpretation is appropriate:

> No robust incremental predictive utility relative to the prespecified kinematic+safety baseline was demonstrated in the power-limited top-five cohort. The apparent favourable direction was reproducible, but nonsignificant, non-generalizing across held-out scenarios, and not IPV-specific under the negative controls.

This execution is useful to the manuscript if it is presented as a bounded external feasibility and diagnostic-alignment result, not as successful formal validation.

## 2. Evidence favourable to the paper

### 2.1 The external validation pipeline is operational and auditable

For the approved top-five cohort, Gate -1 reports clean usable mappings for all 150 planned `team × scenario` cells. The full 20-team universe remains outside the analysis-ready scope, but the approved cohort has a traceable replay/outcome mapping.

Gate 0 passed all 13 unit tests and the key deployment contracts:

- correct competitive-shortfall and over-yielding orientation;
- frozen human conditional norm;
- the same estimator contract across domains;
- rolling-to-rolling comparison;
- no future leakage;
- support/OOD/abstention accounting;
- conformal calibration restricted to InterHub;
- no NSFC-outcome tuning;
- separation of empirical verifier and policy guard;
- no renormalization against the competitor distribution.

This supports the manuscript claim that the verifier architecture is auditable and deployable as a monitored pipeline, even when transfer validity is incomplete.

Primary evidence:

- `02_process/02_gate_minus1/gate_minus1_status.json`
- `02_process/04_gate0_measurement/gate0_status.json`
- `01_results/figures/fig01_provenance_coverage.*`
- `01_results/figures/fig03_support_ood_abstention.*`

### 2.2 Directional IPV produced a coherent favourable numerical trend

In the corrected leave-one-team-out primary analysis (`N=53`, 9 teams, 14 scenarios), adding `D_comp_auc` and `D_yield_auc` to the prespecified state + causal-kinematics + safety baseline improved all three reported metrics numerically:

- delta Spearman: `+0.1368`, 95% interval `[-0.0388, 0.3058]`, `p=0.30`;
- MAE reduction: `+0.4793`, from `7.4165` to `6.9372`, interval `[0.0635, 0.8819]`, one-sided `p=0.07`;
- delta CV-R²: `+0.0862`, interval `[-0.0047, 0.2463]`, `p=0.13`.

The coherent direction across rank correlation, absolute error and CV-R² is useful hypothesis-supporting evidence. It is not statistically decisive and must not be described as a robust increment.

Primary evidence:

- `01_results/tables/confirmatory_results.csv`
- `01_results/figures/fig05_scenario_fix_before_after.*`

### 2.3 Independent replication preserved the favourable direction

The reported-alpha refit reproduced the corrected result essentially exactly. A separately tuned independent implementation also retained favourable directions:

- delta Spearman: `+0.0547`;
- MAE reduction: `+0.4014`;
- delta CV-R²: `+0.0681`.

This is valuable implementation evidence. It reduces concern that the favourable trend is a one-off coding artifact, although it does not resolve significance, generalization or specificity.

Primary evidence:

- `02_process/17_independent_replication/replication2/replication2_status.json`
- `01_results/figures/fig08_independent_replication.*`

### 2.4 The two-tail IPV interpretation is operationally consistent

RQ003_6 gives the paper a defensible directional contract:

- `D_comp` marks below-norm competitive shortfall;
- `D_yield` marks above-norm yielding excess;
- both are defined relative to the InterHub human conditional norm.

This directly supports replacing an ambiguous single signed deviation and one-sided trigger with two interpretable diagnostic channels.

Primary evidence:

- `02_process/04_gate0_measurement/ipv_sign_contract.md`
- `02_process/08_directional_ipv/g0r_cond_001_status.json`
- `01_results/figures/fig04_directional_ipv_signature.*`

### 2.5 Abstention is a meaningful cross-domain result

Across 14,127 estimated conflict frames, only 1,850 ego frames were high-support and 12,277 were assigned to abstention; optimizer-error frames were zero. This is not a coverage success, but it supports a useful runtime-verification property: under domain shift the system can expose insufficient support rather than emit uniformly strong social-compliance judgements.

This result is paper-useful only when framed as a transfer boundary and quality-gating demonstration, not as broad NSFC coverage.

Primary evidence:

- `01_results/figures/fig03_support_ood_abstention_source.csv`

## 3. Findings that block a stronger claim

### 3.1 The apparent increment is not IPV-specific

Several negative controls matched or exceeded the primary delta Spearman, including future-leaky full-window IPV, IPV time shuffle, counterpart swap, role flip and sign flip. Degradation controls also failed to behave as clean diagnostics. Therefore the favourable primary direction cannot be attributed specifically to the intended directional IPV mechanism.

Primary evidence:

- `01_results/tables/negative_controls.csv`
- `01_results/figures/fig06_negative_controls.*`
- `02_process/16_red_team_fixes/red_team3/red_team3_status.json`

### 3.2 New-scenario generalization was not demonstrated

Leave-one-scenario-out performance was approximately null:

- delta Spearman: `+0.0167`;
- interval `[-0.1781, 0.1645]`;
- MAE and CV-R² changes were null or adverse.

RQ003_6 therefore does not establish transfer to unseen scenarios.

### 3.3 Behavioural mechanism validation remains blocked

H3 is blocked because the package contains no real two-human blind annotation result. No inter-rater agreement or event–IPV test is available. Consequently, the report cannot claim that `D_comp` identifies aggressive intrusion or that `D_yield` identifies freezing/over-yielding in independently labelled behaviour.

### 3.4 No stable favourable state stratum was found

The exploratory state-dependence analysis produced no interpretable row with `q <= 0.10`. State dependence did not rescue the primary NSFC result.

### 3.5 The outcome is not demonstrated to be expert-rated social judgement

The provenance audit supports describing coordination as an official/generated report score or rule-kinematic report metric. Human expert coordination rating, common Beijing/Shanghai judges and a shared formal rubric were not established. Manuscript wording must not call this outcome expert-rated coordination.

### 3.6 Scope is narrow

The confirmatory result is restricted to a power-limited top-five cohort and only 53 primary cells. The full 20-team universe is not analysis-ready. Safe-subset robustness was also not established: S1 and S2 duplicate the primary sample, while S3 has only six cells and is null/reverse.

## 4. Recommended manuscript use

### Main-text-safe conclusion

A bounded sentence may be used in Results or Discussion:

> In a power-limited top-five real-vehicle challenge cohort, adding state- and counterpart-conditioned directional IPV summaries produced coherent but nonsignificant improvements over a prespecified kinematic and safety baseline under leave-one-team-out evaluation. The favourable direction was independently reproduced, but was not IPV-specific and did not generalize across held-out scenarios; we therefore interpret the challenge as an external feasibility and diagnostic-alignment test rather than definitive criterion validation.

### Recommended placement

- **Main Results:** at most one bounded sentence plus a reference to Extended Data.
- **Extended Data:** primary LOTO estimates, LOSO null result, support/OOD/abstention, negative controls and independent replication.
- **Discussion:** domain-transfer limits, outcome provenance, lack of blind labels and the need for matched behavioural criterion data.
- **Abstract/title:** do not use RQ003_6 as evidence of successful external validation or predictive superiority.

## 5. Claims classification

| Claim | Review decision |
|---|---|
| The verifier can be executed and audited on an independent challenge workflow | **Supported within the top-five cohort** |
| Directional IPV has a weak favourable association with the official coordination score | **Supported descriptively; nonsignificant** |
| The favourable direction is reproducible across implementations | **Supported** |
| IPV adds robust incremental predictive value beyond kinematics+safety | **Rejected for RQ003_6** |
| The increment is IPV-specific | **Rejected** |
| The result generalizes to unseen scenarios | **Rejected** |
| Blind behavioural consequences validate `D_comp` / `D_yield` | **Blocked** |
| The official coordination target is expert-rated social judgement | **Not established** |
| The transferred human envelope has broad NSFC support | **Rejected; high abstention / narrow support** |

## 6. Final recommendation

Retain RQ003_6 as **Tier B external diagnostic-alignment and transfer-boundary evidence**. It strengthens the paper's methodological credibility, quality-gating story, directional-sign correction and reproducibility record. It should not be used to claim formal external validation, robust incremental utility, unseen-scenario generalization or an independently verified social mechanism.
