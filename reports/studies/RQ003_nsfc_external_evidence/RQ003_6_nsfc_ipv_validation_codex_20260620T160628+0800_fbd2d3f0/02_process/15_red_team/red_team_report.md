# Phase 7 Independent Red Team Report

Worker: `RQ003_phase7_red_team_001`
Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`
Generated: `2026-06-20T22:09:15+0800`

## Verdict

Status: **BLOCKERS_FOUND**.

The computed Phase 4 result is internally coherent under its current structural scenario map, but the analysis cannot be accepted as the governing RQ003 null/reverse result because the structural scenario labels used for fixed effects, A1 exclusion, and leave-one-scenario analysis disagree with the official score-map scenario labels in 120 / 150 cells. The mismatch moves the two zero-score official A1 cells into structural `C2`, makes the confirmatory report's A1/safety statements false, and changes a scratch leave-one-team recomputation from null/reverse to directionally favorable when official scenario labels are used for residualization and inclusion.

The NULL is therefore **not robust enough to call final**. The honest current statement is: under one structural scenario relabeling, IPV adds no detected utility over the baseline, but that result is blocked pending a scenario-label reconciliation and full rerun.

## Blocking Findings

### RT-BLOCK-001: Scenario-label mismatch invalidates scenario fixed effects, A1 exclusion, and LOSO interpretation

Severity: **BLOCKING**

Evidence:

- Direct crosswalk of `scenario_map_outcome_free.csv` to `replay_score_mapping.csv` by `team, area, case_id` found 30 / 150 scenario-label matches and 120 / 150 mismatches.
- The mismatch is symmetric across areas: 60 Beijing mismatches and 60 Shanghai mismatches.
- Example: `T15_C2_task6924_case2333` is structural `C2` but official-score scenario `A1` (`12-...pedestrian crossing` in source name), with safety=0, efficiency=0, coordination=0.
- The current Phase 4 model residualizes by structural scenario and excludes structural `A1`; a scratch spot recomputation using official scenario labels for inclusion and fixed effects changed primary LOTO:
  - Current structural labels: N=48, baseline Spearman=0.354, full=0.244, delta=-0.110; MAE reduction=-0.739.
  - Official-label spot check: N=53, baseline Spearman=-0.191, full=-0.054, delta=+0.137; MAE reduction=+0.479.
- This is not a replacement analysis, but it proves the headline null/reverse is label-sensitive.

Recommendation:

Freeze a verified scenario crosswalk that reconciles structural case order, case names, and official scenario codes without outcome leakage. Rerun Phase 4, negative controls, state-dependence, A1/collision handling, and independent replication using one consistent label system.

### RT-BLOCK-002: A1 zero-score veto handling and report language are false under the actual joined tables

Severity: **BLOCKING**

Evidence:

- `confirmatory_analysis_report.md` states that the only non-100 safety rows are two zero-safety A1 rows already excluded by the non-A1 rule.
- Recomputed assembled Phase 4 frame shows all structural A1 rows have safety=100; the two safety=0 rows are `T15_C2_task6924_case2333` and `T20_C2_task6941_case2333`.
- Those two rows are official scenario `A1` in `replay_score_mapping.csv`, but structural `C2` in `scenario_map_outcome_free.csv`.
- The same false statement is hardcoded in `run_confirmatory_analysis.py` report text and repeated in `worker_report.json`.

Recommendation:

After the scenario crosswalk fix, regenerate all primary-sample and safe-subset statements. Explicitly report the zero-score/catastrophic rows by both official scenario and structural case identity until the crosswalk is resolved.

## Attack Surface Review

### 1. D_comp / D_yield sign and historical one-sided cost sign bug

Finding: **No current sign blocker.**

Evidence:

- `ipv_sign_contract.md` and `real_optimizer_unit_tests.csv` pass the real-theta sign checks: theta > 0 is prosocial; theta < 0 is competitive.
- Formula spot check on 14,127 estimated conflict-window frame rows found max absolute recomputation error of `8.9e-16` for `D_comp=max(0,(q_low-theta)/w)` and `1.3e-15` for `D_yield=max(0,(theta-q_high)/w)`.
- Current features therefore do not reproduce the historical reversed one-sided cost bug.

Severity: **NON-BLOCKING**

Recommendation: Preserve the dual-tail formulas and keep sign tests in any rerun.

### 2. Future leakage in features, baseline, residualization, or CV

Finding: **No direct future-leak blocker found, but rerun after RT-BLOCK-001 is required.**

Evidence:

- `compute_phase4_features.py` calls `estimate_ipv_current` on rolling prefixes `start:i`.
- `baseline_feature_manifest.csv` lists current/past kinematic sources; no official scores/ranks appear in baseline features.
- Confirmatory code fits residualization, imputation, scaling, and alpha inside training folds.
- The previously invalid future-leaky negative control was repaired; feature health reports nonconstant features and it is explicitly excluded from NULL-robustness claims.

Severity: **NON-BLOCKING**

Recommendation: Repeat leakage checks after scenario crosswalk rerun.

### 3. Rolling vs full-window mixing

Finding: **No primary mixing found.**

Evidence:

- Primary model uses `D_comp_auc` and `D_yield_auc` from high-support rolling frames.
- Full/future-window IPV appears only in `future_leaky_full_window_ipv`, labelled non-deployable and excluded from robustness claims.

Severity: **NON-BLOCKING**

Recommendation: Keep future-inclusive diagnostics separate from deployable/primary claims.

### 4. Is IPV merely a recoding of speed / TTC / stop?

Finding: **Not a simple recoding, but overlap with kinematics remains a real interpretation boundary.**

Evidence:

- On the current primary sample, the largest absolute Spearman correlation between an IPV tail and a baseline kinematic feature was 0.334 (`D_yield_auc` vs lateral gap AUC).
- Selected correlations for `D_sum_auc`: ego speed -0.227, relative speed -0.219, TTC risk -0.093, lateral-gap risk 0.161, conflict duration 0.206, estimated frames 0.218.
- `D_comp_auc` and `D_yield_auc` are weakly negatively related (Spearman -0.145), so the two tails are not collapsed into one speed/stop proxy.

Severity: **NON-BLOCKING**

Recommendation: Do not claim IPV is independent information beyond trajectories. Phrase it as incremental utility relative to the prespecified baseline.

### 5. Is coordination circularly defined by efficiency / scenario difficulty / scoring formula?

Finding: **Criterion-source circularity risk remains.**

Evidence:

- Gate -1 found coordination is an official generated/report metric with background-traffic safety/efficiency/comfort components, not proven expert rating.
- Recomputed correlations show coordination tracks comprehensive score (Spearman 0.596 in top-five), efficiency (0.425), and kinematic heading difference (-0.420).
- This does not invalidate a prediction task against the official score, but it weakens any claim that the outcome is an independent social-ground-truth criterion.

Severity: **NON-BLOCKING for official-score prediction; BLOCKING for any expert/human criterion-validity wording.**

Recommendation: Use only "official generated coordination score" unless stronger source evidence is found.

### 6. Team / scenario / area nesting

Finding: **Team and area handling are structurally explicit, but scenario nesting is blocked by RT-BLOCK-001.**

Evidence:

- Fold assignment checks found no train/test group overlap within structural LOTO/LOSO/LOFO folds.
- Primary structural sample has 48 cells, 9 teams, and 14 structural scenarios; T8 has no primary held-out cell.
- Because 120 / 150 scenario labels mismatch the official score-map labels, scenario fixed effects and LOSO folds are not trustworthy until scenario identity is reconciled.

Severity: **BLOCKING**

Recommendation: Rebuild folds and fixed-effect labels from the corrected scenario map.

### 7. Frame-level pseudoreplication

Finding: **No pseudoreplication blocker.**

Evidence:

- 16,771 frame rows are aggregated to 150 cell-level feature rows.
- Primary LOTO predictions contain 48 rows and 48 unique cells, with no duplicate prediction cells.

Severity: **NON-BLOCKING**

Recommendation: Keep all inferential claims at cell/fold level.

### 8. A1 zero-score veto handling

Finding: **BLOCKING; see RT-BLOCK-002.**

Evidence:

- The actual joined structural frame places zero-safety official A1 cases in structural C2.
- Current report statements saying the low-safety rows are structural A1 are false.

Severity: **BLOCKING**

Recommendation: Fix scenario crosswalk, then recompute A1/collision-free exclusions and safe subsets.

### 9. Post-treatment NPC matching

Finding: **No causal overclaim found.**

Evidence:

- `npc_feasibility_and_boundary.md` concludes pre-onset matching is not identifiable and no NPC effect analysis was run.
- It forbids realized NPC trajectory matching and permits only future "matched opportunity structure" wording if missing fields are later obtained.

Severity: **NON-BLOCKING**

Recommendation: Keep NPC analysis out of causal claims.

### 10. Outcome-selected window / threshold / sample

Finding: **No direct outcome-tuned threshold found, but scenario-map construction is a blocker.**

Evidence:

- Freeze artifacts report outcome-clean thresholds and a single confirmatory comparison.
- The scenario map was built outcome-free by sort order, but that rule conflicts with official scenario labels; the issue is not outcome tuning but wrong scenario identity.

Severity: **BLOCKING via RT-BLOCK-001**

Recommendation: Rerun only after a corrected crosswalk is frozen before outcome modeling.

### 11. Low-support forced violation judgments

Finding: **No blocker found.**

Evidence:

- Primary uses high-support cells only; fallback-inclusive columns are suffixed and non-confirmatory.
- State-dependence report labels low-support/fallback-heavy rows as abstention or boundary, not validation.

Severity: **NON-BLOCKING**

Recommendation: Continue to abstain on low-support/fallback-heavy contexts.

### 12. Non-exchangeability and conformal coverage

Finding: **No wrongful nominal coverage claim found.**

Evidence:

- Freeze and Gate 0 documents explicitly forbid nominal NSFC conformal coverage claims.
- Narrative-facing `00_entry`/`90_report` are placeholders and do not assert coverage.

Severity: **NON-BLOCKING**

Recommendation: Keep NSFC coverage empirical/OOD only.

### 13. Multiple comparisons / cherry-picking

Finding: **No FDR promotion blocker found.**

Evidence:

- `state_dependence_results.csv` has 97 interpretable rows.
- Favorable local-alignment rows have no `q <= 0.10`; minimum favorable q was 0.508.
- The only `q <= 0.10` row was reverse/context-mismatch: A1 fallback-inclusive rho=0.936, q=0.0065, labelled abstain.

Severity: **NON-BLOCKING**

Recommendation: Keep state-dependence as exploratory boundary mapping.

### 14. Leave-one-scenario mechanical overlap

Finding: **Structural LOSO has no group overlap, but LOSO is blocked by wrong scenario labels.**

Evidence:

- Fold checks found no train/test structural scenario overlap.
- Because official and structural scenario labels mismatch in 120 / 150 cells, LOSO does not test official scenario generalization.

Severity: **BLOCKING via RT-BLOCK-001**

Recommendation: Rebuild LOSO after scenario crosswalk fix.

### 15. Train / held-out contamination

Finding: **No direct train/held-out contamination found.**

Evidence:

- `run_confirmatory_analysis.py` fits residualization and preprocessing inside each training fold.
- Independent stats review reported held-out outcome perturbation changed predictions by 0.
- Feature tables contain no coordination/rank/residual predictor columns.

Severity: **NON-BLOCKING**

Recommendation: Repeat contamination check after rerun.

### 16. Prior exploration mislabeled as confirmatory

Finding: **No blocker found.**

Evidence:

- `plan_snapshot.md`, `exploratory_prior_manifest.md`, `tried.md`, and `claims_register.md` identify prior values such as signed approximately 0.255 as exploratory context.
- Confirmatory family contains one comparison.

Severity: **NON-BLOCKING**

Recommendation: Preserve exploratory labels.

### 17. Missingness / selection bias

Finding: **Selection limits are material.**

Evidence:

- Top-five cohort has no replay-mapping missingness, but full 20-team universe is nonrandom and not analysis-ready.
- Current structural primary sample is 48 / 150 cells; T8 has no primary cells.
- Included cells have higher mean coordination (83.93) and efficiency (83.98) than excluded cells (coordination 81.23, efficiency 77.95), though primary-inclusion vs coordination Spearman is small (0.077).

Severity: **NON-BLOCKING limitation; can become blocking if generalized beyond the scoped high-support top-five cohort.**

Recommendation: State the estimand as top-five, mapped, high-support, non-A1/collision-free cells only.

### 18. Narrative exceeding evidence

Finding: **No final overclaim exists yet, but Phase 4 process narrative contains blocking false statements.**

Evidence:

- `00_entry/index.html` and `90_report/index.html` remain placeholder pages saying no formal statistics have been run.
- `confirmatory_analysis_report.md` is cautious about null/reverse and power, but contains false A1/safety assertions tied to RT-BLOCK-002.

Severity: **BLOCKING for Phase 4 report text; NON-BLOCKING for final reader report because it is not built.**

Recommendation: Do not build a final report until blockers are fixed.

### 19. Is the NULL itself an artifact?

Finding: **Yes, the current NULL/REVERSE may be an artifact of scenario-label handling.**

Evidence:

- Baseline signal is not broken under the current structural labels: baseline Spearman=0.354, and state-shuffle/wrong-state controls degrade it.
- However, official-label spot recomputation changes primary sample and model behavior substantially: N=53, baseline Spearman=-0.191, full=-0.054, delta=+0.137, MAE reduction=+0.479.
- This does not establish a positive IPV finding, but it is enough to block the current null/reverse headline.

Severity: **BLOCKING**

Recommendation: Treat the null as provisional until the corrected rerun and independent replication are complete.

### 20. Solver preset and N=48 power

Finding: **Material limitation, not the main blocker.**

Evidence:

- Feature worker used `balanced` SLSQP after prior environment constraints; sign reconfirmation passed.
- Current primary sample has 48 cells and uneven team coverage; a corrected scenario-label variant has 53 cells.
- Existing report correctly calls power limited.

Severity: **NON-BLOCKING limitation.**

Recommendation: Keep wording as "power-limited no detected incremental utility" if the corrected rerun remains null.

## Required Fix Summary

1. Reconcile scenario identity before any further interpretation.
2. Recompute all Phase 4/6 outputs after the scenario fix.
3. Regenerate A1, collision, safe-subset, LOSO, and state-dependence text.
4. Re-run independent replication and red team on the corrected package.
5. Only then decide Tier placement.

## Bottom Line

The current package is red-team complete, but the analysis is not ready for Tier review. The primary null/reverse result is honest for the current structural-label computation, but it is not robust to a plausible and source-backed scenario-label correction. This is a blocking analysis-integrity issue, not a cosmetic reporting issue.
