# Independent source audit: Figures 2, 5 and 6

Date: 2026-09-25. Status: **PASS for the source-data and statistical-display scope below**.

This figure-revision task reorganizes the existing manuscript evidence into six main figures. This independent, single-pass check covers the completed Figure 2, Figure 5 and Figure 6 exports and their supplied aggregate sources. It checks numerical correspondence, denominators, estimands and interval provenance; it is not visual QA, a manuscript-wide review, or a new validation of empirical provenance.

## Scope and method

- Read `input_plan.md`, the three plotting scripts and their source maps/notes. Independently compared source-data CSVs with the referenced frozen CSV/JSON inputs using Python assertions and ordinary arithmetic. Numerical comparisons allowed only floating-point serialization tolerance; no estimates, tests, intervals or bootstrap samples were regenerated.
- The Figure 2 joint-density check repeated only the displayed 48 × 48 bin aggregation of the supplied real case values. It did not synthesize points, smooth a density or perform statistical estimation.
- Verified all recorded source SHA-256 values in the source-map snapshot: 8 Figure 2 records, 7 combined Figure 4/5 map records and 25 Figure 6 records. Hash agreement is an input-integrity check, not scientific validation of Figure 4.
- Subjective reads stayed within `data/manually_organized/RQ028_v1_主观评分_20260914/论文可用结果提取_20260925/`. Did not read individual/pair rating rows, other subjective versions, RQ007 held-out data or RQ014 blinded rating data. No frozen source was edited.

All run-relative paths below refer to `reports/knowledge/PAPER001_online_sociality_verification_manuscript/figure_revision_20260925/`.

## Figure 2

- `source_data/fig2a_joint_distribution_bins.csv`: all 2,304 bins match the real case-source aggregation. The source has 34,850 rows and 34,850 unique `source_row` values. Symmetrization yields 69,700 display points; the independent-case count remains 34,850.
- `source_data/fig2b_complementarity.csv`: all point estimates and supplied interval endpoints match `dataset_dyad_metrics.csv` and `round3_c1_static_implementation_comparison.csv`. Source counts are nuPlan 7,330, AV2 2,289, Lyft 4,738 and Waymo 20,493, summing to 34,850. The primary pooled matched-support estimate uses 34,757 cases; the independent implementation uses 34,645. The independent observed interval is absent and stays absent. Matched permutation-null intervals are kept distinct from bootstrap CIs.
- `source_data/fig2c_early_role_auc.csv`: all 18 rows and fields match the frozen AUC table, including the combined model retained as source data. Six plotted windows span 24,872–31,831 cases. The source implementation defines the target from `progress >= 0.75` and the sign of `final_ipv1_mean - final_ipv2_mean`; it is not measured passage order. Source code uses grouped out-of-fold logistic predictions and 300 scene-cluster bootstrap draws. Definition anchors: `agent_dynamics/code/analyze_dynamics_round2.py:59,732,776–778,783–812,939–955` within the accepted RQ004 second study process directory.
- `source_data/fig2d_turn_straight.csv`: all five displayed sample counts, differences and interval endpoints match the overall and four-source frozen tables. The nuPlan interval includes zero and is retained.
- `source_data/fig2e_priority_pet.csv`: all three ALL-source rows and fields match the frozen PET-stratified table; counts remain 9,799, 22,833 and 4,863. The existing intervals are reused and PET remains an offline descriptive stratum.
- `source_data/fig2f_geometry.csv`: all eight selected source-specific rows and displayed fields match the `primary_contrast`, `scope=dataset` frozen rows for `MP_minus_nonMP` and `SS_minus_nonSS`. No pooled geometry effect was created.

## Figure 5

- `source_data/fig5_rates.csv`: all 18 arm × coverage × side rows match the source `flag_counts` numerators and denominators. At the main 90% working point, human assertive/accommodating counts are 435/15,598 and 351/15,598; AV counts are 519/14,099 and 869/14,099. Total outside-range counts are 786/15,598 and 1,388/14,099, respectively. Direction-specific absent intervals remain blank.
- The human total-rate CI is the supplied 3.4–5.5%, now exported in the human/90/outside row, with the existing driver-by-scenario resampling metadata, B=1,000. This corresponds to `reports/knowledge/RQ022_matched_scenario_human_arm/decision.md:29–30` and `human_arm_data.json.flag_rate_ci95_alpha90`.
- `source_data/fig5_rate_ratio.csv` preserves the reported 1.95 [1.62, 2.39] AV/human ratio and its separate scenario-cluster bootstrap metadata (20,000 draws, seed 20260820). The source is the accepted paper handoff at `imported_from_paper_repo_20260620/agent_handoff.md:1101–1103`; the interval was not reconstructed from arm-level intervals.
- `source_data/fig5_scenarios.csv`: all 15 scenario rows match both source JSONs, including supplied rates and counts. AV rate exceeds human rate in 15 of these 15 matched scenarios. This is an observed set-level count, not a claim about future scenarios.
- `source_data/fig5_consequence_signature.csv`: all 12 ratios match the supplied aggregate values. AV ego median/upper-quartile ratio intervals match `ego_three_second_window.json`; both arms' counterpart speed ratio intervals match their source JSONs. Human ego ratio intervals and both arms' emergency-tail ratio intervals remain unavailable. Existing share-difference intervals were not converted to ratio intervals.
- The human source remains an accepted `REAL_VERIFIED` aggregate. This check does not establish that the unavailable human row-level archive is present or independently recomputed here.

## Figure 6 and subjective sensitivity

- The nine complete `source_data/fig6_*.csv` copies corresponding to rating descriptives, paired contrasts, paper-core results, design counts, common-scenario source/summary, alternative-code sensitivity, source-label sensitivity and scenario/class coverage are byte-identical to the supplied final-package tables.
- `source_data/fig6_plotted_means.csv`: all 15 displayed rows/fields match the source. The plotting code uses `subject_equal_mean`, not `raw_mean`. All rows retain 40 participants, 20 pairs and the supplied 10,000 pair-cluster percentile-bootstrap interval method. `q5_cost_cp` has only counterpart rows; no ego value or zero was inserted.
- `source_data/fig6_participant_paired_contrasts.csv` contains 30 distinct seat/metric/contrast rows with all 30 `p_holm_30` values. All eight `paper_core_results` rows agree with the corresponding complete-table estimate, CI endpoints and adjusted p value. Ego atypicality A−W retains Holm p=0.07038382684.
- Source weighting/methodology was checked in the supplied `process/extract_results.py:116–159`: participant category means precede pair aggregation; within-participant contrasts are formed before pair resampling; the complete 30-comparison family is corrected together. No Holm adjustment or test was rerun by this audit.
- Both supplementary display exports contain the correct nine counterpart rows: `figS2_equal_participant_pair_bootstrap.csv` retains pair resampling, while `figS2_equal_scenario_scenario_bootstrap.csv` retains scenario resampling with fixed observed raters. They are not described as the same estimand or independent replications.
- All 30 common-scenario summary rows have positive/negative/zero counts matching `common_scenario_source.csv`. Main counterpart contrasts use 8 common scenarios for A−W, 9 for C−W and 12 for A−C. In particular, comfort has 8/8 negative A−W and 12/12 negative A−C differences; cost has 8/8 positive A−W and 12/12 positive A−C differences. Atypicality A−W is 6 positive and 2 negative, not 8/8.
- Design-table checks reproduce 1,192 position-rating records; counterpart-only summation gives 596 completed trials and 90 segments. The scenario/class coverage table has 15 unique scenarios. Participant and pair counts are shared counts, not summed across design cells.

## Resolved display issue and limits

The only issue found was presentation precision for counterpart Q1-reversed C−W: the frozen estimate is exactly `0.6825`, whereas default three-decimal half-even formatting printed `+0.682`. The leader updated the supplement formatter to preserve `+0.6825`. The closing check confirmed `process/make_supplement_tables.py:10–12` implements this and the regenerated `assets/subjective_supplement_tables.tex:35` prints `CP & Atypicality & C--W & $+0.6825 ...$`. No source value changed.

No unresolved numerical-source blocker was identified for the checked figures. The source/statistical-display scope passes. Figure aesthetics, full-paper panel callouts, compilation and visual readability remain the leader's separate acceptance checks. Missing questionnaire/protocol documentation and unavailable human row-level archives are not resolved by correct figure export.
