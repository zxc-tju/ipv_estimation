# IPV Analysis Handoff (for Next Agent)

Last updated: 2026-03-07  
Owner context: multiple rounds of alignment, filtering, Interhub core analysis, Argoverse re-analysis, and cross-dataset comparison.

## 1. Scope of What Has Been Done

This handoff summarizes completed work across:
- JSON/CSV alignment checks for Interhub outputs.
- Regeneration of a strict JSON-matchable CSV.
- Filling `mean_ipv` for the two matched vehicles into CSV.
- Interhub core sociality analysis notebook (PCS/PVS, AV vs HV, PET/APET).
- Sign-convention correction (`IPV > 0` cooperative, `IPV < 0` competitive).
- Direct Argoverse re-analysis from xlsx results under the same sign convention.
- Cross-dataset comparison table generation (Argoverse vs Interhub).
- Code-level comparison of IPV estimation pipelines (`process_argoverse.py` vs `process_interhub.py`).

## 2. Key Interpretation Convention (Important)

Use this convention consistently in all text/conclusions:
- `IPV > 0`: cooperative tendency
- `IPV < 0`: competitive tendency
- `IPV = 0`: neutral tendency

This convention was explicitly re-applied and notebook outputs were refreshed after correction.

## 3. Canonical Data and Output Paths

### 3.1 Interhub inputs / aligned CSV
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\interhub_traj_lane\intersection_results_pass_only.csv`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\interhub_traj_lane\intersection_results_pass_only_json_matched.csv`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\interhub_traj_lane\intersection_results_pass_only_json_matched_with_mean_ipv.csv`

### 3.2 Interhub notebook + exported analysis
- Notebook:
  - `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\interhub_traj_lane\paper_plot_interhub.ipynb`
  - `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\interhub_traj_lane\paper_plot_interhub.executed.ipynb`
- Outputs:
  - `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\interhub_traj_lane\analysis_outputs\interhub_core\interhub_analysis_ready.csv`
  - `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\interhub_traj_lane\analysis_outputs\interhub_core\sample_quality_report.csv`
  - `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\interhub_traj_lane\analysis_outputs\interhub_core\summary_stats.csv`
  - `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\interhub_traj_lane\analysis_outputs\interhub_core\hypothesis_tests.csv`
  - `figA/figB/figC` png+pdf in same directory.

### 3.3 Argoverse re-analysis outputs
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\argoverse\analysis_outputs\argoverse_reanalysis\file_ingestion_stats.csv`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\argoverse\analysis_outputs\argoverse_reanalysis\argoverse_ipv_reanalysis_summary.csv`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\argoverse\analysis_outputs\argoverse_reanalysis\argoverse_ipv_reanalysis_tests.csv`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\argoverse\analysis_outputs\argoverse_reanalysis\argoverse_sign_convention_interpretation.csv`

### 3.4 Cross-dataset comparison outputs
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\analysis_outputs\cross_dataset\argoverse_vs_interhub_comparison_long.csv`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\analysis_outputs\cross_dataset\argoverse_vs_interhub_A_role_wide.csv`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\analysis_outputs\cross_dataset\argoverse_vs_interhub_B_role_wide.csv`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\analysis_outputs\cross_dataset\argoverse_vs_interhub_comparison.md`

## 4. Interhub JSON/CSV Alignment Work (Completed)

## 4.1 Initial consistency checks
- Directory-vs-CSV naming overlaps were partial.
- `waymo_0-299` JSON content check example:
  - total pairs in JSON side: `6400`
  - matched in CSV: `1723`
  - unmatched: `4677`

## 4.2 Pairing rule diagnosis vs `process_interhub.py`
- `process_interhub.py` does **not** parse CSV `track_id` when selecting pair.
- Actual selection logic in code:
  - if AV + HV exists: select `(first AV, first HV)`
  - else if at least 2 HV: select first two HV
  - else skip
- This depends on JSON vehicle insertion/order, not CSV `track_id` order directly.

## 4.3 Using `intersection_results_pass_only.csv` with per-dataset dedup logic
- User rule adopted: duplicate `scenario_idx` handling uses last row, but dedup should be per dataset (`folder + scenario_idx`) to avoid cross-folder collisions.
- After per-dataset dedup and comparison:
  - overlap rows compared: `25274`
  - matched: `20596`
  - mismatched: `4678`
  - missing JSON scenario: `807`
  - overall match rate: `81.49%`

## 4.4 Unmatched categories (per-dataset dedup path)
- total unmatched: `4678`
- `last_row_overwrite_mismatch`: `3340`
- `missing_json_scenario`: `807`
- `av_rule_json_has_ego_but_not_first_hv`: `240`
- `hv_fallback_hv_pair_different`: `193`
- `hv_fallback_json_selected_ego_hv`: `95`
- `av_rule_json_selected_hv_hv`: `3`

Main reason: keeping only the last CSV row per `(folder, scenario_idx)` can drop the row that actually matches JSON-selected pair.

## 5. Filtered CSV Regeneration and Mean IPV Fill (Completed)

## 5.1 Strict JSON-matchable CSV generation
- Source: `intersection_results_pass_only.csv` (`42061` rows).
- Group key: `(folder, scenario_idx)`.
- Keep one row per group that can match JSON pair (if multiple matchable rows, keep last matchable).
- Output:
  - `intersection_results_pass_only_json_matched.csv`
  - rows: `23936`
  - verified:
    - duplicate `(folder,scenario_idx)` count = `0`
    - mismatch vs JSON pair = `0`

## 5.2 Exclusion accounting
- excluded groups due to:
  - no JSON folder: `6941`
  - missing JSON scenario: `807`
  - no pair match within group: `531`

## 5.3 JSON totals vs retained
- JSON total across all top-level dirs: `38043`
- JSON in current CSV scope (lyft + waymo splits): `26222`
- kept in matched CSV: `23936`
- excluded in scope:
  - no CSV group: `1755`
  - has CSV group but no pair match: `531`
- out-of-scope JSON (e.g., interaction datasets) not covered by current CSV mapping: `11821`

## 5.4 Mean IPV backfill to matched CSV
- Input: `intersection_results_pass_only_json_matched.csv` (`23936` rows)
- Added columns:
  - `ipv_vehicle_id_1`
  - `ipv_mean_ipv_1`
  - `ipv_vehicle_id_2`
  - `ipv_mean_ipv_2`
- Output:
  - `intersection_results_pass_only_json_matched_with_mean_ipv.csv`
- fill status: `23936/23936` rows successful.

## 6. Interhub Core Notebook Analysis (Completed)

Notebook: `paper_plot_interhub.ipynb`  
Input baseline: `intersection_results_pass_only_json_matched_with_mean_ipv.csv`

## 6.1 Derived schema and sample quality
From `sample_quality_report.csv`:
- raw rows: `23936`
- parseable PCS/PVS events: `16029`
- main sample (both IPV numeric): `7429`
- supplement sample (any IPV numeric): `14482`
- excluded rows: `7907`
- invalid reasons:
  - `priority_label_equal`: `7138`
  - `priority_not_in_pair`: `765`
  - `priority_label_missing_or_unknown`: `4`

Breakdowns in parseable events:
- by folder:
  - `waymo_0-299: 4100`
  - `waymo_500-799: 3620`
  - `waymo_300-499: 2826`
  - `lyft_train_full: 2810`
  - `waymo_800-999: 2673`
- by AV inclusion:
  - `all_HV: 11196`
  - `AV: 4833`
- by event label:
  - `PVS: 9218`
  - `PCS: 6811`

## 6.2 Core statistical results (Interhub)
From `hypothesis_tests.csv` (already BH-FDR corrected):

### A) Priority vs low-priority IPV
- PCS:
  - priority mean = `0.004` (slightly cooperative/near neutral)
  - low-priority mean = `-0.072` (competitive)
  - FDR p = `4.726e-10` (significant)
- PVS:
  - priority mean = `-0.021` (competitive)
  - low-priority mean = `0.039` (cooperative)
  - FDR p = `1.179e-4` (significant)

### B) AV vs HV under same role
- PCS / priority:
  - HV = `0.010` vs AV = `-0.048`
  - FDR p = `0.0502` (not significant at 0.05, borderline)
- PCS / low-priority:
  - HV = `-0.078` vs AV = `-0.035`
  - FDR p = `0.0597` (not significant)
- PVS / priority:
  - HV = `-0.093` vs AV = `0.138`
  - FDR p = `2.108e-83` (significant)
- PVS / low-priority:
  - HV = `0.026` vs AV = `0.121`
  - FDR p = `2.872e-9` (significant)

### C) PET and calculated_PET (APET proxy)
- PET: PCS vs PVS
  - PCS = `0.771`, PVS = `0.803`
  - FDR p = `1.030e-4` (significant)
- PET: AV vs all_HV
  - AV = `0.827`, all_HV = `0.773`
  - FDR p = `3.751e-12` (significant)
- calculated_PET: PCS vs PVS
  - PCS = `1.408`, PVS = `1.384`
  - FDR p = `0.0502` (not significant at 0.05; borderline)
- calculated_PET: AV vs all_HV
  - AV = `1.376`, all_HV = `1.402`
  - FDR p = `0.124` (not significant)

## 6.3 Meaning of "A类角色模式"
"A类角色模式" here means section A of analysis: within each event type (PCS/PVS), compare IPV of:
- priority agent (`ipv_priority`)
- low-priority agent (`ipv_low_priority`)

Interhub observed pattern:
- PCS: priority more cooperative than low-priority.
- PVS: low-priority more cooperative than priority.

## 7. Argoverse Re-analysis (Unified Sign Convention, Completed)

Data source:
- Recomputed directly from xlsx files under `argoverse/1_experiment_result/ipv_estimation`.
- Applied original filter: `step >= 6` and `ipv_error < 0.6` for both roles.

Re-analysis ingestion summary:
- xlsx files ingested: `5258`
- valid events kept: `2779`

Key Argoverse results (from `argoverse_ipv_reanalysis_summary.csv` + tests):
- A-role:
  - PCS: priority mean `-0.045` (competitive), low-priority `0.038` (cooperative), significant
  - PVS: priority mean `0.228` (cooperative), low-priority `-0.045` (competitive), significant
- B AV-vs-HV:
  - PVS low-priority: HV `0.013` vs AV `-0.216`, significant
  - PVS priority: HV `0.278` vs AV `0.079`, significant
  - PCS two role comparisons: not significant

## 8. Cross-Dataset Comparison (Argoverse vs Interhub, Completed)

See:
- `analysis_outputs/cross_dataset/argoverse_vs_interhub_comparison_long.csv`
- `analysis_outputs/cross_dataset/argoverse_vs_interhub_A_role_wide.csv`
- `analysis_outputs/cross_dataset/argoverse_vs_interhub_B_role_wide.csv`
- `analysis_outputs/cross_dataset/argoverse_vs_interhub_comparison.md`

Most important comparison point:
- A-role pattern direction is opposite between datasets:
  - PCS:
    - Argoverse priority competitive / low-priority cooperative
    - Interhub priority cooperative / low-priority competitive
  - PVS:
    - Argoverse priority cooperative / low-priority competitive
    - Interhub priority competitive / low-priority cooperative

## 9. Code-Level Difference Check: Argoverse vs Interhub IPV Pipelines

Conclusion:
- Core estimator is shared: both call `estimate_ipv_pair` from `ipv_estimation.py`.
- Effective core parameters are aligned (`history_window=10`, `min_observation=4`).

Important differences are upstream:
- Interhub has additional data filtering (`missing_reference`, lane-distance threshold, invalid entry).
- Interhub pair selection is AV/HV-first rule over JSON vehicle order, not CSV `track_id` parsing.
- Interhub uses heading-based dynamic labels (`lt/rt/gs`), Argoverse uses fixed role mapping (`lt_argo/gs_argo`).
- Therefore algorithm core is same, but sample composition and role labeling differ.

## 10. Workflow Logs to Trust

Primary full-history log for these rounds:
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\main_workflow.log`

This includes:
- Interhub Stage1~Stage6 runs
- sign convention update run
- Argoverse re-analysis run
- cross-dataset table generation run

## 11. Suggested Next Actions for Next Agent

1. Validate whether `priority_label_equal` cases (`7138`) should be re-labeled or excluded permanently in manuscript statistics.
2. Add a reproducible script (non-notebook) that regenerates:
   - Interhub `hypothesis_tests.csv`
   - Argoverse reanalysis tables
   - cross-dataset comparison tables
3. Add confidence intervals / bootstrap effect-size CI to core tests for paper-grade reporting.
4. If manuscript claims differ from current sign convention, resolve wording mismatch explicitly.

## 12. Minimal Reproduction Checklist

1. Ensure input CSV exists:
   - `interhub_traj_lane/intersection_results_pass_only_json_matched_with_mean_ipv.csv`
2. Execute:
   - `interhub_traj_lane/paper_plot_interhub.ipynb`
3. Verify outputs under:
   - `interhub_traj_lane/analysis_outputs/interhub_core/`
4. Verify Argoverse re-analysis outputs exist under:
   - `argoverse/analysis_outputs/argoverse_reanalysis/`
5. Verify cross-dataset files under:
   - `analysis_outputs/cross_dataset/`

