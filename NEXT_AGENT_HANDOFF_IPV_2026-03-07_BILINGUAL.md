# IPV Analysis Handoff (Bilingual 中英对照)

Last updated / 最后更新: 2026-03-07

## 1) Scope / 工作范围

- EN: Completed Interhub JSON/CSV alignment diagnostics, mismatch categorization, and strict JSON-matchable CSV regeneration.
- 中文：已完成 Interhub 的 JSON/CSV 对齐诊断、未匹配分类，并重建“严格可与 JSON 匹配”的 CSV。

- EN: Filled two-agent `mean_ipv` back into matched CSV rows.
- 中文：已将每条匹配样本中两车的 `mean_ipv` 回填到 CSV。

- EN: Implemented and executed Interhub core analysis notebook (PCS/PVS role pattern, AV vs HV, PET/APET proxy).
- 中文：已实现并执行 Interhub 核心分析 notebook（PCS/PVS 角色模式、AV vs HV、PET/APET 代理指标）。

- EN: Corrected interpretation convention to `IPV>0 cooperative`, `IPV<0 competitive`.
- 中文：已统一并修正解释口径为 `IPV>0 合作`、`IPV<0 竞争`。

- EN: Re-analyzed Argoverse directly from result xlsx under the same sign convention.
- 中文：已在同一符号口径下，直接读取 Argoverse 结果 xlsx 进行重分析。

- EN: Generated cross-dataset comparison outputs (Argoverse vs Interhub).
- 中文：已生成跨数据集对比输出（Argoverse vs Interhub）。

- EN: Audited code-path differences between `process_argoverse.py` and `process_interhub.py`.
- 中文：已审查 `process_argoverse.py` 与 `process_interhub.py` 的代码路径差异。

## 2) Interpretation Convention / 解释口径（必须统一）

- EN: `IPV > 0` = cooperative tendency; `IPV < 0` = competitive tendency; `IPV = 0` = neutral.
- 中文：`IPV > 0` = 合作倾向；`IPV < 0` = 竞争倾向；`IPV = 0` = 中性。

## 3) Canonical Files / 关键文件路径

## 3.1 Interhub input and aligned CSV / Interhub 输入与对齐 CSV
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\interhub_traj_lane\intersection_results_pass_only.csv`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\interhub_traj_lane\intersection_results_pass_only_json_matched.csv`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\interhub_traj_lane\intersection_results_pass_only_json_matched_with_mean_ipv.csv`

## 3.2 Interhub notebook and exports / Interhub notebook 与导出
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\interhub_traj_lane\paper_plot_interhub.ipynb`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\interhub_traj_lane\paper_plot_interhub.executed.ipynb`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\interhub_traj_lane\analysis_outputs\interhub_core\`
  - `interhub_analysis_ready.csv`
  - `sample_quality_report.csv`
  - `summary_stats.csv`
  - `hypothesis_tests.csv`
  - `figA/figB/figC` (PNG + PDF)

## 3.3 Argoverse reanalysis outputs / Argoverse 重分析输出
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\argoverse\analysis_outputs\argoverse_reanalysis\file_ingestion_stats.csv`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\argoverse\analysis_outputs\argoverse_reanalysis\argoverse_ipv_reanalysis_summary.csv`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\argoverse\analysis_outputs\argoverse_reanalysis\argoverse_ipv_reanalysis_tests.csv`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\argoverse\analysis_outputs\argoverse_reanalysis\argoverse_sign_convention_interpretation.csv`

## 3.4 Cross-dataset outputs / 跨数据集对比输出
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\analysis_outputs\cross_dataset\argoverse_vs_interhub_comparison_long.csv`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\analysis_outputs\cross_dataset\argoverse_vs_interhub_A_role_wide.csv`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\analysis_outputs\cross_dataset\argoverse_vs_interhub_B_role_wide.csv`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\analysis_outputs\cross_dataset\argoverse_vs_interhub_comparison.md`

## 4) Interhub Alignment Findings / Interhub 对齐结论

- EN: `process_interhub.py` pair selection is not parsed from CSV `track_id`.
- 中文：`process_interhub.py` 的配对不直接解析 CSV 的 `track_id`。

- EN: Actual rule is: if AV+HV exist -> `(first AV, first HV)`; else if HV>=2 -> first two HV; else skip.
- 中文：实际规则是：若有 AV+HV 则取 `(第一个 AV, 第一个 HV)`；否则若 HV>=2 则取前两个 HV；否则跳过。

- EN: Per-dataset dedup key used for validation is `(folder, scenario_idx)`; keeping only last row can overwrite matchable rows.
- 中文：校验时按 `(folder, scenario_idx)` 去重；仅保留最后一行会覆盖掉原本可匹配行。

### 4.1 Match stats (per-dataset dedup path) / 匹配统计（分 dataset 去重口径）
- compared / 对比行数: `25274`
- matched / 匹配行数: `20596`
- mismatched / 不匹配行数: `4678`
- missing_json_scenario / 缺失 JSON 场景: `807`
- match rate / 匹配率: `81.49%`

### 4.2 Mismatch categories / 不匹配类型
- `last_row_overwrite_mismatch`: `3340`
- `missing_json_scenario`: `807`
- `av_rule_json_has_ego_but_not_first_hv`: `240`
- `hv_fallback_hv_pair_different`: `193`
- `hv_fallback_json_selected_ego_hv`: `95`
- `av_rule_json_selected_hv_hv`: `3`

## 5) Regenerated CSV and mean_ipv fill / 重建 CSV 与 mean_ipv 回填

- EN: Source `intersection_results_pass_only.csv` has `42061` rows.
- 中文：源文件 `intersection_results_pass_only.csv` 共 `42061` 行。

- EN: Generated `intersection_results_pass_only_json_matched.csv` with `23936` rows.
- 中文：生成 `intersection_results_pass_only_json_matched.csv`，共 `23936` 行。

- EN: Verification: duplicate `(folder,scenario_idx)` count = `0`, mismatch vs JSON pair = `0`.
- 中文：校验通过：输出中 `(folder,scenario_idx)` 重复数 `0`，与 JSON 配对不一致数 `0`。

- EN: Exclusion reasons: no JSON folder=`6941`; missing JSON scenario=`807`; no pair match in group=`531`.
- 中文：剔除原因：无 JSON folder=`6941`；缺失 JSON scenario=`807`；组内无可匹配配对=`531`。

- EN: Filled columns `ipv_vehicle_id_1/2`, `ipv_mean_ipv_1/2` into `intersection_results_pass_only_json_matched_with_mean_ipv.csv`; success `23936/23936`.
- 中文：已在 `intersection_results_pass_only_json_matched_with_mean_ipv.csv` 回填 `ipv_vehicle_id_1/2` 与 `ipv_mean_ipv_1/2`；成功 `23936/23936`。

## 6) Interhub Core Results / Interhub 核心结果

Source / 来源: `interhub_traj_lane/analysis_outputs/interhub_core`

### 6.1 Sample quality / 样本质量
- raw_rows / 原始行: `23936`
- parseable_events (PCS/PVS) / 可解析事件: `16029`
- main_samples_both_numeric / 主样本（双侧 IPV 数值）: `7429`
- supplement_samples_any_numeric / 补充样本（至少一侧数值）: `14482`
- excluded_rows / 排除行: `7907`

Invalid reasons / 无效原因:
- `priority_label_equal`: `7138`
- `priority_not_in_pair`: `765`
- `priority_label_missing_or_unknown`: `4`

### 6.2 A-role pattern (priority vs low-priority) / A类角色模式（priority vs low-priority）
- EN PCS: priority mean `0.004` (near neutral/slightly cooperative), low-priority `-0.072` (competitive), FDR significant.
- 中文 PCS：priority 均值 `0.004`（近中性/略合作），low-priority `-0.072`（竞争），FDR 显著。

- EN PVS: priority mean `-0.021` (competitive), low-priority `0.039` (cooperative), FDR significant.
- 中文 PVS：priority 均值 `-0.021`（竞争），low-priority `0.039`（合作），FDR 显著。

### 6.3 B-role AV vs HV / B类同角色 AV vs HV
- EN PCS/priority and PCS/low-priority are not significant (borderline near 0.05).
- 中文 PCS/priority 与 PCS/low-priority 在 FDR 下均不显著（接近 0.05）。

- EN PVS/priority and PVS/low-priority are significant, with AV showing more cooperative tendency in both roles.
- 中文 PVS/priority 与 PVS/low-priority 均显著，且 AV 在两种角色下都更偏合作。

### 6.4 C-role safety metrics (PET/APET proxy) / C类安全指标（PET/APET 代理）
- EN PET: PCS vs PVS significant; AV vs all_HV significant.
- 中文 PET：PCS vs PVS 显著；AV vs all_HV 显著。

- EN calculated_PET: PCS vs PVS not significant (borderline); AV vs all_HV not significant.
- 中文 calculated_PET：PCS vs PVS 不显著（边缘）；AV vs all_HV 不显著。

## 7) Argoverse Reanalysis / Argoverse 重分析

- EN: Reanalysis ingested `5258` xlsx events and kept `2779` valid events (`step>=6` and both-role `ipv_error<0.6`).
- 中文：重分析读取 `5258` 个 xlsx 事件，保留 `2779` 个有效事件（`step>=6` 且双角色 `ipv_error<0.6`）。

- EN PCS: priority competitive, low-priority cooperative (significant).
- 中文 PCS：priority 竞争、low-priority 合作（显著）。

- EN PVS: priority cooperative, low-priority competitive (significant).
- 中文 PVS：priority 合作、low-priority 竞争（显著）。

## 8) Cross-Dataset Comparison / 跨数据集对比

- EN: A-role pattern direction is largely opposite between Argoverse and Interhub.
- 中文：A类角色模式在 Argoverse 与 Interhub 之间方向基本相反。

- EN PCS:
  - Argoverse: priority competitive / low-priority cooperative
  - Interhub: priority cooperative / low-priority competitive
- 中文 PCS：
  - Argoverse：priority 竞争 / low-priority 合作
  - Interhub：priority 合作 / low-priority 竞争

- EN PVS:
  - Argoverse: priority cooperative / low-priority competitive
  - Interhub: priority competitive / low-priority cooperative
- 中文 PVS：
  - Argoverse：priority 合作 / low-priority 竞争
  - Interhub：priority 竞争 / low-priority 合作

## 9) Code Difference Audit / 代码差异审查结论

- EN: Core solver is shared (`estimate_ipv_pair`) and effective parameters are aligned (`history_window=10`, `min_observation=4`).
- 中文：核心求解器相同（`estimate_ipv_pair`），有效参数一致（`history_window=10`、`min_observation=4`）。

- EN: Differences come from upstream pipeline: filtering, pair-selection policy, and maneuver labeling.
- 中文：差异主要来自上游流程：过滤规则、配对策略、机动标签定义。

- EN: So distribution/conclusion differences are expected from sample composition and role semantics, not solver core.
- 中文：因此分布与结论差异主要来自样本组成与角色语义，不是求解器本体差异。

## 10) Primary Logs / 主日志

- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\main_workflow.log`
- `C:\Users\xiaocongzhao\.codex\worktrees\bb52\2_sociality_estimation\main_workflow.log`

## 11) Recommended Next Steps / 建议下一步

1. EN: Decide manuscript policy for `priority_label_equal` (`7138`) cases: exclude permanently or define a dedicated category.
   中文：先确定文稿对 `priority_label_equal`（`7138`）样本的处理：永久剔除或单独建类。

2. EN: Convert notebook outputs into script-based reproducible pipeline.
   中文：将 notebook 流程脚本化，形成可重复的一键流水线。

3. EN: Add confidence intervals (bootstrap CI) for key effect sizes.
   中文：为关键效应量补充置信区间（bootstrap CI）。

4. EN: Ensure manuscript language is fully consistent with the sign convention above.
   中文：确保文稿措辞与当前符号口径完全一致。

