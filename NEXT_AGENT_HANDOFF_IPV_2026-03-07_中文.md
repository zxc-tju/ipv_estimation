# IPV 分析交接文档（给下一位 Agent）

最后更新：2026-03-07

## 1. 已完成工作范围

本轮已完成以下链路：
- Interhub 的 JSON/CSV 对齐检查与不匹配原因分析。
- 重新生成“只保留可与 JSON 配对成功”的 CSV。
- 从 JSON 回填两车 `mean_ipv` 到 CSV。
- 完成 Interhub 核心分析（PCS/PVS、AV vs HV、PET/APET）。
- 明确并修正解释口径：`IPV>0` 合作，`IPV<0` 竞争。
- 直接读取 Argoverse 结果 xlsx 进行同口径重分析。
- 生成 Argoverse vs Interhub 对比表。
- 审核两套 IPV 计算代码差异（Argoverse vs Interhub）。

## 2. 统一解释口径（重要）

- `IPV > 0`：合作倾向
- `IPV < 0`：竞争倾向
- `IPV = 0`：中性

后续所有文字结论都应保持该口径一致。

## 3. 关键数据与产物路径

## 3.1 Interhub 输入与对齐后 CSV
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\interhub_traj_lane\intersection_results_pass_only.csv`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\interhub_traj_lane\intersection_results_pass_only_json_matched.csv`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\interhub_traj_lane\intersection_results_pass_only_json_matched_with_mean_ipv.csv`

## 3.2 Interhub notebook 与导出
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\interhub_traj_lane\paper_plot_interhub.ipynb`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\interhub_traj_lane\paper_plot_interhub.executed.ipynb`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\interhub_traj_lane\analysis_outputs\interhub_core\`

主要文件：
- `interhub_analysis_ready.csv`
- `sample_quality_report.csv`
- `summary_stats.csv`
- `hypothesis_tests.csv`
- `figA/figB/figC` 的 PNG + PDF

## 3.3 Argoverse 重分析输出
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\argoverse\analysis_outputs\argoverse_reanalysis\file_ingestion_stats.csv`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\argoverse\analysis_outputs\argoverse_reanalysis\argoverse_ipv_reanalysis_summary.csv`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\argoverse\analysis_outputs\argoverse_reanalysis\argoverse_ipv_reanalysis_tests.csv`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\argoverse\analysis_outputs\argoverse_reanalysis\argoverse_sign_convention_interpretation.csv`

## 3.4 跨数据集对比输出
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\analysis_outputs\cross_dataset\argoverse_vs_interhub_comparison_long.csv`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\analysis_outputs\cross_dataset\argoverse_vs_interhub_A_role_wide.csv`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\analysis_outputs\cross_dataset\argoverse_vs_interhub_B_role_wide.csv`
- `C:\Users\xiaocongzhao\OneDrive\Desktop\Projects\1_Codes\2_sociality_estimation\analysis_outputs\cross_dataset\argoverse_vs_interhub_comparison.md`

## 4. Interhub 对齐与筛选结论

## 4.1 对齐与配对规则核对
- `process_interhub.py` 不是按 CSV 的 `track_id` 直接选车。
- 代码实际规则：
  - 有 AV + HV：取 `(第一个 AV, 第一个 HV)`
  - 否则 HV 数量 >=2：取前两个 HV
  - 否则跳过

## 4.2 基于 `intersection_results_pass_only.csv` 的分 dataset 处理
- 去重键采用 `(folder, scenario_idx)`，组内只看最后一条（并注意跨 folder 同 `scenario_idx` 冲突）。
- 对齐统计（重算后）：
  - compared: `25274`
  - matched: `20596`
  - mismatched: `4678`
  - missing_json_scenario: `807`
  - match rate: `81.49%`

## 4.3 未匹配类型（4678）
- `last_row_overwrite_mismatch`: `3340`
- `missing_json_scenario`: `807`
- `av_rule_json_has_ego_but_not_first_hv`: `240`
- `hv_fallback_hv_pair_different`: `193`
- `hv_fallback_json_selected_ego_hv`: `95`
- `av_rule_json_selected_hv_hv`: `3`

主要原因是“同组只留最后一条”造成的覆盖错配。

## 5. 已生成的可用 CSV 与回填状态

## 5.1 严格可匹配 CSV
- 输入：`intersection_results_pass_only.csv`（`42061` 行）
- 输出：`intersection_results_pass_only_json_matched.csv`（`23936` 行）
- 验证：
  - 输出内 `(folder, scenario_idx)` 重复为 `0`
  - 输出与 JSON 配对不匹配为 `0`

## 5.2 剔除原因
- no JSON folder: `6941`
- missing JSON scenario: `807`
- 组内无任何可匹配行: `531`

## 5.3 JSON 规模与覆盖
- JSON 全量（所有顶层目录）：`38043`
- 当前 CSV 覆盖范围内（lyft+waymo）：`26222`
- 保留到输出 CSV：`23936`
- 范围内被排除：
  - 无 CSV 分组：`1755`
  - 有 CSV 但无配对匹配：`531`
- 范围外 JSON（如 interaction_*）：`11821`

## 5.4 mean_ipv 回填
- 输出：`intersection_results_pass_only_json_matched_with_mean_ipv.csv`
- 新增列：
  - `ipv_vehicle_id_1`
  - `ipv_mean_ipv_1`
  - `ipv_vehicle_id_2`
  - `ipv_mean_ipv_2`
- 回填成功：`23936/23936`

## 6. Interhub 核心分析结果（已执行）

来源：`interhub_core/sample_quality_report.csv`、`summary_stats.csv`、`hypothesis_tests.csv`

## 6.1 样本质量
- raw rows: `23936`
- parseable events (PCS/PVS): `16029`
- main samples（两侧 IPV 都是数值）: `7429`
- supplement samples（至少一侧是数值）: `14482`
- excluded: `7907`

无效原因：
- `priority_label_equal`: `7138`
- `priority_not_in_pair`: `765`
- `priority_label_missing_or_unknown`: `4`

## 6.2 A类角色模式（priority vs low-priority）
- PCS：
  - priority mean = `0.004`（近中性/略合作）
  - low-priority mean = `-0.072`（竞争）
  - FDR 显著
- PVS：
  - priority mean = `-0.021`（竞争）
  - low-priority mean = `0.039`（合作）
  - FDR 显著

说明：这里的 “A类角色模式” 就是固定在 PCS/PVS 条件下，比 `ipv_priority` 和 `ipv_low_priority` 的分布与均值方向。

## 6.3 B类（同角色下 AV vs HV）
- PCS / priority：未显著（边缘）
- PCS / low-priority：未显著
- PVS / priority：显著，HV 更竞争、AV 更合作
- PVS / low-priority：显著，AV 更合作

## 6.4 C类（PET / calculated_PET）
- PET：PCS vs PVS 显著
- PET：AV vs all_HV 显著
- calculated_PET：PCS vs PVS 不显著（边缘）
- calculated_PET：AV vs all_HV 不显著

## 7. Argoverse 同口径重分析结果

处理口径：
- 直接读取 `argoverse/1_experiment_result/ipv_estimation` 下 xlsx
- 过滤规则：`step>=6` 且双方 `ipv_error<0.6`

规模：
- xlsx 读入：`5258`
- valid events：`2779`

关键方向：
- PCS：priority 竞争、low-priority 合作（显著）
- PVS：priority 合作、low-priority 竞争（显著）

## 8. Argoverse vs Interhub 对比要点

最关键差异：A类角色模式方向基本相反。
- PCS：
  - Argoverse：priority 竞争 / low-priority 合作
  - Interhub：priority 合作 / low-priority 竞争
- PVS：
  - Argoverse：priority 合作 / low-priority 竞争
  - Interhub：priority 竞争 / low-priority 合作

详见 `analysis_outputs/cross_dataset` 目录下三份 CSV 与一份 MD。

## 9. 代码差异结论（IPV 计算流程）

结论：
- 两套都调用同一个核心求解函数 `estimate_ipv_pair`。
- 有效核心参数一致（`history_window=10`, `min_observation=4`）。

真正差异在上游：
- Interhub 有更多过滤（缺参考线、车道距离阈值、无效轨迹等）。
- Interhub 选车依赖 JSON 车辆顺序 + AV/HV 规则，不直接使用 CSV `track_id` 解析。
- Interhub 标签是动态航向分类（`lt/rt/gs`），Argoverse 多为固定角色映射（`lt_argo/gs_argo`）。

因此：核心算法一致，但样本组成和角色定义不同，会导致分布/结论差异。

## 10. 建议下一步

1. 先确认 `priority_label_equal`（7138 条）在论文口径中是永久剔除还是另建类别。  
2. 将 notebook 结果固化为脚本化 pipeline（可一键复现 Interhub + Argoverse + cross-dataset）。  
3. 给核心效应量加置信区间（bootstrap CI）用于论文级报告。  
4. 核查文稿中“egoistic/cooperative”表述是否与当前符号口径完全一致。  

## 11. 复现最小清单

1. 确认输入存在：`intersection_results_pass_only_json_matched_with_mean_ipv.csv`
2. 运行 `paper_plot_interhub.ipynb`
3. 检查 `interhub_traj_lane/analysis_outputs/interhub_core/` 文件完整
4. 检查 `argoverse/analysis_outputs/argoverse_reanalysis/` 文件完整
5. 检查 `analysis_outputs/cross_dataset/` 文件完整

