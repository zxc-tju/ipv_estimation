# 可追溯记录

## 数据来源

- `data/derived/interhub/20260612_sigma_0_1_full_rerun/00_hpc_outputs/sigma01_ipv_timeseries.csv`
- `data/derived/interhub/20260612_sigma_0_1_full_rerun/00_hpc_outputs/sigma01_case_summary.csv`
- 既有报告：`reports/studies/RQ004_ipv_state_space/` 与
  `reports/studies/RQ001_online_ipv_interval/RQ001_2_interval_query_20260618/`

## 主分析脚本

- `02_process/run_current_ipv_distribution_experiment.py`
  - 构建 case-agent-frame 级别特征；
  - 评估 global、lag1 persistence、EWMA、residual HGB、paired/context、source-guard residual conformal 方法；
  - 输出 `01_results/current_ipv_interval_metrics.csv`。

- `02_process/run_selected_cqr_methods.py`
  - 复用 frame-level feature parquet；
  - 评估 self-history、paired-history、full-context、source-guard CQR 方法；
  - 输出 `01_results/current_ipv_selected_cqr_metrics.csv`。

## Fleet 子实验

- H1 temporal IPV-history agent：`02_process/agents/h1_temporal_ipv_history/`
  - 证明 lag-history CQR 能达到 P90 coverage 0.900、mean width 约 0.845。

- H2 paired-joint agent：`02_process/agents/h2_paired_joint_ipv/`
  - 证明 paired-history-only 不是 self-history 的稳定替代方案；P90 coverage 小幅提升，但 width 和 interval score 略变差。

- H3 trajectory/reference agent：`02_process/agents/h3_trajectory_reference_context/`
  - 证明 trajectory/reference alone 弱于 IPV history，但与 IPV history 合并后可小幅改善 interval score。

- H4 source calibration agent：`02_process/agents/h4_source_calibration/`
  - 证明 source/domain shift 是主要风险；Waymo LWO baseline P90 coverage 下降到 0.856，source-guard residual calibration 可恢复到 0.906。

## 在线可用性边界

实验中允许：

- 当前和历史轨迹状态；
- 自身 lagged IPV history，时间戳必须早于 `t`；
- 对方 lagged IPV history，时间戳必须早于 `t`；
- 查询时可得到的 source/context/reference-line 代理变量；
- 由 lagged IPV 和当前轨迹派生的 paired 结构特征。

部署时禁止作为输入：

- observed PET；
- actual order；
- closest-frame labels；
- post-hoc phase；
- 当前目标 IPV；
- 使用未来轨迹或未来参考线计算的统计量。

## 已完成验证

- `python -m py_compile Z:\02_reports\online_current_ipv_distribution_20260618\02_process\run_current_ipv_distribution_experiment.py`
- `python Z:\02_reports\online_current_ipv_distribution_20260618\02_process\run_current_ipv_distribution_experiment.py`
- `python -m py_compile Z:\02_reports\online_current_ipv_distribution_20260618\02_process\run_selected_cqr_methods.py`
- `python Z:\02_reports\online_current_ipv_distribution_20260618\02_process\run_selected_cqr_methods.py`

## 当前读者入口

- `00_entry/index.html`
- 为解决本地 file 页面图片不可见的问题，入口页使用 `00_entry/assets/` 中的图片副本。
