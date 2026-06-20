# Traceability

## Entry

- `00_entry/index.html`

## Final Results

- Summary memo: `01_results/final_summary.md`
- Metrics table: `01_results/online_interval_metrics.csv`
- Prediction sample: `01_results/online_interval_predictions_sample.csv`
- Feature extract: `01_results/online_prefix_case_agent_features.parquet`
- Feature extraction summary: `01_results/feature_extract_summary.json`
- Recommendation JSON: `01_results/model_recommendation_summary.json`
- Figure PNG/PDF: `01_results/figures/coverage_width_comparison.*`

## Process Files

- Reproducible experiment script:
  `02_process/run_online_ipv_interval_experiment.py`
- Query algorithm specification:
  `02_process/agent_algorithm_spec/online_ipv_interval_query_spec.md`
- Fleet board:
  `archived/report_local_state/interhub_20260620/codex_fleet/online-ipv-query/board/`

## Claim Map

| Claim | Evidence |
|---|---|
| PET/risk lookup is not the accuracy bottleneck. | `online_interval_metrics.csv`: oracle PET and online TTC cell lookup are only slightly narrower than global floor. |
| Direct online-safe conditional distribution modeling improves IPV interval sharpness. | `strict_online_kinematic_cqr` rows in metrics table and Figure 1. |
| Rolling-IPV self-anchor is the strongest candidate signal. | `rolling_ipv_anchor_cqr` rows plus subagent causality audit in validation board. |
| Deployment-grade source-shift validity is not solved. | Leave-Waymo-Out rows in metrics table; all methods under nominal 90% coverage. |
