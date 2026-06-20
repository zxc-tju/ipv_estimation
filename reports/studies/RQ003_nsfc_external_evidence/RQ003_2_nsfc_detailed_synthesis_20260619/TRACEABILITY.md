# Traceability

## Entry

- `00_entry/index.html`

## Final Results

- Source-of-truth synthesis: `01_results/FINDINGS.md`
- Result table index: `01_results/README.md`
- Report-rendered figures: `02_process/figures/fig1.png` through
  `02_process/figures/fig6.png`
- Editable vector figures: `02_process/figures/fig1.svg` through
  `02_process/figures/fig6.svg`
- Plotting script: `02_process/make_submission_figures.py`

## Figure Sources

- Fig1 Tier-A premise:
  `02_process/agent_results/ws2-outcomes-tierA/master_outcome_table.csv`,
  `safety_saturation_spread.csv`, `safe_but_bad_social_cells.csv`
- Fig2 NSFC scoring structure:
  `02_process/agent_results/ws2-outcomes-tierA/score_correlation_matrix_long.csv`,
  `score_pca_loadings.csv`, `score_partial_correlation_matrix_long.csv`
- Fig3 Tier-B null:
  `02_process/agent_results/ws3-stats/results_table.csv`,
  `merged_analysis_table.csv`
- Fig4 1-D do-simulation:
  `02_process/agent_results/ws4-interhub-dosim/do_intervention_dose_response_summary.csv`
- Fig5 2-D failure basin / best response / human band:
  `02_process/agent_results/ws5-2d-bestresponse/2d_sweep_per_cell_surface.csv`,
  `coordination_failure_basin.csv`, `best_response_curve.csv`,
  `human_ipv_marginal_summary.csv`
- Fig6 temporal precedence:
  `02_process/agent_results/ws6-temporal-precedence/lagged_precedence_results.csv`,
  `cross_lagged_results.csv`, `negative_control_results.csv`

## Process Files

- Agent results: `02_process/agent_results/`
- Fleet reports: `02_process/fleet_reports/`
- Prompts and orchestration records: `02_process/prompts/`,
  `02_process/plan.md`, `02_process/tried.md`, `02_process/knowledge.md`

The HTML report references figures as `../02_process/figures/figN.png` so the
entry remains directly viewable from disk while the figure-generation process
stays traceable.
