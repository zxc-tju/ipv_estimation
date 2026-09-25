# RQ027 Decision

Date: 2026-08-28  
Status: `ACCEPTED / CLOSED`  
Final state: `CLOSED_BY_PI_SCOPE_DECISION`  
Execution verdict retained: `PILOT_NO_GO`  
Further recovery research in this RQ: `STOPPED`

## Problem and stage

RQ027 tested whether the frozen online IPV estimator could recover a simulation-controlled IPV value when the trajectory generator did not share its planner, search, cost implementation or likelihood, and whether candidate-weight concentration could select lower-error readings. The bounded `240 interactive + 48 negative-control` pilot executed completely and was independently recomputed.

## PI ruling

For the current Nature Machine Intelligence manuscript, IPV recovery will not be developed as a new research line. The estimator is treated as an established method component introduced and validated in the published T-ITS study:

> Zhao, X., Sun, J. & Wang, M. Measuring sociality in driving interaction. IEEE Transactions on Intelligent Transportation Systems 25, 9224–9237 (2024).

The validation scope used by the manuscript is the controlled, model-consistent VGIM evidence already published there. The present paper focuses on the downstream human-reference monitor.

## Accepted interpretation

1. The RQ027 pilot is retained as a bounded diagnostic of cross-model numerical transportability for one frozen estimator, one independent generator family and one concentration policy.
2. The pilot does not support a general claim that IPV cannot be estimated, and it does not overturn the published VGIM-consistent recovery evidence.
3. The pilot does not reopen or revise the accepted RQ017, RQ018, RQ019, RQ021, RQ024 or RQ025 decisions.
4. Candidate-weight concentration may be used as the frozen operational readability rule of the present monitor; it is not presented as a calibrated guarantee of numerical recovery accuracy.
5. S2 perturbation expansion, the 3,120/14,040-run scale-up, sealed confirmatory testing and within-RQ retuning remain stopped.

## Manuscript directive

- Main text: cite the T-ITS paper in one short sentence and state that the estimator is held fixed as the behavioural reading supplied to the monitor.
- Supplementary Information: summarise the published controlled validation, the present online implementation, interaction-specific diagnostics and episode-summary sensitivity.
- Keep RQ027 figures and negative findings in the research record; they are not part of the paper's frontstage evidence chain.
- Use `readable/readability` and `candidate-discrimination` for the operational gate. Do not frame the current paper as a new universal parameter-recovery study.
- Follow the publication-conference principle: centre the paper on the context-conditioned human-reference monitor, without a defensive excursus on an abandoned comparison dimension.

## Closure

RQ027 is complete and closed. No additional recovery experiment is authorised under this RQ. Reopening requires an explicit new PI instruction and a new research contract.

## Evidence

- Plan: `reports/plans/RQ027_plan_v0_known_truth_ipv_recovery_20260828.md`
- Execution: `reports/studies/RQ027_known_truth_ipv_recovery/RQ027_1_known_truth_recovery_20260828T094332Z_889b49be/00_entry/index.html`
- Report: `reports/studies/RQ027_known_truth_ipv_recovery/RQ027_1_known_truth_recovery_20260828T094332Z_889b49be/01_results/REPORT.md`
- Independent validation: `reports/studies/RQ027_known_truth_ipv_recovery/RQ027_1_known_truth_recovery_20260828T094332Z_889b49be/01_results/independent_validation.md`
- Synthesis: `reports/knowledge/RQ027_known_truth_ipv_recovery/synthesis.md`
