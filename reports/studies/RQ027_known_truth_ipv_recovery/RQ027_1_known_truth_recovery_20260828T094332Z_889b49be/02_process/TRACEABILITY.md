# Traceability

| Item | Path | Notes |
|---|---|---|
| Approved pilot plan | `reports/plans/RQ027_plan_v0_known_truth_ipv_recovery_20260828.md` | Source-plan hash and frozen measurement contract |
| Plan review | `reports/knowledge/RQ027_known_truth_ipv_recovery/reviews/plan_rationality_review.md` | `REVISE -> GO FOR BOUNDED PILOT` |
| Official entry | `00_entry/index.html` | Offline result page |
| Formal report | `01_results/report.md` | Self-contained result interpretation |
| Automated summary | `01_results/summary.json` | Executor verdict and gates |
| Independent validation | `01_results/independent_validation.json` | Independent recomputation |
| Run ledger | `01_results/run_level_results.csv` | One row per simulation run |
| Frame ledger | `01_results/frame_level_results.parquet` | Target-side attempted frames |
| Generator | `pipelines/simulation/rq027_independent_generator.py` | stdlib + NumPy only |
| Runner | `pipelines/simulation/run_rq027_pilot.py` | Frozen estimator adapter and summaries |
| Test | `tests/test_rq027_known_truth_recovery.py` | Independence, matrix, determinism and exact smoke |
