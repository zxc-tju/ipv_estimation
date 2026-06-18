# START_HERE: Current Operating Brief

Last reviewed: 2026-06-18.

Use this file as the first stop for a new agent thread. Keep it short, current,
and operational; put durable policy in `AGENTS.md` and deeper architecture notes
in `PROJECT_STRUCTURE.md`.

## Current Active Batch

- Primary active context: realtime IPV estimator validation and documentation for
  InterHub CSV/pkl motion data.
- Recommended online sign mode: `RealtimeIPVEstimator.for_realtime_sign(...)`
  with `history_window=10`, `max_workers=10`, and the five-candidate sign grid.
- Accuracy-preserving online value mode: `solver_preset="parallel_accurate"`
  with the legacy seven-candidate grid.
- Production/full-dataset result context currently centers on
  `interhub_traj_lane/1_ipv_estimation_results/full_datasets/batches/20260612_sigma_0_1_full_rerun/`.

## Canonical Data Paths

- InterHub subset CSV: `interhub_traj_lane/0_raw_data/subsets_for_yiru/selected_interactive_segments_equalized.csv`
- InterHub subset pkl root: `interhub_traj_lane/0_raw_data/subsets_for_yiru/pkl/`
- InterHub full-dataset raw data: `interhub_traj_lane/0_raw_data/full_datasets/`
- Legacy Argoverse source data: `argoverse/0_souce_data/` (typo is part of the current path)
- Onsite competition local payload: `onsite_competition_results/` (large/untracked local data)

## Canonical Output Paths

- InterHub subset IPV outputs: `interhub_traj_lane/1_ipv_estimation_results/subsets_for_yiru/`
- InterHub full-dataset batches: `interhub_traj_lane/1_ipv_estimation_results/full_datasets/batches/`
- Realtime validation artifacts: `interhub_traj_lane/1_ipv_estimation_results/_codex_parallel_accurate_realtime_check/`
- Runtime smoke outputs: `interhub_traj_lane/1_ipv_estimation_results/_codex_runtime_smoke/`
- Legacy Argoverse IPV outputs: `argoverse/1_experiment_result/ipv_estimation/`
- Argoverse analysis outputs: `argoverse/analysis_outputs/`

## How To Run Tests

- Current working tree has no active `tests/` suite after cleanup. If tests are
  restored, run: `python -m pytest tests -q`.
- Baseline syntax check for active code: `python -m py_compile agent.py ipv_estimation.py process_interhub.py`.
- Legacy scripts are archived under `archived/legacy_scripts/`; restore and
  path-check them before treating them as active commands.
- One-case InterHub smoke check:
  `python process_interhub.py --limit 1 --workers 1 --solver-preset realtime --no-plots --output-root interhub_traj_lane/1_ipv_estimation_results/_codex_runtime_smoke`.
- For scientific or publication claims, rerun the relevant pipeline on a
  representative sample and compare generated CSV/figure/report artifacts.

## How To Start App/Server

- No persistent app or web server is required. This repository is script- and
  report-oriented.
- Static report HTML files can be opened directly from disk.

## What Not To Delete

- Raw data under `interhub_traj_lane/0_raw_data/`, `argoverse/0_souce_data/`,
  and `onsite_competition_results/`.
- Reader-facing reports, evidence bundles, and summary tables under
  `interhub_traj_lane/1_ipv_estimation_results/` and `argoverse/analysis_outputs/`.
- Report-linked process archives, especially `01_process/` folders and
  `hpc_run_files/` READMEs/scripts.
- `main_workflow.log`, `AGENTS.md`, and this `START_HERE.md`.
- Large archives such as curated valid case zips unless the user explicitly
  approves cleanup. Delete only reproducible caches by default.

## Latest Stable Report/Result

- Realtime IPV estimator summary: `docs/realtime_ipv_estimator.md`.
- Latest mixed-distribution realtime sign validation:
  `interhub_traj_lane/1_ipv_estimation_results/_codex_parallel_accurate_realtime_check/current5_50pd_w10_pkl_*`.
- Key stable numbers from the documented 200-case mixed validation: overall
  sign accuracy 92.3%, Wilson 95% lower bound 91.3%, mean latency 0.110 s,
  median latency 0.099 s.
- Stable subset report: `interhub_traj_lane/1_ipv_estimation_results/subsets_for_yiru/ipv_distribution_report/ipv_distribution_report.html`.
- Full-dataset sigma 0.1 rerun package:
  `interhub_traj_lane/1_ipv_estimation_results/full_datasets/batches/20260612_sigma_0_1_full_rerun/`.
- Organized report entries for that package are indexed at:
  `interhub_traj_lane/1_ipv_estimation_results/full_datasets/batches/20260612_sigma_0_1_full_rerun/02_reports/README.md`.
  Each report package uses `00_entry/index.html`, `01_results/`, and
  `02_process/` plus a `TRACEABILITY.md` map.

## Known Weak Spots

- NuPlan is the weakest realtime IPV slice; current validation does not support
  a dataset-specific NuPlan >90% sign-accuracy guarantee.
- The online estimator is a statistical realtime solution, not a hard-deadline
  guarantee for every frame. Use asynchronous scheduling for strict control
  deadlines.
- `archived/legacy_scripts/batch_process_ipv.py` defaults to
  `interhub_traj_lane/ipv_estimation_results`, while current InterHub outputs
  use `interhub_traj_lane/1_ipv_estimation_results/`. Check paths before
  restoring or running it.
- `simulator.py` references external resources such as `NDS_analysis` that are
  not present in this repository snapshot.
- `requirements-minimal.txt` is referenced by older guidance but is not present
  in the current file list; use `requirements.txt` unless a minimal file is
  restored.
