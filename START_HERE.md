# START_HERE: Current Operating Brief

Last reviewed: 2026-06-20.

Use this file as the first stop for a new agent thread. Keep durable policy in
`AGENTS.md`, architecture notes in `PROJECT_STRUCTURE.md`, and the research
question index in `STUDIES.md`.

## Current Active Context

- Primary technical context: realtime IPV estimator validation and InterHub
  CSV/pkl motion-data pipelines.
- Recommended online sign mode: `RealtimeIPVEstimator.for_realtime_sign(...)`
  with `history_window=10`, `max_workers=10`, and the five-candidate sign grid.
- Accuracy-preserving online value mode: `solver_preset="parallel_accurate"`
  with the legacy seven-candidate grid.
- The 20260612 sigma 0.1 full-rerun data source is now under
  `data/derived/interhub/20260612_sigma_0_1_full_rerun/00_hpc_outputs/`.

## Canonical Code Entrypoints

- Core IPV package: `src/sociality_estimation/core/`.
- Planning and geometry helpers: `src/sociality_estimation/planning/`.
- Active InterHub CSV/pkl pipeline:
  `pipelines/interhub/process_interhub.py`.
- Active simulation entrypoint: `pipelines/simulation/simulator.py`.
- InterHub helper/report scripts: `pipelines/interhub/tools/`.
- Old root wrappers are archived under `archived/compat_wrappers_20260619/`.

## Convenience Launchers

- macOS double-click Terminal launchers live at
  `scripts/launch_claude.command` and `scripts/launch_codex.command`.
- Each launcher enters this project root, then starts the corresponding CLI.
  If the CLI command is missing, the Terminal window stays open with the
  current directory and PATH for diagnosis.

## Canonical Knowledge And Report Paths

- Study index and status board: `STUDIES.md`.
- Execution/report layer: `reports/studies/`.
- Interpretation/review/decision layer: `reports/knowledge/`.
- `reports/` intentionally has only two first-level directories:
  `studies/` and `knowledge/`.
- Large derived outputs live under `data/derived/`.
- Report-linked process archives and local agent state live under
  `archived/report_process/` and `archived/report_local_state/`.
- Manuscript/paper drafting lives in the standalone paper repository:
  `../9_overleaf/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle`.
  Do not recreate a top-level `paper/` directory here.

## Active Study Map

| RQ | Study folder | Knowledge folder |
|---|---|---|
| RQ001 online IPV interval | `reports/studies/RQ001_online_ipv_interval/` | `reports/knowledge/RQ001_online_ipv_interval/` |
| RQ002 self-anchor group norm | `reports/studies/RQ002_self_anchor_group_norm/` | `reports/knowledge/RQ002_self_anchor_group_norm/` |
| RQ003 NSFC external evidence | `reports/studies/RQ003_nsfc_external_evidence/` | `reports/knowledge/RQ003_nsfc_external_evidence/` |
| RQ004 IPV state space | `reports/studies/RQ004_ipv_state_space/` | `reports/knowledge/RQ004_ipv_state_space/` |
| RQ005 NMI evidence gap | `reports/studies/RQ005_nmi_evidence_gap/` | `reports/knowledge/RQ005_nmi_evidence_gap/` |
| RQ006 sigma sensitivity | `reports/studies/RQ006_sigma_sensitivity/` | `reports/knowledge/RQ006_sigma_sensitivity/` |

For parallel agent runs under one RQ, use execution names like
`RQ003_5_nsfc_open_explore_codex_20260619`; the number after the underscore is
the execution version.

## Canonical Data Paths

- InterHub subset CSV:
  `data/interhub/raw/subsets_for_yiru/selected_interactive_segments_equalized.csv`
- InterHub subset pkl root: `data/interhub/raw/subsets_for_yiru/pkl/`
- InterHub full-dataset raw data: `data/interhub/raw/full_datasets/`
- InterHub sigma 0.1 derived full-rerun outputs:
  `data/derived/interhub/20260612_sigma_0_1_full_rerun/00_hpc_outputs/`
- Onsite competition local payload: `data/onsite_competition/raw/`
- Onsite competition manifests: `data/onsite_competition/00_manifest/` and
  `data/onsite_competition/top5_research_subset/`
- Legacy Argoverse source data:
  `archived/argoverse/0_souce_data/` (typo is historical).

## Key Report Entries

- RQ001 deployable online interval report:
  `reports/studies/RQ001_online_ipv_interval/RQ001_3_online_interval_lock_20260619/00_entry/index.html`
- RQ002 main self-anchor validation:
  `reports/studies/RQ002_self_anchor_group_norm/RQ002_1_self_anchor_validation_main_20260619/00_entry/index.html`
- RQ002 parallel Codex validation:
  `reports/studies/RQ002_self_anchor_group_norm/RQ002_2_self_anchor_validation_codex_20260619/00_entry/index.html`
- RQ003 core NSFC evidence:
  `reports/studies/RQ003_nsfc_external_evidence/RQ003_1_nsfc_core_evidence_20260618/00_entry/core_results_nature.html`
- RQ003 detailed synthesis:
  `reports/studies/RQ003_nsfc_external_evidence/RQ003_2_nsfc_detailed_synthesis_20260619/00_entry/index.html`
- RQ003 parallel Codex open exploration:
  `reports/studies/RQ003_nsfc_external_evidence/RQ003_5_nsfc_open_explore_codex_20260619/00_entry/index.html`
- RQ003 Tier B NSFC IPV validation final reader:
  `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/00_entry/index.html`

## How To Run Tests

- Lightweight shortcut-script checks:
  `python3 -m unittest tests.test_shortcut_scripts -q`.
- If a broader pytest suite is restored, run: `python -m pytest tests -q`.
- Baseline syntax check for active code:
  `python -m py_compile src/sociality_estimation/core/agent.py src/sociality_estimation/core/ipv_estimation.py src/sociality_estimation/planning/Lattice.py src/sociality_estimation/planning/lattice_planner.py src/sociality_estimation/planning/utility.py pipelines/interhub/process_interhub.py pipelines/simulation/simulator.py`.
- One-case InterHub smoke check:
  `python pipelines/interhub/process_interhub.py --limit 1 --workers 1 --solver-preset realtime --no-plots --output-root data/derived/interhub/_codex_runtime_smoke`.
- If a verification command fails only because a Python dependency is missing,
  install it in the active project environment and continue; record durable
  dependency changes in requirements or `main_workflow.log`.

## What Not To Delete

- Raw data under `data/interhub/raw/`, `data/onsite_competition/raw/`,
  `data/onsite_competition/top5_research_subset/teams/`, and
  `archived/argoverse/0_souce_data/`.
- Derived InterHub full-rerun outputs under `data/derived/interhub/`.
- Reader-facing study report packages under `reports/studies/`.
- Knowledge decisions and manuscript context under `reports/knowledge/`.
- Report-linked process archives under `archived/report_process/`.
- `main_workflow.log`, `AGENTS.md`, `START_HERE.md`, `PROJECT_STRUCTURE.md`,
  and `STUDIES.md`.

## Known Weak Spots

- NuPlan remains the weakest realtime IPV slice; current validation does not
  support a dataset-specific NuPlan >90% sign-accuracy guarantee.
- Online IPV interval deployment is route-conditioned. No-lane/no-route cases
  should fall back to no-roll kinematic CQR.
- Self-anchor is useful for sharpness but not sufficient alone as a validated
  group-norm verifier; use situation floor plus out-of-support abstention.
- NSFC evidence is useful for external design and hypothesis mining, but current
  reports do not freeze formal verifier-validation claims.
