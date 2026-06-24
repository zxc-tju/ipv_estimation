# START_HERE: Current Operating Brief

Last reviewed: 2026-06-24.

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
- Each launcher enters this project root, then starts the corresponding CLI via
  `cmux codex-teams` or `cmux claude-teams`. If `cmux` or the CLI command is
  missing, the Terminal window stays open with the current directory and PATH
  for diagnosis.

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
| RQ007 interaction-conditioned IPV estimability | `reports/studies/RQ007_interaction_conditioned_ipv_estimability/` | `reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/` |
| RQ008 InterHub temporal IPV discovery | `reports/studies/RQ008_interhub_temporal_ipv_discovery/` | `reports/knowledge/RQ008_interhub_temporal_ipv_discovery/` |
| RQ010 WOD-E2E tracking feasibility | `reports/studies/RQ010_wod_e2e_tracking_feasibility/` | `reports/knowledge/RQ010_wod_e2e_tracking_feasibility/` |
| RQ011 onsite full-universe readiness | `reports/studies/RQ011_onsite_full_universe_readiness/` | `reports/knowledge/RQ011_onsite_full_universe_readiness/` |
| RQ012 onsite event annotation readiness | `reports/studies/RQ012_onsite_event_annotation_readiness/` | `reports/knowledge/RQ012_onsite_event_annotation_readiness/` |

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
- Onsite competition current all-team package, generated locally and ignored:
  `data/onsite_competition/all_teams_dataset/` (rebuild with
  `scripts/build_onsite_all_teams_dataset.py`)
- Onsite competition lightweight manifests: `data/onsite_competition/00_manifest/`
- Onsite competition archived raw/top-five subset payload:
  `archived/onsite_competition_raw_and_top5_subset_20260623/`
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
- RQ007 interaction-conditioned IPV estimability report (development/guard estimability boundary;
  knowledge `decision.md` frozen 2026-06-24; held-out sealed):
  `reports/studies/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/00_entry/index.html`
- RQ008 InterHub temporal IPV discovery report (negative discovery-only result;
  knowledge `decision.md` frozen 2026-06-24; 0/24 candidates survived,
  confirmation split remains unopened):
  `reports/studies/RQ008_interhub_temporal_ipv_discovery/RQ008_1_temporal_ipv_discovery_20260622T234914+0800_3e3e776a/00_entry/index.html`
- RQ010 WOD-E2E tracking feasibility report (`T2_FULL_TRACKING_REQUIRED`;
  knowledge `decision.md` frozen 2026-06-24; Route 4 preferred,
  Route 5 fallback, HPC blocked pending access):
  `reports/studies/RQ010_wod_e2e_tracking_feasibility/RQ010_1_wod_tracking_feasibility_20260623T073830+0800_14f21d3e/00_entry/index.html`
- RQ012A OnSite event annotation readiness Wave-A package (9 automatic events;
  gates 012-0/012-1 pass, 012-2 text-cleared, 012-3 ready-pending-humans,
  012B blocked; knowledge `decision.md` freezes the deferral, not a full PASS):
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37/90_report/index.html`
- RQ011A OnSite full-universe readiness (re-run on complete data; `READY_WITH_FROZEN_EXCLUSIONS`:
  outcome universe full 300 / replay 285 with T19 excluded; run-level & repeated-run not identifiable
  by design; knowledge `decision.md` frozen 2026-06-24; supersedes the suspended
  RQ011_1 incomplete-data run):
  `reports/studies/RQ011_onsite_full_universe_readiness/RQ011_2_onsite_readiness_20260623T201415+0800_efdd75a5/90_report/index.html`

## Latest Review Packets

- RQ001 Codex review:
  `reports/knowledge/RQ001_online_ipv_interval/reviews/codex_review.md`
- RQ002 Codex review:
  `reports/knowledge/RQ002_self_anchor_group_norm/reviews/codex_review.md`
- RQ004 Codex review:
  `reports/knowledge/RQ004_ipv_state_space/reviews/codex_review.md`
- RQ005 Codex review:
  `reports/knowledge/RQ005_nmi_evidence_gap/reviews/codex_review.md`
- RQ006 Codex review:
  `reports/knowledge/RQ006_sigma_sensitivity/reviews/codex_review.md`
- RQ007 Codex review:
  `reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/reviews/codex_review.md`
- RQ008 Codex review:
  `reports/knowledge/RQ008_interhub_temporal_ipv_discovery/reviews/codex_review.md`
- RQ010 Codex review:
  `reports/knowledge/RQ010_wod_e2e_tracking_feasibility/reviews/codex_review.md`
- RQ011 Codex review:
  `reports/knowledge/RQ011_onsite_full_universe_readiness/reviews/codex_review.md`
- RQ012 Codex review:
  `reports/knowledge/RQ012_onsite_event_annotation_readiness/reviews/codex_review.md`

These review packets are evidence-boundary reviews, not accepted
`decision.md` freezes.

## Latest Decision Packets

- RQ007 accepted development/guard estimability boundary:
  `reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/decision.md`
- RQ008 accepted negative temporal-discovery boundary:
  `reports/knowledge/RQ008_interhub_temporal_ipv_discovery/decision.md`
- RQ010 accepted WOD-E2E feasibility/tracking-route boundary:
  `reports/knowledge/RQ010_wod_e2e_tracking_feasibility/decision.md`
- RQ011 accepted OnSite readiness/scope boundary:
  `reports/knowledge/RQ011_onsite_full_universe_readiness/decision.md`
- RQ012 frozen Wave-A readiness deferral:
  `reports/knowledge/RQ012_onsite_event_annotation_readiness/decision.md`

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

- Raw/local data under `data/interhub/raw/`,
  `data/onsite_competition/all_teams_dataset/`,
  `archived/onsite_competition_raw_and_top5_subset_20260623/`, and
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
- RQ004 supports a state-conditioned IPV response surface, not a universal
  state-space law; RQ005 supports a verifier framework/gap record, not deployed
  early warning or planner performance; RQ006 supports sigma=0.1 as the healthier
  current source, while sigma remains a numeric sensitivity boundary.
- RQ007 supports an estimability/measurement boundary, not causal interaction
  timing or held-out confirmation; RQ008 supports a negative directional
  temporal-discovery result, not proof that all temporal dynamics are absent.
- RQ010 is feasibility-only and requires full tracking before WOD-E2E IPV use;
  RQ011 separates full_300 outcomes from clean_285 replay/IPV surfaces; RQ012 is
  annotation-readiness only and remains blocked for human labels.
