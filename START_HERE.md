# START_HERE: Current Operating Brief

Last reviewed: 2026-06-24.

Use this file as the first stop for a new agent thread. Keep durable policy in
`AGENTS.md`, architecture notes in `PROJECT_STRUCTURE.md`, and the compact research
question index in `STUDIES.md`.

## Current Active Context

- **Primary active research:** RQ009 estimability-aware dynamic counterpart-conditioned human
  envelope. PI authorized launch; independent plan review is the first gate.
- RQ009 plan:
  `reports/plans/RQ009_plan_v0_dynamic_counterpart_conditioned_envelope_20260624.md`.
- RQ009 main-agent prompt:
  `reports/plans/prompts/RQ009_prompt_claude_codex_orchestration_20260624.md`.
- RQ007 held-out remains sealed. RQ009 must freeze all rules and stop at
  `READY_FOR_SEALED_TEST` until a new PI authorization opens it.
- RQ008B is not authorized; no RQ008 motif may enter RQ009.
- External-validation priority after RQ009: **OnSite first**, WOD-E2E tracking pilot in
  parallel.
- Two-human RQ012 annotation is deferred; RQ012 remains `BLOCKED_FOR_HUMAN_LABELS`.
- The current paper baseline is paper-repository `main` merge `c6783577`; `structure.md` is
  v4.1 estimability-aware dynamic norm and must supersede v3 self-anchor round-trips.
- The 20260612 sigma 0.1 full-rerun data source is under
  `data/derived/interhub/20260612_sigma_0_1_full_rerun/00_hpc_outputs/`.

## Canonical Code Entrypoints

- Core IPV package: `src/sociality_estimation/core/`.
- Planning and geometry helpers: `src/sociality_estimation/planning/`.
- Active InterHub CSV/pkl pipeline: `pipelines/interhub/process_interhub.py`.
- Active simulation entrypoint: `pipelines/simulation/simulator.py`.
- InterHub helper/report scripts: `pipelines/interhub/tools/`.
- Old root wrappers are archived under `archived/compat_wrappers_20260619/`.

## Convenience Launchers

- macOS launchers: `scripts/launch_claude.command` and `scripts/launch_codex.command`.
- They enter the project root and start the corresponding CLI through the current team launcher.
- If the launcher or CLI is unavailable, leave the Terminal window open for diagnosis.

## Canonical Research Paths

- Compact index: `STUDIES.md`.
- Program dashboard: `reports/knowledge/RQ_PROGRESS_DASHBOARD.md`.
- Machine registry: `reports/knowledge/rq_progress_registry.csv`.
- Centralized plans/prompts: `reports/plans/`.
- Execution/report layer: `reports/studies/`.
- Interpretation/review/decision layer: `reports/knowledge/`.
- `reports/` has three governed first-level directories: `plans/`, `studies/`, `knowledge/`.
- Large derived outputs: `data/derived/`.
- Report-linked process archives and local agent state:
  `archived/report_process/` and `archived/report_local_state/`.
- Manuscript drafting lives in the standalone paper repository:
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
| RQ008 temporal IPV discovery | `reports/studies/RQ008_interhub_temporal_ipv_discovery/` | `reports/knowledge/RQ008_interhub_temporal_ipv_discovery/` |
| RQ009 dynamic counterpart envelope | `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/` | `reports/knowledge/RQ009_dynamic_counterpart_conditioned_envelope/` |
| RQ010 WOD-E2E tracking feasibility | `reports/studies/RQ010_wod_e2e_tracking_feasibility/` | `reports/knowledge/RQ010_wod_e2e_tracking_feasibility/` |
| RQ011 OnSite readiness | `reports/studies/RQ011_onsite_full_universe_readiness/` | `reports/knowledge/RQ011_onsite_full_universe_readiness/` |
| RQ012 event/annotation readiness | `reports/studies/RQ012_onsite_event_annotation_readiness/` | `reports/knowledge/RQ012_onsite_event_annotation_readiness/` |
| RQ013 beyond-safety utility | `reports/studies/RQ013_beyond_safety_incremental_validity/` | `reports/knowledge/RQ013_beyond_safety_incremental_validity/` |

For parallel agent runs under one RQ, the number after the RQ stem is the execution version.
Each execution must create a unique atomically locked RUN_ID/RUN_ROOT.

## Current PI Decisions

- Launch RQ009 now.
- Do not run RQ008B.
- Keep RQ007 held-out sealed until RQ009 reaches its pre-opening freeze; request a new PI
  authorization before any read.
- Defer two-human RQ012 annotation.
- Authorize WOD-E2E signed-in manifest/pilot work in principle; account/licence/login must be
  completed by the user.
- Prioritize OnSite RQ011B after RQ009; WOD proceeds in parallel.
- Use paper `main` commit `c6783577` as the current v4.1 baseline.

## Canonical Data Paths

- InterHub subset CSV:
  `data/interhub/raw/subsets_for_yiru/selected_interactive_segments_equalized.csv`
- InterHub subset pkl root: `data/interhub/raw/subsets_for_yiru/pkl/`
- InterHub full-dataset raw data: `data/interhub/raw/full_datasets/`
- InterHub sigma 0.1 time-series and full-rerun outputs:
  `data/derived/interhub/20260612_sigma_0_1_full_rerun/00_hpc_outputs/`
- OnSite all-team package, generated locally and ignored:
  `data/onsite_competition/all_teams_dataset/`
  (rebuild with `scripts/build_onsite_all_teams_dataset.py`).
- OnSite lightweight manifests: `data/onsite_competition/00_manifest/`.
- OnSite archived raw/top-five payload:
  `archived/onsite_competition_raw_and_top5_subset_20260623/`.
- Legacy Argoverse source data: `archived/argoverse/0_souce_data/`.

## Key Report And Decision Entries

- RQ003 Tier B validation:
  `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/00_entry/index.html`
- RQ007 estimability report:
  `reports/studies/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/00_entry/index.html`
- RQ007 decision:
  `reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/decision.md`
- RQ008 negative temporal-discovery report:
  `reports/studies/RQ008_interhub_temporal_ipv_discovery/RQ008_1_temporal_ipv_discovery_20260622T234914+0800_3e3e776a/00_entry/index.html`
- RQ008 decision:
  `reports/knowledge/RQ008_interhub_temporal_ipv_discovery/decision.md`
- RQ010 tracking-feasibility report:
  `reports/studies/RQ010_wod_e2e_tracking_feasibility/RQ010_1_wod_tracking_feasibility_20260623T073830+0800_14f21d3e/00_entry/index.html`
- RQ010 decision:
  `reports/knowledge/RQ010_wod_e2e_tracking_feasibility/decision.md`
- RQ011 complete-data readiness report:
  `reports/studies/RQ011_onsite_full_universe_readiness/RQ011_2_onsite_readiness_20260623T201415+0800_efdd75a5/90_report/index.html`
- RQ011 decision:
  `reports/knowledge/RQ011_onsite_full_universe_readiness/decision.md`
- RQ012 readiness report:
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37/90_report/index.html`
- RQ012 decision:
  `reports/knowledge/RQ012_onsite_event_annotation_readiness/decision.md`

## How To Run Tests

- Launcher checks: `python3 -m unittest tests.test_shortcut_scripts -q`.
- Broader suite when available: `python -m pytest tests -q`.
- Syntax check:
  `python -m py_compile src/sociality_estimation/core/agent.py src/sociality_estimation/core/ipv_estimation.py src/sociality_estimation/planning/Lattice.py src/sociality_estimation/planning/lattice_planner.py src/sociality_estimation/planning/utility.py pipelines/interhub/process_interhub.py pipelines/simulation/simulator.py`.
- One-case InterHub smoke:
  `python pipelines/interhub/process_interhub.py --limit 1 --workers 1 --solver-preset realtime --no-plots --output-root data/derived/interhub/_codex_runtime_smoke`.
- Record any durable dependency change in requirements or `main_workflow.log`.

## What Not To Delete

- Raw/local data under `data/interhub/raw/`, `data/onsite_competition/all_teams_dataset/`,
  `archived/onsite_competition_raw_and_top5_subset_20260623/`, and
  `archived/argoverse/0_souce_data/`.
- Derived InterHub full-rerun outputs under `data/derived/interhub/`.
- Plans/prompts under `reports/plans/`.
- Reader-facing study report packages under `reports/studies/`.
- Knowledge decisions and manuscript context under `reports/knowledge/`.
- Report-linked process archives under `archived/report_process/`.
- `main_workflow.log`, `AGENTS.md`, `START_HERE.md`, `PROJECT_STRUCTURE.md`, and `STUDIES.md`.

## Known Weak Spots

- NuPlan remains the weakest realtime IPV slice; no dataset-specific >90% guarantee.
- Self-anchor remains M4 ablation only, not normative authority.
- RQ007 is a development/guard estimability boundary; held-out is sealed and most gross
  concentration is proximity-driven.
- RQ008 supports a negative directional temporal-discovery boundary, not proof that all temporal
  dynamics are absent; RQ008B is currently not authorized.
- RQ009 must not read RQ007 sealed data until all rules/code/thresholds are frozen and the PI
  explicitly authorizes opening.
- RQ010 requires full tracking; exact data/HPC scale remains sign-in gated.
- RQ011 supports full_300 outcomes and clean_285 replay/IPV with T19 replay-only exclusion;
  run-level/repeated-run/causal claims are unavailable.
- RQ012 is readiness-only and human annotation is deferred.
- Paper `main` is v4.1 but still carries evidence/external-pending markers and is not submission-ready.
