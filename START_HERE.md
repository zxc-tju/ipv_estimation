# START_HERE: Current Operating Brief

Last reviewed: 2026-06-19.

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
  `reports/interhub/ipv_estimation_results/full_datasets/batches/20260612_sigma_0_1_full_rerun/`.

## Canonical Code Entrypoints

- Core IPV package: `src/sociality_estimation/core/`.
- Planning and geometry helpers: `src/sociality_estimation/planning/`.
- Active InterHub CSV/pkl pipeline:
  `pipelines/interhub/process_interhub.py`.
- Active simulation entrypoint: `pipelines/simulation/simulator.py`.
- InterHub helper/report scripts now live under `pipelines/interhub/tools/`.
- Old root compatibility wrappers were archived under
  `archived/compat_wrappers_20260619/`; do not use root-level commands/imports
  as active entrypoints.

## Canonical Data Paths

- InterHub subset CSV: `data/interhub/raw/subsets_for_yiru/selected_interactive_segments_equalized.csv`
- InterHub subset pkl root: `data/interhub/raw/subsets_for_yiru/pkl/`
- InterHub full-dataset raw data: `data/interhub/raw/full_datasets/`
- Legacy Argoverse source data: `archived/argoverse/0_souce_data/` (typo is part of the historical path)
- Onsite competition local payload:
  `data/onsite_competition/raw/` (large/untracked local data)
- Onsite competition organization manifests:
  `data/onsite_competition/00_manifest/`
- Onsite competition top-five manifests:
  `data/onsite_competition/top5_research_subset/`
- Onsite competition top-five raw/team payload:
  `data/onsite_competition/top5_research_subset/teams/`

## Canonical Output Paths

- InterHub subset IPV outputs: `reports/interhub/ipv_estimation_results/subsets_for_yiru/`
- InterHub full-dataset batches: `reports/interhub/ipv_estimation_results/full_datasets/batches/`
- Realtime validation artifacts: `reports/interhub/ipv_estimation_results/ipv_rt_final/`
- Runtime smoke outputs: `reports/interhub/ipv_estimation_results/_codex_runtime_smoke/`
- Online IPV interval-query report:
  `reports/interhub/ipv_estimation_results/full_datasets/batches/20260612_sigma_0_1_full_rerun/02_reports/online_ipv_interval_query_20260618/`
- Legacy Argoverse IPV outputs: `archived/argoverse/1_experiment_result/ipv_estimation/`
- Argoverse analysis outputs: `archived/argoverse/analysis_outputs/`
- NSFC top-five social-compliance evidence package:
  `reports/onsite_competition/social_compliance_report/competition_evidence_from_raw_data/`

## How To Run Tests

- Current working tree has no active `tests/` suite after cleanup. If tests are
  restored, run: `python -m pytest tests -q`.
- If a verification command fails only because a Python dependency is missing,
  install the missing package in the active project environment and continue;
  record durable dependency changes in requirements or `main_workflow.log`.
- Baseline syntax check for active code:
  `python -m py_compile src/sociality_estimation/core/agent.py src/sociality_estimation/core/ipv_estimation.py src/sociality_estimation/planning/Lattice.py src/sociality_estimation/planning/lattice_planner.py src/sociality_estimation/planning/utility.py pipelines/interhub/process_interhub.py pipelines/simulation/simulator.py`.
- Legacy scripts are archived under `archived/legacy_scripts/`; restore and
  path-check them before treating them as active commands.
- One-case InterHub smoke check:
  `python pipelines/interhub/process_interhub.py --limit 1 --workers 1 --solver-preset realtime --no-plots --output-root reports/interhub/ipv_estimation_results/_codex_runtime_smoke`.
- For scientific or publication claims, rerun the relevant pipeline on a
  representative sample and compare generated CSV/figure/report artifacts.

## How To Start App/Server

- No persistent app or web server is required. This repository is script- and
  report-oriented.
- Static report HTML files can be opened directly from disk.

## What Not To Delete

- Raw data under `data/interhub/raw/`, `data/onsite_competition/raw/`,
  `data/onsite_competition/top5_research_subset/teams/`, and
  `archived/argoverse/0_souce_data/`.
- Reader-facing reports, evidence bundles, and summary tables under
  `reports/interhub/ipv_estimation_results/`,
  `reports/onsite_competition/`, and `argoverse/analysis_outputs/`.
- Lightweight manifest tables under `data/onsite_competition/`.
- Report-linked process archives, especially `01_process/` folders and
  `hpc_run_files/` READMEs/scripts.
- `main_workflow.log`, `AGENTS.md`, and this `START_HERE.md`.
- Large archives such as curated valid case zips unless the user explicitly
  approves cleanup. Delete only reproducible caches by default.

## Latest Stable Report/Result

- Realtime IPV estimator summary: `docs/realtime_ipv_estimator.md`.
- Latest mixed-distribution realtime sign validation:
  `reports/interhub/ipv_estimation_results/ipv_rt_final/evidence/current5_50pd_w10_pkl_*`.
- Key stable numbers from the documented 200-case mixed validation: overall
  sign accuracy 92.3%, Wilson 95% lower bound 91.3%, mean latency 0.110 s,
  median latency 0.099 s.
- Stable subset report: `reports/interhub/ipv_estimation_results/subsets_for_yiru/ipv_distribution_report/ipv_distribution_report.html`.
- Full-dataset sigma 0.1 rerun package:
  `reports/interhub/ipv_estimation_results/full_datasets/batches/20260612_sigma_0_1_full_rerun/`.
- Organized report entries for that package are indexed at:
  `reports/interhub/ipv_estimation_results/full_datasets/batches/20260612_sigma_0_1_full_rerun/02_reports/README.md`.
  Each report package uses `00_entry/index.html`, `01_results/`, and
  `02_process/` plus a `TRACEABILITY.md` map.
- Latest deployable online IPV interval report (2026-06-19, SUPERSEDES the 2026-06-18 interval-query result below):
  `reports/interhub/ipv_estimation_results/full_datasets/batches/20260612_sigma_0_1_full_rerun/02_reports/3-online_ipv_interval-2/00_entry/index.html`
  (structured package: `00_entry/` + `01_results/` + `02_process/` + `README.md`/`TRACEABILITY.md`).
  Self-contained Nature/NMI-style HTML (4 inline SVG figures + locked tables). Recommended
  scheme = lane-referenced causal rolling-IPV self-anchor + split-conformal. LOCKED on a
  balanced 5,000-case / 10,000-row causal rebuild (Wilson 95% CI): primary TEST coverage
  0.899 [.885,.911] / width 0.591 (−31.8% vs oracle PET, −19.9% vs no-roll kinematics);
  Leave-Waymo-Out coverage 0.902 [.894,.910] / width 0.628 — the ONLY method reaching
  nominal 0.90 under cross-dataset shift. The prefix-only `RealtimeIPVEstimator` rebuild the
  prior report flagged as required is DONE: with a map-lane-centerline reference, prefix-only
  causal IPV reproduces the offline signal at corr 0.993. Production estimator
  (`.../02_reports/3-online_ipv_interval-2/01_results/model_balanced_lock_cqr.joblib` + `02_process/scripts/predict_interval_balanced_lock.py`),
  verifier-integration module, and A/B vs the PET-bin envelope are included; fall back to
  no-roll kinematic CQR (−25–30%) when lane/route is unknown (~26% of cases).
- Latest self-anchor group-norm validation report (2026-06-19):
  `reports/interhub/ipv_estimation_results/full_datasets/batches/20260612_sigma_0_1_full_rerun/02_reports/4-self_anchor_validation-1/00_entry/index.html`.
  Formal E1-E5 preregistered validation returns **MUST REVISE** for the strong
  "self-anchor alone is a group norm" narrative. E1 fails strict time separation
  because the locked full-window label overlaps the early 2 s anchor; E2 true
  held-out-individual validation is not assessable without persistent driver IDs;
  E3 stress + independent replication support out-of-support flag/abstention
  against self-consistent deviators; E4 flags disposition-residual exposure; E5
  fails because self-pass/situation-flag disagreement cases are enriched for bad
  outcome proxies. Use hybrid wording/design: self-anchor for sharpness plus
  situation floor and out-of-support abstention.
- Parallel independent Codex validation package for the same E1-E5 plan
  (2026-06-19; separate write scope to avoid concurrent-agent interference):
  `reports/interhub/ipv_estimation_results/full_datasets/batches/20260612_sigma_0_1_full_rerun/02_reports/4-self_anchor_validation-codex-20260619_182448-01/00_entry/index.html`.
  It also returns **MUST REVISE**: E1 overlap mean 0.952, E2 LWO coverage 0.902
  and scenario-family coverage 0.903, E3 main stress passes but independent
  empirical-bin replication flags medium-delta washout risk, E4 confirms
  self-anchor sharpness with material residual exposure, and E5 fails because
  self-anchor does not outperform situation-only on `oracle_pet<=1` bad-outcome
  tracking.
- Prior online IPV interval-query result (2026-06-18, superseded; kept for history):
  `reports/interhub/ipv_estimation_results/full_datasets/batches/20260612_sigma_0_1_full_rerun/02_reports/online_ipv_interval_query_20260618/00_entry/index.html`.
  Tier 1 strict-online kinematic CQR + split conformal; Tier 0 global floor fallback; Tier 2
  rolling-IPV was the candidate pending the prefix-only rebuild now completed above.
- Latest current-frame online IPV reasonableness result:
  `reports/interhub/ipv_estimation_results/full_datasets/batches/20260612_sigma_0_1_full_rerun/02_reports/online_current_ipv_distribution_20260618/00_entry/index.html`.
  Best normal-operation method is full-context paired source-guard CQR
  (primary P90 coverage 0.904, mean width 0.859, Winkler 1.243). For
  unknown/shifted sources, use residual full-context paired source-guard
  conformal fallback (Leave-Waymo-Out P90 coverage 0.906, mean width 1.203).
- Latest NSFC top-five external-validation evidence package:
  `reports/onsite_competition/social_compliance_report/competition_evidence_from_raw_data/00_entry/core_results_nature.html`.
  Nature/NMI-style HTML entry with two embedded SVG figure plates, plus editable
  SVG/PDF/PNG exports under
  `reports/onsite_competition/social_compliance_report/competition_evidence_from_raw_data/01_results/nature_core_html/`.
- Latest NSFC + InterHub social-compliance submission synthesis:
  `reports/onsite_competition/social_compliance_report/competition_evidence_detailed_analysis/00_entry/index.html`.
  Six Python/matplotlib submission figures are exported as PNG/SVG under
  `reports/onsite_competition/social_compliance_report/competition_evidence_detailed_analysis/02_process/figures/`.
  Locked bounds: NSFC Tier-A premise is strong, NSFC Tier-B verifier criterion
  validity is null, and causal language is limited to triangulated support rather
  than identified causation.
- Source synthesis for the NSFC top-five evidence package:
  `reports/onsite_competition/social_compliance_report/competition_evidence_from_raw_data/00_entry/NSFC_evidence_synthesis_for_NMI.md`.
  It confirms official outcome anchors and feasible case-to-scenario alignment,
  but its `trajectory_friction_z` metric is an exploratory kinematic proxy, not
  formal IPV-envelope deviation.

## Known Weak Spots

- NuPlan is the weakest realtime IPV slice; current validation does not support
  a dataset-specific NuPlan >90% sign-accuracy guarantee.
- The online estimator is a statistical realtime solution, not a hard-deadline
  guarantee for every frame. Use asynchronous scheduling for strict control
  deadlines.
- Online IPV interval lookup should not rely on predicted PET/risk-bin lookup as
  the main accuracy lever: even ORACLE PET cell lookup is only ~3% narrower than the
  global interval (risk is not the lever). Direct conditional interval models are better;
  the 2026-06-19 lane-referenced causal rolling-IPV + conformal method now reaches nominal
  Leave-Waymo-Out coverage (0.902). The no-roll fallback and general unknown sources can
  still fall below nominal under shift and need source-specific calibration.
- RESOLVED 2026-06-19: rolling-IPV is now a prefix-causal deployable sharpness signal,
  but not sufficient alone as a validated group-norm verifier.
  Rebuilding the prefix features with `RealtimeIPVEstimator` + a map-lane-centerline
  reference reproduces the offline signal at corr 0.993; the locked balanced rebuild gives
  Leave-Waymo-Out coverage 0.902 (only method at nominal under shift). It is
  route-conditioned (lane/route known at decision time); ~26% of no-lane cases fall back to
  no-roll kinematic CQR. The self-anchor group-norm validation report requires a
  hybrid guardrail before manuscript/verifier claims: situation floor plus
  out-of-support abstention. Optional follow-ups: full-56k rebuild,
  source-specific recalibration of residual cross-source gaps, production CQR/HGB
  replay of E3-E5, and stronger external outcome anchors. See the 2026-06-19
  report entries above.
- Current-frame IPV reasonableness now has a stronger distribution-model result,
  but the lagged IPV history used in the experiment is still an HPC label proxy.
  Before deployment claims, repeat the same CQR/source-guard evaluation using
  prefix-only realtime IPV estimates and online reference-line features.
- `archived/legacy_scripts/batch_process_ipv.py` defaults to the old
  `interhub_traj_lane/ipv_estimation_results` layout, while current InterHub
  outputs use `reports/interhub/ipv_estimation_results/`. Check paths before
  restoring or running it.
- `pipelines/simulation/simulator.py` references external resources such as `NDS_analysis` that are
  not present in this repository snapshot.
- `requirements-minimal.txt` is referenced by older guidance but is not present
  in the current file list; use `requirements.txt` unless a minimal file is
  restored.
- The local onsite competition payload is now materialized under
  `data/onsite_competition/raw/shanghai/` and
  `data/onsite_competition/raw/beijing/`, with organization manifests under
  `data/onsite_competition/00_manifest/`. Use `session_manifest.csv` as the
  stable replay entry point. Current caveat: scored team `T18`/`bitsite`
  (`data/onsite_competition/raw/beijing/5-BIT_Site`) has scores/media but no
  materialized replay session directory; `raw/shanghai/长沙理工大学_csust` and
  `raw/beijing/8-高中部` appear to be extra media-only/non-scored folders.
- NSFC social-compliance evidence currently supports external-validation design
  and hypothesis mining, not a frozen manuscript claim that the formal IPV
  verifier has been validated on NSFC. Confirm Shanghai scenario mapping and run
  true InterHub-trained IPV normative-envelope deviation before main-text claims.
