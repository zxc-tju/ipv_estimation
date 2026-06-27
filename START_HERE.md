# START_HERE: Current Operating Brief

Last reviewed: 2026-06-27.

Use this file as the first stop for a new agent thread. Keep durable policy in
`AGENTS.md`, architecture notes in `PROJECT_STRUCTURE.md`, and the research
question index in `STUDIES.md`.

## Current Active Context

- RQ012B Stage 3+ OnSite M3-anchor enabling build now has AV-perspective
  clean_285 anchors under
  `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors/onsite_m3_av_anchors.parquet`:
  267/285 units covered, 267 anchors, 30/32 M3 required inputs fully populated.
  This build computes pinned-estimator IPV features/targets only; no
  event/outcome join or harm association has been run.
- RQ012B Stage 3+ frozen-M3 OOD/support gate is complete for the 267 OnSite
  AV anchors. As-is gate pass is 51/267 anchors/units, with 216 abstentions;
  units with usable deviation are 51. Dominant frozen hard-fail causes are
  k=25 distance over threshold 1.6072176695 for 136 category-eligible anchors
  and unsupported joint cells for 80 anchors (`F|equal|AV;HV`=60,
  `CP|equal|AV;HV`=20). `priority_role=equal`, geometry levels CP/F/MP, and
  `agent_type_pair=AV;HV` are individually supported in RQ009
  `ood_gate.json`; `apet_online_proxy` NaN is common (184 abstaining anchors)
  but the frozen distance gate already uses RQ009 train-median imputation.
  Sensitivities: drop `apet_online_proxy` from distance gives 61/267 usable;
  train-median imputation and literal equal-supported are no-ops; drop-apet
  plus equal-supported is also 61/267. Report:
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/ood_gate/ood_gate_report.md`.
  Scored data:
  `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/ood_gate/`.
- Primary technical context: realtime IPV estimator validation and InterHub
  CSV/pkl motion-data pipelines.
- Recommended online sign mode: `RealtimeIPVEstimator.for_realtime_sign(...)`
  with `history_window=10`, `max_workers=10`, and the five-candidate sign grid.
- Accuracy-preserving online value mode: `solver_preset="parallel_accurate"`
  with the legacy seven-candidate grid.
- The 20260612 sigma 0.1 full-rerun data source is now under
  `data/derived/interhub/20260612_sigma_0_1_full_rerun/00_hpc_outputs/`.
- HPC reuse: HPC (Tongji) reusable assets + how-to → `reports/knowledge/INFRA_hpc_tongji_reuse.md`.
- RQ010B WOD-E2E Tongji HPC basic parser access is now working under
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/`. The 2026-06-26 four-shard
  ratings-sealed structural pre-flight sampled all 12 candidate-bearing
  scene frames found in shards 00000..00003 and passed the five t* structural
  checks on those 12; the full 479-scene gate remains pending.
- RQ010B StreamPETR Route 4 Tongji HPC setup is now available under
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/`: code at `code/StreamPETR`,
  env at `envs/streampetr`, checkpoint at
  `checkpoints/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth`, and
  full R50 flash checkpoint dummy 6-camera forward passed on an L40 GPU node
  (`logs/streampetr_checkpoint_forward_flash_l40_20260626.log`, output
  `boxes_tensor_shape=[20, 9]`). Key versions: torch 1.13.0+cu117,
  CUDA module 11.8, mmcv 1.6.0, mmdet 2.28.2, mmdet3d 1.0.0rc6.
- RQ010B StreamPETR Route 4 real Waymo Perception lead-config smoke now passes
  on Tongji HPC L40: converter/dataset/config/smoke scripts live under
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/code/StreamPETR/`; one training
  segment was converted to
  `data/waymo_stream_petr/waymo_infos_train_1seg.pkl` with five forward cameras
  (`FRONT`, `FRONT_LEFT`, `FRONT_RIGHT`, `SIDE_LEFT`, `SIDE_RIGHT`), sample
  shape `img=[1,5,3,256,704]`, forward output `boxes=[300,9]`, and two-step
  forward/backward smoke loss decreased `400.0814 -> 316.3276`. Runbook:
  `code/StreamPETR/tools/waymo_perception/RQ010B_ROUTE4_WAYMO_STREAM_PETR_SMOKE.md`;
  key logs:
  `logs/waymo_sample_shape_l40_20260626.log`,
  `logs/waymo_forward_l40_20260626.log`,
  `logs/waymo_train_overfit_l40_20260626.log`.
- RQ010B Waymo Perception v1.4.3 small dev subset is now available on Tongji
  HPC under `/share/home/u25310231/ZXC/RQ010B_wod_e2e/data/perception/`: first
  4 deterministic-sorted training segments plus first 2 validation segments
  from `gs://waymo_open_dataset_v_1_4_3/individual_files/`, exact-size and
  crc32c verified. Manifest:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/manifests/perception_dev.tsv`.
- RQ010B Waymo Perception v1.4.3 finetune subset is now complete on Tongji
  HPC under `/share/home/u25310231/ZXC/RQ010B_wod_e2e/data/perception/`: first
  64 deterministic-sorted `.tfrecord` training segments plus first 16
  validation segments, exact-size and crc32c verified against Mac `gsutil stat`.
  Manifest:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/manifests/perception_finetune.tsv`
  (80 rows, 80 ok, total bytes 80,523,139,102). Transient token/temp/lock
  state was removed after completion; final rescue log:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/logs/perception_finetune_rescue_20260626T174924Z.log`.
- RQ010B StreamPETR Route 4 dev6 dry-run finetune now passes end-to-end on
  Tongji HPC L40. Converted infos are
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/data/waymo_stream_petr/waymo_infos_train_4seg.pkl`
  (794 train samples, 4 scenes) and
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/data/waymo_stream_petr/waymo_infos_val_2seg.pkl`
  (397 val samples, 2 scenes). Dry-run config:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/code/StreamPETR/projects/configs/StreamPETR/stream_petr_r50_flash_704_waymo_5cam_dev6_dryrun.py`;
  work dir:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/work_dirs/streampetr_waymo_dev6_dryrun_20260626/`.
  The clean 40-iter run saved `iter_20.pth` and `iter_40.pth`, loss decreased
  `73.2736 -> 40.4284`, and the lightweight Waymo center-distance smoke eval
  completed with `waymo_center_recall_2m=0.0` on 397 val samples. This metric
  validates the eval path only; it is not an accuracy claim.
- RQ010B StreamPETR Route 4 lead-config 64-train/16-val finetune was
  resubmitted as detached Tongji Slurm job `1707389` on 2026-06-27 02:48 CST
  after job `1707307` failed at MMCV config parse with
  `TypeError: cannot pickle '_io.BufferedReader' object`. The config fix
  deletes the closed `_handle` left by the parse-time train-info `with open`
  block, and login-node `Config.fromfile(...)` now returns
  `CONFIG_PARSE_OK`. Job `1707389` was `RUNNING` on L40 node `gpu4037` at
  launch check. Sbatch:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/scripts/streampetr_waymo_finetune64_leadcfg_20260627.sbatch`;
  config:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/code/StreamPETR/projects/configs/StreamPETR/stream_petr_r50_flash_704_waymo_5cam_finetune64_leadcfg.py`;
  work dir:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/work_dirs/streampetr_waymo_finetune64_leadcfg_20260627/`;
  Slurm logs:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/logs/streampetr_waymo_finetune64_1707389.out`
  and `.err`. The job first converts
  `waymo_infos_train_64seg.pkl` and `waymo_infos_val_16seg.pkl`, then runs a
  12-pass fixed-iteration one-L40 schedule with eval/checkpoint each pass,
  `max_keep_ckpts=3`, best-checkpoint saving on `waymo_center_recall_2m`,
  `--time=36:00:00`, estimated about 30 GPU-hours. This is an engineering
  finetune launch only; no result or research claim is available yet.
- RQ009 Phase 3 features gate is now PASS. The hw=4 target source remains
  the verified frame-level `sigma01_hw4_ipv_timeseries.csv` with 3,695,981
  data rows, exact key overlap with sigma01, SHA-256
  `cf970f01455905000dac4f24909e69f532e21014987a52a541466a2748fd34fc`,
  and 12-case hw=4 parity `max_abs_diff=0.0`; the assembled feature matrix
  has 6,397,266 perspective-anchor rows under
  `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix/`.
  Independent Phase 3.5 matrix audit `RQ009-W3-matrix-audit` is PASS with
  target t*+6 re-derivation `max_abs_diff=0.0` on 400 anchors, full-scale
  case-split no-bleed, max numeric M3 feature-target |corr| `0.1146074993`,
  and leakage-probe test R2 `0.2811922275`; report:
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/03_features/matrix_audit.md`.
- RQ009 IPV estimator divergence investigation (`INV-ipv-code-diff`, 2026-06-27)
  found genuine local logic drift from the sigma01-generation estimator. The
  package refactor was `0827f504` on 2026-06-19, but the post-sigma01 numeric
  drift was introduced by `a0fee535` on 2026-06-15 in the vectorized
  `cal_individual_cost`/`cal_group_cost` objective helpers. Matched-parameter
  direct A/B on 4 real parity cases gave `max_abs_diff=0.2811681005`; restoring
  the legacy loop helpers reproduced the pinned legacy diagnostic exactly.
  Follow-up `INV-ipv-accel-hyperparam` tested solver/grid accuracy knobs and
  found no config-only high-accuracy setting restores sigma01 parity
  (`parallel_accurate` is speed-only; strict SLSQP and denser IPV grids still
  differ). Use the pinned HPC legacy estimator for sigma01-compatible IPV.
  Notes: `reports/knowledge/ipv_estimator_divergence_investigation.md` and
  `reports/knowledge/ipv_accel_hyperparam_finding.md`.
- RQ009 Phase 4 calibration and independent Phase 4.5 calibration-integrity
  audit are now PASS. M3 test coverage reproduces at 80=`0.8162154701`,
  90=`0.8986657101`, and 95=`0.9496345436`; M3 conformal radii reproduce
  from calibration only at 80=`-0.0041994299`, 90=`-0.0080911424`, and
  95=`-0.0054183006`; no test-fold leakage was detected and calibration/test
  scene/case overlap is zero. Audit caveat: the `1e-10` endpoint nudge changes
  exact M3 80% boundary-tie coverage but only for rows within the 1e-10
  tolerance; this is recorded as nonblocking. Reports:
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/04_calibration/calibration_report.md`
  and
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/04_calibration/calibration_audit.md`.
- RQ012B Stage 3+ callable frozen-M3 scorer build is PASS. The scorer reuses
  RQ009 calibration code, refits only M3 per-quantile HGB models with the
  frozen selected hyperparameters and seed `20260626`, uses saved M3 conformal
  radii and saved OOD gate threshold/support parameters, and touches no OnSite
  outcomes. Serialized scorer/helper/contract:
  `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/m3_scorer/`.
  Refit/parity/provenance:
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/m3_refit/`.
  Saved RQ009 M3 calibration+test prediction parity is exact for materialized
  q/interval/abstain columns (`max_abs_diff=0.0`, fraction within `1e-5` =
  `1.0`); scorer SHA-256
  `bf9a0c7ae41ba9efcb2ad997aaac1b7881d7788cf8dadd01252c17ed7a6b0ba5`.
- RQ009 Phase 6 M3-vs-M4 gate is PASS with no stable incremental
  counterpart-IPV interval value over `ipv_removed` (M3 90% Winkler
  difference `-0.000211426`, case-cluster 95% CI
  `[-0.0018861798657293647, 0.00150497909450504]`). Phase 6.5b
  exploration group G5/C15 is complete on guard_tune only: dependency is not
  approximately zero by the partial-Pearson threshold (`0.0315120479`), but
  point and interval screens do not improve the matched control
  (`dR2=-0.003274902`, `dMAE=-0.0001594282`,
  `dWinkler90%=0.1898183`, coverage delta `0.00001536`). Phase 6.5b
  exploration group G2/C05,C08,C09 is complete on guard_tune only:
  dependencies were nonzero (`partial_r=-0.1411410`, `-0.1160929`,
  `-0.1009250`), all point screens were worse than matched controls
  (least-bad `C09 dR2=-0.0068356`, `dMAE=0.0002454`), and the best interval
  screen was C05 with a small Winkler reduction (`dWinkler90%=-0.9070%`, 95%
  CI `[-1.0933%, -0.7357%]`, coverage rule OK) below the 5% meaningful-effect
  threshold. Phase 6.5b
  exploration group G3/C06-C07/C10-C12 is also complete on guard_tune only:
  C10/C11 were sparse/ineligible with zero matching guard rows; C06/C07/C12
  dependencies were nonzero (`partial_r=-0.1492908`, `-0.1591437`,
  `-0.1182023`), point screens were worse than matched controls (least-bad
  `C12 dR2=-0.0132819`; least-bad MAE `C07 dMAE=0.0008816`), and only C06
  showed a small interval Winkler reduction (`dWinkler90%=-0.5748%`, 95% CI
  `[-0.7061%, -0.4280%]`, coverage rule OK) below the 5% meaningful-effect
  threshold. Phase 6.5b
  exploration group G1/C01-C04 is complete on guard_tune only: all dependency
  probes were nonzero (`partial_r=-0.1390962`, `-0.2381406`,
  `-0.2212075`, `-0.0319065`), all point screens were worse than matched
  controls (least-bad `C04 dR2=-0.0009120`, `dMAE=0.0007248`), and the best
  interval screen was C03 with a small Winkler reduction
  (`dWinkler90%=-1.1902%`, 95% CI `[-1.5128%, -0.8373%]`, coverage rule OK)
  below the 5% meaningful-effect threshold. Phase 6.5b
  exploration group G4/C13-C14 is complete on guard_tune only: both dependency
  probes were nonzero (`partial_r=-0.1623032` for both C13 and C14); C13 was
  worse on point metrics (`dR2=-0.0075118`, `dMAE=0.0004988`), while C14 had a
  tiny MAE reduction but worse R2/MSE (`dR2=-0.0009504`,
  `dMAE=-0.0001637`, `dMSE=0.0001786`). Both interval screens showed small
  Winkler reductions with coverage rule OK (`C13 dWinkler90%=-0.4868%`, 95%
  CI `[-0.6499%, -0.3213%]`; `C14 dWinkler90%=-0.3591%`, 95% CI
  `[-0.4368%, -0.2757%]`), below the 5% meaningful-effect threshold. Phase
  6.5c exploration synthesis is PASS and formalizes the guard_tune verdict as
  `null_confirmed`: all 15 candidates were aggregated, no point or interval
  screen was both BH-significant and pre-registered meaningful, the best point
  dR2 remained negative (`C04=-0.0009120`), the best interval Winkler reduction
  was small (`C03=-1.1902%`, below the 5% bar), and no test confirmation was
  triggered. The test fold remains untouched. Dependency caveat: raw engineered
  dependency probes were often nonzero (max absolute partial r `C02=0.2381406`);
  C15's orthogonalized probe was small (`partial_r=0.0315120`,
  `spearman=0.0142350`, `CMI=0.0006338`) but above the strict preregistered
  `<0.02` partial-r approximate-zero threshold, so the robust-null claim is a
  performance/adaptation null rather than a literal independence claim.
  Artifacts:
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/06_m3_vs_m4/m3_vs_m4_verdict.md`
  and
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/06_m3_vs_m4/exploration/results/results_G5.md`
  plus
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/06_m3_vs_m4/exploration/results/results_G2.md`
  plus
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/06_m3_vs_m4/exploration/results/results_G3.md`
  plus
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/06_m3_vs_m4/exploration/results/results_G1.md`
  plus
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/06_m3_vs_m4/exploration/results/results_G4.md`
  plus
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/06_m3_vs_m4/exploration/exploration_verdict.md`
  and
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/06_m3_vs_m4/exploration/exploration_master_table.csv`.
- RQ009 Phase 6.5d longer-horizon sweep is complete under
  `02_process/06_m3_vs_m4/exploration/horizon/`. Registered horizons
  `h={6,8,11,13,16,18,21}` were run on train/calibration/guard_tune only with
  lookup targets from the frame-level hw=4 time series; the test fold remains
  untouched because no horizon/encoding cleared the guard_tune bar. Decision:
  `null_across_horizons` for point/interval adaptation. Eligibility shrank from
  5,126,700 analysis anchors / 30,566 cases at h=6 to 4,215,974 anchors /
  29,972 cases at h=21. Best point screen was h=6 C08
  (`dR2=0.001715`, below the 0.02 bar and MAE worse); best interval screen was
  h=6 C08 (`dWinkler90%=-0.2405%`, far below the 5% bar). Dependency caveat:
  the registered raw/ego-controlled counterpart-current probe did not reproduce
  the prior C15-style h=6 near-zero partial-r sanity (`h6=-0.1090`, max abs
  `0.1254` at h=18), so treat the horizon conclusion as a guarded
  point/interval adaptation null rather than a literal independence result.
  Artifacts:
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/06_m3_vs_m4/exploration/horizon/horizon_verdict.md`,
  `horizon_verdict.json`, `horizon_sweep.csv`, and
  `horizon_partial_r_curve.csv`/`.svg`.
- RQ009 Phase 6.5e dependency reconciliation is complete under
  `02_process/06_m3_vs_m4/exploration/reconcile/`. Canonical residualization
  uses `counterpart_ipv_current` only, with base context plus M4 ego
  self-anchor current/error/slope, and HistGradientBoosting residualizers for
  ego target and counterpart current. The h=6 canonical partial r is
  in-sample `-0.039023`, train-to-guard row-level `-0.037680`, and held-out
  case-level `0.057521`; across h={6,11,16,21}, train-to-guard row-level r
  stays small (`-0.0345` to `-0.0393`) and does not grow with horizon, while
  held-out case-level r flips positive and is not monotone (`0.0226` to
  `0.0575`). C15 vs horizon differed because C15 used an orthogonalized
  five-column counterpart-block PCA component screen with max-absolute
  component correlations, while the horizon sweep used signed raw
  `counterpart_ipv_current` with Ridge residualization and ego-self controls.
  Predictive reconciliation: Ridge dR2 is small but positive
  (`0.010609`, `0.014090`, `0.015037` for h=6,16,21), below the 2% bar, while
  flexible HGB dR2 is negative at all checked horizons (`-0.005250`,
  `-0.005854`, `-0.012110`). Verdict: `robust_null` for a real,
  cross-case-generalizing counterpart-current dependency; row-level negative
  sign, where present, would be compensatory under the local convention
  `theta>0` prosocial/yielding. The test fold remains untouched. Artifacts:
  `dependency_reconcile.md`, `dependency_reconcile.json`, and
  `run_dependency_reconcile.py`.
- RQ009 Phase 7 perturbation sensitivity is complete under
  `02_process/07_perturbation/`. The worker used outcome-blind,
  source-stratified case subsamples from train/calibration/guard_tune only
  (targets 50k/35k/45k rows before full-case overshoot; guard_tune reporting;
  test fold not read) and refit/recalibrated M2/M3/M4/`ipv_removed` for
  feature-window, counterpart-noise, missingness, OOD-gate, subsample-seed,
  and target-horizon perturbations. M3 90% guard_tune coverage ranged
  `0.8754..0.9119`, mean width `0.9703..1.1144`, Winkler `1.5265..1.7217`,
  and paired M3-vs-`ipv_removed` relative Winkler gain ranged
  `-0.541%..1.193%`. No validity break outside +/-3 pp and no meaningful
  counterpart-null flip were found; gate booleans are
  `validity_robust=true`, `null_robust=true`. Artifacts:
  `perturbation_results.csv`, `perturbation_report.md`,
  `perturbation_gate.json`, and `perturbation.py`.
- RQ009 Phase 8 independent end-to-end review (`RQ009-W8-review`) is PASS
  under `02_process/08_review/`. The review found no blocking or major issues:
  contract adherence is OK, leakage controls are clean, M3 marginal
  gate-passing conformal validity is sound, and the counterpart-IPV null is
  defensible as a bounded performance/adaptation null rather than a literal
  independence claim. Independent spot checks reproduced the feature matrix
  counts (`6,397,266` rows, no case split bleed), target timing (`+2/+6`),
  M3 calibration radii and test coverage, calibration/test case overlap `0`,
  and the effective scored target zero atom (`273,819 / 1,270,566`). Two minor
  hygiene findings remain for final packaging: top-level run status/index
  artifacts are stale, and exploration p-values were reconstructed from saved
  CIs because raw bootstrap draws/case-level paired differences were not
  retained. Artifacts: `independent_review.md`, `review_findings.csv`,
  `review_gate.json`, and `execution_log.md`.
- RQ009 Phase 10 clean-room replication (`RQ009-W10-replication`) is complete
  under `02_process/10_replication/` with status `FAIL` for a documented
  divergence rather than a leakage/blocking failure. The independent route used
  HGB quantile CQR, linear conformal quantiles, and a train/guard robust-distance
  support gate without importing original calibration/evaluation code. M3
  coverage agrees at 90% (`0.898762` vs original `0.898666`) with 90% width
  `1.067353` vs original `1.016152`; M3-vs-M4 agrees in direction and scale
  (`width +4.929%`, `Winkler +1.979%`, both below PI escalation bars, vs
  original `+2.960%` and `+2.784%`). The paired counterpart-null diverges:
  M3-minus-`ipv_removed` interval-score difference is small in practical terms
  (`-0.004664`, `-0.334%`) but case sign-test p=`1.72e-10`, unlike the original
  near-zero/sign p=`0.8629`; held-out row-level dependency remains small and
  agrees (`r=-0.0234` vs canonical about `-0.04`). Artifacts:
  `replicate.py`, `replication_results.csv`, `replication_report.md`,
  `replication_gate.json`, and `execution_log.md`.
- RQ009 Phase 10b replication-null reconciliation (`RQ009-W10b-recon`) is
  complete under `02_process/10_replication/` with the practical null
  reconciled. Frozen M3 and `ipv_removed` test predictions reproduce the
  original 90% paired row-weighted Winkler difference
  `-0.000211426` (`-0.014856%` of `ipv_removed`), case sign-test
  p=`0.862943`, case Wilcoxon p=`0.522202`, and case-cluster CI containing
  zero (recomputed `[-0.001848, 0.001454]`; original Phase 6
  `[-0.001886, 0.001505]`). Row-level sign/Wilcoxon tests are tiny-p because
  they count 1,209,857 autocorrelated anchors; the naive paired t-test is not
  significant (`p=0.534857`). The clean-room `-0.333852%` result remains far
  below the 5% meaningful-effect bar, and its saved p-value `1.7158e-10` is
  labeled `paired_case_sign_p` in the replication code/results, not row-level.
  Artifacts: `replication_reconcile.md` and `replication_reconcile.json`.
- RQ009 Phase 11 visualization/report package (`RQ009-W11-report`) is PASS.
  Nature-skill figure generation was available and used; the offline bilingual
  report package has seven conclusion-owned figure groups, 14 evidence rows,
  and a PASS `report_gate.json` with `offline_ok=true`. Entry point:
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/00_entry/index.html`.
  English/Chinese reports:
  `90_report/index.html` and `90_report/index.zh.html`. Figure manifest:
  `01_results/figures/figure_manifest.csv`. The package headline is: marginally
  valid CQR envelope, counterpart-IPV practically null, and adaptation encoded
  mainly in kinematics/context.
- RQ009 Phase 12a final independent report review
  (`RQ009-W12a-final-review`) is PASS and `ready_to_register=true` after the
  W11b report fix. The final review verified offline EN/ZH link resolution,
  bilingual correspondence, C1-C7 figure binding, headline number consistency,
  honest practical-null/marginal-validity boundaries, and `evidence.csv`
  consistency with zero blocking, major, or minor findings. Artifacts:
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/12_final_review/final_review.md`,
  `final_review_findings.csv`, and `final_review_gate.json`. No report,
  registry, contract, or paper-repository files were edited by the final
  reviewer.

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
- RQ009 dynamic envelope hw=4 target source:
  `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/target_hw4/sigma01_hw4_ipv_timeseries.csv`
  (verification report:
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/03_features/target_hw4_fetch.md`).
- RQ009 dynamic envelope Phase 3 feature matrix:
  `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix/`
  with gate:
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/03_features/features_gate.json`.
  Independent Phase 3.5 audit:
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/03_features/matrix_audit.md`
  and
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/03_features/matrix_audit.json`.
- RQ010B WOD-E2E Tongji HPC work root:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/`; parser env at
  `envs/e2e`, structural pre-flight code at `src/e2e_structural_preflight.py`,
  and latest four-shard result at
  `results/e2e_structural_preflight_4shards_20260626.json`. StreamPETR Route 4
  setup is at `code/StreamPETR` with env `envs/streampetr`, checkpoint
  `checkpoints/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth`, and latest
  checkpoint smoke log `logs/streampetr_checkpoint_forward_flash_l40_20260626.log`.
  Waymo Perception v1.4.3 dev and finetune subsets for StreamPETR
  dataloader/calibration work are at `data/perception/{training,validation}/`
  with manifests `manifests/perception_dev.tsv` (6 files, all crc32c matched on
  2026-06-26) and `manifests/perception_finetune.tsv` (64 training plus 16
  validation files, all crc32c matched on 2026-06-27; total bytes
  80,523,139,102).
  Latest Route 4 real-Waymo StreamPETR smoke artifacts are
  `data/waymo_stream_petr/waymo_infos_train_1seg.pkl`,
  `checkpoints/stream_petr_waymo3_reinit_cls.pth`, and
  `code/StreamPETR/tools/waymo_perception/RQ010B_ROUTE4_WAYMO_STREAM_PETR_SMOKE.md`.
  Latest Route 4 dev6 train/eval dry-run artifacts are
  `data/waymo_stream_petr/waymo_infos_train_4seg.pkl`,
  `data/waymo_stream_petr/waymo_infos_val_2seg.pkl`,
  `projects/configs/StreamPETR/stream_petr_r50_flash_704_waymo_5cam_dev6_dryrun.py`,
  `work_dirs/streampetr_waymo_dev6_dryrun_20260626/iter_40.pth`, and summary
  `logs/streampetr_waymo_dev6_dryrun_summary_20260626.md`.
  Active Route 4 64/16 finetune launch artifacts are Slurm job `1707389`,
  config
  `projects/configs/StreamPETR/stream_petr_r50_flash_704_waymo_5cam_finetune64_leadcfg.py`,
  script `scripts/streampetr_waymo_finetune64_leadcfg_20260627.sbatch`, work
  dir `work_dirs/streampetr_waymo_finetune64_leadcfg_20260627/`, logs
  `logs/streampetr_waymo_finetune64_1707389.out`/`.err`, and existing converted
  infos `data/waymo_stream_petr/waymo_infos_train_64seg.pkl` plus
  `data/waymo_stream_petr/waymo_infos_val_16seg.pkl`.
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
- RQ009 dynamic counterpart-conditioned envelope bilingual report (Nature-style
  conclusion-owned figures; Phase 11 report gate PASS, offline EN/ZH):
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/00_entry/index.html`
- RQ010 WOD-E2E tracking feasibility report (`T2_FULL_TRACKING_REQUIRED`;
  knowledge `decision.md` frozen 2026-06-24; Route 4 preferred,
  Route 5 fallback; basic Tongji HPC parser/pre-flight access verified on
  four validation shards 2026-06-26, full gate pending):
  `reports/studies/RQ010_wod_e2e_tracking_feasibility/RQ010_1_wod_tracking_feasibility_20260623T073830+0800_14f21d3e/00_entry/index.html`
- RQ012A OnSite event annotation readiness Wave-A package (9 automatic events;
  gates 012-0/012-1 pass, 012-2 text-cleared, 012-3 ready-pending-humans,
  012B blocked; knowledge `decision.md` freezes the deferral, not a full PASS):
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37/90_report/index.html`
- RQ012B W0 frozen automatic extractor health bilingual report (no
  outcome/IPV/deviation association; scientific endpoint
  `BLOCKED_PENDING_M3`; clean_285 attempted 285, succeeded 280; precedence
  suppression 2.6569%, identity stability 100% raw intervals; coarser sampling
  unstable):
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_1_event_harm_20260625T202307+0800_38f47437/00_entry/index.html`
- RQ012B W0 independent blind replication of frozen extractor health (no
  outcome/IPV/deviation association; native counts near but not exact versus
  W0, computability 280/285 agrees; principled uniform-grid resampling remains
  unstable at 5 Hz +88.1% and 20 Hz -30.4% total primary events):
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_1_event_harm_20260625T202307+0800_38f47437/02_process/08_replication/replication_report.md`
- RQ012B W0 extractor-health publication figures (Nature-style, extractor
  evidence only; PNG/PDF/SVG plus source data and manifest; no outcome/IPV/
  ranking/harm endpoint plotted):
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_1_event_harm_20260625T202307+0800_38f47437/01_results/figures/figure_manifest.md`
- RQ011A OnSite full-universe readiness (re-run on complete data; `READY_WITH_FROZEN_EXCLUSIONS`:
  outcome universe full 300 / replay 285 with T19 excluded; run-level & repeated-run not identifiable
  by design; knowledge `decision.md` frozen 2026-06-24; supersedes the suspended
  RQ011_1 incomplete-data run):
  `reports/studies/RQ011_onsite_full_universe_readiness/RQ011_2_onsite_readiness_20260623T201415+0800_efdd75a5/90_report/index.html`
- RQ011B OnSite matched-scenario run paused after phases 0-2 at the phase-3 gate (P1 FAIL blockers B001-B005; P2 PASS; resume requires RQ009 M3 downstream clearance plus PI-approved SAP/controls): `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/00_meta/PAUSE_STATE.md`

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
