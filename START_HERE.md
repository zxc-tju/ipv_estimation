# START_HERE: Current Operating Brief

Last reviewed: 2026-07-03.

Use this file as the first stop for a new agent thread. Keep durable policy in
`AGENTS.md`, architecture notes in `PROJECT_STRUCTURE.md`, and the compact research
question index in `STUDIES.md`.

## Current Active Context

- **RQ010B COMPLETE (2026-07-03; 10Hz sensitivity closed 2026-07-04) = bounded NULL.** Reframed WOD-E2E human-preference
  validity: candidate IPV does NOT predict human preference and is not comparable to
  physics (Scheme 1 future-only n=75 rho=0.148 p=0.10; Scheme 2 history+future >=1s
  n=98 rho=0.031 p=0.69; max-stat permutation p=1.0 both). M3 does NOT transfer to
  WOD-E2E (<=15% in-support) -> path-type HV norm. Review PASS, red-team null ROBUST,
  replication exact. Report `reports/studies/RQ010_wod_e2e_tracking_feasibility/RQ010B_1_tracking_preference_20260625T201647+0800_695fa83f/90_report_reframed_preference/index.html` (+`.zh.html`);
  decision `reports/knowledge/RQ010_wod_e2e_tracking_feasibility/decision.md`. Full
  pipeline on HPC `/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/` (retained). The PI-flagged
  4Hz->10Hz caveat is now checked under
  `/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/phase_10hz_sensitivity/`: re-estimated
  candidate IPV at dt=0.1 with no counterpart extrapolation and joined ratings only for
  the final test. Null holds at 10Hz (Scheme 1 n=75 rho=0.165 p=0.0626; Scheme 2 10Hz
  effective n=47 rho=0.128 p=0.241; max-stat p=1.0 both; IPV-vs-4Hz Spearman 0.308/0.289).
  Deliverables: `candidate_ipv_10hz.csv` and `tenhz_sensitivity_report.md`. No active
  RQ010B compute; token relay stopped.
- RQ012B Stage 4/5 deviation-to-harm association and negative-control battery is
  complete for the expanded all-valid frozen-M3 OnSite deviation table. Analysis
  set is the pre-registered gate-passing units: `n=245` units across 19 teams;
  exclusions are 18 replay-eligible units that failed IPV/anchor build before
  deviation plus 22 built units with no gate-passing anchor. Final verdict is
  `NULL`: no primary objective harm co-primary deviation effect is reliable
  after BH-FDR or label-permutation control, and none passes the full
  stage-5 battery. Primary co-primary effects: official_safety
  `frac_outside_90` increment `1.1595e-05`, 95% CI
  `[9.728e-08, 0.0013195]`, permutation p `0.7429`, q `0.999`;
  official_safety `max_abs_exceedance_90` increment `0.0001303`, 95% CI
  `[6.687e-07, 0.0013279]`, p `0.6485`, q `0.9947`;
  collision/intervention indicator increments were effectively zero with p
  `0.3845`/`0.6941` and q `0.9422`/`0.999`. W0 event-count associations are
  secondary only; E16 sparse event rows included some low p-values, but
  automatic-event counts alone are not a scientific outcome and several
  controls failed. Stage 4b full interaction-failure consequence battery is now
  complete over the 8 non-inert automatic behavioural manifestations, 4
  behavioural groupings, and 4 official subscores with kinematic-only +
  exposure baseline, seed `20260628`, 5,000 team-block permutations, and 300
  team-cluster bootstraps. Full-battery verdict is `BOUNDED` with `0`
  SUPPORTED endpoints: strongest powered channel is NEAR-MISS/CONTACT
  `max_abs_exceedance_90` IRR `1.2239`, 95% CI `[1.0314, 1.3450]`,
  permutation p `0.0018`, BH q `0.05119`, baseline-incremental and beating
  placebo/label but failing M2; E09 near-miss similarly has IRR `1.2329`, p
  `0.0018`, q `0.05119` but fails placebo and M2. E16 no-progress/deadlock is
  bounded and control-passing (IRR `1.4967`, p `0.002599`, q `0.05119`) but is
  explicitly UNDERPOWERED. No official subscore or abrupt/discomfort channel
  passes BH-FDR/control requirements, and no interaction-failure channel is
  IPV-specifically supported by deviation. Full-battery artifacts:
  `data/derived/onsite_competition/RQ012B_event_harm/stage4b/full_battery/{full_battery_results.csv,endpoint_summary.csv,negative_control_results_full_battery.csv}`
  and
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/04_harm_association/{harm_association_full_battery_report.md,results_full_battery.json}`.
  Earlier Stage 4/5 artifacts:
  `data/derived/onsite_competition/RQ012B_event_harm/stage4plus/` and
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/04_harm_association/{harm_association_report.md,results.json}`;
  stage-5 detail:
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/05_negative_controls/negative_control_detail.csv`.
  Publication figure package for HA-1/HA-2/HA-3 plus the intuitive Stage-4b
  partial-rank Fig. 4:
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/01_results/figures/`
  with PNG/PDF/SVG figure groups, per-panel source CSVs, manifest, and plotting
  scripts. Latest added figure:
  `fig4_deviation_vs_failures_intuitive.{png,pdf,svg}`, computed from
  `unit_analysis_table.parquet` as exposure-controlled partial Spearman
  correlations with event rates and `100 - official_score`; all point estimates
  are positive but weak, and too-passive lower-tail deviation is larger than
  too-aggressive upper-tail deviation in 9/10 displayed consequences.
  Bilingual offline-openable report package:
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/90_report/index.html`,
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/90_report/index.zh.html`,
  and entry page
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/00_entry/index.html`.
  Independent blind replication by a different route also reproduces NULL:
  team-block outcome-profile permutation plus exposure-controlled rank/logistic
  tests gave official_safety p `0.0762`/`0.2529` for
  `frac_outside_90`/`max_abs_exceedance_90`, collision/intervention p
  `0.3421`/`0.8956`, all `AGREE`; artifacts:
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/08_replication/`.
  Independent full-battery replication/red-team recheck also `AGREE`s with the
  bounded/null story while adding two wording caveats: the displayed/powered
  consequences have uniform positive worse-direction partial-r signs, but the
  full 16-endpoint family has sparse underpowered E18/E19 exceptions; and the
  strict `partial r <= 0.17` shorthand holds for the displayed simple-rank view
  but not for the NEAR-MISS/CONTACT grouping (`r=0.205`, still small). E16
  lower-tail deadlock remains the only all-control-passing row in the
  independent Poisson check (M3 increment `0.03817`, M2 `0.03606`, placebo
  `0.01534`, within-team permutation p `0.0010`, 52 units with E16>0), but is
  underpowered and published BH q is `0.05119`; near-miss and
  NEAR-MISS/CONTACT max-exceedance lose to M2 (`0.04335` vs `0.04536`, and
  `0.04879` vs `0.05145`). Recheck artifacts:
  `data/derived/onsite_competition/RQ012B_event_harm/stage4b/recheck/` and
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/08_replication/full_battery_recheck_report.md`.
  Reproduce the original Stage 4/5 run with
  `/Users/xiaocong/.rq009_codex_fleet/venv/bin/python reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/04_harm_association/run_harm_association.py --seed 20260628 --n-permutations 5000 --bootstrap 300`.
  Reproduce the full-battery Stage 4b run with
  `/Users/xiaocong/.rq009_codex_fleet/venv/bin/python reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/04_harm_association/run_harm_association_full_battery.py --seed 20260628 --n-permutations 5000 --bootstrap 300`.
  Reproduce the full-battery independent recheck with
  `/Users/xiaocong/.rq009_codex_fleet/venv/bin/python data/derived/onsite_competition/RQ012B_event_harm/stage4b/recheck/recheck_full_battery.py --seed 20260628 --n-permutations 5000`.
- RQ012B Stage 3+ OnSite all-valid M3-anchor enabling build now has
  AV-perspective clean_285 anchors under
  `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet`:
  267/285 units covered, 67,861 anchors, 29/32 M3 required inputs fully
  populated, dense IPV rows at
  `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_ipv_timeseries_multi_allvalid.parquet`.
  This HPC build used pinned legacy estimator HEAD `5edd2810` with
  process-pool Slurm job `1710800` on one AMD 192-core node. The expanded
  frozen-M3 OOD/support gate and deviation scan is complete: gate pass
  19,044/67,861 anchors and 245/267 units; abstain 48,817 anchors. At the 90%
  band, 840 gate-passing anchors across 149 units are out-of-band; 80%/95%
  counts are 2,475/447 anchors and 193/116 units. Per-unit max absolute 90%
  exceedance is >0 for 149 units (nonzero min/median/max
  0.00158/0.24593/1.06895). Abstention remains structural: distance over
  threshold for 47,166 category-eligible anchors and unsupported joint cells
  for 1,651 anchors; imputed-NaN distance features are common (64,040 anchors,
  45,301 abstainers) but are not a separate frozen-gate hard-fail. Stage 4/5
  harm association has now been run from this expanded deviation table; see the
  current RQ012B Stage 4/5 bullet above. Stage-3 gate report:
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/ood_gate_multi/ood_gate_multi_report.md`.
  Scored data:
  `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/ood_gate_multi/`.
- Prior RQ012B Stage 3+ frozen-M3 OOD/support gate is complete only for the
  earlier one-anchor-per-unit 267-anchor OnSite AV build. As-is gate pass is
  51/267 anchors/units, with 216 abstentions;
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
- HPC reuse: shared Tongji HPC usage guide for all local projects →
  `../HPC_TONGJI_USAGE_GUIDE.md`; InterHub/IPV-specific reusable assets remain
  in `reports/knowledge/INFRA_hpc_tongji_reuse.md`. On HPC, durable work lives
  under `/share/home/u25310231/ZXC`, and newly submitted Slurm job names must
  start with `zxc-`.
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
- RQ010B Waymo Perception v1.4.3 finetune subset is now 256 training plus
  16 validation `.tfrecord` segments on Tongji HPC under
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/data/perception/`, crc32c verified
  with manifest
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/manifests/perception_256.tsv`
  (272/272 ok). This supersedes the earlier 64-train/16-val
  `perception_finetune.tsv` subset for current StreamPETR finetuning.
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
- RQ010B StreamPETR Route 4 lead-config 64-train/16-val finetune was stopped
  early after best 16-val smoke recall plateaued around epoch 4. Frozen best
  checkpoint:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/work_dirs/streampetr_waymo_finetune64_leadcfg_20260627/best_waymo_center_recall_2m_iter_50732.pth`.
  The original Slurm job was `1707389` (L40 node `gpu4037`), launched
  2026-06-27 02:48 CST after job `1707307` failed at MMCV config parse with
  `TypeError: cannot pickle '_io.BufferedReader' object`. The config fix
  deletes the closed `_handle` left by the parse-time train-info `with open`
  block, and login-node `Config.fromfile(...)` returns `CONFIG_PARSE_OK`.
  Sbatch:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/scripts/streampetr_waymo_finetune64_leadcfg_20260627.sbatch`;
  config:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/code/StreamPETR/projects/configs/StreamPETR/stream_petr_r50_flash_704_waymo_5cam_finetune64_leadcfg.py`;
  work dir:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/work_dirs/streampetr_waymo_finetune64_leadcfg_20260627/`;
  Slurm logs:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/logs/streampetr_waymo_finetune64_1707389.out`
  and `.err`. The job first converts
  `waymo_infos_train_64seg.pkl` and `waymo_infos_val_16seg.pkl`.
- RQ010B §5 detector-quality/error-model gate is complete for the frozen
  Route 4 best checkpoint on 16 Perception validation segments. Evaluation job
  `1710088` ran on one L40 for 00:28:41 (0.478 GPU-h allocated; script runtime
  estimate 0.460 GPU-h). Method: StreamPETR single-GPU inference, classwise
  rotated BEV NMS (`score>=0.05`, IoU `0.25`, max `100` detections/frame),
  center-distance AP/matching at 2 m because official Waymo LET-3D-AP metrics
  ops were unavailable in the current env. Validation-selected operating point
  is score threshold `0.15` (max micro-F1 on the same 16-val segments). Result:
  overall AP `0.00328`, recall `0.08034`, precision `0.03276`; Vehicle AP
  `0.00432`, recall `0.10585`, precision `0.03276`; Pedestrian and Cyclist AP,
  recall, and precision all `0.0`. Verdict: this 64-segment detector is not
  adequate for tracker + HOTA/AMOTA QA; add the remaining 734 Perception
  training segments (full 798 total) with class-balanced checks and retrain
  before retesting Route 4. Route 5 remains fallback if full-data Route 4 still
  leaves Pedestrian/Cyclist near zero. Outputs:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/results/rq010b_route4_detector_quality_best50732_20260627_summary.json`,
  `_metrics_by_class_range.csv`, `_error_model.json`, and `_error_model.csv`.
- RQ010B improved StreamPETR Waymo Perception 256-train/16-val finetune is
  complete. Train-only 4-L40 DDP Slurm job `1712698` completed on `gpu4011`
  in `20:27:18` (about 81.8 allocated L40 GPU-h) and saved all 12 raw-equivalent
  epoch checkpoints through `iter_152124.pth` (`latest.pth -> iter_152124.pth`).
  Previous jobs failed only in in-loop distributed
  evaluation: `1712416` on `gpu4009` failed after epoch-1 eval with NCCL
  watchdog timeout, and `1712590` on `gpu4025` saved `iter_25354.pth` then
  failed at DistEvalHook with `TypeError: 'NoneType' object is not iterable` in
  `projects/mmdet3d_plugin/datasets/waymo_dataset.py evaluate()`. Training was
  healthy with loss around 21 and checkpoints `iter_12677.pth` and
  `iter_25354.pth`. Current no-eval fix applied on HPC: `tools/train.py` keeps
  `timeout=datetime.timedelta(hours=4)` for `init_dist`; the active config sets
  `evaluation=None`, `custom_hooks=[]`, `raw_equivalent_epochs=12`,
  `max_iters=152124`, `checkpoint_config.interval=12677`, and
  `max_keep_ckpts=-1`; `projects/mmdet3d_plugin/core/apis/mmdet_train.py`
  skips eval-hook registration when `evaluation is None`; the resume sbatch
  also passes `--no-validate` and resumes explicitly from `iter_25354.pth`.
  Saved checkpoints must still be evaluated separately on 1 GPU because the
  DDP eval path has the known `NoneType` bug. Config:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/code/StreamPETR/projects/configs/StreamPETR/stream_petr_r50_flash_704_waymo_5cam_finetune256_balanced_warminit.py`;
  warm-init checkpoint:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/checkpoints/stream_petr_waymo3_warminit_nusc_car_ped_bicycle.pth`.
  Recipe uses ClassBalancedDataset `oversample_thr=0.70`, nuScenes class-row
  warm init (`car->Vehicle`, `pedestrian->Pedestrian`, `bicycle->Cyclist`),
  5x LR for `pts_bbox_head.cls_branches`, grid mask plus resize/flip and
  BEV rot/scale augmentation. The config hard-points to
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/data/waymo_stream_petr/waymo_infos_train_256seg.pkl`
  and
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/data/waymo_stream_petr/waymo_infos_val_16seg.pkl`
  and fails if they are absent. Converted-info assertion:
  train 50,708 samples/256 scenes/360,505,501 bytes; val 3,151 samples/16
  scenes/22,943,129 bytes; JSON
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/logs/waymo_infos_256_16_assert_1712385.json`.
  `Config.fromfile` passes with 4-GPU schedule
  `raw_iters_per_epoch=12677`, `checkpoint_interval=12677`,
  `eval_interval=164801` (disabled sentinel greater than `max_iters`),
  `max_iters=152124`, `evaluation is None`, and zero custom hooks. Quick
  4-L40 DDP smoke job `1712408` on `gpu4011` passed warm-init +
  class-balanced sampler
  + DDP backprop with averaged loss decreasing
  `78.1665 -> 66.1600 -> 59.0420`;
  logs:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/logs/streampetr_waymo_finetune256_ddp_smoke_1712408.log`
  and `.jsonl`. Original full-run sbatch:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/scripts/streampetr_waymo_finetune256_balanced_warminit_ddp4_20260628.sbatch`;
  failed timeout-fix resume sbatch:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/scripts/streampetr_waymo_finetune256_balanced_warminit_ddp4_resume_timeoutfix_20260628.sbatch`;
  completed no-eval resume sbatch:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/scripts/streampetr_waymo_finetune256_balanced_warminit_ddp4_resume_noeval_20260628.sbatch`;
  original resume checkpoint:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/work_dirs/streampetr_waymo_finetune256_balanced_warminit_ddp4_20260628/iter_25354.pth`
  (`latest.pth` now points to final `iter_152124.pth`);
  work dir:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/work_dirs/streampetr_waymo_finetune256_balanced_warminit_ddp4_20260628/`;
  Slurm logs:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/logs/streampetr_waymo_finetune256_ddp4_1712416.out`
  and `.err` for the first failed run,
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/logs/streampetr_waymo_finetune256_ddp4_resume_1712590.out`
  and `.err` for the failed eval-resume job, and
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/logs/streampetr_waymo_finetune256_ddp4_resume_noeval_1712698.out`
  and `.err` for the completed train-only resume job. Separate single-GPU
  detector-quality Slurm job `1745613` (`zxc-rq010b-eval256`) evaluated
  checkpoints `iter_76062`, `iter_101416`, `iter_126770`, and `iter_152124`
  on the 16 Perception validation segments in `01:42:26` (about 1.71 allocated
  L40 GPU-h; script runtime sum about 1.64 GPU-h). Best by mean AP over the
  9 class x range cells is ep12 `iter_152124`: `mAP_9=0.08454`, pooled
  center-distance AP `0.10835`, overall recall `0.21916`, precision `0.23675`
  at score threshold `0.225`. Class all-range matched-TP recall/precision:
  Vehicle `0.24363`/`0.23469`, Pedestrian `0.14515`/`0.24725`, Cyclist
  `0.05644`/`0.40000`. Pedestrian and Cyclist are nonzero, so warm init plus
  class balance worked enough to clear the zero-class failure; `grad_norm:nan`
  during training was benign for detector output. The detector is now adequate
  to proceed to tracker + HOTA/AMOTA QA on the 16-val pilot, while still weak
  at 50+ m and not yet a final detector-quality solution. Best outputs:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/results/rq010b_route4_detector_quality_256_balwarm_ep12_iter_152124_20260629_summary.json`,
  `_metrics_by_class_range.csv`, `_error_model.json`, `_error_model.csv`,
  `_matched_tp_errors.csv`, `_threshold_sweep.csv`, and `_post_nms_records.pkl`;
  checkpoint ranking summary:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/results/rq010b_route4_detector_quality_256_balwarm_20260629_checkpoint_summary.{json,csv}`.
- RQ010B WOD-E2E IPV-rating pilot degeneracy investigation/fix is complete on
  Tongji HPC. The original
  `results/rq010b_wod_e2e_ipv_rating_pilot_20260629/` IPV-rating result is
  invalid for IPV conclusions: WOD-E2E state sequences were sampled at
  `dt=0.25` s while the legacy IPV estimator still integrated with global
  `dt=0.1` s; probability-space trajectory likelihoods underflowed to all-zero
  candidate weights and forced the uniform `ego_ipv=0.0` fallback; and the
  adapter used each evaluated candidate as its own reference line instead of
  the RQ010B §6 scene-level route reference. The patched HPC adapter is
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/code/rq010b_ipv_rating_pilot_20260629/analyze_wod_e2e_ipv_rating_pilot.py`
  with backup `.bak_20260629_dtfix`; it sets estimator `dt=0.25`, uses
  log-domain trajectory-likelihood normalization, and builds the §6
  past-pose-plus-routing constant-curvature ego reference. Final fixed outputs:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/results/rq010b_wod_e2e_ipv_rating_pilot_routefix_20260629T124941/`.
  Distribution is now finite and varied but still partially uninformative:
  `n=33`, range `[-1.1781, 1.1781]`, 17/33 `abs(ego_ipv)>1e-6`,
  16 rounded-distinct IPV values, and 4 uniform-fallback rows. Re-run
  IPV-rating association remains weak/null: pooled Spearman `rho=0.123`,
  95% bootstrap CI `[-0.224, 0.452]`, `p=0.495`; mean within-scene Spearman
  `-0.0787` over 11 usable scenes; IPV single-feature R2 `0.0110`, below the
  best physics feature `driven_ade_m` R2 `0.0634`. Reproducer artifacts:
  `debug_reproducer_dt_route_log_20260629.{md,json}` in the final result dir.
  Applicability caveat remains: this is an exploratory one-frame StreamPETR
  velocity-extrapolated counterpart pilot, not the full RQ010B validated
  tracker/M3 preference test.
- RQ010B WOD-E2E multi-frame ceiling investigation is complete on Tongji HPC
  and overturns the four-shard hard-ceiling interpretation. A single
  `E2EDFrame` contains one timestamp's 8 camera JPEGs, one per surround camera,
  not an internal recent-frame sequence; ego `past_states`/`future_states` are
  16/20 samples at 4 Hz. The sparse four-shard finding is an interleaved
  shard-access artifact: adding four CRC-clean validation probe shards (`00004`,
  `00005`, `00007`, `00010`) increased unique pre-t* frame coverage for all 12
  rated pilot segments (min/median/max `2/6/9` -> `6/12/16`). The 8 clean
  shards still did not reconstruct a 10-frame contiguous run ending at t*
  (end-contiguous max 2; best pre-t* contiguous max 3), so the concrete next
  path is full/targeted validation shard indexing over
  `val_202504211843.tfrecord-00000..00092-of-00093`. Four attempted extra probe
  shards (`00006`, `00008`, `00009`, `00011`) were excluded after CRC failures
  from an interrupted transfer. Artifacts:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/results/wod_e2e_temporal_ceiling_probe_20260629/{rated_record_structure_probe.json,shard_growth_probe_clean_extra.json,shard_growth_probe_contiguous_clean_extra.json}`
  and manifest
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/manifests/validation_probe_00004_00011.tsv`.
- RQ010B full WOD-E2E validation rated479 streaming ingest/extract is complete
  on Tongji HPC. Slurm job `1746449` (`zxc-rq010b-full93`, `amd`, `cpua277`)
  completed with exit `0:0` in `11:47:32`; it was not cancelled. The apparent
  post-loop hang was useful finalizer work: after `all-shard loop complete
  receipts_ok=93/93` at 2026-06-30 04:37 CST, the job wrote sorted
  per-segment `frames.tfrecord` and `frames.index.tsv`, logged
  `segments_finalized=479`, and exited at 05:11. The buggy final summarize
  left `manifests/rated479_segment_counts.tsv` at 0 bytes; it was regenerated
  atomically from the independent readiness table and is now nonzero.
  Extracted data is under
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/data/rated479_segments/<segment_key>/`
  with exactly 479 rated segment directories and zero leftover raw validation
  shards. The stray 480th directory was empty `_tmp` and was removed; the stale
  GCS token at
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/secrets/gcs_token` was deleted
  after finalization.
  Readiness artifacts:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/manifests/rated479_segment_readiness.tsv`
  and `.json`. Gate result: 479/479 segments have at least 10 strict
  contiguous pre-t* frames; min/median/max pre-t* contiguous history is
  91/228/229 frames, histogram `50-99:3`, `100-199:100`, `>=200:376`, and
  there are no short/abstain-worthy segments. All 479 segments have the five
  forward-arc cameras/calibrations (`FRONT`, `FRONT_LEFT`, `FRONT_RIGHT`,
  `SIDE_LEFT`, `SIDE_RIGHT`) on every indexed frame. Native cadence is 10 Hz
  from adjacent `context_step` deltas (`mode=1` for all segments). Current
  directory size is about 354.8 GB apparent / 330.5 GiB because final
  tracker-facing TFRecords and shard TFRecords are both retained; the
  tracker-facing final `frames.tfrecord` set is 177.4 GB / 165.2 GiB, median
  374,013,480 bytes (356.7 MiB) per segment. Downstream tracker should read
  each segment through `frames.index.tsv` sorted by `context_step`, then stream
  the matching records from `frames.tfrecord`; each E2EDFrame contains the
  pruned five-camera images/calibrations plus ego past/future states and
  preference trajectories. Shard archives remain under `shards/` for audit or
  rebuild only.
- RQ010B WOD-E2E 12-segment dense multiframe tracking -> IPV counterpart
  selection repair is complete on Tongji HPC, using cached detections/tracks
  from
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/results/rq010b_wod_e2e_multiframe_tracking_ipv_pilot_20260630T053507/`
  and the fixed IPV adapter under
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/code/rq010b_ipv_rating_pilot_20260629/`.
  The active selector code is
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/code/rq010b_multiframe_tracking_ipv_20260630/analyze_multiframe_tracking_ipv.py`
  with backups `.bak_counterpart_gates_20260630T0603` and
  `.bak_vehicle_class_gate_20260630T0610`; focused regression tests are in
  `test_counterpart_selection_gates.py` and pass (`4 passed`). Final cached
  L40 Slurm rerun `1751326` (`zxc-rq010b-cp-gates`) completed with exit `0:0`
  in `00:01:58`, writing
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/results/rq010b_wod_e2e_multiframe_tracking_ipv_pilot_counterpart_gates_20260630T060925/`.
  Gates now require vehicle class, >=10 hits, >=1.0 s span, <=0.5 s stale,
  >=2.0 m displacement, >=0.5 m/s path speed, jitter ratio <=4, step p95
  <=1.75 m, observation gap <=0.5 s, history coverage >=0.4, current distance
  <=35 m, and predicted ego-path conflict gap <=8 m / TTC <=6 s or compatible
  crossing/leading/opposing geometry. Retained interaction rate on the 12 pilot
  segments is 8/12 vehicle counterparts (24 candidate IPV rows) and 4
  abstentions: two no quality-passing tracks, one no interacting vehicle after
  conflict gate, and one pedestrian-only/conflict-gate case after the vehicle
  class gate. Selected vehicle tracks are all moving/interacting
  (displacement 2.51-6.36 m, 14-48 observed points, min predicted gap
  0.221-7.71 m, geometries crossing/opposing/leading-or-merging). Ego IPV on
  retained candidates is finite with range `[-1.17810, 1.14898]`, median `0`,
  13/24 nonzero above 1e-6, and pooled rating association remains small/null
  (Spearman rho `0.1269`, 95% bootstrap CI `[-0.3147, 0.5425]`, p `0.5547`).
  This pilot gate has now been scaled to all 479; use the audited full-run
  result below for current RQ010B operating facts.
- RQ010B WOD-E2E full479 scored-target multiframe tracking -> gated
  counterpart -> fixed ego-IPV audit is complete under Tongji HPC `/ZXC`
  boundaries, status `audited_not_frozen` pending review. Canonical result dir:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/results/rq010b_wod_e2e_multiframe_tracking_ipv_full479_scored_audited_20260630T063600/`.
  Detector array `1751377_[0-7]` used one L40 per shard on
  `gpu4006/gpu4008/gpu4010` and completed cleanly; CPU job `1751378` wrote the
  full analysis artifacts then failed only in posthoc audit markdown formatting,
  which was patched and rerun successfully on the completed CSV/JSON outputs.
  Frozen gates retained 302/479 scenes and abstained 177/479. Abstention
  reasons: no interacting track after conflict gate 92, no track after motion
  gate 36, no track after quality filter 34, no track after history-coverage
  gate 9, no track after smoothness gate 4, and 2 data-level target abstentions
  with no scored preference frame plus 50-frame history. Selected counterparts
  are all `Vehicle`, all pass real-moving gates, and all pass interaction gates;
  selected-track displacement median 3.34 m, mean speed median 1.75 m/s, and
  predicted min-gap median 3.37 m. Ego IPV distribution over 906 retained
  candidate rows has mean 0.00799, median 0, q25/q75 -0.0938/0.1041, and range
  [-1.1781, 1.1781]. Primary IPV-rating association is weak/null: all-retained
  pooled Spearman rho -0.0384, 95% CI [-0.1016, 0.0256], p=0.2477; fresh
  confirmatory subset excluding the 12 pilot segments has 294 scenes / 882 rows,
  pooled Spearman rho -0.0445, 95% CI [-0.1078, 0.0183], p=0.1872. Within-scene
  rank correlations are also weak (fresh mean Spearman -0.0678, p=0.1198;
  mean Kendall -0.0554, p=0.1590). Shape check is only suggestive: quadratic
  term is negative but not conventionally significant (fresh quadratic p=0.0623,
  delta R2=0.00394). Comparability verdict: IPV is not comparable to the best
  single physical feature in this audited run; `driven_fde_m` is best with fresh
  R2=0.02085 and Spearman -0.2461, while ego IPV has fresh R2=0.00265 and
  Spearman -0.0445. Open-loop/closed-loop bias summary: driven trajectory is
  closest to the top-rated candidate in 181/302 retained scenes (59.9%) and
  driven IPV lies inside the candidate IPV range in 214/302 scenes (70.9%).
  Main audit table:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/results/rq010b_wod_e2e_multiframe_tracking_ipv_full479_scored_audited_20260630T063600/rq010b_wod_e2e_multiframe_tracking_ipv_full479_audited_selected_counterpart_summary.csv`;
  audit summary JSON/MD:
  `rq010b_wod_e2e_multiframe_tracking_ipv_full479_audited_audit_summary.{json,md}`
  in the same result directory.
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
- The 2026-06-27 `INV-ipv-code-diff` finding describes the pre-fix state after
  `a0fee535`: the first vectorized cost helpers drifted from the pinned
  sigma01-generation estimator. Commit `67f4c543` later restored the legacy
  loop backend as default `solver_mode="exact"` and repaired the vectorized
  `fast` backend. Current local estimator/profile tests pass (`6 passed`, one
  Linux-only strict check skipped), and verifier tests pass `8/8`. Final HPC job
  `1912947` reproduces sigma01 with `exact` at `max_abs_diff=4.44e-16`; the
  non-canonical `fast` backend differs from `exact` by `0.0016531` on that ABI.
  Cross-platform SLSQP still moves local exact output by about `0.0587`, so
  formal production uses the cloned sigma01 binary ABI and `solver_mode=exact`.
  Reproduction preserves `sigma=0.1`, `history_window`, `min_observation=4`,
  reference clip/max/smooth `60/40/40`, NuPlan 20-to-10 Hz downsampling, and the
  tracked `configs/ipv_sigma01_exact.json`; InterHub CLI reference defaults are
  now aligned to `60/40/40`.
- Git-based HPC deployment is active at
  `/share/home/u25310231/ZXC/sociality_estimation/code/repo` as a clean detached
  checkout of `codex/unify-ipv-pipeline`; runtime and lock content was validated
  at `a5af68d2`. Exact and verifier environments are isolated under
  `envs/ipv-exact-sigma01` (Python 3.9.24) and
  `envs/ipv-verifier` (Python 3.9.6); their conda/pip locks are tracked under
  `environments/`. Portable private scorer SHA-256
  `b04999aba29a82fb71a97ac22c728479a7734e24a0b32189d08f95184d74f253`
  is checksum-bound at `checkpoints/rq009_m3/`; final verifier job `1912948`
  passed `8/8`. The historical `/share/home/u25310231/ZXC/ipv_estimation`
  checkout remains untouched at `5edd2810`. Deployment guide:
  `docs/reproducible_ipv_pipeline.md`. Historical investigation notes:
  `reports/knowledge/_analysis/ipv_estimator_divergence_investigation.md` and
  `reports/knowledge/_analysis/ipv_accel_hyperparam_finding.md`.
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
  outcomes. The original serialized scorer/helper/contract are provenance-only
  and were moved out of the active derived topology to
  `data/derived/_provenance_archive/rq009_m3_legacy_source_20260711/`.
  Runtime code must use the manifest-verified portable bundle under
  `models/rq009_m3/` locally or `checkpoints/rq009_m3/` on HPC.
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
  2026-06-26), `manifests/perception_finetune.tsv` (64 training plus 16
  validation files, all crc32c matched on 2026-06-27; total bytes
  80,523,139,102), and current `manifests/perception_256.tsv` (256 training
  plus 16 validation files, 272/272 crc32c ok).
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
  Latest Route 4 64/16 finetune artifacts are config
  `projects/configs/StreamPETR/stream_petr_r50_flash_704_waymo_5cam_finetune64_leadcfg.py`,
  script `scripts/streampetr_waymo_finetune64_leadcfg_20260627.sbatch`, work
  dir `work_dirs/streampetr_waymo_finetune64_leadcfg_20260627/`, logs
  `logs/streampetr_waymo_finetune64_1707389.out`/`.err`, and existing converted
  infos `data/waymo_stream_petr/waymo_infos_train_64seg.pkl` plus
  `data/waymo_stream_petr/waymo_infos_val_16seg.pkl`. Latest §5 detector
  quality/error-model outputs are under `results/` with prefix
  `rq010b_route4_detector_quality_best50732_20260627`, especially
  `_summary.json`, `_metrics_by_class_range.csv`, `_error_model.json`,
  `_error_model.csv`, `_matched_tp_errors.csv`, and `_threshold_sweep.csv`.
  Latest improved recipe config/smoke/train/eval artifacts are
  `projects/configs/StreamPETR/stream_petr_r50_flash_704_waymo_5cam_finetune256_balanced_warminit.py`,
  warm-init checkpoint
  `checkpoints/stream_petr_waymo3_warminit_nusc_car_ped_bicycle.pth`,
  converted infos `data/waymo_stream_petr/waymo_infos_train_256seg.pkl` plus
  `data/waymo_stream_petr/waymo_infos_val_16seg.pkl`,
  support files
  `projects/mmdet3d_plugin/datasets/waymo_ap_dataset.py`,
  `projects/mmdet3d_plugin/core/evaluation/waymo_early_stopping.py`,
  `tools/waymo_perception/make_waymo_warminit_checkpoint.py`, and
  `tools/waymo_perception/smoke_train_waymo_ddp.py`, DDP smoke logs
  `logs/streampetr_waymo_finetune256_ddp_smoke_1712408.log`/`.jsonl`,
  failed Slurm jobs `1712416` and `1712590`, completed train-only resume Slurm
  job `1712698`, single-GPU detector-quality eval job `1745613`, no-eval resume sbatch
  `scripts/streampetr_waymo_finetune256_balanced_warminit_ddp4_resume_noeval_20260628.sbatch`,
  work dir
  `work_dirs/streampetr_waymo_finetune256_balanced_warminit_ddp4_20260628/`,
  final checkpoint `latest.pth -> iter_152124.pth`, checkpoint ranking outputs
  `results/rq010b_route4_detector_quality_256_balwarm_20260629_checkpoint_summary.{json,csv}`,
  best detector-quality/error-model outputs under prefix
  `results/rq010b_route4_detector_quality_256_balwarm_ep12_iter_152124_20260629`,
  single-GPU eval sbatch
  `scripts/run_rq010b_detector_quality_256_eval.sbatch`, and Slurm logs
  `logs/streampetr_waymo_finetune256_ddp4_resume_noeval_1712698.out`/`.err`
  (failed run logs remain
  `logs/streampetr_waymo_finetune256_ddp4_1712416.out`/`.err` and
  `logs/streampetr_waymo_finetune256_ddp4_resume_1712590.out`/`.err`).
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
- RQ010B Route 4 64-segment StreamPETR is not tracker-ready: §5 detector
  quality on 16 Perception validation segments has overall 2 m center-distance
  AP `0.00328`, recall `0.08034`, precision `0.03276`, and zero
  Pedestrian/Cyclist detections at the selected operating point. The improved
  256-seg balanced/warm-init checkpoint ep12 `iter_152124` is the current best
  Route 4 pilot detector and is tracker-QA-ready for the 16-val pilot:
  `mAP_9=0.08454`, pooled AP `0.10835`, recall `0.21916`, precision `0.23675`,
  with nonzero Pedestrian and Cyclist detections. Remaining weak spots are
  far-range quality, small Cyclist sample size, and the fact that this is still
  a 16-val pilot rather than a final full-data detector validation.
- RQ011 supports full_300 outcomes and clean_285 replay/IPV with T19 replay-only exclusion;
  run-level/repeated-run/causal claims are unavailable.
- RQ012 is readiness-only and human annotation is deferred.
- Paper `main` is v4.1 but still carries evidence/external-pending markers and is not submission-ready.
