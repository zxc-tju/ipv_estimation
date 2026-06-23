# RQ008 Run-Level Traceability Ledger

Run ID: `RQ008_1_temporal_ipv_discovery_20260622T234914+0800_3e3e776a`
Reconciled by: `RQ008-W12b-ledger` at `2026-06-23T08:12:15+08:00`

Scope: phases 0-12 only. This reconciliation is pure bookkeeping: no conclusions, figures, statistics, frozen protocol core fields, confirmation data, or paper-repository files were changed.

| Phase | Objective | Key artifacts | Evidence / verdict |
|---|---|---|---|
| 0_bootstrap | Initialize RQ008A discovery-only run, pin plan/input metadata, and create operating skeleton. | `02_process/00_meta/run_manifest.json`; `02_process/00_meta/input_manifest.csv`; `02_process/00_meta/spec_snapshot.md`; `02_process/00_meta/worker_schema.md`; `README.md` | PASS. Run created by `RQ008-W00-bootstrap`; analysis_performed=false and confirmation_subset_opened=false. |
| 1_plan_review | Audit RQ008A plan before data work and identify any blocking enforceability gaps. | `02_process/01_plan_review/plan_review.md`; `02_process/01_plan_review/findings.csv`; `02_process/01_plan_review/status.json` | BLOCKED_FINDING_REMEDIATED. Phase 1 recorded F001 confirmation-isolation blocker; Phase 2 closes it with a physical discovery-only split. |
| 2_split | Create protected discovery/confirmation split and close Gate 008-0 before discovery workers read time-series rows. | `02_process/02_split/split_manifest.json`; `02_process/02_split/leakage_audit.md`; `02_process/02_split/access_control.md`; `02_process/02_split/split_assignment.csv` | GATE 008-0 PASS. Discovery=22,937 cases / 2,218,410 frames; confirmation=15,291 cases / 1,477,571 frames; leakage audit ALL_PASS. Confirmation path/hash recorded from metadata only. |
| 3_atlas | Build discovery-only temporal atlas and normalized case substrate. | `02_process/03_atlas/atlas_report.md`; `02_process/03_atlas/data_health.md`; `01_results/atlases/`; derived `02_atlas/case_normalized_timeseries.csv.gz` | PASS. Atlas substrate SHA-256 `c229798295f974da0a704cf1b6fcfe85b05390b7514a46f9497b4724cfa74765`; all outputs labeled exploratory. |
| 4_alignment | Compare temporal alignment references and fence circular/oracle references. | `02_process/04_alignment/alignment_report.md`; `01_results/tables/alignment_concentration.csv`; `01_results/tables/alignment_circularity_risk.csv`; `01_results/figures/alignment_*.svg` | PASS. `estimability_onset_proxy` and `offline_oracle_phase` flagged as circular/not-confirmation; independent references retained only as exploratory inputs. |
| 5_role | Quantify role/phase dynamics using discovery-only atlas and kinematic risk summaries. | `02_process/05_role/role_dynamics_report.md`; `01_results/tables/role_dynamics_summary.csv`; `01_results/tables/role_dynamics_topline.csv`; `01_results/figures/role_*.svg` | PASS. Role outcomes: early ordering continuous, risk rise/release null, lead/lag null, asymmetry/hysteresis/post-resolution continuous/asymmetric candidates only. |
| 6_motifs | Catalogue exploratory temporal motifs and null/failed motif attempts. | `02_process/06_motifs/motifs_report.md`; `02_process/06_motifs/motif_catalogue.csv`; `02_process/06_motifs/negative_results.csv`; derived `06_motifs/*assignments.csv` | PASS. Six method families attempted; 18 exploratory motif rows; 16 within-source survivors before controls; no positive claim frozen. |
| 7_controls | Apply mechanical, compositional, uncertainty, source-balance, and estimability controls to fixed candidate catalogue. | `02_process/08_controls/controls_report.md`; `02_process/08_controls/control_results.csv`; `02_process/08_controls/survival_summary.csv`; `01_results/figures/controls_survival_summary.svg` | PASS. 0/24 candidates survived all relevant controls; this is the negative headline and the basis for the primary null. |
| 8_hypothesis_freeze | Freeze bounded confirmation protocol and candidate register without opening Wave-B data. | `02_process/09_hypothesis_freeze/freeze_synthesis.md`; `02_process/09_hypothesis_freeze/freeze_register.csv`; `02_process/09_hypothesis_freeze/confirmation_protocol.yaml` | FROZEN_RQ008A after Phase-9 ratification. Frozen rows: `NULL-PRIMARY-NO_SURVIVORS`, `FALS-MOTIF-008-JOINT_UPWARD`, `FALS-UNCERTAINTY-EARLY_DECLINE`; Wave B not authorized. |
| 9_review | Independently review and ratify the freeze package. | `02_process/10_review/discovery_review.md`; `02_process/10_review/review_findings.csv`; `02_process/10_review/ratification_decision.md`; `02_process/09_hypothesis_freeze/confirmation_protocol.yaml` | PASS. 8/8 review checklist items passed; decision RATIFY; protocol hash after ratification `ebcdb92ce811c0af1ec85c9c19d4bf447aaecf0ef8ea2e85b97fd7232532214b`. |
| 10_red_team | Run adversarial discovery review and leakage/false-negative checks. | `02_process/11_red_team/red_team_report.md`; `02_process/11_red_team/red_team_findings.csv`; `02_process/11_red_team/red_team_attack_summary.csv`; `01_results/figures/red_team_*.svg` | PASS_WITH_FIXES. Ten attacks addressed; 0 blocking findings. Attack-10 reversed-time/SNR scope risk carried into `pre_wave_b_amendments.md`. |
| 11_replication | Independently reimplement selected checks and recompute split membership without opening holdout contents. | `02_process/12_replication/replication_report.md`; `02_process/12_replication/replication_compare.csv`; `01_results/tables/replication_compare.csv` | PARTIAL but nonblocking for negative headline. 4/5 items AGREE; change-point distribution DIVERGENCE recorded as exploratory and amended before Wave B. |
| 12_figures_html | Render Nature-skill publication figures and assemble local/offline HTML report. | `01_results/figures/FIGURE_PROVENANCE.md`; `01_results/figures/figure_manifest.csv`; `01_results/figures/publication/fig1_temporal_atlas.*` through `fig4_replication_agreement.*`; `00_entry/index.html`; `90_report/index.html` | PASS. Figures via Nature skill with source CSV hashes; HTML is offline/local and keeps the negative `0/24` headline. |

## Evidence Rows

- `evidence.csv` contains four exploratory rows only: the negative headline plus the three frozen candidates from `freeze_register.csv`.
- No positive temporal IPV claim was added.

## Confirmation Boundary

- `confirmation_PROTECTED` was not opened during this reconciliation.
- The protected confirmation holdout path and SHA-256 are retained only from `split_manifest.json` metadata.
- Wave B remains gated on explicit user authorization and the pre-Wave-B amendments.
