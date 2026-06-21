# Phase 7 Red Team v2 Report on Corrected Results

Worker: `RQ003_phase7_red_team2_001`
Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`
Status: `BLOCKERS_FOUND`

## Scope And Gate

Pre-write identity passed: `RUN_ROOT` existed, `run_manifest.json` matched the requested run ID, `plan_sha256.txt` matched `98900786a4819e58e91f16eb8da84724c928ca5a6d2c40b2252cf106b56625e1`, `fix_status.json` reported both `rt_block_001_resolved=true` and `rt_block_002_resolved=true`, and corrected result tables existed.

This pass reviewed the corrected crosswalk, A1/safety correction, before/after quarantined tables, Phase 4 freeze contracts, corrected confirmatory/negative-control/state reports, Phase 7 `stats_rereview2` artifacts, raw SQL spot checks, score CSV mappings, and Phase 8 independent replication artifacts.

## Verdict

RT-BLOCK-001 is resolved as a label-correction blocker: the corrected labels match the official score CSV scenario codes and raw SQL case-name sequences. I found no residual label mismatch that would reverse the crosswalk. There is a non-blocking provenance defect: only 30/150 stored `scenario_map_sql_line` pointers hit the exact mapped session row; the other pointers hit another session with the same area/case-name sequence.

RT-BLOCK-002 is resolved: official A1 has 10 rows; the only safety<100 rows are `T15` and `T20` Beijing case 2333, both official A1, zero-score/catastrophic rows. Collision-free membership as `safety >= 100` is consistent with the corrected frame.

The corrected favorable direction is not robust enough to accept as a corrected conclusion. It is nonsignificant and underpowered (`delta Spearman=+0.136833`, scenario-cluster bootstrap CI `[-0.038781, +0.305797]`, permutation `p=0.30`), secondary LOSO is effectively null (`delta Spearman=+0.016732`, MAE and CV-R2 worse), safe-subset agreement is vacuous because S1 and S2 are identical to the primary sample, and negative controls often match or exceed the primary effect. Existing independent replication is also `DISCREPANT` and does not reproduce the corrected N=53 analysis.

## Blocking Findings

### RT2-BLOCK-001: Negative controls do not support a robust IPV-specific favorable signal

The primary delta Spearman is `+0.136833`. Multiple controls equal or exceed it:

| control | delta Spearman | CI | p | note |
|---|---:|---|---:|---|
| future_leaky_full_window_ipv | +0.231817 | [0.007696, 0.482317] | 0.09 | non-deployable, incomplete cache |
| ipv_time_shuffle | +0.196823 | [0.027138, 0.381227] | 0.29 | time/cell shuffled IPV exceeds primary |
| counterpart_swap | +0.168441 | [-0.001504, 0.347969] | 0.10 | swapped conditioning exceeds primary |
| role_flip | +0.136833 | [-0.039084, 0.329710] | 0.27 | identical to primary |
| sign_flip | +0.136833 | [-0.047064, 0.314614] | 0.33 | identical to primary |

The two degradation controls also failed their expectation. `state_shuffle` improved the baseline from reference Spearman `-0.190937` to `+0.105547`, improved MAE from `7.416451` to `6.851252`, and improved CV-R2 from `-0.141026` to `+0.013881`. `wrong_state` also failed. This means the control battery does not distinguish real directional IPV information from noise, baseline instability, or model artifacts.

`stats_rereview2` treated the degradation anomaly as a non-blocking caveat because the baseline is weak/anti-predictive. This red-team pass classifies the same evidence as blocking for robustness, not as a detected pipeline defect: a control battery that improves when corrupted cannot validate an IPV-specific favorable conclusion.

Impact: The corrected result can be described only as a weak, nonsignificant favorable direction. It must not be called robust, validated, or confirmatory evidence of incremental IPV utility.

### RT2-BLOCK-002: The corrected model path lacks independent reproduction

Phase 7 explicitly reused `02_process/10_confirmatory_analysis/run_confirmatory_analysis.py` and replaced scenario/family/A1 labels mechanically. The only independent replication artifact in the run, `02_process/17_independent_replication/independent_replication_report.md`, has verdict `DISCREPANT` and reports material CV prediction discrepancies for the pre-fix N=48 structural-label analysis. It does not reproduce the corrected N=53 rerun.

Impact: The favorable flip is plausible as a label-fix consequence, but the corrected CV model cannot be accepted as final until an independent implementation reproduces the corrected N=53 predictions or explains the Phase 4 prediction discrepancy.

### RT2-BLOCK-003: Safe-subset agreement is not meaningful robustness evidence

The freeze rule requires at least two outcome-independent safe subsets to agree in IPV direction. Under the corrected frame:

- `primary_inclusion`: 53 cells.
- `safe_s1_inclusion`: 53 cells, exactly the same cell set as primary.
- `safe_s2_inclusion`: 53 cells, exactly the same cell set as primary.
- `safe_s3_inclusion`: 6 cells, `delta Spearman=0`, MAE worsens by `-0.091855`.

S1 and S2 duplicate the primary because all `safety < 100` rows are official A1 and are already excluded by the non-A1 primary rule. The reported `safe_subset_agreement_count=2/3` is therefore a bookkeeping artifact, not an independent robustness check.

Impact: Any wording that says the safe-subset requirement is meaningfully met is overclaiming. The corrected result lacks a meaningful safe-subset robustness pass.

### RT2-BLOCK-004: Current result language can overclaim the nonsignificant direction

The corrected markdown correctly reports the CI and p-value, but it also labels the row `confirmatory`, says `direction=favorable`, reports `safe_subset_requirement_met=True`, and refers to the "corrected confirmatory result" without clearly stating that the favorable direction is nonsignificant, underpowered, LOSO-near-null, and control-anomalous.

Impact: Before any reader-facing synthesis, paper wording, evidence row, or final decision uses this run, the language must be rewritten to "suggestive/nonsignificant favorable direction only" and must explicitly state that robustness is blocked.

## Non-Blocking Findings And Checks

### Scenario crosswalk

No blocking residual label mismatch found. `replay_score_mapping.csv` has 150 top-five team-area-case rows and 15 official scenario codes, each appearing 10 times. The official code spaces are `A1-A7`, `B1-B4`, and `C1-C4`. Spot checks against raw `tjjhs_referee_scoring` lines confirm the raw case names:

- Shanghai case 2325: `1-行人横穿道路`, official A1.
- Beijing case 2333: `12-行人横穿道路`, official A1.
- Beijing case 2344: `1-施工区避障`, official A7.

The stored SQL pointer is not exact-session provenance for 120/150 rows, but the case-name/code mapping itself remains supported by the official score CSV plus raw SQL case-name sequences.

### A1 and safety

The corrected A1/safety identity is consistent. The only top-five safety<100 rows are:

| team | area | case_id | official scenario | raw name | safety | coordination |
|---|---|---:|---|---|---:|---:|
| T15 | beijing | 2333 | A1 | 12-行人横穿道路 | 0 | 0 |
| T20 | beijing | 2333 | A1 | 12-行人横穿道路 | 0 | 0 |

### Flip source

The before/after primary sample changed from 48 to 53 cells. The five added cells are old-label A1 rows for Beijing teams T14/T15/T16/T17/T20 that are official A7 case 2344. No old-primary cells were dropped. This supports the view that the primary flip came from the legitimate scenario/A1 label correction and inclusion change, not from future-leaky imputation or LOSO regeneration.

### Fold regeneration

Primary LOTO remains team-based and has no train/test team overlap. Corrected LOSO/LOFO folds were regenerated in memory from official scenario/family labels because the frozen fold file contains the old erroneous A1-C5 grid. This is a legitimate error-fix for secondary/boundary analyses, but it should be formalized in an amended freeze/fold contract before any final report relies on LOSO or family transfer.

### State strata and multiple comparisons

No interpretable state-dependence row reached `q <= 0.10`; minimum FDR q was `0.628317`. The corrected state report's abstention framing is appropriate.

### Prior stats re-review

`stats_rereview2` found no internal statistical pipeline defect and passed the corrected run with caveats. That result is useful for internal validity, but it does not remove the red-team blockers above because it did not provide a corrected clean-room replication and did not require negative controls or safe subsets to support a robust favorable conclusion.

### IPV versus kinematics and criterion boundaries

In the corrected primary sample, the largest absolute Spearman correlation between directional IPV and a baseline kinematic feature was about `0.31` (`D_yield_auc` with lateral-gap AUC). This does not reduce IPV to a single obvious kinematic recoding, but the official coordination outcome remains a generated score, not an independent human social-ground-truth measure. Coordination was correlated with comprehensive score (`rho=0.420`) and heading-difference AUC (`rho=-0.522`) in the corrected primary sample.

### Stale reader metadata

`00_entry/index.html`, `90_report/index.html`, `README.md`, and `execution_status.json` still describe the run as initialized/pending/no formal statistics. This is not a blocker for the corrected tables, but it is a traceability issue before reader-facing release.

## Acceptance Criteria Results

| Criterion | Result |
|---|---|
| Pre-write identity | PASS |
| RT-BLOCK-001 resolved? | PASS for labels; non-blocking provenance pointer defect remains |
| RT-BLOCK-002 resolved? | PASS |
| Flip legitimacy | Legitimate label/inclusion correction for primary LOTO, but not independently reproduced |
| Favorable robustness | FAIL, blocking |
| Overclaim check | FAIL, blocking if any final wording treats favorable direction as established |
| Degradation-control anomaly | FAIL, blocking for robustness |
| Safe-subset >=2 agreement | Technically true but not meaningful; blocking for robustness wording |
| State strata FDR | PASS, no q<=0.10 promotion |
| Train/heldout overlap | PASS in fold assignment checks |

## Required Bottom Line

The corrected analysis should read:

> After correcting official scenario labels and A1/safety membership, the primary LOTO comparison changes from null/reverse to a weak favorable direction, but the effect is nonsignificant, underpowered, not supported by LOSO, not meaningfully supported by independent safe subsets, not robust to negative controls, and not yet independently reproduced. No verifier-validation or robust incremental-utility claim is accepted from this run.
