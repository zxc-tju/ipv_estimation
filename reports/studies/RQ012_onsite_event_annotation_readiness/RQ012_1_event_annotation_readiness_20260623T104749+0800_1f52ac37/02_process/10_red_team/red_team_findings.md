# RQ012A W16 Red Team Findings

Worker ID: RQ012-W16-red-team  
Role: reviewer, adversarial red team  
Run ID: RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37  
Git HEAD reviewed: 13aafc86b29295187372173f408b6160b133c3c7

## Scope Guard

I reviewed RUN_ROOT artifacts, pilot extractor code/tests, merge validation code/tests, release transport evidence, and derived custody/mode evidence only. I did not read the paper repository. I did not read or reproduce controlled identity-map contents. I did not compute any event-IPV association.

## Prioritized Blocking List

1. **B11 - Required readiness HTML report is missing/non-substantive.** The SPEC requires `onsite_event_annotation_readiness_report.html`, but the run contains only `90_report/index.html`, whose body is the bootstrap text `RQ012A run initialized`. This is already confirmed by phase-10 review and remains unresolved.

## Per-Vector Findings

### V01 - Recycled-threshold circularity

Severity: **MAJOR**

Evidence:

- E01 counterpart hard braking and E02 high deceleration are distinct ontology entries, but their core central kinematic threshold is the same deceleration magnitude and duration/gap family: E01 uses `T_hard_brake_decel_E01`, `D_min_E01`, `G_merge_E01` (`01_results/event_ontology.yaml`, lines 48-56), and E02 uses `T_decel_E02`, `D_min_E02`, `G_merge_E02` (`01_results/event_ontology.yaml`, lines 89-95).
- The threshold table gives both E01 and E02 central deceleration `>=3.4 m/s^2`, `D_min=0.3 s`, `G_merge=0.3 s`, and `G_missing=0.3 s` (`01_results/event_threshold_rationale.md`, lines 23-33).
- The current pilot prevents one collapse by refusing to emit E01 until a frozen counterpart relation exists (`02_process/05_extractor_pilot/extractor_pilot.py`, lines 709-716). That is a current guard, not a permanent ontology distinction.
- E18 is more differentiated centrally because it requires ego stop/brake support (`extractor_pilot.py`, lines 1137-1169), but its high sensitivity band becomes broad `decel or brake` and the pilot report shows high-band E18 count 202 while central is 0 (`01_results/automatic_event_pilot_report.md`, lines 71-75).

Risk:

Once E01 becomes computable, the same non-ego hard-deceleration episode can become both E01 and E02 unless a precedence or subset rule is added. Treating those as independent endpoints would inflate apparent evidence.

Required fix:

Define a cross-event hierarchy before any later RQ012B use: either collapse E01/E02 into one canonical deceleration event with role/context qualifiers, or explicitly mark E01 as an E02 subset and forbid using both as independent primary endpoints for the same actor-window. Apply the same subset/precedence logic to E18 versus E02 for ego hard-stop cases.

### V02 - Threshold outcome-tuning

Severity: **NOT_FOUND**

Evidence:

- The threshold rationale explicitly states no IPV, deviation, official score, rank, team identity, later event outcome, annotation label, agreement result, or event-IPV association was used (`01_results/event_threshold_rationale.md`, line 8).
- The B02 section states no data-derived confirmatory threshold value was used and describes the required future firewall if data-derived thresholds are introduced (`02_process/04_thresholds/threshold_rationale_detail.md`, lines 62-76).
- The extractor read-scope audit says the extractor read OnSite trajectory logs and RQ003 item IDs for exclusion only, and did not read IPV, official score, rank, team identity, labels, agreement, or event-outcome associations (`data/derived/.../extractor_pilot/read_scope_audit.json`, lines 5-6).

Required fix:

None for current artifacts. Preserve the B02 firewall if thresholds are changed later.

### V03 - Timestamp misalignment

Severity: **MAJOR**

Evidence:

- Pair extraction aligns each world row to the nearest ego row within 100 ms (`extractor_pilot.py`, lines 1000-1025), then records the pair row using the world timestamp (`extractor_pilot.py`, lines 1049-1059).
- Ego rows selected by `nearest_ego_rows` are only sorted; duplicate ego timestamps are not rejected there (`extractor_pilot.py`, lines 983-986). The pair path calls `row_quality_for_emission` on the matched ego row with `previous_t_s=None`, so duplicate/non-increasing ego timestamps are not detected in pair events (`extractor_pilot.py`, lines 1026-1031).
- W12 tests cover duplicate timestamps for E02 actor-series extraction, not for E09/E15 pair alignment (`tests/test_extractor_pilot.py`, lines 460-470).
- Red-team synthetic probe: duplicate ego timestamp with conflicting ego geometry emitted one E09 interval with `event_count=1`, `raw_frame_hits=2`, `impossible_values=0`, and `missing_data_failures=0`. That is a silent pair-event detection from an ambiguous ego time base.

Required fix:

Pre-validate ego time series for pair events before nearest-neighbor alignment. Reject or deterministically collapse duplicate/non-monotonic ego timestamps, record the impossible-value diagnostic, and add W12 tests for E09/E15 duplicate ego timestamps, tied nearest-neighbor selection, and mismatched actor time bases.

### V04 - Actor attribution errors

Severity: **MAJOR**

Evidence:

- `read_world_rows` chooses `actor_id = id or originId`, stores `origin_id` and `name`, then groups rows solely by `actor_id` (`extractor_pilot.py`, lines 371-394).
- E09/E15 pair extraction iterates each `world_by_actor` key and emits `counterpart_id=world:<actor_id>` without checking stable `origin_id`, name, source, or neutral relation over the interval (`extractor_pilot.py`, lines 1009-1014 and 1089-1093).
- E01 is safely deferred because the frozen counterpart relation is unavailable (`extractor_pilot.py`, lines 709-716), but E09/E15 still treat every non-ego actor key as a pair candidate.
- No team identity was found in the corrected annotator-facing templates or release zip, so this is an extractor actor-stability risk rather than a neutral-ID/team leakage finding.

Required fix:

Add an actor-stability guard for world actors: require stable `id` plus `originId`/name where available, split actor windows on identity changes, count actor-attribution failures per affected row/window, and avoid using E09/E15 pair intervals as primary evidence when actor identity is unresolved.

### V05 - Frame-level duplicate events

Severity: **MAJOR**

Evidence:

- The interval merge logic deduplicates runs within one event result only (`extractor_pilot.py`, lines 625-665). The pilot report also reports duplicate rates per event, not across event types (`01_results/automatic_event_pilot_report.md`, lines 27-39).
- E09 contact precedence is implemented using a fixed `central_contact_tolerance = 0.0` even when the E15 band has a different overlap tolerance (`extractor_pilot.py`, lines 1004 and 1079-1082).
- The high E15 band uses `overlap_tolerance_m=0.1`, while high E09 uses `distance_m=1.0` (`extractor_config.json`, lines 36-37). Therefore a same-pair clearance of 0.05 m is both an E15 high-band contact candidate and an E09 high-band near miss.
- Red-team synthetic probe with the same pair/time at 0.05 m clearance for 0.3 s produced high-band counts `E09=1` and `E15=1` for the same interval. Central counts were `E09=1`, `E15=0`, showing that the duplication is setting-dependent and not caught by current per-event overlap checks.

Required fix:

Compute E15 first under the same active band and suppress E09 for any same pair/time interval classified as contact by that band. Add a cross-event duplicate audit table for E09/E15, E01/E02, and E02/E18, with tests proving same physical intervals cannot inflate endpoint counts.

### V06 - Filename/media metadata leakage and release transport

Severity: **MAJOR**

Evidence:

- Direct scan of the corrected release zip entries for team/score/rank/IPV/area/scenario/path/proxy terms produced no hits. The recorded `zipinfo` shows only four neutral text/CSV files and zero-byte extra fields (`02_process/06_blind_package/release/zipinfo.txt`, lines 19-133).
- The controlled identity map and internal stratification record are present only under the derived blind-package directory and are mode `600`; contents were not read or reproduced.
- However, the issuance and release manifests still carry `UNSIGNED_PENDING_INDEPENDENT_REAUDIT` auditor fields (`blind_issuance_manifest.csv`, lines 1-42; `blind_issuance_release_transport_manifest.csv`, lines 1-6), while the re-audit nevertheless closes Gate 012-2 for text/issuance surfaces (`blind_package_reaudit.md`, lines 51-64 and 101-109).
- Final neutral media/card issuance remains explicitly pending (`blind_package_reaudit.md`, lines 97-99; `phase_status_reaudit.json`, lines 5-7). The current release proves only the four text/CSV surfaces, not future media/card metadata.

Required fix:

Before any real label acceptance, create the final neutral media/card package, run the same no-proxy and metadata checks on those files, refresh support-file checksums, and replace unsigned auditor placeholders with a signed re-audit record. If this is not done, downgrade Gate 012-2 to text-only cleared and keep formal issuance blocked.

### V07 - Annotator training contamination

Severity: **NOT_FOUND**

Evidence:

- The training package states it contains fictional neutral-ID stubs, no real labels, and no formal validation items (`01_results/annotator_training_package/README.md`, lines 1-15).
- The illustrative key says it is not derived from real study outcomes, prior annotations, formal validation items, or automated annotations (`illustrative_training_key.md`, lines 1-14).
- The separation rule states training answers must not be copied, adapted, or used as evidence for formal items (`training_vs_formal_separation.md`, lines 1-7).

Required fix:

None for current artifacts. Keep training stubs fictional and outside formal raw-label custody.

### V08 - Copied labels / duplicate annotators

Severity: **MAJOR**

Evidence:

- `merge_validate.py` rejects only byte/hash-identical annotation files as copied duplicate submissions (`merge_validate.py`, lines 422-428).
- It rejects a duplicate human only when the two provenance sidecars report the same `coordinator_verified_submitter_id` (`merge_validate.py`, lines 406-418). If sidecars claim different identities, the script has no independent channel-log verification.
- The blind annotation protocol and checklist require checks for near-identical label patterns, impossible completion timing, controlled channel evidence, and duplicate/model anomalies (`blind_annotation_protocol.md`, lines 96-106; `human_coordination_checklist.md`, lines 35-43), but W8 tests cover only byte-identical copied files and missing provenance (`test_merge_validation.py`, lines 142-154 and 353-357).

Required fix:

Implement a near-duplicate/collusion screen in merge validation or a mandatory pre-merge provenance report: row-level label-vector similarity, timing/notes similarity, completion-time plausibility, template-order checks, and controlled-channel submitter evidence. Treat high-similarity non-identical pairs as quarantined pending coordinator review, not agreement-ready inputs.

### V09 - Simulated agreement

Severity: **NOT_FOUND**

Evidence:

- The agreement protocol is frozen before results, records `labels_generated_by_this_worker=false` and `agreement_computed_by_this_worker=false`, and does not authorize event-IPV analysis (`agreement_analysis_protocol.yaml`, lines 1-10 and 124-152).
- Merge validation initializes and preserves `agreement_computed=false`, `event_ipv_association_computed=false`, and `merge_or_agreement_output_created=false` (`merge_validate.py`, lines 444-455).
- Merge tests assert those flags remain false for every fixture (`test_merge_validation.py`, lines 297-310), and the published merge-test report repeats that no agreement or association was computed (`annotation_merge_validation_tests.md`, lines 22-29).

Required fix:

None for current artifacts. Keep the agreement computation in a later separate gate after two accepted real human submissions.

### V10 - Independence over-claim

Severity: **NOT_FOUND**

Evidence:

- The binding addendum explicitly says the behavioural reference must not be described as fully independent truth and should be described as outcome-blind behavioural evidence unless a later stricter tier is approved (`plan_resolution_v0_1.md`, line 11).
- The codebook and label-detail files use `independent_consequence_endpoint` as a tier meaning conceptually more separable from the IPV construct, not ground truth (`annotation_codebook_v2.md`, lines 31-40; `codebook_v2_label_detail.md`, lines 24-30).
- Searches over the key report/protocol/codebook surfaces found no `ground truth` claim.

Required fix:

None for current artifacts. Continue using "outcome-blind behavioural evidence" and avoid "ground truth" language.

### B11 - Bonus: required readiness report missing

Severity: **BLOCKING**

Evidence:

- The SPEC requires `onsite_event_annotation_readiness_report.html` and formal report figure artifacts (`reports/plans/RQ012_plan_v0_onsite_event_annotation_readiness_20260622.md`, lines 230-253).
- The run contains no `onsite_event_annotation_readiness_report.html`; the only `90_report/index.html` is a one-line bootstrap placeholder: `RQ012A run initialized`.
- Phase-10 independent review already recorded this as blocking and required creation of the SPEC-named readiness HTML or an approved path substitution (`02_process/09_review/independent_review.md`, lines 52-56). That blocker is still present.
- `01_results/annotation_readiness_status.json` remains `DRAFT` and `BLOCKED_FOR_HUMAN_LABELS` (`annotation_readiness_status.json`, lines 1-12).

Required fix:

Create the SPEC-named readiness HTML report, or record an explicit SPEC-approved path substitution, and make it substantive: gate statuses, open human-label boundary, leakage and metadata status, extractor limitations, merge-validation limits, final media/card dependency, and evidence map. Then rerun phase-10/phase-11 review.

## Final Verdict

RED TEAM: BLOCKERS_PRESENT
