# RQ014 — WOD-E2E rating↔IPV-deviation lost-result recovery

Status: v1.5 `FORMAL_G1_PASS` for the first managed declassification step; no accepted empirical or manuscript claim.

## Current contract

The active candidate is base v1 + v1.3 provenance/hardening +
`reports/plans/RQ014_plan_v1p5_amendment_20260712.md` + the primary
`reports/plans/RQ014_recovery_lane_v2.json`. v1.3 bytes remain immutable and replayable.
The attempted v1.4 launch is retained only as provenance: its waiver intent survives, while its illegal
`INACCESSIBLE_PI_WAIVED` state, self-declared G1, ambiguous booleans, mutated v1p3 paths, and retired HPC
execution paths are superseded.

G0 is represented legally as `CLOSED_WITH_INACCESSIBLE_SURFACES`: F01–F04 are
`NOT_FOUND_ON_SCANNED_SURFACES`; F05/F06/F07/F08/F10 are `INACCESSIBLE` with orthogonal PI-waiver
metadata; F09 is independently inaccessible. Negative-findings language covers F01–F04 only.

Formal G1 was reached on the sixth fresh-review round. Statistics and execution/governance reviewers both
returned `NO_BLOCKER` over the same 68-file manifest; the machine-readable adjudication is
`reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/RQ014_formal_G1_v1p5_20260712.yaml`.
Any reviewed-byte drift makes this adjudication stale.

The primary science route is now explicitly specification recovery, not null-hypothesis discovery. A
rating-blind 960-cell grid covers 4/10 Hz, true causal-history, look-ahead-future, two-sided combined,
t*-prefix and full-future windows, three envelope constructions, two horizons, and ten deviation readouts.
After a separately authorized one-time full-rated479 join, three association definitions yield 2,880
append-only rows; rank 1 is frozen and independently reimplemented in a clean replay. The old split/power/
confirmation machinery is optional non-gating follow-up and cannot hide a historically correct recipe.

## Rating-blind boundary

Read-only source-code verification found that the original rated479 TFRecord payloads still embed
`preference_score`; the old “no scores” reader also read full score-bearing CSV rows before dropping the
column. G2 therefore cannot mount TFRecord, protobuf, pickle, scored targets, ratings CSV, or joined tables.

The first staged operation is `rq014_g2_declassification_export`. It may read only the eight exact
score-omitting Phase-1 bundles, the structural readiness table, and the selected counterpart table whose
hashes are frozen in the source inventory. It exports a strict allowlisted CSV/JSON bundle and runs a full
validator before atomic publication. `rq014_g2_contract_preflight` remains denied until that export has a
validated completion receipt. The launcher binds each source path/size/SHA to the reviewed inventory,
binds the exact Python binary through a structured environment manifest, and requires formal-review bytes
to equal executed bytes.

A1–A4 are public aggregate receipts in G2; rho is not recomputed. Any G3R recovery join and G4R clean replay
remain separately authorized future operations.

## Current execution status

- Focused v1.5 tests: 194 passed.
- Broader non-shortcut suite excluding the locally absent ignored RQ009 scorer-only module:
  202 passed, 1 skipped, 2 deselected.
- No rating value read; no RQ014 production run root created; no Slurm job submitted.
- HPC source inventory, restricted-unpickle structural audit, and producer data-flow audit are complete;
  476/479 scenes are geometry-available and three remain explicit structural attrition.
- Formal review, G1, and the 74-row final checksum bundle are complete; bundle SHA-256 is
  `1ee1e1d121b8d24cef7fdca93f05ddcccfcb3282b70727c606ce03c36984c933`. PR #5 merged the exact contract
  commit `24be08278adf43371fda14e7ec23a95b986b2fb1` to `origin/main` at
  `a738de44715abb118e5571eec42af30d9b1c6786`. Remaining launch gates are an exact read-only v2 run spec
  and launcher validate-only evidence; publication is no longer a blocker.
- The append-only Lead/Sub-Agent execution runbook is
  `reports/plans/RQ014_plan_v1p6_execution_handoff_20260712.md` (SHA-256
  `f007c290ea6bb1130b2df1b49c63e482e34cfc7147716f8d68dd4c918e81de0c`). It does not alter
  v1.5 science or authorization. It freezes Waves 0–8, independent reviewer roles, HPC sync/spec/
  validate-only/export acceptance, later implementation and 960/2,880-row execution, failure handling,
  and user decisions D1–D6. No user decision is required before the already authorized export bounded
  report; D1 is mandatory before enabling contract preflight. Independent execution/HPC and science/
  governance reviewers returned `NO_BLOCKER_AFTER_REMEDIATION`; review record:
  `reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/RQ014_v1p6_execution_handoff_review_20260712.md`.

## v1.6 Waves 0–3 execution record (2026-07-13)

- Managed checkout synced by reviewed incremental git bundle: HPC HEAD detached at exact `24be0827…`,
  `refs/remotes/origin/main` CAS-updated `b1476bd0…` → `eb1ade2b…` under the exclusive maintenance lock;
  sync script survived four adversarial review rounds before GO; attestation + bundle retained.
- Immutable spec `RQ014_0_score_stripped_export_20260712T154921Z_1ee1e1d1.json` (SHA-256 `0e6ca130…31f62b`)
  derived byte-identically by two independent agents, published read-only via staging hard-link no-replace.
- Validate-only evidence matched the frozen expected table (Lead + W1-A + fresh W1-D, 14/14; zero side effects).
- Single authorized submission: Slurm `1919412` `zxc-rq014-export-0e6ca13094ad`, COMPLETED 0:0 in 3m52s.
- Output: nine-file score-stripped bundle under the managed input root; universe 479, geometry 476,
  structural attrition 3 (reasons `MISSING_DECLASSIFIED_PHASE1_SCENE`), candidate distribution {0:3, 3:476};
  forbidden/unexpected/duplicate/nonfinite scans all zero; receipts hash-chained DONE→export→{sanitization,
  file_manifest}; sanitization receipt carries the 17 contract attestation fields 1:1.
- Dual W3 review (distinct fresh identities): statistics NO_BLOCKER (science primitives preserved; no
  association computed), execution/governance NO_BLOCKER (receipt schema conform; W2-C literal checklist
  items adjudicated as overreach with blob citations; one record-keeping deviation — W1-B round-2 verdict
  persisted late — disclosed and accepted on timeline evidence).
- Bounded report:
  `reports/studies/RQ014_wod_e2e_rating_recovery/RQ014_1_declassification_export_20260712T165224Z_0e6ca130/report.md`.
- Status: `PASS_RATING_BLIND_EXPORT_READY_FOR_PI_DECISION`; execution stopped at D1. No rating value was
  read; contract preflight remains centrally DENIED pending D1 and its own authorization loop.

No `decision.md` should be created until empirical evidence reaches the registered claim gates.

## Preflight wave record (2026-07-14)
- D1→D2 loop complete: PRs #10–#14 (allowlist, cross-commit provenance field, amendment v1.7/lane v3 M3-fixed
  320 cells, WOD path-type freeze 254/222@F/3@K, blind-anchor fixed-root + shared cross-phase validator), each
  with fresh dual review and regenerated FORMAL_G1_PASS. First submission failed fail-closed (cross-phase
  defect; RUN_ID burned, root preserved); fix reviewed; resubmission job 1924193 COMPLETED 0:0. All 12 bindings
  materialized; M3 delivery verified pre-deserialization with immutable receipt. Bounded report + evidence:
  reports/studies/RQ014_wod_e2e_rating_recovery/RQ014_2_contract_preflight_20260714T003336Z_72dd4362/.
  D2 accepted: resource-pilot authorization loop + managed environment closure v4 design authorized; pilot
  submit retains an explicit PI stop. Rating access: NONE throughout.
