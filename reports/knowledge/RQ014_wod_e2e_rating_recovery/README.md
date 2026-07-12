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

No `decision.md` should be created until empirical evidence reaches the registered claim gates.
