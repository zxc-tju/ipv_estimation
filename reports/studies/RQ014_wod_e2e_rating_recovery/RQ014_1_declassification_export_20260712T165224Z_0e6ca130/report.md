# RQ014 G2 declassification export — bounded report (v1.6 Waves 0–3)

- STATUS: PASS_RATING_BLIND_EXPORT_READY_FOR_PI_DECISION
- OPERATION: rq014_g2_declassification_export (sole centrally allowlisted operation; scoped PI decision v1.5)
- RUN_ID: RQ014_0_score_stripped_export_20260712T154921Z_1ee1e1d1
- GIT_COMMIT: 24be08278adf43371fda14e7ec23a95b986b2fb1 (exact reviewed contract commit; detached managed HEAD)
- SPEC_SHA256: 0e6ca13094adc14286f8a1f39d57013322fa2e0e2fb9edc12494c470bb31f62b (3,098 bytes; canonical JSON;
  published read-only 0444 inode 95871301641 as direct child of manifests/RQ014/run_specs/; W1-A and W1-B derived
  byte-identical bytes independently; binding rule: final spec is never chmod/rewritten/replaced/unlinked/renamed)
- JOB_ID_AND_STATE: 1919412 zxc-rq014-export-0e6ca13094ad — COMPLETED ExitCode 0:0, elapsed 00:03:52,
  partition amd node cpua102, 1 CPU / 8 GiB / 2 h, all six thread limits =1, --export=NIL on directive AND
  sbatch command (sacct SubmitLine verified); single submission by Lead-as-W2-A via the frozen clean-environment
  fd8/fd9 bootstrap (validate-only first, then same spec bytes with --submit)
- SOURCE_HASH_VERDICT: 10/10 exact — 8 phase1_post_scene_bundle.pkl + rated479_segment_readiness.tsv +
  selected_counterpart_tracks.csv match frozen inventory path/size/SHA-256 (W0-C on-HPC re-hash; launcher enforced
  again in-job); v3 environment manifest 30de86f7…, python 616aea77…, stdlib 14,326/307,357,072/0 symlinks/zip
  absent, native closure 20 rows — all exact; code snapshot 75/75 blobs (74-row contract bundle + self-pinned
  bundle) match registered digests incl. wrapper d8036336…, launcher 6b3cf6da…, preflight f91bbd2a…,
  materializer d8cac79f…, exporter 39e8c812…
- OUTPUT_BUNDLE_PATH_AND_HASHES: /share/home/u25310231/ZXC/sociality_estimation/inputs/RQ014/
  wod_rated479_score_stripped/v1 — exactly 9 regular non-symlink files (7 CSV + 2 JSON); every CSV hash equals its
  file_manifest.json row; export receipt seals file_manifest_sha256=4c41172f… and
  sanitization_receipt_sha256=4dfc8105…; DONE.json (rq014-managed-operation-done-v1, status PASS) seals
  receipt_sha256=a3839a7a…; local copies + hashes archived under .codex-fleet/rq014-execution-v1p6/board/w2_evidence/
- UNIVERSE_AND_GEOMETRY_COUNTS: universe_segment_count=479 exact; geometry available=476; structural
  attrition=3; candidate_count_distribution {0:3, 3:476}; blind_scene_manifest rows=479, structural_attrition
  rows=3, candidate_states rows=29,529, ego_history rows=7,616, ego_future rows=9,520, tstar_pose rows=7,616,
  counterpart rows=14,665 (all equal manifest row_count)
- FORBIDDEN_FIELD_AND_SCHEMA_SCANS: in-job scans forbidden/unexpected/duplicate/nonfinite = 0/0/0/0; independent
  W2-C re-scan of all 9 files for the 17 forbidden field patterns = 0 matches; parent-absolute-path scan of CSVs
  = 0; formats ASCII/UTF-8 CSV/JSON only (no TFRecord/protobuf/pickle/symlink); 7 CSV headers exactly match
  RQ014_score_stripped_schema_v1.json
- RATING_ACCESS: NONE (export receipt field; no rated479_segments TFRecord, ratings CSV, joined table or observed
  statistic was opened at any wave). OBSERVED_STATISTICS: NONE.
- DEVIATIONS: (1) validate-only attempt 1 was cut by a local 2-minute tool timeout at stdlib-check 7000/14326 —
  side-effect-free by contract and verified (no run root/receipt/process); clean rerun attempt 2 passed.
  (2) W2-C receipt verifier returned two literal FAILs against the Lead's over-strict checklist (identity fields
  expected inside export receipt/DONE; manifest rows expected for the 2 JSONs); blob-level adjudication
  (W3 execution reviewer + Lead) confirmed the produced receipts conform exactly to the reviewed authority:
  sanitization receipt carries all 17 attestation_required_fields 1:1, export receipt/DONE schema ids equal the
  contract's required_prior_receipts ids, and the two JSONs are hash-sealed by the export receipt (self-listing
  impossible); no receipt was hand-modified. (3) Foreign non-RQ014 job 1919345 "DW_export_feat" (workdir
  /share/home/u25310231/ZZ) ran on the account during the window; disjoint from managed roots; RQ014 job matching
  used exact name+workdir. (4) Managed-checkout sync script went through 4 red-team rounds (v1→v4) before GO; sync
  executed once, attestation archived; bundle retained at manifests/RQ014/bootstrap/ pending this report's
  acceptance. (5) Output bundle files are mode 0600 on HPC (owner-only; no mode clause in contract). (6) W1-B
  round-2 NO_BLOCKER (publication-rule re-review) was received 2026-07-12T16:06:45Z, before spec publication at
  16:09:30Z, but was persisted to the fleet board only after publication; the W3 execution reviewer flagged the
  stale "re-review pending" ledger line during report review. Durable artifact now persisted
  (board/reports/w1b-spec-adversarial-reviewer-round2.md, verbatim capture + capture-file mtime + publication-log
  timeline); validation ledger amended. Record-keeping deviation only; §5.2 ordering was honored in fact.
- REVIEWS: W0 A/B/C PASS (13/13, 12/12 + 74/74, 10/10+closures); W1-A/W1-B byte-identical spec + W1-C command
  re-derivation; validate-only parsed independently by W1-A and fresh W1-D (14/14, NO_BLOCKER both); W2-B monitor
  identity checks EXACT; W2-C artifact verification (data-clean, hash-chain PASS); W3 statistics reviewer and W3
  execution/governance reviewer (distinct fresh identities): NO_BLOCKER / NO_BLOCKER.
- NEXT_CENTRAL_AUTH_DECISION: D1 — PI decides whether to accept this rating-blind export and, separately, whether
  to authorize preparing the addition of rq014_g2_contract_preflight to the central allowlist (that authority-byte
  change then requires new scoped decision artifact, rebuilt candidate manifest, fresh dual review with distinct
  identities, new Formal G1 and final bundle, immutable spec, validate-only, before any preflight submit).
  Recommended default per v1.6 §8: accept (all scans zero, counts exact). If no reply: STOP_AND_PRESERVE.
  No rating access, no new operation, no allowlist change will occur before explicit D1 acceptance.
