# RQ014 G2 contract preflight — bounded report (v1.6 Wave 4 / §9)

- STATUS: PASS_PREFLIGHT_READY_FOR_PI_DECISION_D2
- OPERATION: rq014_g2_contract_preflight (added to central allowlist via the D1-authorized loop: PR #10/#11;
  science amendment v1.7 + lane v3 + registries v1p6 via PR #12; WOD path-type freeze package via PR #13;
  blind-anchor cross-phase fix via PR #14 — each with fresh dual review, regenerated Formal G1 and final bundle)
- RUN_ID: RQ014_1_wod_rating_recovery_20260713T161542Z_41ac5280 (a prior RUN_ID …T115004Z_bdad1e4c was burned by
  the cross-phase defect failure of job 1922605 and is preserved unmodified per §6; never reused)
- GIT_COMMIT: b06a243eea7e1418622f89e5ea80d3da4fe3bc58 (exact reviewed commit; HPC HEAD detached there;
  published ancestor of origin/main dc2f6bd3)
- SPEC_SHA256: 72dd4362f9546dcf24b792bd3c0f1c6413f1232f069e57891bcbb293fd316c14 (2,499 bytes, canonical, 0444,
  inode 79545352196, direct child of run_specs/; derived byte-identically by two independent agents; includes
  declassification_export_commit=24be0827… and the preflight-only m3_artifact binding)
- JOB_ID_AND_STATE: 1924193 zxc-rq014-pre-72dd4362f954 — COMPLETED ExitCode 0:0, 00:03:26, partition amd node
  cpua041, 2 CPU / 4 GiB / 1 h, all six thread limits =1, --export=NIL on directive AND sbatch command; single
  submission after explicit user confirmation at the promised hard stop
- FORMAL_G1_AND_BUNDLE: RQ014_formal_G1_v1p6_preflight_20260713.yaml sha 755e6a34… = FORMAL_G1_PASS
  (real-validator run; reviewed manifest 4fb4af0c…, 101 rows); final bundle RQ014_plan_v1p6_checksums_20260713
  sha 41ac5280… (105 rows)
- VERIFIED_INPUT_ROLES (all PASS in preflight receipt 1e2d0cf6…, sealed by DONE a138754f…):
  wod bundle file_manifest 4c41172f…; sanitization receipt 4dfc8105…; blind anchor 80e393f7…/1,752 B at the
  fixed root inputs/RQ014/blind_anchor/v1 (cross-phase shared validator, launcher prepare:1936 = runtime
  preflight:1713); WOD path-type mapping manifest 8c48d0eb… + csv 6e689647… (254 rows CP115/HO90/MP48/F1;
  222 undecidable excluded at F, 3 structural at K); registry bindings 12/12 materialized (ledger 2413a457…,
  input manifest 1be6248e…); M3 artifact verified path/size 88,306,301/sha b04999ab… with retained-fd
  O_NOFOLLOW pre-deserialization check (deserialized:false) and immutable M3 input receipt written
- SNAPSHOT: 106-file closed snapshot (105 bundle rows + self-pinned bundle) at exact commit blobs; wrapper
  3ad13e11…/launcher f7eb384a…/preflight 0492a5e5…/materializer f4d4f621…/exporter 39e8c812… digests exact
- RATING_ACCESS: FORBIDDEN enforced; receipt scans zero; slurm .err empty; no rating/rank/preference value
  appeared anywhere (names-only tokens verified benign). OBSERVED_STATISTICS: NONE.
- DEVIATIONS: (1) first submission failed fail-closed at the G2 input-manifest gate due to a reviewed
  cross-phase validator defect (launcher-vs-runtime root asymmetry made the blind-anchor receipt unsatisfiable);
  root-caused with blob citations, fixed via PR #14 with 3 new test functions plus strengthened assertions in
  one existing contract test, dual NO_BLOCKER, G1/bundle regenerated;
  failed run root preserved. (2) Two transient infra events during the wave (codex backend stream disconnect;
  permission-classifier outage) — no state impact, retries clean. (3) Local /tmp review-clone accumulation
  (~72 GB) cleaned post-wave; future review rounds will avoid full clones.
- NEXT_CENTRAL_AUTH_DECISION: D2 — PI decides whether to accept this preflight and authorize the next
  operation loop (rq014_g2_resource_pilot per §10, which requires the same per-operation authorization chain:
  scoped decision → allowlist commit → candidate manifest → fresh dual review → Formal G1 → bundle → PR →
  immutable spec → validate-only → submit). Also pending from §10: managed environment closure v4 (scientific
  stack for G2R) as its own gated work item. Recommended default per §8: accept (no contract drift; all
  verifications exact). If no reply: STOP_AND_PRESERVE.
