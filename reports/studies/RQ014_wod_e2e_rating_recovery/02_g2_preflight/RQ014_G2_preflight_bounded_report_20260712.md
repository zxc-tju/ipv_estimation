# RQ014 G2 bounded report — preflight contract stop

STATUS: BLOCKED

PHASE: G2_PREFLIGHT_CONTRACT_CONSISTENCY (before phase 1; RUN_ID not allocated)

FILES_WRITTEN:

- `reports/studies/RQ014_wod_e2e_rating_recovery/02_g2_preflight/RQ014_G2_preflight_bounded_report_20260712.md`
- `START_HERE.md` (current operating status only)
- `STUDIES.md` (RQ014 status only)
- `main_workflow.log` (append-only workflow record)

KEY_EVIDENCE:

- The v1.4 checksum manifest verified 16/16 files. Its SHA-256 is
  `d7f738d791f755d4240a0eb3d587b51aaa2d76aa4f1a7f52dc543aaad2ab190f`.
- Kickoff prose states F05–F10 are all `INACCESSIBLE_PI_WAIVED`:
  `reports/plans/prompts/RQ014_G2_kickoff_prompt_20260711.md`.
- The checksum-bound forensic registry instead records F09 as `INACCESSIBLE`; only
  F05–F08/F10 are `INACCESSIBLE_PI_WAIVED`:
  `reports/plans/RQ014_forensic_registry_v1p3.yaml`.
- The checksum-bound PI decision agrees with the registry and waives only F05–F08/F10:
  `reports/plans/RQ014_PI_decision_G0_waiver_launch_20260711.md`.
- The kickoff requires an immediate stop when prose and a resolved registry conflict.
- Authorization was otherwise consistent across all three registries: G2 blind build and
  scientific compute are true; the other six scopes are false.
- Tongji access reached `tongji-hpc`; only path/file-name/size metadata was queried.
  No rating-bearing file contents or rating-derived statistics were read.
- No RQ014 HPC directory was created, no RUN_ID was allocated, no file was transferred,
  and no `sbatch` job was submitted.

GATES_PASSED_FAILED:

- PASS — repository guidance and Tongji shared usage guide read.
- PASS — v1.4 checksum verification, 16/16.
- PASS — authorization-vector consistency across the three resolved registries.
- FAIL — prose-versus-registry consistency (F09 terminal state).
- NOT_RUN — phase 1 hash freeze and all later G2/G2P phases.

DEVIATIONS:

- Execution deviation: none; fail-closed behavior followed the kickoff stop rule.
- Contract gap for the next revision: phase 1 says to fill every `TO_FREEZE_AT_G2`
  registry field while source-registry modification and registry SHA drift are forbidden.
  A sanctioned run-scoped materialized-registry copy/transform must be specified.
- Contract gap for the next revision: A1–A4 exact rho recomputation requires rating
  information, while G2 forbids rating fields and FL05-like rating statistics. Register a
  blind parity receipt/fixture or explicitly redefine the permitted anchor check.
- Contract gap for the next revision: base v1 requires ratings/split manifests in the
  input manifest, but G2 cannot mount `contains_rating=true` files and selected partitions
  do not freeze until G2P. Specify a metadata-only handoff and staged-finalization rule.
- Contract gap for the next revision: all three registries retain
  `execution_authorized: false`; document precedence of scoped authorization booleans.
- Contract gap for the next revision: X02 declares two source artifacts but the extension
  registry provides one aggregate `artifact_sha256` without a canonical digest rule.

NEXT_DECISION_FOR_PI:

- Issue a checksum-bound clarification that (1) makes the kickoff F09 statement match the
  registry/PI decision, and (2) resolves the five execution gaps above without enabling
  rating access. After the new manifest verifies, restart at G2 preflight; do not resume
  from phase 1 under the current v1.4 bundle.
