# RQ015 v1.1 Execution Review — Codex Lane B

Date: 2026-07-26
Reviewer: Codex independent execution review lane B
Verdict: **PASS_WITH_CONDITIONS**

## Scope

Frozen baseline verified against:

- `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md`
- `reports/plans/RQ015_plan_v1p1_checksums_20260726.sha256`

Reviewed surfaces:

- RQ015 plan v1.1
- `src/sociality_estimation/core/reliability_logdomain.py`
- `tests/test_rq015_reliability_logdomain.py`
- current live estimator entrypoints/callers:
  - `src/sociality_estimation/core/agent.py`
  - `src/sociality_estimation/core/ipv_estimation.py`
  - `pipelines/interhub/process_interhub.py`
  - `src/sociality_estimation/verifier/anchors.py`
  - `scripts/rq014/build_wod_m3_anchors.py`
- managed HPC/run-spec governance:
  - `configs/run_specs/research_run_spec_v2.schema.json`
  - `configs/run_specs/README.md`
  - `scripts/hpc/prepare_research_run.py`
  - `scripts/hpc/submit_research_run.sh`
  - `tests/test_hpc_run_launcher.py`

Old v1 review material was used only as a closure checklist, then re-verified against current code.

## Summary

- Blockers: 0
- Majors: 1
- Minors: 0

The frozen v1.1 plan hash matches the manifest. The isolated `reliability_logdomain.py` prototype closes several v1 defects: `grid_id` is now present, the result object is versioned, fail-closed input checks exist, D1 partial-underflow behavior is explicitly tested, and the two underflow boundaries are separated. After re-checking the gate semantics, I do **not** treat the missing production compatibility bridge or missing RQ015-managed HPC lane as plan-review blockers, because the plan now states both explicitly as not-yet-delivered execution prerequisites and keeps `execution_authorized=false`. The remaining approval condition is a real text/API inconsistency: `AT_GRID_BOUNDARY` is still described as both a status and a diagnostic flag.

## What Passes

- Frozen plan hash matches the manifest:
  - `de68bd15eb560a428d3146b4f68a88263eaaf168d3e7880f53989d692a0f8d21  reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md`
- The prototype now carries the core B1/B2 metadata fields and fail-closed guards:
  - `grid_id`, `schema_version`, `estimator_version`, `mse_per_candidate`, optional `step_sq_residuals` at `src/sociality_estimation/core/reliability_logdomain.py:62-81`
  - explicit validation and typed failure returns at `src/sociality_estimation/core/reliability_logdomain.py:88-99` and `src/sociality_estimation/core/reliability_logdomain.py:181-195`
- The executable D1/D2/D3/D4 classifier is materially stronger than in v1:
  - current classifier and priority order at `src/sociality_estimation/core/reliability_logdomain.py:231-260`
  - targeted tests for total underflow, harmless partial underflow, harmful boundary-crossing partial underflow, D2 vs D3 split, fail-closed paths, sufficiency limits, and two threshold definitions at `tests/test_rq015_reliability_logdomain.py:79-205` and `tests/test_rq015_reliability_logdomain.py:231-272`
- The prototype remains safely isolated:
  - module header says it is not imported by any production path at `src/sociality_estimation/core/reliability_logdomain.py:3-7`
  - current tree import search found only the unit test import: `tests/test_rq015_reliability_logdomain.py:14`

## Findings

### [MAJOR] The frozen B2 contract is still internally inconsistent on boundary semantics

Evidence:

- Plan §4.2 lists `AT_GRID_BOUNDARY` inside the status-code vocabulary: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:139-143`.
- The same plan later describes an “orthogonal result contract” with one mutual-exclusive `status` plus coexisting diagnostic `flags`: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:195-214`.
- The current implementation consistently treats boundary-hit as a flag rather than a terminal status: `src/sociality_estimation/core/reliability_logdomain.py:46-49`, `src/sociality_estimation/core/reliability_logdomain.py:207-214`.
- The tests also validate boundary-hit as a flag coexisting with `STATUS_MODEL_MISFIT`: `tests/test_rq015_reliability_logdomain.py:231-240`.

Why this matters:

This is no longer a code bug in the prototype; the code and tests are internally consistent. The remaining problem is that the frozen plan text still exposes two incompatible public contracts for the same concept. That is enough to create downstream schema/API ambiguity if B2 is approved as-written.

Minimal fix:

- Freeze one public rule before approval:
  - either `AT_GRID_BOUNDARY` is a diagnostic flag outside the terminal status vocabulary,
  - or it is a true terminal status and the code/tests must be updated accordingly.
- Make §4.2, §4.4b, and §8 use the same vocabulary.

## Execution Gates

The following items remain real **execution prerequisites**, but under the current v1.1 text I no longer treat them as plan-review blockers because the plan states them explicitly and keeps execution denied.

### [NON-BLOCKING EXECUTION GATE] The old-interface compatibility / abstention bridge is not delivered yet

Evidence:

- The plan now explicitly says `BUILD_WHILE_DENY / B1_PROTOTYPE / B2_SCAFFOLD_NOT_WIRED`, states that production wiring is not delivered, and says it must not be represented as a deployable abstention chain: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:195-214`.
- The prototype itself says the production wiring is not delivered and that it is not imported by any production path: `src/sociality_estimation/core/reliability_logdomain.py:3-7`.
- The live estimator still uses legacy `cal_traj_reliability()` and only produces scalar `ipv` / `ipv_error`: `src/sociality_estimation/core/agent.py:243-255`, `src/sociality_estimation/core/agent.py:1078-1141`.
- `estimate_ipv_pair()` still returns only two numeric arrays, seeds warm-up rows as `ipv_values=0` and `ipv_errors=1`, and has no status/flags/version channel: `src/sociality_estimation/core/ipv_estimation.py:213-215`, `src/sociality_estimation/core/ipv_estimation.py:251-252`, `src/sociality_estimation/core/ipv_estimation.py:334-372`.
- InterHub outputs still persist only numeric IPV/error means plus coarse case status, not per-row estimator status/reason/version/ledger fields: `pipelines/interhub/process_interhub.py:58-70`, `pipelines/interhub/process_interhub.py:863-885`, `pipelines/interhub/process_interhub.py:1236-1245`.
- Verifier/M3 anchor assembly still blindly converts `counterpart_ipv` and `counterpart_ipv_error` to floats, with no abstention/status contract: `src/sociality_estimation/verifier/anchors.py:21-35`, `src/sociality_estimation/verifier/anchors.py:130-135`.

Gate meaning:

This is still unfinished implementation work, and B2 should not be claimed complete until the adapter and integration tests exist. But because the plan now says exactly that and forbids production enablement before completion, this is an execution gate rather than a reason to reject the v1.1 plan text outright.

Required before execution:

- Define the canonical adapter from `ReliabilityResult` to live estimator outputs.
- Add persisted status/reason/version/ledger fields where the plan requires them.
- Add explicit verifier-side abstention handling before any M3 envelope scoring.
- Add integration tests that exercise non-`OK` rows through `estimate_ipv_pair()`, InterHub export, and verifier/M3 anchor construction.

### [NON-BLOCKING EXECUTION GATE] The managed run-spec / HPC execution surface is still RQ014-only

Evidence:

- The plan itself says the current run-spec and launcher surface is hard-wired to `RQ014`, the legacy sigma01 checkout is tombstoned, and the RQ015 HPC path is “字面不可执行” until the execution surface is extended or replaced: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:326-334`.
- The schema still fixes `rq_id` to `RQ014` and only enumerates RQ014 operations: `configs/run_specs/research_run_spec_v2.schema.json:4-30`.
- The run-spec README still documents only the `RQ014` bootstrap command and explicitly frames the wrapper as an `RQ014` authorization boundary: `configs/run_specs/README.md:10-28`.
- The loader rejects any v2 spec whose `rq_id` is not `RQ014`: `scripts/hpc/prepare_research_run.py:865-876`.
- Submission API and CLI entry mode both reject anything except schema-v2 `RQ014`: `scripts/hpc/prepare_research_run.py:4774-4780`, `scripts/hpc/prepare_research_run.py:4796-4803`.
- The shell wrapper still invokes the launcher only through `--rq014-only`: `scripts/hpc/submit_research_run.sh:179`.
- Focused launcher tests still encode that only `RQ014` v2 is allowed and that cross-label/generic submission is rejected: `tests/test_hpc_run_launcher.py:931-977`, `tests/test_hpc_run_launcher.py:980-1024`.

Gate meaning:

This is a genuine execution-governance gap, but the plan now labels it correctly as a start-up prerequisite rather than pretending the lane already exists. Since `execution_authorized=false` and §10 says the current HPC path is literally non-executable, I treat this as an explicit stop gate before future Phase B replay, not as a defect in the reviewed v1.1 plan text itself.

Required before execution:

- Keep `execution_authorized=false` until an RQ015-specific managed operation or an explicitly approved alternate governed lane exists.
- Freeze and test the corresponding schema/template/launcher/allowlist path before any Formal G1 that implies executability.

## Closure Check Against Old v1 Review

The following old-v1 issues appear closed in the current prototype and should not be re-opened:

- `grid_id` is now present in `ReliabilityResult`: `src/sociality_estimation/core/reliability_logdomain.py:72-81`.
- Independent `schema_version` / `estimator_version` are now present: `src/sociality_estimation/core/reliability_logdomain.py:79-81`.
- D3 is no longer silently disabled by a nullable default; `min_mse_misfit` is required and validated: `src/sociality_estimation/core/reliability_logdomain.py:160-179`, `tests/test_rq015_reliability_logdomain.py:154-161`.
- Partial-underflow D1 behavior is now explicitly covered, including the harmful boundary-crossing case the old review called out: `tests/test_rq015_reliability_logdomain.py:97-134`.
- The two distinct underflow boundaries are now named and tested separately: `src/sociality_estimation/core/reliability_logdomain.py:264-277`, `tests/test_rq015_reliability_logdomain.py:248-256`.

## Verification Performed

### Frozen hash check

- `shasum -a 256 reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md`
- Observed hash matched `reports/plans/RQ015_plan_v1p1_checksums_20260726.sha256`.

### Diagnostics

`mcp__omx_code_intel.lsp_diagnostics` reported 0 diagnostics for:

- `src/sociality_estimation/core/reliability_logdomain.py`
- `tests/test_rq015_reliability_logdomain.py`
- `src/sociality_estimation/core/agent.py`
- `src/sociality_estimation/core/ipv_estimation.py`
- `pipelines/interhub/process_interhub.py`
- `src/sociality_estimation/verifier/anchors.py`
- `scripts/hpc/prepare_research_run.py`

Tooling note:

- The available diagnostics backend reported `npx tsc --noEmit --pretty false`; this is not a Python typechecker, so I treated it as a lightweight sanity pass rather than full Python static typing evidence.
- `ast-grep` is not installed in this environment, so AST pattern checks were unavailable.

### Safe read-only tests

- `PYTHONPATH=src .venv_ipv_local_test/bin/python -m pytest -q tests/test_rq015_reliability_logdomain.py`
  - Result: `18 passed in 0.10s`
- `PYTHONPATH=src .venv_ipv_local_test/bin/python -m pytest -q tests/test_ipv_estimator_parity.py`
  - Result: `5 passed, 1 skipped in 14.48s`
  - Notes: emitted only dependency deprecation warnings from matplotlib/pyparsing/scipy, no test failures.
- `PYTHONPATH=src .venv_ipv_verifier/bin/python -m pytest -q tests/test_hpc_run_launcher.py -k 'rq014_legacy_schema_is_rejected_before_filesystem_or_git_checks or v2_base_override_is_rejected_before_alternate_checkout_access or cli_entry_mode_accepts_only_rq014_v2_submission or cli_entry_mode_rejects_rq014_v2_validate_without_internal_wrapper_mode'`
  - Result: `4 passed, 72 deselected in 0.13s`

Correction to my earlier lane-B note:

- The prior statement that no repo-local test environment was available was incorrect. This rerun used the existing `.venv_ipv_local_test` and `.venv_ipv_verifier` environments successfully.

## Recommendation

**PASS_WITH_CONDITIONS**

Conditions:

1. Resolve the remaining public-contract ambiguity on `AT_GRID_BOUNDARY` before treating B2 as frozen API/schema text.
2. Preserve the current gate semantics: the missing compatibility bridge and missing RQ015 HPC lane are still mandatory execution prerequisites, but the plan may pass only as a non-executable phased package with `execution_authorized=false` until those gates are closed.

---

## Post-Verdict Addendum (2026-07-26)

This addendum responds to a post-verdict evidence challenge. The original verdict text above is preserved for audit. After independently re-checking the cited probes against the current tree, my **final disposition supersedes the earlier recommendation and changes to `BLOCKED`**.

### Updated severity

- Blockers: 3
- Majors: 3
- Minors: 0

### [BLOCKER] Fail-closed input validation is not actually closed

Evidence:

- `estimate_reliability()` validates only `ipv_range.shape == (K,)`, not finiteness: `src/sociality_estimation/core/reliability_logdomain.py:175-177`.
- `grid_id` is required by signature but never validated for non-empty string semantics: `src/sociality_estimation/core/reliability_logdomain.py:160-163`, `src/sociality_estimation/core/reliability_logdomain.py:223-228`.
- `k_eff_flat_ratio` is accepted verbatim and used in the status gate with no finiteness/range validation: `src/sociality_estimation/core/reliability_logdomain.py:162`, `src/sociality_estimation/core/reliability_logdomain.py:218-221`.
- `legacy_divergence_tol` is accepted verbatim and used in D1 classification with no validation: `src/sociality_estimation/core/reliability_logdomain.py:239-257`.
- Independent local probe reproduced:
  - nonfinite `ipv_range` can yield `status=OK` with `ipv=NaN`;
  - empty string and `None` both pass through as `grid_id`;
  - `k_eff_flat_ratio=NaN` / `inf` can suppress flat-likelihood detection and return `OK`;
  - `legacy_divergence_tol=NaN` / `inf` silently changes D1 behavior.

Why this blocks approval:

The reviewed module is explicitly described as fail-closed and as the frozen contract scaffold for B2, but these parameter surfaces still admit silent contract corruption and `OK` rows with invalid payloads.

Minimal fix:

- Validate `ipv_range` finiteness before weighting and reject any nonfinite entry.
- Require `grid_id` to be a non-empty string.
- Require finite bounded `k_eff_flat_ratio` and finite nonnegative `legacy_divergence_tol`.
- Add explicit tests for all four surfaces.

### [BLOCKER] Finite positive tiny-`sigma` inputs still fail via raw `AssertionError`

Evidence:

- `weights_from_mse()` computes `-mse / (2.0 * sigma ** 2)` and then asserts on the denominator, rather than mapping this numeric collapse to a typed fail-closed result: `src/sociality_estimation/core/reliability_logdomain.py:115-127`.
- Independent local probe reproduced:
  - `sigma=1e-200` -> `AssertionError: stable softmax denominator collapsed`
  - `sigma=1e-300` -> `AssertionError: stable softmax denominator collapsed`
- The current tests only cover `sigma in {0.0, -0.1, NaN, inf}` and therefore miss finite-positive underflowing `sigma`: `tests/test_rq015_reliability_logdomain.py:210-215`.

Why this blocks approval:

This directly contradicts the module’s fail-closed claim. A finite positive input should not escape as an untyped assertion failure.

Minimal fix:

- Detect numerically degenerate `sigma**2` / `logw` regimes before the assert path.
- Return a typed `EstimatorInputError` or mapped `SOLVER_FAILURE` / `NON_FINITE_INPUT` outcome instead of `AssertionError`.
- Add regression tests for tiny but positive `sigma`.

### [BLOCKER] The frozen B2 schema/contract is broaderly inconsistent, not just on `AT_GRID_BOUNDARY`

Evidence:

- The plan’s explicit per-row schema listing is only:
  `ipv, ipv_error, status, reason_code, K, grid_id, min_mse, loglike_gap, at_grid_boundary, mse_per_candidate[K], estimator_version`: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:121-124`.
- The same plan also says the delivered contract is an orthogonal `status + flags + reason_code + schema_version/grid_id/estimator_version` triple and mentions optional `step_sq_residuals`: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:202-207`.
- The implementation exposes additional contract fields not present in the frozen schema listing:
  `weights`, `flags`, `k_eff`, `step_sq_residuals`, `schema_version`, `sufficiency_scope`: `src/sociality_estimation/core/reliability_logdomain.py:62-81`, `src/sociality_estimation/core/reliability_logdomain.py:223-228`.
- `AT_GRID_BOUNDARY` remains contradictory between status vocabulary and diagnostic-flag implementation: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:139-143`, `src/sociality_estimation/core/reliability_logdomain.py:46-49`, `src/sociality_estimation/core/reliability_logdomain.py:207-214`.

Why this blocks approval:

The plan says B2 freezes the schema before implementation. The current text still does not describe the actual object shape the prototype returns, so the review cannot approve this as a frozen contract.

Minimal fix:

- Publish one exact contract shape and vocabulary.
- Either remove non-schema fields from the implementation surface or add them to the frozen schema text.
- Resolve `AT_GRID_BOUNDARY` consistently as status or flag, not both.

### [MAJOR] The `boundary="zero"` threshold and its tests encode the wrong cutoff

Evidence:

- The code uses `limits = {"subnormal": np.finfo(float).tiny, "zero": 5e-324}`: `src/sociality_estimation/core/reliability_logdomain.py:272`.
- That computes:
  - `n=5 -> 1.7336185332654104`
  - `n=11 -> 1.1752447977715996`
- For actual round-to-zero behavior, the cutoff should be half the smallest subnormal, which gives:
  - `n=5 -> 1.7344180025598073`
  - `n=11 -> 1.1757808479007583`
- The test suite currently locks the smaller numbers as the expected `zero` threshold: `tests/test_rq015_reliability_logdomain.py:248-253`.

Why this matters:

The module comment says `boundary="zero"` means the product rounds to exact zero. The implementation and tests instead encode the minimum-subnormal boundary, not the actual rounding boundary.

Minimal fix:

- Rename the current boundary to what it actually means, or adjust the implementation to match the documented round-to-zero semantics.
- Update the tests and plan text accordingly.

### [MAJOR] My earlier closure check against the old v1 checksum manifest was no longer valid on current paths

Evidence:

- Current bytes are:
  - `de68bd15...  reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md`
  - `da9cb010...  src/sociality_estimation/core/reliability_logdomain.py`
  - `45dd60fb...  tests/test_rq015_reliability_logdomain.py`
- The old v1 checksum manifest still records different hashes for those same paths:
  - `2c214b0d...` for the plan
  - `a02b1a6e...` for the module
  - `fb527b6c...` for the test
  at `reports/plans/RQ015_plan_v1_checksums_20260726.sha256:1-4`.
- The current v1p1 manifest matches the current tree for those paths: `reports/plans/RQ015_plan_v1p1_checksums_20260726.sha256:1-4`.

Why this matters:

This does not by itself break the v1p1 review, but it means my earlier use of old-v1 checksum-backed closure language was too loose. The old v1 package is no longer verifiable against the current path contents because v1.1 overwrote those paths.

Minimal fix:

- Treat old-v1 manifest checks as historical only unless the exact v1 bytes are restored elsewhere.
- Use only the v1p1 manifest for current-path integrity claims.

### [MAJOR] Plan §10 claims an archived WOD adapter copy that is not present

Evidence:

- The plan says there is a copy under `archived/report_process/RQ010B_ipv_rating_pilot_20260629/`: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:331-332`.
- The directory currently contains only:
  `README.md`, `analyze_wod_e2e_ipv_rating_pilot.py`, `infer_streampetr_e2e_pilot.py`, `prepare_wod_e2e_pilot_assets.py`, `run_rq010b_wod_e2e_ipv_rating_pilot.sbatch`.

Why this matters:

This is a concrete documentation error in the execution guidance section.

Minimal fix:

- Correct the path claim or remove it from §10.

## Final disposition

The earlier `PASS_WITH_CONDITIONS` no longer stands. The newly verified probes show that the reviewed prototype still has real fail-closed defects and a still-unfrozen contract surface. My final disposition is:

**BLOCKED**
