# RQ015 v1 Execution Review — Codex Lane B

Date: 2026-07-26
Reviewer: Codex independent execution review lane B
Verdict: **BLOCKED**

## Scope

Reviewed the frozen v1 package against `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md` (SHA-256 `2c214b0dccaa126a009876c7aeec2d6895862e593a9478ba02adab139bf57cd6`), including:

- `reports/plans/prompts/RQ015_portrait_scan_v1.sh`
- `src/sociality_estimation/core/reliability_logdomain.py`
- `tests/test_rq015_reliability_logdomain.py`
- current estimator call sites in `src/sociality_estimation/core/agent.py` and `src/sociality_estimation/core/ipv_estimation.py`
- current InterHub output surfaces in `pipelines/interhub/process_interhub.py`
- current verifier anchor contract in `src/sociality_estimation/verifier/anchors.py`
- current managed HPC execution surface in `scripts/hpc/prepare_research_run.py`, `scripts/hpc/submit_research_run.sh`, and focused launcher tests

Files reviewed: 10

## Summary

- Blockers: 2
- Majors: 1
- Minors: 0

The new log-domain kernel is mathematically coherent and its isolated unit suite passes. The package is still blocked because the implemented schema/status surface does not yet match the frozen v1 contract, and the promised NaN/abstention compatibility layer is not delivered on any live estimator, InterHub, or verifier path.

## Findings

### [BLOCKER] B2 schema/status contract is not fully implemented

Evidence:

- The frozen plan requires per-row output fields `ipv, ipv_error, status, reason_code, K, grid_id, min_mse, loglike_gap, at_grid_boundary, mse_per_candidate[K], estimator_version` and status values `OK / NOT_ATTEMPTED / SOLVER_FAILURE / NON_FINITE_INPUT / MODEL_MISFIT / FLAT_LIKELIHOOD / AT_GRID_BOUNDARY` at [reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:118-138](/Volumes/ZHITAI%202T/.CloudStorage/Data/OneDrive-%E4%B8%AA%E4%BA%BA/Desktop/Projects/1_Codes/2_sociality_estimation/reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:118).
- `ReliabilityResult` omits `grid_id` entirely at [src/sociality_estimation/core/reliability_logdomain.py:46-61](/Volumes/ZHITAI%202T/.CloudStorage/Data/OneDrive-%E4%B8%AA%E4%BA%BA/Desktop/Projects/1_Codes/2_sociality_estimation/src/sociality_estimation/core/reliability_logdomain.py:46).
- The module defines `STATUS_NOT_ATTEMPTED` and `STATUS_SOLVER_FAILURE`, but `estimate_reliability()` never emits either one at [src/sociality_estimation/core/reliability_logdomain.py:33-39](/Volumes/ZHITAI%202T/.CloudStorage/Data/OneDrive-%E4%B8%AA%E4%BA%BA/Desktop/Projects/1_Codes/2_sociality_estimation/src/sociality_estimation/core/reliability_logdomain.py:33) and [src/sociality_estimation/core/reliability_logdomain.py:108-158](/Volumes/ZHITAI%202T/.CloudStorage/Data/OneDrive-%E4%B8%AA%E4%BA%BA/Desktop/Projects/1_Codes/2_sociality_estimation/src/sociality_estimation/core/reliability_logdomain.py:108).
- The frozen plan lists `AT_GRID_BOUNDARY` as a status value, but the implementation only emits a boolean `at_grid_boundary` and leaves status as `OK` unless another branch fires at [reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:134-138](/Volumes/ZHITAI%202T/.CloudStorage/Data/OneDrive-%E4%B8%AA%E4%BA%BA/Desktop/Projects/1_Codes/2_sociality_estimation/reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:134) and [src/sociality_estimation/core/reliability_logdomain.py:140-157](/Volumes/ZHITAI%202T/.CloudStorage/Data/OneDrive-%E4%B8%AA%E4%BA%BA/Desktop/Projects/1_Codes/2_sociality_estimation/src/sociality_estimation/core/reliability_logdomain.py:140).

Why this blocks approval:

The reviewed package claims B2 implementation, but the emitted object cannot represent all frozen contract states/fields. That means downstream consumers still cannot distinguish warm-up abstention, solver failure, and boundary-clamped estimates using the schema the plan says is frozen.

Minimal required fix:

- Add the missing `grid_id` field.
- Decide and encode the final boundary semantics: either `AT_GRID_BOUNDARY` is a real terminal status or the plan must be corrected before approval.
- Implement or explicitly bridge `NOT_ATTEMPTED` and `SOLVER_FAILURE` instead of only defining dead constants.
- Add tests that assert the final frozen status vocabulary, not just the current subset.

### [BLOCKER] The promised NaN/abstention compatibility layer is not delivered on live estimator, InterHub, or verifier surfaces

Evidence:

- The frozen plan makes B2 contingent on “schema frozen and implemented; NaN and old-interface compatibility layer delivered and tested; sufficient statistics persisted” at [reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:256-263](/Volumes/ZHITAI%202T/.CloudStorage/Data/OneDrive-%E4%B8%AA%E4%BA%BA/Desktop/Projects/1_Codes/2_sociality_estimation/reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:256).
- The plan also states that non-`OK` rows must set `ipv=NaN`, and that consumers must explicitly handle abstentions because current InterHub/M3/WOD outputs assume numeric results at [reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:136-138](/Volumes/ZHITAI%202T/.CloudStorage/Data/OneDrive-%E4%B8%AA%E4%BA%BA/Desktop/Projects/1_Codes/2_sociality_estimation/reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:136).
- The live estimator path still calls legacy `cal_traj_reliability()` and writes only scalar `subject.ipv` and `subject.ipv_error` at [src/sociality_estimation/core/agent.py:243-255](/Volumes/ZHITAI%202T/.CloudStorage/Data/OneDrive-%E4%B8%AA%E4%BA%BA/Desktop/Projects/1_Codes/2_sociality_estimation/src/sociality_estimation/core/agent.py:243).
- `estimate_ipv_pair()` still initializes pre-estimation rows as `ipv_values=0` and `ipv_errors=1`, then returns only two numeric arrays `(ipv_values, ipv_errors)` at [src/sociality_estimation/core/ipv_estimation.py:213-215](/Volumes/ZHITAI%202T/.CloudStorage/Data/OneDrive-%E4%B8%AA%E4%BA%BA/Desktop/Projects/1_Codes/2_sociality_estimation/src/sociality_estimation/core/ipv_estimation.py:213), [src/sociality_estimation/core/ipv_estimation.py:251-252](/Volumes/ZHITAI%202T/.CloudStorage/Data/OneDrive-%E4%B8%AA%E4%BA%BA/Desktop/Projects/1_Codes/2_sociality_estimation/src/sociality_estimation/core/ipv_estimation.py:251), and [src/sociality_estimation/core/ipv_estimation.py:334-372](/Volumes/ZHITAI%202T/.CloudStorage/Data/OneDrive-%E4%B8%AA%E4%BA%BA/Desktop/Projects/1_Codes/2_sociality_estimation/src/sociality_estimation/core/ipv_estimation.py:334).
- InterHub persistence/export still writes only `ipv_key_agent_*` and `ipv_key_agent_*_error` numeric columns, with no status, reason, `K`, `mse_per_candidate`, or version fields, at [pipelines/interhub/process_interhub.py:863-885](/Volumes/ZHITAI%202T/.CloudStorage/Data/OneDrive-%E4%B8%AA%E4%BA%BA/Desktop/Projects/1_Codes/2_sociality_estimation/pipelines/interhub/process_interhub.py:863).
- InterHub plotting still treats every row as a numeric band around IPV at [pipelines/interhub/process_interhub.py:903-916](/Volumes/ZHITAI%202T/.CloudStorage/Data/OneDrive-%E4%B8%AA%E4%BA%BA/Desktop/Projects/1_Codes/2_sociality_estimation/pipelines/interhub/process_interhub.py:903).
- The verifier anchor builder still blindly casts `counterpart_ipv` and `counterpart_ipv_error` to floats at [src/sociality_estimation/verifier/anchors.py:130-131](/Volumes/ZHITAI%202T/.CloudStorage/Data/OneDrive-%E4%B8%AA%E4%BA%BA/Desktop/Projects/1_Codes/2_sociality_estimation/src/sociality_estimation/verifier/anchors.py:130).

Why this blocks approval:

The package’s isolated kernel can abstain, but none of the live call sites or persistence surfaces currently carry, store, or gate on that abstention. Without the compatibility layer the plan explicitly requires, the “deployable” part of Phase B is still missing.

Minimal required fix:

- Define the canonical adapter between `ReliabilityResult` and existing estimator outputs.
- Add status/version/ledger persistence for InterHub outputs.
- Add verifier-side abstention handling before M3 envelope scoring.
- Cover the bridge with integration tests that exercise `estimate_ipv_pair()`, InterHub export, and verifier anchor construction on non-`OK` rows.

### [MAJOR] The managed HPC execution surface is still RQ014-only, so Phase B remains intentionally non-executable

Evidence:

- The frozen plan explicitly says the current run-spec/launcher surface is hard-wired to RQ014 and therefore “字面不可执行” for RQ015 until the execution surface is extended or a separate lane is created at [reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:275-283](/Volumes/ZHITAI%202T/.CloudStorage/Data/OneDrive-%E4%B8%AA%E4%BA%BA/Desktop/Projects/1_Codes/2_sociality_estimation/reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:275).
- The managed launcher constants and scopes are still exclusively RQ014-specific at [scripts/hpc/prepare_research_run.py:171-245](/Volumes/ZHITAI%202T/.CloudStorage/Data/OneDrive-%E4%B8%AA%E4%BA%BA/Desktop/Projects/1_Codes/2_sociality_estimation/scripts/hpc/prepare_research_run.py:171).
- The wrapper still executes the launcher only with `--rq014-only` at [scripts/hpc/submit_research_run.sh:175-179](/Volumes/ZHITAI%202T/.CloudStorage/Data/OneDrive-%E4%B8%AA%E4%BA%BA/Desktop/Projects/1_Codes/2_sociality_estimation/scripts/hpc/submit_research_run.sh:175).
- Focused launcher fixtures still build only `rq_id="RQ014"` specs at [tests/test_hpc_run_launcher.py:83-96](/Volumes/ZHITAI%202T/.CloudStorage/Data/OneDrive-%E4%B8%AA%E4%BA%BA/Desktop/Projects/1_Codes/2_sociality_estimation/tests/test_hpc_run_launcher.py:83) and [tests/test_hpc_run_launcher.py:118-135](/Volumes/ZHITAI%202T/.CloudStorage/Data/OneDrive-%E4%B8%AA%E4%BA%BA/Desktop/Projects/1_Codes/2_sociality_estimation/tests/test_hpc_run_launcher.py:118).

Why this matters:

This is not a hidden defect; the plan states it clearly. But it remains a real feasibility gap. No reviewer should interpret the current package as ready for HPC submission or full sigma01 replay until a governed RQ015 execution lane exists.

Minimal required fix:

- Keep `execution_authorized=false` until an RQ015-specific managed operation is defined and tested, or until the work is explicitly routed through a separate approved execution channel.

## Verification Performed

### Frozen artifact checks

- `shasum -a 256 reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md reports/plans/prompts/RQ015_portrait_scan_v1.sh src/sociality_estimation/core/reliability_logdomain.py tests/test_rq015_reliability_logdomain.py`
  - Observed digests matched `reports/plans/RQ015_plan_v1_checksums_20260726.sha256` for the frozen plan, portrait script, module, and test file.

### Diagnostics

- `mcp__omx_code_intel.lsp_diagnostics` on:
  - `src/sociality_estimation/core/reliability_logdomain.py`
  - `tests/test_rq015_reliability_logdomain.py`
  - `src/sociality_estimation/core/agent.py`
  - `src/sociality_estimation/core/ipv_estimation.py`
  - `pipelines/interhub/process_interhub.py`
- Result: 0 reported diagnostics on all five files.

### Safe read-only tests

- `PYTHONPATH=src ./.venv_ipv_verifier/bin/python -m pytest -q tests/test_rq015_reliability_logdomain.py`
  - Result: `7 passed in 0.14s`
- `PYTHONPATH=src ./.venv_ipv_verifier/bin/python -m pytest -q tests/test_ipv_estimator_parity.py`
  - Result: `5 passed, 1 skipped in 21.95s`
- `./.venv_ipv_verifier/bin/python -m pytest -q tests/test_hpc_run_launcher.py -k 'rq014_legacy_schema_is_rejected_before_filesystem_or_git_checks or v2_base_override_is_rejected_before_alternate_checkout_access'`
  - Result: `2 passed, 74 deselected in 0.13s`

### Tooling note

- `mcp__omx_code_intel.ast_grep_search` could not run because `ast-grep` is not installed in this environment. I did not treat that as a blocker because the core review is already grounded by direct source inspection, checksums, diagnostics, and targeted tests.

## Recommendation

**BLOCKED**

The isolated numerical repair is credible, but the package does not yet satisfy the frozen B2 contract or the deployable abstention/compatibility surface described in the plan. The minimum unblock is:

1. Finish the frozen schema/status implementation (`grid_id`, final status vocabulary, real `NOT_ATTEMPTED` / `SOLVER_FAILURE` handling).
2. Deliver and test the old-interface compatibility bridge across `estimate_ipv_pair()`, InterHub export, and verifier ingestion.
3. Keep HPC execution denied until an RQ015-governed execution surface exists.
