# Codex Statistical Review — RQ015 v1

Date: 2026-07-26
Reviewer: Codex (independent stats/scientific lane A)
Plan under review: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md`
Frozen manifest: `reports/plans/RQ015_plan_v1_checksums_20260726.sha256`

## Verdict

**BLOCKED**

Counts: **1 blocker, 2 major, 2 minor**

The core scientific framing is materially improved versus v0: the proxy boundary is now faithful to RQ007, held-out isolation is explicit, the D0 warm-up artifact is separated from estimability, and the sigma discussion is corrected from a prior absolute claim to an outcome-blind empirical question. The blocker is narrower but real: the package currently overstates B2 completeness. The checked-in draft module does not yet satisfy the schema/status/consumer contract that the plan itself says is required before abstention is scientifically safe.

I did **not** rely on prior review outputs for this verdict. I reviewed the frozen v1 plan, checksum manifest, portrait script, BUILD_WHILE_DENY module/tests, the legacy estimator path, and representative current consumers.

## What Passes

- The proxy boundary is correctly constrained to the frozen RQ007 dev/guard result and explicitly not promoted to a direct IPV-error interpretation: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:61-78`, `reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/decision.md:3`, `reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/decision.md:28-34`.
- The warm-up/D0 correction is real and consistent with the current estimator initialization path: `START_HERE.md:16-22`, `src/sociality_estimation/core/ipv_estimation.py:247-252`.
- The legacy underflow mechanism is real: the production estimator still multiplies per-step Gaussian terms and falls back to uniform weights when `sum(var)==0`: `src/sociality_estimation/core/agent.py:1104-1140`.
- The new log-domain draft is mathematically aligned with the stated rewrite and the frozen checksums match the manifest: `src/sociality_estimation/core/reliability_logdomain.py:7-18`, `src/sociality_estimation/core/reliability_logdomain.py:74-82`, `reports/plans/RQ015_plan_v1_checksums_20260726.sha256:1-4`.
- The RQ014-only HPC execution-surface blocker is real and correctly called out: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:275-283`, `configs/run_specs/README.md:10-27`, `scripts/hpc/prepare_research_run.py:3379-3388`, `scripts/hpc/prepare_research_run.py:4774-4804`.

## Findings

### Blocker 1 — B2 is not actually complete enough to support the scientific abstention contract

The plan says B2 requires a frozen schema, explicit status codes, `grid_id`, `mse_per_candidate`, and a compatibility layer so non-OK rows become `NaN` without being silently re-consumed as ordinary numeric IPV in existing outputs: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:116-138`, `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:256-263`.

The checked-in draft module is still short of that contract:

- `ReliabilityResult` has no `grid_id`: `src/sociality_estimation/core/reliability_logdomain.py:46-61`.
- The plan lists `AT_GRID_BOUNDARY` as a status code, but the code exposes only a boolean `at_grid_boundary`; no such status is emitted or tested: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:134-138`, `src/sociality_estimation/core/reliability_logdomain.py:33-39`, `src/sociality_estimation/core/reliability_logdomain.py:140-158`, `tests/test_rq015_reliability_logdomain.py:118-135`.
- `SOLVER_FAILURE` and `NOT_ATTEMPTED` exist only as constants or plan concepts; the draft entrypoint cannot emit them because it has neither solver-failure plumbing nor frame/min-observation context: `src/sociality_estimation/core/reliability_logdomain.py:33-39`, `src/sociality_estimation/core/reliability_logdomain.py:108-158`.
- The compatibility layer is not present. Representative consumers still assume dense numeric `ipv_values`/`ipv_errors` arrays and no status channel:
  - InterHub export/plot writes plain columns and plots `ipv ± ipv_error`: `pipelines/interhub/process_interhub.py:854-930`.
  - The WOD M3 builder hard-fails on nonfinite terminal IPV/error: `scripts/rq014/build_wod_m3_anchors.py:546-573`.

Why this is blocking: the scientific claim in §7 is that “measured-not-measured” becomes an explicit abstention before verifier use, not another silent recoding. With the current package, an executor still has to invent the schema/compatibility behavior that is supposed to make that claim true.

Minimal required fix:

1. Either narrow the package claim from “B1/B2 implementation has landed” to “B1 landed, B2 draft scaffold only”, or
2. Land the missing B2 pieces before approval: `grid_id`, full emitted status taxonomy, explicit serialization schema, and tested adapters for current numeric consumers.

### Major 1 — The knowledge-layer RQ015 README is still partially on v0 facts, despite the plan saying it must be corrected in the same batch

The plan says the v0 error corrections have already spread into `START_HERE.md` and `reports/knowledge/RQ015_ipv_estimability_contract/README.md` and that both must be corrected together: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:35-36`.

`START_HERE.md` is updated to v1 values and wording: `START_HERE.md:11-32`.

But the knowledge README still points to the v0 plan and still carries the stale `73.8%` tension number:

- stale v0 plan pointer: `reports/knowledge/RQ015_ipv_estimability_contract/README.md:3`
- stale `P(IPV=0 | error≥0.61)=73.8%`: `reports/knowledge/RQ015_ipv_estimability_contract/README.md:22`
- v1 authoritative value is `71.65%`: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:270`, `START_HERE.md:17-19`

Why this matters statistically: this knowledge-layer README is exactly where future downstream readers will look for the boundary claim. Leaving it half-on-v0 risks reintroducing the wrong quantitative framing after the current review.

Minimal required fix: update the README so the plan pointer, numeric tension line, and version summary all point only to v1.

### Major 2 — One cited evidence path is stale, so the parity-gate rationale is not traceable from the package as written

The plan cites `reports/knowledge/ipv_estimator_divergence_investigation.md` as the rationale for the hard parity gate: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:110-112`.

That file does not exist at the cited path. The actual investigation currently lives at `reports/knowledge/_analysis/ipv_estimator_divergence_investigation.md`, where the 0.281 sigma01 numeric drift is indeed documented: `reports/knowledge/_analysis/ipv_estimator_divergence_investigation.md:7-13`, `reports/knowledge/_analysis/ipv_estimator_divergence_investigation.md:97-113`.

Why this matters: the parity gate is one of the main reasons the review should trust B1 as “science-preserving”. A stale citation weakens that audit trail unnecessarily.

Minimal required fix: repair the path in the plan (or add a stable alias note) so the cited evidence is directly resolvable.

### Minor 1 — The plan’s “two existing log-domain implementations” statement is only partially verifiable from the repository

The live RQ014 adapter does still retain uniform fallback behavior on nonfinite or zero-total paths: `scripts/rq014/wod_ipv_adapter.py:61-91`.

But the plan states there are “two existing log-domain implementations (`scripts/rq014/wod_ipv_adapter.py` and archived copy)”: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:280-281`.

I could verify the live file, but I could not find an `archived/.../wod_ipv_adapter.py` counterpart in the repository tree during review. This is a traceability issue, not a scientific-contract blocker.

Minimal required fix: either name the archived path explicitly or drop the archived-copy clause.

### Minor 2 — The package’s “7 tests all pass” statement is not rerunnable in the current local interpreter without additional test tooling

The test file exists and its logic is coherent: `tests/test_rq015_reliability_logdomain.py:1-141`. The frozen checksum also matches: `reports/plans/RQ015_plan_v1_checksums_20260726.sha256:4`.

However, the current local interpreter does not have `pytest` installed, so I could not replay the exact test file by `python3 -m pytest`. I did run a manual Python smoke check that reproduced the key intended behaviors:

- parity sample: `max_abs_diff = 5.55e-17`
- underflow threshold sample: `1.6915 m`
- underflow case: legacy uniform fallback, new path finite with `k_eff = 1.0`
- flat-likelihood case: `status = FLAT_LIKELIHOOD`, `ipv = NaN`

This is a verification-environment gap, not a conceptual flaw in the package itself.

Minimal required fix: none for the plan; optionally note the expected test environment in future package metadata.

## Simulated Execution Checks

I mentally simulated three representative tasks against the actual codebase:

1. **Enable B1/B2 in the production estimator path.**
   The worker would have to change or wrap `src/sociality_estimation/core/agent.py:1078-1140` and likely `src/sociality_estimation/core/ipv_estimation.py:335-338`, but the current draft does not yet define how status/schema should leave the estimator in a way that downstream consumers can safely handle.

2. **Push abstention into current InterHub outputs.**
   `pipelines/interhub/process_interhub.py:854-930` still assumes two numeric arrays and directly exports/plots `ipv_values` and `ipv_errors`. A worker cannot complete this safely without additional B2 compatibility design that is not yet present in the package.

3. **Protect the WOD/M3 path from silent center-collapse semantics.**
   `scripts/rq014/build_wod_m3_anchors.py:546-573` explicitly rejects nonfinite terminal IPV/error, which is directionally safer than silently converting them to zero, but it means the “upstream abstention gate” contract in the plan still needs a concrete adapter boundary before any deployment story is valid.

## Bottom Line

Scientifically, the v1 direction is substantially better and mostly well-bounded:

- proxy boundary: acceptable
- held-out isolation: acceptable
- sigma derivation framing: acceptable
- verifier conditional-validity logic: acceptable in principle

What prevents approval is narrower: the checked-in package still describes B2 as if the scientifically critical schema/compatibility layer were already there, but the repository only contains a draft core module plus tests. Until that is either implemented or the package wording is narrowed, I cannot mark the v1 package PASS or PASS_WITH_CONDITIONS.
