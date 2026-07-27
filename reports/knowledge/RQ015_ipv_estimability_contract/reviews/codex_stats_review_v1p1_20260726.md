# RQ015 Plan v1.1 — Codex independent Lane A review

Reviewed plan: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md`  
Frozen baseline: SHA-256 `de68bd15eb560a428d3146b4f68a88263eaaf168d3e7880f53989d692a0f8d21`, verified against `reports/plans/RQ015_plan_v1p1_checksums_20260726.sha256`.

I did not inspect any v1p1 execution-review artifact and did not use old v1 review text beyond the closure claims already embedded in v1.1.

**PASS_WITH_CONDITIONS**

**Justification**: Most v1 blockers are genuinely closed. The revised plan now matches the real RQ007 dev/guard-only identifiability boundary, correctly downgrades frozen RQ009/M3 coverage to historical ungated marginal performance, and honestly presents the new log-domain work as a non-production B1 prototype plus B2 scaffold. I verified the live scaffold and ran the committed RQ015 unit suite in the project environment (`18 passed`). One material contract inconsistency remains inside the plan itself: §4.2 still treats `AT_GRID_BOUNDARY` as a status that would force `ipv=NaN`, while §4.4b and the implementation treat it as an orthogonal diagnostic flag. Executors should not have to guess which schema is authoritative.

Counts: blocker `0`, major `1`, minor `2`.

**Summary**:
- Clarity: Pass with condition. One internal schema contradiction remains.
- Verifiability: Pass. Key claims are grounded in code, frozen decision artifacts, and runnable tests.
- Completeness: Pass with minor gap. Phase A would be cleaner if it named the already-frozen split artifact paths directly.
- Big Picture: Pass. Measurement-vs-deployment and gate-vs-coverage boundaries are now correctly separated.

**What passes**:
- The D0 warm-up reclassification is genuine. `estimate_ipv_pair` still initializes `ipv_values=np.zeros` and `ipv_errors=np.ones`, then only overwrites from `t >= min_observation`: `src/sociality_estimation/core/ipv_estimation.py:251-252`, `271`, `334-338`.
- The RQ007 boundary cited by the plan is real: dev `19,258`, guard `7,628`, sealed `11,342`; estimability is explicitly frozen as an identifiability proxy, not a standard deviation: `reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/decision.md:3`, `14-16`, `24`, `30`. The concrete split artifact exists and is frozen: `reports/studies/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_process/00_meta/split_freeze.json:36-42`, `76-109`.
- The RQ009/M3 coverage caveat is correctly tightened. Accepted coverage is historical ungated marginal performance (`≈0.899`), subgroup/LODO validity is uneven, and the original plan explicitly forbids claiming conditional or source-shift nominal coverage: `reports/knowledge/RQ009_dynamic_counterpart_conditioned_envelope/decision.md:12`, `23-25`; `reports/plans/RQ009_plan_v0_dynamic_counterpart_conditioned_envelope_20260624.md:174-187`.
- The revised B1/B2 scaffold is honestly non-production and backed by code/tests. The live module is not imported by production paths and implements the orthogonal result contract, D1/D2/D3/D4 classifier, sufficiency boundary, and underflow thresholds: `src/sociality_estimation/core/reliability_logdomain.py:1-7`, `34-49`, `62-81`, `154-228`, `231-277`.
- I verified the revised unit suite under the project env: `PYTHONPATH=src /Users/xiaocong/.rq009_codex_fleet/venv/bin/python -B -m pytest -p no:cacheprovider tests/test_rq015_reliability_logdomain.py -q` -> `18 passed`.
- The plan is correct that a compatibility bridge and abstention audit are still required before deployment. Current consumers remain numeric-first: InterHub case processing saves/plots raw `ipv +/- ipv_error` and only reports `"ok"`/`"failed"` case statuses: `pipelines/interhub/process_interhub.py:58-70`, `1168-1245`. M3 anchor builders require finite counterpart IPV/error: `src/sociality_estimation/verifier/anchors.py:130-135`; `scripts/rq014/build_wod_m3_anchors.py:546-575`. Verifier deviation usability depends on finite observed IPV: `src/sociality_estimation/verifier/scorer.py:165-180`.

**Findings**:
- Major, definitely missing/contradictory: the plan still gives two incompatible B2 contracts for `AT_GRID_BOUNDARY`. §4.2 lists it as a status code and says non-`OK` rows carry `ipv=NaN`: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:139-143`. But §4.4b and the live scaffold define an orthogonal `status + flags` contract with `AT_GRID_BOUNDARY` as a coexisting flag, not a main abstention state: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:202-214`; `src/sociality_estimation/core/reliability_logdomain.py:34-49`, `63-70`, `207-225`. This is material because boundary rows may remain valid measurements or coexist with `MODEL_MISFIT`; downstream NaN propagation depends on this choice.
- Minor, definitely stale: the public `estimate_ipv_pair` docstring still says pre-`min_observation` rows are filled with `np.nan`, but the implementation returns the warm-up `0/1` placeholders that RQ015 now reclassifies as D0: `src/sociality_estimation/core/ipv_estimation.py:213-215`, `251-252`.
- Minor, possibly unclear: Phase A correctly says freeze/tune only on dev+guard and stop if fold provenance is unavailable: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:55-56`. But it still makes the executor search for the already-frozen split artifacts even though the exact source paths are recorded in `split_freeze.json`: `reports/studies/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_process/00_meta/split_freeze.json:71-74`, `84-109`.

**Minimal fixes**:
- Edit §4.2 so the authoritative B2 contract matches §4.4b and the scaffold: `AT_GRID_BOUNDARY` should be a diagnostic flag, not a main status. State explicitly whether `OK + AT_GRID_BOUNDARY` keeps finite `ipv`, and whether `MODEL_MISFIT + AT_GRID_BOUNDARY` is allowed.
- Update `estimate_ipv_pair` docstring to describe either the current legacy warm-up placeholders or the intended future `NOT_ATTEMPTED` emission path.
- In Phase A, cite the exact RQ007 split artifacts (`.../02_process/00_meta/split_freeze.json` and `data/derived/.../case_split_assignment.csv`) as the canonical sealed-exclusion source.

---

## Post-Verdict Addendum (2026-07-26, new-evidence challenge)

This addendum preserves the original review text above and records the result of an independent reassessment against the five enumerated challenge items only. I did **not** read the execution-lane review or any v1p1 synthesis artifact during this reassessment.

### Final disposition after reassessment

**BLOCKED**

The original `PASS_WITH_CONDITIONS` no longer holds. Items (1)-(5) together show that the remaining issues are not merely cleanup or wording defects; they leave the plan without a single-valued scientific contract for Phase B classification and without a frozen, testable acceptance rule for the post-selection M3 coverage audit.

Updated counts: blocker `2`, major `3`, minor `1`.

### Reassessment findings

- **Blocker 1, definite**: the B2 result contract is still internally contradictory at the core schema level.
  `AT_GRID_BOUNDARY` is listed as a terminal status in plan §4.2, which would force `ipv=NaN`: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:139-143`.
  But plan §4.4b and the implemented scaffold define `status + flags`, with `AT_GRID_BOUNDARY` explicitly in the orthogonal diagnostics lane, not the terminal state lane: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:202-214`; `src/sociality_estimation/core/reliability_logdomain.py:34-49`, `63-70`.
  This is blocker-grade because downstream NaN propagation, aggregation, and abstention behavior depend on which contract is authoritative.

- **Blocker 2, definite**: the plan requires `min_mse_misfit` as a frozen scientific threshold but never freezes the rule that selects it.
  The live scaffold correctly makes `min_mse_misfit` mandatory: `src/sociality_estimation/core/reliability_logdomain.py:160-179`, and plan §4.4b states D2/D3 separation depends on it: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:222-223`.
  But the plan never specifies the unit, statistic, split artifact, pass rationale, or decision rule that will freeze that threshold. Phase A says only that some rows must be re-run under a sampling precision contract: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:260-263`.
  This leaves the executor guessing how `min_mse_misfit` becomes scientifically valid. The over-strong D2 wording in §4.4 ("IPV 对轨迹无杠杆——真实发现") also exceeds what the current Gaussian-grid model can support before that threshold rule is frozen: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:188-190`.

- **Major, definite**: the post-selection coverage audit is still not operationalized enough to verify deployment readiness.
  Plan §4.6 correctly says the frozen `≈0.899` RQ009 result is only a historical ungated marginal result and requires a new outcome-blind audit on `gate-pass × estimator_version`: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:234-251`.
  But the plan still does not freeze the evaluation split, target artifact, nominal levels, CI/tolerance band, or pass/fail criterion for that audit.
  At the same time, §7 still uses deployable language: "本 RQ 的可部署交付即是把它改为'无法判定'": `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:300-301`.
  Without a machine-actionable audit success rule, the executor cannot know when this deployment claim becomes true.

- **Major, definite**: the plan overstates the current verifier failure mode.
  Section 7 says every unmeasured frame is silently judged compliant because `IPV=0` falls near the human center: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:296-301`.
  A direct read-only audit of the frozen RQ009 M3 test artifact shows this is too absolute. Using
  `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/04_calibration/predictions/tier=M3/fold=test/predictions.parquet`,
  filtered to supported rows (`abstain=false`), nominal `0.90`, and `|y|<1e-6`, I computed
  `520,826 / 522,219 = 99.7333%` intervals containing `0`, with `1,393` not containing `0`.
  This does not weaken the central concern that zero and unestimability are entangled; it does refute the universal "每个" phrasing and therefore requires narrower wording.

- **Major, definite**: "B1 不改变任何科学结论" is too absolute and is contradicted by the plan’s own later boundary language.
  The absolute claim appears at `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:115`.
  But §4.6 later states B1 can change observed IPV, M3 inputs, and final deviations on legacy-underflow rows: `reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md:238-241`.
  The scientifically defensible claim is narrower: B1 is mathematically equivalent on non-underflow rows and is intended not to reinterpret already-valid measurements. The current sentence overclaims.

- **Minor, definite**: the public `estimate_ipv_pair` docstring remains stale on warm-up semantics.
  It still says earlier rows are filled with `np.nan`: `src/sociality_estimation/core/ipv_estimation.py:213-215`, while the implementation still emits the legacy `0/1` placeholders: `src/sociality_estimation/core/ipv_estimation.py:251-252`.

### Bottom line

The plan is scientifically closer than v1, but it is **not yet actionable** for implementation/review handoff. Before re-review, it needs:

1. one authoritative B2 contract (`status` vs `flags`, especially `AT_GRID_BOUNDARY`);
2. a frozen rule for selecting and justifying `min_mse_misfit`;
3. a frozen coverage-audit protocol with explicit target artifact, split, nominal levels, tolerance/CI, and pass/fail rule;
4. tightened wording where the current text overstates universality or invariance (`每个测不出帧都合规`, `B1 不改变任何科学结论`, `D2 = 真实发现`).
