# RQ015A v3 Independent Review (Reviewer 3: readability, execution, governance)

Date: 2026-07-26  
Scope: `RQ015A_plan_v3_concentration_audit_20260726.md`, checksum manifest, ledger schema, run spec, contracts implementation, fixtures, sealed-exposure disclosure, `RQ007` knowledge README/decision, `configs/research_authorization.json`, and first-party repository paths needed to verify execution wiring.

## Review setup

I reviewed the v3 frozen baseline as a cross-disciplinary readability / execution / governance reviewer. I did not run any RQ015A data analysis, replay, HPC job, or held-out measurement inspection.

Mechanical checks performed:

- Manifest bytes matched the published checksum bundle for all 6 bound files (`reports/plans/RQ015A_plan_v3_checksums_20260726.sha256:1-6`).
- All five declared input roots exist (`reports/plans/RQ015A_run_spec_v1.json:26-31`).
- `scripts/rq015a/rq015a_contracts.py` imports, schema loading, and a synthetic `c0_route_with_sensitivity(...)` probe succeed (`scripts/rq015a/rq015a_contracts.py:165-300`).
- `py_compile` succeeds for the contracts module and fixture file when the bytecode cache is redirected to a writable temporary prefix.
- Direct fixture replay in the base shell fails: `python3 -m pytest -q tests/test_rq015a_contracts.py` returns `No module named pytest`, even though validate-only requires “fixtures 16/16 pass” (`reports/plans/RQ015A_run_spec_v1.json:38-43`; `tests/test_rq015a_contracts.py:9`).

## Shared claim

The v3 package is materially better than the earlier prose-only versions. It narrows the object to a concentration audit rather than an estimability claim, freezes a concrete ledger schema, and supplies deterministic fail-closed contract code. However, Formal G1 should remain blocked because the execution contract is still not self-sufficient and the governance authority surfaces are still internally inconsistent.

## Evidence and missing

Evidence now present:

- The scientific object is correctly narrowed to continuous `q_eff` concentration, with explicit prohibition on using report bins for downstream routing (`reports/plans/RQ015A_plan_v3_concentration_audit_20260726.md:20-35`).
- Artifact-specific expansion/collapse rules, split semantics, and non-applicable external-product handling are frozen in a machine-readable schema (`reports/plans/RQ015A_ledger_schema_v1.json:48-168`).
- The contracts implementation is deterministic and fail-closed for conservation, local-position logic, bins sensitivity, and C0 routing (`scripts/rq015a/rq015a_contracts.py:46-300`).

Evidence still missing from a reproducibility/governance standpoint:

- No exact runnable command or launcher is bound in the run spec, despite the plan claiming that the immutable run spec freezes the command (`reports/plans/RQ015A_plan_v3_concentration_audit_20260726.md:74-76`; `reports/plans/RQ015A_run_spec_v1.json:1-78`).
- The authorization anchor named by the run spec does not exist in the central authorization file (`reports/plans/RQ015A_run_spec_v1.json:7`; `configs/research_authorization.json:1-27`).
- The split allowlist source is referenced only symbolically as `RQ007 case_split_assignment.csv`, not as a checksum-bound repository path, even though this file is the gatekeeper for `held_out_parsed_rows = 0` (`reports/plans/RQ015A_run_spec_v1.json:33,57-64`; `reports/plans/RQ015A_plan_v3_checksums_20260726.sha256:1-6`).

## Overall

My overall judgement is `REQUEST_CHANGES`. The package is close in scientific framing, but not yet acceptable as a cross-disciplinary, execution-ready governance artifact. The remaining problems are not about the concentration construct itself; they are about whether another reviewer or operator can rerun the preflight, identify the exact authorized object, and reconcile the RQ007 authority story without relying on insider memory.

## Audience

For an insider who already knows RQ007/RQ009/RQ014, the v3 package is readable. For a broader methods, governance, or reproducibility audience, it is not yet self-contained enough. The main failure mode is not jargon density alone; it is that the package still requires implicit knowledge to resolve which split file is authoritative, which authorization object permits execution, and which RQ007 document should be trusted when the waiver and “untouched” language conflict.

## Strengths

- The construct boundary is much cleaner: “candidate-weight concentration” is no longer conflated with “estimability,” and the package explicitly bans stronger wording (`reports/plans/RQ015A_plan_v3_concentration_audit_20260726.md:80-85`).
- The schema is a real contract rather than prose. Expansion/collapse factors, role crosswalks, split applicability, and cross-artifact pooling prohibition are all explicit (`reports/plans/RQ015A_ledger_schema_v1.json:48-168`).
- The implementation is meaningfully executable as a contract library: conservation checks, local-position logic, deterministic averaging, bins-stability calculation, and C0 routing are all present and synthetically probeable (`scripts/rq015a/rq015a_contracts.py:84-300`).

## Findings

### Blocker

1. **Run spec is not yet a self-sufficient executable/governance contract.**

   Evidence: the plan says the immutable run spec freezes the exact command, environment, output root, receipt, and authorization object (`reports/plans/RQ015A_plan_v3_concentration_audit_20260726.md:74-78`). The run spec actually contains phases, checks, and produced filenames, but no runnable command, no entrypoint, and no launcher field anywhere in the document (`reports/plans/RQ015A_run_spec_v1.json:1-78`). In the same file, `authorization_object` points to `configs/research_authorization.json#rq015a_concentration_audit` (`reports/plans/RQ015A_run_spec_v1.json:7`), but the central authorization file contains only `INFRA` and `RQ014` entries (`configs/research_authorization.json:3-25`).

   Why this matters: an execution contract that cannot identify its own runnable entrypoint or its own valid authorization object is not reproducible and not governable. A later operator would have to reconstruct intent from prose, which is precisely what v3 claims to have eliminated.

   Fix: add an explicit execution stanza to the run spec with the exact command/argv or runner path for both `validate_only` and `execute`, and add the matching `rq015a_concentration_audit` authorization object to `configs/research_authorization.json` (or change the run spec to point to an existing authority).

2. **The RQ007 authority surfaces still contradict the sealed-exposure governance record.**

   Evidence: the disclosure requires `decision.md` and the knowledge README to add an append-only pointer and to stop claiming held-out is untouched (`reports/knowledge/RQ015A_ipv_estimability_labelling/sealed_exposure_disclosure_20260726.md:100-106`). The README now complies and explicitly says future confirmation must not be described as “pristine untouched” (`reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/README.md:6-12,40-50`). But the frozen `decision.md` still says “Held-out sealed,” “Sealed split untouched,” and “sealed (11,342) untouched” (`reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/decision.md:3,24,30`). Meanwhile the RQ015A plan says v3 must not modify any owning RQ `decision.md` (`reports/plans/RQ015A_plan_v3_concentration_audit_20260726.md:109-110`).

   Why this matters: the package currently has no coherent answer to the question “what is the authoritative RQ007 governance statement after the waiver?” A cross-disciplinary reader can legitimately read the README and `decision.md` and reach opposite conclusions.

   Fix: resolve the authority rule explicitly. Either:
   - append the waiver pointer to `decision.md` under explicit governance approval, or
   - amend the RQ015A/RQ007 governance text to state that `decision.md` remains frozen for claim content while the README is the authoritative append-only governance addendum after 2026-07-26.

### Major

1. **The split allowlist source is still not checksum-bound in the v3 execution contract.**

   Evidence: the run spec names the split source only as `RQ007 case_split_assignment.csv` (`reports/plans/RQ015A_run_spec_v1.json:33`). The checksum manifest binds only six files and does not include the actual split-assignment artifact (`reports/plans/RQ015A_plan_v3_checksums_20260726.sha256:1-6`). An independent repository lookup shows that the concrete file does exist at `data/derived/interhub/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_outputs/splits/case_split_assignment.csv`, but that path and hash are not bound inside the execution contract.

   Why this matters: `held_out_parsed_rows = 0` is only as strong as the exact whitelist file being used. If the split source is symbolic rather than bound, the most important exclusion control can drift without invalidating the v3 manifest.

   Fix: promote the exact split-assignment path and SHA-256 into `bound_artifacts` and the checksum bundle, or add a frozen split-freeze metadata file and bind that instead.

2. **The fixture/validate-only story is not reproducible from the declared environment.**

   Evidence: validate-only requires `fixtures 16/16 pass` (`reports/plans/RQ015A_run_spec_v1.json:38-43`), and the fixture file indeed contains 16 tests (`tests/test_rq015a_contracts.py:27-244`). But the same test file imports `pytest` directly (`tests/test_rq015a_contracts.py:9`), while the run spec advertises a pure-standard-library environment and lists only stdlib modules (`reports/plans/RQ015A_run_spec_v1.json:18-23`). In the current shell, `python3 -m pytest -q tests/test_rq015a_contracts.py` fails immediately with `No module named pytest`.

   Why this matters: the v3 package currently overstates how easy it is to reproduce preflight validation. A reviewer following the declared environment literally cannot rerun the mandatory fixtures without additional undocumented setup.

   Fix: either document `pytest` as a required validation dependency and give the exact validation command, or provide a stdlib-only self-test runner consistent with the “pure stdlib” claim.

3. **The executable aggregation helpers do not themselves enforce the schema’s anti-pooling contract.**

   Evidence: the schema says cross-artifact pooling is forbidden and that sigma01 is the only valid corpus-level source (`reports/plans/RQ015A_ledger_schema_v1.json:156-160`). The ledger primary key also includes `artifact_id` and `measurement_role` (`reports/plans/RQ015A_ledger_schema_v1.json:7-35`). But `aggregate_l2` groups only by `(case_id, perspective, configuration)` and `aggregate_l3` then collapses only by `case_id` (`scripts/rq015a/rq015a_contracts.py:165-213`). The fixture helper rows likewise omit `artifact_id` and `measurement_role`, so the test surface never checks that mixed duplicate-derived rows are rejected (`tests/test_rq015a_contracts.py:95-123`).

   Why this matters: the anti-pooling rule is presently enforced by prose/schema discipline, not by the “unique algorithm” surface. A caller can accidentally pass mixed derivative rows and still obtain a valid-looking L2/L3 summary.

   Fix: either make the aggregation helpers artifact-aware and fail closed on mixed-artifact inputs, or state explicitly in the API contract that these helpers only accept a single artifact/role slice and add tests that reject mixed inputs.

### Minor

- None beyond the issues above. The remaining readability limitations are secondary to the execution/governance blockers.

## Nature axes

| Axis | Verdict | Note |
|---|---|---|
| Scope discipline | PASS | The package now audits concentration rather than overclaiming estimability. |
| Mechanical contract design | PASS with caveat | Schema and contract functions are concrete, deterministic, and fail-closed where implemented. |
| Execution reproducibility | BLOCKED | No exact runnable command; authorization target missing; validate-only dependency story incomplete. |
| Governance coherence | BLOCKED | README and `decision.md` disagree after the waiver, and the current plan/disclosure surfaces do not resolve that conflict. |
| Cross-disciplinary readability | MAJOR CONCERN | A non-insider still needs unstated context to identify the authoritative split file and authority chain. |

## Risk / unsupported claims

The current package does **not** yet support the following claims:

- “The v3 run spec is a fully frozen executable contract.”
- “RQ015A preflight can be rerun from the declared standard-library environment.”
- “The RQ007 authority surfaces are now harmonized after the held-out exposure waiver.”
- “Cross-artifact pooling is mechanically impossible in the executable aggregation path.”

## Verdict / count

Verdict: `REQUEST_CHANGES`  
Blockers: 2  
Majors: 3  
Minors: 0

`formal_g1_eligible=false`  
`execution_authorized=false`
