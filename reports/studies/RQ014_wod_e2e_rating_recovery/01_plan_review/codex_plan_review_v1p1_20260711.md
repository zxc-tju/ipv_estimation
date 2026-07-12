# RQ014 Plan v1.1 — Codex independent re-review

Review date: 2026-07-11  
Review type: adversarial statistics, registry/measurement, execution/HPC, and lead adjudication  
Reviewed amendment: `reports/plans/RQ014_plan_v1p1_amendment_20260710.md`  
Amendment SHA-256: `8bde5081d43ba0caca059395f1653e6b7554c6c43a80802ed8bcc59e3a560198`  
Valid registry SHA-256: `5aa69cbb0eb21fc3a48e00521136d3efa089148910e4e9c6a176e3b176bf3061`  
Forensic registry SHA-256: `e14bb777b887f90174e2fd84b55d2124a03dc82aeb39ad78a40bad3eb605b46a`  
Recovery-extension registry SHA-256: `16299cd4fb2cf9b28cc4dce221417250a19357fa03f64590d723e8fb19d3e9bb`  
FL05 script SHA-256: `d96f97ee8132008cf95f8b236452cfc9d15845e6e53b63033965237c479822b7`

## Verdict

**`BLOCKED_PENDING_MAJOR_REVISION`.**

v1.1 的修改方向有价值：增加独立 recovery-extension family 能区分“valid family 未找回”与
“旧设置可能落在 valid family 之外”；把“必须逐项胜过所有运动学变量”改成构造有效性、简约性、
增量性三层，也正确识别了运动学可能是 IPV→评分中介这一概念问题。

但是，当前 amendment 仍不能通过 G1。阻断原因不是 G0/G2 尚未实际运行这一正常状态，而是若干
**计划合同本身还不能唯一执行或不能支持其标签**：FL05 没有实现所注册的完整 CSV 索引；extension
的 X02–X05 和六项 maxT 未闭合；required-cell 仍可能循环定义；Tier C 把应检验不变性的
role/sign flip 当成必须击败的 placebo；Tier P 的 non-inferiority null、comparator、margin 和标签
不匹配。

因此：

- 不批准关闭 v1.1 independent re-review；
- 不批准 G1 PASS；
- 不批准 rating join、valid discovery、extension scan、forensic compute 或 HPC submission；
- 允许继续只读 G0 取证、修订 v1.2、以及 rating-blind G2 contract/manifest 准备。

## What improved and is retained

| Item | Re-review assessment |
|---|---|
| Base v1 remains immutable | PASS。base-v1 checksum 仍全部匹配。 |
| Three registries are separated | PASS。valid、forensic、extension 不共享 promotion/confirmation。 |
| Execution stays disabled | PASS。三个 registry 均为 proposed/blocked 且 `execution_authorized=false`。 |
| Base valid grid | PASS。12 unique configs × 3 readouts = 36 rows；V04 references exist。 |
| Recovery-coverage motivation | PASS IN PRINCIPLE。新增第三臂是合理治理修复。 |
| Mediation objection to old specificity | PASS IN PRINCIPLE。不能把“控制中介后无增量”自动写成 IPV 无效。 |
| Checksums | PASS for the five listed artifacts；范围仍需扩充。 |

## Blocking findings

### B1. FL05 does not implement the registered “all-statistics CSV”

Amendment lines 46–60 and forensic registry lines 48–57 require
`historical_stats_index.csv` with eight structured fields and an exhaustive recursive index. The supplied script
does something materially different:

- it emits sectioned console text, not CSV rows or a CSV header;
- `head -200/-400/-120/-80` silently truncates every important section;
- the parsers do not populate `n`, `unit`, `config_fields_as_recorded`, `mtime`, or a per-row source hash;
- the first statistics regex omits standalone `corr`, while the hash regex omits Pearson/Kendall;
- the `-0.[3-9]` heuristic misses `-1`, `-.35`, scientific notation and split-line JSON, and may match an unrelated number on the same line;
- errors are redirected away and the script uses only `set -u`.

Verified counterexample: running the script where both hard-coded input roots do not exist returns exit code `0`,
prints six empty section headings, and produces no CSV header. A missing source tree can therefore be mistaken for a
successful negative scan.

Minimum fix:

1. build an untruncated, NUL-safe file manifest with count/bytes and SHA;
2. fail closed if either root is missing or any parser stage fails (`set -euo pipefail` plus explicit audit);
3. parse CSV/JSON/Markdown by format into one row per statistic;
4. include `parse_status` and `raw_locator` so unparsed candidates are visible;
5. output the exact registered CSV schema, plus a completeness summary;
6. preflight tree size and move to a small `zxc-` CPU Slurm job if the frozen login-node budget is exceeded;
7. remove all `head` truncation from the canonical artifact.

Until this is fixed and run, B2/F10 are not closed.

### B2. The recovery-extension registry is not yet an executable six-cell family

The new family is conceptually useful, but four cells and the joint inference remain under-specified:

- **X02** changes source, form, conditioning and likely estimator scale at once. It lacks state variables, WOD mapping,
  tau rule, quantiles, weighting, scale-compatibility rule, builder hash and blind support gate. A legacy sigma01
  envelope cannot be compared to WOD-fixed candidate IPV merely by changing `envelope_source`.
- **X03** specifies only the median of three candidates. The primary normalized exceedance requires `L/M/U`; no
  lower/upper band exists. If the intended endpoint is distance from the candidate median, then both envelope and
  endpoint change and the row must be registered as a verifier-contract bundle, not a one-field source change.
- **X04/X05** name central 80/95 bands but do not freeze `q10/q50/q90` and `q025/q50/q975`, new envelope IDs,
  half-width gates, or reuse of the exact human sample/weight/tau contract. The valid registry currently hard-codes
  q05/q50/q95.
- **X06** uses pooled Spearman, whereas X01–X05 use `rho_WS`. A single unstudentized within-scene permutation maxT
  is not defined across these heterogeneous statistics. X06’s pooled association also contains between-scene
  structure that within-scene label permutation intentionally removes.
- X02 may be unavailable, yet the family is fixed at six cells with no frozen shrink/stop policy.
- The amendment says an extension signal is “exclusive”, but the YAML flag does not compare it with V04 or the
  other extension cells.

Minimum fix: fully enumerate X01–X05 as hashed verifier contracts; separate X06 into a pooled forensic diagnostic;
then freeze synchronized permutations, common/missing scene rules, studentization, lower-tail maxT, p-value formula
and unavailable-cell policy for the homogeneous family.

### B3. Required-envelope cells remain potentially circular

Amendment lines 69–74 and valid registry lines 115–120 define required cells from “primary-eligible” scenes, but
`primary-eligible` has no independent pre-envelope definition. In base v1, `N_feature_complete` means the candidate
endpoint exists; that endpoint itself depends on a valid envelope cell.

Replace the term with an explicit rating-blind state such as:

> `pre_envelope_trajectory_eligible`: exactly three candidates, frozen path type, solver-finite IPV, shared
> counterpart/common support and scheduled tau grid all pass, but no envelope has been queried and no rating has
> been read.

The required set must be the union of scheduled `(path_type,tau)` cells for those scenes. Only after that immutable
manifest is hashed may envelope gating determine config eligibility.

### B4. Tier C confuses placebo superiority with invariance

`shuffled_ipv` and a properly frozen counterpart swap break candidate correspondence and can serve as placebo
controls. `sign_flip` and often `role_flip` are different:

- with a near-zero, symmetric envelope, sign-flipped IPV can yield exactly the same deviation;
- role symmetry may be a desired property rather than a failed construction.

Requiring `S_IPV-S_control>0` for these cells makes a correct invariance look like failure. Tier C should either:

1. keep only genuine correspondence-breaking placebos in the superiority family; or
2. first prove rating-blind non-alias behaviour and separately freeze equivalence/invariance margins for sign/role
   diagnostics.

Tier C also lacks a universal common-scene manifest or an explicit master-ID synchronized bootstrap when controls
have different support.

### B5. Tier P’s non-inferiority calibration uses the wrong null

The amendment and registry propose checking Tier P size under a global null. But when
`S_IPV=S_K=0`, `A_P=S_IPV-S_K+delta_NI=0.10`; this is inside the non-inferiority alternative, not on the null
boundary. A correct size calibration must simulate
`S_IPV-S_K=-delta_NI`, including settings with `S_K>0` and the observed tie/missingness structure.

Tier C likewise needs equality-boundary simulations `S_IPV=S_control>0`, not only the all-zero global null. The
one-sided 95% lower-bound wording also implies nominal alpha 0.05, while G2 requires empirical size ≤0.03. The plan
must choose one coherent alpha/interval contract rather than relying on unexplained conservatism.

### B6. Tier P’s comparator, margin and claim do not align

The plan compares IPV with one scalar `kinematics_combined_cost`; this is not a fitted seven-feature model. The
registry lists six constituent metrics, while the prose repeatedly says “seven-feature battery”. The exact composite
formula and code SHA are not present locally.

Consequences:

- current PASS can support only “not inferior to the frozen RQ010B kinematics composite”, not “matches a
  seven-feature battery” or “one index expresses the comprehensive tendency”;
- `delta_NI=0.10` is justified by a discovery threshold, not by the maximum scientifically acceptable loss. Against
  existing kinematic effects around 0.16–0.26, it permits a large relative loss;
- no Tier-P-specific non-inferiority power requirement is frozen.

Either narrow the comparator and label, or build a genuinely frozen/cross-fitted multivariable kinematic comparator.
In both cases, justify one primary margin before ratings and simulate its boundary-null size and power.

### B7. Four valid configs still lack a canonical sequence execution rule

V01/V02/V07/V08 use `sequence_mode: trailing_only`, but `sequence_contract.local_window_estimator_rule` defines
only `local_span_future_only_s` and `local_span_history_future_s`. Add
`local_span_trailing_only_s: W` (and its splice rule), or map `trailing_only` explicitly to the W=1 canonical alias.
Without this, 4/12 valid configs cannot be produced uniquely from the registry.

### B8. Current execution readiness remains blocked by design

Forensic surfaces F05–F08 and F10 are still `OPEN`; amendment lines 146–152 correctly require zero OPEN surfaces.
G2 code/adapter/builder/input hashes also remain placeholders. These are not defects in amendment logic, but they are
hard launch gates. Current state must remain:

```text
G0 = OPEN
G1 = RE-REVIEW_BLOCKED
execution_authorized = false
```

## Major findings

1. Tier I controls only the frozen kinematics composite. The label must be
   `INCREMENTAL_BEYOND_FROZEN_KINEMATICS_COMPOSITE` unless a full kinematic model is actually controlled.
2. Tier I should run only after the preceding tiers under a frozen hierarchy; this ordering is implied by the label
   ladder but not stated as an execution rule.
3. Forensic FT01 and FT04 do not point to exact canonical field paths. FT01 must distinguish input grid dt from the
   internal estimator timing defect; FT04 currently changes both counterpart source and extrapolation contract.
4. The checksum manifest omits the FL05 script and future file manifest/parser/output. A gate-critical script can drift
   without invalidating the registered five hashes.
5. START_HERE and STUDIES still identify v1 rather than recording “v1.1 candidate amendment, review blocked”.
6. Historical workflow entries say G1/NOT_FOUND complete even though the current registry correctly exposes open
   surfaces. Add a superseding status entry; do not rewrite historical lines.
7. Amendment line 27 says extension enters no alpha family, while line 34 defines an arm-internal maxT. Correct wording:
   it does not share alpha with scientific/forensic families but has its own exploratory family.

## Claim–evidence–boundary audit

| Proposed claim | Current evidence contract | Ruling |
|---|---|---|
| Recovery-extension closes the old-spec coverage gap | Six proposed variants around V04 | Directionally supported, but cells/inference not executable |
| Tier C proves construction robustness | Four controls, all required to be weaker | Rejected as written; two controls are invariance diagnostics |
| IPV is a parsimonious index matching a kinematic battery | NI against one unrecovered scalar composite | Claim exceeds comparator and margin evidence |
| IPV adds information beyond kinematics | Nested model controls only one composite | Narrow to “beyond frozen composite” |
| FL05 exhaustively recovers historical statistics | Truncated grep text | Rejected; registered CSV is not produced |

## Required revision order

1. Replace FL05 with a fail-closed, exhaustive structured indexer; hash the script and outputs; run it and close F10.
2. Close or formally mark F05–F08; issue an explicit G0 closure matrix.
3. Rewrite extension rows as complete verifier contracts; move X06 out of the homogeneous maxT family.
4. Replace `primary-eligible` with pre-envelope structural eligibility and freeze the required-cell manifest algorithm.
5. Split Tier C into placebo-superiority and invariance/equivalence checks with synchronized support.
6. Rebuild Tier P around the correct boundary null, a defensible margin and an accurately named comparator; add NI power.
7. Narrow Tier I wording or strengthen its comparator; add the missing `trailing_only` execution rule.
8. Regenerate checksums, update START_HERE/STUDIES, and request a fresh independent re-review.

## Re-review acceptance checklist

- [ ] FL05 produces the registered untruncated CSV and fails if coverage is incomplete.
- [ ] F05–F08/F10 have terminal states and evidence hashes.
- [ ] X01–X05 are fully enumerable, scale-compatible verifier contracts.
- [ ] X06 is separately governed or has a statistically valid joint-family derivation.
- [ ] required cells use a non-circular pre-envelope eligibility definition.
- [ ] Tier C separates superiority from invariance and freezes common support.
- [ ] Tier P calibrates the NI boundary null, freezes alpha, margin, power and exact comparator.
- [ ] Tier P/I labels match the comparator actually used.
- [ ] `trailing_only` has an explicit local-window rule.
- [ ] all gate-critical scripts/manifests/outputs are checksum-bound.
- [ ] canonical indexes state v1.1’s current blocked status.

## Verification performed

- Parsed all three v1.1 YAML registries successfully.
- Verified 12/12 unique valid config IDs and tuples, 36 materialized valid readouts, 6/6 unique extension IDs,
  5 forensic lookup cells and 4 one-factor twins.
- Verified all five entries in `RQ014_plan_v1p1_checksums_20260710.sha256`.
- Verified `bash -n` for the FL05 script.
- Demonstrated the FL05 false-success case: missing input roots still return exit code 0 and no CSV.
- Confirmed `execution_authorized=false` in all three registries.
- Confirmed no RQ014 compute or rating join was launched during this review.

## Final ruling

v1.1 should not be discarded: B1’s recovery-extension arm and B5’s separation of association, parsimony and
incrementality are worthwhile. But the current documents are still a **design draft**, not an executable SAP. The
shortest path to PASS is to repair contracts and claims, not to add more configurations.

Until the checklist above is satisfied, the correct project status is
`BLOCKED_PENDING_MAJOR_REVISION / NO_COMPUTE_AUTHORIZED`.
