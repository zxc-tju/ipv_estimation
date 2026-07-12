# RQ014 Plan v1.2 — Codex independent re-review

Review date: 2026-07-11  
Review type: adversarial statistics, registry/measurement, forensic execution, and lead adjudication  
Reviewed amendment: `reports/plans/RQ014_plan_v1p2_amendment_20260711.md`  
Amendment SHA-256: `2a2ac758f1d7c83d634584e8c6195775a11bbd6167d025a41e11bd0f05c5408f`  
Valid registry SHA-256: `3ea525e614a1f832166f50fc9d2ab8c2af9a0a47d8a7a3478e878ce8837377eb`  
Forensic registry SHA-256: `71eefbf506a5bb105ca7d250ad72f9a0330fbcaeb4d17fe7ef087745817fcd23`  
Recovery-extension registry SHA-256: `1cbd25b4456170ad9432e65254b898840970aa28fc2e32737a4eb583a9463b62`  
FL05 indexer SHA-256: `095ac326eea0534041b67dccb291c888ac01d559071e4b12adc1151a49f2505b`

## Verdict

**`BLOCKED_PENDING_MAJOR_REVISION / NO_COMPUTE_AUTHORIZED`.**

v1.2 对 v1.1 的主要概念问题作出了实质修复。required-cell 已改为评分盲、包络前的结构资格；
`trailing_only`、FT01/FT04、X06 的归属都已闭合；Tier C 已正确拆成 correspondence-breaking
placebo 与 invariance diagnostics；Tier P/I 的比较器和标签也明显收窄。保留这些修改。

但 G1 仍不能通过。当前剩余问题集中在两处：

1. **G0 取证闭环不能可靠证明“完整扫描”。** FL05 会漏掉常见长表 CSV/JSON，Markdown 会取错数字，
   审计也不是逐文件；F05–F08 的 closure scripts 在输入面缺失时仍可退出 0，并且大量截断输出。
2. **重构后的统计 family 还不能唯一执行。** extension 的 studentized maxT、异支持同步置换和单格
   adjusted p 未定义，却用一个 omnibus p 给具体 cell 发 flag；`p<0.05` 又与全案单侧
   `alpha=0.025` 直接冲突。Tier P/C 的边界校准和 NI power 仍只有目标效应，没有可复现的联合 DGP。

因此，本轮不批准 G1 PASS，不批准 rating join、valid discovery、extension scan、forensic recompute
或 HPC submission。允许的下一步仅限修订 v1.3、只读 G0 取证脚本修复，以及评分盲的 G2
contract/manifest 准备。

## What v1.2 fixed successfully

| Item | Ruling |
|---|---|
| Base v1 remains immutable | PASS；base-v1 checksum 仍匹配。 |
| Registry separation and launch guard | PASS；三个 registry 均保持 `execution_authorized=false`。 |
| Valid grid | PASS；12 个 unique configs，3 个 readouts，共 36 行。 |
| Non-circular required cells | PASS；由 `pre_envelope_trajectory_eligible` 场景物化，明确不查 envelope、不读 rating。 |
| W=1 alias / `trailing_only` | PASS；V01/V02/V07/V08 已有唯一 local-window 执行规则。 |
| Tier C conceptual split | PASS IN PRINCIPLE；shuffle/swap 与 sign/role invariance 已分开。 |
| Tier P/I claim scope | PASS IN PRINCIPLE；比较器和标签已收窄到 frozen kinematics composite。 |
| Forensic twin fields | PASS；FT01 指 estimator-internal dt，FT04 明示历史双字段 bundle 及 purity waiver。 |
| Extension membership | PASS IN PRINCIPLE；X06 已移至 FD01，X01/X03/X04/X05 单格基本可枚举。 |
| Static artifact integrity | PASS；v1.2 checksum manifest 8/8 匹配，三个 YAML 可解析，三个 shell 脚本语法通过。 |

## Blocking findings

### B1. FL05 is neither exhaustive nor value-safe

Indexer lines 60–113 only support a subset of the registered formats:

- CSV is parsed only when a **column header** contains `rho/correlation`. A normal long table such as
  `statistic_name,value,n / spearman_rho,-0.42,75` yields no row.
- JSON is emitted only when the statistic itself is a numeric-valued key. A record such as
  `{"statistic_name":"spearman_rho","value":-0.42,"n":75}` is missed.
- malformed CSV rows are silently skipped; a non-numeric value such as `NA` is labeled `PARSED_CSV`.
- Markdown takes the first number anywhere on the line. `At 4 Hz, Spearman rho=-0.42 (n=75)` is indexed as
  `value=4`, not `-0.42`.

The mock review reproduced all four cases. This violates the amendment's “all recorded correlation statistics”
and `UNPARSED_CANDIDATE` visibility contract. It also makes the downstream rule
“any recorded rho <= -0.30” unsafe.

Minimum fix: support both wide and long CSV/JSON; validate numeric values and bounds; parse Markdown from a
statistic-local `key=value` expression; emit every malformed candidate explicitly; add fixture tests covering all
accepted schemas and false-positive numbers.

### B2. FL05 is not an atomic, per-file-auditable fail-closed artifact

The indexer writes its header and rows directly to canonical STDOUT before the final completeness check. A failure
therefore leaves a partial file at the registered output path. Its audit reports only total file count/bytes and
aggregate statistic rows; it does not provide the required per-file path/bytes/SHA/mtime/parser-status/row-count
manifest, so files with zero extracted rows are invisible.

The >200 MB path only aborts. No `zxc-` sbatch wrapper, compute-node override, output contract, or job evidence is
registered; setting `SLURM_JOB_ID` would still hit the same limit. Thus “rerun as sbatch” is currently an instruction
without an executable path.

Minimum fix: parse to a temporary spool, validate completeness, then atomically publish; emit a full per-file audit
including zero-hit files; provide and checksum a real CPU Slurm wrapper whose job name starts with `zxc-` and whose
compute-node budget is explicit.

### B3. F05–F08 closure scripts can turn missing surfaces into a negative scan

`RQ014_forensics_hpc_pass3_20260710.sh` and
`RQ014_forensics_mac_pass3b_20260710.sh` use only `set -u`. Missing directories, failed reads and failed greps are
suppressed. The HPC script returned exit 0 locally with all hard-coded `/share/...` paths absent. Both scripts also
truncate canonical evidence (`head -20`, `head -60`, first 2,500/4,000/5,000 bytes, first 20 Cowork hits).
The Cowork step lists matching filenames rather than archiving the matched fulltext needed to adjudicate F08.

These scripts may be useful discovery probes, but they cannot legally close F05/F06/F07/F08 as
`NOT_FOUND_ON_SCANNED_SURFACES`.

Minimum fix: one fail-closed status record per surface; explicit required-root checks; untruncated file manifests and
hashes; separate `FOUND`, `NOT_FOUND_ON_SCANNED_SURFACES`, and `INACCESSIBLE`; no terminal state when a read fails.

### B4. Extension maxT and cell-level flag are not a defined procedure

Extension registry lines 88–108 name a studentized lower-tail maxT but do not define `T_obs,k`, `T_bk`, the SE
estimator, whether SE is re-estimated, the null centering, or zero-SE/undefined policies. Cells use per-cell scene
manifests, while “same master seed” does not guarantee that an overlapping `scene_id` receives the same within-scene
rating permutation in every cell.

More importantly, the registry defines only one omnibus `minT` p-value and then reuses it to flag an individual
cell. A signal in X01 could make the omnibus p significant while a raw-threshold-passing X04 receives
`SPEC_RECOVERY_CANDIDATE` without cell-specific adjusted significance.

Minimum fix:

1. freeze a union master-ID manifest and a deterministic `(replicate, scene_id)` keyed permutation;
2. define the exact studentized statistics and numerical-failure policies;
3. emit single-step maxT-adjusted `p_k` for every cell (or restrict the flag to the registered argmin cell);
4. if “exclusive versus V04” is intended, add a paired cell-vs-V04 contrast on the exact same scene set.

### B5. Extension alpha is internally inconsistent

Amendment lines 91–94 state that **all** one-sided tests use nominal alpha 0.025. Amendment line 59 and extension
registry lines 105–108 use `within_arm_maxT_p < 0.05`. The extension lower-tail maxT is a one-sided test. An
implementer can therefore choose two different legal thresholds.

Use 0.025 consistently, or explicitly declare and justify an extension-family exception while removing the claim
of a whole-plan unified alpha contract.

### B6. Tier C/P calibration and Tier-P power lack a reproducible joint DGP

The registry now names the right boundary targets—`S_IPV-S_K=-delta_NI` and
`S_IPV=S_placebo>0`—but does not say how one set of simulated ratings jointly realizes IPV, K and placebo
strengths while preserving their observed dependence. It also lacks per-grid replicate/seed rules and a frozen
worst-case-versus-average pass rule across `S_K × tie` cells. The NI power item similarly states only
`S_IPV=S_K` and probability >=0.80, without a generating model.

Minimum fix: freeze one joint copula/latent-rank or resampling DGP, calibrate every boundary cell, state replicate
counts/seeds and require every registered boundary cell (not the average) to meet the size/FWER/power threshold.

### B7. X02 still has an unresolved scale eligibility decision

X02 refers to legacy PET bins and a future G2 WOD mapping, but does not freeze bin edges, mapping semantics, the
coverage denominator, or the “12-case parity set”. Scale compatibility is only recorded as a caveat; there is no
metric or threshold that makes an incompatible static sigma01 envelope ineligible. Such a cell can still enter maxT
and trigger a recovery flag.

Minimum fix: freeze the source bins, WOD mapping, parity sample, metric, threshold and denominator before rating
access; route failures to an explicit `INELIGIBLE_SCALE_INCOMPATIBLE` state or move X02 to forensic diagnostics.

## Major findings

1. **The invariance “equivalence test” is a point-estimate rule.** `|S_flip-S_IPV|<=0.10` is not evidence of
   equivalence. Define paired TOST/simultaneous CI, alpha, multiplicity and common support, or make both diagnostics
   fully descriptive and unable to cap labels.
2. **Tier P and Tier I lack frozen paired scene manifests.** Recompute all compared statistics on the exact
   comparator-complete intersection and hash separate P/I manifests, or freeze one universal confirmation
   intersection.
3. **Tier I has no boundary calibration.** The highest label needs type-I calibration at `beta_D=0` under observed
   Q–K collinearity, plus singular/near-collinear and undefined-bootstrap policies.
4. **The historical fingerprint predicate is too broad.** A negative rho elsewhere in the two large RQ010B trees is
   not automatically a rating↔IPV-deviation fingerprint. Require variable pair, direction, unit, N, WOD-E2E
   candidate scope and source/config fingerprint; otherwise label it
   `UNATTRIBUTED_NEGATIVE_CORRELATION_CANDIDATE`.
5. **The NI margin citation is not anchored precisely.** `delta_NI=0.08` may be retained as a normative margin, but
   “half the weakest published physics effect rho≈0.16” is not a stable description of all registered RQ010B
   controls/rates. Cite one exact frozen row and explain why it is the scientific loss benchmark, or describe 0.08
   explicitly as a PI-chosen tolerance and include sensitivity results without changing the primary margin.

## Minor finding

Amendment lines 132–133 say “FL05 indexer plus three pass3/pass3b scripts”, while the checksum contains FL05,
HPC pass3 and Mac pass3b—three gate scripts total, only two of which are named pass3/pass3b. Change the prose to
“three gate-critical scripts (FL05, HPC pass3, Mac pass3b)” or add the missing third pass script.

## Claim–evidence–boundary audit

| Proposed claim/status | Current evidence contract | Ruling |
|---|---|---|
| FL05 exhaustively indexes historical statistics | Partial format parser; aggregate audit | Rejected as exhaustive |
| F05–F08 are legally closable by pass3/pass3b | Fail-open and truncated probes | Rejected as closure evidence |
| One extension cell is an exclusive recovery candidate | Omnibus p reused at cell level | Not identified |
| Tier C proves placebo superiority | Concept correct; bootstrap family mostly inherited | Retain after exact DGP/calibration closure |
| Tier P supports non-inferiority to frozen composite | Correctly narrowed comparator/label | Retain after comparator hash, common set, DGP and power closure |
| Tier I supports incrementality beyond frozen composite | Correctly narrowed label | Retain after null calibration and collinearity policy |

## Required revision order

1. Replace FL05 with fixture-tested, atomic, per-file-audited parsing and add a real `zxc-` CPU wrapper.
2. Replace F05–F08 discovery probes with fail-closed surface-specific closure procedures.
3. Freeze extension union-scene synchronization, exact studentization, adjusted `p_k`, V04 contrast and alpha.
4. Freeze the Tier C/P joint boundary DGP, Tier-P NI power DGP and worst-case pass rules.
5. Close X02 mapping/scale eligibility; finish invariance, P/I common-set and Tier-I calibration contracts.
6. Regenerate checksums, update v1.3 status, and request one targeted independent re-review.
7. Only after G1 passes, execute G0; only after G0 and G2 close may any rating join or compute be authorized.

## Re-review acceptance checklist

- [ ] FL05 long/wide CSV and JSON plus Markdown fixtures all parse correctly; malformed values are visible.
- [ ] FL05 publishes atomically and audits every scanned file, including zero-hit files.
- [ ] A checksum-bound `zxc-` CPU wrapper can execute the over-budget path.
- [ ] F05–F08 cannot return a negative terminal state when a required surface/read is missing.
- [ ] Extension maxT has exact statistics, scene-ID synchronization and cell-adjusted p-values.
- [ ] Extension uses one unambiguous alpha threshold.
- [ ] Tier C/P boundary size and Tier-P NI power use a frozen joint DGP and worst-case gate.
- [ ] X02 has a frozen mapping and quantitative scale-compatibility gate.
- [ ] Invariance, Tier P and Tier I have frozen common-set and inferential contracts.
- [ ] Tier I is calibrated at its null with collinearity/numerical policies.
- [ ] Historical fingerprint classification is variable- and scope-specific.
- [ ] All gate-critical scripts and regenerated artifacts are checksum-bound.

## Verification performed

- Parsed all three v1.2 YAML registries.
- Verified 12/12 unique valid config IDs and tuples, 36 materialized readouts, 5 extension cells, 5 forensic lookup
  cells and 4 one-factor twins.
- Verified all eight entries in `RQ014_plan_v1p2_checksums_20260711.sha256`.
- Ran `bash -n` on FL05, HPC pass3 and Mac pass3b.
- Confirmed all three registries keep `execution_authorized=false` and G0 surfaces F05–F08/F10 remain OPEN.
- Ran local parser mocks demonstrating long-form CSV/JSON omission, non-numeric CSV acceptance, malformed-row loss
  and Markdown first-number misparsing.
- Confirmed missing FL05 roots exit nonzero, but confirmed HPC pass3 exits 0 with all hard-coded roots absent.
- Confirmed no RQ014 ratings were joined, no scientific/forensic computation was run, and no HPC job was submitted
  during this review.

## Final ruling

v1.2 is a meaningful improvement and should be used as the base for a **targeted v1.3**, not discarded. The
remaining work is contract hardening, not expansion of the search grid. Until the checklist passes, the correct
status remains `BLOCKED_PENDING_MAJOR_REVISION / NO_COMPUTE_AUTHORIZED`.
