# RQ014 plan v1.6 execution handoff — Lead Agent + Sub Agent staged recovery

日期：2026-07-12

状态：`EXECUTION_HANDOFF / NOT_AN_AUTHORIZATION_ARTIFACT / NOT_EMPIRICAL`

适用顺序：base v1 → v1.3 scientific hardening → v1.5 amendment →
`RQ014_recovery_lane_v2.json` → 本 v1.6 execution handoff。

## 0. 文档地位与当前结论

本文件把 RQ014 从当前状态推进到最终历史结果恢复所需的全部工作、Lead Agent/Sub Agent 分工、验收标准、
用户决策点和停止条件一次写清。它不修改 v1.5 的 960-cell 科学空间、评分隔离、rank 规则、Formal G1、
中央授权或最终 bundle，也不自行授权任何新 operation。

现有 v1.5 reviewed commit中的 artifact bytes与74-row bundle不得改写历史。versioned science JSON如需改变，必须
使用新版本路径；canonical `configs/research_authorization.json`只能在新分支/新commit中变更，并保留旧commit可寻址。
任何会改变以下机器权威的后续工作都必须形成新commit、新 candidate manifest、两路 fresh review、Formal G1和final bundle：

- `reports/plans/RQ014_recovery_lane_v2.json`；
- `reports/plans/RQ014_envelope_builder_contract_v2.json`；
- `reports/plans/RQ014_execution_contract_v1p5.json`；
- `configs/research_authorization.json`；
- managed launcher、run-spec schema、科学 entrypoint 或测试。

截至本文件冻结时：

- handoff编写时本地 `origin/main`快照=`802f05c599163a59c425e707262c97c66b39226f`；执行同步时必须
  fetch/attest当时最新的published `origin/main`，不得假定该快照仍是tip；
- exact reviewed contract commit=`24be08278adf43371fda14e7ec23a95b986b2fb1`，已是 `origin/main` 祖先；
- Tongji managed checkout 仍为 `b1476bd0b4e345c15ffdb9582bf7bf6bfa67fef0`，且尚无 v1.5 execution contract；
- Formal G1=`FORMAL_G1_PASS`；
- 当前中央 allowlist 只允许 `rq014_g2_declassification_export`；
- 尚未读取评分、创建 RQ014 production run root 或提交 RQ014 Slurm job；
- 当前不能直接运行 preflight、resource pilot、960-cell feature bank、评分 join 或 clean replay。

因此现在**可以执行，但只可以自动推进到评分盲 export 的 bounded report**。首次需要用户/PI 决策的节点是
export PASS evidence 完整形成之后，而不是 HPC 同步之前。

## 1. 不变研究问题、分析单位与通过定义

研究任务不是提出新假设，而是在 WOD-E2E rated479 的同一数据集上找回已观察到但设置遗失的负相关 recipe：
人类评分越高，轨迹偏离 human IPV envelope 的程度越低；评分越低，偏离越高。

- 分析单位：`segment × candidate trajectory`；聚类/重采样单位与三种 association 的权重由
  `RQ014_recovery_lane_v2.json` 冻结。
- 评分盲 predictor 空间：2 个频率 × 8 个 temporal recipes × 3 个 envelopes × 2 个 horizons ×
  10 个 deviation readouts = 960 cells。
- 评分访问后的 terminal screen：RWS、pooled Spearman、pooled Pearson，共 2,880 行；全部行必须 terminal
  后才能机械排序。
- 通过定义：完整screen的全部2,880行无条件按冻结typed comparator机械赋予ranks 1..2,880。只有至少一行同时为
  `OBSERVED`且`recovery_compatible=true`时，全体rank 1才允许生成`selected_recovery_recipe.json`并进入fresh
  clean replay。若无兼容行，仍发布完整rank table，但终止为`RECOVERY_INCONCLUSIVE_COVERAGE`或
  `NOT_RECOVERED_WITHIN_FROZEN_RECOVERY_GRID`，禁止生成selected recipe或运行clean replay。
- 结论边界：最多支持“同一 WOD-E2E 数据集上的历史关联结果已计算复现”；不等于新数据 confirmation、因果效应
  或外部泛化。

## 2. Lead Agent 与 Fleet 运行合同

未来执行必须由一个 Lead Agent 负责判断、集成和用户沟通；Sub Agent 只承担边界清晰的执行、核验或独立复现。
Lead 开始时创建：

```text
.codex-fleet/rq014-execution-v1p6/
  board/plan.md
  board/knowledge.md
  board/tried.md
  board/validation.md
  board/reports/
```

`board/plan.md` 逐 wave 记录 `PENDING / IN_PROGRESS / PASS / BLOCKED / USER_DECISION_REQUIRED`；
`knowledge.md` 记录新发现的路径、环境和反证；`tried.md` 记录失败与不可重试状态；`validation.md` 记录每次
独立 review/replication 的输入 SHA、reviewer identity 和 verdict。

每个 Sub Agent prompt 必须明确：目标、输入 SHA、只读/写入范围、禁止路径、预期 artifact、测试、停止条件；
每个 final report 必须包含 `TL;DR / Result / Artifacts / Verification / Confidence & caveats / Needs from lead`。
同一路径不得由两个写入 agent 并行修改。重要结果不能由原 executor 自审；统计与执行治理 reviewer 必须是不同身份。

Lead 可以自行推进已授权且不改变机器权威的步骤；不得以多数投票代替判断，不得把 Sub Agent 的“建议”当成授权。
low-level cell workers 永远看不到 partial leaderboard，也不得修改 axes、ranking、ledger、mask、weights 或 resampling。

## 3. 自动推进范围与用户沟通规则

### 3.1 当前无需用户批准即可推进

在所有 fail-closed gate 通过的前提下，Lead 可连续执行 Wave 0–3：

1. 只读核对本地/HPC/Git/环境/source hashes；
2. 将 managed checkout 同步到包含 exact reviewed commit 的已发布 Git 历史；
3. 创建 immutable v2 export run spec；
4. 运行 launcher validate-only；
5. 在独立 reviewer 给出无 blocker verdict 后，提交当前已中央授权的 declassification export；
6. 监控 Slurm、验证 receipts/bundle，并形成 bounded report。

网络瞬断、只读查询失败和可证明无状态变化的重试不需要用户决定。任何 reviewed byte、中央 allowlist、评分边界、
input universe、统计空间、resource profile 或不可重试 submission state 的变化都不属于自动修复。

### 3.2 必须提前通知用户

Lead 在每个 `D1–D6` 决策点到达前，应先发送简短预告；证据包完成后再提交正式决策请求。请求必须包含：

- 当前 gate 与 machine verdict；
- exact commit/spec/artifact SHA；
- 已通过和失败的检查；
- 推荐选项及理由；
- 其他可选项的影响；
- 若用户暂不回复，默认 `STOP_AND_PRESERVE`，不得自动扩大权限。

rating leakage、source/hash drift、`SUBMISSION_STATE_UNKNOWN`、review blocker、unexpected universe/duplicate/nonfinite、
任何 forbidden path/field 或无法证明 Slurm 唯一状态时，不等到正常决策点，立即通知用户并停止。

## 4. Wave 0 — 只读再确认与 managed checkout 同步

### 4.1 并行 Sub Agents

- **W0-A Git/HPC sync auditor（只读）**：确认本地 `origin/main` 包含 `24be0827`；记录 HPC HEAD、
  `refs/remotes/origin/main`、clean status、filesystem free space 和 maintenance lock 状态。
- **W0-B contract auditor（只读）**：验证 Formal G1、68-row review manifest、74-row final bundle、两位 reviewer identity、
  central allowlist 和 exact wrapper/launcher hashes。
- **W0-C source/runtime auditor（只读）**：复核 8 个 scene bundles、readiness TSV、counterpart CSV 的 path/size/SHA，
  v3 environment receipt、14,326-file stdlib closure、20-row native closure；不得解析 rating-bearing payload。

三者独立报告后，Lead 生成 W0 gate table。任何一项不一致均 `BLOCKED`。

### 4.2 Git 同步的固定实现

当前 HPC `refs/remotes/origin/main`仍停在旧提交，因此不能只checkout `24be0827`；launcher还要求run commit是
HPC remote-main祖先。Lead使用已验证的incremental Git bundle，不在HPC依赖公网pull。固定变量：

```text
SYNC_OLD=b1476bd0b4e345c15ffdb9582bf7bf6bfa67fef0
RUN_COMMIT=24be08278adf43371fda14e7ec23a95b986b2fb1
SYNC_NEW=<执行时本地 fetch 后 attest 的 origin/main 40-hex>
TEMP_REF=refs/rq014/bootstrap/origin-main-<SYNC_NEW前12位>
REMOTE_MAIN=refs/remotes/origin/main
LOCK=/share/home/u25310231/ZXC/sociality_estimation/manifests/runtime_maintenance.lock
```

固定算法：

1. 本地clean checkout先fetch；验证`SYNC_OLD`和`RUN_COMMIT`均为`SYNC_NEW`祖先；创建以
   `refs/remotes/origin/main`为唯一head、以`SYNC_OLD`为prerequisite的incremental bundle。
2. 本地`git bundle verify`与`git bundle list-heads`必须显示exactly one head：
   `SYNC_NEW refs/remotes/origin/main`；记录bundle size/SHA-256/head/prerequisite。
3. 传输到`/share/home/u25310231/ZXC/sociality_estimation/manifests/RQ014/bootstrap/`固定regular non-symlink
   文件；HPC重验size/SHA、`bundle verify`、唯一head和prerequisite。
4. 修改managed repo前，确认`LOCK`是fixed regular non-symlink file；clean shell打开同一路径为fd8，核对
   path/fd target与device-inode-mode identity，并执行非阻塞`flock -x -n 8`。获取失败即`BLOCKED`，不得循环抢锁。
   该独占锁覆盖fetch、临时ref、CAS、detached checkout和全部最终attestation；完成后才释放。wrapper/job继续使用
   同一锁的共享模式，因此同步不能与validate/submit/job并发。
5. 在禁用hooks/fsmonitor/filters的固定Git环境中，只fetch bundle head到全新`TEMP_REF`；若ref已存在即停止。
6. 验证`TEMP_REF=SYNC_NEW`、当前`REMOTE_MAIN=SYNC_OLD`、`SYNC_OLD`是`TEMP_REF`祖先、`RUN_COMMIT`是
   `TEMP_REF`祖先；任何不等均停止。
7. 仅用compare-and-swap执行`git update-ref REMOTE_MAIN SYNC_NEW SYNC_OLD`；CAS失败即停止，禁止force覆盖。
8. 禁hooks/filters后detached checkout exact `RUN_COMMIT`；删除`TEMP_REF`前先记录SHA，删除后再次核对remote-main。
9. 最终必须验证`HEAD=RUN_COMMIT`、`REMOTE_MAIN=SYNC_NEW`、ancestor gate成功、
   `git status --porcelain=v1 --untracked-files=all`完全为空、v1.5 contract/launcher存在、wrapper SHA为
   `d8036336c354b202f388c5fd9dbc05a80a1e8292a574cc4c47400d0a772d7a1d`，并重验launcher/preflight/materializer hashes。
10. attestation固定记录old/new/run refs、bundle size/SHA/head/prerequisite、lock path/fd identity、CAS结果、HEAD、
    full-clean stdout、wrapper/launcher hashes。bundle在export bounded report接受前不得删除。

不得把handoff编写时的`802f05c5`快照或执行时`SYNC_NEW`自动当作run commit。运行代码固定取正式审查过的
`24be...` commit tree blobs；较新的remote-main只承担published-ancestor证明。

### 4.3 W0 通过标准

- HPC managed HEAD 精确为 `24be...`，clean；
- HPC remote-main ref 包含该 commit；
- 68/74 manifests 和 Formal G1 全通过；
- 10 个 source files 与 v3 runtime closure 全匹配；
- rating payload 未打开；
- 未创建 production run root、未调用 `sbatch`。

## 5. Wave 1 — immutable export spec 与 validate-only

### 5.1 Run ID 与固定字段

使用：

```text
RQ014_0_score_stripped_export_<YYYYMMDDTHHMMSSZ>_1ee1e1d1
```

从 exact commit 的 `configs/run_specs/RQ014_g2_declassification_export.template.json` 生成 schema-v2 spec，固定：

- `git_commit=24be08278adf43371fda14e7ec23a95b986b2fb1`；
- Formal G1 SHA=`a24e69c240700840d908514580ba6f43cb6555d1de02d825121475bdeda88713`；
- contract bundle SHA=`1ee1e1d121b8d24cef7fdca93f05ddcccfcb3282b70727c606ce03c36984c933`；
- environment manifest SHA=`30de86f702101fbfc8065f6a0d7fd4378daf526d0e55c1197a6a0a147752877a`；
- template 中已冻结的 8+1+1 source hashes；
- 实际 UTC seconds 与 `rq014-g2-declassify-cpu-v1`。

spec 必须是 sorted-key compact canonical JSON、唯一 trailing newline、无 placeholder/duplicate key。固定 staging dir为
`/share/home/u25310231/ZXC/sociality_estimation/manifests/RQ014/run_spec_staging/`：目录必须与`run_specs/`同filesystem、
regular directory、non-symlink、mode `0700`。candidate固定名`<RUN_ID>.json.candidate`；若candidate或final已存在即停止。
写完后核对bytes/SHA并设mode `0444`。在4.2同一maintenance lock的独占模式下，以hard-link no-replace方式把candidate
一次发布为`run_specs/<RUN_ID>.json`（final存在时link必须失败），核对两路径inode/bytes/mode相同后删除candidate并同步目录
metadata。最终文件必须是direct-child、regular、non-symlink、read-only；发布后不得chmod、改写或替换。

### 5.2 Agent 分工

- **W1-A spec builder**：生成候选 bytes、SHA 和 field table，不运行 launcher。
- **W1-B spec adversarial reviewer**：从 schema/template/authority 独立重算，检查 placeholder、路径、hash、UTC、mode。
- **W1-C bootstrap reviewer**：从 exact `24be...` commit 重新读取完整 operator command 与 wrapper digest，禁止复制旧 shell history。
- **Lead**：只有三份报告均无 blocker，才原子发布 spec。

### 5.3 Validate-only

只使用 exact commit 的 `configs/run_specs/README.md` 与 v1.5 kickoff 中同一条 clean-environment fd8/fd9 bootstrap；
不直接运行 wrapper，不直接调用 `sbatch`，不拆分 hash check 与 execution。首次调用不带 `--submit`。

validate-only evidence 必须包含并匹配：operation、run ID、exact commit、Formal G1、bundle、10 个 source expectations、
v3 environment、stdlib/native closure、closed-snapshot file plan、固定 exporter、job name
`zxc-rq014-export-<specsha12>`、1 CPU/8 GiB/2 h、所有 threads=1、`--export=NIL` 双重计划、固定 output root、
`rating_access=FORBIDDEN`。它不得创建 run root、snapshot receipt、rendered sbatch 或 job ID。

W1-A 与 fresh W1-D validator 分别解析输出；Lead 对照 exact expected table。任何差异都停止，不得通过改 spec 绕过。

## 6. Wave 2 — 已授权 export 提交、监控与失败处置

只有 W1 validate-only bounded evidence 得到 Lead + fresh execution reviewer `NO_BLOCKER` 后，才对**同一 spec bytes**
运行同一 bootstrap 的 `--submit`。

- **W2-A sole submitter**：唯一允许执行 submit command 的 agent；提交一次后只报告原始 stdout/stderr、spec SHA、job ID。
- **W2-B monitor（只读）**：用 `squeue/sacct` 监控；job name 必须以 `zxc-` 开头，workdir/spec/commit 精确匹配。
- **W2-C receipt verifier（不提交）**：完成后验证 run root、snapshot receipt、rendered NIL sbatch、export receipt、DONE、
  output bundle 和 source/output hashes。

失败规则：

- 在 `sbatch` 开始前失败：launcher 必须清理 partial snapshot/run root并写 retryable failure；修复后需要新的 validate-only review；
- `sbatch` 已开始而状态不明：保留 namespace、原始响应和已知 job ID，标记 `SUBMISSION_STATE_UNKNOWN`，禁止盲重试，
  立即通知用户；
- job 明确 FAILED/CANCELLED/TIMEOUT/OOM：保留证据，不能覆盖同一 RUN_ID；先根因 review，再决定是否新 RUN_ID；
- job COMPLETED 但 receipt/DONE/validator 任一失败：整体 `BLOCKED`，不得手工补 receipt 宣称完成。

## 7. Wave 3 — Export 验收与 bounded report

Export PASS 必须同时满足：

- Slurm `COMPLETED/0:0`，job/workdir/commit/spec SHA 全匹配；
- exactly 479 universe、geometry available=476、structural attrition=3；任一计数不同均为冻结source/universe drift，
  立即`BLOCKED`并要求新amendment/review，不得用本次新receipt改写预期；
- fixed 7 CSV + 2 JSON bundle 完整；
- `file_manifest.json` 与 `sanitization_receipt.json` hashes 封存；
- duplicate/unexpected/nonfinite/forbidden-field/path scans 全为 0；
- 输出无 TFRecord/protobuf/pickle/symlink/parent absolute path；
- `rating_access=NONE`、`observed_statistics=NONE`；
- export receipt 与 `DONE.json` schema/hash/identity 全通过。

统计 reviewer 只检查 universe/attrition/schema 是否保留科学原语；执行 reviewer 独立检查 provenance、Slurm、hash chain、
failure semantics。Lead 形成不超过 100 行的 bounded report，写入新的
`reports/studies/RQ014_wod_e2e_rating_recovery/RQ014_1_declassification_export_<UTC>_<sha8>/`，并更新
`START_HERE.md`、`STUDIES.md`、knowledge README、`main_workflow.log`。报告结论只能是：

```text
PASS_RATING_BLIND_EXPORT_READY_FOR_PI_DECISION
BLOCKED
```

随后无条件停止，进入 D1。

## 8. D1–D6 用户/PI 决策门

| 决策 | 触发证据 | 用户需要决定 | 推荐默认 | 未回复时 |
|---|---|---|---|---|
| D1 Export acceptance | Wave 3 双审查 PASS、receipt/DONE/bounded report | 是否接受 export，并允许准备把 `rq014_g2_contract_preflight` 加入中央 allowlist及重新 Formal G1 | 接受，仅当所有扫描为 0 | STOP_AND_PRESERVE |
| D2 Preflight acceptance | 新授权审查完成、preflight job PASS、bounded report | 是否接受 preflight并授权 resource pilot/implementation closure | 仅在无 contract drift 时接受 | STOP_AND_PRESERVE |
| D3 Compute budget | resource pilot给出 CPU-hour、内存、失败率、预计总成本 | 是否批准完整 960-cell rating-blind feature build预算 | 选择能覆盖全部 cells 且保留重试余量的最小预算 | STOP_AND_PRESERVE |
| D4 Rating access | 960 cells 全 terminal、feature bank/ledger frozen、rating-blind reviewers PASS | 是否授权一次 G3R full-rated479 join和 2,880-row terminal screen | 只授权固定 entrypoint、一次 join、无 partial leaderboard | STOP_AND_PRESERVE |
| D5A Clean replay | 2,880行全terminal，至少一行`OBSERVED && recovery_compatible=true`，合法`selected_recovery_recipe.json`已冻结 | 是否授权fresh agent执行唯一recipe的G4R clean replay | rank/freeze无违规时授权 | STOP_AND_PRESERVE |
| D5B No compatible recipe | 2,880行全terminal但不存在合法selected recipe，终态为`RECOVERY_INCONCLUSIVE_COVERAGE`或`NOT_RECOVERED_WITHIN_FROZEN_RECOVERY_GRID` | 是否接受有界non-recovery终态并进入最终结论审查；G4R必须保持禁止 | 接受机器终态、不扩网格 | STOP_AND_PRESERVE |
| D6 Claim acceptance | clean replay PASS/mismatch，或D5B no-recipe终态；统计/执行/反证review完成 | 接受/拒绝/限定RQ014结论，是否创建`decision.md`并允许paper handoff | 仅接受同数据集历史恢复或有界未恢复表述 | 不创建decision.md、不改论文 |

D1 是预计第一次需要用户做出的决定。Lead 必须在 Wave 2 job 接近完成或 Wave 3 evidence assembling 时提前预告，
不能在报告完成后才首次通知。

### 8.1 每个新 operation 的强制授权闭环

用户在D1/D2/D3/D4/D5A选择“继续”只表达PI意图，不等于machine authorization。对preflight、resource pilot、
G2R feature build、G3R rating join和conditional G4R clean replay中的**每一个**operation，Lead都必须完整执行：

1. 创建versioned scoped PI/user decision artifact，只命名一个目标operation；
2. 在新branch/new commit中只向canonical central allowlist新增该operation，保留旧commit可寻址；
3. 将该operation的entrypoint、tests、schema、contract、decision、authority bytes纳入同一个新candidate manifest；
4. 派发不同身份且均非原executor的fresh statistics reviewer与fresh execution/governance reviewer；
5. 两者对同一manifest均`NO_BLOCKER`且无major后，生成新Formal G1和final bundle；
6. 发布exact commit，安全同步managed checkout，生成immutable spec并先validate-only；
7. fresh validator确认无差异后，才允许single submit。

任一环缺失时中央机器结论仍为`DENY`。预算获批不能替代G2R授权；评分访问获批不能替代G3R授权；存在selected recipe
也不能替代G4R授权。

## 9. Wave 4 — D1 后的 central authorization 与 contract preflight

D1 明确接受后才开始：

1. 创建新的 scoped PI/user decision artifact；
2. 只把 `rq014_g2_contract_preflight` 加入中央 allowlist，不顺带开放 resource pilot/G2R/G3R/G4R；
3. 因 authority byte 改变，重建 candidate review manifest；
4. fresh statistics reviewer 与 fresh execution/governance reviewer读取同一 manifest；
5. 两者身份不同且均 `NO_BLOCKER` 后，Lead 生成新的 Formal G1 与 final bundle；
6. 从 preflight template 填入 exact export receipt、DONE、sanitization receipt、G2 input manifest和 materialization ledger；
7. immutable spec validate-only PASS 后，才提交 preflight；
8. preflight只验证/物化 rating-blind inputs、registry bindings、golden fixtures和资源前件，不运行 960 cells。

preflight PASS report 完成后停止进入 D2。任何为了通过 preflight而修改 v1.5 science axes、rating boundary 或旧 reviewed
registry bytes 的做法都必须另立 amendment，不能作为“修路径”处理。

## 10. Wave 5 — D2 后的 implementation closure 与 resource pilot

当前 repository 只实现 export/preflight边界；resource pilot、960-cell G2R、G3R rating join和 G4R clean replay并未全部形成
可执行且中央授权的 entrypoints。这是计划内未完成工作，不是隐藏的 runtime bug。

D2 后由不同 agents完成：

- **science implementer**：严格从 recovery/envelope JSON contracts实现 window-local derivatives、16 feature families、3 envelopes、
  2 horizons、10 readouts和 support/terminal rules；不得读取评分；
- **test engineer**：golden fixtures、4/10 Hz、t*=0 seam、no derivative halo、HFEAS/TF finite、path-type scene mapping、
  bootstrap occurrence identity、ledger chain和失败终态测试；
- **execution engineer**：closed snapshot、immutable spec、resource profile、resume/idempotency、per-cell atomic outputs；
- **statistics reviewer**：独立核对 estimand、mask/weights/denominators、BM90/BT90、BL90、rank/freeze；
- **execution reviewer**：独立核对无评分访问、无 partial leaderboard、HPC failure/retry语义。

代码通过PR合并后，D2仍必须按8.1为`rq014_g2_resource_pilot`完成独立scoped decision、central allowlist、双审、
Formal G1、bundle、immutable spec和validate-only闭环，之后才可提交小规模rating-blind pilot。pilot须覆盖最昂贵和最轻的
代表cells，报告walltime、CPU、peak RSS、I/O、失败率和全量估算，不产生评分统计。D3决定预算和是否启动下一operation授权闭环。

## 11. Wave 6 — D3 后的 960-cell rating-blind feature bank

D3批准预算后，仍须按8.1为冻结名称的G2R feature-build operation完成完整授权闭环；中央allowlist只新增该operation。
Lead再冻结batch map、cell IDs、expected outputs、retry policy、aggregation schema和成本上限；之后低级workers可并行执行
互不重叠的cell batches。

硬规则：

- workers 只读 score-stripped WOD bundle、frozen InterHub source和 frozen contracts；
- ratings、rating-derived split、rho、partial leaderboard完全不可见；
- 每个 cell输出 predictor、coverage/support、health metrics、receipt和 terminal status；
- retry沿用同 cell identity但新 attempt identity，不能覆盖成功 artifact；
- Lead 只在全部 960 cells terminal后聚合；失败/不可估计也是 terminal row，不能静默删除；
- numerical health、coverage、duplicates、constants、impossible values、sparsity、context imbalance和参数敏感性必须审计；
- feature bank、cell ledger、code/spec/environment/input hashes一次冻结后不可变。

两个 fresh rating-blind reviewers确认完整性后进入 D4。此时仍不能读取评分。

## 12. Wave 7 — D4 后的一次性 G3R rating join 与 2,880-row screen

D4必须明确接受固定rating manifest、固定entrypoint和一次full-rated479 join，并按8.1为G3R operation完成完整授权闭环。
实现/执行必须与G2R agents隔离，且只能是一个固定entrypoint的一次job：

- join agent只能读取 frozen feature bank与明确授权的 rating table；
- key/cardinality/tie/completeness检查先于统计；
- 三种association对960 cells一次性运行；2,880行先写入job-private mode-0700临时目录中的hash-chain ledger；
- stdout/stderr只能输出无数值的完成计数、health状态和failure code；禁止输出association、方向、rank、p值或cell metric；
- 临时ledger路径不得返回给Lead/workers/user。只有terminal validator证明row_count=2,880、全行terminal、hash chain和
  schema/completeness通过后，才以一次atomic publication同时发布terminal ledger、rank table和selection/no-selection artifact；
- job失败时不得发布或展示partial metric bytes；删除未发布的metric temp files，只保留无统计值的failure receipt、attempt identity
  和partial-chain digest。execution tests必须证明日志和failure artifacts不含metric/rank/direction；
- 不以 p value、power或符号提前停止；
- 2,880行全部terminal后，无条件对全部2,880个unique leaderboard rows按`RQ014_recovery_lane_v2.json`冻结的
  typed comparator机械赋予唯一ranks 1..2,880；`recovery_compatible_rank`是排序键，不是筛选条件，禁止对子集重排；
- 若至少一行`OBSERVED && recovery_compatible=true`，则按冻结排序键全体rank 1必为兼容行；只冻结该全体rank 1为
  `selected_recovery_recipe.json`，并绑定完整参数、input/code/spec/environment hashes和selection ledger；
- 若不存在兼容行，仍原样发布完整1..2,880 rank table，但禁止创建selected recipe，冻结`RECOVERY_INCONCLUSIVE_COVERAGE`或
  `NOT_RECOVERED_WITHIN_FROZEN_RECOVERY_GRID`及no-selection receipt，进入D5B；
- fresh reviewers只能在上述atomic publication成功后读取terminal metrics。

fresh statistics reviewer复算completeness/compatibility/ranking或no-selection终态；execution reviewer检查rating access、
non-disclosure和ledger atomicity。存在合法selected recipe时进入D5A；否则进入D5B并保持G4R=`DENY`。

## 13. Wave 8 — D5A 后的 G4R clean replay

D5A明确接受后仍须按8.1为G4R operation完成完整授权闭环。只有合法`selected_recovery_recipe.json`存在时，才由未参与
G2R/G3R实现的fresh agent在独立、commit-addressed、closed-snapshot执行面中，只读取frozen rank-1 recipe与机器合同，
独立重写 resampling、window、envelope、deviation、join和 statistic。不得 import/reuse原 screen实现的计算函数；只允许共享
schema和 frozen inputs。

D5B no-recipe终态禁止进入本wave；Lead直接组织最终有界non-recovery审查并进入D6，不得为了获得recipe扩大网格。

Clean replay必须验证：单位/样本数、coverage/support、deviation分布、association方向/数值容差、bootstrap/uncertainty、
negative controls和 boundary cases。原结果与 replay不一致时结论为 `BLOCKED_REPLICATION_MISMATCH`，不得挑选更好的一次。

Lead用 Nature-style evidence-first结构生成最终 Markdown + HTML report；每个结论有独立 evidence row和 figure/evidence bundle，
明确 positive evidence、boundary、uncertainty、robustness/negative checks。完成多轮 no-blocker review后进入 D6。

## 14. Git、报告与论文边界

- 每个 wave从最新 `main` 建 `codex/rq014-execution-<wave>`，通过 PR发布；不得在 HPC dirty checkout直接开发。
- HPC只执行已发布 exact commit的 registered blobs；每个新 operation均需 immutable spec和 managed launcher。
- 每个执行 wave在 `reports/studies/RQ014_wod_e2e_rating_recovery/` 建新 execution report目录；解释/接受结论只写
  `reports/knowledge/RQ014_wod_e2e_rating_recovery/`。
- 每次 workflow更新 `main_workflow.log`；当前 operating fact改变时同步更新 `START_HERE.md`和 `STUDIES.md`。
- 未经 D6 不创建/修改 `decision.md`；研究仓库与 paper repository不得在同一 PR修改。
- 只有 D6 接受的 claims才可交给论文仓库；本研究计划本身不是 claim evidence。

## 15. 全局立即停止条件

以下任一发生立即停止、保全现场并通知用户：

- Formal G1/review/bundle/authority/hash drift；
- managed HEAD不等于 spec commit，或 run commit不是 HPC remote-main祖先；
- managed Git同步未持同一maintenance lock的独占fd8，bundle head/prerequisite/temp ref/CAS任一不符，或full clean包含
  tracked/untracked entry；
- production spec非 canonical/direct-child/read-only/regular/non-symlink；
- wrapper fd8/fd9、runtime closure、`--export=NIL`或 thread/resource contract漂移；
- direct wrapper、direct sbatch、retired checkout、unmanaged code或 forbidden root；
- source size/hash改变、symlink/special file、rating semantic、raw TFRecord/protobuf/pickle进入 G2；
- universe/cardinality/duplicate/nonfinite/unexpected field/forbidden field不符合冻结合同；
- A1–A4 在 G2被重算，或 partial leaderboard被读取；
- 任一新operation缺scoped decision、central allowlist、fresh双审、Formal G1、final bundle、immutable spec或validate-only；
- G3R stdout/stderr/failure artifact泄露metric/direction/rank，或2,880行未全terminal即发布ledger；
- 无合法`selected_recovery_recipe.json`却尝试G4R，或为了制造selected recipe扩大/重排冻结网格；
- 两个 reviewer identity相同、任一 reviewer给出 blocker/major；
- `SUBMISSION_STATE_UNKNOWN` 或无法证明唯一 job identity；
- 任何 agent试图为了得到预期负相关而增加 cell、改变 rank规则、删除 terminal failures或提前查看评分结果。

## 16. Lead Agent 启动指令

Lead 接手后依次执行：

1. 读 `AGENTS.md`、`START_HERE.md`、Tongji HPC shared guide、本文件和全部 v1.5 machine authorities；
2. 创建 fleet board并登记当前事实，不沿用旧 agent memory作为证据；
3. 并行派发 W0-A/W0-B/W0-C只读 audits；
4. 汇总无 blocker后完成 incremental bundle同步；
5. 派发 W1 spec builder/reviewer/bootstrap reviewer；
6. validate-only双检；
7. 已授权 export单次提交、监控、receipt双检；
8. bounded report双审查；
9. 在 D1前提前通知用户并停止。

Lead 不得询问“是否先做只读检查”；这些已在本计划授权范围内。Lead 也不得在 D1之前请求评分或扩大 allowlist。

## 17. 本文件的成功标准

本 handoff成功不是完成实验，而是让后续 Lead Agent无需自行决定执行顺序、角色、验收、失败策略或用户停点。当前立即目标是：

```text
HPC_SYNC_PASS
→ IMMUTABLE_EXPORT_SPEC_PASS
→ VALIDATE_ONLY_PASS
→ AUTHORIZED_EXPORT_COMPLETED
→ EXPORT_BOUNDED_REPORT_PASS
→ USER_DECISION_D1
```

任何后续经验结论仍为未运行。
