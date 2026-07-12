# RQ014 v1.5 formal G1 round-5 remediation

日期：2026-07-12  
状态：第五轮整改已闭合，等待第六轮 fresh formal review；本文不是 G1 PASS，也不授权 HPC。

## 冻结的第五轮证据

第五轮 fresh formal review 使用 64-row manifest
`RQ014_plan_v1p5_review_manifest_round5_blocked_20260712.sha256`，SHA-256 为
`213026f36faa56c7ee397c029274102265a04941309fea60dcd21f8d14096ce8`。原字节结论：

- statistics：`BLOCKED`，0 blocker / 1 major；artifact SHA-256
  `f7a38e1b1bebc96722b3aaf7002c314b40aa2ea8b653a493da70196bd46c0178`；
- execution/governance：`NO_BLOCKER`，0 blocker / 0 major；artifact SHA-256
  `cbbca13f75c8a36d2900cbe0c785c53d134d1c47e010da87697810165b0da17e`。

两份 review 与 manifest 均为 append-only provenance；不得改写，第五轮 reviewer 不参与下一轮计票。

## Finding-to-fix map

| Finding | 冻结风险 | 本轮整改目标 |
|---|---|---|
| STAT5-M01 | AVAILABLE group 逐行限制 `path_type_or_NA∈{CP,HO,MP,F}`，却没有显式要求同一 `segment_id×feature_id×horizon_id` 的所有 tau rows 使用同一 path type；混合 CP/HO 可改变 aggregate、envelope 与 rank | 冻结 AVAILABLE group 的 `path_type_set` cardinality 必须等于 1，并绑定 checksum-bound segment path lookup；TF 的 `h_common_tick` 继续 cardinality=1。Validator/test 必须让合法同值组通过、混合 path type 失败；aggregate 每个 scene group 只能贡献给一个 path-type 分支。 |

第五轮 execution reviewer 已独立确认：retained-fd bootstrap、wrapper capability、dependency-import 前拒绝、
private token、exact import、G0/G1/final bundle 与运行时闭包均无 blocker/major。省略 internal flag 的 direct
validate/submit 与携带 flag 但缺 capability 的路径都在 dependency marker 前拒绝。同账号主动仿造
descriptor 仍是已声明 NOTE residual，不被表述为密码学安全边界。

## 整改闭包证据

- `scene_anchor_domain_contract.available_group` 现在要求每组所有 AVAILABLE rows 的
  `path_type_or_NA` 集合基数严格等于 1，并等于 checksum-bound `(segment_id,tstar_context_step)`
  path lookup；同一 segment 的所有 feature/horizon groups 复用该 frozen value。TF 组内
  `h_common_tick_or_NA` 集合基数也继续严格等于 1。
- Aggregate generator 明确每个 per-scene group 只能贡献到一个 path-type branch。测试 validator 同时
  执行 lookup equality 与 `len(path_types)==1`；同值 CP/HO fixture 通过，混合 CP/HO 以及同值但不匹配
  lookup 均失败。15,328 groups 与 16/48/960/2,880 counts 不变，STAT4-m01 residual 不变。
- Recovery、envelope 与专属 test SHA-256 分别为
  `c1d3a8c4faeb04871e15d7d1d0f07edfd45b8e6904bdd5ac7e05fa3f1f412d7d`、
  `407d63209764896a673aa94811f9dd8b60a57a047d17e8cee0a3465c55b8c8a4`、
  `33756813a35523176e6c897f754e4d4a6f889ea7bb31411a529f5a7c09a2458b`。
- 第五轮四份 provenance artifact 进入 required set；required=57，与 v1.3 inheritance union=68。
  Launcher 为 116,287 bytes、SHA-256
  `6b3cf6da1d31bb4304617fe5305435fd656b9a2ca41020556ad0a8bba42bd91f`；wrapper 为 8,108
  bytes、SHA-256 `d8036336c354b202f388c5fd9dbc05a80a1e8292a574cc4c47400d0a772d7a1d`。

主代理独立复跑的 8-file RQ014 聚焦套件为 `194 passed`；全仓 non-shortcut 回归在排除本地未同步、
git-ignored 的 RQ009 scorer bundle 所专属 `test_verifier_runtime.py` 后为
`202 passed, 1 skipped, 2 deselected`，并已恢复测试产生的 RQ012 fixture 输出。Python compilation、
wrapper/operator syntax、strict JSON、exact hash/retained-fd assertions 与 `git diff --check` 均通过。

## 下一轮 gate

整改完成后必须重建新的 canonical manifest，并由两名未参与前五轮的 fresh reviewers 独立复核。
只有两路均为 `NO_BLOCKER` 且无
unresolved blocker/major，才可生成 formal G1。当前未读取 rating、未创建 production run root、未提交
Slurm。
