# RQ014 v1.5 formal G1 round-2 remediation

日期：2026-07-12  
状态：第二轮整改已闭合，等待第三轮 fresh formal review；本文不是 G1 PASS，也不授权 HPC。

## 冻结的 BLOCKED 证据

第二轮 formal review 读取 51-row candidate manifest
`RQ014_plan_v1p5_review_manifest_round2_blocked_20260712.sha256`，其 SHA-256 为
`5a6b34f85904b3cedfb59d1d949058148869ea01fe2daa552637fd1d1618dbed`。该轮结论必须保留：

- statistics：`BLOCKED`，0 blocker / 6 major / 2 minor；artifact SHA-256
  `8a43140dd6cbd108aba8cfc40905f51651af15307af2fd1821599beaf56d4ea9`；
- execution/governance：`BLOCKED`，1 blocker / 4 major；artifact SHA-256
  `0ced6fd9bd815827cedd5e81215f3204d51c56474eea63ad9ee5822b4f9a0c4a`。

两份 artifact、该轮 manifest 与第一轮 BLOCKED 证据均为 append-only provenance；不得改写成 PASS，
也不得在下一轮 formal G1 计票中复用 reviewer 结论。

## Finding-to-fix map

| Finding | 冻结风险 | 本轮整改目标 |
|---|---|---|
| EG2-B01 | source path 先 hash、后另开解析，receipt 与实际去密字节可分离 | 每个 pickle/TSV/CSV 以 `O_NOFOLLOW` 单 FD 读一次，fstat continuity、expected size/SHA 与解析 bytes 完全相同；从 BytesIO/StringIO 解析并补 swap tests。 |
| EG2-M01 | F01–F04 阴性关闭证据不在 review/final bundle | 把 exact forensics report 纳入 required set，并在 formal G1 validator 中逐项核 registry evidence path/hash。 |
| EG2-M02 | spec 声称 one-read，但 submit 又重开；路径也未限制在 immutable managed root | 只接受 managed `run_specs/` 下只读 regular non-symlink；单 FD 读取并把 retained canonical bytes 直接 seal。 |
| EG2-M03 | kickoff 要求 validate-only 产生只有 submit 才能产生的 snapshot receipt/rendered Slurm | 将前置证据改为 validate-only 实际输出的 snapshot file plan、job/resource/runtime metadata；submit 内部证据另行 fail-closed。 |
| EG2-M04 | 文档称 receipt 后只改 central allowlist，但 execution contract 仍会拒绝 preflight | 预先把 preflight 注册为 receipt-conditional operation；中央 allowlist 继续是当前唯一 operation 开关。 |
| STAT2-M01 | between-cell support、NA cluster、cluster/shard/common-support 分母不完整 | 明确所有 rating-blind support 变化；冻结 NA 处理、整数分母和 common-support artifact schema。 |
| STAT2-M02 | adaptive extension 与固定 2,880-row 排名/冻结状态机冲突 | v2 禁止 adaptive extension；仅 compatible OBSERVED rank 1 可 freeze/replay，否则终结为覆盖或未恢复状态。 |
| STAT2-M03 | BL90 rank 1 缺少合法 clean-replay legacy envelope 输入 | selected recipe 与 replay allowlist 条件式绑定 exact legacy artifact/manifest/L-M-U bytes。 |
| STAT2-M04 | 未定义 placebo/robustness 名称会改变最终 verdict | 只保留已定义的 history-only structural gate；其余项目 future DENY、non-gating，不进入当前 verdict。 |
| STAT2-M05 | ledger chain 缺少 exact preimage、编码、genesis 与 append order | 冻结 exact keys、canonical JSON-LF、domain、self-hash exclusion、zero genesis 与单一 aggregator order。 |
| STAT2-M06 | envelope bootstrap occurrence weight 与 degenerate replicate 成功规则不唯一 | 冻结 occurrence/role/observation 权重、1000 draws 无重抽，以及 finite collapsed/zero-width 与 failure 判据。 |
| STAT2-m01/m02 | fold seed bytes 与 PPR rank-vector tolerance 不明确 | 冻结 UTF-8 seed preimage；rank-vector exact 只适用于 RWS/PSP，PPR 为 NOT_APPLICABLE。 |

## 整改闭包证据

统计合同已把固定搜索空间、support/common-support 分母、冻结与 replay 状态机、BL90 legacy
输入、当前 verdict gate、ledger byte semantics、bootstrap occurrence 权重与 fold seed 全部转成可机检
条款；v2 明确禁止自适应扩展，基础屏固定为 960 predictor cells × 3 association methods = 2,880
rows。运行合同已完成以下 fail-closed 闭包：

- 10 个去密输入均由 launcher 传入 reviewed role/path/size/SHA-256；exporter 对每个输入只用一个
  `O_RDONLY|O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK` descriptor，核验前后 `fstat`、精确长度与摘要，
  只解析 retained bytes，并把同一摘要写入 receipt；
- F01–F04 的 exact 阴性证据字节进入 review/final required set，formal G1 validator 同时核验 registry
  状态、evidence path 与三方 SHA；
- production run spec 仅允许 managed `run_specs/` 的 direct-child、只读 regular non-symlink
  canonical JSON；首次单 FD 读取的 retained bytes 贯穿 validation、sealing 与 submission，不重开路径；
- validate-only 输出实际可获得的 closed-snapshot file plan 与 submission/runtime plan；snapshot receipt
  和 rendered sbatch 仍只在 submit 内生成并再次 fail closed；
- preflight 已在 execution contract 中预注册为 receipt-conditional，但中央 allowlist 当前仍只允许
  declassification export。未来修改中央授权字节后，必须重建 candidate manifest、重新双审、重建
  formal G1 与 final bundle，不能仅改 allowlist 后运行。

闭包后的 exact 运行字节为：launcher 106,863 bytes，SHA-256
`4f73b1f98f6320e509e84569e867ef2416d17cdb32741abc979e8649504b9d8b`；operator wrapper
7,138 bytes，SHA-256 `2df7458bbaf5e92489cbba4b682185b670da77a950dfc31c6f03961fe62df032`；
preflight 69,166 bytes，SHA-256
`f91bbd2aef8ab6678109c1391d46672c41a41a03c5b1a9d67c35d01ce3de4102`。

主代理独立复跑的 8-file RQ014 聚焦套件为 `173 passed`。全仓 non-shortcut 回归在排除本地
未同步、git-ignored 的 RQ009 scorer bundle 所专属的 `test_verifier_runtime.py` 后为
`181 passed, 1 skipped, 2 deselected`；保留该模块时为 `187 passed, 1 skipped, 2 failed,
2 deselected`，两项失败均精确来自缺失的 `models/rq009_m3/m3_scorer.joblib`，与本轮 RQ014
修改无关。Python compilation、shell/bootstrap syntax、JSON parsing 与 `git diff --check` 均通过。
广泛测试产生的 RQ012 fixture 输出已恢复，没有纳入 RQ014 diff。

一名未参与本轮实现的只读 closure auditor 又逐项复核 A–F，返回 `NO_BLOCKER`、0 blocker、
0 major；该结论只是冻结前内部审计，不参与第三轮 formal G1 计票。

## 下一轮 gate

整改完成后必须重建新的 canonical review manifest，并由不同的 fresh
statistics 与 execution/governance reviewers 独立审查。只有两路均为 `NO_BLOCKER`、0 unresolved
blocker、0 unresolved major，lead 才能生成 formal G1 artifact。当前仍未读取 rating value、未创建
production run root、未提交 Slurm。
