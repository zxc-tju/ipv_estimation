# RQ026 Frozen Monitor Runtime

状态：`APPROVED / EXECUTING`；`FORMAL_PILOT_PASS`；`FULL_NOT_SUBMITTED`。

RQ026 要解决的问题不是模型精度，而是 frozen monitor 配置在本地与 Tongji HPC 上的
runtime 路径能否被复现、拆分执行、独立复核并长期留痕。当前整体阶段已经走到：
WP4 本地 no-refit reproduction 已通过，repair14 exact formal pilot 已正式提升为 PASS，
repair15 full exact split-runtime 包已完成本地独立验证，但 full exact 还没有提交到 HPC。

本知识层文件夹只记录 runtime 证据，不构成 deployment、accuracy、alert effectiveness
或 paper claim。

## 当前已确认

- formal pilot 已由 repair14 split-runtime 路径正式通过。顶层
  `.codex-fleet/rq026-frozen-monitor-runtime/work/hpc_spec/PILOT_PASS.json`
  为 `PASS`，sha256=`5b71d3a594a3e4392bc9ac89c724e007986498067b21ee6dc67288efa29fcc40`。
- promotion receipt 记录 candidate 原样提升为顶层 PASS，且 full 未启动。
  receipt 路径为 `.codex-fleet/rq026-frozen-monitor-runtime/work/hpc_spec/PROMOTION_RECEIPT.json`，
  sha256=`9b394807f17399246f491c79a3eb7986712b39df450bc1b7ce313d9ae2d154d2`。
- independent validation r3 证实 Stage2 job `2349471` 已 `COMPLETED 0:0`，并得到
  `context/status/reason = 256/256`、`comparable = 167`、`max parity ipv abs diff = 0.0`、
  `unhandled = 0`、`stage2_score_calls = 256`。验证文件路径为
  `.codex-fleet/rq026-frozen-monitor-runtime/work/hpc_spec/repair14_split_runtime/hpc_pilot/INDEPENDENT_VALIDATION_R3.json`，
  sha256=`479b248cca74657a169648cde46a3252a56db1ae304d0ef6c4ea90da04d5c8be`。
- repair15 split-full 本地独立验证已通过，说明 full exact 包已经能稳定渲染：
  `136` 个 shard、`67,861` 行、首 shard `full_0001=500`、末 shard `full_0136=361`，
  `pytest -q` 为 `11 passed`，exact-only aggregator validate-only 为 `PASS`。验证文件路径为
  `.codex-fleet/rq026-frozen-monitor-runtime/work/hpc_spec/repair15_split_full/INDEPENDENT_VALIDATION.json`，
  sha256=`1ba09c7c0af927837f7e669afbbdc36e4aae9e648c011d98f58b29d3e757a727`。
- 最新 live preflight 仍是 fail-closed：由于本轮 SSH refresh 失败，`recommended_array_concurrency`
  只能保守取 `1`，不能把任何 live cluster capacity 写成已验证事实。对应文件为
  `.codex-fleet/rq026-frozen-monitor-runtime/work/hpc_spec/repair15_split_full/REMOTE_PREFLIGHT.json`，
  sha256=`5fa57be1734bfe9348465938406493f9cd002810fbf5711ef85601b6032c1755`。

## 当前不支持

- 不支持把这些结果写成 accuracy 修复、性能优越、部署成功或报警有效性证明。
- 不支持声称 full exact 已提交、已运行或已通过；当前状态仍然是 `FULL_NOT_SUBMITTED`。
- WP8 `TRUSTED_SIGN_ONLY` 只能作为 sign-side 辅助证据，不能替代 frozen exact runtime 证据。

## 目录角色

- [synthesis.md](./reviews/synthesis.md)：汇总当前证据能支持什么、不能支持什么。
- [codex_independent_validation.md](./reviews/codex_independent_validation.md)：独立验证摘要与哈希锚点。
- `reports/plans/RQ026_plan_v0_frozen_monitor_runtime_20260824.md`：本轮唯一执行合同。
