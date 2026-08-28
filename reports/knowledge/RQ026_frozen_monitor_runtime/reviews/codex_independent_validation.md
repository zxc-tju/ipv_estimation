# RQ026 Codex Independent Validation Summary

本文件把 2026-08-27 至 2026-08-28 已形成的 RQ026 关键 runtime 证据收成一个独立摘要，
用于回答三个问题：formal pilot 是否真的通过、promotion 是否改动了内容、repair15 full 包
是否已经到了可提交但未提交的状态。

## 1. Formal pilot 是否真的通过

- 顶层 PASS 文件：
  `.codex-fleet/rq026-frozen-monitor-runtime/work/hpc_spec/PILOT_PASS.json`
  sha256=`5b71d3a594a3e4392bc9ac89c724e007986498067b21ee6dc67288efa29fcc40`
- 文件内 formal gate：
  `context_cell_present_rows=256`，
  `parity_status_match_rows=256`，
  `parity_reason_match_rows=256`，
  `parity_ipv_comparable_rows=167`，
  `parity_ipv_max_abs_diff=0.0`，
  `unhandled_failure_rows=0`
- 顶层状态：`status=PASS`

这些数字与 r3 独立验证文件完全一致，说明 formal pilot PASS 不是口头结论，而是有顶层 receipt
与独立复核双重支撑的结果。

## 2. Promotion 是否改动了内容

- promotion receipt：
  `.codex-fleet/rq026-frozen-monitor-runtime/work/hpc_spec/PROMOTION_RECEIPT.json`
  sha256=`9b394807f17399246f491c79a3eb7986712b39df450bc1b7ce313d9ae2d154d2`
- candidate PASS：
  `.codex-fleet/rq026-frozen-monitor-runtime/work/hpc_spec/repair14_split_runtime/hpc_pilot/PILOT_PASS_candidate.json`
  sha256=`5b71d3a594a3e4392bc9ac89c724e007986498067b21ee6dc67288efa29fcc40`
- promoted PASS：
  `.codex-fleet/rq026-frozen-monitor-runtime/work/hpc_spec/PILOT_PASS.json`
  sha256=`5b71d3a594a3e4392bc9ac89c724e007986498067b21ee6dc67288efa29fcc40`

promotion receipt 明确写明 `content_identical_to_candidate = true`。因此这次提升只是把已通过的
candidate 正式放到顶层，不是重算，也不是改写结果。

## 3. repair15 是否已经到了“可提交但未提交”的状态

- repair15 独立验证：
  `.codex-fleet/rq026-frozen-monitor-runtime/work/hpc_spec/repair15_split_full/INDEPENDENT_VALIDATION.json`
  sha256=`1ba09c7c0af927837f7e669afbbdc36e4aae9e648c011d98f58b29d3e757a727`
- 结论：`verdict=PASS`，`ready_for_live_preflight=true`，`p0=0`，`p1=0`
- 关键渲染结果：`136` 个 shard，`67,861` 行，首 shard `full_0001=500`，末 shard `full_0136=361`
- 关键本地验证：`py_compile PASS`，`pytest 11 passed`，`full_exact_aggregator.py --validate-only PASS`

这说明 full exact 的提交材料已经可用，但并不等于 full 已经提交。是否能把 Stage1/Stage2 的数组并发
从 fail-closed 的 `1` 放大，仍取决于 live HPC preflight。

## 4. 当前 live preflight 的真实边界

- live preflight 文件：
  `.codex-fleet/rq026-frozen-monitor-runtime/work/hpc_spec/repair15_split_full/REMOTE_PREFLIGHT.json`
  sha256=`5fa57be1734bfe9348465938406493f9cd002810fbf5711ef85601b6032c1755`
- 当前状态：`status=FAIL`
- 当前可执行建议：`recommended_array_concurrency=1`
- 含义：因为 SSH refresh 没成功，这个 `1` 是 fail-closed 保守值，不是 live cluster 实测最优值。

## 5. 本摘要刻意不做的事

- 不把 runtime evidence 写成 accuracy、deployment 或 alert effectiveness 结论。
- 不声称 full exact 已提交或已完成。
- 不把 WP8 sign-only 辅助线替换成 exact runtime 主证据。
