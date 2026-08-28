# RQ026 Review Synthesis

RQ026 关注的是 frozen monitor runtime evidence，而不是模型表现好坏。到 2026-08-28 为止，
整体已经完成了三层递进证据：第一层是 WP4 本地 no-refit reproduction 通过；第二层是
repair14 split-runtime exact formal pilot 被独立验证后正式提升为 PASS；第三层是
repair15 split-full 包已在本地完成独立验证，说明 full exact 的拆分提交材料已经具备。
当前尚未完成的唯一核心门槛，是 live HPC preflight refresh 后的 full exact 正式提交。

## 当前证据支持的结论

- `FORMAL_PILOT_PASS` 是成立的。formal pilot 的顶层 PASS 文件、promotion receipt、
  r3 独立验证三者互相对齐：Stage1/Stage2 job 链为 `2336919 -> 2336920 -> 2349471`，
  最终 exact formal gate 达到 `context/status/reason = 256/256`，`167/256` 行可比，
  最大 IPV 差 `0.0`，`unhandled = 0`，且 Stage2 realtime M2 scorer 被调用 `256` 次。
- full exact 还不能写成“已跑”。repair15 当前只证明 full split 包在本地可渲染、可校验、
  可 fail-close，并且 exact-only aggregator 的 gate 已定义清楚；它没有提供任何已提交或已完成的
  Slurm 证据。
- 最新 live preflight 只能支持“保守并发=1”的 fail-closed 结论。因为 SSH refresh 失败，
  这次没有拿到可信的 `squeue`、`sinfo` 或 account snapshot，所以任何更大的并发值都只能算
  公式上界，不是 live-validated recommendation。

## 当前证据不支持的结论

- 不支持把 RQ026 写成 RQ024 的 accuracy 修复。RQ024 仍是
  `ACCEPTED / MIXED_DIAGNOSTIC / Tier2 blocked`，RQ026 只能讨论 runtime 链路是否可复现。
- 不支持把 WP8 sign-side 结果升级成正式 frozen-monitor runtime 成功。sign lane 与 exact lane
  的接口和 claim 范围不同，不能互相替代。
- 不支持 deployment、线上稳定性、告警收益或 human/AV 行为优劣主张。

## 下一道门

下一步不是重写结论，而是等待 live HPC refresh 成功后，用新的
`recommended_array_concurrency` 覆写 repair15 的 `%1` fail-closed 默认值，然后再按
`stage1 -> stage2 -> full exact aggregator` 的顺序提交 full exact。只要这一步还没发生，
当前知识层状态就应保持为 `FORMAL_PILOT_PASS / FULL_NOT_SUBMITTED`。
