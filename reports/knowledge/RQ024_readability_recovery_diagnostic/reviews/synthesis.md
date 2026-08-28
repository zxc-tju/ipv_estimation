# RQ024 Synthesis

Status: WAITING_ON_PI
Verdict: MIXED_DIAGNOSTIC
Tier2: blocked

## What The Reports Establish

- 本轮只诊断 WP2 Tier1 sealed synthetic 结果，分析分母固定为 `288` 行、噪声成对分母固定为 `144` 对。
- Gate A 在既有 adjudication 中已失败：相邻更严格阈值比较共 `42` 组，其中 `36/42` 组 `risk_mae_rad` 上升。
- 噪声混杂与指标失配同时存在：`115/144` 个噪声成对单元出现“误差更差，但至少一个 gate 指标朝更安全方向移动”。
- `q_eff`、`k_eff`、`ipv_error` 近乎同一单调量；严格 gate 同时富集边界/近边界高误差行。

## What The Reports Do Not Establish

- 本轮不能把失败机制精确归因为 argmax 选边界，因为 `tier1_results.csv` 不含原始 `weights`，`286/288` 行只能标 `INCONCLUSIVE_ARGMAX`。
- 本轮不提供任何 Tier2 授权，不提供真实 AV / human 数据上的新主张，也不重开 RQ017/RQ018/RQ019/RQ021 已接受结论。

## Boundary Conditions

- 仅允许 sealed 288 行与只读 pilot candidate grid。
- 不改 comparator，不改阈值，不重估，不新生成 synthetic。
- 知识层状态仅为 `WAITING_ON_PI`；本目录不生成 `decision.md`。

## Manuscript-Safe Language

- 可写：`MIXED_DIAGNOSTIC`、`Tier2 blocked`、`strict gates retain boundary or near-boundary high-error rows under this sealed synthetic grid`。
- 不可写：对真实系统的性能判断、因果措辞、任何已接受 RQ 的重判。
