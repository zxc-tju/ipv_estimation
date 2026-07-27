RQ015A 执行层：v3 三路独立复审于 2026-07-26 完成，三路均判定
`BLOCKED / REQUEST_CHANGES`，`formal_g1_eligible=false`、`execution_authorized=false`。

现行计划为 `reports/plans/RQ015A_plan_v3_concentration_audit_20260726.md`
（SHA-256 `75912bc1433a5efb5b0520af492e27579e9a1f6652074d3f37eb3a77befff264`）；
综合复审为
`reports/knowledge/RQ015A_ipv_estimability_labelling/reviews/rq015a_three_reviewer_synthesis_v3_20260726.md`。

v3 的 continuous `q_eff` primary、bins-downstream decoupling、三恒等式骨架、OnSite
local-position、deterministic mean 与 explicit zero-support 已被接受，不得退回分档主产物或
estimability/“测出 IPV”命名。当前 blocker 是 PI rederivation condition 未被 append-only
supersede；逐产物 path/hash/key/role 与真实产物不一致；invalid `ipv_error` 仍会生成有效 q；
L2/L3 可跨 artifact pooling；C0 无 q 时可 fail open；以及不存在 exact command、有效 authorization、
完整 executor/validator/receipt/factor implementation。

因此仍未建立 RUN_ID 报告目录。仅允许起草并冻结新修订版及关闭包；新一轮复审无 blocker
且另获针对精确 run-spec SHA 的 scoped single-use authorization 后，才可创建运行目录。
在此之前不得构建最终 ledger/连续画像，不得生成 policy bins、episode summary 或 C0 routing，
不得解析新的 RQ007 held-out measurement fields，也不得 replay WOD/RQ014。
