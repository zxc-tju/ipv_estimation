# RQ015A — IPV 估计尝试状态与候选权重集中度回溯审计（知识层）

现行复审对象：`reports/plans/RQ015A_plan_v3_concentration_audit_20260726.md`  
计划 SHA-256：`75912bc1433a5efb5b0520af492e27579e9a1f6652074d3f37eb3a77befff264`  
基线 manifest：`reports/plans/RQ015A_plan_v3_checksums_20260726.sha256`（`6/6 OK`）

状态：`INDEPENDENT_REVIEW_COMPLETE / BLOCKED / REQUEST_CHANGES`  
`formal_g1_eligible=false`｜`execution_authorized=false`｜未创建 `decision.md`

## v3 复审结论

2026-07-26 三路以同一冻结 v3 事实基线独立完成复审，三路均未放行
（R1/R2 为 `BLOCKED`；R3 为含 2 个 blocker 的 `REQUEST_CHANGES`）：

- 技术正确性：`reviews/reviewer1_technical_v3_20260726.md`（4 blocker / 3 major / 1 minor）
- 科学意义与主张边界：`reviews/reviewer2_significance_v3_20260726.md`（2 blocker / 3 major / 2 minor）
- 可读性、执行与治理：`reviews/reviewer3_readability_execution_v3_20260726.md`（2 blocker / 3 major / 0 minor）
- 综合：`reviews/rq015a_three_reviewer_synthesis_v3_20260726.md`
- 校验：`reviews/rq015a_review_manifest_v3_20260726.sha256`

v3 的 continuous `q_eff=K_eff/K` primary、bins-downstream decoupling、三条守恒恒等式骨架、
OnSite local-position、`sorted + math.fsum` 和 L3 zero-support handling 均是实质进步，应保留。
v1 已关闭的构念收窄继续成立；不得退回分档主产物，不得把 attempt/concentration-only
命名为 estimability 或解释为“是否测出 IPV”。

当前决定性阻断为：

1. v3 单方面声称 PI 的 dev+guard rederivation condition 已解除，但 manifest-bound disclosure
   与 RQ007 README 仍把它写成 mandatory；没有 append-only supersession。
2. run spec 没有 exact command/entrypoint，引用的 authorization fragment 不存在，split 只是
   未绑定的符号串；executor、validator、receipt writer、factor/Spearman/bootstrap 也未实现。
3. schema 未绑定逐产物 exact path/hash/source columns，且与真实产物存在硬 mismatch：OnSite
   实际为 `case_key`，RQ014 anchor identity 含 segment/feature/horizon/tau/candidate，M3 alpha collapse
   与三 measurement roles 被混为同一计数维度。
4. `q_eff` 实现没有执行 plan 的 `ipv_error>=1 -> None`：例如 `q_eff(1.1,7)=1.0`，并接受
   negative error/bool K，使用 clipping 把 invalid domain 变成有效值。
5. L2/L3 丢掉 `artifact_id`、`measurement_role`，synthetic mixed rows 可跨 artifact pooling；
   schema 又未定义代码强制读取的 perspective/configuration。
6. C0 在没有任何 q evidence 时仍可返回 `NO_AUDIT_TRIGGER_DETECTED`，不校验重叠/越界计数，
   sensitivity 翻转只给 `stable=false` 而不 withhold action。
7. validate-only 声称纯 stdlib 且必须跑 16 fixtures，但 fixtures 依赖未声明的 `pytest`；
   16 项 synthetic tests 也未覆盖以上闭合风险。

准确 gate 状态：G1 `CLOSED`；G2–G6 `PARTIAL / BLOCKING`；G7 `OPEN / BLOCKING`。
v3 的 `6/6 OK` 与 16 fixtures 不能升级成 Formal G1 或执行授权。

## held_out 暴露治理状态

`sealed_exposure_disclosure_20260726.md` 已记录 PI 于 2026-07-26 采用判读 A 并给予治理豁免；
附加条件是必须在排除 held-out 的 development/guard 范围内重导出阈值。v3 复审不推翻
该 PI 裁定，但明确它是 disclosed governance waiver，不会把已观察的聚合信息变回
科学上 pristine untouched。准确的事实措辞应是：程序曾解析并聚合 held-out 逐行字段，
但未显示、导出、落盘或人工检视 held-out 单行值，也未作 held-out 效应估计/拟合/检验。

RQ007 knowledge README 已有 pointer，但 frozen `decision.md` 仍写 sealed/untouched；v3 又禁止修改
owning-RQ decision。下一版须由 PI 明确 README addendum 的权威优先级，或授权一个窄化的
append-only decision pointer；不得由执行者自行改写 accepted claim ledger。

## 当前执行边界

仅允许起草新的 checksum-frozen closure package。完整包经新一轮独立复审无 blocker、且对
精确 run-spec SHA 另获 scoped single-use authorization 之前，不得构建最终 ledger/连续画像，
不得生成 policy bins、episode summary 或 C0 routing，不得解析新的 RQ007 held-out measurement
fields，不得 replay WOD/RQ014，不得改写任何 owning RQ accepted claim/decision。

## 历史 v2

v2 计划、三路复审与综合保留为历史：

- `reports/plans/RQ015A_plan_v2_concentration_audit_20260726.md`
- `reviews/rq015a_three_reviewer_synthesis_v2_20260726.md`
- `reviews/rq015a_review_manifest_v2_20260726.sha256`

v2 建立 continuous-primary 方向，但承诺的 closure objects 当时尚不存在；v3 将其中一部分
写成真实文件，却仍未关闭其语义、数据绑定与执行链。

## 历史 v1

v1 计划与三路复审保留为历史：

- `reports/plans/RQ015A_plan_v1_attempt_status_and_weight_concentration_audit_20260726.md`
- `reviews/rq015a_three_reviewer_synthesis_v1_20260726.md`
- `reviews/rq015a_review_manifest_v1_20260726.sha256`

v1 的中心贡献是关闭 v0 的构念越界；其 blocker 由 v2 继续响应，但没有被当前 v2
六文件 manifest 完成性关闭。

## 历史 v0

v0 计划 `reports/plans/RQ015A_plan_v0_ipv_estimability_labelling_20260726.md` 及其三路复审仅作历史记录：

- `reviews/reviewer1_technical_v0_20260726.md`
- `reviews/reviewer2_significance_v0_20260726.md`
- `reviews/reviewer3_readability_execution_v0_20260726.md`
- `reviews/rq015a_three_reviewer_synthesis_v0_20260726.md`

v0 的 estimability 命名与 `ESTIMABLE/NOT_ESTIMABLE` 标签已作废，不得再作为执行或主张依据。
