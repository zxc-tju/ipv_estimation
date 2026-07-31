# RQ007: Interaction-Conditioned IPV Estimability

Status: study final review PASS (development/guard); held-out confirmatory evaluation unopened, with
aggregate IPV/error exposure disclosed and PI-waived below; knowledge `decision.md` frozen 2026-06-24

Governance addendum (2026-07-26): an RQ015/RQ015A whole-corpus scan programmatically parsed and
aggregated IPV/error fields that included held-out cases. No held-out single-row values were displayed,
exported, persisted, or manually inspected, and no held-out effect estimation, fitting, or testing was
performed. The PI recorded a governance waiver (reading A) with a mandatory dev/guard-only threshold
rederivation condition in `RQ015A_sealed_exposure_disclosure_20260726.md`. Any future held-out
confirmation must proceed under this disclosed waiver and new explicit PI authorization; it must not be
described as pristine untouched. The frozen 2026-06-24 claim decision is not otherwise modified here.

Execution layer: `reports/studies/RQ007_interaction_conditioned_ipv_estimability/`
Latest run: `RQ007_1_ipv_estimability_20260622T155229Z_289d9a99` (`COMPLETE`; all gates PASS)
Plan: `reports/plans/RQ007_plan_v0_interaction_conditioned_ipv_estimability_20260622.md`

Paper section: Methods / v4.1 estimability contract (PAPER001/PAPER002).

## Research Question

Is IPV equally estimable at every timestamp, or is per-frame IPV identifiability
interaction-conditioned — and is estimability distinct from behavioural settling and from the
choice of episode summary?

## Current Interpretation

Study-level review accepted the development/guard evidence boundary: estimability is
interaction-conditioned (the estimator concentration index is lower within causal opportunity
windows) but mostly proximity-driven, with a small conflict-geometry-specific residual whose
case-clustered CIs exclude zero; estimability is not behavioural settling; and episode-level
IPV summaries are definition-dependent. See `synthesis.md` for the consolidated read,
`decision.md` for the frozen claim ledger, `report_index.md` for the execution package, and
`reviews/` for the review boundary.

This knowledge folder is the single synthesis point for RQ007. The claims are frozen in
`decision.md` (2026-06-24); apply the proximity-bounded caveat. Do not use sealed/held-out,
PET, intensity, order, priority, or outcome fields as claim sources.

## Append-only pointer（2026-07-26，RQ015A 暴露登记）

RQ007 的 held_out 分割（11,342 cases）曾被 RQ015 立项扫描以**语料级聚合形式**触及：
程序解析并聚合了逐行字段，未显示/导出/人工检视任何单行值，未在 held_out 上做推断。
PI 于 2026-07-26 裁定为**判读 A：不构成对封存目的的实质破坏，记录豁免**，
held_out 确认路径不受影响。完整登记与裁定：
`RQ015A_sealed_exposure_disclosure_20260726.md`（本目录）。

**因此本 RQ 的文档不应再无条件声称 held_out 为 `untouched`**；
应表述为"held_out 未被用于任何估计/拟合/检验，仅曾进入语料级聚合统计（见上述登记）"。
另：本 RQ split 的真实标签为 `development / guard / held_out`，"sealed" 一词停用。
