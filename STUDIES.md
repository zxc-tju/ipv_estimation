# Research Study Index

This repository is the source of truth for IPV estimation research plans, execution reports,
review records, and evidence decisions. Manuscript text lives in the separate paper repository;
durable research knowledge lives here.

## Program Progress Dashboard

Use the following files to synchronize program-level progress, dependencies, blockers, latest
artifacts, and next gates:

- `reports/knowledge/RQ_PROGRESS_DASHBOARD.md`
- `reports/knowledge/rq_progress_registry.csv`
- `reports/plans/README.md`

`STUDIES.md` remains the compact study index. The dashboard carries the operational status view
and must not override an accepted RQ `decision.md`.

## Three-Layer Research Governance

- `reports/plans/`: centralized active/proposed plans and Claude→Codex orchestration prompts.
- `reports/studies/`: execution layer. Each independent run uses a versioned directory such as
  `RQ001_1_current_ipv_distribution_20260618`.
- `reports/knowledge/`: interpretation layer. Each RQ has one knowledge folder containing reviews,
  synthesis, and the accepted/rejected/deferred claim ledger.

One RQ may have multiple execution reports. One RQ should have one knowledge folder. The suffix
after the RQ stem in an execution folder is an execution version, not a new research question.

## Study Index

| RQ | Topic | Status | Execution layer | Knowledge layer | Paper use |
|---|---|---|---|---|---|
| RQ001 | Online IPV interval deployability | review | `reports/studies/RQ001_online_ipv_interval/` | `reports/knowledge/RQ001_online_ipv_interval/` | legacy interval engineering / M4 ablation boundary |
| RQ002 | Self-anchor as group norm | review | `reports/studies/RQ002_self_anchor_group_norm/` | `reports/knowledge/RQ002_self_anchor_group_norm/` | reject self-anchor-only normative authority |
| RQ003 | NSFC external evidence | accepted | `reports/studies/RQ003_nsfc_external_evidence/` | `reports/knowledge/RQ003_nsfc_external_evidence/` | Tier B boundary; no robust increment |
| RQ004 | IPV state-space conclusions | review | `reports/studies/RQ004_ipv_state_space/` | `reports/knowledge/RQ004_ipv_state_space/` | episode-level state organization |
| RQ005 | NMI draft evidence gap | review | `reports/studies/RQ005_nmi_evidence_gap/` | `reports/knowledge/RQ005_nmi_evidence_gap/` | leakage and claim governance |
| RQ006 | Sigma sensitivity | archived-review | `reports/studies/RQ006_sigma_sensitivity/` | `reports/knowledge/RQ006_sigma_sensitivity/` | robustness appendix |
| RQ007 | Interaction-conditioned IPV estimability | accepted dev/guard boundary; held-out confirmatory evaluation unopened, but aggregate IPV/error exposure is disclosed and PI-waived (2026-07-26) | `reports/studies/RQ007_interaction_conditioned_ipv_estimability/` | `reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/` | v4.1 estimability contract; any held-out confirmation proceeds under disclosed waiver and new PI authorization |
| RQ008 | InterHub temporal IPV discovery | accepted negative boundary; RQ008B not authorized | `reports/studies/RQ008_interhub_temporal_ipv_discovery/` | `reports/knowledge/RQ008_interhub_temporal_ipv_discovery/` | no positive temporal law from RQ008A |
| RQ009 | Estimability-aware dynamic counterpart-conditioned envelope | accepted (R3 context-conditioned conformal envelope; IPV-conditioning channels internal null) | `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/` | `reports/knowledge/RQ009_dynamic_counterpart_conditioned_envelope/` | R3 context-conditioned envelope; conditioning channels internal ablation |
| RQ010 | WOD-E2E tracking feasibility and preference validity | feasibility accepted; **RQ010B COMPLETE — bounded NULL** (2026-07-03): candidate IPV does not predict human preference and is not comparable to physics; M3 does not transfer to WOD-E2E | `reports/studies/RQ010_wod_e2e_tracking_feasibility/` | `reports/knowledge/RQ010_wod_e2e_tracking_feasibility/` | registered external-validity boundary (bounded null); manuscript R4 WOD-E2E leg = negative, not a positive claim |
| RQ011 | OnSite full-universe readiness | accepted `READY_WITH_FROZEN_EXCLUSIONS` | `reports/studies/RQ011_onsite_full_universe_readiness/` | `reports/knowledge/RQ011_onsite_full_universe_readiness/` | OnSite universe/scope decision |
| RQ011B | OnSite moment-level IPV monitor validity | closed-out `PROVISIONAL_NULL / UNDER_IDENTIFIED` (measurement-limited; not a frozen manuscript claim) | `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/` | `reports/knowledge/RQ011_onsite_full_universe_readiness/` | not demonstrated on OnSite pending adequate interaction-failure segment retrieval/segmentation |
| RQ012 | OnSite automatic-event harm (scope revised: automatic events + official outcomes; human labels deprecated) | accepted; RQ012B COMPLETE — deviation→harm BOUNDED/NULL across the full behavioural battery (no IPV-specific channel; passivity→deadlock unconfirmed hint); [RQ012B report](reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/00_entry/index.html) | `reports/studies/RQ012_onsite_event_annotation_readiness/` | `reports/knowledge/RQ012_onsite_event_annotation_readiness/` | realised interaction consequence (bounds R5; bounded/null) |
| RQ013 | Beyond-safety incremental validity | planning | `reports/studies/RQ013_beyond_safety_incremental_validity/` | `reports/knowledge/RQ013_beyond_safety_incremental_validity/` | final baseline-relative utility |
| RQ014 | WOD-E2E lost rating↔IPV-deviation result recovery | R3 screen complete (960 rows): one secondary R04N/NMD/RWS candidate (`r=-0.384`, `n=42`), primary NEX 0; R10L `DEFECT`, probe ceiling `UNCERTAIN`; [Codex result review](reports/knowledge/RQ014_wod_e2e_rating_recovery/reviews/codex_review.md) = `PENDING_REPLAY / NOT ACCEPTED` | `reports/studies/RQ014_wod_e2e_rating_recovery/` | `reports/knowledge/RQ014_wod_e2e_rating_recovery/` | no manuscript claim; selected-recipe freeze, clean replay, durable evidence package, and claim acceptance remain pending |
| RQ015 | ~~IPV 可估计性契约与估计器修复（合并版）~~ **已按 PI 决策 2026-07-26 拆分为 RQ015A / RQ015B**；v1/v1.2 保留为历史记录，不再作为执行依据 | superseded-split | n/a | n/a | 见 RQ015A / RQ015B |
| RQ015A | **IPV 估计尝试状态与候选权重集中度回溯审计**（v8 计划 + run spec **v7** + ledger **schema v4**；continuous `q_eff` primary；bins 不进入 episode/C0） | **executed 2026-07-31** — 审计已首次真正运行完毕，`run_receipt = PASS`；四产物合计 **14,473,982** measurement 行（onsite 281,268 / wod 906 / sigma01 5,197,072 / rq009_feature_matrix 8,994,736），`held_out_parsed_rows = 0`；`q_eff` 值域两端逐位命中理论界 [1/7, 1.0]；报告 `bounded_report.md` 已交付 | `reports/studies/RQ015A_ipv_estimability_labelling/RQ015A_1_concentration_audit_20260731T093746Z_e82091ce/` | `reports/knowledge/RQ015A_ipv_estimability_labelling/` | 可审计范围 **4/6 产物**（InterHub sigma01 + RQ009 feature matrix + OnSite + 已取回的 WOD full479 906 行，**WOD 为部分覆盖**）；**C0 路由不稳定**：sigma01 与 rq009_feature_matrix primary `NO_AUDIT_TRIGGER_DETECTED` 但 `stable=false`，一档敏感性 → `OWNER_REANALYSIS_REQUIRED`（是否重估 RQ009 属 PI 决策）；OnSite 仅 **1.06%** 行携带 IPV 数值；四产物中三个 `BINS_WITHHELD_UNSTABLE`；不得声称测出/未测出 IPV，禁用 estimability 表述 |
| RQ015B | IPV 估计器数值修复与 verifier 弃权闸（log 域改写 + 正交结果契约 + 生产兼容层 + 机制拆分 + 覆盖审计；计划 `reports/plans/RQ015B_plan_v0_estimator_repair_and_abstain_gate_20260726.md`） | planning / 待独立双路复审；无计算授权；实现为 BUILD_WHILE_DENY（测试 36/36） | `reports/studies/RQ015B_estimator_repair_and_abstain_gate/` | `reports/knowledge/RQ015B_estimator_repair_and_abstain_gate/` | 不重训 M3；弃权闸部署须先过 gate-pass 条件覆盖审计 |
| RQ016 | 用只过门的样本重建机制二的人类 envelope（去除伪零后的覆盖行为、区间宽度、两道门串联弃权率） | **executed 2026-08-03** — 单 agent 单轮，监督方已独立复算放行。在 `development + guard` 域同法跑两臂（唯一变量为样本口径）：90% 名义层 coverage 0.898832 → 0.902689，平均区间宽度 1.016189 → 1.300967（**+28.02%**）；两道门合并弃权 **32.0583%（284,964/888,892）**，机制一 28.4932% / 机制二 3.5651%。变宽机制经独立证实：目标四分位距 0.0493 → 0.2017 | `reports/studies/RQ016_human_envelope_rebuild/RQ016_1_envelope_rebuild_20260803T134808Z_d23fa836/` | `reports/knowledge/RQ016_human_envelope_rebuild/` | **尚无 `decision.md`，无已接受手稿主张**；与 RQ009 已发表数**不构成复现关系**（其 test 域含 held_out，本轮仅 dev+guard）；零点聚集只减半未消除（`\|y\|<1e-6` 由 42.39% 降至 29.63%）；支持门用 12 项而非 RQ009 原 15 项；描述性结果，禁用 estimability 表述 |
| RQ016B | 把重建后的 envelope 用到 WOD 与 OnSite 的可行性审计（含 envelope 目标值 ego 身份查证） | **audited 2026-08-04** — 只读审计，监督方已独立复算。**直接套用不可行**：WOD 与 OnSite 一行都没有七候选 MSE（`mse_0..6`/`status`/`reason_code` 非空计数全为 0），机制一判不了。**WOD 本地只有 4 列 906 行**，29 个 M2 特征全 MISSING，需重做脱敏投影且触及 RQ014 致盲边界（PI 裁定本轮放弃）。**OnSite 可行**：67,861 行 AV 锚点、29 个 M2 特征一个不缺、类别取值 100% 被 InterHub 覆盖，缺的只有 materializer。**另查实 RQ016 的 envelope 里 10.9009%（69,288/635,618）的目标值是自动驾驶车自己的 IPV** | `reports/studies/RQ016B_wod_onsite_feasibility/RQ016B_1_feasibility_20260804T001351Z_7480c173/` | `reports/knowledge/RQ016B_wod_onsite_feasibility/` | **尚无 `decision.md`**；WOD 判定只针对本地脱敏产物；**无同源迁移证据**（RQ009 LODO 4 个留出源均不含 OnSite 与该 WOD 产物，90% coverage 波动 0.7484–0.9921）；`apet_online_proxy` 填充率 OnSite 7.90% vs InterHub 40.26%；描述性结果，禁用 estimability 表述 |
| RQ016C | 只用纯人-人样本构建供 OnSite 使用的参照 envelope（PI 2026-08-04 裁定：envelope 是查询机制，不同目标可建不同 envelope） | **executed 2026-08-04** — 监督方已独立复算放行。参照池 2,442,625 行（dev 1,752,509 + guard 690,116，held_out 0）；90% coverage 0.898038（414,837/461,937）、平均宽度 1.238468、机制二弃权 5.0801%（24,723/486,660）。**产物已在真实 OnSite 全量 67,861 行上跑通打分（只加载不重拟）**，支持门通过 **21,936/67,861 = 32.3249%** | `reports/studies/RQ016C_human_only_envelope/RQ016C_1_human_only_envelope_20260804T005716Z_7480c173/` | `reports/knowledge/RQ016C_human_only_envelope/` | **尚无 `decision.md`**；打分演练**不构成对任何一辆自动驾驶车的判定**（OnSite 无机制一判据）；特征集较 RQ009 M2 移除 `agent_type_pair`/`av_included`/`vehicle_type_list` 三列；与 RQ009 已发表数**不构成复现关系**；无同源迁移证据；OnSite 有 7 行坐标系异常（`relative_distance_anchor` ≈ 570,762 m）待处理；模型本体 164 MB 未入库（sha256 `bc25302b…`） |
| PAPER001 | Manuscript context | reference | n/a | `reports/knowledge/PAPER001_online_sociality_verification_manuscript/` | historical paper context |
| PAPER002 | Dynamic-IPV v4.1 evidence architecture | writing; verified paper `main` at `c6783577` | n/a | `reports/knowledge/PAPER002_dynamic_ipv_evidence_architecture/` | active `structure.md`/claims-register baseline |

## Current PI Decisions (2026-06-24)

- Launch RQ009 using the new plan at
  `reports/plans/RQ009_plan_v0_dynamic_counterpart_conditioned_envelope_20260624.md`.
- Keep RQ007 held-out sealed until RQ009 reaches its independently reviewed pre-opening freeze;
  opening requires another explicit PI authorization.
- Do not run RQ008B at present.
- Defer RQ012 two-human annotation; keep `BLOCKED_FOR_HUMAN_LABELS`.
- Authorize WOD-E2E signed-in manifest/pilot work in principle; account/licence/login remains a
  user action.
- Prioritize OnSite RQ011B after RQ009 freezes; WOD proceeds in parallel.
- Use the paper-repository `main` merge `c6783577` as the current v4.1 manuscript baseline.

## Status Vocabulary

| Status | Meaning |
|---|---|
| planning | Research question is being scoped. |
| approved | PI has authorized launch; independent plan review is the first gate. |
| running | Execution is in progress. |
| review | Execution reports exist and need synthesis/review. |
| accepted | Accepted claims are frozen in `reports/knowledge/<RQ>/decision.md`. |
| writing | Verified manuscript baseline is active and being updated. |
| done | Paper-side work is complete. |
| archived-review | Preserved for traceability; not an active headline result. |
| blocked | Missing data, authority, or design decision prevents progress. |
| reference | Context/archive only; not an active claim decision. |

## Boundary Rule

`reports/` has three governed first-level directories: `plans/`, `studies/`, and `knowledge/`.
Large derived data lives under `data/derived/`; report process archives and local agent state live
under `archived/report_process/` and `archived/report_local_state/`.
