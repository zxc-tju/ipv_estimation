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
| RQ007 | Interaction-conditioned IPV estimability | accepted; dev/guard boundary, held-out sealed | `reports/studies/RQ007_interaction_conditioned_ipv_estimability/` | `reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/` | v4.1 estimability contract |
| RQ008 | InterHub temporal IPV discovery | accepted negative boundary; RQ008B not authorized | `reports/studies/RQ008_interhub_temporal_ipv_discovery/` | `reports/knowledge/RQ008_interhub_temporal_ipv_discovery/` | no positive temporal law from RQ008A |
| RQ009 | Estimability-aware dynamic counterpart-conditioned envelope | accepted (R3 context-conditioned conformal envelope; IPV-conditioning channels internal null) | `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/` | `reports/knowledge/RQ009_dynamic_counterpart_conditioned_envelope/` | R3 context-conditioned envelope; conditioning channels internal ablation |
| RQ010 | WOD-E2E tracking feasibility and preference validity | feasibility accepted; **RQ010B COMPLETE — bounded NULL** (2026-07-03): candidate IPV does not predict human preference and is not comparable to physics; M3 does not transfer to WOD-E2E | `reports/studies/RQ010_wod_e2e_tracking_feasibility/` | `reports/knowledge/RQ010_wod_e2e_tracking_feasibility/` | registered external-validity boundary (bounded null); manuscript R4 WOD-E2E leg = negative, not a positive claim |
| RQ011 | OnSite full-universe readiness | accepted `READY_WITH_FROZEN_EXCLUSIONS` | `reports/studies/RQ011_onsite_full_universe_readiness/` | `reports/knowledge/RQ011_onsite_full_universe_readiness/` | OnSite universe/scope decision |
| RQ011B | OnSite moment-level IPV monitor validity | closed-out `PROVISIONAL_NULL / UNDER_IDENTIFIED` (measurement-limited; not a frozen manuscript claim) | `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/` | `reports/knowledge/RQ011_onsite_full_universe_readiness/` | not demonstrated on OnSite pending adequate interaction-failure segment retrieval/segmentation |
| RQ012 | OnSite automatic-event harm (scope revised: automatic events + official outcomes; human labels deprecated) | accepted; RQ012B COMPLETE — deviation→harm BOUNDED/NULL across the full behavioural battery (no IPV-specific channel; passivity→deadlock unconfirmed hint); [RQ012B report](reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/00_entry/index.html) | `reports/studies/RQ012_onsite_event_annotation_readiness/` | `reports/knowledge/RQ012_onsite_event_annotation_readiness/` | realised interaction consequence (bounds R5; bounded/null) |
| RQ013 | Beyond-safety incremental validity | planning | `reports/studies/RQ013_beyond_safety_incremental_validity/` | `reports/knowledge/RQ013_beyond_safety_incremental_validity/` | final baseline-relative utility |
| RQ014 | WOD-E2E lost rating↔IPV-deviation result recovery | v1.5 FORMAL_G1_PASS; v1.6 Lead/Sub-Agent execution handoff frozen; HPC managed checkout sync → immutable export spec → validate-only → rating-blind export are next; no empirical run yet; first user decision is D1 after export PASS | `reports/studies/RQ014_wod_e2e_rating_recovery/` | `reports/knowledge/RQ014_wod_e2e_rating_recovery/` | no manuscript claim; preflight, 960-cell feature bank, rating join, clean replay and claim acceptance remain separately gated |
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
