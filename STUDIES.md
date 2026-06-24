# Research Study Index

This repository is the source of truth for IPV estimation research, execution
reports, review records, and evidence decisions. Manuscript text lives in the
separate paper repository; durable knowledge lives here.

## Program Progress Dashboard

Use the following files to synchronize program-level progress, dependencies,
blockers, latest artifacts, and next gates:

- `reports/knowledge/RQ_PROGRESS_DASHBOARD.md`
- `reports/knowledge/rq_progress_registry.csv`

`STUDIES.md` remains the compact study index. The dashboard carries the more
operational status view and must not override an accepted RQ `decision.md`.

## Two-Layer Knowledge Base

- `reports/studies/`: execution layer. Each research question has one folder,
  and each independent report run uses a versioned directory such as
  `RQ001_1_current_ipv_distribution_20260618`.
- `reports/knowledge/`: interpretation layer. Each research question has one
  corresponding knowledge folder with the same RQ stem, for example
  `RQ001_online_ipv_interval`.

One RQ may have multiple execution reports, especially when several agents run
in parallel. One RQ should have only one knowledge folder; that folder
synthesizes the multiple reports into accepted, rejected, or deferred claims.

## Naming Contract

```text
reports/studies/RQ001_online_ipv_interval/
  RQ001_1_current_ipv_distribution_20260618/
  RQ001_2_interval_query_20260618/
  RQ001_3_online_interval_lock_20260619/

reports/knowledge/RQ001_online_ipv_interval/
  report_index.md
  synthesis.md
  decision.md
```

The RQ stem links reports and knowledge. The suffix after the underscore in an
execution folder is the execution version number, not a new research question.

## Study Index

| RQ | Topic | Status | Execution layer | Knowledge layer | Paper use |
|---|---|---|---|---|---|
| RQ001 | Online IPV interval deployability | review | `reports/studies/RQ001_online_ipv_interval/` | `reports/knowledge/RQ001_online_ipv_interval/` | legacy interval engineering / M4 ablation boundary |
| RQ002 | Self-anchor as group norm | review | `reports/studies/RQ002_self_anchor_group_norm/` | `reports/knowledge/RQ002_self_anchor_group_norm/` | verifier validity/limitations; reject self-anchor-only normative authority |
| RQ003 | NSFC external evidence | accepted | `reports/studies/RQ003_nsfc_external_evidence/` | `reports/knowledge/RQ003_nsfc_external_evidence/` | external validation -- Tier B; top-five; no robust increment |
| RQ004 | IPV state-space conclusions | review | `reports/studies/RQ004_ipv_state_space/` | `reports/knowledge/RQ004_ipv_state_space/` | episode-level state organization / norm framing |
| RQ005 | NMI draft evidence gap | review | `reports/studies/RQ005_nmi_evidence_gap/` | `reports/knowledge/RQ005_nmi_evidence_gap/` | leakage and claim-governance boundary |
| RQ006 | Sigma sensitivity | archived-review | `reports/studies/RQ006_sigma_sensitivity/` | `reports/knowledge/RQ006_sigma_sensitivity/` | robustness appendix |
| RQ007 | Interaction-conditioned IPV estimability | planning | `reports/studies/RQ007_interaction_conditioned_ipv_estimability/` | `reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/` | online IPV measurement and valid-window contract |
| RQ008 | InterHub temporal IPV discovery and confirmation | planning | `reports/studies/RQ008_interhub_temporal_ipv_discovery/` | `reports/knowledge/RQ008_interhub_temporal_ipv_discovery/` | within-interaction temporal organization |
| RQ009 | Dynamic counterpart-conditioned human envelope | planning | `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/` | `reports/knowledge/RQ009_dynamic_counterpart_conditioned_envelope/` | main dynamic verifier (M3); M4 self-history ablation |
| RQ010 | WOD-E2E tracking feasibility and human preference validity | feasibility-audit complete | `reports/studies/RQ010_wod_e2e_tracking_feasibility/` | `reports/knowledge/RQ010_wod_e2e_tracking_feasibility/` | independent human-preference validity |
| RQ011 | OnSite full-universe and run-level readiness | readiness complete (re-run on complete data); `READY_WITH_FROZEN_EXCLUSIONS` — outcome universe full 300 / replay 285 (T19 excluded); run-level & repeated-run not identifiable by design; [RQ011A report](reports/studies/RQ011_onsite_full_universe_readiness/RQ011_2_onsite_readiness_20260623T201415+0800_efdd75a5/90_report/index.html) (supersedes suspended RQ011_1 incomplete-data run) | `reports/studies/RQ011_onsite_full_universe_readiness/` | `reports/knowledge/RQ011_onsite_full_universe_readiness/` | matched-scenario algorithm validity readiness |
| RQ012 | OnSite event ontology and blind-annotation readiness | Wave-A readiness complete; `BLOCKED_FOR_HUMAN_LABELS`; [RQ012A report](reports/studies/RQ012_onsite_event_annotation_readiness/RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37/90_report/index.html) | `reports/studies/RQ012_onsite_event_annotation_readiness/` | `reports/knowledge/RQ012_onsite_event_annotation_readiness/` | realised interaction consequence / blind behaviour reference |
| RQ013 | Beyond-safety incremental validity | planning | `reports/studies/RQ013_beyond_safety_incremental_validity/` | `reports/knowledge/RQ013_beyond_safety_incremental_validity/` | final incremental utility relative to safety/kinematic baselines |
| PAPER001 | Manuscript context | reference | n/a | `reports/knowledge/PAPER001_online_sociality_verification_manuscript/` | paper-side claim map and drafts |
| PAPER002 | Dynamic-IPV v4 evidence architecture | planning | n/a | `reports/knowledge/PAPER002_dynamic_ipv_evidence_architecture/` | structure v4, claims register, figure/evidence map |

## Status Vocabulary

| Status | Meaning |
|---|---|
| planning | Research question is being scoped. |
| approved | Plan is approved and ready for execution. |
| running | Execution is in progress. |
| review | Execution reports exist and need synthesis/review. |
| accepted | Accepted claims are frozen in `reports/knowledge/<RQ>/decision.md`. |
| writing | Accepted claims are being applied in the paper repo. |
| done | Paper-side work is complete. |
| archived-review | Preserved for traceability; not an active headline result. |
| blocked | Missing data, authority, or design decision prevents progress. |
| reference | Context/archive only; not an active claim decision. |

## Boundary Rule

`reports/` intentionally has only two first-level directories: `studies/` and
`knowledge/`. Large derived data lives under `data/derived/`; report process
archives and local agent state live under `archived/report_process/` and
`archived/report_local_state/`.
