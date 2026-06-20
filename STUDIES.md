# Research Study Index

This repository is the source of truth for IPV estimation research, execution
reports, review records, and evidence decisions. Manuscript text lives in the
separate paper repository; durable knowledge lives here.

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
| RQ001 | Online IPV interval deployability | review | `reports/studies/RQ001_online_ipv_interval/` | `reports/knowledge/RQ001_online_ipv_interval/` | verifier method/results |
| RQ002 | Self-anchor as group norm | review | `reports/studies/RQ002_self_anchor_group_norm/` | `reports/knowledge/RQ002_self_anchor_group_norm/` | verifier validity/limitations |
| RQ003 | NSFC external evidence | review | `reports/studies/RQ003_nsfc_external_evidence/` | `reports/knowledge/RQ003_nsfc_external_evidence/` | external validation |
| RQ004 | IPV state-space conclusions | review | `reports/studies/RQ004_ipv_state_space/` | `reports/knowledge/RQ004_ipv_state_space/` | state-space / norm framing |
| RQ005 | NMI draft evidence gap | review | `reports/studies/RQ005_nmi_evidence_gap/` | `reports/knowledge/RQ005_nmi_evidence_gap/` | narrative risk review |
| RQ006 | Sigma sensitivity | archived-review | `reports/studies/RQ006_sigma_sensitivity/` | `reports/knowledge/RQ006_sigma_sensitivity/` | robustness appendix |
| PAPER001 | Manuscript context | reference | n/a | `reports/knowledge/PAPER001_online_sociality_verification_manuscript/` | paper-side claim map and drafts |

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

## Boundary Rule

`reports/` intentionally has only two first-level directories: `studies/` and
`knowledge/`. Large derived data lives under `data/derived/`; report process
archives and local agent state live under `archived/report_process/` and
`archived/report_local_state/`.
