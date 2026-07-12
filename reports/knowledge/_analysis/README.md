# Cross-RQ Analysis Notes

Engineering / analysis notes that do not belong to a single research question. Moved
here from the `knowledge/` root on 2026-06-30 (see `../KNOWLEDGE_GOVERNANCE_PLAN_20260630.md`).

These are interpretation-layer notes. Raw machine artifacts (`.json`) ideally belong in
`reports/studies/`; they are kept here next to their notes for now and may be promoted to
`studies/` later (the "strict" upgrade).

## Inventory

| File(s) | Topic | Inbound refs within `knowledge/` (2026-06-30) |
|---|---|---|
| `INFRA_hpc_tongji_reuse.md` | Tongji HPC reuse / infra notes | none — **orphan**; confirm whether to keep or archive |
| `ipv_accel_hyperparam.json` + `ipv_accel_hyperparam_finding.md` | IPV acceleration hyperparameter finding | 1 |
| `ipv_estimator_api_map.json` + `ipv_estimator_api_map.md` | IPV estimator API surface map | none — **orphan**; confirm whether to keep or archive |
| `ipv_estimator_divergence.json` + `ipv_estimator_divergence_investigation.md` | IPV estimator divergence investigation | 1 |
| `PROGRAM_REVIEW_20260707_claude.md` | Program-level progress review + prioritized recommendations (Claude, 2026-07-07); synthesis only, no new claims | new (2026-07-07) |

"Inbound refs" counts only references from other files under `reports/knowledge/`; a file
may still be referenced elsewhere in the repo. Orphans are retained pending a keep-or-archive
decision, not deleted.

## Convention

Name notes `<topic>.md`, with an optional same-stem `<topic>.json`. Add new cross-RQ notes here
rather than at the `knowledge/` root.
