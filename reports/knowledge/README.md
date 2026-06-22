# Research Knowledge

This is the interpretation layer of the research knowledge base. It is now the
shared home for ChatGPT, Claude Code, Codex, and human review records.

## Program Dashboard

The program-level status, dependencies, blockers, and next gates for all active
and planned research questions are tracked here:

- [`RQ_PROGRESS_DASHBOARD.md`](RQ_PROGRESS_DASHBOARD.md) — human-readable board.
- [`rq_progress_registry.csv`](rq_progress_registry.csv) — machine-readable status registry.

Update both files whenever an RQ changes program status, stage, blocker, latest
artifact, or next action. An RQ may be marked `accepted` only when its
`decision.md` freezes the accepted/rejected/deferred claim slate.

Each RQ has exactly one knowledge folder, and the folder stem matches the
corresponding execution folder in `reports/studies/`.

```text
reports/studies/RQ002_self_anchor_group_norm/
reports/knowledge/RQ002_self_anchor_group_norm/
```

## Standard RQ Files

- `README.md`: question, scope, and current state.
- `report_index.md`: all execution reports for the RQ.
- `reviews/`: Claude/GPT/Codex/human review notes when available.
- `synthesis.md`: consolidated interpretation across report versions.
- `decision.md`: accepted, rejected, and deferred claims.

## Manuscript Context

The former paper-repository `knowledge/` folder has been moved here:

`reports/knowledge/PAPER001_online_sociality_verification_manuscript/imported_from_paper_repo_20260620/`

The paper repository should not recreate a local `knowledge/` directory. Paper
agents should read manuscript context here, then edit only manuscript files in
the paper repo.
