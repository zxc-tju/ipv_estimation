# Wave A Claude → Codex Orchestration Prompts

These files are standalone prompts for the main Claude agent. Claude is an
orchestrator only; Codex CLI is the sole executor.

| RQ | Plan | Prompt |
|---|---|---|
| RQ007 | `../RQ007_plan_v0_interaction_conditioned_ipv_estimability_20260622.md` | `RQ007_prompt_claude_codex_orchestration_20260622.md` |
| RQ008 | `../RQ008_plan_v0_interhub_temporal_ipv_discovery_20260622.md` | `RQ008_prompt_claude_codex_orchestration_20260622.md` |
| RQ010 | `../RQ010_plan_v0_wod_e2e_tracking_feasibility_20260622.md` | `RQ010_prompt_claude_codex_orchestration_20260622.md` |
| RQ011 | `../RQ011_plan_v0_onsite_full_universe_readiness_20260622.md` | `RQ011_prompt_claude_codex_orchestration_20260622.md` |
| RQ012 | `../RQ012_plan_v0_onsite_event_annotation_readiness_20260622.md` | `RQ012_prompt_claude_codex_orchestration_20260622.md` |

## Shared execution contract

Every prompt requires:

- Claude only decomposes, invokes Codex CLI, evaluates Codex status reports, and reports progress.
- All repository inspection, coding, analysis, testing, review, plotting, and report generation are performed by Codex.
- Each execution creates a unique atomically locked `RUN_ID` and `RUN_ROOT`.
- Worker roles are separated; implementers cannot act as independent reviewers, red teams, or replication workers.
- Each Codex task uses the fixed contract: `ROLE`, `OBJECTIVE`, `INPUTS`, `READ_SCOPE`, `WRITE_SCOPE`, `DENYLIST`, `TASKS`, `DELIVERABLES`, `ACCEPTANCE_CRITERIA`, `NON_GOALS`, `STOP_CONDITIONS`, and `RETURN_FORMAT`.
- Final reader-facing output is an offline HTML report at `${RUN_ROOT}/90_report/index.html`.
- Every reader-facing figure must be generated using the Nature skill; no silent plotting fallback is permitted.
- Null, reverse, failed, and blocked findings remain visible and are recorded in `tried.md`.
- Git safety rules prohibit destructive resets, cleans, force pushes, overwriting uncommitted work, and mixing research-repository and paper-repository edits.
