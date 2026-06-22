# Research Plans

This directory is the centralized planning layer for active and proposed research questions.
It contains plans only; execution artifacts remain under `reports/studies/`, and reviews,
synthesis, and accepted/rejected claims remain under `reports/knowledge/`.

## Wave A plans and execution prompts

| RQ | Topic | Status | Plan | Main-agent prompt |
|---|---|---|---|---|
| RQ007 | Interaction-conditioned IPV estimability | planning | `RQ007_plan_v0_interaction_conditioned_ipv_estimability_20260622.md` | `prompts/RQ007_prompt_claude_codex_orchestration_20260622.md` |
| RQ008 | InterHub temporal IPV discovery | planning | `RQ008_plan_v0_interhub_temporal_ipv_discovery_20260622.md` | `prompts/RQ008_prompt_claude_codex_orchestration_20260622.md` |
| RQ010 | WOD-E2E data and tracking feasibility | planning | `RQ010_plan_v0_wod_e2e_tracking_feasibility_20260622.md` | `prompts/RQ010_prompt_claude_codex_orchestration_20260622.md` |
| RQ011 | OnSite full-universe and run-level readiness | planning | `RQ011_plan_v0_onsite_full_universe_readiness_20260622.md` | `prompts/RQ011_prompt_claude_codex_orchestration_20260622.md` |
| RQ012 | OnSite event ontology and blind-annotation readiness | planning | `RQ012_plan_v0_onsite_event_annotation_readiness_20260622.md` | `prompts/RQ012_prompt_claude_codex_orchestration_20260622.md` |

Prompt index: `prompts/README.md`.

## Planning rules

- A `v0` plan is a scoped proposal, not a frozen confirmatory specification.
- Each plan must define inputs, denylisted outcomes, work packages, gates, deliverables,
  stop conditions, and claim boundaries.
- Discovery and confirmation must be separated whenever exploratory freedom is allowed.
- A plan becomes `approved` only after independent review and an explicit status update.
- Execution runs use unique versioned directories under `reports/studies/<RQ>/`.
- Final reader-facing execution reports are offline HTML; formal figures use the Nature skill.
- Null, failed, blocked, and reverse findings must be retained.
- The program dashboard is `reports/knowledge/RQ_PROGRESS_DASHBOARD.md`.
