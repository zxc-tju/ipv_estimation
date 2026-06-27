# RQ009: Estimability-Aware Dynamic Counterpart-Conditioned Human Envelope

Question: can the primary M3 model (`causal context + counterpart current IPV`) provide a
calibrated, selective human current-IPV envelope among active and estimable interaction windows?

Plan:
`reports/plans/RQ009_plan_v0_dynamic_counterpart_conditioned_envelope_20260624.md`

Main-agent prompt:
`reports/plans/prompts/RQ009_prompt_claude_codex_orchestration_20260624.md`

## Executions

| Version | Report package | Role | Status |
|---|---|---|---|
| — | — | — | Not started; PI authorized launch on 2026-06-24. Independent plan review is the first gate. |

## Execution boundary

The first execution must create a unique atomically locked run directory. It may complete plan
review, measurement audit, split/feature freeze, M0–M5 implementation, conformal calibration,
negative controls, and pre-opening review. It must not access the RQ007 sealed split without a new
explicit PI authorization. Until then the strongest permissible execution state is
`READY_FOR_SEALED_TEST`.
