# RQ007: Interaction-Conditioned IPV Estimability

Status: study final review PASS (development/guard; held-out sealed); knowledge `decision.md` missing

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
`report_index.md` for the execution package, and `reviews/codex_review.md` for the current
knowledge-layer gap.

This knowledge folder is the single synthesis point for RQ007. Do not treat the claims as
paper-frozen until a knowledge-layer `decision.md` exists. Do not use sealed/held-out, PET,
intensity, order, priority, or outcome fields as claim sources.
