# PAPER002: Dynamic-IPV v4.1 Evidence Architecture

Status: `writing` (registry stage `S8 Paper handoff`).

This folder is the **paper-architecture** layer for the dynamic-IPV v4.1 manuscript:
it tracks how accepted RQ evidence is assembled into the manuscript spine. It is
**not** a claim ledger — the canonical claim records remain each RQ's `decision.md`.
If anything here conflicts with an RQ `decision.md`, the RQ decision controls.

## Evidence chain

```text
online IPV time-series
→ interaction-conditioned estimability                 (RQ007, accepted)
→ estimability-aware dynamic context-conditioned envelope (RQ009, accepted; R3)
→ OnSite matched-scenario validity                     (RQ011/RQ011B/RQ012, first external priority)
→ WOD-E2E human-preference validity                    (RQ010 → RQ010B, parallel engineering path)
→ incremental value vs prespecified safety/kinematic baselines (RQ013, planning)
```

RQ008 is a negative temporal-discovery boundary, not a positive link in this chain.

## Constituent records (canonical)

- `../RQ007_interaction_conditioned_ipv_estimability/decision.md`
- `../RQ009_dynamic_counterpart_conditioned_envelope/decision.md`
- `../RQ010_wod_e2e_tracking_feasibility/decision.md`
- `../RQ011_onsite_full_universe_readiness/decision.md`
- `../RQ011B_onsite_moment_monitor/` (registered close-out)
- `../RQ012_onsite_event_annotation_readiness/decision.md`

## Files here

- `synthesis.md` — cross-RQ architecture narrative.
- `decision.md` — which accepted claims enter the v4.1 spine (pointer ledger).
- `report_index.md` — index of architecture-level reviews and inputs.
- `reviews/` — cross-RQ paper-architecture reviews.
