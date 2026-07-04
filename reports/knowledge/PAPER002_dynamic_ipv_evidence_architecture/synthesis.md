# PAPER002 Architecture Synthesis

Status: `writing`. Architecture-level summary across accepted RQ evidence. Canonical claims live in each RQ `decision.md`; this is a narrative integration, not a new claim.

## Spine

The v4.1 manuscript argues an **online, estimability-aware, context-conditioned dynamic-IPV verifier** and validates it on external surfaces:

1. **Estimability gate (RQ007, accepted).** Interaction-conditioned estimability defines where current-IPV is reliably measurable; development/guard claims carry a proximity-bounded caveat.
2. **Dynamic envelope (RQ009, accepted; R3).** An estimability-aware, context-conditioned split-conformal envelope is sharp and near-nominal (width −42.3%, Winkler −35.6%, coverage ≈0.899, abstention 4.78% at 90%). It is presented as context-conditioned; the IPV-conditioning channels were null internal ablations and are not manuscript claims.
3. **OnSite validity (RQ011/RQ011B/RQ012, first external priority).** Matched-scenario readiness is frozen at `algorithm×scenario` with `full_300` outcomes and `clean_285` replay (T19 replay-excluded); consequence evidence uses automatic events + official outcomes (human blind annotation deprecated).
4. **WOD-E2E human-preference validity (RQ010 → RQ010B, parallel).** Feasibility/route only so far (`T2_FULL_TRACKING_REQUIRED`, Route 4); carries the human-alignment leg jointly with InterHub.
5. **Beyond-safety increment (RQ013, planning).** Incremental value relative to prespecified safety/kinematic baselines — not yet executed.

## Standing caveats for the manuscript

InterHub σ=0.1 setting; empirical runtime monitor, not a formal proof; OnSite run-level effects not identifiable; WOD-E2E not yet a completed human-preference validation. RQ008 forbids inheriting unconfirmed temporal-IPV motifs as laws.

Sources: `reviews/chatgpt_review_wave_b.md`; constituent RQ `decision.md` files; `../RQ_PROGRESS_DASHBOARD.md`.
