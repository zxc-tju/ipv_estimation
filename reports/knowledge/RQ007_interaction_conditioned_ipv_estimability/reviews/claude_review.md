# Claude Code Review

Status: filed (2026-06-24)

Reviewer role: research reviewer / repository integrator.

Run reviewed: `RQ007_1_ipv_estimability_20260622T155229Z_289d9a99` (overall `COMPLETE`; all gates PASS).
Reader entry: `reports/studies/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/90_report/index.html`.

## Verdict

Concur with the registered ACCEPTED decision (C1 MODERATE-with-strong-boundaries; C2/C3 SUPPORTED). The
key calibration is correct and well supported: the headline interaction-aligned concentration gap
(~ -0.13) is **not** wholly interaction-specific. Most of it (~ -0.096) is reproduced by a nearby
non-conflicting actor (proximity/history), and only a small conflict-geometry-specific residual
(-0.032 to -0.036, case-clustered CIs excluding zero) survives. The study resisted the tempting
over-claim ("time-locked interaction effect") and shrank to a defensible proximity-bounded residual.
That discipline is the main reason the claim is trustworthy.

## Key Findings

| Check | Result | Reading |
|---|---|---|
| Total gap (dev/guard) | -0.132 / -0.129 (replicated -0.134 / -0.133) | Real and reproducible, but composite. |
| Time-shift control | +0.006 | Effect is time-locked, not an alignment artifact. |
| Counterpart permutation / re-est. switch | +0.021 / +0.122 | Counterpart-specific, not arbitrary pairing. |
| Nearby non-conflicting (proximity) | ~ -0.096 | Majority of the gap is proximity, not conflict geometry. |
| Conflict-geometry residual | -0.032 to -0.036, CI excludes 0 | Small but nonzero specific increment — the actual claim. |
| C2 estimability ≠ settling | low-index \|dθ\| ~0.30/0.31; high-index ~0.17 | Estimable ≠ stable; high index ≠ IPV 0. |
| C3 episode-summary dependence | ~0.26 rad diff, ~22% sign flips | "Episode IPV" is not definition-free. |

## Boundaries And Watch-Items (confirmed)

- Development/guard only; the sealed held-out split was never opened. All claims are pre-confirmation.
- "Estimability" is the estimator **concentration index** (an identifiability proxy), not a standard
  deviation; do not read it as IPV uncertainty in physical units.
- Estimator-input reruns are **sanity checks only** (recompute mismatch mean ~0.11, p95 ~0.45). They must
  not be cited as rigorous proof; the robust evidence is the 26/26 analysis-level perturbations.
- The lifecycle (concentration min near resolution; ~44–46% of onsets before opportunity) is descriptive,
  not causal precedence — consistent with RQ008's negative temporal-discovery boundary.
- Single InterHub setting; no map/lane, PET, intensity, order, priority, or outcome fields used.

## Reproducibility / Process Assessment

- Identity gates verified; plan SHA-256 pinned. Independent review PASS (0 blocking, 0 major, 2 minor).
  Red team PASS, explicitly forcing the shrinkage to the proximity-bounded residual. Replication PASS/MIXED
  (headline gaps reproduce; minor window/bin and rounded-τ differences only).
- Claim–evidence matrix rows all `verified=true`; figure source data staged. Process is auditable.

## Supporting Role For The Program / Manuscript

- Underwrites the v4.1 **estimability contract**: IPV is not equally interpretable at every timestamp;
  separate interaction opportunity, estimability, human-reference support, and deviation; never read high
  uncertainty as neutral IPV.
- Feeds RQ009 (dynamic counterpart-conditioned envelope) as the valid-window/estimability input.
- Manuscript-safe wording is in `synthesis.md`. Do not inflate the small conflict-geometry residual, do not
  describe the lifecycle as causal, and keep all numbers provisional until held-out confirmation.

## Recommendation

Accept as registered. Use only the C1/C2/C3 frozen claims with the proximity-bounded caveat attached.
Re-open at the held-out confirmation stage under the frozen contract before any timestamp-level estimability
claim is hardened for submission.

## Source Pointers

- `.../11_final_review/conclusions.md`, `.../11_final_review/claim_evidence_matrix.csv`
- `.../05_controls/controls_results.csv`, `.../09_red_team/red_team_cluster_control_probes.csv`
- `.../10_replication/replication_compare.csv`, `01_results/figure_manifest.csv`
- knowledge: `decision.md`, `synthesis.md`
