# Claude Code Review — RQ010B (WOD-E2E human-preference validity)

Status: filed (2026-07-03)

Reviewer role: research reviewer / repository integrator.

Run reviewed: `RQ010B_1_tracking_preference_20260625T201647+0800_695fa83f`, Phase-3 reframed preference test (`02_process/05_reframed_preference_results/`).
Predictor per PI reframe: **magnitude of IPV deviation from the context-conditioned envelope** (not raw IPV), within-segment across the 3 rated candidates.

## Verdict

Concur with the run's own `FAIL_BOUND_OR_NULL` across all three schemes. On the **full rated WOD-E2E set**, IPV
deviation from the human envelope does **not** predict human preference, and is **not IPV-specific** — simple
kinematics predict preference better than the deviation signal. This is a clean, well-executed negative, not a
pipeline artifact.

## Key findings

| Scheme (N segments) | Primary \|deviation\| effect | IPV-specificity | Verdict |
|---|---|---|---|
| S1 future-only (75) | Spearman ρ=0.148 (p=0.10), **wrong sign**; BT β(smaller-pref)=−0.67 (p=0.124); hit-rate −0.04 (perm p=0.76) | IPV strength 0; best control `min_ttc` 0.165 (also `min_gap`); **IPV beats all controls = False**; max-stat p=1 | FAIL |
| S2 hist+future ≥1s (98) | ρ=0.031 (p=0.69) | best control `curvature` 0.255 > IPV | FAIL |
| S2 ≥2s (66) | ρ=0.00 (p=1.0) | best control `jerk` 0.242 > IPV | FAIL |

Raw signed IPV is also null (ρ≈−0.02/0.07); the inverted-U/neutral-IPV shape term is non-significant. The
pre-registered PASS rule (|ρ|≥0.10 & p<0.05, IPV beats every kinematic/shuffle control, max-stat + advantage
permutation p<0.05) is met on **no** count.

## Boundaries / measurement caveat

- Ran on the **full 479 rated segments** (not a subset), so the null is not data-availability-limited; the
  effective N (75–98) is set by the estimability filter, not by missing data.
- **Measurement weakness:** candidate IPV (finite-differenced from 4 Hz positions) is rate-unstable — 4 Hz vs
  10 Hz IPV correlate only ≈0.29–0.31 — which independently caps how much signal a candidate IPV could carry.
- Open-loop opportunity structure (shared frozen counterpart); not closed-loop or realised harm.

## Reproducibility / process

Rating join is verified: 479 rated segments, 477 with exactly one scored frame, candidate hash-alignment
**1428/1428 (0 mismatch)** — the null is not a join error. M3 transfer outputs were not read (ratings-blind
until the pre-registered test). Fixed-seed within/across-segment shuffle controls + within-segment rating
permutation with a max statistic. The reframe (deviation-primary, raw-IPV as control) was implemented as
specified.

## Manuscript role — NOT in the paper (PI decision 2026-07-03)

Per PI, the R4 / WOD human-preference-validity result is **not featured in the manuscript for now** — kept as
an internal null (same treatment as the RQ009 counterpart-channel null). Do not write R4 as a claim, positive
or negative. Consistent with the program pattern: no external validation leg (WOD preference R4, OnSite
consequence R5) shows IPV-specific validity over kinematics/context; what survives is R1 (state-dependence),
the estimability framing (RQ007), and R3 (the sharp, calibrated context-conditioned envelope as a monitor).

## Recommendation

Hold as an internal null. If ever revisited, it needs IPV-estimation-quality work (rate stability), window
redefinition, and a WOD-domain estimability/OOD treatment — high risk and unlikely to flip given kinematics
already dominate at this N. Do not upgrade by secondary patterns.

## Source pointers

- `02_process/05_reframed_preference_results/phase3_preference_report.md`, `phase3_effect_summary.csv`, `phase3_controls_summary.csv`
- `.../10hz_sensitivity/tenhz_sensitivity_summary.json`; `rating_join_verification.md`
