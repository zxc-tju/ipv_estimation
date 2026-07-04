# RQ010 Decision: WOD-E2E Tracking Feasibility

Status: ACCEPTED feasibility/route boundary; **RQ010B AUTHORIZED by PI 2026-06-24** (signed-in pilot + build multi-camera tracker (Route 4 preferred) + HPC approved). Not behavioural-validity evidence yet.

Run ID: `RQ010_1_wod_tracking_feasibility_20260623T073830+0800_14f21d3e` (feasibility-only)
Plan SHA-256: `d9988309a803dc443cfd2c9bcde2f75e078c8cd87e57a036828f0a13e1778d87`
Basis: independent review PASS_WITH_FIXES; red team reverify CLEAR; final review PASS; `reviews/claude_review.md` + `reviews/codex_review.md`; PI authorization 2026-06-24.

## PI Decision (2026-06-24): authorize RQ010B

WOD-E2E is now a priority external-validation surface (it carries the human-alignment leg jointly with InterHub after RQ012 human annotation was dropped). Authorized: (1) signed-in WOD-E2E pilot to resolve official sizes/shards/throughput; (2) build the Route 4 multi-camera 3D/BEV tracker (Route 5 fallback); (3) commit HPC for the tracking pipeline. Keep all work ratings-blind.

## Accepted Claims (feasibility)

| ID | Claim |
|---|---|
| RQ010-KC-T2 | The public WOD-E2E schema exposes ego/camera/preference context but no surrounding-actor tracks → `T2_FULL_TRACKING_REQUIRED`; T0/T1 rejected (no verified WOMD/WOD crosswalk). |
| RQ010-KC-ROUTE | Route 4 (adapt multi-camera 3D/BEV tracker) preferred; Route 5 (custom) fallback. |
| RQ010-KC-ACCESS | Gate 010-0 PASS: non-commercial research/publication path exists (production/vehicle use forbidden). |
| RQ010-KC-PROTOCOL | RQ010B is a shared open-loop, ratings-blind opportunity structure with explicit abstention; candidate-conditioned forecasting is sensitivity-only; predicted responses ≠ realised harm. |

## Open Risks Carried Into RQ010B (not resolved by authorization)

Map conflict geometry MISSING (M3 risk — needs RQ009 fallback); critical-frame index within the 20 s run unverified; official sizes were sign-in gated (pilot resolves); camera-only 3D depth error (tracker accuracy risk); HPC scale only bounded by derived estimate (0.15–17 TB full / 0.1–11 GB pilot). RQ009 interface must be frozen before the M3 preference test.

## Paper Handoff

Still feasibility/route until the pilot + tracker deliver; `\externalpending{R4}` in the manuscript. Do not cite WOD-E2E as a completed human-preference validation until RQ010B produces results.

---

## RQ010B Reframed Preference-Validity — DECISION 2026-07-03: bounded NULL (registered negative)

Run: `RQ010B_1_tracking_preference_20260625T201647+0800_695fa83f` (reframed preference analysis).
Pre-registration: `02_process/04_reframed_preference_prereg/PREREGISTRATION_20260630.md`.
Reader report: `90_report_reframed_preference/index.html` + `index.zh.html`. Result artifacts: `02_process/05_reframed_preference_results/`.
Basis: within-segment preference test, ratings-blind until the final join (1428/1428 alignment); wild-cluster-bootstrap 5000 + max-statistic permutation 5000; independent review PASS; adversarial red-team null ROBUST (10 vectors); clean-room replication exact.

### Accepted claims (RQ010B)

| ID | Claim | Strength |
|---|---|---|
| RQ010B-KC-NULL | On WOD-E2E, a candidate trajectory's implied IPV (and its deviation from a human norm) does NOT predict human preference among the 3 rated candidates, and does NOT reach parity with weak physical features (IPV strength 0 vs physics ρ≈0.16–0.26; max-stat permutation p=1.0). Holds for both operationalizations — Scheme 1 future-only (n=75), Scheme 2 history+future ≥1 s (n=98) — and the ≥2 s robustness subset (n=66). | Bounded/underpowered NULL (rigorous) |
| RQ010B-KC-M3NOTRANSFER | The frozen RQ009 InterHub M3 context-conditioned human-norm envelope does NOT transfer to camera-derived WOD-E2E: categorical support alignable via pure HV-HV (684/684) but the numeric kinematic distribution is OOD (≤15 % in-support even after all construction fixes) and the norm center collapses to near-constant. A path-type-conditioned pure-HV IPV norm was used instead (human center ≈ neutral 0 across geometries; spread varies). | Methodological finding |
| RQ010B-KC-TRACKER | A working camera-only 3D detector + multi-frame tracker on WOD-E2E was built (StreamPETR fine-tuned on Waymo; forward-arc; ~1.2 m error). Route 4 is feasible. Per-candidate preference N ceiling ≈ 75–98 (counterpart limited to the forward-camera arc; only untapped lever = 360° tracking). | Methodological byproduct |

### Not supported / manuscript bound

The reframed hypothesis (ego IPV as an interpretable single variable comparable to physical features in explaining human preference) is **NOT SUPPORTED** on WOD-E2E. Consistent with the project pattern: RQ009 counterpart-IPV null, RQ003 prior null, RQ012B deviation→harm null. This **bounds the manuscript R4 human-alignment leg on WOD-E2E** — do NOT cite WOD-E2E as positive human-preference validation for IPV; it is a registered external-validity boundary (bounded null).

### Caveats (must travel)
N ceiling ~75–98 (underpowered for |ρ| < ~0.28); counterpart trajectories camera-detector-derived (~1.2 m), not ground truth; candidate-conditioned IPV over an open-loop opportunity structure (not closed-loop causal, not real harm); this is a null for the executed operationalization, not a universal claim.

### 4 Hz → 10 Hz robustness — CAVEAT CLOSED (2026-07-04)
The 4 Hz-vs-10 Hz sampling caveat was tested: trajectories interpolated to 10 Hz (dt=0.1), IPV re-estimated, and the within-segment test + controls re-run for both schemes. **The null HOLDS at 10 Hz** — Scheme 1 (n=75) deviation↔rating rho=0.165 [0.0005, 0.32] p=0.063 (still n.s., still wrong-sign for H1); Scheme 2 (n=47 under the stricter 10 Hz gate) rho=0.128 p=0.24; IPV beats no physical control in either (max-stat permutation p=1.0). Honest nuance: the per-candidate IPV VALUE is rate-sensitive (10 Hz-vs-4 Hz Spearman only ~0.29–0.31), so the estimator's absolute IPV depends on sampling rate — but the downstream preference-validity NULL is robust to the rate. Artifacts: `05_reframed_preference_results/10hz_sensitivity/`.
