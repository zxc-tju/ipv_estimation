# Support, OOD, Abstention, and Operational Parameter Definition

Gate 0 freezes all operational parameters without NSFC outcomes.

Primary estimator and windows:

- Primary estimator: `estimate_ipv_pair` in `src/sociality_estimation/core/ipv_estimation.py:158-335`.
- Primary window: 10 rolling frames, from existing estimator/InterHub defaults.
- Sensitivity windows: short=5, medium=10, long=20, derived as 0.5x/1x/2x of the primary window.
- Minimum observation: 4 frames.

Human conditional norm:

- Calibration split: deterministic InterHub-only split using SHA-256 of `scene_unique_id` and the fixed plan hash; calibration fraction 0.80.
- Records used: 5676048 perspective-conditioned InterHub human rolling IPV records from 30638 calibration scenes.
- `Q_low=Q0.25`, `m=Q0.5`, `Q_high=Q0.75` within `(theta_npc_bin, state_condition, tau_bin)`.
- `w=max((Q_high-Q_low)/2, w_min)`, with `w_min=0.196349540849` rad.

Support:

- High support requires exact InterHub cell count >= 30, both estimator errors <= 0.621630886982, theta_npc inside calibration q01/q99 [-1.17809654905, 1.1780972451], and current distance <= 41.3104272038 m.
- Monitor-only frames have at least 10 exact-cell records but miss high-support requirements.
- Low/OOD frames abstain from primary scalar summaries.

Fallback order:

1. Exact cell `(theta_npc_bin, state_condition, tau_bin)`.
2. `(theta_npc_bin, tau_bin)`.
3. `tau_bin` marginal.
4. Global InterHub calibration marginal.

Separate boundaries:

- Self-anchor is not used as the external-validation expectation.
- Empirical verifier is only `D_comp` and `D_yield` against the InterHub human conditional norm.
- Safety guard is recorded separately (`TTC=1.5s`, lateral gap `2.0m`) and is not added to Gate 0 empirical deviations.
