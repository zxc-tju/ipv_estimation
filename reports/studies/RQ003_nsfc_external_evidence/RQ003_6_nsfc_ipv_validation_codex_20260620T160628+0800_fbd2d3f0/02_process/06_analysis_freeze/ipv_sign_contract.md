# RQ003 Frozen IPV Sign Contract

Worker: `RQ003_phase3_freeze_002`

## Orientation

- `theta > 0` means more prosocial.
- `theta < 0` means more competitive.
- Competitive shortfall and over-yielding excess are separate one-sided deviations.

## Directional Deviations

Let the human conditional norm be defined from InterHub calibration data:

```text
m(t) = Q_0.5(theta_ego | theta_npc, s, tau)
w(t) = max((Q_high - Q_low) / 2, w_min)
D_comp(t) = max(0, (Q_low(theta_npc, s, tau) - theta_ego) / w(t))
D_yield(t) = max(0, (theta_ego - Q_high(theta_npc, s, tau)) / w(t))
```

Interpretation:

- `D_comp > 0`: ego is below the lower conditional human envelope and is therefore more competitive than expected.
- `D_yield > 0`: ego is above the upper conditional human envelope and is therefore more prosocial/yielding than expected.
- `D_comp` and `D_yield` must never be collapsed into a signed scalar for the confirmatory predictor block.

## Required Invariance Checks

Before confirmatory interpretation:

- obvious forcing/aggressive intrusion cases must map to `D_comp > 0`;
- obvious freezing/over-yielding cases must map to `D_yield > 0`;
- role exchange, mirror transform, and time truncation must not flip signs;
- no future frames, observed PET, realized order, or post-hoc phase labels may enter online metrics.

## Gate 0 Review Condition

`G0R-COND-001` remains a Phase 4 gating condition:

> Install or restore scipy/matplotlib, run the real optimizer path, and reconfirm the sign contract on real theta outputs before any confirmatory NSFC result is trusted.

Phase 4 may compute measurement traces before final interpretation, but it must not trust or report confirmatory NSFC IPV results until this condition passes.
