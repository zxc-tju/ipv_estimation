# Realtime IPV Estimator

This project now has a realtime-safe IPV path based on the original candidate
trajectory model.

## Recommended Modes

Use `RealtimeIPVEstimator.for_realtime_sign()` when the control loop only needs
the IPV sign and must stay close to 10 Hz:

```python
from ipv_estimation import MotionSequence, RealtimeIPVEstimator

with RealtimeIPVEstimator.for_realtime_sign(
    history_window=10,
    max_workers=10,
) as estimator:
    signs, ipv, err = estimator.estimate_sign_current(
        primary_sequence,
        counterpart_sequence,
        threshold=0.05,
    )
```

This uses the named five-candidate grid
`SIGN_REALTIME_CANDIDATE_IPV_VALUES = [-3, -1, 0, 1, 3] * pi / 8`, keeps the
accurate SLSQP solver, and batches both agents' candidate optimizations into
one persistent worker pool call per frame.

Use the seven-candidate `parallel_accurate` path when IPV value fidelity matters
more than strict latency:

```python
with RealtimeIPVEstimator(
    history_window=10,
    solver_preset="parallel_accurate",
    max_workers=20,
) as estimator:
    ipv, err = estimator.estimate_current(primary_sequence, counterpart_sequence)
```

`parallel_accurate` keeps the accurate SLSQP settings and the legacy seven IPV
candidates. This is the accuracy-preserving path.

For one-off calls, `estimate_ipv_current()` now defaults to `parallel_accurate`.
For sustained loops, prefer `RealtimeIPVEstimator` so worker processes are
reused across frames.

## Validation Snapshot

Validation artifacts are under:

`interhub_traj_lane/1_ipv_estimation_results/ipv_rt_final/`

The final CSV evidence is in `ipv_rt_final/evidence/`, and the rebuild script is
in `ipv_rt_final/scripts/run_full_reference_validation.py`.

The full-reference labels are prior HPC batch outputs, not hand labels. These
checks use pkl motion source, `history_window=10`, full-run reference settings
(`clip_margin=60`, `max_points=40`, `smooth_points=40`), and sign threshold
`0.05`.

### Mode Summary

| Mode | Sample | Candidates | Workers | Sign accuracy | Wilson 95% lower | Mean latency | Median latency | P95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Five-candidate sign mode | 200 cases / 3,200 agent-steps | 5 | 10 | 92.3% | 91.3% | 0.110s | 0.099s | 0.235s |
| Seven-candidate accuracy mode | 80 cases / 1,280 agent-steps | 7 | 20 | 94.1% | 92.6% | 0.121s | 0.108s | 0.253s |

The five-candidate mode is the recommended sign-only online path. The
seven-candidate mode is preferable when the IPV value itself matters more than
latency.

### IPV Value Bias

Bias is `pred_ipv - reference_ipv`. A negative value means the online estimator
is lower than the full-reference label.

| Mode | N | Cases | Sign accuracy | Bias | MAE | Median abs error | RMSE | P95 abs error | Max abs error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Five-candidate sign mode | 3,200 | 200 | 92.3% | -0.0050 | 0.0802 | 0.0085 | 0.1764 | 0.3924 | 1.3771 |
| Seven-candidate accuracy mode | 1,280 | 80 | 94.1% | -0.0026 | 0.0538 | 0.0002 | 0.1393 | 0.3332 | 1.1057 |

Overall bias is near zero in both modes, so there is no strong global upward or
downward shift. The five-candidate sign mode trades IPV magnitude fidelity for
latency: it keeps sign accuracy above 90% on the mixed sample, but its MAE and
tail errors are larger than the seven-candidate mode.

### Dataset Breakdown

Five-candidate sign mode, 200 cases:

| Dataset | N | Cases | Sign accuracy | Bias | MAE | Median abs error | RMSE | P95 abs error | Max abs error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AV2 | 800 | 50 | 94.1% | -0.0021 | 0.1119 | 0.0148 | 0.2269 | 0.5683 | 1.2538 |
| Lyft | 800 | 50 | 98.1% | +0.0027 | 0.0271 | 0.0000 | 0.0977 | 0.1518 | 0.9661 |
| NuPlan | 800 | 50 | 81.1% | -0.0143 | 0.1349 | 0.0792 | 0.2187 | 0.4531 | 1.1767 |
| Waymo | 800 | 50 | 95.8% | -0.0062 | 0.0468 | 0.0022 | 0.1253 | 0.1890 | 1.3771 |

Seven-candidate accuracy mode, 80 cases:

| Dataset | N | Cases | Sign accuracy | Bias | MAE | Median abs error | RMSE | P95 abs error | Max abs error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AV2 | 320 | 20 | 93.1% | -0.0020 | 0.0825 | 0.0010 | 0.1931 | 0.4103 | 1.1057 |
| Lyft | 320 | 20 | 99.7% | -0.0001 | 0.0029 | 0.0000 | 0.0153 | 0.0088 | 0.1765 |
| NuPlan | 320 | 20 | 84.1% | -0.0050 | 0.1193 | 0.0598 | 0.1893 | 0.4324 | 0.8056 |
| Waymo | 320 | 20 | 99.4% | -0.0033 | 0.0104 | 0.0000 | 0.0655 | 0.0470 | 1.0592 |

NuPlan is the dominant weak slice for both sign and IPV magnitude. Lyft and
Waymo are stable in the seven-candidate mode; AV2 retains good sign accuracy
but has larger tail magnitude errors.

### Bias By Reference Sign

Five-candidate sign mode:

| Reference sign | N | Sign accuracy | Bias | MAE | Pred mean | Ref mean | P95 abs error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| -1 | 725 | 89.5% | +0.0542 | 0.1435 | -0.4711 | -0.5252 | 0.5449 |
| 0 | 1,485 | 94.3% | -0.0027 | 0.0119 | -0.0028 | -0.0002 | 0.0634 |
| 1 | 990 | 91.2% | -0.0517 | 0.1363 | +0.5090 | +0.5607 | 0.4900 |

Seven-candidate accuracy mode:

| Reference sign | N | Sign accuracy | Bias | MAE | Pred mean | Ref mean | P95 abs error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| -1 | 300 | 91.3% | +0.0212 | 0.0889 | -0.5018 | -0.5231 | 0.4033 |
| 0 | 563 | 95.2% | +0.0016 | 0.0083 | +0.0019 | +0.0002 | 0.0423 |
| 1 | 417 | 94.5% | -0.0255 | 0.0898 | +0.5205 | +0.5459 | 0.4246 |

Both modes slightly shrink extreme IPV magnitudes toward zero: negative labels
are predicted less negative on average, and positive labels are predicted less
positive on average. This shrinkage is stronger in the five-candidate sign mode.

The guarantee supported by these checks is overall sign accuracy on the sampled
mixed distribution. Dataset-specific guarantees are not yet supported: NuPlan is
the weakest subset (`81.1%` in the 200-case five-candidate sign check and
`84.1%` in the 80-case seven-candidate accuracy check). Follow-up probes found
that simple threshold tuning, temporal smoothing, thread execution, log-distance
reliability, and denser IPV grids did not push NuPlan over `90%` without new
modeling work. The current sign mode is an online statistical realtime solution,
not a hard-deadline guarantee for every frame; use asynchronous scheduling if a
strict control deadline must never be missed.
