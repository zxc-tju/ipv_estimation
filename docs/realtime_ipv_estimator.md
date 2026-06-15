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

`interhub_traj_lane/1_ipv_estimation_results/_codex_parallel_accurate_realtime_check/`

Current full-reference checks:

- Sign-mode sample: 200 full-dataset cases, 50 per dataset, 3,200
  agent-step labels.
- Accuracy-mode sample: 80 full-dataset cases, 20 per dataset, 1,280
  agent-step labels.
- Shared config: `history_window=10`, pkl motion source, full-run reference
  settings (`clip_margin=60`, `max_points=40`, `smooth_points=40`), sign
  threshold `0.05`.
- Five-candidate sign mode: overall sign accuracy `92.1%`, Wilson 95% lower
  bound `90.5%`, mean pair-step latency `0.099s`, median `0.087s` on the
  80-case check. On the larger 200-case check with `max_workers=10`, overall
  sign accuracy was `92.3%`, Wilson 95% lower bound `91.3%`, mean latency
  `0.110s`, median `0.099s`, and p95 latency `0.235s`.
- Seven-candidate accuracy mode: overall sign accuracy `94.1%`, Wilson 95%
  lower bound `92.6%`, mean pair-step latency `0.121s`, median `0.108s`.

The full-reference labels are prior HPC batch outputs, not hand labels. The
parallel estimator preserves the current accurate objective; remaining
differences reflect numerical/environment differences against that historical
batch.

The guarantee supported by these checks is overall sign accuracy on the sampled
mixed distribution. Dataset-specific guarantees are not yet supported: NuPlan is
the weakest subset (`81.1%` in the 200-case five-candidate sign check and
`84.1%` in the 80-case seven-candidate accuracy check). Follow-up probes found
that simple threshold tuning, temporal smoothing, thread execution, log-distance
reliability, and denser IPV grids did not push NuPlan over `90%` without new
modeling work. The current sign mode is an online statistical realtime solution,
not a hard-deadline guarantee for every frame; use asynchronous scheduling if a
strict control deadline must never be missed.
