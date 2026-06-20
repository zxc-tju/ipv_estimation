# Online IPV Interval Query Report

Primary entry: `00_entry/index.html`.

This package answers the 2026-06-18 research task: replace inaccurate online
IPV range lookup based on predicted PET/risk with a more accurate online query
scheme for human-driver IPV intervals in the same scene.

## Main Result

The recommended scheme is direct conditional interval modeling with split
conformal calibration:

1. Tier 0: global or coarse empirical conformal floor.
2. Tier 1: strict-online kinematic CQR interval, current deployable default.
3. Tier 2: rolling-IPV self-anchor CQR interval, best candidate pending a
   prefix-only `RealtimeIPVEstimator` rebuild.

The local reproducible experiment shows that predicted-risk lookup is not the
main lever. On the primary test split, oracle PET cell lookup has mean interval
width 0.835, online TTC cell lookup 0.839, and global floor 0.854. Direct
strict-online CQR narrows this to 0.686, while rolling-IPV CQR reaches 0.599,
both near nominal 90% coverage. Leave-Waymo-Out remains under-calibrated for all
methods, so source-shift calibration must be treated as an open deployment
problem.

## Layout

- `00_entry/index.html`: reader-facing report.
- `01_results/`: metrics, feature extract, prediction sample, figure exports,
  and final summary.
- `02_process/`: experiment script, algorithm spec, and process artifacts.
- `archived/report_local_state/interhub_20260620/codex_fleet/online-ipv-query/board/`:
  fleet plan, ledgers, and validation.
- `TRACEABILITY.md`: map from claims to result and process files.
