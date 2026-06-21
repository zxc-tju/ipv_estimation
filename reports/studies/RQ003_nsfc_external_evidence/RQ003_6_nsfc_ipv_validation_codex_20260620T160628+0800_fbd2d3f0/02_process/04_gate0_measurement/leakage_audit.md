# Gate 0 Leakage Audit

Verdict: PASS

Evidence:

- `src/sociality_estimation/core/ipv_estimation.py:237-250` loops over `t`, sets `start=max(0,t-history_window)`, and passes only `start:t+1` tracks to the estimator.
- `src/sociality_estimation/core/ipv_estimation.py:240-260` creates fresh `Agent` instances per timestep, avoiding state carryover between windows.
- `src/sociality_estimation/core/ipv_estimation.py:263-301` estimates primary and counterpart from the same rolling prefix.
- `src/sociality_estimation/core/ipv_estimation.py:369-388` implements the current-frame API by setting `min_observation=steps-1`, so current estimation sees only the prefix supplied by the caller.
- The monkeypatched unit test `estimate_ipv_pair_rolling_window_prefix_only` verified every call length was at most `history_window+1` and no call included frame indices after `t`.

Forbidden online features:

- `ipv_trace_sample.csv` excludes observed PET, realized order, post-hoc phase labels, official outcomes, ranks, and outcome-derived fields.
- The InterHub source table contains some retrospective metadata, but Gate 0 did not use those columns in norm fitting, support, or trace output.
- No NSFC outcome or score-joined file was opened.

Rolling-to-rolling:

- Dynamic traces use rolling IPV values from the InterHub time-series source and do not compare rolling NSFC values to a full-window envelope.
- Historical `ipv_*_mean` summary columns were inspected only as schema context and were not used to fit the Gate 0 conditional norm.

Test environment note: import-based rolling/leakage tests used local stubs for optional plotting/optimizer modules because matplotlib/scipy are unavailable; the real optimizer path was not invoked.
