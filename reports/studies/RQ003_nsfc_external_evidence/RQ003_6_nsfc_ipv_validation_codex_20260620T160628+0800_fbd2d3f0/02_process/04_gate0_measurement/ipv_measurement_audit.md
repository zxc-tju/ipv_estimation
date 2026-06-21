# Gate 0 IPV Measurement Audit

Status: PASS
Worker: RQ003_phase2_gate0_audit_001
Run: RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0

Identity checks passed: run root, Gate 0 directory, traces directory, run manifest RUN_ID, and plan SHA-256 matched the fixed context.

Outcome firewall:

- Denylist file read and obeyed.
- Denylisted file content reads: 0.
- NSFC outcome values, official outcome files, ranks, and outcome-joined tables were not opened.
- The raw plan was not opened because the Gate 0 denylist instructs using the sanitized spec instead.

Operational freeze summary:

- Calibration source: InterHub rolling IPV time-series only.
- Calibration scenes: 30638; perspective records used: 5676048; rows streamed: 3695981.
- Quantiles: Q_low=0.25, Q_mid=0.5, Q_high=0.75; w_min=0.196349540849 rad.
- Conformal threshold: 2.06774367457, from InterHub calibration only.
- High-support exact-cell threshold: n >= 30; estimator error max 0.621630886982.

|check|result|evidence|
|---|---|---|
|1 sign contract theta>0 prosocial|PASS|ipv_sign_contract.md; unit_test_results.csv|
|2 D_comp orientation|PASS|formula tests; operational_parameters.yaml|
|3 D_yield orientation|PASS|formula tests; operational_parameters.yaml|
|4 human conditional norm|PASS|InterHub calibration split; no self-anchor expectation|
|5 canonical unit tests|PASS|13/13 pass|
|6 conditioned differences|PASS|same_theta_different_conditioned_norms_differ|
|7 mirror/role/time tests|PASS|mirror, role-swap, rolling prefix tests|
|8 same estimator/window/sampling|PASS|same estimate_ipv_pair contract frozen for InterHub and NSFC; NSFC must resample to 10 Hz rolling contract|
|9 rolling-to-rolling only|PASS|source slice audit and monkeypatch test|
|10 only info before t|PASS|leakage_audit.md|
|11 no PET/order/phase online|PASS|trace columns exclude forbidden online fields|
|12 self-anchor/verifier/guard separated|PASS|support_definition.md|
|13 expectation human norm|PASS|InterHub-only conditional median|
|14 conformal threshold InterHub only|PASS|threshold=2.06774367457|
|15 no nominal NSFC conformal coverage|PASS|operational_parameters.yaml disallows claim|
|16 no NSFC outcome tuning|PASS|denylisted reads=0; InterHub-only params|
|17 counterpart-conditioned IPV|PASS|two perspectives per frame in trace|
|18 support/OOD/abstention computed|PASS|ipv_trace_sample.csv columns|
|19 three views recomputable|PASS|unit test recomputed marginal/conditional/scalar|
|20 event summaries computable|PASS|trace has D_comp/D_yield by frame for onset/persistence/AUC/reciprocity|
|21 missing params frozen outcome-free|PASS|operational_parameters.yaml|
|22 representative trace sample|PASS|ipv_trace_sample.csv|

Result views:

- Marginal view: recompute means/rates over high-support non-abstained frames from `ipv_trace_sample.csv` or full trace.
- Conditional view: group by `theta_npc_bin`, `state_condition`, `tau_bin`, and `perspective`.
- Scalar view: integrate or average `max(D_comp,D_yield)` over rolling frames; onset/persistence use first and consecutive nonzero deviation frames.

Known scope note:

- This Gate 0 pass did not fit norms from NSFC algorithm trajectories and did not compute predictor-outcome associations. That is intentional.
- Top-five trajectory logs were inspected only for outcome-free replay schema; the representative trace sample is InterHub-only, which the task explicitly permits.
