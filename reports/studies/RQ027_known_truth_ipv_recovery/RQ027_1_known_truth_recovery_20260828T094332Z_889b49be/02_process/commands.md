# Commands

## Baseline and new tests

```bash
PYTHONPATH=src:. .venv_ipv_local_test/bin/python -m pytest -q \
  tests/test_rq027_known_truth_recovery.py \
  tests/test_ipv_estimator_parity.py \
  tests/test_rq015_reliability_logdomain.py \
  tests/test_ipv_execution_profile.py
```

Result: `50 passed, 1 skipped`.

## Full pilot

```bash
PYTHONPATH=src:. .venv_ipv_local_test/bin/python \
  pipelines/simulation/run_rq027_pilot.py \
  --output-dir reports/studies/RQ027_known_truth_ipv_recovery/RQ027_1_known_truth_recovery_20260828T094332Z_889b49be/01_results \
  --solver-mode exact \
  --workers 8
```

Result: `PILOT_NO_GO`; elapsed `590.815410709 s`.

## Independent recomputation

```bash
PYTHONPATH=src:. .venv_ipv_local_test/bin/python \
  reports/studies/RQ027_known_truth_ipv_recovery/RQ027_1_known_truth_recovery_20260828T094332Z_889b49be/02_process/independent_verify.py \
  --run-dir reports/studies/RQ027_known_truth_ipv_recovery/RQ027_1_known_truth_recovery_20260828T094332Z_889b49be
```

Result: `validation_status=PASS`.
