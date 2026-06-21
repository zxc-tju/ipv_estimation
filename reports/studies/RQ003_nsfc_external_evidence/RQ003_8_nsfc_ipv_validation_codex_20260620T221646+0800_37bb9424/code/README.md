# Gate 0 Measurement Audit Code

Run from the repository root:

```bash
python3 reports/studies/RQ003_nsfc_external_evidence/RQ003_8_nsfc_ipv_validation_codex_20260620T221646+0800_37bb9424/code/run_gate0_measurement_audit.py
```

The script is outcome-blind: it reads the sanitized Gate 0 spec, the denylist,
the measurement-only plan section, read-only estimator source, InterHub rolling
IPV calibration columns, NSFC raw trajectory logs, and replay routing columns
with score/rank columns excluded.

If the active Python environment cannot import the core estimator dependencies
(`scipy`, `matplotlib`, `shapely`), the script still emits the audit package but
marks Gate 0 as non-PASS because same-estimator NSFC IPV recomputation is not
proven.
