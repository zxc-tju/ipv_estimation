# RQ014 PI scoped decision — D2 resource-pilot authorization loop

Date: 2026-07-14. Source: user/PI decision captured by the RQ014 Lead session after the completed contract-preflight wave.

## Accepted preflight evidence

The PI accepts the `PASS_PREFLIGHT_READY_FOR_PI_DECISION_D2` result in
`reports/studies/RQ014_wod_e2e_rating_recovery/RQ014_2_contract_preflight_20260714T003336Z_72dd4362/report.md`.
The accepted run was Slurm job `1924193`, which finished `COMPLETED` with `ExitCode 0:0`. Receipt verification and
the corrected bounded report both closed with dual `NO_BLOCKER`. The archived prerequisite receipt schema IDs are
`rq014-g2-contract-preflight-receipt-v1` and `rq014-managed-operation-done-v1`.

## Exactly one operation

This decision authorizes starting the full §8.1 authority-change and implementation-closure loop for exactly:

```text
rq014_g2_resource_pilot
```

It authorizes adding only that operation as the third RQ014 central allowlist entry. It does not authorize any G2R
feature-bank operation, rating-bearing operation, scientific/input-bundle regeneration, HPC submission, or pilot
result claim. It does permit the §8.1 final checksum-bundle regeneration referenced below after fresh review and
Formal G1; that governance bundle does not grant pilot execution.

## Mandatory stop before pilot submission

Central allowlisting is necessary but not sufficient for execution. The resource pilot remains fail-closed until its
resource profile, run-spec schema and template, entrypoint, rating-blind inputs/outputs, representative heaviest and
lightest cells, resource measurements, tests, candidate manifest, fresh distinct statistics and execution/governance
reviews, new Formal G1, final bundle, publication/sync, immutable spec, and validate-only evidence all exist and agree.
There is an explicit PI/user stop after validate-only and before the first pilot submit. No reply means
`STOP_AND_PRESERVE`.

## Rating boundary and D3

The pilot may not mount, read, join, derive, report, or infer ratings, preferences, ranks, rating-bearing inputs, or
rating-derived statistics. Its outputs are limited to rating-blind correctness evidence and walltime, CPU time, peak
RSS, I/O, failure rate, and full-build cost estimates. D3 is the separate compute-budget gate after an accepted pilot;
it decides whether to start a later authorization loop for the full rating-blind G2R feature build. This D2 decision
does not pre-authorize D3 or any downstream operation.
