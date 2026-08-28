# RQ027 Process Record

## Scope

This process implemented and ran the bounded independent-generator feasibility pilot only. It did not read real trajectory data, RQ007 held-out data, RQ014 blinded rating fields, paper files or accepted decision files.

## Code

- `pipelines/simulation/rq027_independent_generator.py`
- `pipelines/simulation/run_rq027_pilot.py`
- `tests/test_rq027_known_truth_recovery.py`
- `02_process/independent_verify.py`

## Verification

- Relevant baseline plus RQ027 tests: `50 passed, 1 skipped`.
- Full matrix: `288/288` run rows and `3,456` target-side attempted frames.
- Independent validation: `PASS`; verdict `PILOT_NO_GO` reproduced.

## Known issues retained

Two post-resolution negative controls were collision-tagged. They were not excluded. The other three negative-control families still produced `27/36=75.0%` persistent false acceptance, so the central negative-control conclusion does not depend on those two runs.
