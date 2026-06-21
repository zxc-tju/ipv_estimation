# A1 / Safety Identity Correction

Worker: `RQ003_phase7_scenario_fix_001`
Generated UTC: `2026-06-20T14:41:56.155118+00:00`

## Corrected Identity

- Official A1 rows: 10.
- Official A1 zero-score/catastrophic rows: 2.
- Safety/collision-free membership is based on official `safety`: `collision_free = safety >= 100`.
- The prior report's claim that old structural A1 captured the non-100 safety rows was false. The zero-safety rows were old-label C2 but official A1.

## Zero-Score / Catastrophic Rows

| cell_id | team | area | old_label | official_scenario | case_id | safety | efficiency | coordination | comprehensive | scenario_name |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| T15_C2_task6924_case2333 | T15 | beijing | C2 | A1 | 2333 | 0 | 0 | 0 | 0 | 12-行人横穿道路 |
| T20_C2_task6941_case2333 | T20 | beijing | C2 | A1 | 2333 | 0 | 0 | 0 | 0 | 12-行人横穿道路 |

## All Official A1 Rows

| cell_id | old_label | safety | coordination | comprehensive |
|---|---|---:|---:|---:|
| T11_A1_task6923_case2325 | A1 | 100 | 87.01 | 86.44 |
| T14_C2_task6922_case2333 | C2 | 100 | 80.16 | 80.57 |
| T15_C2_task6924_case2333 | C2 | 0 | 0 | 0 |
| T16_C2_task6926_case2333 | C2 | 100 | 81.39 | 82.05 |
| T17_C2_task6931_case2333 | C2 | 100 | 73.6 | 79.42 |
| T20_C2_task6941_case2333 | C2 | 0 | 0 | 0 |
| T5_A1_task6925_case2325 | A1 | 100 | 83.24 | 69.66 |
| T6_A1_task6921_case2325 | A1 | 100 | 93.25 | 96.83 |
| T7_A1_task6938_case2325 | A1 | 100 | 77.11 | 67.02 |
| T8_A1_task6932_case2325 | A1 | 100 | 77.73 | 70.94 |
