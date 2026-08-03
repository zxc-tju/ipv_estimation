## L1: RQ009 Exact-Zero Target Atom Split

This section supports RQ015, whose role in online verification is to decide whether a frame-level IPV value carries discriminating information among the candidate IPV values before the accepted RQ009 envelope support rule is applied. IPV means Interaction Preference Value, a scalar for social interaction tendency. RQ015 A-K have already produced the K2 row ledger with row-level log-domain gate status; L1 is the read-only step that checks whether RQ009's exact-zero target atom can be connected to that ledger and then separates passing rows from rows where the gate says the value should not be interpreted as discriminating.

### Join Feasibility

The RQ009 scoring target used for the accepted exact-zero atom is the `y` column in `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/04_calibration/predictions/tier=M3/fold=test/predictions.parquet` after filtering `alpha` to 0.10 in the same way as `compute_target_atoms()` in `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/05_evaluation/evaluate.py`. That gives 1,270,566 rows and 273,819 exact `y == 0.0` rows, which is 21.551% (273,819/1,270,566). The row key is `case_key + anchor_frame_index + perspective`; it is unique in the one-alpha target rows. For connecting to K2, `source_dataset` is added because K2's `product_row_key` includes it, and the role suffix `|role=target_future` gives the K2 `canonical_key`.

The K2 `artifact_id=rq009_feature_matrix` ledger has 8,994,736 rows. Its `canonical_key` is unique at 8,994,736/8,994,736 rows. Its `product_row_key` and `interhub_canonical_key` each have 4,497,368 distinct values, and each appears exactly twice because K2 stores two measurement roles, `target_future` and `counterpart_current`. In the `target_future` subset, `canonical_key`, `product_row_key`, and `interhub_canonical_key` are all unique at 4,497,368/4,497,368 rows.

The exact left join is feasible as a one-to-zero-or-one join, not as full coverage of all RQ009 target rows. For all RQ009 target rows, 888,892 rows match K2 and 381,674 rows do not, so the match rate is 69.960% (888,892/1,270,566) and the missing rate is 30.040% (381,674/1,270,566). The 3,811,698 row identity is not a coincidence: it is 1,270,566 target rows times three alpha levels. The K2-matched part is 888,892 x 3 = 2,666,676, and the missing part is 381,674 x 3 = 1,145,022.

For the exact-zero atom, 192,271 of 273,819 rows match K2, and 81,548 of 273,819 rows do not. The missing rows are kept as a separate category; no approximate key or relaxed join was used.

### Split Result

Among the 273,819 RQ009 target rows with `y == 0.0`:

| Category | Rows | Share of zero atom | Share of all RQ009 targets |
|---|---:|---:|---:|
| Passing zero, K2 `status == OK` | 99,938 | 36.498% (99,938/273,819) | 7.866% (99,938/1,270,566) |
| Non-OK K2 status | 92,333 | 33.720% (92,333/273,819) | 7.267% (92,333/1,270,566) |
| Non-OK reason `NEAR_UNIFORM` | 90,490 | 33.047% (90,490/273,819) | 7.122% (90,490/1,270,566) |
| Non-OK reason `NO_IPV_EFFECT` | 1,796 | 0.656% (1,796/273,819) | 0.141% (1,796/1,270,566) |
| Engineering failure `SOLVER_FAILURE` | 47 | 0.017% (47/273,819) | 0.004% (47/1,270,566) |
| Join miss | 81,548 | 29.782% (81,548/273,819) | 6.418% (81,548/1,270,566) |

The directly answerable part of RQ009's 21.551% (273,819/1,270,566) exact-zero atom is therefore: 36.498% (99,938/273,819) are passing zero rows, 33.720% (92,333/273,819) are non-OK rows, and 29.782% (81,548/273,819) are outside the K2 target-future ledger join.

### Zero Definitions

The exact-zero definition used above is RQ009's actual scoring target: `y == 0.0` in `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/04_calibration/predictions/tier=M3/fold=test/predictions.parquet` after the `alpha == 0.10` filter. That count is 21.551% (273,819/1,270,566). The upstream matrix column `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix` / `target_ipv_future` has a slightly different exact-zero count because the prediction `y` column is stored as a float column; this is why L1 uses the RQ009 evaluation input, not a redefinition from the matrix.

For the K2 `target_future` subset, `ipv_log == 0.0` exactly occurs in 3.498% (157,318/4,497,368); using `abs(ipv_log) < 1e-12` gives 6.939% (312,062/4,497,368). Within the matched RQ009 exact-zero rows, `ipv_log == 0.0` exactly occurs in 7.123% (13,696/192,271); using `abs(ipv_log) < 1e-12` gives 13.992% (26,903/192,271). These are different quantities from RQ009's `y == 0.0` atom and should not be substituted for it.

### Fingerprint Check

The legacy `ipv_error` fingerprint was checked with both exact equality to `0.6220355269907728` and a tolerance rule `abs(ipv_error - 0.6220355269907728) <= 1e-12`. In the matched RQ009 exact-zero rows, the tolerance rule hits 10.903% (29,854/273,819). The hit rows are not highly coincident with K2 non-OK status: 25.109% (7,496/29,854) of fingerprint-hit rows are non-OK, while 74.891% (22,358/29,854) are `status == OK`. Conversely, only 8.118% (7,496/92,333) of non-OK rows hit the fingerprint.

The non-overlap has two parts. First, 22,358 matched exact-zero rows have the fingerprint but `status == OK`, so the old error value alone is not sufficient for the RQ015 gate decision. Second, 84,837 matched exact-zero rows are non-OK without the fingerprint; most are `NEAR_UNIFORM` under K2's log-domain status fields. The crosstab is in `L1_fingerprint_crosstab.csv`.

### Conclusions

The join key is exact and duplicate-free for the rows K2 covers, but K2 does not cover every RQ009 test target row. L1 therefore reports an exact left join with explicit misses, not a full one-to-one coverage claim.

The RQ009 exact-zero atom is mixed. Of 273,819 exact-zero target rows, 99,938 are passing zero rows, 92,333 are non-OK rows, and 81,548 have no K2 target-future ledger row.

### Pending Decisions

No RQ009 accepted result was changed. The remaining decision for the leader is how L3 should phrase the 81,548 join-miss rows: either keep them as an explicit out-of-ledger category, or restrict the headline split to the 192,271 matched rows and report the missing rows in the coverage sentence. If omitted, the report would overstate how much of RQ009's exact-zero atom K2 can classify.

state: WAITING_ON_LEADER
timestamp_utc: 2026-08-03T03:17:47Z
