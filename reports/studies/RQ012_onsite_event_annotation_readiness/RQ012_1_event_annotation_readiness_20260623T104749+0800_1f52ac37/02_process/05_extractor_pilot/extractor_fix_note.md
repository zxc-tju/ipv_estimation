# RQ012A W10b Extractor Invalid-Row Fix Note

Run ID: RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37
Worker: RQ012-W10b-extractor-fix

## Finding

W12 found that `split_valid_segments` counted impossible rows but still left
them in the rows used to build event flags. The minimal repro was an E02
synthetic series with `speed=-1.0` and `accel=-3.5` for 0.3 s: the pre-fix
extractor reported `impossible_values=3` but still emitted one E02 interval
with `raw_frame_hits=3`.

## Fix

The extractor now applies a shared emission-quality guard before event flags or
intervals are built. Rows are excluded from emission when they have missing
required fields, NaN/inf kinematics, negative speed magnitudes,
duplicate/non-increasing timestamps after per-series timestamp ordering, or
impossible geometry dimensions when geometry is required. Excluded rows close
the current valid segment so intervals cannot bridge through an invalid sample.
They still increment the missing-data or impossible-value diagnostics.

The same guard is used while constructing E09/E15 pair rows, so invalid ego or
counterpart geometry/speed cannot feed geometric pair emission.

## W12 Repro Closure

| case | event_count | raw_frame_hits | impossible_values | missing_data_failures |
|---|---:|---:|---:|---:|
| pre-fix E02 negative speed | 1 | 3 | 3 | 0 |
| post-fix E02 negative speed | 0 | 0 | 3 | 0 |

## Pilot Rerun Delta

The pilot was rerun with the unchanged seed `202606230510` and the same five
selected sessions. Central event counts were unchanged for all events. Raw hit
counts moved only where invalid rows had previously entered flag construction:
E03 decreased by 1 raw hit and E06 decreased by 2 raw hits. No thresholds were
changed; E15 proxy, E16 no-progress, and E18 candidate semantics were preserved.
