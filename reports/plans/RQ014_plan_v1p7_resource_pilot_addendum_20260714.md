# RQ014 v1.7 addendum — W5b resource-pilot environment split

Date: 2026-07-14. This addendum narrows the already scoped
`rq014_g2_resource_pilot` operation. It does not authorize submission, G2R,
rating access, M3 scoring, or environment-v4 construction.

## Environment decision

The pilot runs only under the existing checksum-bound v3 standard-library
runtime. The W5 environment-closure proposal records that v3 imports zero site
packages and cannot deserialize or execute the frozen RQ009 M3 scorer
(`.codex-fleet/rq014-execution-v1p6/board/reports/w5-env-closure-v4-proposal.md`).
Accordingly, W5b measures only these rating-blind stages:

1. `source_load`: read and parse the score-stripped CSV sources and frozen path
   mapping;
2. `window_assembly`: construct the exact sampled branches, horizons, and
   temporal windows selected below; and
3. `feature_prep`: derive window-local kinematic state from positions without a
   derivative halo.

The separate `m3_scoring` stage is disabled. Its receipt fields are fixed to
`m3_stage_enabled=false`, `env_v4_required=true`, and
`m3_cost_estimate=EXPLICITLY_UNMEASURED`. The frozen M3 file may be byte-verified
for lineage but must not be imported, deserialized, or scored. Both M3 cost and
the combined G2R total cost are `EXPLICITLY_UNMEASURED`; no numeric surrogate is
permitted. D3 therefore may use the receipt only for the measured non-M3 budget.
An M3-inclusive budget requires a separately accepted v4 closure and a new
measurement.

## Deterministic representative-cell rule

Lane v3 defines the 320-cell axes and canonical order but did not previously
name cost extremes. W5b freezes `LANE_V3_NON_M3_COST_EXTREMES_V1`, derived from
those lane semantics:

- sampling workload increases from `R04N` to `R10L`;
- the minimum temporal workload is `CH-W10`, while the maximum is `TF`, whose
  full `[0,H_common]` position window applies at every retained finite anchor;
- horizon workload increases from bounded `H20` to all-feasible `HFEAS`;
- readouts are tied because deviation/readout execution depends on disabled M3;
  ties use the first lane-declared readout, `NEX_MEAN`.

The exact selected cells are therefore:

- lightest: `RR3-R04N-CH-W10-H20-NEX_MEAN`;
- heaviest: `RR3-R10L-TF-HFEAS-NEX_MEAN`.

The runner must reconstruct all 320 IDs from the lane-v3 axes, verify that these
two IDs are present, and emit this rule ID and both IDs in the receipt. Any axis,
count, or ID drift fails closed.

## Resource profile

`rq014-g2-resource-pilot-cpu-v1` is fixed to the `amd` partition, one node, one
task, 16 allocated CPUs, `32G`, and `04:00:00`, while every registered per-worker
thread limit remains `1`. This sizing implements the PI guidance recorded in
`.codex-fleet/rq014-execution-v1p6/board/knowledge.md` on 2026-07-14:

> 2026-07-14 PI resource guidance: CPU is ample on the cluster — prefer
> efficiency; pilot and G2R profiles may request generous CPU (parallel cells;
> per-worker thread caps preserved for determinism). Recorded for W5b+.

The two selected endpoint cells therefore run concurrently in separate
single-threaded processes, with the process pool capped at 16 workers for the
production-cost extrapolation. Sixteen CPUs leave one core per possible worker;
32 GiB protects concurrent source scans and the R10L/TF/HFEAS in-memory window
work without asserting that M3 fits or ran.

## Measurement and projection contract

Each measured stage records elapsed seconds, CPU seconds, process peak RSS, and
process I/O deltas. The receipt also records every selected cell's serial sum of
its stage wall/CPU timings and the aggregate wall-clock around the concurrent
process-pool execution. Stage failures use the fixed taxonomy
`INPUT_CONTRACT_FAILURE`, `SOURCE_LOAD_FAILURE`, `WINDOW_ASSEMBLY_FAILURE`, or
`FEATURE_PREP_FAILURE`; successful stages use `NONE`. A PASS receipt requires
all six cell-stage executions to succeed and a zero failure rate.

The non-M3 full-grid projection is explicit: source-load cost is counted once;
the larger observed light/heavy per-cell `window_assembly + feature_prep` cost is
multiplied by 320 for the serial wall/CPU estimate and by `ceil(320 / 16)` for
the 16-worker parallel wall estimate. The measured aggregate endpoint
wall-clock remains separate evidence of process startup and concurrent I/O, so
D3 can qualify both extrapolations rather than treating the theoretical worker
division as observed 16-way scaling. This is a conservative two-endpoint
extrapolation, not a G2R total. The receipt-to-`DONE.json` hash chain follows the
managed sibling operations, and `DONE.json` is written only for PASS.
