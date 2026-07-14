# RQ014 v1.7 addendum — W5b resource-pilot environment split

Date: 2026-07-14. This addendum narrows the already scoped
`rq014_g2_resource_pilot` operation. It does not authorize submission, G2R, or
rating access. Its W5b environment decision was superseded by the W5d changelog
below after the separately reviewed environment-v4 candidate became available.

## W5b environment decision (superseded by W5d)

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
- readouts are tied because the W5d M3 cost substrate and its row count do not
  vary by readout; ties use the first lane-declared readout, `NEX_MEAN`.

The exact selected cells are therefore:

- lightest: `RR3-R04N-CH-W10-H20-NEX_MEAN`;
- heaviest: `RR3-R10L-TF-HFEAS-NEX_MEAN`.

Before deriving either endpoint, the runner verifies the complete ordered axis
payload at SHA-256
`72216349fe299a31c7f00d534e129b19e9a7c0cf8ac1ec3fb0876d71e09413a1`,
then reconstructs all 320 IDs and verifies their LF-delimited canonical digest
`db280b77a5fba7e7bb8546da9d2d22337e66c1b1d267d8f8acd281326eaaadee`.
It then verifies that the two endpoints are present and emits this rule ID and
both IDs in the receipt. Any axis value, order, count, or ID drift fails closed.

## Frozen anchor eligibility

The measured window workload directly implements the checksum-bound
`RQ014_envelope_builder_contract_v2.json` rules because no anchor-domain
artifact is among the pilot's bound inputs. For each scene, `H_common` is the
largest grid tick jointly supported by C1, C2, C3, and the counterpart. Every
anchor requires every exact closed tick in all four position timelines. `H20`
is emitted only as its complete `rate_hz..2*rate_hz` set; `HFEAS` is unavailable
unless that complete H20 set exists. Window-local velocity and acceleration use
the frozen one-sided/centered operator. Heading is defined only above the exact
`1e-9` speed threshold, stationary ticks use the frozen earlier-heading/leading
backfill rules, all-stationary windows are ineligible, and exact `-pi` is stored
as `+pi`. No derivative halo or cross-window state reuse is permitted.

The R04N counterpart path is governed specifically by
`RQ014_recovery_lane_v3.json#/rating_blind_feature_bank/sampling_axis/0/history_and_counterpart_rule`,
which requires linear interpolation inside observed support to the full
tstar-anchored 0.25 s grid, and by
`RQ014_envelope_builder_contract_v2.json#/timeline_and_state_contract/wod_tstar/counterpart`,
`#/timeline_and_state_contract/linear_interpolation`, and
`#/timeline_and_state_contract/maximum_source_gap/default` plus
`#/timeline_and_state_contract/maximum_source_gap/extrapolation`.
The latter fixes componentwise interpolation from immediately bracketing source
timestamps, forbids extrapolation, and permits a bracketing gap of at most
`2*dt`. The runner therefore accepts the bit-exact `0.5 s` R04N boundary and
makes the query ineligible for any strictly larger float, without tolerance.

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
32 GiB protects the once-loaded shared source and M3 payloads plus concurrent
R10L/TF/HFEAS in-memory window and scoring work.

## Measurement and projection contract

Each measured stage records elapsed seconds, CPU seconds, process peak RSS, and
process I/O deltas. `source_load` runs exactly once in the parent and is emitted
as the separate `SHARED` measurement row. The read-only payload is installed
before a POSIX fork process pool and inherited copy-on-write by its workers; it
is not reopened, reparsed, or serialized once per cell. The receipt records
every selected cell's serial sum of
its `window_assembly + feature_prep + m3_scoring` wall/CPU timings, the worker-pool wall-clock,
and aggregate wall-clock spanning the shared load plus concurrent execution.
Stage failures use the fixed taxonomy
`INPUT_CONTRACT_FAILURE`, `SOURCE_LOAD_FAILURE`, `WINDOW_ASSEMBLY_FAILURE`,
`FEATURE_PREP_FAILURE`, or `M3_STAGE_FAILURE`; successful stages use `NONE`.
A PASS receipt requires the two shared stages and six cell-stage executions to succeed with zero
failure rate.

The non-M3 full-grid projection mirrors that execution exactly: the measured
shared source-load cost is counted once;
the larger observed light/heavy per-cell `window_assembly + feature_prep` cost is
multiplied by 320 for the serial wall/CPU estimate and by `ceil(320 / 16)` for
the 16-worker parallel wall estimate. The measured aggregate endpoint
wall-clock remains separate evidence of process startup and concurrent I/O, so
D3 can qualify both extrapolations rather than treating the theoretical worker
division as observed 16-way scaling. This is a conservative two-endpoint
extrapolation. W5d adds the once-only M3 model-load cost and the larger endpoint
M3-scoring cost to separate M3 and combined G2R projections, using the same
serial and `ceil(320 / 16)` parallel formulas. The receipt-to-`DONE.json` hash chain follows the
managed sibling operations, and `DONE.json` is written only for PASS.

## W5d changelog — accepted environment-v4 closure

W5d supersedes this addendum's W5b M3-disabled environment split and every
dependent readout, memory-sizing, measurement-count, taxonomy, timing, and
projection clause above. The W5c
build report and environment-closure proposal bind
`managed_python_environment_v4.json` at SHA-256
`1fbca7709d0ae913cf0bb73621fffa666981486973267fce51d221fce0f6d7d9`.
Export and contract-preflight stay on v3; the resource pilot binds v4 and must
verify, before scientific Python starts, 1,849 stdlib files / 40,860,773 bytes /
0 symlinks, 12,206 site-package files / 487,535,728 bytes / 0 symlinks, and the
94-row native closure (45 unique resolved files, 31 link rows).

After that gate, the pilot deserializes M3 once in the parent and measures one
shared `m3_model_load` plus per-cell `m3_scoring` in single-threaded fork
workers. Scoring uses a deterministic rating-free vector derived from frozen
model/support metadata, tiled only to the measured cell's window count; it is a
cost substrate, not scientific evidence. D3 receives separate non-M3, M3, and
combined serial/16-worker projections. Any v4, model, or scoring mismatch is a
global abort with a FAIL receipt and no DONE, following the lane-v3 W4
`m3_artifact_mismatch_policy=GLOBAL_ABORT` precedent; a failed measured load or
score attempt is recorded in a FAIL receipt, but no numeric M3/combined cost or
DONE is emitted. Silent fallback is
forbidden and numeric M3/combined estimates become `EXPLICITLY_UNMEASURED`.

The M3 manifest names no parity fixture. The reviewed v4 standard instead
adopts `tests/fixtures/m3_verifier_portable_fixture.json` at SHA-256
`ae62b9fddba53308d319ccef5a70d56a9f0ae243fe009aa3f85e36cb20fcee37`
with its recorded absolute tolerance `1e-7`; W5c achieved raw maximum absolute
difference `4.952394050405928e-11`. The fixture is closure evidence only and is
never opened as pilot workload input because it contains a frozen target field.
