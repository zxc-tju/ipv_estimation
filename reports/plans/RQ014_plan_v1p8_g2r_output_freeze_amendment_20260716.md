# RQ014 v1.8 amendment — G2R W1 output freeze

Date: 2026-07-16.

Status: authority/schema freeze only; `rq014_r2_blind_feature_build` remains
DENY and has no runnable surface in this wave.

## Authority and precedence

This amendment supplements, and does not rewrite, these exact science bytes:

1. `RQ014_recovery_lane_v3.json`, SHA-256
   `a23141e27e43f4c718f75ad48fb0356beac7ce8fb705243cf18191187efbd4ba`,
   controls the G2R sampling, temporal, horizon and readout axes, deviations,
   masks, canonical order, NC gate, and staged rating boundary.
2. `RQ014_envelope_builder_contract_v2.json`, SHA-256
   `407d63209764896a673aa94811f9dd8b60a57a047d17e8cee0a3465c55b8c8a4`,
   controls only its safe-primitive timeline/state and scene-anchor-domain
   clauses for G2R; its InterHub envelope axes do not replace lane v3.
3. `models/rq009_m3/feature_spec_contract.json`, SHA-256
   `3ad8ba8ab4c51422a7b2ef208683b7552b68f9e949f0087542ba208065677cce`,
   controls the 32-column M3 model order and named output surface.
4. `RQ014_config_space_v1p6.yaml`, SHA-256
   `9818f20b5a844a4dfd4ba21233f5dd9f00eee2373034fac7f869481511ea490e`,
   controls the frozen M3/estimator bindings and existing DENY state.
5. `RQ014_execution_contract_v1p5.json`, SHA-256
   `acc44ac831ffafc22ad5f8b641204ed5ee1e0944b9693e2609ad5938e862cc28`,
   controls the rating boundary, managed lineage, v4 requirement, and current
   operation denial.

Within that source set, lane v3 supersedes legacy recovery axes; envelope v2
governs only the clauses named above; the feature contract governs M3 column
order; and this amendment plus `RQ014_g2r_output_contract_v1.json` governs only
the newly resolved WOD-port and physical-output bytes. Any unlisted conflict or
missing fixture binding fails closed. The PI/Lead resolutions are recorded in
`RQ014_PI_decision_D3_G2R_W1_output_freeze_20260716.md`.

## Operation and stage boundary

G2R produces the rating-blind 320-cell feature bank, anchor evidence,
availability masks, common-support blind manifest, immutable predictor
manifest, scene-anchor-domain binding, NC history-only receipt, umbrella output
manifest, operation receipt, and PASS-only managed DONE. It reads no rating,
performs no rating join, and computes no observed rating statistic.

G2R ends **before G3R/D4**. It must not produce the 960 RWS/PSP/PPR leaderboard
rows, associations, ranks, selected recipe, `recovery_ledger.jsonl`, terminal
recovery-ledger digest, or any rating-derived field/status. The G3R recovery
ledger hash chain is not reused for G2R: G2R binds whole files and contiguous
slices only. `rating_access=NONE` in the contract and
`rating_access=FORBIDDEN` in the future operation receipt are equivalent
prohibitions, not an authorization surface.

## Recovered WOD-to-M3 port

The recovery reference is the ignored local forensic file
`reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/hpc_onsite_ipv/build_onsite_m3_anchors_hpc.py`,
68,645 bytes, SHA-256
`5abc1429ad82db7699bbc3cd1b488f4d0667e2cd976e23533445b12b31404d10`.
It builds OnSite AV-perspective anchors against the same RQ009 M3 feature
contract and recovers the portable 25 numeric formulas, H10/minimum-4/tail-10/
slope-tail-5 mechanics, seven categorical surface, and external-dataset row
assembly (`board/g2r_w1_wod_port_design.md:20-179,220-323`). It is evidence for
the port, not a runtime input and not authority over WOD transport.

WOD transport remains frozen: consume the exported single checksum-bound
counterpart, preserve C1/C2/C3 source order, R04N/R10L position-only
within-support interpolation and gap rules, t*=0 seam, exact window-local state
derivation, route reference, H-common, and the immutable anchor-domain
memberships. For every retained tau, the primary M3 context is a separate
history-only branch ending at tau. The preregistered blind sensitivity uses the
named `terminal_minus_6_rows` alignment. The lane's family-specific focal IPV
remains comparison `v`; no H4 target is recomputed.

Actual-trajectory kinematics produce M3 geometry, turn, and priority tokens;
the checksum-bound WOD CP/HO/MP/F path type stays separate. The exact HV–HV
tokens implement the PI rationale that deviation is from the human-population
IPV envelope. The output contract records every 32-column source/method and the
canonical input-row preimage.

## Resolved format and governance choices

The logical feature bank is normalized into anchor-score and scalar-cell JSONL
tables. Every JSON row is strict UTF-8 canonical sorted compact JSON plus one
LF; artifacts have no blank lines and are hashed over complete stored bytes.
All 459,840 cell/scene/candidate slots are materialized in cell-major,
raw-UTF-8 scene, candidate-ordinal order; terminal values use typed NA rather
than row deletion, null, NaN, infinity, or sentinels. CSV is retained only for
the already frozen scene-anchor and common-support blind artifacts.

Only finite strict `L<M<U` permits NEX/NMD/AMD. Equality at either boundary,
nonfinite inputs, or an invalid interval produces row-level
`M3_SCORING_NUMERICAL_FAILURE`; this A09 choice remains explicitly subject to
statistics review before execution authorization. Failures are candidate
granular, make the all-three scene mask false, and do not globally abort unless
authority/schema/hash integrity or the NC leakage gate fails. The decision
record's `D10` reference is treated as A10 because D1–D8 are the complete D
ledger.

The output contract instantiates the otherwise unnamed A05/A10 status/reason
namespace and the A12/A14 receipt failure objects as deterministic governance
identifiers. Those identifiers do not add a science branch. A pre-runtime gate
failure aborts without receipt/DONE; a runtime failure writes an immutable FAIL
receipt, no DONE, and publishes no staged output. PASS publication is atomic,
and DONE is written only after the PASS receipt and every output are durable.

For A11, the normative `wod_scene_anchor_domain.csv` `reason_code` registry is
the lane-v3/G2R F-stage registry frozen in
`RQ014_g2r_output_contract_v1.json#/status_contract`; the stale
`RQ014_envelope_builder_contract_v2.json#/scene_anchor_domain_contract/reason_code_registry`
pointer to recovery lane v2 is superseded for this G2R consumer and must not be
followed. The immutable anchor-domain artifact bytes are not rewritten.

## W1/W1b and future-wave boundary

This W1a wave freezes authority, contract, and seven JSON Schemas only. Fixture
paths/hashes are explicit `PENDING_W1B` bindings and cannot pass a gate. W1b
must add the deterministic golden payloads, exact hashes, exact-key tests,
formula/boundary tests, status propagation tests, and candidate-manifest/G1/
bundle cascade.

No environment v5 is required: future G2R remains bound to the reviewed v4
closure when separately authorized. This amendment does not edit
`configs/research_authorization.json`, the run-spec schema, launcher,
preflight, managed entrypoint, resource profiles, or any frozen science
contract. It therefore cannot make G2R executable or allowed.
