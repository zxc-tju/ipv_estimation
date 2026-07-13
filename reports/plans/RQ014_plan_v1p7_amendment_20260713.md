# RQ014 plan v1.7 amendment — PI-identified M3 envelope and science-byte freeze

Date: 2026-07-13. This amendment implements
`RQ014_PI_decision_envelope_identification_20260713.md`. It supersedes the envelope-axis, recovery-grid,
G4R-envelope, active-registry, binding, active recovery-authority, staged-count, and rating-blind input-role clauses
of v1.5/v1.6. All rating isolation, operation authorization, append-only ledger, fixed association definitions,
ordering, weights, thresholds, and stop rules remain unchanged.
The byte-addressable v2 lane and v1.5 registries remain historical authority for their own versions.

## Frozen M3 envelope

The sole envelope input is the exact RQ009 M3 bundle. Its external private managed scorer is fixed at
`/share/home/u25310231/ZXC/sociality_estimation/checkpoints/rq009_m3/m3_scorer.joblib`, SHA-256
`b04999aba29a82fb71a97ac22c728479a7734e24a0b32189d08f95184d74f253` (88,306,301 B), tracked manifest
`2efbdd0c39edabc419aad815a1eb7529af3623a06c4d3a0b0a99782bcb2f40f4`, and tracked feature contract
`3ad8ba8ab4c51422a7b2ef208683b7552b68f9e949f0087542ba208065677cce`. The manifest itself binds the scorer and
feature contract (`models/rq009_m3/manifest.json#/artifact` and `#/feature_contract`); the model outputs are fixed by
`feature_spec_contract.json#/output_columns` and `#/intervals/90`.

For every otherwise eligible WOD evaluation row, G2R must consume the unmasked prediction result: set `M=q_0p5`,
`L=lo_90`, and `U=hi_90` from those three arrays **before** the scorer applies its OOD/support NaN mask. A post-mask
NaN is never a scoring input, and no implementation may silently patch the frozen scorer bytes; support flags remain
separate diagnostics. Retain the v2 NEX, NMD, AMD, and ten readout formulas without renumbering or reweighting.
`support_gate_pass` and `ood_abstain` are diagnostic only: score the whole eligible WOD domain under the
PI-acknowledged extrapolation semantics, including the bounded historical 0/228-in-support result. No InterHub
envelope is built or queried.

## Grid and dependent counts

`RQ014_recovery_lane_v3.json` removes `envelope_axis`. Its declared order is the unchanged two-element sampling axis,
unchanged eight-element temporal axis, unchanged two-element horizon axis, and unchanged ten-element readout axis:
`2 × 8 × 2 × 10 = 320` predictor cells. Each cell retains the unchanged association order RWS, PSP, PPR, giving
`320 × 3 = 960` terminal rows.

Every dependent count is changed in v3 at these machine locations: predictor enumeration; predictor-manifest
prerequisite; common-support cell universe/predicate/count; registered leaderboard count; ledger row index, cell
universe, stored-row count, base order, terminal row/count, and visibility; total rank domain; adaptive-extension base
ledger; selected-recipe prerequisite/source; no-observed verdict; and G2R/G3R summaries. Therefore valid row indices
are `0..959`, the terminal record is row 959, and ranks are exactly `1..960`. The v2 file remains byte-identical at
SHA-256 `c1d3a8c4faeb04871e15d7d1d0f07edfd45b8e6904bdd5ac7e05fa3f1f412d7d`.

G4R must verify and load the exact frozen M3 scorer/manifest/feature-spec bytes. It must independently rewrite
resampling, exact-window state and M3-feature assembly, deviation/readout reduction, rating join, and association
code. It may not reimplement or retrain M3, read screen caches, or fall through to a different ranked recipe.

`M3_ARTIFACT_MISMATCH` is global source drift, not row ineligibility. Exact path containment, no-follow regular-file
opening, size and SHA-256 verification must finish before any input-manifest, materialization-ledger, predictor-cell,
or rating processing. A mismatch aborts with zero cells, zero ledger rows, and zero rating values read.

## Runtime closure and current preflight scope

G2R requires a future, separately reviewed **managed-environment closure v4** that checksum-freezes the scientific
runtime needed by M3, including `joblib`, `numpy`, `pandas`, `scipy`, and `scikit-learn`, and passes its own explicit
gate before any M3 import or scoring. The current stdlib-only v3 runtime is not scientific-compute authority.
Current contract preflight never deserializes or scores M3: it only performs retained-descriptor artifact
verification, repeats the check at job start, and writes the exact path/size/SHA evidence into an immutable receipt.

## Normative A-group byte definitions

The following five rules are normative and replace every former undefined A-group recipe.
Source-line citations in rules 2–4 refer to the read-only verified 42,665-byte historical file
`/share/home/u25310231/ZXC/RQ010B_wod_e2e/code/rq010b_ipv_rating_pilot_20260629/analyze_wod_e2e_ipv_rating_pilot.py`,
SHA-256 `7c60676effd0b3787cba8825790b6648a3de9d462e571677bf12dbb3920cdba2`.

1. **Estimator core tree.** The exact import graph is the historical WOD adapter →
   `src/sociality_estimation/core/ipv_estimation.py` → `core/agent.py` → `planning/utility.py` and
   `planning/Lattice.py`, including the three executable package initializers. The exact seven rows and component
   hashes are in `RQ014_estimator_core_tree_v1p7.json`. `core_tree_sha` is SHA-256 of the `files` array serialized as
   UTF-8 canonical JSON with sorted keys, compact separators, paths sorted by raw UTF-8 bytes, and no terminal LF:
   `ffc83befe2f0e45cccd236965166bb14b71a3f258a49897fef49c0468946fb5e`.
2. **WOD adapter.** `scripts/rq014/wod_ipv_adapter.py`, SHA-256
   `cafb4e4131e0c4069120a785fa851fd813c7bc01afb22a635fd98df3bf7ea106`, is the sole adapter. Its provenance note
   cites the 42,665-byte historical source SHA
   `7c60676effd0b3787cba8825790b6648a3de9d462e571677bf12dbb3920cdba2`; it fixes exact solver mode,
   `history_window=min_observation=point_count-1`, common-window equality, explicit ego route reference, observed
   counterpart reference, and stable log-domain reliability.
3. **Preprocessing.** `scripts/rq014/wod_ipv_preprocessing.py`, SHA-256
   `8e8aa1ffb1123982fc3b61b5cd8f4447384d2cbc16caa3747244021ac077e633`, is the sole state contract. Slice the exact
   closed position window first; then apply first/last one-sided and interior centered differences to position and
   derived velocity, use the 1e-9 moving threshold, normalize `-pi` to `+pi`, earlier-carry/backfill heading within
   that window, and fail all-stationary rows. Source dynamics, derivative halos, cross-window reuse, and extrapolation
   are forbidden, as in `RQ014_envelope_builder_contract_v2.json#/timeline_and_state_contract`.
4. **Reference builder.** `scripts/rq014/wod_reference_builder.py`, SHA-256
   `b1dd28ab49d3bc8546c122df867b984e948d403464cf04635c80d09d9b2d596d`, is the sole scene-level WOD route builder.
   It extracts the historical straight/left/right 12 m-radius, 80 m-extension rule from the same 42,665-byte source;
   one scene reference is shared across all three candidates and the estimator alone prepares it once.
5. **Spearman implementation/version.** `scripts/rq014/spearman_average_midranks.py`, SHA-256
   `c0f035429a86d14b6fd09700e07fd95048b5cbd6296dbba4d4bb8c7f76abff20`, is the exact implementation. Its canonical
   four-key `{package,version,function,options}` manifest is
   `scripts/rq014/spearman_version_manifest_v1.json`, whose exact file SHA-256 is
   `4bfca132554b995672d51773768a40b25c4372f69109eec38f2bb6b2bd5f01a9`. It requires finite equal-length vectors,
   at least three rows and three distinct paired rows, average midranks for exact ties, nonconstant rank vectors, and
   finite Pearson-of-midranks output. Both active registries mirror this same implementation/version digest.

## Active binding contract and X02 disposition

Option (b) is adopted. The ten X02-prefixed bindings are removed from the active requirement; every former X02 hash
site in the v1.6 registries is the literal `LEGACY_INACTIVE_UNBOUND` with a provenance note. X02 cannot enter the
320-cell grid or any materialized active binding ledger.

The exact active count is **9**: four fixed-estimator hashes; three frozen-M3 hashes; and two general Spearman
implementation/version mirror sites (valid-scientific and recovery-extension) constrained equal. The required IDs,
targets, and sole equality are machine-defined in
`RQ014_execution_contract_v1p5.json#/registry_binding_contract`. The registries store the reviewed real hashes;
run-scoped materialization verifies exact equality rather than inventing or overwriting science bytes.

Option (c) fallback is review-only: if reviewers require X02 for a later legacy analysis, a new checksum-bound
amendment must define its missing source, mapping, composite, builder, and input artifacts and explicitly restore its
bindings. No reviewer or operator may reinterpret `LEGACY_INACTIVE_UNBOUND` as a hash, placeholder, or permission.

## Authority and execution boundary

This amendment creates review-candidate science bytes only. It creates no verdict, Formal G1, final bundle, run
specification, rating authorization, or HPC permission. The current central allowlist remains limited to export and
contract preflight. Fresh dual `NO_BLOCKER`, regenerated Formal G1/final bundle, publication, sync, immutable spec,
and validate-only are still required before any permitted submit; G2R/G3R/G4R remain denied.

The adopted reviewer ruling removes InterHub from active G2/G2R inputs and retains its v2 source-manifest contract as
historical provenance only. WOD path-type mapping is scientifically distinct from an InterHub envelope and remains a
separate checksum-pinned rating-blind input under
`RQ014_execution_contract_v1p5.json#/wod_path_type_mapping_manifest_contract`.

## Round-6 byte-change changelog and registry-delta disposition

The reviewer identified twelve substantive v1p6 config-registry pointer changes. Their complete disposition is:

1. `/sequence_contract/state_derivation` is reverted byte-for-byte to v1p5 because the M3 identification does not
   authorize that independent state-rule change. Exact-window preprocessing remains normative only in this amendment,
   the v3 lane, and the separately hashed managed preprocessing module pending review.
2. `/envelope/source` survives because the PI replaced the InterHub envelope with frozen RQ009 M3.
3. `/envelope/form` survives because M3 supplies one median and one calibrated central-90 interval.
4. `/envelope/path_types` is removed because it was an InterHub-envelope axis; WOD path type remains a separately
   pinned input and is not an envelope dimension.
5. `/envelope/quantiles` is removed because fixed named M3 outputs `q_0p5`, `lo_90`, and `hi_90` replace empirical
   InterHub quantile construction.
6. `/envelope/matched_fields` is removed because no InterHub cell matching remains after the M3 fix.
7. `/envelope/human_episode_weighting` is removed because no InterHub episodes enter the active envelope.
8. `/envelope/builder_contract` is removed because no InterHub envelope is built; the v2 builder survives only at
   explicitly cited timeline/reference provenance pointers.
9. `/envelope/envelope_gate` is removed because M3 support/OOD is diagnostic under the PI's extrapolation ruling.
10. `/envelope/support_semantics` is required to state that diagnostics do not mask or drop otherwise eligible rows.
11. `/envelope/wod_transfer_semantics` is required to record the PI-acknowledged out-of-support boundary.
12. `/envelope/frozen_m3` is required to bind the selected scorer, manifest, feature contract, size, and named outputs.

The surviving M3 block now uses the external managed checkpoint path, and `/envelope/quantile_consumption` explicitly
binds pre-OOD-mask outputs. The pre-envelope role text now identifies M3 scoring plus the separately pinned WOD
mapping, and the optional statistical-contract role key now names lane v3; these are direct coherence repairs, not new
science axes. Both v1p6 registries forward-bind the v1.6 preflight Formal-G1 and D1 decision names. All X02 inactive
sentinels remain required by adopted option (b).

Earlier Phase-B changes to `scripts/rq014/materialize_registry.py` are declared in full: its docstring was generalized
from v1.5-only materialization; the active X02 composite check was removed because X02 became inactive/unbound; and a
`VERIFY_PREFILLED_EXACT` source-binding branch was added while retaining the historical
`MATERIALIZE_PLACEHOLDERS` branch and adding fail-closed rejection of unknown modes. Earlier Phase-B changes to
`scripts/rq014/preflight.py` mirrored that exact binding-mode split, imported `get_pointer`, removed active X02
composite validation, and retained placeholder materialization for historical contracts.

Round 6 changes to `scripts/rq014/preflight.py` are also declared in full: import `stat`; add the retained-descriptor
M3 path/containment/symlink/regular/size/hash/identity validator and receipt; add the canonical checksum-bound WOD
path-type mapping validator; require the M3 ref as the first `run_preflight` gate; remove the active InterHub root and
validator call; and return M3 and WOD-mapping evidence. The historical InterHub validator remains addressable but is
not called. `scripts/rq014/materialize_registry.py` receives no Round-6 byte change. The launcher/schema/template and
managed entrypoint propagate the exact M3 block, repeat job-start verification, and make both output receipts read-only.
