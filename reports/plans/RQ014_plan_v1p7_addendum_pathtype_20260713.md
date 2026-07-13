# RQ014 v1.7 addendum — rating-blind WOD path-type freeze

Date: 2026-07-13
Status: **REVIEW CANDIDATE; PR gate retained; no execution authority**
Scope: the PI's standing science-freeze authorization for a scene-level WOD `CP/HO/MP/F` lookup. This addendum
does not authorize an HPC write, launcher invocation, rating access, G2R, G3R, G4R, or installation under the
managed input root. It supersedes only the v1.7 amendment's formerly unresolved WOD-mapping recipe and its
nine-binding count; every other v1.7 clause remains in force.

## 1. Source and semantic boundary

The only data inputs are the nine published score-stripped bundle files whose file-manifest and sanitization
receipt SHA-256 values are respectively
`4c41172fd16f00d48187df711ad6435063d80c4bbeb81c5e6e07025d63c78ef9` and
`4dfc81056fe97db66d0ba0df04955a9c7c3a5464237010f298f3fa64363911a3`. The transport schema fixes t*=0,
the 0.25 s grids, the history zero owner, and the finite-difference convention
(`reports/plans/RQ014_score_stripped_schema_v1.json:12-65`); it identifies the route-intent and pose fields
(`:68-98,176-189`) and makes `counterpart_tracks.csv` observed-support-only (`:191-210`). The envelope contract
states that intent is exactly `1/GO_STRAIGHT`, `2/GO_LEFT`, or `3/GO_RIGHT`, and identifies past/future, pose and
selected-counterpart XY/timestamps as the rating-blind primitives sufficient for a separately frozen mapping
(`reports/plans/RQ014_envelope_builder_contract_v2.json:26-36`). It fixes the canonical label order and prohibits
runtime inference or fallback after materialization (`:57-77`).

No candidate ordinal, candidate trajectory, detector confidence, scenario label, rating, rank, tie, completeness,
or outcome statistic enters the classifier. `candidate_states.csv` and the other non-consumed bundle files are
still read and byte-verified so the derivation is bound to the complete published nine-file package. The classifier
uses the actual driven ego future, hence emits one scene-level value for `(segment_id,tstar_context_step)`; it is
not a candidate-level relabeling.

## 2. Historical rule adopted normatively

The recovered full479 conflict-geometry source is immutable HPC file
`/share/home/u25310231/ZXC/RQ010B_wod_e2e/code/rq010b_multiframe_tracking_ipv_20260630/analyze_multiframe_tracking_ipv.py`,
57,725 bytes, SHA-256 `0e4891f38766b631fde4f62c23bf878025e6f1b45e44cbf1dd8f54af6b9d2901`.
Its effective direction is the least-squares slope over the last 1.5 s of observed track, or all observations when
fewer than three lie in that window (`:331-371`); its t* state plus effective velocity defines the constant-velocity
future (`:401-410`); and its closest-point angle/lateral/longitudinal classifier and exact thresholds are at
`:467-513`. The later Phase-1 source explicitly trims the selected track to observations at or before t* before
this computation (`/share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/phase1_ipv_build/analyze_phase1_candidate_ipv.py:137-145`,
30,922 bytes, SHA-256 `9a04a012909db815b871a8021f7c94b1c09bd38ca12f2ea7369e7b6c0764fd97`).

The following is the complete normative algorithm. Any implementation difference is source drift.

1. Verify all nine files as regular no-follow files, their exact names, canonical JSON receipts, exact CSV headers,
   registered sizes, SHA-256 values and row counts. Require the sorted unique 479-row blind universe. Malformed or
   drifting source bytes are a **global abort**, not a scene exclusion.
2. For a geometry-available scene require the exact route-intent code/name pair; 16 finite t*-pose elements in
   source row-major order; 16 ordered ego-history rows ending at exactly 0 s; and 20 ordered positive-time
   ego-future rows. These are eligibility gates. Because the transported coordinate frame is already
   `ego_at_tstar` and the schema forbids inferring a transform convention (`RQ014_score_stripped_schema_v1.json:176-189`),
   the pose is verified but not applied a second time. Route intent is likewise verified but does not override
   observed geometry.
3. Retain only the selected counterpart observations with `time_s<=0`. Require one track identity and at least two
   strictly time-ordered finite observations. Fit separate ordinary least-squares slopes `vx,vy` over rows whose
   times are at least `last_observed_time-1.5`; if fewer than three rows qualify, use all retained rows. A zero or
   non-finite denominator makes the scene undecidable.
4. If the last retained row is at t*=0, its XY is the counterpart t* position. Otherwise propagate that last XY to
   t*=0 using that row's observed `vx_mps,vy_mps`. At every ego-future time `t>0`, set counterpart XY to t* XY plus
   `(vx*t,vy*t)`. This is the historical analytic conflict ray (`analyze_multiframe_tracking_ipv.py:401-410`), not
   a transported or emitted extrapolated observation; the input table remains observed-only.
5. Pair this ray with the actual driven ego-future XY. Choose the first sample attaining the minimum Euclidean
   separation. Derive ego direction there by the registered endpoint/central finite-difference rule
   (`RQ014_score_stripped_schema_v1.json:53-64`); normalize it and the fitted counterpart velocity. Speed at or below
   `1e-9` is undecidable.
6. Let `angle` be `acos(clamp(dot(ego_dir,cp_dir),-1,1))` in degrees; let `rel=cp_xy-ego_xy` at the chosen sample;
   `lateral=abs(cross2d(ego_dir,rel))`; and `longitudinal=dot(ego_dir,rel)`. Apply these branches in order, exactly
   matching the historical conditions (`analyze_multiframe_tracking_ipv.py:492-506`):

   - **CP (crossing):** `45 <= angle <= 135`.
   - **HO (opposing/head-on):** `angle > 135` and `lateral <= 5 m`.
   - **MP (merging):** `angle < 45`, `lateral <= 4 m`, and `longitudinal >= -8 m`; this is the historical
     `leading_or_merging` branch and is named MP to match the canonical InterHub vocabulary.
   - **F (following):** `angle < 45`, `lateral <= 4 m`, and `longitudinal < -8 m`; this is the historical
     `same_lane_or_following` branch.

The terminology therefore aligns exactly as `crossing→CP`, `opposing→HO`, `leading_or_merging→MP`, and
`same_lane_or_following→F`. `opposing_nearby` (lateral >5 m), `parallel_nearby` (lateral >4 m), low-motion,
missing-counterpart, and structural-NA scenes are not forced into a class.

The frozen 479-scene derivation maps 254 scenes (`CP=115, HO=90, MP=48, F=1`) and excludes 225. The historical
Phase-2f reference (`CP=90, HO=88, MP=36, F=14`) counted 228 candidate-level trajectories, whereas this table
classifies each scene once from its actual driven ego future. Those different units and future paths make class-count
equality inappropriate—in particular, historical candidate following alternatives need not be the scene's driven
future—so the `F=1` scene count is not by itself evidence of classifier drift.

## 3. UNMAPPED-EXCLUDED and downstream fail closure

Every scene terminates as either one mapped class or an explicit derivation status beginning
`UNMAPPED_EXCLUDED_`. The latter is represented by **absence** from `wod_path_type_mapping.csv`, because the
reviewed manifest schema permits only `CP/HO/MP/F` (`reports/plans/RQ014_execution_contract_v1p5.json:339-350`).
The canonical distribution summary counts every exclusion reason and proves mapped plus excluded equals 479.

At runtime an absent exact lookup becomes `MISSING_WOD_PATH_TYPE`, never an inferred fallback
(`RQ014_envelope_builder_contract_v2.json:67-74`). Lane v3 requires the checksum-bound lookup for every AVAILABLE
scene-feature-horizon group and emits no feature/readout for a terminal group
(`reports/plans/RQ014_recovery_lane_v3.json:194-219`). Thus an UNMAPPED-EXCLUDED scene is excluded at stage F from
all 320 predictor cells and all ten readouts for that scene; the frozen rollup is `INELIGIBLE_BLIND / F_MISSING_WOD_PATH_TYPE`
(`:509-530`), and the 479-scene denominator remains auditable under the ordered attrition rules (`:378-410`).

## 4. Frozen artifacts and review gate

The managed implementation is `scripts/rq014/derive_wod_path_type_mapping.py`; golden inputs/outputs are
`tests/fixtures/rq014_wod_path_type_mapping_golden_v1.json` and
`tests/test_rq014_wod_path_type_mapping.py`. Its stdlib-only runtime, float and serialization contract is frozen in
`scripts/rq014/wod_path_type_mapping_version_manifest_v1.json`. The local review candidate is under
`reports/studies/RQ014_wod_e2e_rating_recovery/02_g2_preflight/wod_path_type_mapping_v1/` and contains exactly
`wod_path_type_mapping.csv`, `manifest.json`, and `distribution_summary.json`. The manifest's mapping path is the
future managed location
`/share/home/u25310231/ZXC/sociality_estimation/inputs/RQ014/wod_path_type_mapping/v1/wod_path_type_mapping.csv`;
creating that root or publishing these bytes remains a later Lead-supervised, no-clobber action after PR/review.

The active v1p6 scientific registry records the source-definition, implementation and mapping-table SHA-256 values,
plus the complete input, fixture, output-manifest and derivation provenance. The active binding count is now exactly
**12**: the prior four estimator, three M3, and two equal Spearman sites remain, and the three new IDs are
`valid.envelope.wod_path_type_mapping.{source_definition_sha256,implementation_sha256,mapping_table_sha256}`.
The execution contract promotes exactly those three new scientific values into the run-scoped registry-binding
ledger. This package neither changes the
two-operation central allowlist nor weakens any Formal-G1, final-bundle, publication, immutable-spec, validate-only,
explicit-confirmation, or rating-access gate.
