# RQ014 PI/Lead decision — D3 G2R W1 output freeze only

Date: 2026-07-16.

Decision source: `.codex-fleet/rq014-execution-v1p6/board/g2r_w1_decision_record.md`,
5,335 bytes, SHA-256
`80181e38c8f37c9c2f2e3c1d74b754648e773485efa189532e5118bc27883d26`
(the “decision record”). The decision record adopts the recommended drafts in
`.codex-fleet/rq014-execution-v1p6/board/g2r_w1_design_proposal.md`, SHA-256
`56f47c46b2808bf0d8da3c6c06c6f7fbc261f5e05fef62b6dc7eb9acd4b62d7d`,
and the port decisions in
`.codex-fleet/rq014-execution-v1p6/board/g2r_w1_wod_port_design.md`, SHA-256
`0a820ae1268f77822d549e40b1d60a85fbc9856bc481329733c6ffb18cb2a17e`.

## Authorization boundary

This decision authorizes only the W1 authority and artifact-schema bytes named
by `RQ014_plan_v1p8_g2r_output_freeze_amendment_20260716.md`. It does **not**
authorize a runnable G2R operation, run-spec surface, launcher branch, profile,
publication, submission, HPC access, rating access, or any G3R/D4 output. The
registered `rq014_r2_blind_feature_build` operation remains
`DENY_PENDING_ACCEPTED_PREFLIGHT_PILOT_AND_PI_BUDGET`; central authorization is
not changed (`RQ014_execution_contract_v1p5.json:171-182`).

## A01–A18 resolutions

The quoted resolution text below is preserved from the decision record or from
the exact recommended-draft row that the decision record adopts at lines
51–57.

| ID | Adopted resolution |
|---|---|
| A01 | “normalized two-table logical bank” (`g2r_anchor_scores.jsonl` plus `g2r_blind_feature_bank.jsonl`). |
| A02 | “canonical JSONL rows+LF+typed values+whole-file SHA (CSV only where already fixed)”. |
| A03 | “FULL 459,840 slots cell-major then scene/candidate, no silent row deletion”. |
| A04 | “typed FINITE_FLOAT/NA”; finite binary64 JSON, with `-0.0` normalized to `0.0`. |
| A05 | “G2R-only status namespace (no rating/G3 statuses)”. |
| A06 | “one row per canonical cell” in the terminal immutable 320-row predictor manifest. |
| A07 | “A07 (WOD→32 M3 features) was NOT in the committed repo (forensics NOT_FOUND) but IS recoverable by PORTING the RQ012B OnSite M3-anchor builder to WOD. All 25 numeric formulas + 7 categoricals exist in the reference and the committed verifier (anchors.py/features.py).” |
| A08 | “build a thin REVIEWED helper returning unmasked q_0p5 + calibrated lo_90/hi_90 via model.predict_tier_quantiles + calibrated_bounds, WITHOUT changing scorer bytes”. |
| A09 | “require finite strict L<M<U; otherwise status M3_SCORING_NUMERICAL_FAILURE (row-level granular per D10 propagation), PENDING statistics-reviewer confirmation; freeze boundary inclusivity + signed-zero.” |
| A10 | “granular candidate status; all-three scene mask fails; only contract/hash drift is global abort”. The decision record's literal `D10` cross-reference in A09 is retained as a provenance note and this amendment binds it to A10 because no D10 exists. |
| A11 | “bind score-stripped file_manifest, mapping manifest (not table alone), spec-pinned UTC; replace stale v2 pointer via amendment”. |
| A12 | “exact IEEE-754 hex arrays in canonical JSON; five registered fixture pairs”. |
| A13 | “one umbrella manifest plus whole-file/slice SHA; no hash chain”. |
| A14 | “pilot-style pre-runtime abort; runtime FAIL receipt/no DONE; publish outputs only on PASS”. |
| A15 | “regenerate pre-mask goldens for the OOD fixture rows under v4 + independent parity check”. |
| A16 | “strict UTF-8 ID + LF including final LF”; the 9,184-byte 320-ID payload has SHA-256 `db280b77a5fba7e7bb8546da9d2d22337e66c1b1d267d8f8acd281326eaaadee`. |
| A17 | “retain support_gate_pass/ood_abstain per-anchor diagnostic (never for availability)”. |
| A18 | “profile/batch/retry kept OUT of W1 (bound in the later operation wave)”. |

### Verbatim A-resolution record

The decision record's A-resolution bullets are preserved verbatim here:

- A08/A15 = build a thin REVIEWED helper returning unmasked q_0p5 + calibrated lo_90/hi_90 via
  model.predict_tier_quantiles + calibrated_bounds, WITHOUT changing scorer bytes; regenerate pre-mask goldens
  for the OOD fixture rows under v4 + independent parity check.
- A09 invalid-interval = require finite strict L<M<U; otherwise status M3_SCORING_NUMERICAL_FAILURE (row-level
  granular per D10 propagation), PENDING statistics-reviewer confirmation; freeze boundary inclusivity +
  signed-zero. (ordinary NEX/NMD/AMD formulas per recovery_lane_v3:203-225.)
- Format A01-A06,A10-A14,A16-A18 = adopt the design-proposal recommended drafts: normalized two-table logical
  bank; canonical JSONL rows+LF+typed values+whole-file SHA (CSV only where already fixed); FULL 459,840 slots
  cell-major then scene/candidate, no silent row deletion; typed FINITE_FLOAT/NA; G2R-only status namespace
  (no rating/G3 statuses); one umbrella manifest + whole-file/slice SHA (NO hash chain); pilot-style pre-runtime
  abort + runtime FAIL-receipt/no-DONE + publish-only-on-PASS; strict UTF-8 cell-id+LF digest; retain
  support_gate_pass/ood_abstain per-anchor diagnostic (never for availability); profile/batch/retry kept OUT of
  W1 (bound in the later operation wave).

The A07 recovery framing is also verbatim: “A07 (WOD→32 M3 features) was NOT in
the committed repo (forensics NOT_FOUND) but IS recoverable by PORTING the
RQ012B OnSite M3-anchor builder to WOD. All 25 numeric formulas + 7
categoricals exist in the reference and the committed verifier
(anchors.py/features.py).”

The exact status/reason identifiers and exact FAIL-object shapes needed to make
A05/A10/A14 machine-valid were not enumerated in the decision record. W1a
therefore freezes only deterministic schema/governance completions in
`RQ014_g2r_output_contract_v1.json`; they add no rating field, model formula, or
scientific branch. Fixture payload hashes remain explicit `PENDING_W1B`
placeholders and cannot satisfy a future execution gate.

## D1–D8 resolutions

The following is a lookup-oriented restatement of the decision record.

- **D1 counterpart:** consume the exported checksum-bound
  `counterpart_track_id` per segment, pair with C1/C2/C3; missing, multiple, or
  drifted identity is TERMINAL, with no nearest/TTC/name fallback.
- **D2 elapsed time:** `elapsed_time_s` starts at the earliest jointly-supported
  resampled context tick for the sampling branch, shared across candidates when
  pre-t* support is identical, and is passed explicitly even if history is
  cropped.
- **D3 row windows:** preserve exact row counts: M3 context tail 10;
  counterpart estimator H10 (at most 11 positions, minimum observation index
  4); slope tail 5. No H4 target estimation occurs in G2R. Rows are never
  converted to seconds.
- **D4 M3-context alignment:** **A, anchor at tau and history-only through tau,
  is PRIMARY for all eight temporal families**; the lane's family-specific
  focal IPV is comparison `v`. Alignment is an explicit named parameter.
  **C, terminal minus 6 rows and RQ009-training-faithful, is a preregistered
  blind sensitivity.** Primary and sensitivity are fixed while
  `rating_access=NONE` and may never be chosen by rating correlation post-hoc.
  This is a PI-approved WOD transfer/extrapolation, not a claim that each lane
  IPV window recreates the RQ009 five-row target label.
- **D5 geometry tokens:** use the reference per-anchor actual-trajectory
  kinematic heuristic at the M3 context anchor. Retain the frozen WOD
  CP/HO/MP/F path type as a separate lineage/mask field, never require textual
  equality, and bind a diagnostic cross-tab in the golden fixtures.
- **D6 agent-type encoding:** `agent_type_pair='HV;HV'`,
  `vehicle_type_list=['HV','HV']`, `av_included='all_HV'`, plus a vehicle-class
  eligibility check on the selected counterpart. The WOD ego is an AV, but the
  envelope is HV–HV because the research question is whether the ego behaves
  normally within the human population: higher human rating is compared with
  lower deviation from the **human** IPV envelope. This is grounded in the
  RQ010B preregistration HV–HV encoding. The exact serialized M3 token is
  `"['HV', 'HV']"`, preserving the reference token spacing.
- **D7 turn/priority:** reuse the reference kinematic rules: last-at-most-ten
  heading-delta `turn_pair_label`; anchor-relative longitudinal-projection
  `priority_role`; preserve literal `equal` within 2 m, and keep category/OOD
  support diagnostic under lane v3.
- **D8 input-row preimage:** canonical JSON with exact keys
  `{schema_version,columns,values}`; `columns` is the 32-name model-order array;
  `values` contains finite binary64 numbers with `-0.0` normalized to `0.0`,
  exact UTF-8 category strings, or the approved typed-NA object. Serialize with
  CPython 3.9 `json.dumps(sort_keys=True,separators=(',',':'),ensure_ascii=False,
  allow_nan=False)` plus one terminal LF and hash every byte.

### Verbatim D-resolution record

For byte-level provenance, the D-resolution bullets below are copied verbatim
from decision-record lines 15–44; the preceding list is only a lookup-oriented
restatement.

- D6 agent-type encoding = **HV–HV**: agent_type_pair='HV;HV', vehicle_type_list=['HV','HV'], av_included=
  'all_HV', + vehicle-class eligibility check on the selected counterpart. RATIONALE (PI): the WOD ego is an AV,
  but the envelope must be computed HV–HV because the research question is whether the ego behaves NORMALLY
  WITHIN THE HUMAN POPULATION. This is the scientific foundation of the whole rating-recovery (higher human
  rating ↔ lower deviation from the HUMAN IPV envelope). Grounded in RQ010B preregistration HV-HV encoding.
- D5 geometry tokens = **actual-trajectory kinematics** (reference per-anchor heuristic on the M3 context
  anchor); frozen WOD CP/HO/MP/F path-type retained as a SEPARATE lineage/mask field, never text-equal;
  diagnostic cross-tab frozen in golden.
- D4 M3-context alignment = **A (anchor at τ, history-only through τ) PRIMARY** for all 8 temporal families;
  lane's family-specific focal IPV = comparison v. Alignment is an EXPLICIT NAMED PARAMETER in the output
  contract. **C (terminal minus 6 rows, RQ009-training-faithful) = a PRE-REGISTERED BLIND SENSITIVITY** run.
  Primary/sensitivity fixed NOW, rating-blind, on principle — NEVER chosen by rating correlation post-hoc
  (protects the rating_access=NONE guarantee). Record D4 explicitly as a PI-approved WOD transfer/extrapolation,
  not a claim that every lane IPV window reproduces the RQ009 5-row target label.
- D1 counterpart = consume the exported checksum-bound counterpart_track_id per segment, pair with C1/C2/C3;
  missing/multiple/drifted identity = TERMINAL, no nearest/TTC/name fallback (score_stripped_schema:191-210;
  derive_wod_path_type_mapping.py:600-632).
- D2 elapsed_time_s origin = earliest jointly-supported resampled context tick for the sampling branch, shared
  across candidates when pre-t* support identical; passed explicitly even if history cropped.
- D3 window rows = preserve EXACT row counts: M3 context tail 10; counterpart estimator H10 (≤11 positions, min
  obs index 4); slope tail 5. No H4 target estimation in G2R (lane defines candidate IPV windows). Do NOT
  convert rows→seconds (would erase R04N/R10L and change the model input distribution).
- D7 turn/priority = reuse reference kinematic rules: last-≤10 heading-delta turn_pair_label; anchor-relative
  longitudinal-projection priority_role; literal 'equal' within 2 m preserved (not coerced); category/OOD
  support diagnostic under lane v3.
- D8 row hash = canonical JSON {schema_version, columns(=32 names feature_spec_contract:104-137), values}:
  finite binary64 JSON, -0.0→0.0, exact UTF-8 category strings, or W1-approved typed-NA; CPython 3.9
  json.dumps(sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+one LF; hash all bytes.

The exact serialized D6 M3 value is "['HV', 'HV']", preserving the reference
token spacing at board/g2r_w1_wod_port_design.md:417-419.

## Rating-blind protection

G2R ends at the blind 320-cell bank, masks, evidence, and manifests.
`rating_access`, `rating_join`, and `observed_rating_statistics` are forbidden;
rating reads must remain zero. No 960-row RWS/PSP/PPR leaderboard, association,
rank, selected recipe, `recovery_ledger.jsonl`, or recovery-ledger hash chain is
a G2R artifact. D4 primary and sensitivity outputs are fixed before rating
access and are never selected or filtered by a rating-derived quantity.
