# RQ025 Plan v0p1 — technical correction for ego contract only

Status: `APPEND-ONLY CORRECTION`
Scope: `RQ025_wp7_consequence`
User approval date: `2026-08-24`

This file supersedes only the ego outcome contract, the ego one-pass check, and the ego output path in v0. All other v0 statements remain unchanged.

## 1. Correction rationale

The accepted ego regeneration contract is not the strict `(anchor, anchor+3s]` ledger. The authoritative accepted-contract window is inclusive `[anchor_frame_index, target_window_end_frame_index]`, with the anchor included.

The strict `(anchor, anchor+3s]` ledger has been marked `invalid` for effect use and must not be treated as the current ego effect input.

## 2. Authority paths

Authoritative ego contract evidence:

- `.codex-fleet/nmi-revision-research-lead/work/RQ025_wp7_consequence/ego_regeneration/accepted_contract/REPORT.md`
- `.codex-fleet/nmi-revision-research-lead/work/RQ025_wp7_consequence/ego_regeneration/accepted_contract/data_health.json`
- `.codex-fleet/nmi-revision-research-lead/work/RQ025_wp7_consequence/ego_regeneration/accepted_contract/RUN_RECEIPT.json`
- `.codex-fleet/nmi-revision-research-lead/work/RQ025_wp7_consequence/ego_regeneration/accepted_contract/source_preflight.json`

Authoritative counterpart evidence:

- `.codex-fleet/nmi-revision-research-lead/work/RQ025_wp7_consequence/counterpart_regeneration/REPORT.md`
- `.codex-fleet/nmi-revision-research-lead/work/RQ025_wp7_consequence/counterpart_regeneration/data_health.json`
- `.codex-fleet/nmi-revision-research-lead/work/RQ025_wp7_consequence/counterpart_regeneration/RUN_RECEIPT.json`
- `.codex-fleet/nmi-revision-research-lead/work/RQ025_wp7_consequence/counterpart_regeneration/source_preflight.json`

## 3. Current accepted ego contract

Current accepted ego TTC window:

- inclusive `[anchor_frame_index, target_window_end_frame_index]`
- anchor included

Full-universe gate, finite counts:

- lower: `472/14099`
- inside: `11669/14099`
- upper: `747/14099`

Full-universe gate, `TTC < 2 s` counts:

- lower: `22/14099`
- inside: `1032/14099`
- upper: `40/14099`

These are the current pre-outcome facts for the accepted ego contract. They supersede the stricter window wording in v0.

## 4. Authoritative effect inputs

Use these effect inputs for the approved consequence analysis:

- ego: `.codex-fleet/nmi-revision-research-lead/work/RQ025_wp7_consequence/ego_regeneration/accepted_contract/ego_outcome_by_product_row_key.parquet`
- counterpart: `.codex-fleet/nmi-revision-research-lead/work/RQ025_wp7_consequence/counterpart_regeneration/counterpart_outcome_by_product_row_key.parquet`

The old v0 output path `ego_contract_outcome_ledger.parquet` is superseded by the accepted-contract ego path above.

## 5. Counterpart boundary correction

Counterpart outcomes inherit the accepted RQ019 non-scripted main boundary.

Rule:

- the non-scripted main ledger is the authoritative effect input for primary consequence analysis;
- any all-row counterpart estimate may appear only as descriptive sensitivity or provenance support;
- any all-row estimate must not replace the non-scripted main boundary;
- scripted rows remain isolated and do not redefine the primary effect contract.

## 6. Execution gates

Proceed only if all gates hold:

1. The ego ledger path is the accepted-contract parquet listed above.
2. The ego window is inclusive and anchor-included.
3. The finite counts remain `472/11669/747` over `14099`.
4. The ego `TTC < 2 s` counts remain `22/1032/40` over `14099`.
5. The counterpart input remains the accepted non-scripted main boundary.
6. No human, protected, causal, NI, equivalence, or paper-edit scope is introduced.

## 7. v0 sentences superseded

The following v0 statements are superseded by this correction:

- v0 §5.1: the implicit strict-window reading is superseded by the inclusive accepted-contract ego window.
- v0 §7: the ego output target line `ego_contract_outcome_ledger.parquet` is superseded by the accepted-contract ego parquet path.
- v0 §8 check 4: “Confirm the ego ledger is future-only” is superseded by the accepted-contract inclusive-window check and the hard gate counts above.

Everything else in v0 remains in force.

## 8. Unchanged boundaries

Unchanged from v0:

- frozen episode/matching counts and parameters
- `protected_data=NONE`
- `human_collection=denied`
- `causal_claim=denied`
- no NI or equivalence
- no paper edits
- no change to the matching contract or the caliper contract

## 9. No placeholders

This correction contains no placeholders, deferred text, or open-ended replacement tokens.
