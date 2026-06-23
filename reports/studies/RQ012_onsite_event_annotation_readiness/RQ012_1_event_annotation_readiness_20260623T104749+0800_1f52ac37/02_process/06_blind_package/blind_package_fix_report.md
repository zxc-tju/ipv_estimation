# Corrected Blind Issuance Package Fix Report

Worker: RQ012-W06-blind-package-fix  
Role: implementer (corrected blind issuance package)  
Run ID: RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37  
Gate: 012-2  
Package version: `RQ012A_BLIND_ISSUANCE_CORRECTED_v0.1`  
Generated UTC: 2026-06-23T05:47:23.607498+00:00  
Git HEAD: `f38c402853d3fcb74c3863487a489750a8f36d28`

## Source And Write Boundaries

- Source materials were read from RQ003_8 annotation/blind-package files and controlled derived files only.
- RQ003 was not modified by this worker.
- Paper repository was not read or written.
- Outputs were written only under the requested RQ012 facing, package, and derived blind-package paths.

## Audit Fix Mapping

1. HIGH 1/2 A02 proxy removal: corrected annotator templates contain only `neutral_item_id` plus blank behavior/timing/confidence/note fields. `neutral_item_manifest.csv` contains only `neutral_item_id`. The annotator-facing codebook issuance notes contain neutral labeling instructions only. Scenario, stratum, selection probability, sample role/design, order columns, material-package names, source paths, and case-card paths are absent from facing files.
2. Internal retention: scenario, sampling-stratum, stratum-count, selection-probability, selection-role/design, and source mapping details are retained only in `data/derived/onsite_competition/RQ012_onsite_event_annotation_readiness/RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37/blind_package/controlled_identity_map.csv` and `data/derived/onsite_competition/RQ012_onsite_event_annotation_readiness/RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37/blind_package/internal_stratification_record.csv` with owner-only permissions.
3. Randomized order: `annotator_01_template.csv` and `annotator_02_template.csv` have independently randomized row sequences. Seeds are recorded in the issuance manifest and the internal stratification record; no facing order column is emitted.
4. Metadata stripping: attempted with `xattr -c`, `xattr -d com.apple.provenance`, and `removexattr(2)`. The environment immediately re-reports `com.apple.provenance`; evidence and the platform probe are in `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37/02_process/06_blind_package/metadata_stripping_evidence.md`. This fix remains partial until an unsandboxed release host clears the attribute.
5. A03 issuance completeness: `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37/02_process/06_blind_package/blind_issuance_manifest.csv` records package version, neutral-item list, per-file checksums, per-annotator randomization seeds, source-to-neutral custody path only, metadata evidence reference, and unsigned auditor sign-off placeholders. Checksums are valid for the current files.
6. MEDIUM 1 identity-map custody: the controlled identity map exists only under the RQ012 derived blind-package directory and is chmod 600. No map contents are reproduced in report or facing files.

## Media Issuance Scope

Final media/card issuance remains `BLOCKED_FOR_HUMAN_LABELS`: RQ003_8 did not contain actual issued case-material packages under its annotation tree, and this worker did not read raw media or fabricate cards. A later authorized material-preparation worker should issue neutral-ID media/cards after the frozen RQ011 analysis universe is available and should add those files to a new checksum/metadata-stripping manifest.

## Phase 7 Codebook Note

The RQ003 source codebook contains language instructing annotators to receive scenario labels and viewing order. Phase 7 should replace that language with neutral-item-only issuance wording. This fix does not fully rewrite the codebook.
