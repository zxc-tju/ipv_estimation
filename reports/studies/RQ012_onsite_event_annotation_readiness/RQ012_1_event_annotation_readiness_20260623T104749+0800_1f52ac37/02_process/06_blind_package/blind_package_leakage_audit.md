# RQ012A Phase 6 Blind Package Leakage Audit

Worker: RQ012-W05-blind-package-audit  
Role: reviewer / privacy-leakage auditor  
Run ID: RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37  
Gate: 012-2 Blind-package integrity  
Verdict: BLOCKED  
Generated UTC: 2026-06-23T05:28:43.533907+00:00  
Git HEAD: f38c402853d3fcb74c3863487a489750a8f36d28

## Scope And Constraints

Read scope was limited to the specified RQ003 annotation/blind-package surfaces and the directory trees needed to enumerate annotator-facing materials. No raw media content was read. Controlled identity-map contents were not copied, printed, summarized, or reproduced here; only custody location, access mode, existence, and checksums are recorded.

Primary package: `reports/studies/RQ003_nsfc_external_evidence/RQ003_8_nsfc_ipv_validation_codex_20260620T221646+0800_37bb9424`  
Reference package: `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`

## Artifact Enumeration

The version-lock manifest `blind_package_versionlock_manifest.csv` enumerates every audited file with sha256, role, neutral-ID status, ordering status, and metadata-stripping status.

Annotator-facing files audited: 52 total.

- RQ003_8 primary facing files: 5 text/CSV files: codebook, two annotator templates, mechanism manifest, validation manifest.
- RQ003_8 primary missing issued case materials: no `blinded_items/` directory and no `rq003_blind_case_*.zip` material files were present under `01_results/annotations/`.
- RQ003_6 reference facing files: 47 text/CSV/Markdown files: codebook, two annotator templates, two manifests, and 42 blinded item cards.
- Media/package files under audited annotation trees: 0. Therefore thumbnail/media-index metadata for actual issued videos, trajectories, zips, or images cannot be version-locked from these packages.

Internal files audited or checked: 17 total, including anonymization audits, status files, artifact manifests, merge/build scripts, and controlled mapping/provenance files. RQ003_8 stores the controlled source-to-neutral map in the derived controlled directory rather than in the primary report package; RQ003_6 stores a controlled identity map under its process directory.

## Leakage Findings

Severity counts: HIGH=4, MEDIUM=1, LOW=0, INFO=3.

### HIGH 1 - Unapproved A02 scenario and strata proxies are annotator-facing

Addendum A02 prohibits area/scenario/run IDs, filenames, paths, item ordering, thumbnails, manifest-derived strata, and borrowed/prior annotation files unless explicitly approved in writing as outcome-blind design variables.

Affected facing surfaces include:

- RQ003_8 templates and manifests expose `scenario_family` and `scenario` fields.
- RQ003_8 validation manifest exposes `sampling_stratum`, stratum counts, draw count, and selection probability.
- RQ003_8 codebook states that annotators receive scenario labels and viewing order.
- RQ003_6 templates/manifests/cards expose scenario family/code or scenario-stratum fields.

No approval record was found in the audited RQ003 materials or in the binding addendum for exposing these specific proxies to annotators.

### HIGH 2 - Unapproved A02 ordering, paths, and manifest-derived sample metadata are annotator-facing

Affected facing surfaces include:

- RQ003_8 annotator templates include a `viewing_order` column. The two annotator orders differ and appear randomized, but order itself is an A02 proxy and no approval record was found.
- RQ003_6 annotator templates do not include a randomized-order seed or explicit order field, and both annotator templates have identical row ordering.
- RQ003_6 annotator templates include `case_card_path`; RQ003_6/RQ003_8 manifests expose sample role, selection role, selection design, or strata metadata.
- Neutral package names and card paths do not carry explicit team/score/rank/IPV values, but they remain filename/path surfaces requiring written approval or demonstrable neutral issuance under A02/A03.

### HIGH 3 - Metadata stripping is not demonstrated

All 52 annotator-facing files carry a `com.apple.provenance` extended attribute. These text/CSV/Markdown files do not contain EXIF-style embedded media metadata, but A03 requires evidence of embedded/extended metadata stripping before release. Current metadata status is therefore not clean.

### HIGH 4 - Issuance is not reproducibility-complete

A03 requires a frozen issuance manifest with neutral IDs, source-to-neutral mapping custody, per-file checksums, randomized order seed, package version, metadata-stripping evidence, and auditor sign-off before formal annotation files are accepted.

Current evidence is incomplete:

- RQ003_8 has checksums in an artifact manifest and records order seeds in the build script, but it does not include issued case-material files under the primary annotation directory.
- RQ003_6 has blinded item cards and checksums, but no actual media/package files are present under the audited annotation directory.
- Neither package provides a single A03-complete issuance manifest with package version, source-to-neutral custody, randomized order seed for every issued annotator surface, metadata-stripping evidence, and auditor sign-off.

### MEDIUM 1 - Controlled identity-map containment is procedural, not uniformly permission-enforced

The controlled identity map is sensitive and must remain internal.

- RQ003_8 primary report package does not contain `02_process/12_blind_annotation/controlled_identity_map.csv`. Its controlled source-to-neutral map exists under the derived controlled directory, has owner-only permissions, and is not under `01_results/annotations/`.
- RQ003_6 reference package contains a controlled identity map under its process directory, not under `01_results/annotations/`, but its file mode is not owner-only. Its contents were not read into this audit and are not reproduced here.
- Annotator-facing cards/templates refer generically to controlled mapping or source paths but do not expose the map contents. If a full run/process directory were distributed, the RQ003_6 map would be reachable by path and file permissions.

### INFO 1 - No explicit team/score/rank/IPV value leakage found in public templates

The sanitized CSV/Markdown scans did not find explicit team identity, official score, rank, area-rank, or numeric IPV value columns in the annotator templates. Label fields in both annotator templates are blank; no simulated labels or borrowed prior labels were found in the templates.

### INFO 2 - Script behavior does not emit explicit secrets into templates

The RQ003_8 build script writes explicit team/area/rank/path fields only to controlled outputs, while public outputs omit explicit team, official score, rank, raw source path, and numeric IPV fields. However, the same script intentionally emits scenario labels, strata, viewing order, and selection-design metadata into public outputs, which are blocked under A02 without approval.

The RQ003_8 merge script aligns two completed human annotation CSVs and does not join score/IPV data or compute event-IPV tests. The RQ003_6 merge script refuses empty/simulated inputs and computes agreement only after completed labels; it does not emit score/IPV data to annotator-facing outputs.

### INFO 3 - Validation sample construction is random/scenario-stratified, not IPV-extreme selected

Based on public manifests, status files, and build-script logic without reading IPV values:

- RQ003_8 validation sample: 15 rows, scenario-stratified random, one case per scenario, fixed seed recorded, per-row selection probabilities recorded, and `selected_by_ipv_or_outcome_extreme` recorded as no.
- RQ003_6 validation sample: 30 rows, scenario-stratified random, two per scenario, fixed seed recorded in status, and the package audit says no IPV or official-outcome extremes were used.

This satisfies the random/scenario-stratified construction check, but the public exposure of scenario strata remains an A02 proxy leakage issue.

## Per-Surface Audit Summary

- Contents: explicit team/score/rank/IPV values not found in facing templates; A02 proxy columns and values are present in facing templates/manifests/cards.
- Filenames: neutral-looking IDs/package names are used, but filename/path surfaces are not accompanied by A02 written approval or full A03 issuance evidence.
- Paths: RQ003_6 templates expose local case-card paths; no raw source paths were found in facing files. Controlled map paths are internal, but RQ003_6 permissions are not owner-only.
- Directory ordering: RQ003_8 annotator template order appears randomized and differs between annotators, but exposes `viewing_order`; RQ003_6 annotator templates have identical row order and no randomized-order evidence.
- Metadata: all facing files have `com.apple.provenance` xattrs; stripping evidence is absent.
- Thumbnails/media-index metadata: no actual media/package files are present under audited annotation trees, so thumbnail/media metadata cannot be audited/version-locked.
- Prior/borrowed labels: no completed labels found; templates are blank.

## Gate 012-2 Verdict

BLOCKED.

Prioritized fix list:

1. Produce a new blind issuance package that removes or explicitly approves A02 proxies: scenario labels/codes, strata, viewing order, manifest-derived sample-role/selection metadata, and annotator-facing paths.
2. Generate actual annotator-facing material packages/cards under neutral IDs and include them in the version-lock manifest; include package version, neutral-item list, source-to-neutral custody location, per-file checksums, randomized-order seeds, and auditor sign-off.
3. Strip extended metadata from every annotator-facing file and record stripping evidence. For media/package files, audit metadata/thumbnails without reading raw content.
4. Keep controlled identity maps outside annotator-facing/package trees with owner-only permissions; fix the RQ003_6 reference-map permission if it remains in any distribution path.
5. Re-run Gate 012-2 only after the corrected package is frozen. Do not accept formal annotation files until the corrected A03 manifest is clean.
