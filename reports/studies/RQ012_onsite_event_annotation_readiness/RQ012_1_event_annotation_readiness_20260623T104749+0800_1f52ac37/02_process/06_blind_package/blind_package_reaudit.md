# RQ012A Gate 012-2 Blind Package Re-Audit

Worker: RQ012-W07b-blind-reaudit  
Role: reviewer, independent blind-package re-audit  
Run ID: RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37  
Gate: 012-2 Blind-package integrity  
Verdict: CLEARED for text/issuance surfaces  
Git HEAD: f38c402853d3fcb74c3863487a489750a8f36d28

## Scope

I audited the corrected RQ012A facing files, blind-package manifests, release transport evidence, W05 prior leakage audit, W06 fix report, W07a metadata evidence, and the A02/A03/A04 addendum. I did not read raw media. I did not read or reproduce controlled identity-map contents. Controlled/internal records are referenced only for custody, mode, and aggregate validation-sample construction checks.

## HIGH-1 / HIGH-2 - A02 Proxy Leakage

Verdict: CLOSED.

Evidence:

- Facing files audited: `annotator_01_template.csv`, `annotator_02_template.csv`, `neutral_item_manifest.csv`, and `codebook_issuance_notes.md`.
- `annotator_01_template.csv` and `annotator_02_template.csv` expose only `neutral_item_id` plus blank behavior, timing, confidence, and note fields. They do not expose `scenario`, `scenario_family`, `sampling_stratum`, strata, selection probability, sample role, selection design, `viewing_order`, case-card paths, source paths, filenames, raw paths, team, score, rank, or IPV fields.
- `neutral_item_manifest.csv` has one column, `neutral_item_id`, and 30 neutral items.
- The codebook issuance notes had no hits for the prohibited proxy terms or path-like material references in the re-audit scan.
- Value scans of all three facing CSVs found zero prohibited proxy values and zero path-like values.
- Randomized order is realized only by row sequence. Both annotator files contain the same 30 neutral IDs, but their row sequences differ. The first 10 neutral IDs differ between annotators in the scan, confirming the two issued sequences are independent rather than copied.
- The issuance manifest records separate order-seed rows for `annotator_01` and `annotator_02`; no public order column is emitted.

Facing-file checksums independently recomputed and matched the issuance manifest:

| File | sha256 | Match |
|---|---:|---|
| `annotator_01_template.csv` | `48b1932fdaa75eaed642fc20a9375357206bc03e8f9308dd69d2b8e94f825036` | yes |
| `annotator_02_template.csv` | `eb030ba4159eb173cb6a60b6adb167af57a4351902380409f041baac2f351ab1` | yes |
| `neutral_item_manifest.csv` | `4ea60346c754ea6000cdcb135d4e45be42e60bfd44d60b751c8b199943bb7909` | yes |
| `codebook_issuance_notes.md` | `9edbef8da7a0dd7014db48e43213f146ac22a1c9f3ceed9d1ab547a1411ebcde` | yes |

## HIGH-3 - Metadata

Verdict: CLOSED.

Evidence:

- Independent `xattr` checks on all four facing files showed exactly one extended attribute: `com.apple.provenance`.
- The residual provenance value was identical across the facing files and no other xattrs were present. This is consistent with the W07a finding that the marker is OS-managed and content-free in this host environment.
- The release artifact `release/RQ012A_facing_content_release.zip` contains only the four facing files and no `__MACOSX` entries.
- `zipinfo -v` reports zero-byte extra fields for each archive entry, so source-host xattrs are not carried in the zip payload.
- Streaming each zip entry with `unzip -p` produced the same content sha256 values as the source facing files and as the pre/post release hash evidence.

Judgment: A03 metadata-leakage intent is satisfied for the text/issuance release path. The host filesystem still reports `com.apple.provenance`, but the verified release transport is content-only, preserves bytes exactly, and carries no source-host xattr payload.

## HIGH-4 - A03 Issuance Completeness

Verdict: CLOSED.

Evidence:

- `blind_issuance_manifest.csv` records package version `RQ012A_BLIND_ISSUANCE_CORRECTED_v0.1`.
- It lists 30 neutral item IDs.
- It includes per-file checksum rows, and the four annotator-facing file checksums were independently reverified as matching.
- It includes per-annotator randomized-order seed rows for `annotator_01` and `annotator_02`.
- It records the source-to-neutral custody location as a controlled derived-path location, not as annotator-facing content.
- It references `metadata_stripping_evidence.md`.
- Auditor sign-off is explicitly `UNSIGNED_PENDING_INDEPENDENT_REAUDIT`, with no auditor name or timestamp filled in.
- `blind_issuance_release_transport_manifest.csv` records the content-only release transport, per-facing-file source and round-trip hashes, passing round-trip status, zero-byte extra fields, and unsigned auditor status.

Non-gating caveat: A wider checksum pass over all `file_checksum` rows found that two non-facing support-file hashes in `blind_issuance_manifest.csv` changed after the W07a evidence/status updates (`metadata_stripping_evidence.md` and `phase_status.json`). This does not reopen HIGH-4 for the issued text surfaces because the four facing-file hashes and release-transport hashes match. Refresh those support rows before any archival freeze that treats the manifest as a complete package-level fixity record.

## MEDIUM-1 - Identity-Map Custody

Verdict: CLOSED.

Evidence:

- `controlled_identity_map.csv` exists only under `data/derived/onsite_competition/RQ012_onsite_event_annotation_readiness/RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37/blind_package/`.
- `internal_stratification_record.csv` is in the same controlled derived directory.
- Both controlled/internal files are mode `600` (`-rw-------`) and owned by the local user.
- A filename search under the RQ012 report/run tree found no `controlled_identity_map.csv` or identity-map copy.
- The annotator-facing directory exposes only the two templates, the neutral manifest, and the codebook issuance notes. The facing scan found no source IDs, paths, scenario fields, strata, or controlled-map contents.

I did not reproduce controlled identity-map contents in this report.

## Validation Sample Construction

Verdict: ACCEPTABLE for Gate 012-2.

Aggregate controlled-record checks, without reading IPV values or reporting source identities, show:

- 15 selected validation items.
- 150 validation-frame rows.
- Design label: `scenario_stratified_random_one_case_per_scenario`.
- One recorded random seed across selected validation items.
- Draw count per stratum is 1; selection probability is 0.1000000000.
- Structured selection-design fields do not contain `ipv` or `extreme`.

This supports the prior claim that the validation sample remains random/scenario-stratified rather than IPV-extreme selected. The scenario/stratification details remain internal and are not annotator-facing.

## Remaining Dependency

`BLOCKED_FOR_HUMAN_LABELS` remains open: final neutral media/card issuance awaits the frozen RQ011 analysis universe and an authorized material-preparation worker. No raw media were read and no cards were fabricated in this re-audit. This dependency is allowed to remain open and does not fail Gate 012-2 for the corrected text/issuance surfaces audited here.

## Final Adjudication

- HIGH-1: CLOSED.
- HIGH-2: CLOSED.
- HIGH-3: CLOSED.
- HIGH-4: CLOSED.
- MEDIUM-1: CLOSED.

GATE 012-2 RE-AUDIT: CLEARED
