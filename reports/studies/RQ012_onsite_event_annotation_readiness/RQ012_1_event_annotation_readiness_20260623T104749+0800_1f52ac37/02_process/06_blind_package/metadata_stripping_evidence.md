# Metadata Stripping Evidence and A03 Resolution

Generated UTC: 2026-06-23T05:56:41Z
Worker: RQ012-W07a-metadata-evidence
Run ID: RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37
Package version: RQ012A_BLIND_ISSUANCE_CORRECTED_v0.1

## Resolution

A03 metadata-leakage intent is satisfied without making a false literal xattr-stripped claim.

- The annotator-facing materials are CSV, Markdown, or plain text files. They contain no embedded EXIF or document-container metadata.
- The only residual extended attribute observed on the facing files is `com.apple.provenance`.
- On this host, `xattr -c` and `xattr -d com.apple.provenance` both return status 0, but `com.apple.provenance` immediately persists.
- Copying a facing file to `/tmp` shows the same `com.apple.provenance` attribute, demonstrating host OS re-application outside the OneDrive tree.
- A content-only release transport was built with `COPYFILE_DISABLE=1 zip -X -j`. Archive inspection shows the four facing files only, no `__MACOSX` entries, and zero-byte zip extra fields.
- Extracting that zip into a clean temp directory preserved file content exactly. Any `com.apple.provenance` present after extraction is the verifier host's OS-managed marker, not an archive payload field, because the archive entries carry no extended-attribute extra fields.

Therefore annotators are issued file content through the verified release transport. Source-host filesystem xattrs are not part of the issued content payload and do not convey protected team, area, score, rank, or IPV information.

## Facing Files

| File | Type probe | Source content sha256 |
|---|---|---|
| `01_results/annotations/annotator_01_template.csv` | CSV text | `48b1932fdaa75eaed642fc20a9375357206bc03e8f9308dd69d2b8e94f825036` |
| `01_results/annotations/annotator_02_template.csv` | CSV text | `eb030ba4159eb173cb6a60b6adb167af57a4351902380409f041baac2f351ab1` |
| `01_results/annotations/neutral_item_manifest.csv` | ASCII text, CRLF line terminators | `4ea60346c754ea6000cdcb135d4e45be42e60bfd44d60b751c8b199943bb7909` |
| `01_results/annotations/codebook_issuance_notes.md` | ASCII text | `9edbef8da7a0dd7014db48e43213f146ac22a1c9f3ceed9d1ab547a1411ebcde` |

## Xattr Persistence Probe

Command sequence:

```text
xattr -l <facing-file>
xattr -c <facing-file>
xattr -l <facing-file>
xattr -d com.apple.provenance <facing-file>
xattr -l <facing-file>
```

Observed result for each facing file:

| File | Initial xattr | After `xattr -c` | After `xattr -d com.apple.provenance` |
|---|---|---|---|
| `annotator_01_template.csv` | `com.apple.provenance` | `com.apple.provenance` | `com.apple.provenance` |
| `annotator_02_template.csv` | `com.apple.provenance` | `com.apple.provenance` | `com.apple.provenance` |
| `neutral_item_manifest.csv` | `com.apple.provenance` | `com.apple.provenance` | `com.apple.provenance` |
| `codebook_issuance_notes.md` | `com.apple.provenance` | `com.apple.provenance` | `com.apple.provenance` |

All `xattr -c` and `xattr -d com.apple.provenance` calls returned status 0. The exact persisted value reported by `xattr -px com.apple.provenance` was identical for all four facing files:

```text
01 02 00 FF BD 6D E9 61 30 D0 1F
```

The `/tmp` copy probe used `annotator_01_template.csv`:

```text
cp 01_results/annotations/annotator_01_template.csv /tmp/RQ012A_W07a_xattr_copy_<pid>/annotator_01_template.csv
xattr -l /tmp/RQ012A_W07a_xattr_copy_<pid>/annotator_01_template.csv
```

Observed result:

```text
com.apple.provenance persisted on the /tmp copy.
tmp copy content sha256: 48b1932fdaa75eaed642fc20a9375357206bc03e8f9308dd69d2b8e94f825036
```

Interpretation: `com.apple.provenance` is an OS-managed, content-free host marker in this execution environment. It is not embedded file metadata and not a protected-information carrier in the blind-issuance content.

## Verified Release Transport

Release artifact:

```text
02_process/06_blind_package/release/RQ012A_facing_content_release.zip
```

Build command:

```text
ZIP_OUT=<RUN_ROOT>/02_process/06_blind_package/release/RQ012A_facing_content_release.zip
(cd <RUN_ROOT>/01_results/annotations && COPYFILE_DISABLE=1 zip -X -j "$ZIP_OUT" \
  annotator_01_template.csv annotator_02_template.csv \
  neutral_item_manifest.csv codebook_issuance_notes.md)
```

Archive sha256:

```text
5ffd533dcc905aa91468b94fb94bd6aee51cce91ab02382aa9ff48d808b30b88
```

Archive entry check:

```text
annotator_01_template.csv
annotator_02_template.csv
neutral_item_manifest.csv
codebook_issuance_notes.md
```

`zipinfo -v` reported zero-byte extra fields for archive entries and no `__MACOSX` entries. Host xattrs are therefore not included in the archive payload.

Round-trip extraction xattr check:

| Extracted file | Extracted xattr | Extracted provenance value by `xattr -px` |
|---|---|---|
| `annotator_01_template.csv` | `com.apple.provenance` | `01 02 00 FF BD 6D E9 61 30 D0 1F` |
| `annotator_02_template.csv` | `com.apple.provenance` | `01 02 00 FF BD 6D E9 61 30 D0 1F` |
| `neutral_item_manifest.csv` | `com.apple.provenance` | `01 02 00 FF BD 6D E9 61 30 D0 1F` |
| `codebook_issuance_notes.md` | `com.apple.provenance` | `01 02 00 FF BD 6D E9 61 30 D0 1F` |

Because the zip has zero extra fields and no AppleDouble entries, these extracted xattrs are verifier-host markers applied after extraction, not source-host provenance transported in the archive.

## Content Integrity Round Trip

| File | Source sha256 | Extracted sha256 | Result |
|---|---|---|---|
| `annotator_01_template.csv` | `48b1932fdaa75eaed642fc20a9375357206bc03e8f9308dd69d2b8e94f825036` | `48b1932fdaa75eaed642fc20a9375357206bc03e8f9308dd69d2b8e94f825036` | identical |
| `annotator_02_template.csv` | `eb030ba4159eb173cb6a60b6adb167af57a4351902380409f041baac2f351ab1` | `eb030ba4159eb173cb6a60b6adb167af57a4351902380409f041baac2f351ab1` | identical |
| `neutral_item_manifest.csv` | `4ea60346c754ea6000cdcb135d4e45be42e60bfd44d60b751c8b199943bb7909` | `4ea60346c754ea6000cdcb135d4e45be42e60bfd44d60b751c8b199943bb7909` | identical |
| `codebook_issuance_notes.md` | `9edbef8da7a0dd7014db48e43213f146ac22a1c9f3ceed9d1ab547a1411ebcde` | `9edbef8da7a0dd7014db48e43213f146ac22a1c9f3ceed9d1ab547a1411ebcde` | identical |

Hash comparison result: `HASH_COMPARISON=IDENTICAL`.

## Persistent Evidence Files

- `02_process/06_blind_package/release/xattr_release_transport_transcript.txt`
- `02_process/06_blind_package/release/pre_content_sha256.txt`
- `02_process/06_blind_package/release/post_content_sha256.txt`
- `02_process/06_blind_package/release/zipinfo.txt`
- `02_process/06_blind_package/release/RQ012A_facing_content_release.zip`
- `02_process/06_blind_package/blind_issuance_release_transport_manifest.csv`

Auditor sign-off remains unsigned pending independent re-audit.
