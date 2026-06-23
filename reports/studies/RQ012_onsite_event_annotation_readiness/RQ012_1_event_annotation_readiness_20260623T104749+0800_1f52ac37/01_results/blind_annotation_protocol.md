# RQ012A Blind Annotation Protocol

Run ID: `RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37`
Worker: `RQ012-W13-annotation-protocol`
Phase: phase8_9a
Status: `FROZEN_BEFORE_HUMAN_LABELS`
Human-label status: `BLOCKED_FOR_HUMAN_LABELS`

This protocol freezes SPEC W6 before any formal labels exist. It defines how two independent human annotations may be collected later. It does not create labels, compute agreement, adjudicate disagreements, or authorize event-IPV analysis.

## Binding Inputs

- Binding addendum: `02_process/01_plan_review/plan_resolution_v0_1.md`
- Annotation codebook: `01_results/annotation_codebook_v2.md`
- Annotator-facing directory: `01_results/annotations/`
- Issuance manifest: `02_process/06_blind_package/blind_issuance_manifest.csv`
- Verified release-transport evidence: `02_process/06_blind_package/blind_issuance_release_transport_manifest.csv`

Version lock for formal issuance:

- Package version: `RQ012A_BLIND_ISSUANCE_CORRECTED_v0.1`
- Allowed release transport: content-only zip export recorded as `02_process/06_blind_package/release/RQ012A_facing_content_release.zip`
- Current Gate 012-2 status: cleared for corrected text/issuance surfaces, with remaining human-label and final-material dependencies.
- If any annotator-facing byte changes before release, the package version, checksums, release-transport manifest, and protocol references must be refreshed before accepting labels.

## Roles

At least two real, independent human annotators are required.

- `annotator_01`: first independent human annotator. Uses only the `annotator_01_template.csv` order and the version-locked blind materials.
- `annotator_02`: second independent human annotator. Must be a different real person from `annotator_01`. Uses only the `annotator_02_template.csv` order and the version-locked blind materials.
- `annotation_coordinator`: custody and process controller. Issues materials, verifies attestations, receives submissions, hashes raw files, logs provenance, and enforces the no-contact and no-mutual-access rules. The coordinator does not fill labels and does not alter raw label files.
- `agreement_statistician`: later analysis role that computes independent agreement only after both raw human submissions are saved append-only. This role does not adjudicate before agreement is complete.
- `adjudicator`: later disagreement-resolution role used only after agreement statistics have been computed and archived. The adjudicator writes a separate adjudicated file and does not edit raw annotator files.

No model, simulated labeler, prior annotation file, copied template, or duplicate human submission may substitute for either required human annotator.

## Blinding And Materials

Annotators may receive only blinded, neutral-ID materials:

- neutral item IDs;
- the assigned blank template;
- the version-locked codebook and issuance notes;
- approved neutral visual or trajectory material when final material issuance is authorized.

The following A02 proxy channels are excluded unless a later written approval specifically authorizes one as outcome-blind for this exact use:

- area IDs, scenario IDs, run IDs, filenames, paths, item ordering columns, thumbnails, manifest-derived strata, prior or borrowed annotation files;
- team identity, official score, rank, area-rank, IPV outputs, score-derived fields, or any event-IPV result;
- source-to-neutral identity-map contents or controlled stratification records.

The row sequence in each assigned template may be used as the review sequence. The sequence itself must not be exposed as a separate public order column.

## Training Phase

Training and formal annotation are separate phases.

Training may use only fictional practice items, neutral-ID stubs, or explicitly training-only examples. Training keys, demonstrations, and discussion notes are not evidence for any formal item and must not be copied into formal templates.

Training completion requirements:

- each annotator confirms they understand the codebook, endpoint-eligibility tiers, confidence scale, insufficient-evidence handling, and no-outside-information rule;
- practice discussion may clarify the codebook but must not reveal formal item provenance, score/IPV information, or another annotator's formal labels;
- training artifacts remain marked `training_only` and outside formal raw-label custody.

## Formal Phase

Formal labels may begin only after the coordinator confirms:

- package version and file checksums match the issuance and release-transport manifests;
- final annotator-facing materials expose only neutral IDs and approved blind content;
- both annotators have signed or otherwise recorded the required human attestation;
- annotators have no access to each other's templates, labels, notes, or submission status beyond generic scheduling logistics;
- the controlled submission channel is ready and audit logging is active.

During formal annotation:

- annotators work independently and do not communicate about formal items;
- annotators do not use outside information, hidden provenance, public results, prior labels, models, or assistance from another person;
- annotators fill only their own assigned template;
- uncertainty is recorded through the codebook rules, confidence, event-time blanks, and concise notes, not by consulting extra context.

Until two real, independently submitted human label files exist and pass provenance checks, the run status remains `BLOCKED_FOR_HUMAN_LABELS`.

## Raw Label Preservation

Raw independent labels are permanent records.

- On receipt, the coordinator stores each submitted file in an append-only raw-label custody location.
- The coordinator records receipt timestamp, submitter identity, submission channel, file name, byte size, sha256, and any available file-provenance metadata.
- Raw files must never be overwritten, edited, normalized in place, sorted in place, or corrected in place.
- Any schema-cleaned, merged, agreement-ready, or adjudicated file must be a derived artifact with its own hash and provenance record.
- Corrections after submission require a new submitted file version and a written reason; the prior raw version remains preserved.

## A04 Anti-Simulation Controls

Every formal label file must pass provenance checks as well as content checks. A non-empty, schema-valid CSV is not sufficient evidence of real human annotation.

Required controls:

- Annotator attestation: each annotator attests that they are a real human, worked independently, used only authorized blind materials, did not use AI/model-generated labels, did not copy or borrow labels, and did not coordinate with the other annotator.
- Controlled submission channel: formal submissions are accepted only through the coordinator-designated channel. The channel must bind a submitter identity or coordinator-verified handoff to each received file.
- File provenance and custody log: the coordinator records receipt metadata, hashes, channel evidence, custody movements, and verifier notes.
- Duplicate/model-screening: the coordinator checks for wrong-channel submissions, missing attestations, identical file hashes, identical or near-identical label patterns, impossible completion timing, template/order mismatch, non-assigned neutral IDs, and other anomalies suggesting duplicate, model, copied, or borrowed submissions.
- Exception handling: any failed provenance check blocks acceptance unless a written exception is approved before agreement computation. Rejected submissions remain logged but are not treated as real human labels.

## Disagreement And Adjudication Order

The order is frozen and mandatory:

1. Issue version-locked blind materials through the verified transport.
2. Receive `annotator_01` and `annotator_02` submissions independently through the controlled channel.
3. Preserve both raw label files append-only and record provenance/hashes.
4. Validate schema and provenance without altering raw files.
5. Compute independent agreement under `01_results/agreement_analysis_protocol.yaml`.
6. Archive the independent agreement report and agreement-ready derived tables.
7. Only after steps 1-6 are complete, open adjudication.
8. Adjudicate disagreements in this order: provenance/schema exceptions, insufficient-evidence and unrelated-failure status, endpoint-eligible label presence/absence, construct-proximal descriptor presence/absence, event timing windows, confidence and note clarifications.
9. Save adjudicated decisions in a separate derived file with reasons, adjudicator identity, timestamp, and hash.
10. Preserve raw independent labels and independent agreement results permanently, even if adjudication changes the downstream consensus file.

Adjudication cannot precede saved independent agreement statistics. Adjudicated labels cannot be used to backfill or modify the independent agreement results.

## Downstream Lock

This protocol does not authorize event-IPV analysis. Later RQ012B work remains blocked unless the authorization chain in `agreement_analysis_protocol.yaml` is satisfied and explicitly recorded.
