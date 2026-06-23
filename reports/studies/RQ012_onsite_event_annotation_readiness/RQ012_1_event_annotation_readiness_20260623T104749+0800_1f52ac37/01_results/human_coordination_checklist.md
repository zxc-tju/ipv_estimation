# RQ012A Human Coordination Checklist

Run ID: `RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37`
Status: `BLOCKED_FOR_HUMAN_LABELS` until two real independent human submissions exist and pass provenance checks.

Use this checklist as the operational order for W6/W7. Do not skip ahead.

## 1. Issuance Readiness

- [ ] Confirm the active package version is `RQ012A_BLIND_ISSUANCE_CORRECTED_v0.1`, or record a newer frozen version before release.
- [ ] Confirm all annotator-facing checksums match the issuance manifest.
- [ ] Confirm the release uses the verified content-only transport recorded in `blind_issuance_release_transport_manifest.csv`.
- [ ] Confirm final annotator-facing materials contain only neutral IDs and approved blind content.
- [ ] Confirm A02 proxies are absent: area/scenario/run IDs, filenames, paths, public order columns, thumbnails, manifest-derived strata, prior labels, team identity, score/rank, and IPV outputs.
- [ ] Confirm the source-to-neutral map and any stratification records remain controlled and non-facing.
- [ ] Record coordinator name, timestamp, package version, release artifact hash, and any open caveats.

## 2. Training

- [ ] Assign `annotator_01` and `annotator_02` to two different real human individuals.
- [ ] Provide training-only fictional or neutral-stub examples, not formal items.
- [ ] Review codebook v2, endpoint eligibility, confidence scale, event-time rules, and insufficient-evidence handling.
- [ ] Confirm training materials and notes are marked `training_only`.
- [ ] Obtain each annotator's training completion acknowledgement.

## 3. Formal Independent Labeling

- [ ] Obtain annotator attestations covering human authorship, independence, no model assistance, no borrowed labels, authorized materials only, and no communication about formal items.
- [ ] Issue `annotator_01_template.csv` only to `annotator_01`.
- [ ] Issue `annotator_02_template.csv` only to `annotator_02`.
- [ ] Instruct annotators not to view or request each other's labels, notes, progress, or submission file.
- [ ] Keep coordinator logistics separate from label discussion.
- [ ] Keep status as `BLOCKED_FOR_HUMAN_LABELS` until both real submissions are received and accepted.

## 4. Controlled Submission And Raw Preservation

- [ ] Accept submissions only through the coordinator-designated controlled channel.
- [ ] Log submitter identity, receipt timestamp, channel evidence, filename, byte size, sha256, and provenance metadata.
- [ ] Store each raw submission in append-only custody.
- [ ] Do not edit, sort, normalize, overwrite, or correct raw submissions in place.
- [ ] If a correction is required, receive a new version through the same controlled channel and retain the original raw file.
- [ ] Screen for missing attestation, wrong channel, duplicate hash, near-duplicate response pattern, wrong template order, non-assigned neutral IDs, and impossible completion/provenance anomalies.
- [ ] Reject or quarantine submissions that fail provenance checks; do not treat them as real human labels.

## 5. Independent Agreement

- [ ] Build agreement-ready derived copies from the preserved raw files.
- [ ] Confirm no adjudication discussion has occurred before agreement computation.
- [ ] Compute agreement exactly under `agreement_analysis_protocol.yaml`.
- [ ] Save agreement outputs and logs separately from raw labels.
- [ ] Record agreement software/command, input hashes, output hashes, and statistician identity.
- [ ] Keep raw independent labels and independent agreement outputs immutable after computation.

## 6. Adjudication

- [ ] Open adjudication only after independent agreement statistics are saved.
- [ ] Adjudicate in the frozen order: provenance/schema exceptions, insufficient-evidence and unrelated-failure status, endpoint-eligible label presence/absence, construct-proximal descriptors, event timing, confidence and notes.
- [ ] Write adjudicated decisions to a separate derived file.
- [ ] Record reason, adjudicator identity, timestamp, and source disagreement for each adjudicated change.
- [ ] Do not change raw annotator files or recompute independent agreement from adjudicated labels.

## 7. Authorization For Later RQ012B

- [ ] Confirm Gates 012-0, 012-1, 012-2, and 012-3 have documented PASS status.
- [ ] Confirm frozen ontology, threshold rationale, blind-package audit, and agreement-analysis protocol are present.
- [ ] Confirm two real independent human annotation files exist and passed provenance checks.
- [ ] Confirm agreement has been computed under the frozen protocol before results were viewed.
- [ ] Confirm RQ007, RQ009, and RQ011 freezes are complete and cited.
- [ ] Confirm an explicit recorded authorization exists for RQ012B event-IPV analysis.
- [ ] If any item above is missing, keep downstream event-IPV analysis blocked.
