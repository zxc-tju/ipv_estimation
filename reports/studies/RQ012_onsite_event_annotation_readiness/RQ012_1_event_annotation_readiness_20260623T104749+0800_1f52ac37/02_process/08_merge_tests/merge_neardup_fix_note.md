# RQ012 W17b Merge Near-Duplicate Fix Note

Run ID: `RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37`

## V08 Mapping

Red-team V08 found that merge validation rejected byte-identical copied files
and same submitted IDs, but did not screen near-duplicate or collusive
submissions.

`merge_validate.py` now runs a near-duplicate / collusion quarantine screen
after schema/provenance validation and before any agreement-ready handoff. Byte
identical files still use the existing hard rejection path
`copied_duplicate_submission`.

## Added Quarantine Checks

- `near_duplicate_label_pattern`: aligned neutral-item label+confidence cell
  similarity exceeds the configured threshold.
- `near_duplicate_notes`: paired nonblank note text similarity exceeds the
  configured threshold.
- `implausible_completion_timing`: provenance reports implausibly short
  completion duration or identical start/end completion windows.
- `template_order_clone`: both submissions use the same neutral-item row order
  where the issued annotator templates should differ.
- `controlled_channel_evidence_anomaly`: sidecar identity is not independently
  supported by controlled-channel submitter or receipt evidence.

## Default Thresholds

- Label+confidence cell similarity: `0.95`
- Minimum comparable nonblank label+confidence cells: `20`
- Free-text note similarity: `0.95`
- Minimum paired note characters: `40`
- Minimum completion duration: `300` seconds
- Identical timing tolerance: `1` second
- Item-order similarity: `1.0`

These defaults are exposed through `merge_validate.py` CLI options and through
the `NearDuplicateConfig` object for tests or coordinator-controlled runs.

## Anti-Fabrication Boundary

All new near-duplicate, timing, order, and channel fixtures are quarantine-test
inputs only. They are not real human labels, not analysis labels, and are never
used for agreement or event-IPV association. Human-label status remains
`BLOCKED_FOR_HUMAN_LABELS`.
