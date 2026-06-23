# RQ012A W17b Annotation Merge-Validation Tests

Run ID: `RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37`
Worker: `RQ012-W17b-merge-neardup`
Status: `PASS`

These tests are merge-validation guardrails only. The adversarial inputs are
quarantined rejection-test fixtures, not human annotations, not analysis labels,
and not inputs to agreement or event-IPV association.
Near-duplicate, timing, and template-order anomalies are quarantined for
coordinator review; quarantine is not acceptance and is not agreement-ready.

| Fixture | Expected result | Actual result | Assert match |
|---|---|---|---|
| valid_structural_blank | PASS_STRUCTURAL_ONLY | PASS_STRUCTURAL_ONLY | PASS |
| adversarial_empty_template_no_rows | empty_submission | empty_submission | PASS |
| adversarial_all_blank_labels_completed_mode | incomplete_required_fields | incomplete_required_fields | PASS |
| adversarial_copied_duplicate_file | copied_duplicate_submission | copied_duplicate_submission | PASS |
| adversarial_simulated_labels_no_provenance | provenance_attestation_missing | provenance_attestation_missing | PASS |
| adversarial_wrong_neutral_item_ids | item_id_set_mismatch | item_id_set_mismatch | PASS |
| adversarial_incomplete_required_fields | incomplete_required_fields | incomplete_required_fields | PASS |
| adversarial_identity_proxy_leakage | protected_identity_or_proxy_leakage | protected_identity_or_proxy_leakage | PASS |
| adversarial_near_duplicate_label_vectors | near_duplicate_label_pattern | near_duplicate_label_pattern | PASS |
| adversarial_identical_timing_clone | implausible_completion_timing | implausible_completion_timing | PASS |
| adversarial_template_order_clone | template_order_clone | template_order_clone | PASS |
| adversarial_controlled_channel_duplicate_submitter | controlled_channel_evidence_anomaly | controlled_channel_evidence_anomaly | PASS |

## No Agreement Or Association Computed

- `agreement_computed` was false for every validation result.
- `event_ipv_association_computed` was false for every validation result.
- `merge_or_agreement_output_created` was false for every validation result.
- Valid blank files passed only the structural gate with test-fixture provenance;
  they were not accepted as real human labels.
- Human-label status remains `BLOCKED_FOR_HUMAN_LABELS`.
