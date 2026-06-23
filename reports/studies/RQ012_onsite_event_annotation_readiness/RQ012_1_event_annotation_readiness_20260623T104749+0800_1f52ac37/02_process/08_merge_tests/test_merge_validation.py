#!/usr/bin/env python3
"""Guardrail tests for the RQ012A merge-validation gate.

All generated CSV/JSON files under fixtures/ are test-only rejection or
structural fixtures. They are not human labels, not analysis labels, and are
never used for agreement or event-IPV computation.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


MERGE_DIR = Path(__file__).resolve().parent
RUN_ROOT = MERGE_DIR.parents[1]
RESULTS_DIR = RUN_ROOT / "01_results"
FACING_DIR = RESULTS_DIR / "annotations"
FIXTURE_DIR = MERGE_DIR / "fixtures"
TEST_RESULTS_DIR = MERGE_DIR / "test_results"
RAW_ARCHIVE_DIR = MERGE_DIR / "raw_input_archive"
EXPECTED_ITEMS = FACING_DIR / "neutral_item_manifest.csv"
REPORT_MD = RESULTS_DIR / "annotation_merge_validation_tests.md"
PHASE_STATUS_JSON = MERGE_DIR / "phase_status.json"

sys.path.insert(0, str(MERGE_DIR))
import merge_validate  # noqa: E402


RUN_ID = merge_validate.RUN_ID
REQUIRED_COLUMNS = merge_validate.REQUIRED_COLUMNS
BEHAVIOR_COLUMNS = merge_validate.BEHAVIOR_COLUMNS
CONFIDENCE_COLUMN = merge_validate.CONFIDENCE_COLUMN


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return [{key: (value if value is not None else "") for key, value in row.items()} for row in reader]


def write_rows(path: Path, rows: Iterable[Mapping[str, str]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or list(REQUIRED_COLUMNS)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def fixture_provenance(role: str, case_name: str, extra: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    provenance: Dict[str, object] = {
        "annotator_role": role,
        "coordinator_verified_submitter_id": f"test-fixture-{case_name}-{role}",
        "adversarial_test_fixture": True,
        "not_real_human_labels": True,
        "structural_gate_only": True,
        "human_attestation": False,
        "provenance_purpose": "RQ012 W8 merge-validation fixture only; not a human label file.",
        "anti_fabrication_notice": (
            "This sidecar supports structural guardrail testing only. It does not "
            "claim a real human attestation and must not be used for agreement."
        ),
    }
    if extra:
        provenance.update(extra)
    return provenance


def write_fixture_provenance(role: str, case_name: str, extra: Optional[Mapping[str, object]] = None) -> Path:
    path = FIXTURE_DIR / f"{case_name}_{role}_provenance.json"
    path.write_text(json.dumps(fixture_provenance(role, case_name, extra=extra), indent=2) + "\n", encoding="utf-8")
    return path


def filled_rows(template_rows: List[Dict[str, str]], *, variant: int, note: str = "") -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for index, source in enumerate(template_rows):
        row = {column: source.get(column, "") for column in REQUIRED_COLUMNS}
        for offset, column in enumerate(BEHAVIOR_COLUMNS):
            row[column] = "1" if (index + offset + variant) % 11 == 0 else "0"
        row[CONFIDENCE_COLUMN] = str(3 + ((index + variant) % 2))
        row["event_start_sec"] = ""
        row["event_end_sec"] = ""
        row["free_text_notes"] = note
        rows.append(row)
    return rows


def patterned_completed_rows(template_rows: List[Dict[str, str]], *, variant: int, note: str = "") -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for index, source in enumerate(template_rows):
        row = {column: source.get(column, "") for column in REQUIRED_COLUMNS}
        for offset, column in enumerate(BEHAVIOR_COLUMNS):
            row[column] = "1" if (index + offset + variant) % 3 == 0 else "0"
        row[CONFIDENCE_COLUMN] = str(1 + ((index + variant) % 5))
        row["event_start_sec"] = ""
        row["event_end_sec"] = ""
        row["free_text_notes"] = note
        rows.append(row)
    return rows


def copied_label_vectors_in_template_order(
    source_rows: List[Dict[str, str]],
    target_template_rows: List[Dict[str, str]],
    *,
    changed_cells: int,
    note: str,
) -> List[Dict[str, str]]:
    source_by_item = {row["neutral_item_id"]: row for row in source_rows}
    copied: List[Dict[str, str]] = []
    for target in target_template_rows:
        source = source_by_item[target["neutral_item_id"]]
        row = {column: source.get(column, "") for column in REQUIRED_COLUMNS}
        row["neutral_item_id"] = target["neutral_item_id"]
        row["free_text_notes"] = note
        copied.append(row)
    for index in range(changed_cells):
        row = copied[index % len(copied)]
        column = BEHAVIOR_COLUMNS[index % len(BEHAVIOR_COLUMNS)]
        row[column] = "0" if row[column] == "1" else "1"
    return copied


def inverted_completed_rows(template_rows: List[Dict[str, str]], *, variant: int, note: str = "") -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for row in patterned_completed_rows(template_rows, variant=variant, note=note):
        for column in BEHAVIOR_COLUMNS:
            row[column] = "0" if row[column] == "1" else "1"
        row[CONFIDENCE_COLUMN] = str(6 - int(row[CONFIDENCE_COLUMN]))
        rows.append(row)
    return rows


def write_readme() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURE_DIR / "README.md").write_text(
        "\n".join(
            [
                "# RQ012 W17b Merge Near-Duplicate Fixtures",
                "",
                "All files in this directory are quarantined test fixtures.",
                "They are not human annotations, not analysis labels, and not inputs",
                "for agreement or event-IPV association. Simulated/fake values exist",
                "only to prove the merge-validation gate rejects unsafe submissions.",
                "",
                "The valid structural pair is blank by design and can pass only with",
                "`structural_only=True` and `allow_test_fixtures=True`; it is still",
                "not accepted as real human labels.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_fixtures() -> Dict[str, Tuple[Path, Path, Optional[Path], Optional[Path]]]:
    write_readme()
    rows_01 = read_rows(FACING_DIR / "annotator_01_template.csv")
    rows_02 = read_rows(FACING_DIR / "annotator_02_template.csv")
    fixtures: Dict[str, Tuple[Path, Path, Optional[Path], Optional[Path]]] = {}

    case = "valid_structural_blank"
    a1 = FIXTURE_DIR / f"{case}_annotator_01.csv"
    a2 = FIXTURE_DIR / f"{case}_annotator_02.csv"
    write_rows(a1, rows_01)
    write_rows(a2, rows_02)
    fixtures[case] = (a1, a2, write_fixture_provenance("annotator_01", case), write_fixture_provenance("annotator_02", case))

    case = "adversarial_empty_template_no_rows"
    a1 = FIXTURE_DIR / f"{case}_annotator_01.csv"
    a2 = FIXTURE_DIR / f"{case}_annotator_02.csv"
    write_rows(a1, [])
    write_rows(a2, rows_02)
    fixtures[case] = (a1, a2, write_fixture_provenance("annotator_01", case), write_fixture_provenance("annotator_02", case))

    case = "adversarial_all_blank_labels_completed_mode"
    a1 = FIXTURE_DIR / f"{case}_annotator_01.csv"
    a2 = FIXTURE_DIR / f"{case}_annotator_02.csv"
    write_rows(a1, rows_01)
    write_rows(a2, rows_02)
    fixtures[case] = (a1, a2, write_fixture_provenance("annotator_01", case), write_fixture_provenance("annotator_02", case))

    case = "adversarial_copied_duplicate_file"
    a1 = FIXTURE_DIR / f"{case}_annotator_01.csv"
    a2 = FIXTURE_DIR / f"{case}_annotator_02.csv"
    write_rows(a1, rows_01)
    write_rows(a2, rows_01)
    fixtures[case] = (a1, a2, write_fixture_provenance("annotator_01", case), write_fixture_provenance("annotator_02", case))

    case = "adversarial_simulated_labels_no_provenance"
    a1 = FIXTURE_DIR / f"{case}_annotator_01.csv"
    a2 = FIXTURE_DIR / f"{case}_annotator_02.csv"
    write_rows(a1, filled_rows(rows_01, variant=1, note="ADVERSARIAL_TEST_FIXTURE_ONLY_NOT_ANALYSIS_LABEL"))
    write_rows(a2, filled_rows(rows_02, variant=2, note="ADVERSARIAL_TEST_FIXTURE_ONLY_NOT_ANALYSIS_LABEL"))
    fixtures[case] = (a1, a2, None, None)

    case = "adversarial_wrong_neutral_item_ids"
    a1 = FIXTURE_DIR / f"{case}_annotator_01.csv"
    a2 = FIXTURE_DIR / f"{case}_annotator_02.csv"
    wrong_rows = [dict(row) for row in rows_01]
    wrong_rows[0]["neutral_item_id"] = "RQ012NI999"
    write_rows(a1, wrong_rows)
    write_rows(a2, rows_02)
    fixtures[case] = (a1, a2, write_fixture_provenance("annotator_01", case), write_fixture_provenance("annotator_02", case))

    case = "adversarial_incomplete_required_fields"
    a1 = FIXTURE_DIR / f"{case}_annotator_01.csv"
    a2 = FIXTURE_DIR / f"{case}_annotator_02.csv"
    incomplete_rows = filled_rows(rows_01, variant=3, note="ADVERSARIAL_TEST_FIXTURE_ONLY_NOT_ANALYSIS_LABEL")
    incomplete_rows[0]["aggressive_intrusion"] = ""
    incomplete_rows[1][CONFIDENCE_COLUMN] = ""
    write_rows(a1, incomplete_rows)
    write_rows(a2, filled_rows(rows_02, variant=4, note="ADVERSARIAL_TEST_FIXTURE_ONLY_NOT_ANALYSIS_LABEL"))
    fixtures[case] = (a1, a2, write_fixture_provenance("annotator_01", case), write_fixture_provenance("annotator_02", case))

    case = "adversarial_identity_proxy_leakage"
    a1 = FIXTURE_DIR / f"{case}_annotator_01.csv"
    a2 = FIXTURE_DIR / f"{case}_annotator_02.csv"
    leaking_columns = list(REQUIRED_COLUMNS) + ["team_id"]
    leaking_rows = [dict(row, team_id="TEAM_SECRET_FIXTURE") for row in rows_01]
    write_rows(a1, leaking_rows, fieldnames=leaking_columns)
    write_rows(a2, rows_02)
    fixtures[case] = (a1, a2, write_fixture_provenance("annotator_01", case), write_fixture_provenance("annotator_02", case))

    case = "adversarial_near_duplicate_label_vectors"
    a1 = FIXTURE_DIR / f"{case}_annotator_01.csv"
    a2 = FIXTURE_DIR / f"{case}_annotator_02.csv"
    near_01 = patterned_completed_rows(rows_01, variant=1, note="")
    near_02 = copied_label_vectors_in_template_order(
        near_01,
        rows_02,
        changed_cells=4,
        note="ADVERSARIAL_NEAR_DUPLICATE_FIXTURE_ONLY_NOT_ANALYSIS_LABEL",
    )
    write_rows(a1, near_01)
    write_rows(a2, near_02)
    fixtures[case] = (a1, a2, write_fixture_provenance("annotator_01", case), write_fixture_provenance("annotator_02", case))

    case = "adversarial_identical_timing_clone"
    a1 = FIXTURE_DIR / f"{case}_annotator_01.csv"
    a2 = FIXTURE_DIR / f"{case}_annotator_02.csv"
    write_rows(a1, patterned_completed_rows(rows_01, variant=2, note=""))
    write_rows(a2, inverted_completed_rows(rows_02, variant=2, note=""))
    identical_timing = {
        "annotation_started_at_utc": "2026-06-23T02:00:00Z",
        "annotation_completed_at_utc": "2026-06-23T02:00:45Z",
        "completion_duration_seconds": 45,
    }
    fixtures[case] = (
        a1,
        a2,
        write_fixture_provenance("annotator_01", case, extra=identical_timing),
        write_fixture_provenance("annotator_02", case, extra=identical_timing),
    )

    case = "adversarial_template_order_clone"
    a1 = FIXTURE_DIR / f"{case}_annotator_01.csv"
    a2 = FIXTURE_DIR / f"{case}_annotator_02.csv"
    write_rows(a1, patterned_completed_rows(rows_01, variant=3, note=""))
    write_rows(a2, inverted_completed_rows(rows_01, variant=3, note=""))
    fixtures[case] = (a1, a2, write_fixture_provenance("annotator_01", case), write_fixture_provenance("annotator_02", case))

    case = "adversarial_controlled_channel_duplicate_submitter"
    a1 = FIXTURE_DIR / f"{case}_annotator_01.csv"
    a2 = FIXTURE_DIR / f"{case}_annotator_02.csv"
    write_rows(a1, rows_01)
    write_rows(a2, rows_02)
    fixtures[case] = (
        a1,
        a2,
        write_fixture_provenance(
            "annotator_01",
            case,
            extra={
                "controlled_channel_submitter_id": "shared-channel-fixture-identity",
                "controlled_channel_receipt_id": "receipt-annotator-01",
            },
        ),
        write_fixture_provenance(
            "annotator_02",
            case,
            extra={
                "controlled_channel_submitter_id": "shared-channel-fixture-identity",
                "controlled_channel_receipt_id": "receipt-annotator-02",
            },
        ),
    )

    return fixtures


def git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=RUN_ROOT.parents[2],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


class MergeValidationGuardrailTests(unittest.TestCase):
    fixtures: Dict[str, Tuple[Path, Path, Optional[Path], Optional[Path]]] = {}
    case_results: List[Dict[str, str]] = []

    @classmethod
    def setUpClass(cls) -> None:
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        RAW_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        cls.fixtures = build_fixtures()
        cls.case_results = []

    @classmethod
    def tearDownClass(cls) -> None:
        expected_rejections = {
            "adversarial_empty_template_no_rows": "empty_submission",
            "adversarial_all_blank_labels_completed_mode": "incomplete_required_fields",
            "adversarial_copied_duplicate_file": "copied_duplicate_submission",
            "adversarial_simulated_labels_no_provenance": "provenance_attestation_missing",
            "adversarial_wrong_neutral_item_ids": "item_id_set_mismatch",
            "adversarial_incomplete_required_fields": "incomplete_required_fields",
            "adversarial_identity_proxy_leakage": "protected_identity_or_proxy_leakage",
        }
        expected_quarantines = {
            "adversarial_near_duplicate_label_vectors": "near_duplicate_label_pattern",
            "adversarial_identical_timing_clone": "implausible_completion_timing",
            "adversarial_template_order_clone": "template_order_clone",
            "adversarial_controlled_channel_duplicate_submitter": "controlled_channel_evidence_anomaly",
        }
        observed_by_fixture = {row["fixture"]: row for row in cls.case_results}
        all_rejections_proven = all(
            observed_by_fixture.get(fixture, {}).get("actual_code") == code
            for fixture, code in expected_rejections.items()
        )
        all_quarantines_proven = all(
            observed_by_fixture.get(fixture, {}).get("actual_code") == code
            and observed_by_fixture.get(fixture, {}).get("actual_status") == "QUARANTINED"
            for fixture, code in expected_quarantines.items()
        )
        valid_passed = observed_by_fixture.get("valid_structural_blank", {}).get("actual_status") == "PASS_STRUCTURAL_ONLY"
        no_agreement = all(row.get("agreement_computed") == "False" for row in cls.case_results)
        verdict = "pass" if all_rejections_proven and all_quarantines_proven and valid_passed and no_agreement else "issues"

        report_lines = [
            "# RQ012A W17b Annotation Merge-Validation Tests",
            "",
            f"Run ID: `{RUN_ID}`",
            "Worker: `RQ012-W17b-merge-neardup`",
            f"Status: `{'PASS' if verdict == 'pass' else 'ISSUES'}`",
            "",
            "These tests are merge-validation guardrails only. The adversarial inputs are",
            "quarantined rejection-test fixtures, not human annotations, not analysis labels,",
            "and not inputs to agreement or event-IPV association.",
            "Near-duplicate, timing, and template-order anomalies are quarantined for",
            "coordinator review; quarantine is not acceptance and is not agreement-ready.",
            "",
            "| Fixture | Expected result | Actual result | Assert match |",
            "|---|---|---|---|",
        ]
        for row in cls.case_results:
            report_lines.append(
                "| {fixture} | {expected} | {actual} | {match} |".format(
                    fixture=row["fixture"],
                    expected=row["expected"],
                    actual=row["actual"],
                    match=row["assert_match"],
                )
            )
        report_lines.extend(
            [
                "",
                "## No Agreement Or Association Computed",
                "",
                "- `agreement_computed` was false for every validation result.",
                "- `event_ipv_association_computed` was false for every validation result.",
                "- `merge_or_agreement_output_created` was false for every validation result.",
                "- Valid blank files passed only the structural gate with test-fixture provenance;",
                "  they were not accepted as real human labels.",
                f"- Human-label status remains `{merge_validate.HUMAN_LABEL_STATUS}`.",
                "",
            ]
        )
        REPORT_MD.write_text("\n".join(report_lines), encoding="utf-8")

        phase_status = {
            "phase": "phase9b_hardening",
            "verdict": verdict,
            "red_team_fixed": ["V08"] if verdict == "pass" else [],
            "tests_pass": verdict == "pass",
            "rejections_proven": [
                {
                    "fixture": fixture,
                    "expected_rejection_code": code,
                    "actual_rejection_code": observed_by_fixture.get(fixture, {}).get("actual_code", ""),
                    "assert_match": observed_by_fixture.get(fixture, {}).get("assert_match") == "PASS",
                }
                for fixture, code in expected_rejections.items()
            ],
            "quarantines_proven": [
                {
                    "fixture": fixture,
                    "expected_quarantine_code": code,
                    "actual_quarantine_code": observed_by_fixture.get(fixture, {}).get("actual_code", ""),
                    "assert_match": observed_by_fixture.get(fixture, {}).get("assert_match") == "PASS",
                }
                for fixture, code in expected_quarantines.items()
            ],
            "valid_structural_pass": valid_passed,
            "no_agreement_or_event_ipv_computed": no_agreement,
            "run_id": RUN_ID,
            "git_head": git_head(),
        }
        PHASE_STATUS_JSON.write_text(json.dumps(phase_status, indent=2) + "\n", encoding="utf-8")

    def run_validation(
        self,
        fixture: str,
        *,
        structural_only: bool = False,
        allow_test_fixtures: bool = False,
    ) -> Dict[str, object]:
        annotator_01, annotator_02, provenance_01, provenance_02 = self.fixtures[fixture]
        result = merge_validate.validate_pair(
            annotator_01,
            annotator_02,
            expected_items=EXPECTED_ITEMS,
            provenance_01=provenance_01,
            provenance_02=provenance_02,
            raw_archive_dir=RAW_ARCHIVE_DIR,
            report_json=TEST_RESULTS_DIR / f"{fixture}.json",
            structural_only=structural_only,
            allow_test_fixtures=allow_test_fixtures,
        )
        self.assertFalse(result["agreement_computed"])
        self.assertFalse(result["event_ipv_association_computed"])
        self.assertFalse(result["merge_or_agreement_output_created"])
        return result

    def record_result(self, fixture: str, expected: str, result: Mapping[str, object]) -> None:
        actual_status = str(result.get("status", ""))
        actual_code = str(result.get("rejection_code", ""))
        if result.get("quarantine_code"):
            actual_code = str(result.get("quarantine_code", ""))
        actual = actual_code if actual_code else actual_status
        assert_match = "PASS" if actual == expected else "FAIL"
        self.case_results.append(
            {
                "fixture": fixture,
                "expected": expected,
                "actual": actual,
                "actual_code": actual_code,
                "actual_status": actual_status,
                "assert_match": assert_match,
                "agreement_computed": str(result.get("agreement_computed")),
                "event_ipv_association_computed": str(result.get("event_ipv_association_computed")),
            }
        )

    def assert_rejection(self, fixture: str, expected_code: str) -> None:
        result = self.run_validation(fixture)
        self.record_result(fixture, expected_code, result)
        self.assertEqual(result["status"], "REJECTED")
        self.assertEqual(result["rejection_code"], expected_code)
        self.assertEqual(result["human_label_status"], merge_validate.HUMAN_LABEL_STATUS)

    def assert_quarantine(self, fixture: str, expected_code: str) -> None:
        result = self.run_validation(fixture, structural_only=True, allow_test_fixtures=True)
        self.record_result(fixture, expected_code, result)
        self.assertEqual(result["status"], "QUARANTINED")
        self.assertEqual(result["quarantine_code"], expected_code)
        self.assertFalse(result["accepted_as_real_human_labels"])
        self.assertEqual(result["human_label_status"], merge_validate.HUMAN_LABEL_STATUS)

    def test_01_valid_schema_pair_passes_structural_gate_only(self) -> None:
        fixture = "valid_structural_blank"
        result = self.run_validation(fixture, structural_only=True, allow_test_fixtures=True)
        self.record_result(fixture, "PASS_STRUCTURAL_ONLY", result)
        self.assertEqual(result["status"], "PASS_STRUCTURAL_ONLY")
        self.assertFalse(result["accepted_as_real_human_labels"])
        self.assertTrue(result["test_fixture_mode"])
        self.assertEqual(result["human_label_status"], merge_validate.HUMAN_LABEL_STATUS)

    def test_02_rejects_empty_template_no_rows(self) -> None:
        self.assert_rejection("adversarial_empty_template_no_rows", "empty_submission")

    def test_03_rejects_all_blank_labels_in_completed_mode(self) -> None:
        self.assert_rejection("adversarial_all_blank_labels_completed_mode", "incomplete_required_fields")

    def test_04_rejects_copied_duplicate_file(self) -> None:
        self.assert_rejection("adversarial_copied_duplicate_file", "copied_duplicate_submission")

    def test_05_rejects_simulated_labels_without_provenance(self) -> None:
        self.assert_rejection("adversarial_simulated_labels_no_provenance", "provenance_attestation_missing")

    def test_06_rejects_wrong_neutral_item_ids(self) -> None:
        self.assert_rejection("adversarial_wrong_neutral_item_ids", "item_id_set_mismatch")

    def test_07_rejects_incomplete_required_fields(self) -> None:
        self.assert_rejection("adversarial_incomplete_required_fields", "incomplete_required_fields")

    def test_08_rejects_identity_proxy_leakage(self) -> None:
        self.assert_rejection("adversarial_identity_proxy_leakage", "protected_identity_or_proxy_leakage")

    def test_09_quarantines_near_duplicate_label_vectors(self) -> None:
        self.assert_quarantine("adversarial_near_duplicate_label_vectors", "near_duplicate_label_pattern")

    def test_10_quarantines_implausible_identical_timing(self) -> None:
        self.assert_quarantine("adversarial_identical_timing_clone", "implausible_completion_timing")

    def test_11_quarantines_template_order_clone(self) -> None:
        self.assert_quarantine("adversarial_template_order_clone", "template_order_clone")

    def test_12_quarantines_controlled_channel_duplicate_submitter(self) -> None:
        self.assert_quarantine(
            "adversarial_controlled_channel_duplicate_submitter",
            "controlled_channel_evidence_anomaly",
        )


if __name__ == "__main__":
    unittest.main()
