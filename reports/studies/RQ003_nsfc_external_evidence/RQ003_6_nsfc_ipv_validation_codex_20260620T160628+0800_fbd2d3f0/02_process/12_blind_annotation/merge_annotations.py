#!/usr/bin/env python3
"""Merge two real human blind annotation files.

The script intentionally refuses empty, placeholder, or simulated inputs. It is
safe to keep in the Phase 5 package because it computes agreement only after two
completed human annotation files are supplied explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ALLOWED_LABELS = {
    "aggressive intrusion",
    "appropriate assertiveness",
    "over-yielding-freeze",
    "oscillation",
    "deadlock",
    "smooth reciprocal negotiation",
    "unrelated failure",
}

FORBIDDEN_PLACEHOLDER_TOKENS = {
    "simulated",
    "synthetic",
    "dummy",
    "fake",
    "fabricated",
    "placeholder",
    "test label",
    "example label",
}

REQUIRED_COLUMNS = {
    "blind_item_id",
    "sample_role",
    "primary_label",
    "secondary_label_optional",
    "confidence",
    "evidence_notes",
    "cannot_code_reason",
}


class AnnotationInputError(ValueError):
    """Raised when an annotation file is not a real completed human input."""


def _norm(value: object) -> str:
    return "" if value is None else str(value).strip()


def _contains_forbidden_placeholder(row: dict[str, str]) -> bool:
    joined = " ".join(_norm(v).lower() for v in row.values())
    return any(token in joined for token in FORBIDDEN_PLACEHOLDER_TOKENS)


def read_annotation_file(path: Path, annotator_name: str) -> list[dict[str, str]]:
    if not path.exists():
        raise AnnotationInputError(f"{annotator_name}: file does not exist: {path}")

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise AnnotationInputError(f"{annotator_name}: CSV has no header")
        missing = sorted(REQUIRED_COLUMNS.difference(reader.fieldnames))
        if missing:
            raise AnnotationInputError(
                f"{annotator_name}: missing required columns: {', '.join(missing)}"
            )
        rows = list(reader)

    if not rows:
        raise AnnotationInputError(f"{annotator_name}: CSV has no rows")

    seen: set[str] = set()
    completed_labels = 0
    for row_index, row in enumerate(rows, start=2):
        blind_item_id = _norm(row.get("blind_item_id"))
        if not blind_item_id:
            raise AnnotationInputError(f"{annotator_name}: blank blind_item_id at row {row_index}")
        if blind_item_id in seen:
            raise AnnotationInputError(
                f"{annotator_name}: duplicate blind_item_id {blind_item_id!r}"
            )
        seen.add(blind_item_id)

        if _contains_forbidden_placeholder(row):
            raise AnnotationInputError(
                f"{annotator_name}: placeholder/simulated marker detected at row {row_index}"
            )

        label = _norm(row.get("primary_label"))
        if not label:
            continue
        completed_labels += 1
        if label not in ALLOWED_LABELS:
            raise AnnotationInputError(
                f"{annotator_name}: invalid primary_label {label!r} at row {row_index}"
            )

        secondary = _norm(row.get("secondary_label_optional"))
        if secondary and secondary not in ALLOWED_LABELS:
            raise AnnotationInputError(
                f"{annotator_name}: invalid secondary_label_optional {secondary!r} "
                f"at row {row_index}"
            )

        confidence = _norm(row.get("confidence"))
        if confidence not in {"high", "medium", "low"}:
            raise AnnotationInputError(
                f"{annotator_name}: confidence must be high, medium, or low at row {row_index}"
            )

        notes = _norm(row.get("evidence_notes"))
        cannot_code_reason = _norm(row.get("cannot_code_reason"))
        if not notes and not cannot_code_reason:
            raise AnnotationInputError(
                f"{annotator_name}: completed row {row_index} lacks evidence_notes "
                "or cannot_code_reason"
            )

    if completed_labels == 0:
        raise AnnotationInputError(
            f"{annotator_name}: no completed human labels found; refusing to merge templates"
        )
    if completed_labels != len(rows):
        raise AnnotationInputError(
            f"{annotator_name}: only {completed_labels}/{len(rows)} rows have labels; "
            "complete all rows before agreement is computed"
        )

    return rows


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float | None:
    n = len(labels_a)
    if n == 0:
        return None
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum((counts_a[label] / n) * (counts_b[label] / n) for label in ALLOWED_LABELS)
    if math.isclose(1.0, expected):
        return None
    return (observed - expected) / (1.0 - expected)


def merge_annotations(
    annotator_01: Path,
    annotator_02: Path,
    merged_csv: Path,
    agreement_json: Path,
) -> None:
    rows_01 = read_annotation_file(annotator_01, "annotator_01")
    rows_02 = read_annotation_file(annotator_02, "annotator_02")

    by_id_01 = {_norm(row["blind_item_id"]): row for row in rows_01}
    by_id_02 = {_norm(row["blind_item_id"]): row for row in rows_02}
    if set(by_id_01) != set(by_id_02):
        only_01 = sorted(set(by_id_01).difference(by_id_02))
        only_02 = sorted(set(by_id_02).difference(by_id_01))
        raise AnnotationInputError(
            "blind_item_id sets differ: "
            f"only_annotator_01={only_01[:10]}, only_annotator_02={only_02[:10]}"
        )

    merged_rows: list[dict[str, str]] = []
    labels_01: list[str] = []
    labels_02: list[str] = []
    for blind_item_id in sorted(by_id_01):
        r1 = by_id_01[blind_item_id]
        r2 = by_id_02[blind_item_id]
        label_01 = _norm(r1["primary_label"])
        label_02 = _norm(r2["primary_label"])
        labels_01.append(label_01)
        labels_02.append(label_02)
        merged_rows.append(
            {
                "blind_item_id": blind_item_id,
                "sample_role": _norm(r1.get("sample_role")),
                "annotator_01_primary_label": label_01,
                "annotator_02_primary_label": label_02,
                "primary_label_agree": str(label_01 == label_02),
                "annotator_01_secondary_label_optional": _norm(
                    r1.get("secondary_label_optional")
                ),
                "annotator_02_secondary_label_optional": _norm(
                    r2.get("secondary_label_optional")
                ),
                "annotator_01_confidence": _norm(r1.get("confidence")),
                "annotator_02_confidence": _norm(r2.get("confidence")),
                "annotator_01_evidence_notes": _norm(r1.get("evidence_notes")),
                "annotator_02_evidence_notes": _norm(r2.get("evidence_notes")),
                "annotator_01_cannot_code_reason": _norm(r1.get("cannot_code_reason")),
                "annotator_02_cannot_code_reason": _norm(r2.get("cannot_code_reason")),
            }
        )

    merged_csv.parent.mkdir(parents=True, exist_ok=True)
    agreement_json.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(merged_rows[0].keys())
    with merged_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    n = len(merged_rows)
    agree_count = sum(row["primary_label_agree"] == "True" for row in merged_rows)
    summary = {
        "status": "computed_from_real_human_inputs",
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_items": n,
        "primary_label_agree_count": agree_count,
        "primary_label_percent_agreement": agree_count / n if n else None,
        "primary_label_cohen_kappa": cohen_kappa(labels_01, labels_02),
        "allowed_labels": sorted(ALLOWED_LABELS),
        "annotator_01_label_counts": dict(Counter(labels_01)),
        "annotator_02_label_counts": dict(Counter(labels_02)),
        "guard": "Inputs passed nonempty, nonsimulated, fully labeled human-data checks.",
    }
    with agreement_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator-01", required=True, type=Path)
    parser.add_argument("--annotator-02", required=True, type=Path)
    parser.add_argument("--merged-csv", required=True, type=Path)
    parser.add_argument("--agreement-json", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        merge_annotations(
            annotator_01=args.annotator_01,
            annotator_02=args.annotator_02,
            merged_csv=args.merged_csv,
            agreement_json=args.agreement_json,
        )
    except AnnotationInputError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
