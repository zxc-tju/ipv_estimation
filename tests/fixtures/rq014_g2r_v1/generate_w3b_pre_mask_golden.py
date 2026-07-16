#!/usr/bin/env python3
"""Generate the RQ014 A08/A15 portable pre-mask M3 golden."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy
import pandas
import pyarrow
import scipy
import sklearn

from scripts.rq014 import build_wod_m3_anchors as W2
from scripts.rq014 import score_wod_m3_deviations as W3


ROOT = Path(__file__).resolve().parents[3]
PORTABLE_FIXTURE_PATH = ROOT / "tests/fixtures/m3_verifier_portable_fixture.json"
PORTABLE_FIXTURE_SHA256 = (
    "ae62b9fddba53308d319ccef5a70d56a9f0ae243fe009aa3f85e36cb20fcee37"
)
PORTABLE_FIXTURE_SIZE_BYTES = 13_179
PARITY_ATOL = 1e-7
PARITY_RTOL = 0.0
OOD_LABELS = ("distance_abstain", "joint_abstain")
EXPECTED_ENVIRONMENT = {
    "joblib": "1.5.3",
    "numpy": "2.0.2",
    "pandas": "2.3.3",
    "pyarrow": "21.0.0",
    "python": "3.9.6",
    "scikit_learn": "1.6.1",
    "scipy": "1.13.1",
}
GENERATION_COMMAND = (
    "PYTHONPATH=src:. /tmp/rq014_w3_v4_assess/bin/python "
    "tests/fixtures/rq014_g2r_v1/generate_w3b_pre_mask_golden.py "
    "--scorer models/rq009_m3/m3_scorer.joblib "
    "--output tests/fixtures/rq014_g2r_v1/m3_pre_mask_expected.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_nonfinite(token: str) -> Any:
    raise ValueError(f"nonfinite JSON token: {token}")


def strict_load(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=_reject_nonfinite,
    )


def exact_v4_environment() -> dict[str, str]:
    observed = {
        "joblib": joblib.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "pyarrow": pyarrow.__version__,
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
    }
    if observed != EXPECTED_ENVIRONMENT:
        raise RuntimeError(
            f"exact-v4 environment mismatch: expected={EXPECTED_ENVIRONMENT} "
            f"observed={observed}"
        )
    return observed


def portable_rows_to_w2(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Represent portable fixture features as checksumable W2 rows."""

    typed_na = {
        "apet_online_proxy": "F_M3_INPUT_APET_UNDEFINED",
        "closing_ttc_anchor": "F_M3_INPUT_TTC_UNDEFINED",
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        values: list[Any] = []
        for column in W2.M3_INPUT_COLUMNS:
            value = row[column]
            if value is None:
                if column not in typed_na:
                    raise ValueError(f"unexpected portable-fixture null: {column}")
                value = {"kind": "NA", "reason_code": typed_na[column]}
            values.append(value)
        output.append(
            {
                "columns": list(W2.M3_INPUT_COLUMNS),
                "schema_version": "rq014-g2r-m3-input-row-v1",
                "values": values,
            }
        )
    return output


def build_payload(scorer_path: Path) -> dict[str, Any]:
    if sha256_file(PORTABLE_FIXTURE_PATH) != PORTABLE_FIXTURE_SHA256:
        raise RuntimeError("portable M3 fixture SHA-256 drift")
    if PORTABLE_FIXTURE_PATH.stat().st_size != PORTABLE_FIXTURE_SIZE_BYTES:
        raise RuntimeError("portable M3 fixture size drift")
    fixture = strict_load(PORTABLE_FIXTURE_PATH)
    rows = fixture["rows"]
    expected = fixture["expected"]
    if [row["fixture_label"] for row in rows] != [
        row["fixture_label"] for row in expected
    ]:
        raise RuntimeError("portable M3 fixture label alignment drift")

    scores = W3.score_pre_mask_m3(portable_rows_to_w2(rows), scorer_path)
    output_rows: list[dict[str, Any]] = []
    for index, (source, post_mask, score) in enumerate(zip(rows, expected, scores)):
        label = source["fixture_label"]
        if bool(post_mask["support_gate_pass"]) != score.support_gate_pass:
            raise RuntimeError(f"portable support-gate drift for {label}")
        if bool(post_mask["ood_abstain"]) != score.ood_abstain:
            raise RuntimeError(f"portable OOD drift for {label}")
        if label in OOD_LABELS:
            output_rows.append(
                {
                    "fixture_label": label,
                    "hi_90": score.hi_90,
                    "lo_90": score.lo_90,
                    "ood_abstain": score.ood_abstain,
                    "portable_fixture_row_index": index,
                    "q_0p5": score.q_0p5,
                    "support_gate_pass": score.support_gate_pass,
                }
            )
    if tuple(row["fixture_label"] for row in output_rows) != OOD_LABELS:
        raise RuntimeError("portable OOD row universe drift")

    generator_path = Path(__file__).resolve()
    return {
        "generation": {
            "command": GENERATION_COMMAND,
            "environment": exact_v4_environment(),
            "generator_path": str(generator_path.relative_to(ROOT)),
            "generator_sha256": sha256_file(generator_path),
            "scorer_path": "models/rq009_m3/m3_scorer.joblib",
            "scorer_sha256": W3.M3_SCORER_SHA256,
            "scorer_size_bytes": W3.M3_SCORER_SIZE_BYTES,
        },
        "parity_standard": {
            "absolute_tolerance": PARITY_ATOL,
            "relative_tolerance": PARITY_RTOL,
            "reviewed_test_reference": "tests/test_verifier_runtime.py:152",
            "source_fixture_path": str(PORTABLE_FIXTURE_PATH.relative_to(ROOT)),
            "source_fixture_sha256": PORTABLE_FIXTURE_SHA256,
            "source_fixture_size_bytes": PORTABLE_FIXTURE_SIZE_BYTES,
        },
        "rows": output_rows,
        "schema_version": "rq014-g2r-m3-pre-mask-portable-golden-v1",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_bytes(W2.canonical_json_bytes(build_payload(args.scorer)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
