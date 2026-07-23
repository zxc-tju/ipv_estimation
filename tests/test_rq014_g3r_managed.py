from __future__ import annotations

import argparse
import csv
import hashlib
import json
import stat
from pathlib import Path

import pytest

from scripts.rq014 import run_managed_g3 as g3


ROOT = Path(__file__).resolve().parents[1]
RECOVERY = ROOT / "reports/plans/RQ014_recovery_lane_v3.json"
FIXTURE = ROOT / "tests/fixtures/rq014_g3r_v1/statistics_and_attrition_goldens.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _json(path: Path, value: object) -> None:
    _write(path, g3.canonical_json_bytes(value))


def _jsonl(path: Path, rows: list[dict]) -> None:
    _write(path, b"".join(g3.canonical_json_bytes(row) for row in rows))


def _csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_case(tmp_path: Path, *, nonfinite: bool = False) -> argparse.Namespace:
    segments = [f"synthetic-{index:02d}" for index in range(5)]
    cells = ["RR3-SYN-00", "RR3-SYN-01"]
    lineage = tmp_path / "lineage"
    scene_path = lineage / "blind_scene_manifest.csv"
    candidate_path = lineage / "candidate_states.csv"
    _csv(
        scene_path,
        ["segment_id", "tstar_context_step", "source_shard_id", "scenario_cluster"],
        [
            {
                "segment_id": segment,
                "tstar_context_step": "10",
                "source_shard_id": f"shard-{index % 2}",
                "scenario_cluster": f"cluster-{index % 2}",
            }
            for index, segment in enumerate(segments)
        ],
    )
    candidate_rows = []
    for segment in segments:
        for ordinal in (1, 2, 3):
            candidate_rows.append(
                {
                    "segment_id": segment,
                    "tstar_context_step": "10",
                    "candidate_ordinal": str(ordinal),
                    "candidate_id": f"C{ordinal}",
                    "geometry_sha256": hashlib.sha256(
                        f"{segment}-{ordinal}".encode()
                    ).hexdigest(),
                }
            )
    _csv(
        candidate_path,
        [
            "segment_id",
            "tstar_context_step",
            "candidate_ordinal",
            "candidate_id",
            "geometry_sha256",
        ],
        candidate_rows,
    )
    file_manifest = lineage / "file_manifest.json"
    _json(
        file_manifest,
        {
            "files": [
                {
                    "relative_path": path.name,
                    "contains_rating": False,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha(path),
                }
                for path in (scene_path, candidate_path)
            ]
        },
    )
    input_manifest = lineage / "input_manifest.json"
    _json(
        input_manifest,
        {
            "entries": [
                {
                    "role": "wod_score_stripped_bundle_manifest",
                    "contains_rating": False,
                    "absolute_path": str(file_manifest),
                    "sha256": _sha(file_manifest),
                }
            ]
        },
    )

    bank = tmp_path / "bank"
    predictors = [
        {"schema_version": "synthetic", "cell_index": index, "cell_id": cell}
        for index, cell in enumerate(cells)
    ]
    features = []
    masks = []
    for cell_index, cell in enumerate(cells):
        for scene_index, segment in enumerate(segments):
            for ordinal, value in zip((1, 2, 3), (1.0, 2.0, 3.0)):
                features.append(
                    {
                        "cell_id": cell,
                        "segment_id": segment,
                        "candidate_ordinal": ordinal,
                        "predictor_value": {
                            "kind": "FINITE_FLOAT",
                            "value": value + cell_index + scene_index * 0.1,
                        },
                        "upstream_status": "AVAILABLE",
                    }
                )
            masks.append(
                {
                    "cell_id": cell,
                    "segment_id": segment,
                    "all_three_available": True,
                    "all_three_deviations_finite": True,
                    "blind_cell_scene_eligible": True,
                    "scene_cell_status": "AVAILABLE",
                }
            )
    _jsonl(bank / "g2r_predictor_manifest.jsonl", predictors)
    _jsonl(bank / "g2r_blind_feature_bank.jsonl", features)
    _jsonl(bank / "g2r_availability_masks.jsonl", masks)
    _csv(bank / "common_support_blind_manifest.csv", ["segment_id"], [{"segment_id": value} for value in segments])
    filler_names = [
        "wod_scene_anchor_domain.csv",
        "wod_scene_anchor_domain_manifest.json",
        "g2r_anchor_scores.jsonl",
        "nc_pretstar_history_only_receipt.json",
    ]
    for name in filler_names:
        _write(bank / name, b"synthetic\n")
    role_paths = {
        "g2r_predictor_manifest": "g2r_predictor_manifest.jsonl",
        "g2r_blind_feature_bank": "g2r_blind_feature_bank.jsonl",
        "g2r_availability_masks": "g2r_availability_masks.jsonl",
        "common_support_blind_manifest": "common_support_blind_manifest.csv",
        "wod_scene_anchor_domain": filler_names[0],
        "wod_scene_anchor_domain_manifest": filler_names[1],
        "g2r_anchor_scores": filler_names[2],
        "nc_pretstar_history_only_receipt": filler_names[3],
    }
    artifacts = {
        role: {
            "relative_path": relative,
            "size_bytes": (bank / relative).stat().st_size,
            "sha256": _sha(bank / relative),
        }
        for role, relative in role_paths.items()
    }
    manifest = bank / "g2r_output_manifest.json"
    _json(
        manifest,
        {
            "operation": g3.PRIOR_OPERATION,
            "run_id": "synthetic-bank",
            "status": "COMPLETE",
            "counts": {"registered_scene_count": 5, "registered_cell_count": 2},
            "lineage": {
                "input_manifest": {
                    "path": str(input_manifest),
                    "size_bytes": input_manifest.stat().st_size,
                    "sha256": _sha(input_manifest),
                }
            },
            "artifacts": artifacts,
        },
    )
    receipt = tmp_path / "bank_receipt.json"
    _json(
        receipt,
        {
            "schema_version": "rq014-r2-blind-feature-build-receipt-v1",
            "operation": g3.PRIOR_OPERATION,
            "run_id": "synthetic-bank",
            "status": "PASS",
            "rating_value_read_count": 0,
            "registered_cell_count": 2,
            "terminal_cell_count": 2,
            "leaderboard_row_count": 0,
            "recovery_ledger_written": False,
            "output_manifest": {"sha256": _sha(manifest)},
        },
    )
    done = tmp_path / "bank_done.json"
    _json(
        done,
        {
            "schema_version": "rq014-managed-operation-done-v1",
            "operation": g3.PRIOR_OPERATION,
            "receipt_sha256": _sha(receipt),
            "status": "PASS",
        },
    )

    ratings = tmp_path / "ratings.csv"
    rating_rows = []
    for row in candidate_rows:
        score = str(4 - int(row["candidate_ordinal"]))
        if nonfinite:
            score = "nan"
        rating_rows.append({**row, "preference_score": score})
    _csv(ratings, g3.RATING_COLUMNS, rating_rows)

    environment = tmp_path / "environment.json"
    _json(
        environment,
        {
            "python_executable": {"sha256": "a" * 64, "version": "Python 3.9.6"},
            "package_versions": {"numpy": "synthetic"},
        },
    )
    implementation = ROOT / "scripts/rq014/run_managed_g3.py"
    code_snapshot = tmp_path / "code_snapshot.json"
    _json(
        code_snapshot,
        {
            "schema_version": "rq014-code-snapshot-v2",
            "git_commit": "a" * 40,
            "files": [
                {
                    "path": "scripts/rq014/run_managed_g3.py",
                    "size_bytes": implementation.stat().st_size,
                    "sha256": _sha(implementation),
                },
                {
                    "path": "tests/fixtures/rq014_g3r_v1/statistics_and_attrition_goldens.json",
                    "size_bytes": FIXTURE.stat().st_size,
                    "sha256": _sha(FIXTURE),
                },
            ],
        },
    )
    return argparse.Namespace(
        run_id="synthetic-g3r",
        git_commit="a" * 40,
        created_at_utc="2026-07-23T00:00:00Z",
        repo_root=ROOT,
        bank_root=bank,
        bank_manifest=manifest,
        bank_manifest_sha256=_sha(manifest),
        bank_receipt=receipt,
        bank_receipt_sha256=_sha(receipt),
        bank_done=done,
        bank_done_sha256=_sha(done),
        ratings_source=ratings,
        ratings_source_size_bytes=ratings.stat().st_size,
        ratings_source_sha256=_sha(ratings),
        recovery_contract=RECOVERY,
        recovery_contract_sha256=_sha(RECOVERY),
        environment_manifest=environment,
        environment_manifest_sha256=_sha(environment),
        code_snapshot=code_snapshot,
        code_snapshot_sha256=_sha(code_snapshot),
        kernel_fixture=FIXTURE,
        kernel_fixture_sha256=_sha(FIXTURE),
        output_root=tmp_path / "outputs",
    )


def _run_synthetic(args: argparse.Namespace):
    return g3.run_g3r_managed(
        args,
        expected_scene_count=5,
        expected_cell_count=2,
        bootstrap_replicates=8,
        expected_bank_run_id=None,
        expected_bank_receipt_sha256_prefix="",
    )


def test_g3r_synthetic_bank_single_join_and_atomic_terminal_publication(tmp_path: Path) -> None:
    args = _synthetic_case(tmp_path)
    receipt, access = _run_synthetic(args)
    assert receipt["status"] == "PASS"
    assert receipt["rating_join"] == "EXACTLY_ONCE"
    assert receipt["terminal_leaderboard_row_count"] == 6
    assert access["rating_value_read_count"] == 15
    assert access["joined_key_count"] == 15
    assert access["source_sha256"] == _sha(args.ratings_source)
    final = args.output_root / "g3r"
    assert final.is_dir()
    assert stat.S_IMODE(final.stat().st_mode) == 0o700
    assert not (args.output_root / ".g3r.private.partial").exists()
    rows = g3._read_canonical_jsonl(final / "recovery_ledger.jsonl")
    assert len(rows) == 6
    assert [row["row_index"] for row in rows] == list(range(6))
    assert [row["association_id"] for row in rows[:3]] == ["RWS", "PSP", "PPR"]
    assert all(row["ledger_status"] == "OBSERVED" for row in rows)
    assert set(receipt["result_artifacts"]) == {
        "recovery_ledger",
        "recovery_ledger_terminal_digest",
        "association_attrition",
        "rank_index",
        "common_support_sensitivity",
        "association_kernel_manifest",
        "association_attrition_manifest",
    }
    receipt_text = json.dumps({"operation": receipt, "access": access})
    assert "preference_score" not in receipt_text
    assert "synthetic-00" not in receipt_text


def test_g3r_null_sha_first_contact_records_governed_source_digest(
    tmp_path: Path,
) -> None:
    args = _synthetic_case(tmp_path)
    expected_sha256 = _sha(args.ratings_source)
    args.ratings_source_sha256 = None
    receipt, access = _run_synthetic(args)
    assert receipt["status"] == "PASS"
    assert access["source_size_bytes"] == args.ratings_source_size_bytes
    assert access["source_sha256"] == expected_sha256
    persisted = json.loads(
        (args.output_root / "g3r/rating_access_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted["source_sha256"] == expected_sha256


def test_g3r_nonfinite_ratings_terminalize_without_run_failure(tmp_path: Path) -> None:
    args = _synthetic_case(tmp_path, nonfinite=True)
    receipt, access = _run_synthetic(args)
    assert receipt["status"] == "PASS"
    assert access["nonfinite_value_count"] == 15
    rows = g3._read_canonical_jsonl(args.output_root / "g3r/recovery_ledger.jsonl")
    assert {row["upstream_status"] for row in rows} == {"RATING_VALUE_NONFINITE"}
    assert {row["ledger_status"] for row in rows} == {
        "INELIGIBLE_RATING_COMPLETENESS"
    }


def test_g3r_pin_drift_is_redacted_and_publishes_no_partial_rows(tmp_path: Path) -> None:
    args = _synthetic_case(tmp_path)
    args.bank_manifest_sha256 = "0" * 64
    receipt, access = _run_synthetic(args)
    assert receipt["status"] == "FAIL"
    assert receipt["failure"] == {
        "kind": "RUNTIME_FAILURE",
        "stage": "INPUT_CONTRACT",
        "failure_class": "INPUT_CONTRACT_FAILURE",
    }
    assert access["rating_value_read_count"] == 0
    assert not (args.output_root / "g3r").exists()
    assert not (args.output_root / ".g3r.private.partial").exists()
    assert "mismatch" not in json.dumps(receipt).lower()


def test_g3r_preexisting_publication_is_not_overwritten_or_disclosed(
    tmp_path: Path,
) -> None:
    args = _synthetic_case(tmp_path)
    final = args.output_root / "g3r"
    final.mkdir(parents=True)
    marker = final / "existing.txt"
    marker.write_text("preserve\n", encoding="utf-8")
    receipt, access = _run_synthetic(args)
    assert receipt["status"] == "FAIL"
    assert receipt["failure"]["stage"] == "INPUT_CONTRACT"
    assert access["rating_value_read_count"] == 0
    assert marker.read_text(encoding="utf-8") == "preserve\n"
    assert not (args.output_root / ".g3r.private.partial").exists()


def test_g3r_synthetic_outputs_are_byte_deterministic(tmp_path: Path) -> None:
    left = _synthetic_case(tmp_path / "left")
    right = _synthetic_case(tmp_path / "right")
    assert _run_synthetic(left)[0]["status"] == "PASS"
    assert _run_synthetic(right)[0]["status"] == "PASS"
    left_files = {
        path.name: path.read_bytes()
        for path in sorted((left.output_root / "g3r").iterdir())
    }
    right_files = {
        path.name: path.read_bytes()
        for path in sorted((right.output_root / "g3r").iterdir())
    }
    assert left_files == right_files


def test_g3r_kernel_fixture_and_rating_join_failure_forms_are_complete() -> None:
    lane = json.loads(RECOVERY.read_text())
    assert len(g3._validate_kernel_fixture(
        fixture_path=FIXTURE,
        fixture_sha256=_sha(FIXTURE),
        recovery_lane=lane,
    )) == 64
    geometry = {("s", ordinal): (10, str(ordinal) * 64) for ordinal in (1, 2, 3)}
    base = {("s", 10, ordinal, str(ordinal) * 64): [float(ordinal)] for ordinal in (1, 2, 3)}
    joined, count = g3._join_ratings_once(segments=["s"], geometry=geometry, ratings=base)
    assert joined["s"][0] == "AVAILABLE" and count == 3
    base[("s", 10, 2, "2" * 64)] = [float("nan")]
    assert g3._join_ratings_once(segments=["s"], geometry=geometry, ratings=base)[0]["s"][0] == "RATING_VALUE_NONFINITE"
    del base[("s", 10, 2, "2" * 64)]
    assert g3._join_ratings_once(segments=["s"], geometry=geometry, ratings=base)[0]["s"][0] == "RATING_JOIN_KEY_MISSING"


def test_g3r_artifact_schemas_accept_production_shapes() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    result_schema = json.loads(
        (ROOT / "configs/artifact_schemas/rq014_g3r_result_row_v1.schema.json").read_text()
    )
    operation_schema = json.loads(
        (ROOT / "configs/artifact_schemas/rq014_g3r_operation_receipt_v1.schema.json").read_text()
    )
    access_schema = json.loads(
        (ROOT / "configs/artifact_schemas/rq014_g3r_rating_access_receipt_v1.schema.json").read_text()
    )
    for schema in (result_schema, operation_schema, access_schema):
        jsonschema.Draft202012Validator.check_schema(schema)
