from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.rq014.export_score_stripped_bundle as exporter_module

from scripts.rq014.export_score_stripped_bundle import (
    COUNTERPART_SOURCE_HEADER,
    METADATA_KEYS,
    READINESS_SOURCE_HEADER,
    SCENE_KEYS,
    STATE_FIELDS,
    ExportError,
    export_bundle,
    trajectory_geometry_sha256,
)
from scripts.rq014.preflight import (
    ContractError,
    canonical_json_bytes,
    require_contained_regular_file,
    secant_kinematics,
    tstar_anchored_linear_resample,
    validate_score_stripped_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "reports" / "plans" / "RQ014_score_stripped_schema_v1.json"


def _states(
    *,
    x: list[float],
    y: list[float],
    z: list[float] | None = None,
    vx: list[float] | None = None,
    vy: list[float] | None = None,
    ax: list[float] | None = None,
    ay: list[float] | None = None,
) -> dict:
    return {
        "pos_x": x,
        "pos_y": y,
        "pos_z": [] if z is None else z,
        "vel_x": [] if vx is None else vx,
        "vel_y": [] if vy is None else vy,
        "accel_x": [] if ax is None else ax,
        "accel_y": [] if ay is None else ay,
    }


def _scene(segment: str = "seg000") -> dict:
    candidates = [
        _states(x=[0.1, 1.0, 2.0], y=[0.0, 0.0, 0.0]),
        _states(
            x=[0.2, 1.0, 2.0],
            y=[0.1, 0.2, 0.3],
            z=[1.0, 1.1, 1.2],
            vx=[3.0, 3.1, 3.2],
            vy=[0.4, 0.5, 0.6],
            ax=[0.01, 0.02, 0.03],
            ay=[0.04, 0.05, 0.06],
        ),
        _states(x=[1.0, 2.0, 3.0], y=[1.0, 1.0, 1.0]),
    ]
    past_x = [float(index - 15) for index in range(16)]
    future_x = [0.25 * (index + 1) for index in range(20)]
    value = {
        "scene_id": segment,
        "segment_key": segment,
        "context_name": segment,
        "context_name_tstar": segment,
        "source_shard": "shard_0",
        "record_index": 0,
        "tstar_context_step": 100,
        "timestamp_micros": 10_000_000,
        "tstar_ego_pose": [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "intent": 1,
        "intent_name": "GO_STRAIGHT",
        "past_states": _states(
            x=past_x,
            y=[0.0] * 16,
            vx=[4.0] * 16,
            vy=[0.0] * 16,
            ax=[0.0] * 16,
            ay=[0.0] * 16,
        ),
        "future_states": _states(
            x=future_x,
            y=[0.1 * index for index in range(20)],
            z=[0.0] * 20,
        ),
        "preference_trajectories": candidates,
        "forward_cameras": [],
        "post_window": {"native_dt_s": 0.1},
    }
    assert set(value) == SCENE_KEYS
    return value


def _metadata(*, ratings_blind: bool = True) -> dict:
    value = {
        "n_shards": 8,
        "native_dt_s": 0.1,
        "notes": "fixture",
        "phase0_provenance_json": "hidden-parent.json",
        "post_frames": 50,
        "ratings_blind": ratings_blind,
        "segment_root": "/hidden/rated479_segments",
        "shard_index": 0,
        "target_csv": "/hidden/full479_targets/scored.csv",
        "version": 1,
    }
    assert set(value) == METADATA_KEYS
    return value


def _write_sources(
    tmp_path: Path,
    *,
    malicious_candidate: bool = False,
    ratings_blind: bool = True,
    prep_abstentions: list[object] | None = None,
) -> tuple[list[Path], Path, Path]:
    bundles: list[Path] = []
    for index in range(8):
        path = tmp_path / f"bundle_{index}.pkl"
        scene = _scene() if index == 0 else None
        if malicious_candidate and scene is not None:
            scene["preference_trajectories"][0]["preference_score"] = 1.0
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "metadata": _metadata(ratings_blind=ratings_blind),
                    "scenes": [] if scene is None else [scene],
                    "prep_abstentions": (prep_abstentions or []) if index == 0 else [],
                },
                handle,
                protocol=4,
            )
        bundles.append(path)

    readiness = tmp_path / "readiness.tsv"
    with readiness.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=READINESS_SOURCE_HEADER, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for index in range(479):
            row = {field: "0" for field in READINESS_SOURCE_HEADER}
            row.update(
                {
                    "segment_key": f"seg{index:03d}",
                    "scenario_cluster": "fixture",
                    "index_source": "fixture",
                    "native_cadence_s": "0.1",
                    "native_cadence_hz": "10.0",
                }
            )
            writer.writerow(row)

    counterpart = tmp_path / "selected_counterpart_tracks.csv"
    with counterpart.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COUNTERPART_SOURCE_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "segment_key": "seg000",
                "tstar_context_step": "100",
                "counterpart_track_id": "track1",
                "context_step": "101",
                "t_rel_s": "0.1",
                "x": "5.0",
                "y": "1.0",
                "vx": "0.0",
                "vy": "0.0",
                "class_name": "VEHICLE",
                "score": "0.9",
            }
        )
    return bundles, readiness, counterpart


def _source_expectations(
    bundles: list[Path],
    readiness: Path,
    counterpart: Path,
) -> dict[str, dict[str, object]]:
    sources = {
        **{f"phase1_scene_bundle_{index:02d}": path for index, path in enumerate(sorted(bundles))},
        "rated479_structural_readiness": readiness,
        "selected_counterpart_tracks": counterpart,
    }
    return {
        role: {
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for role, path in sources.items()
    }


def _export(tmp_path: Path, *, exporter_git_commit: str = "a" * 40) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    bundles, readiness, counterpart = _write_sources(tmp_path)
    output = tmp_path / "base" / "inputs" / "RQ014" / "wod_rated479_score_stripped" / "v1"
    export_bundle(
        bundle_paths=bundles,
        readiness_path=readiness,
        counterpart_path=counterpart,
        schema_path=SCHEMA,
        output_root=output,
        exporter_git_commit=exporter_git_commit,
        exporter_environment_sha256="b" * 64,
        created_at_utc="2026-07-12T00:00:00Z",
        source_expectations=_source_expectations(bundles, readiness, counterpart),
    )
    return output


def test_preflight_provenance_accepts_receipt_commit_distinct_from_current_head(
    tmp_path: Path,
) -> None:
    export_commit = "a" * 40
    current_head = "b" * 40
    output = _export(tmp_path, exporter_git_commit=export_commit)
    result = validate_score_stripped_bundle(
        bundle_root=output,
        schema_path=SCHEMA,
        file_manifest_path=output / "file_manifest.json",
        receipt_path=output / "sanitization_receipt.json",
        expected_exporter_git_commit=export_commit,
    )
    assert result["sanitization_receipt_sha256"]
    assert export_commit != current_head


def test_preflight_provenance_rejects_receipt_commit_mismatch(tmp_path: Path) -> None:
    output = _export(tmp_path, exporter_git_commit="a" * 40)
    with pytest.raises(ContractError, match="exporter_git_commit"):
        validate_score_stripped_bundle(
            bundle_root=output,
            schema_path=SCHEMA,
            file_manifest_path=output / "file_manifest.json",
            receipt_path=output / "sanitization_receipt.json",
            expected_exporter_git_commit="b" * 40,
        )


def _replace_once(path: Path, old: str, new: str) -> None:
    assert len(old) == len(new)
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _rewrite_csv_cell(
    path: Path,
    *,
    predicate,
    column: str,
    value: str,
) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    matches = [row for row in rows if predicate(row)]
    assert len(matches) == 1
    matches[0][column] = value
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_geometry_hash_matches_the_frozen_legacy_byte_rule() -> None:
    trajectory = _states(x=[0.1, 1.0], y=[2.0, 3.0])
    expected = hashlib.sha256()
    for field in STATE_FIELDS:
        values = trajectory[field]
        expected.update(field.encode("utf-8"))
        expected.update(b"|")
        expected.update(str(len(values)).encode("ascii"))
        expected.update(b":")
        for value in values:
            expected.update(format(float(value), ".17g").encode("ascii"))
            expected.update(b",")
        expected.update(b";")
    assert trajectory_geometry_sha256(trajectory) == expected.hexdigest()


def test_frozen_r10_cross_seam_and_derivative_endpoint_operators() -> None:
    contract = json.loads(SCHEMA.read_text(encoding="utf-8"))["scientific_time_series_contract"]
    assert contract["derivatives"]["application_stage"] == (
        "derive_once_on_each_complete_resampled_branch_before_temporal_window_selection"
    )
    assert contract["derivatives"]["branch_scope"] == (
        "one_complete_R04N_or_R10L_branch_per_candidate"
    )
    assert contract["derivatives"]["window_boundary_rederivation"] == "forbidden"
    grid, values = tstar_anchored_linear_resample(
        [0.0, 0.25, 0.5],
        [0.0, 1.0, 1.5],
    )
    assert grid == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    assert values == pytest.approx([0.0, 0.4, 0.8, 1.1, 1.3, 1.5])

    velocity, acceleration = secant_kinematics([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    assert velocity == pytest.approx([1.0, 2.0, 3.0])
    assert acceleration == pytest.approx([1.0, 1.0, 1.0])


def test_export_is_canonical_rating_free_and_preserves_time_semantics(tmp_path: Path) -> None:
    output = _export(tmp_path)
    result = validate_score_stripped_bundle(
        bundle_root=output,
        schema_path=SCHEMA,
        file_manifest_path=output / "file_manifest.json",
        receipt_path=output / "sanitization_receipt.json",
    )
    assert result["scene_count"] == 479
    assert result["geometry_available_scene_count"] == 1
    assert result["canonical_csv_count"] == 7
    assert not any(path.suffix in {".pkl", ".tfrecord"} for path in output.iterdir())
    all_headers = "\n".join(
        path.read_text(encoding="utf-8").splitlines()[0]
        for path in output.glob("*.csv")
    )
    assert "preference_score" not in all_headers
    assert "candidate_scores" not in all_headers
    assert "detector_confidence" in all_headers

    with (output / "candidate_states.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    c1 = [row for row in rows if row["segment_id"] == "seg000" and row["candidate_id"] == "C1"]
    assert c1[0]["dropped_as_tstar_duplicate"] == "true"
    assert c1[0]["raw_time_s"] == "0.0"
    assert c1[0]["included_in_effective_future"] == "false"
    assert c1[0]["effective_time_s"] == "NA"
    assert c1[1]["effective_sample_index"] == "1"
    assert c1[1]["effective_time_s"] == "0.25"
    c2 = [row for row in rows if row["segment_id"] == "seg000" and row["candidate_id"] == "C2"]
    assert [row["vel_x_mps"] for row in c2] == ["3.0", "3.1", "3.2"]
    assert [row["accel_y_mps2"] for row in c2] == ["0.04", "0.05", "0.06"]

    with (output / "ego_history_states.csv").open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    scene_history = [row for row in history if row["segment_id"] == "seg000"]
    assert [row["time_s"] for row in scene_history] == [
        repr((index - 15) * 0.25) for index in range(16)
    ]
    assert all(row["vel_x_mps"] == "4.0" for row in scene_history)
    assert all(row["pos_z_m"] == "NA" for row in scene_history)

    with (output / "ego_future_states.csv").open(encoding="utf-8", newline="") as handle:
        future = list(csv.DictReader(handle))
    assert len(future) == 20
    assert future[0]["time_s"] == "0.25"
    assert future[-1]["time_s"] == "5.0"
    assert future[0]["pos_z_m"] == "0.0"
    assert future[0]["vel_x_mps"] == "NA"

    with (output / "tstar_ego_pose.csv").open(encoding="utf-8", newline="") as handle:
        pose = list(csv.DictReader(handle))
    assert len(pose) == 16
    assert pose[3] == {
        "segment_id": "seg000",
        "tstar_context_step": "100",
        "matrix_row": "0",
        "matrix_column": "3",
        "value": "10.0",
    }

    with (output / "blind_scene_manifest.csv").open(encoding="utf-8", newline="") as handle:
        manifest = {row["segment_id"]: row for row in csv.DictReader(handle)}
    assert manifest["seg000"]["path_type"] == "UNMAPPED"
    assert manifest["seg000"]["source_shard_id"] == "shard_0"
    assert manifest["seg000"]["route_intent_code"] == "1"
    assert manifest["seg000"]["route_intent_name"] == "GO_STRAIGHT"
    assert manifest["seg000"]["ego_future_state_count"] == "20"
    assert manifest["seg000"]["tstar_ego_pose_element_count"] == "16"
    receipt_text = (output / "sanitization_receipt.json").read_text(encoding="utf-8")
    assert "/hidden/rated479_segments" not in receipt_text
    assert "/hidden/full479_targets" not in receipt_text


def test_reviewed_source_swap_after_open_uses_only_retained_descriptor_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundles, readiness, counterpart = _write_sources(tmp_path)
    expectations = _source_expectations(bundles, readiness, counterpart)
    replacement = tmp_path / "replacement.tsv"
    replacement.write_text("rating_value\t999\n", encoding="utf-8")
    original_open = os.open
    swapped = False

    def swap_after_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        descriptor = original_open(path, flags, *args, **kwargs)
        if Path(path) == readiness and not swapped:
            os.replace(replacement, readiness)
            swapped = True
        return descriptor

    monkeypatch.setattr(exporter_module.os, "open", swap_after_open)
    output = tmp_path / "out"
    receipt = export_bundle(
        bundle_paths=bundles,
        readiness_path=readiness,
        counterpart_path=counterpart,
        schema_path=SCHEMA,
        output_root=output,
        exporter_git_commit="a" * 40,
        exporter_environment_sha256="b" * 64,
        created_at_utc="2026-07-12T00:00:00Z",
        source_expectations=expectations,
    )
    assert swapped is True
    assert receipt["source_artifact_ids_and_sha256"][
        "rated479_structural_readiness"
    ] == expectations["rated479_structural_readiness"]["sha256"]
    assert readiness.read_text(encoding="utf-8") == "rating_value\t999\n"


def test_reviewed_source_reader_rejects_fstat_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"reviewed bytes")
    original_fstat = os.fstat
    calls = 0

    def drifting_fstat(descriptor: int) -> object:
        nonlocal calls
        calls += 1
        result = original_fstat(descriptor)
        if calls == 1:
            return result
        values = {
            field: getattr(result, field)
            for field in ("st_mode", "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        }
        values["st_ctime_ns"] += 1
        return SimpleNamespace(**values)

    monkeypatch.setattr(exporter_module.os, "fstat", drifting_fstat)
    with pytest.raises(ExportError, match="descriptor identity drift"):
        exporter_module._read_reviewed_source_bytes(
            source,
            expected_size=source.stat().st_size,
            expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            label="fixture",
        )


@pytest.mark.parametrize(("drift", "message"), [("short", "ended"), ("grow", "exceeds")])
def test_reviewed_source_reader_rejects_short_or_growing_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
    message: str,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"reviewed bytes")
    expected = source.read_bytes()
    original_read = os.read
    calls = 0

    def drifting_read(descriptor: int, size: int) -> bytes:
        nonlocal calls
        calls += 1
        if drift == "short" and calls == 1:
            return b""
        if drift == "grow" and calls == 2:
            return b"x"
        return original_read(descriptor, size)

    monkeypatch.setattr(exporter_module.os, "read", drifting_read)
    with pytest.raises(ExportError, match=message):
        exporter_module._read_reviewed_source_bytes(
            source,
            expected_size=len(expected),
            expected_sha256=hashlib.sha256(expected).hexdigest(),
            label="fixture",
        )


def test_export_never_reopens_reviewed_sources_via_path_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundles, readiness, counterpart = _write_sources(tmp_path)
    expectations = _source_expectations(bundles, readiness, counterpart)
    reviewed = {Path(os.path.abspath(path)) for path in [*bundles, readiness, counterpart]}
    original_path_open = Path.open

    def forbid_source_path_open(path: Path, *args: object, **kwargs: object):
        if Path(os.path.abspath(path)) in reviewed:
            raise AssertionError(f"reviewed source was reopened: {path}")
        return original_path_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", forbid_source_path_open)
    export_bundle(
        bundle_paths=bundles,
        readiness_path=readiness,
        counterpart_path=counterpart,
        schema_path=SCHEMA,
        output_root=tmp_path / "out",
        exporter_git_commit="a" * 40,
        exporter_environment_sha256="b" * 64,
        created_at_utc="2026-07-12T00:00:00Z",
        source_expectations=expectations,
    )


def test_export_rejects_symlink_and_wrong_reviewed_digest(
    tmp_path: Path,
) -> None:
    bundles, readiness, counterpart = _write_sources(tmp_path)
    expectations = _source_expectations(bundles, readiness, counterpart)
    real_bundle = bundles[0]
    alias = tmp_path / "alias.pkl"
    alias.symlink_to(real_bundle)
    swapped_bundles = [alias, *bundles[1:]]
    with pytest.raises(ExportError, match="without following links"):
        export_bundle(
            bundle_paths=swapped_bundles,
            readiness_path=readiness,
            counterpart_path=counterpart,
            schema_path=SCHEMA,
            output_root=tmp_path / "symlink-out",
            exporter_git_commit="a" * 40,
            exporter_environment_sha256="b" * 64,
            created_at_utc="2026-07-12T00:00:00Z",
            source_expectations=expectations,
        )
    bad_expectations = _source_expectations(bundles, readiness, counterpart)
    bad_expectations["rated479_structural_readiness"]["sha256"] = "0" * 64
    with pytest.raises(ExportError, match="SHA-256 mismatch"):
        export_bundle(
            bundle_paths=bundles,
            readiness_path=readiness,
            counterpart_path=counterpart,
            schema_path=SCHEMA,
            output_root=tmp_path / "digest-out",
            exporter_git_commit="a" * 40,
            exporter_environment_sha256="b" * 64,
            created_at_utc="2026-07-12T00:00:00Z",
            source_expectations=bad_expectations,
        )


def test_export_rejects_an_unexpected_preference_score_key(tmp_path: Path) -> None:
    bundles, readiness, counterpart = _write_sources(tmp_path, malicious_candidate=True)
    with pytest.raises(ExportError, match="Rating-semantic source key"):
        export_bundle(
            bundle_paths=bundles,
            readiness_path=readiness,
            counterpart_path=counterpart,
            schema_path=SCHEMA,
            output_root=tmp_path / "out",
            exporter_git_commit="a" * 40,
            exporter_environment_sha256="b" * 64,
            created_at_utc="2026-07-12T00:00:00Z",
            source_expectations=_source_expectations(bundles, readiness, counterpart),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("state_length", "must be empty or match pos_x length"),
        ("pose_shape", "exact 4x4 list"),
        ("route_pair", "Route intent code/name mismatch"),
        ("source_shard", "Unsafe or empty source shard ID"),
    ],
)
def test_export_rejects_lossy_or_ambiguous_safe_science_primitives(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    bundles, readiness, counterpart = _write_sources(tmp_path)
    with bundles[0].open("rb") as handle:
        payload = pickle.load(handle)
    scene = payload["scenes"][0]
    if mutation == "state_length":
        scene["past_states"]["vel_x"] = [1.0]
    elif mutation == "pose_shape":
        scene["tstar_ego_pose"] = [[1.0, 0.0], [0.0, 1.0]]
    elif mutation == "route_pair":
        scene["intent_name"] = "GO_LEFT"
    else:
        scene["source_shard"] = "../unsafe"
    with bundles[0].open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    with pytest.raises(ExportError, match=message):
        export_bundle(
            bundle_paths=bundles,
            readiness_path=readiness,
            counterpart_path=counterpart,
            schema_path=SCHEMA,
            output_root=tmp_path / "out",
            exporter_git_commit="a" * 40,
            exporter_environment_sha256="b" * 64,
            created_at_utc="2026-07-12T00:00:00Z",
            source_expectations=_source_expectations(bundles, readiness, counterpart),
        )


def test_validator_rejects_tampered_sanitization_receipt(tmp_path: Path) -> None:
    output = _export(tmp_path)
    receipt_path = output / "sanitization_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["forbidden_field_scan"] = 1
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    with pytest.raises(ContractError, match="Sanitization scan did not pass"):
        validate_score_stripped_bundle(
            bundle_root=output,
            schema_path=SCHEMA,
            file_manifest_path=output / "file_manifest.json",
            receipt_path=receipt_path,
        )


@pytest.mark.parametrize(
    "source_kwargs",
    [
        {"ratings_blind": False},
        {"prep_abstentions": [{"preference_score": 5.0}]},
    ],
)
def test_export_rejects_nonblind_or_rating_bearing_bundle_metadata(
    tmp_path: Path,
    source_kwargs: dict[str, object],
) -> None:
    bundles, readiness, counterpart = _write_sources(tmp_path, **source_kwargs)
    with pytest.raises(ExportError, match="rating-blind|Rating-semantic|zero prep abstentions"):
        export_bundle(
            bundle_paths=bundles,
            readiness_path=readiness,
            counterpart_path=counterpart,
            schema_path=SCHEMA,
            output_root=tmp_path / "out",
            exporter_git_commit="a" * 40,
            exporter_environment_sha256="b" * 64,
            created_at_utc="2026-07-12T00:00:00Z",
            source_expectations=_source_expectations(bundles, readiness, counterpart),
        )


def test_restricted_unpickler_rejects_pickle_global(tmp_path: Path) -> None:
    bundles, readiness, counterpart = _write_sources(tmp_path)
    with bundles[0].open("wb") as handle:
        pickle.dump(os.system, handle, protocol=4)
    with pytest.raises(ExportError, match="Pickle global is forbidden"):
        export_bundle(
            bundle_paths=bundles,
            readiness_path=readiness,
            counterpart_path=counterpart,
            schema_path=SCHEMA,
            output_root=tmp_path / "out",
            exporter_git_commit="a" * 40,
            exporter_environment_sha256="b" * 64,
            created_at_utc="2026-07-12T00:00:00Z",
            source_expectations=_source_expectations(bundles, readiness, counterpart),
        )


def test_duplicate_threshold_is_strict_and_effective_future_caps_at_twenty(
    tmp_path: Path,
) -> None:
    bundles, readiness, counterpart = _write_sources(tmp_path)
    with bundles[0].open("rb") as handle:
        payload = pickle.load(handle)
    candidate = payload["scenes"][0]["preference_trajectories"][0]
    candidate["pos_x"] = [0.75 + index for index in range(21)]
    candidate["pos_y"] = [0.0] * 21
    with bundles[0].open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    output = tmp_path / "base" / "inputs" / "RQ014" / "wod_rated479_score_stripped" / "v1"
    export_bundle(
        bundle_paths=bundles,
        readiness_path=readiness,
        counterpart_path=counterpart,
        schema_path=SCHEMA,
        output_root=output,
        exporter_git_commit="a" * 40,
        exporter_environment_sha256="b" * 64,
        created_at_utc="2026-07-12T00:00:00Z",
        source_expectations=_source_expectations(bundles, readiness, counterpart),
    )
    with (output / "candidate_states.csv").open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["segment_id"] == "seg000" and row["candidate_id"] == "C1"
    ]
    assert rows[0]["dropped_as_tstar_duplicate"] == "false"
    assert rows[0]["raw_time_s"] == "0.25"
    assert rows[0]["effective_sample_index"] == "1"
    assert sum(row["included_in_effective_future"] == "true" for row in rows) == 20
    assert rows[19]["effective_sample_index"] == "20"
    assert rows[20]["effective_sample_index"] == "NA"


@pytest.mark.parametrize(
    ("relative_path", "old", "new", "message"),
    [
        (
            "candidate_states.csv",
            "seg000,100,C1,1,",
            "seg000,101,C1,1,",
            "Candidate row tstar differs",
        ),
        (
            "blind_scene_manifest.csv",
            ",10.0,4.0,4.0,3,true,",
            ",10.0,4.0,5.0,3,true,",
            "4 Hz candidate rate",
        ),
    ],
)
def test_validator_rejects_cross_file_contract_drift(
    tmp_path: Path,
    relative_path: str,
    old: str,
    new: str,
    message: str,
) -> None:
    output = _export(tmp_path)
    _replace_once(output / relative_path, old, new)
    with pytest.raises(ContractError, match=message):
        validate_score_stripped_bundle(
            bundle_root=output,
            schema_path=SCHEMA,
            file_manifest_path=output / "file_manifest.json",
            receipt_path=output / "sanitization_receipt.json",
            full_hash=False,
        )


def test_validator_recomputes_candidate_set_hash(tmp_path: Path) -> None:
    output = _export(tmp_path)
    manifest = output / "blind_scene_manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    original = rows[0]["candidate_set_sha256"]
    replacement = ("0" if original[0] != "0" else "1") + original[1:]
    _replace_once(manifest, original, replacement)
    with pytest.raises(ContractError, match="Candidate-set SHA-256 mismatch"):
        validate_score_stripped_bundle(
            bundle_root=output,
            schema_path=SCHEMA,
            file_manifest_path=output / "file_manifest.json",
            receipt_path=output / "sanitization_receipt.json",
            full_hash=False,
        )


@pytest.mark.parametrize(
    ("relative_path", "column", "value", "message", "predicate"),
    [
        (
            "candidate_states.csv",
            "raw_time_s",
            "0.1",
            "raw-time interpretation mismatch",
            lambda row: row["segment_id"] == "seg000"
            and row["candidate_id"] == "C1"
            and row["raw_sample_index"] == "0",
        ),
        (
            "candidate_states.csv",
            "vel_x_mps",
            "NA",
            "mixes empty and populated encoding",
            lambda row: row["segment_id"] == "seg000"
            and row["candidate_id"] == "C2"
            and row["raw_sample_index"] == "0",
        ),
        (
            "ego_future_states.csv",
            "time_s",
            "0.5",
            "Ego-future positive 0.25 s time axis mismatch",
            lambda row: row["segment_id"] == "seg000" and row["sample_index"] == "0",
        ),
        (
            "blind_scene_manifest.csv",
            "route_intent_name",
            "GO_LEFT",
            "Route intent code/name differs",
            lambda row: row["segment_id"] == "seg000",
        ),
        (
            "blind_scene_manifest.csv",
            "source_shard_id",
            "NA",
            "nonempty safe source shard ID",
            lambda row: row["segment_id"] == "seg000",
        ),
    ],
)
def test_validator_rejects_lossless_state_route_and_seam_drift(
    tmp_path: Path,
    relative_path: str,
    column: str,
    value: str,
    message: str,
    predicate,
) -> None:
    output = _export(tmp_path)
    _rewrite_csv_cell(
        output / relative_path,
        predicate=predicate,
        column=column,
        value=value,
    )
    with pytest.raises(ContractError, match=message):
        validate_score_stripped_bundle(
            bundle_root=output,
            schema_path=SCHEMA,
            file_manifest_path=output / "file_manifest.json",
            receipt_path=output / "sanitization_receipt.json",
            full_hash=False,
        )


def test_validator_rejects_scientific_time_contract_drift(tmp_path: Path) -> None:
    output = _export(tmp_path)
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    schema["scientific_time_series_contract"]["r10l"]["step_s"] = 0.2
    drifted_schema = tmp_path / "drifted_schema.json"
    drifted_schema.write_bytes(canonical_json_bytes(schema))
    with pytest.raises(ContractError, match="Scientific time-series contract drifted"):
        validate_score_stripped_bundle(
            bundle_root=output,
            schema_path=drifted_schema,
            file_manifest_path=output / "file_manifest.json",
            receipt_path=output / "sanitization_receipt.json",
            full_hash=False,
        )


def test_validator_rejects_incomplete_tstar_pose(tmp_path: Path) -> None:
    output = _export(tmp_path)
    pose_path = output / "tstar_ego_pose.csv"
    with pose_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    with pose_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows[:-1])
    with pytest.raises(ContractError, match="not an exact 4x4 matrix"):
        validate_score_stripped_bundle(
            bundle_root=output,
            schema_path=SCHEMA,
            file_manifest_path=output / "file_manifest.json",
            receipt_path=output / "sanitization_receipt.json",
            full_hash=False,
        )


def test_validator_rejects_noncanonical_csv_and_rating_semantic_values(tmp_path: Path) -> None:
    output = _export(tmp_path)
    candidate_path = output / "candidate_states.csv"
    candidate_path.write_bytes(candidate_path.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(ContractError, match="UTF-8 LF"):
        validate_score_stripped_bundle(
            bundle_root=output,
            schema_path=SCHEMA,
            file_manifest_path=output / "file_manifest.json",
            receipt_path=output / "sanitization_receipt.json",
            full_hash=False,
        )

    output = _export(tmp_path / "second")
    scene_path = output / "blind_scene_manifest.csv"
    _replace_once(scene_path, ",fixture,", ",rating_,")
    with pytest.raises(ContractError, match="Forbidden rating semantic"):
        validate_score_stripped_bundle(
            bundle_root=output,
            schema_path=SCHEMA,
            file_manifest_path=output / "file_manifest.json",
            receipt_path=output / "sanitization_receipt.json",
            full_hash=False,
        )


def test_validator_rejects_noncanonical_json_receipt(tmp_path: Path) -> None:
    output = _export(tmp_path)
    receipt_path = output / "sanitization_receipt.json"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="not canonical JSON"):
        validate_score_stripped_bundle(
            bundle_root=output,
            schema_path=SCHEMA,
            file_manifest_path=output / "file_manifest.json",
            receipt_path=receipt_path,
            full_hash=False,
        )


def test_resolved_raw_rated479_path_is_denied(tmp_path: Path) -> None:
    raw = tmp_path / "rated479_segments" / "seg000" / "frames.tfrecord"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"rating-bearing fixture")
    with pytest.raises(ContractError, match="Denied RQ014 G2 path"):
        require_contained_regular_file(raw, [tmp_path])


def test_symlink_component_is_rejected_even_when_target_stays_inside_root(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    real = root / "real"
    real.mkdir(parents=True)
    source = real / "input.json"
    source.write_text("{}\n", encoding="utf-8")
    (root / "alias").symlink_to(real, target_is_directory=True)
    with pytest.raises(ContractError, match="crosses a symlink"):
        require_contained_regular_file(root / "alias" / "input.json", [root])


def test_main_rolls_back_published_bundle_when_run_receipt_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundles, readiness, counterpart = _write_sources(tmp_path)
    output = tmp_path / "managed" / "inputs" / "RQ014" / "score_stripped" / "v1"
    receipt_root = tmp_path / "managed" / "work_dirs" / "RQ014" / "run" / "outputs"
    receipt_root.mkdir(parents=True)
    original_write_bytes = Path.write_bytes

    def fail_first_run_receipt(path: Path, payload: bytes) -> int:
        if path.name == "rq014_g2_declassification_export_receipt.json":
            raise OSError("injected run-receipt failure")
        return original_write_bytes(path, payload)

    argv = [str(Path(exporter_module.__file__).resolve())]
    for bundle in bundles:
        argv.extend(["--scene-bundle", str(bundle)])
    for role, expectation in _source_expectations(bundles, readiness, counterpart).items():
        argv.extend(
            [
                "--source-expectation",
                role,
                str(expectation["size_bytes"]),
                str(expectation["sha256"]),
            ]
        )
    argv.extend(
        [
            "--readiness-tsv",
            str(readiness),
            "--counterpart-tracks",
            str(counterpart),
            "--schema",
            str(SCHEMA),
            "--output-root",
            str(output),
            "--run-receipt-root",
            str(receipt_root),
            "--exporter-git-commit",
            "a" * 40,
            "--exporter-environment-sha256",
            "b" * 64,
            "--created-at-utc",
            "2026-07-12T00:00:00Z",
        ]
    )
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(Path, "write_bytes", fail_first_run_receipt)

    with pytest.raises(OSError, match="injected run-receipt failure"):
        exporter_module.main()

    assert not output.exists()
    assert list(receipt_root.iterdir()) == []


def test_main_preserves_lexical_source_path_and_rejects_final_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundles, readiness, counterpart = _write_sources(tmp_path)
    expectations = _source_expectations(bundles, readiness, counterpart)
    readiness_alias = tmp_path / "readiness_alias.tsv"
    readiness_alias.symlink_to(readiness.name)
    output = tmp_path / "managed" / "inputs" / "RQ014" / "score_stripped" / "v1"
    receipt_root = tmp_path / "managed" / "work_dirs" / "RQ014" / "run" / "outputs"
    receipt_root.mkdir(parents=True)
    argv = [str(Path(exporter_module.__file__).resolve())]
    for bundle in bundles:
        argv.extend(["--scene-bundle", str(bundle)])
    for role, expectation in expectations.items():
        argv.extend(
            [
                "--source-expectation",
                role,
                str(expectation["size_bytes"]),
                str(expectation["sha256"]),
            ]
        )
    argv.extend(
        [
            "--readiness-tsv",
            str(readiness_alias),
            "--counterpart-tracks",
            str(counterpart),
            "--schema",
            str(SCHEMA),
            "--output-root",
            str(output),
            "--run-receipt-root",
            str(receipt_root),
            "--exporter-git-commit",
            "a" * 40,
            "--exporter-environment-sha256",
            "b" * 64,
            "--created-at-utc",
            "2026-07-12T00:00:00Z",
        ]
    )
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ExportError, match="without following links"):
        exporter_module.main()
    assert not output.exists()
