#!/usr/bin/env python3
"""Build the frozen RQ014 G2R rating-blind output artifacts.

This W4 module is a standalone, deterministic orchestration kernel.  It binds
the W2 WOD-to-M3 feature construction and anchor domain to the W3 unmasked M3
scorer, pointwise deviations, readouts, A09/A10 status propagation, and the W1
artifact schemas.  It deliberately does not expose a managed operation, read a
rating, join a rating, write a leaderboard or recovery ledger, publish DONE, or
change central authorization.  Managed input loading, atomic publication,
operation receipts, retry, and launcher integration remain W5 work.

Production output schemas fix 479 scenes and 320 cells.  ``fixture_mode`` is a
test-only seam for small deterministic universes: it exercises the identical
row/order/hash machinery but is rejected by the production cardinality gate.
Only that seam accepts injected producers; production hard-binds the W2 feature
and IPV implementations plus a checksum-verified W3 scorer.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Mapping, Sequence

import numpy as np

from scripts.rq014 import build_wod_m3_anchors as W2
from scripts.rq014 import build_wod_scene_anchor_domain as DOMAIN
from scripts.rq014 import score_wod_m3_deviations as W3
from scripts.rq014.wod_ipv_adapter import configure_ipv_estimator_timing, estimate_ego_ipv
from scripts.rq014.wod_ipv_preprocessing import state_sequence_from_window_xy


G2R_OUTPUT_CONTRACT_PATH = Path("reports/plans/RQ014_g2r_output_contract_v1.json")
G2R_OUTPUT_CONTRACT_SHA256 = (
    "b066c090ab90316595574890c2b8a8a5ed1bd87d9041f0ce829501e9e4ed1116"
)
OPERATION = "rq014_r2_blind_feature_build"
EXPECTED_SCENE_COUNT = 479
EXPECTED_CELL_COUNT = 320
EXPECTED_FEATURE_BANK_ROWS = 459_840
EXPECTED_MASK_ROWS = 153_280
EXPECTED_ANCHOR_GROUP_COUNT = 15_328
CANDIDATES = ((1, "C1"), (2, "C2"), (3, "C3"))
DIRECT_ARTIFACT_NAMES = (
    "g2r_anchor_scores.jsonl",
    "g2r_blind_feature_bank.jsonl",
    "g2r_availability_masks.jsonl",
    "common_support_blind_manifest.csv",
    "g2r_predictor_manifest.jsonl",
    "g2r_output_manifest.json",
)
LINEAGE_KEYS = frozenset(
    {
        "input_manifest",
        "sanitization_receipt",
        "materialization_ledger",
        "wod_path_type_mapping_manifest",
        "m3_artifact",
        "m3_manifest",
        "m3_feature_spec_contract",
        "environment_manifest",
        "python_executable",
        "contract_preflight_receipt",
        "contract_preflight_done",
        "resource_pilot_receipt",
        "resource_pilot_done",
        "code_snapshot",
    }
)
PREREQUISITE_ARTIFACTS = {
    "wod_scene_anchor_domain": (
        "wod_scene_anchor_domain.csv",
        "rq014-wod-scene-anchor-domain-v1",
    ),
    "wod_scene_anchor_domain_manifest": (
        "wod_scene_anchor_domain_manifest.json",
        "rq014-wod-scene-anchor-domain-manifest-v1",
    ),
    "nc_pretstar_history_only_receipt": (
        "nc_pretstar_history_only_receipt.json",
        "rq014-nc-pretstar-history-only-receipt-v1",
    ),
}
SOURCE_CONTRACT_IDS = (
    "recovery_lane_v3",
    "envelope_builder_contract_v2",
    "m3_feature_spec_contract",
    "config_space_v1p6",
    "execution_contract_v1p5",
)
ANCHOR_SCORE_KEYS = frozenset(
    {
        "schema_version",
        "segment_id",
        "feature_index",
        "feature_id",
        "sampling_id",
        "temporal_id",
        "horizon_id",
        "tau_tick",
        "candidate_ordinal",
        "candidate_id",
        "path_type",
        "candidate_ipv",
        "m3_q_0p5",
        "m3_lo_90",
        "m3_hi_90",
        "support_gate_pass",
        "ood_abstain",
        "nex",
        "nmd",
        "amd",
        "m3_input_row_sha256_or_NA",
        "upstream_status",
        "reason_code",
    }
)
FEATURE_ROW_KEYS = frozenset(
    {
        "schema_version",
        "cell_index",
        "cell_id",
        "segment_id",
        "candidate_ordinal",
        "candidate_id",
        "feature_id",
        "sampling_id",
        "temporal_id",
        "horizon_id",
        "readout_id",
        "predictor_value",
        "upstream_status",
        "reason_code",
        "anchor_row_count",
        "anchor_slice_sha256",
    }
)
MASK_ROW_KEYS = frozenset(
    {
        "schema_version",
        "cell_index",
        "cell_id",
        "segment_id",
        "candidates",
        "all_three_available",
        "all_three_deviations_finite",
        "deviation_vector_nonconstant",
        "blind_cell_scene_eligible",
        "scene_cell_status",
        "reason_code",
    }
)
PREDICTOR_ROW_KEYS = frozenset(
    {
        "schema_version",
        "cell_index",
        "cell_id",
        "sampling_id",
        "temporal_id",
        "horizon_id",
        "readout_id",
        "registered_scene_count",
        "registered_candidate_slot_count",
        "terminal_candidate_slot_count",
        "available_candidate_slot_count",
        "all_three_available_scene_count",
        "finite_nonconstant_scene_count",
        "cell_terminal_status",
        "cell_fatal_upstream_status_or_NA",
        "reason_code",
        "anchor_score_row_count",
        "anchor_score_slice_sha256",
        "feature_bank_row_count",
        "feature_bank_slice_sha256",
        "availability_mask_row_count",
        "availability_mask_slice_sha256",
    }
)


class G2ROrchestrationError(ValueError):
    """Fail-closed W4 authority, input, status, or output-integrity error."""


class CandidateTerminalError(ValueError):
    """Explicit candidate-granular terminal state raised by an adapter."""

    def __init__(self, status: str, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.reason_code = reason_code


@dataclass(frozen=True)
class CellSpec:
    """One frozen predictor cell in canonical lane-v3 order."""

    cell_index: int
    cell_id: str
    feature_index: int
    feature_id: str
    sampling_id: str
    temporal_id: str
    horizon_id: str
    readout_id: str
    rate_hz: int


@dataclass(frozen=True)
class _SlotSummary:
    anchor_row_count: int
    anchor_slice_sha256: str
    upstream_status: str
    reason_code: str
    readouts: Mapping[str, float] | None


@dataclass(frozen=True)
class SceneBlindInput:
    """One verified rating-free WOD scene plus its route-intent token."""

    domain: DOMAIN.SceneDomainInput
    route_intent: str
    counterpart_is_vehicle: bool = True


@dataclass(frozen=True)
class ArtifactReference:
    """A checksum-bound input or output artifact reference."""

    relative_path: str
    schema_version: str
    size_bytes: int
    sha256: str
    row_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "schema_version": self.schema_version,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "row_count": self.row_count,
        }


@dataclass(frozen=True)
class BlindOutputBuild:
    """Complete standalone W4 byte result and row evidence."""

    artifacts: Mapping[str, bytes]
    row_counts: Mapping[str, int]
    anchor_score_rows: tuple[Mapping[str, Any], ...]
    feature_rows: tuple[Mapping[str, Any], ...]
    mask_rows: tuple[Mapping[str, Any], ...]
    predictor_rows: tuple[Mapping[str, Any], ...]
    common_support_ids: tuple[str, ...]


FeatureBuilder = Callable[..., Mapping[str, Any]]
CandidateIpvEstimator = Callable[..., float]
ScoreRows = Callable[[Sequence[Mapping[str, Any]]], Sequence[W3.PreMaskM3Score]]
_UTC_SECONDS = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")


def _repo_root(repo_root: Path | None = None) -> Path:
    return (repo_root or Path(__file__).resolve().parents[2]).resolve()


def _strict_json(path: Path) -> Mapping[str, Any]:
    import json

    def reject_constant(token: str) -> None:
        raise G2ROrchestrationError(f"nonfinite JSON token in {path}: {token}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise G2ROrchestrationError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise G2ROrchestrationError(f"unreadable JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise G2ROrchestrationError(f"JSON root must be an object: {path}")
    return value


def _load_contract(repo_root: Path | None = None) -> Mapping[str, Any]:
    root = _repo_root(repo_root)
    path = root / G2R_OUTPUT_CONTRACT_PATH
    if W2.sha256_file(path) != G2R_OUTPUT_CONTRACT_SHA256:
        raise G2ROrchestrationError("G2R output contract SHA-256 drift")
    contract = _strict_json(path)
    if (
        contract.get("authority_status")
        != "W5B_G2R_OPERATION_AUTHORIZED_RATING_BLIND_ONLY"
        or contract.get("operation") != OPERATION
        or contract.get("future_operation_binding", {}).get("central_authorization")
        != "ALLOWED"
    ):
        raise G2ROrchestrationError("G2R authorized-operation authority drift")
    if tuple(contract["grid_contract"]["readout_axis"]) != W3.READOUT_ORDER:
        raise G2ROrchestrationError("W2/W3 readout order drift")
    return contract


def canonical_cells(repo_root: Path | None = None) -> tuple[CellSpec, ...]:
    """Enumerate and checksum-lock all 320 cells from the frozen axes."""

    contract = _load_contract(repo_root)
    grid = contract["grid_contract"]
    cells: list[CellSpec] = []
    feature_index = 0
    for sampling in grid["sampling_axis"]:
        for temporal in grid["temporal_families"]:
            feature_id = f"F-{sampling['sampling_id']}-{temporal['temporal_id']}"
            for horizon in grid["horizon_axis"]:
                for readout_id in grid["readout_axis"]:
                    cell_id = (
                        f"RR3-{sampling['sampling_id']}-{temporal['temporal_id']}-"
                        f"{horizon['horizon_id']}-{readout_id}"
                    )
                    cells.append(
                        CellSpec(
                            cell_index=len(cells),
                            cell_id=cell_id,
                            feature_index=feature_index,
                            feature_id=feature_id,
                            sampling_id=sampling["sampling_id"],
                            temporal_id=temporal["temporal_id"],
                            horizon_id=horizon["horizon_id"],
                            readout_id=readout_id,
                            rate_hz=int(sampling["rate_hz"]),
                        )
                    )
            feature_index += 1
    payload = b"".join((cell.cell_id + "\n").encode("utf-8") for cell in cells)
    if (
        len(cells) != EXPECTED_CELL_COUNT
        or len(payload) != grid["canonical_cell_id_payload_size_bytes"]
        or hashlib.sha256(payload).hexdigest() != grid["canonical_cell_ids_sha256"]
        or cells[0].cell_id != grid["first_cell_id"]
        or cells[-1].cell_id != grid["last_cell_id"]
    ):
        raise G2ROrchestrationError("canonical 320-cell enumeration drift")
    return tuple(cells)


def load_verified_scorer_rows(scorer_path: Path) -> ScoreRows:
    """Verify/load the frozen scorer once and return the W3 batch callback."""

    from sociality_estimation.verifier.scorer import load_scorer

    path = scorer_path.resolve()
    if scorer_path.is_symlink() or not path.is_file():
        raise G2ROrchestrationError("M3 scorer must be a regular non-symlink file")
    if (
        path.stat().st_size != W3.M3_SCORER_SIZE_BYTES
        or W2.sha256_file(path) != W3.M3_SCORER_SHA256
    ):
        raise G2ROrchestrationError("M3 scorer size or SHA-256 mismatch")
    bundle = load_scorer(path, verify_hash=True)

    def score_rows(rows: Sequence[Mapping[str, Any]]) -> Sequence[W3.PreMaskM3Score]:
        return W3.score_pre_mask_from_bundle(rows, bundle)

    return score_rows


def _finite_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise G2ROrchestrationError("typed finite value is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise G2ROrchestrationError("typed finite value is nonfinite")
    return {"kind": "FINITE_FLOAT", "value": 0.0 if result == 0.0 else result}


def _na_value(reason_code: str) -> dict[str, str]:
    return {"kind": "NA", "reason_code": reason_code}


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(W2.canonical_json_bytes(dict(row)) for row in rows)


def _common_support_bytes(segment_ids: Sequence[str]) -> tuple[bytes, str]:
    handle = io.StringIO(newline="")
    writer = csv.writer(handle, dialect="excel", lineterminator="\n")
    writer.writerow(["segment_id"])
    writer.writerows([[segment_id] for segment_id in segment_ids])
    stored = handle.getvalue().encode("utf-8")
    if b"\r" in stored:
        raise G2ROrchestrationError("common-support CSV contains CR bytes")
    preimage = b"".join((segment_id + "\n").encode("utf-8") for segment_id in segment_ids)
    return stored, hashlib.sha256(preimage).hexdigest()


def _status_tables(contract: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, int]]:
    rows = contract["status_contract"]["candidate_upstream_statuses"]
    status = {row["reason_code"]: row["status"] for row in rows}
    priority = {row["reason_code"]: int(row["reason_priority"]) for row in rows}
    if len(status) != len(rows) or set(status) != set(priority):
        raise G2ROrchestrationError("candidate status namespace is malformed")
    return status, priority


def _scene_focal_reference(
    scene: SceneBlindInput, available_sampling_ids: Collection[str]
) -> np.ndarray:
    """Derive the scene reference from the first frozen-order AVAILABLE branch.

    Recovery lane v3 orders feature families by its declared sampling axis and
    requires exact AVAILABLE anchor groups to execute while terminal groups emit
    no value.  A merely present terminal sampling branch is therefore never a
    legal source for the reference.
    """

    available = set(available_sampling_ids)
    registered = {sampling_id for sampling_id, _ in DOMAIN.SAMPLING_AXIS}
    if not available or not available <= registered:
        raise G2ROrchestrationError("scene has no registered AVAILABLE sampling branch")
    sampling_id = next(
        sampling_id
        for sampling_id, _ in DOMAIN.SAMPLING_AXIS
        if sampling_id in available
    )
    sampling = scene.domain.sampling.get(sampling_id)
    if sampling is None or sampling.terminal_status is not None:
        raise G2ROrchestrationError("AVAILABLE sampling branch is absent or terminal")
    candidate_ticks = [set(sampling.candidates[candidate]) for _, candidate in CANDIDATES]
    shared_ticks = sorted(set.intersection(*candidate_ticks))
    history_ticks = [tick for tick in shared_ticks if tick <= 0]
    if len(history_ticks) < 2:
        raise G2ROrchestrationError("scene focal reference needs two pre-tstar ticks")
    rows: list[tuple[float, float]] = []
    for tick in history_ticks:
        values = [tuple(map(float, sampling.candidates[candidate][tick])) for _, candidate in CANDIDATES]
        if values[1:] != values[:-1]:
            raise G2ROrchestrationError("candidate histories are not scene-level byte-identical")
        rows.append(values[0])
    try:
        return W2.build_scene_focal_reference(rows, route_intent=scene.route_intent)
    except W2.WodM3KernelError as exc:
        raise G2ROrchestrationError("scene focal reference construction failed") from exc


def _joint_tick_bounds(sampling: DOMAIN.SamplingTimelines) -> tuple[int, int]:
    tick_sets = [set(sampling.counterpart)] + [
        set(sampling.candidates[candidate_id]) for _, candidate_id in CANDIDATES
    ]
    joint = set.intersection(*tick_sets)
    if not joint:
        raise G2ROrchestrationError("sampling branch has no joint tick support")
    return min(joint), max(joint)


def _position_track_sha256(positions: Mapping[int, Sequence[float]]) -> str:
    payload = [
        [int(tick), float(positions[tick][0]), float(positions[tick][1])]
        for tick in sorted(positions)
    ]
    return hashlib.sha256(W2.canonical_json_bytes(payload)).hexdigest()


def _default_candidate_ipv_estimator(
    *,
    candidate_positions: Mapping[int, Sequence[float]],
    counterpart_positions: Mapping[int, Sequence[float]],
    temporal_id: str,
    tau_tick: int,
    rate_hz: int,
    h_common_tick: int,
    scene_focal_reference: Sequence[Sequence[float]],
) -> float:
    start_tick, end_tick = W2.temporal_window_bounds(
        temporal_id, tau_tick, rate_hz, h_common_tick
    )
    ticks = tuple(range(start_tick, end_tick + 1))
    if any(tick not in candidate_positions or tick not in counterpart_positions for tick in ticks):
        raise CandidateTerminalError(
            "INELIGIBLE_TIMELINE_SUPPORT",
            "F_TIMELINE_SUPPORT",
            "registered candidate IPV window is incomplete",
        )
    try:
        candidate_xy = np.asarray([candidate_positions[tick] for tick in ticks], dtype=float)
        counterpart_xy = np.asarray([counterpart_positions[tick] for tick in ticks], dtype=float)
        candidate_state = state_sequence_from_window_xy(candidate_xy, 1.0 / rate_hz)
        counterpart_state = state_sequence_from_window_xy(counterpart_xy, 1.0 / rate_hz)
        configure_ipv_estimator_timing(1.0 / rate_hz)
        value, _ = estimate_ego_ipv(
            candidate_state,
            counterpart_state,
            ego_reference=np.asarray(scene_focal_reference, dtype=float),
        )
    except (ValueError, FloatingPointError) as exc:
        raise CandidateTerminalError(
            "INELIGIBLE_IPV_NUMERICAL",
            "F_IPV_NUMERICAL",
            "candidate exact IPV estimation failed",
        ) from exc
    if not math.isfinite(value):
        raise CandidateTerminalError(
            "INELIGIBLE_IPV_NUMERICAL",
            "F_IPV_NUMERICAL",
            "candidate exact IPV result is nonfinite",
        )
    return 0.0 if value == 0.0 else float(value)


def _candidate_terminal_from_exception(exc: Exception) -> CandidateTerminalError | None:
    if isinstance(exc, CandidateTerminalError):
        return exc
    if isinstance(exc, (W2.M3ScoringNumericalFailure, W3.M3ScoringNumericalFailure)):
        return CandidateTerminalError(
            "M3_SCORING_NUMERICAL_FAILURE",
            "F_M3_SCORING_NUMERICAL_FAILURE",
            str(exc),
        )
    if isinstance(exc, W2.WodM3KernelError):
        message = str(exc).lower()
        if "ipv" in message:
            return CandidateTerminalError(
                "INELIGIBLE_IPV_NUMERICAL", "F_IPV_NUMERICAL", str(exc)
            )
        if "reference" in message:
            return CandidateTerminalError(
                "INELIGIBLE_REFERENCE", "F_INELIGIBLE_REFERENCE", str(exc)
            )
        if "vehicle" in message:
            return CandidateTerminalError(
                "INELIGIBLE_COUNTERPART_CLASS", "F_COUNTERPART_NOT_VEHICLE", str(exc)
            )
        if "heading" in message:
            return CandidateTerminalError(
                "INELIGIBLE_UNDEFINED_HEADING", "F_UNDEFINED_HEADING", str(exc)
            )
        if any(token in message for token in ("tick", "timeline", "window", "support", "precedes")):
            return CandidateTerminalError(
                "INELIGIBLE_TIMELINE_SUPPORT", "F_TIMELINE_SUPPORT", str(exc)
            )
        if "nonfinite" in message or "finite" in message:
            return CandidateTerminalError(
                "INELIGIBLE_STATE_NONFINITE", "F_STATE_NONFINITE", str(exc)
            )
    return None


def _failure_anchor_row(base: Mapping[str, Any], failure: CandidateTerminalError) -> dict[str, Any]:
    na = _na_value(failure.reason_code)
    return {
        **base,
        "candidate_ipv": na,
        "m3_q_0p5": na,
        "m3_lo_90": na,
        "m3_hi_90": na,
        "support_gate_pass": "NA",
        "ood_abstain": "NA",
        "nex": na,
        "nmd": na,
        "amd": na,
        "m3_input_row_sha256_or_NA": "NA",
        "upstream_status": failure.status,
        "reason_code": failure.reason_code,
    }


def _score_with_numerical_isolation(
    pending: Sequence[tuple[int, Mapping[str, Any], Mapping[str, Any], float]],
    score_rows: ScoreRows,
) -> list[W3.PreMaskM3Score | CandidateTerminalError]:
    """Batch normally, then bisect only an A09 numerical batch failure."""

    output: list[W3.PreMaskM3Score | CandidateTerminalError | None] = [None] * len(pending)

    def score_indices(indices: tuple[int, ...]) -> None:
        try:
            observed = tuple(score_rows([pending[index][2] for index in indices]))
            if len(observed) != len(indices) or not all(
                isinstance(item, W3.PreMaskM3Score) for item in observed
            ):
                raise G2ROrchestrationError("M3 scorer returned a malformed row set")
            for index, item in zip(indices, observed):
                output[index] = item
        except (W2.M3ScoringNumericalFailure, W3.M3ScoringNumericalFailure) as exc:
            if len(indices) == 1:
                output[indices[0]] = CandidateTerminalError(
                    "M3_SCORING_NUMERICAL_FAILURE",
                    "F_M3_SCORING_NUMERICAL_FAILURE",
                    str(exc),
                )
                return
            midpoint = len(indices) // 2
            score_indices(indices[:midpoint])
            score_indices(indices[midpoint:])

    score_indices(tuple(range(len(pending))))
    if any(item is None for item in output):
        raise G2ROrchestrationError("M3 scorer isolation left an unresolved row")
    return [item for item in output if item is not None]


def build_anchor_score_rows(
    scenes: Sequence[SceneBlindInput],
    anchor_domain_rows: Sequence[Mapping[str, str]],
    *,
    score_rows: ScoreRows,
    feature_builder: FeatureBuilder = W2.build_feature_family_m3_input_row,
    candidate_ipv_estimator: CandidateIpvEstimator = _default_candidate_ipv_estimator,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Run W2->W3 for every concrete AVAILABLE domain anchor and candidate."""

    contract = _load_contract(repo_root)
    status_by_reason, _ = _status_tables(contract)
    ordered_scenes = sorted(scenes, key=lambda item: item.domain.segment_id.encode("utf-8"))
    domain_scenes = [scene.domain for scene in ordered_scenes]
    DOMAIN.validate_anchor_domain_rows(
        anchor_domain_rows,
        expected_scene_count=len(scenes),
        source_scenes=domain_scenes,
    )
    by_segment = {scene.domain.segment_id: scene for scene in ordered_scenes}
    if len(by_segment) != len(scenes):
        raise G2ROrchestrationError("duplicate scene input")
    available_sampling_by_segment: dict[str, set[str]] = {}
    for row in anchor_domain_rows:
        if row["membership_status"] != "AVAILABLE":
            continue
        segment_id = str(row["segment_id"])
        sampling_id = str(row["feature_id"]).split("-", 2)[1]
        available_sampling_by_segment.setdefault(segment_id, set()).add(sampling_id)
    focal_references = {
        segment_id: _scene_focal_reference(
            by_segment[segment_id], available_sampling_by_segment[segment_id]
        )
        for segment_id in sorted(
            available_sampling_by_segment, key=lambda value: value.encode("utf-8")
        )
    }
    feature_order = {feature_id: index for index, feature_id in enumerate(DOMAIN.FEATURE_FAMILIES)}
    pending: list[tuple[int, dict[str, Any], Mapping[str, Any], float]] = []
    emitted: dict[tuple[int, int], dict[str, Any]] = {}
    canonical_feature_builder = feature_builder is W2.build_feature_family_m3_input_row
    canonical_ipv_estimator = candidate_ipv_estimator is _default_candidate_ipv_estimator
    track_digests: dict[tuple[str, str, str], str] = {}
    m3_rows: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
    candidate_ipvs: dict[tuple[str, str, str, int, int], float] = {}

    available_domain_index = -1
    for domain_row in anchor_domain_rows:
        if domain_row["membership_status"] != "AVAILABLE":
            continue
        available_domain_index += 1
        segment_id = domain_row["segment_id"]
        scene = by_segment.get(segment_id)
        if scene is None:
            raise G2ROrchestrationError("anchor domain names an unknown scene")
        feature_id = domain_row["feature_id"]
        try:
            sampling_id, temporal_id = feature_id.split("-", 2)[1:]
            feature_index = feature_order[feature_id]
            rate_hz = dict(DOMAIN.SAMPLING_AXIS)[sampling_id]
            tau_tick = int(domain_row["tau_tick_or_NA"])
            sampling = scene.domain.sampling[sampling_id]
        except (KeyError, TypeError, ValueError) as exc:
            raise G2ROrchestrationError("malformed AVAILABLE anchor-domain row") from exc
        case_start_tick, h_common_tick = _joint_tick_bounds(sampling)
        if temporal_id == "TF" and int(domain_row["h_common_tick_or_NA"]) != h_common_tick:
            raise G2ROrchestrationError("TF h_common binding drift")
        for candidate_ordinal, candidate_id in CANDIDATES:
            base = {
                "schema_version": "rq014-g2r-anchor-score-row-v1",
                "segment_id": segment_id,
                "feature_index": feature_index,
                "feature_id": feature_id,
                "sampling_id": sampling_id,
                "temporal_id": temporal_id,
                "horizon_id": domain_row["horizon_id"],
                "tau_tick": tau_tick,
                "candidate_ordinal": candidate_ordinal,
                "candidate_id": candidate_id,
                "path_type": domain_row["path_type_or_NA"],
            }
            try:
                candidate_positions = sampling.candidates[candidate_id]
                track_key = (segment_id, sampling_id, candidate_id)
                if track_key not in track_digests:
                    track_digests[track_key] = _position_track_sha256(candidate_positions)
                track_sha256 = track_digests[track_key]
                if canonical_ipv_estimator:
                    start_tick, end_tick = W2.temporal_window_bounds(
                        temporal_id, tau_tick, rate_hz, h_common_tick
                    )
                    ipv_key = (
                        segment_id,
                        sampling_id,
                        track_sha256,
                        start_tick,
                        end_tick,
                    )
                    if ipv_key not in candidate_ipvs:
                        candidate_ipvs[ipv_key] = candidate_ipv_estimator(
                            candidate_positions=candidate_positions,
                            counterpart_positions=sampling.counterpart,
                            temporal_id=temporal_id,
                            tau_tick=tau_tick,
                            rate_hz=rate_hz,
                            h_common_tick=h_common_tick,
                            scene_focal_reference=focal_references[segment_id],
                        )
                    candidate_ipv = candidate_ipvs[ipv_key]
                else:
                    candidate_ipv = candidate_ipv_estimator(
                        candidate_positions=candidate_positions,
                        counterpart_positions=sampling.counterpart,
                        temporal_id=temporal_id,
                        tau_tick=tau_tick,
                        rate_hz=rate_hz,
                        h_common_tick=h_common_tick,
                        scene_focal_reference=focal_references[segment_id],
                    )
                if canonical_feature_builder:
                    _, _, context_tick = W2.select_feature_family_context(
                        candidate_positions,
                        sampling.counterpart,
                        temporal_id=temporal_id,
                        tau_tick=tau_tick,
                        rate_hz=rate_hz,
                        h_common_tick=h_common_tick,
                        case_start_tick=case_start_tick,
                        alignment=W2.ALIGNMENT_PRIMARY,
                    )
                    m3_key = (segment_id, sampling_id, track_sha256, context_tick)
                    if m3_key not in m3_rows:
                        m3_rows[m3_key] = feature_builder(
                            candidate_positions,
                            sampling.counterpart,
                            temporal_id=temporal_id,
                            tau_tick=tau_tick,
                            rate_hz=rate_hz,
                            h_common_tick=h_common_tick,
                            case_start_tick=case_start_tick,
                            scene_focal_reference=focal_references[segment_id],
                            counterpart_is_vehicle=scene.counterpart_is_vehicle,
                            m3_context_alignment=W2.ALIGNMENT_PRIMARY,
                        )
                    m3_row = m3_rows[m3_key]
                else:
                    m3_row = feature_builder(
                        candidate_positions,
                        sampling.counterpart,
                        temporal_id=temporal_id,
                        tau_tick=tau_tick,
                        rate_hz=rate_hz,
                        h_common_tick=h_common_tick,
                        case_start_tick=case_start_tick,
                        scene_focal_reference=focal_references[segment_id],
                        counterpart_is_vehicle=scene.counterpart_is_vehicle,
                        m3_context_alignment=W2.ALIGNMENT_PRIMARY,
                    )
                W2.m3_input_row_bytes_and_sha256(m3_row)
                if not math.isfinite(float(candidate_ipv)):
                    raise CandidateTerminalError(
                        "INELIGIBLE_IPV_NUMERICAL",
                        "F_IPV_NUMERICAL",
                        "candidate IPV is nonfinite",
                    )
            except Exception as exc:  # classification is explicit and fail closed below
                failure = _candidate_terminal_from_exception(exc)
                if failure is None:
                    raise G2ROrchestrationError("unexpected W2 candidate failure") from exc
                if status_by_reason.get(failure.reason_code) != failure.status:
                    raise G2ROrchestrationError("candidate failure namespace drift")
                emitted[(available_domain_index, candidate_ordinal)] = _failure_anchor_row(
                    base, failure
                )
                continue
            pending.append(
                (available_domain_index, base, m3_row, float(candidate_ipv))
            )

    if pending:
        scores = _score_with_numerical_isolation(pending, score_rows)
        for (domain_index, base, m3_row, candidate_ipv), score in zip(
            pending, scores
        ):
            domain_key = (domain_index, int(base["candidate_ordinal"]))
            if isinstance(score, CandidateTerminalError):
                emitted[domain_key] = _failure_anchor_row(base, score)
                continue
            try:
                nex, nmd, amd = W3.pointwise_deviations(
                    candidate_ipv, score.lo_90, score.q_0p5, score.hi_90
                )
            except W3.M3ScoringNumericalFailure as exc:
                failure = _candidate_terminal_from_exception(exc)
                assert failure is not None
                emitted[domain_key] = _failure_anchor_row(base, failure)
                continue
            _, row_sha256 = W2.m3_input_row_bytes_and_sha256(m3_row)
            if row_sha256 != score.m3_input_row_sha256:
                raise G2ROrchestrationError("M3 score/input row hash mismatch")
            emitted[domain_key] = {
                **base,
                "candidate_ipv": _finite_value(candidate_ipv),
                "m3_q_0p5": _finite_value(score.q_0p5),
                "m3_lo_90": _finite_value(score.lo_90),
                "m3_hi_90": _finite_value(score.hi_90),
                "support_gate_pass": bool(score.support_gate_pass),
                "ood_abstain": bool(score.ood_abstain),
                "nex": _finite_value(nex),
                "nmd": _finite_value(nmd),
                "amd": _finite_value(amd),
                "m3_input_row_sha256_or_NA": row_sha256,
                "upstream_status": "AVAILABLE",
                "reason_code": "F_AVAILABLE_CONTINUE",
            }

    output: list[dict[str, Any]] = []
    domain_index = -1
    for domain_row in anchor_domain_rows:
        if domain_row["membership_status"] != "AVAILABLE":
            continue
        domain_index += 1
        for candidate_ordinal, _ in CANDIDATES:
            try:
                row = emitted[(domain_index, candidate_ordinal)]
            except KeyError as exc:
                raise G2ROrchestrationError("anchor-score row was silently deleted") from exc
            _validate_anchor_score_row(row, status_by_reason)
            output.append(row)
    return output


def _validate_anchor_score_row(
    row: Mapping[str, Any], status_by_reason: Mapping[str, str]
) -> None:
    if set(row) != ANCHOR_SCORE_KEYS:
        raise G2ROrchestrationError("anchor-score exact-key mismatch")
    reason = str(row["reason_code"])
    if status_by_reason.get(reason) != row["upstream_status"]:
        raise G2ROrchestrationError("anchor-score status/reason mismatch")
    available = row["upstream_status"] == "AVAILABLE"
    typed_fields = (
        "candidate_ipv",
        "m3_q_0p5",
        "m3_lo_90",
        "m3_hi_90",
        "nex",
        "nmd",
        "amd",
    )
    for field in typed_fields:
        value = row[field]
        if available:
            if set(value) != {"kind", "value"} or value["kind"] != "FINITE_FLOAT":
                raise G2ROrchestrationError("AVAILABLE anchor lacks finite typed value")
            _finite_value(value["value"])
        elif value != _na_value(reason):
            raise G2ROrchestrationError("terminal anchor typed-NA drift")
    if available:
        if not isinstance(row["support_gate_pass"], bool) or not isinstance(
            row["ood_abstain"], bool
        ):
            raise G2ROrchestrationError("AVAILABLE support diagnostics are malformed")
    elif (
        row["support_gate_pass"] != "NA"
        or row["ood_abstain"] != "NA"
        or row["m3_input_row_sha256_or_NA"] != "NA"
    ):
        raise G2ROrchestrationError("terminal anchor diagnostics are not typed NA")


def _anchor_rows_by_slot(
    anchor_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str, int], list[Mapping[str, Any]]]:
    output: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = {}
    for row in anchor_rows:
        key = (
            str(row["segment_id"]),
            str(row["feature_id"]),
            str(row["horizon_id"]),
            int(row["candidate_ordinal"]),
        )
        output.setdefault(key, []).append(row)
    return output


def _select_anchor_failure(
    rows: Sequence[Mapping[str, Any]], priority: Mapping[str, int]
) -> tuple[str, str]:
    failures = [row for row in rows if row["upstream_status"] != "AVAILABLE"]
    if not failures:
        return "AVAILABLE", "F_AVAILABLE_CONTINUE"
    selected = min(
        failures,
        key=lambda row: (priority[str(row["reason_code"])], str(row["reason_code"]).encode()),
    )
    return str(selected["upstream_status"]), str(selected["reason_code"])


def _direct_artifact_ref(
    name: str, schema_version: str, data: bytes, row_count: int
) -> ArtifactReference:
    return ArtifactReference(
        relative_path=name,
        schema_version=schema_version,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        row_count=row_count,
    )


def _validate_file_ref(value: Mapping[str, Any], *, label: str) -> None:
    if set(value) != {"path", "size_bytes", "sha256"}:
        raise G2ROrchestrationError(f"{label} file reference exact-key mismatch")
    if (
        not isinstance(value["path"], str)
        or not value["path"]
        or isinstance(value["size_bytes"], bool)
        or not isinstance(value["size_bytes"], int)
        or value["size_bytes"] < 0
        or not isinstance(value["sha256"], str)
        or len(value["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in value["sha256"])
    ):
        raise G2ROrchestrationError(f"{label} file reference is malformed")


def _source_contract_refs(
    contract: Mapping[str, Any], repo_root: Path | None
) -> dict[str, dict[str, Any]]:
    root = _repo_root(repo_root)
    output: dict[str, dict[str, Any]] = {}
    for source_id in SOURCE_CONTRACT_IDS:
        binding = contract["source_bindings"][source_id]
        path = root / binding["path"]
        if (
            path.stat().st_size != binding["size_bytes"]
            or W2.sha256_file(path) != binding["sha256"]
        ):
            raise G2ROrchestrationError(f"source contract drift: {source_id}")
        output[source_id] = {
            "relative_path": binding["path"],
            "size_bytes": binding["size_bytes"],
            "sha256": binding["sha256"],
        }
    return output


def _validate_prerequisite_refs(
    refs: Mapping[str, ArtifactReference], *, fixture_mode: bool
) -> None:
    if set(refs) != set(PREREQUISITE_ARTIFACTS):
        raise G2ROrchestrationError("prerequisite artifact universe drift")
    for artifact_id, (path, schema) in PREREQUISITE_ARTIFACTS.items():
        ref = refs[artifact_id]
        if (
            ref.relative_path != path
            or ref.schema_version != schema
            or ref.size_bytes < 1
            or len(ref.sha256) != 64
            or any(character not in "0123456789abcdef" for character in ref.sha256)
        ):
            raise G2ROrchestrationError(f"malformed prerequisite artifact: {artifact_id}")
    if not fixture_mode:
        if refs["wod_scene_anchor_domain"].row_count < EXPECTED_ANCHOR_GROUP_COUNT:
            raise G2ROrchestrationError("production anchor-domain row count is too small")
        if refs["wod_scene_anchor_domain_manifest"].row_count != 1:
            raise G2ROrchestrationError("anchor-domain manifest row count drift")
        if refs["nc_pretstar_history_only_receipt"].row_count != 1:
            raise G2ROrchestrationError("NC receipt row count drift")


def build_blind_output_artifacts(
    scenes: Sequence[SceneBlindInput],
    anchor_domain_rows: Sequence[Mapping[str, str]],
    *,
    run_id: str,
    git_commit: str,
    created_at_utc: str,
    lineage: Mapping[str, Mapping[str, Any]],
    prerequisite_artifacts: Mapping[str, ArtifactReference],
    scorer_path: Path | None = None,
    score_rows: ScoreRows | None = None,
    feature_builder: FeatureBuilder | None = None,
    candidate_ipv_estimator: CandidateIpvEstimator | None = None,
    fixture_mode: bool = False,
    repo_root: Path | None = None,
) -> BlindOutputBuild:
    """Build all six standalone W4 artifacts in canonical stored-byte order."""

    if not fixture_mode:
        if any(
            producer is not None
            for producer in (score_rows, feature_builder, candidate_ipv_estimator)
        ):
            raise G2ROrchestrationError(
                "production G2R forbids injected feature/IPV/scorer producers"
            )
        if scorer_path is None:
            raise G2ROrchestrationError("production G2R requires the pinned M3 scorer path")
        score_rows = load_verified_scorer_rows(scorer_path)
        feature_builder = W2.build_feature_family_m3_input_row
        candidate_ipv_estimator = _default_candidate_ipv_estimator
    else:
        if score_rows is not None and scorer_path is not None:
            raise G2ROrchestrationError("fixture scorer path/callback is ambiguous")
        if score_rows is None:
            if scorer_path is None:
                raise G2ROrchestrationError("fixture mode requires a scorer path or callback")
            score_rows = load_verified_scorer_rows(scorer_path)
        feature_builder = feature_builder or W2.build_feature_family_m3_input_row
        candidate_ipv_estimator = (
            candidate_ipv_estimator or _default_candidate_ipv_estimator
        )

    contract = _load_contract(repo_root)
    status_by_reason, priority = _status_tables(contract)
    cells = canonical_cells(repo_root)
    scene_ids = [scene.domain.segment_id for scene in scenes]
    if len(scene_ids) != len(set(scene_ids)):
        raise G2ROrchestrationError("duplicate segment_id")
    ordered_scenes = sorted(scenes, key=lambda item: item.domain.segment_id.encode("utf-8"))
    scene_ids = [scene.domain.segment_id for scene in ordered_scenes]
    if fixture_mode and len(ordered_scenes) >= EXPECTED_SCENE_COUNT:
        raise G2ROrchestrationError("fixture mode cannot emit a production-size universe")
    if not fixture_mode and len(ordered_scenes) != EXPECTED_SCENE_COUNT:
        raise G2ROrchestrationError("production G2R requires exactly 479 scenes")
    if set(lineage) != LINEAGE_KEYS:
        raise G2ROrchestrationError("output-manifest lineage exact-key mismatch")
    for lineage_id, reference in lineage.items():
        _validate_file_ref(reference, label=lineage_id)
    _validate_prerequisite_refs(prerequisite_artifacts, fixture_mode=fixture_mode)
    if (
        not run_id
        or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-" for character in run_id)
        or len(git_commit) != 40
        or any(character not in "0123456789abcdef" for character in git_commit)
        or _UTC_SECONDS.fullmatch(created_at_utc) is None
    ):
        raise G2ROrchestrationError("run identity or timestamp is malformed")

    anchor_rows = build_anchor_score_rows(
        ordered_scenes,
        anchor_domain_rows,
        score_rows=score_rows,
        feature_builder=feature_builder,
        candidate_ipv_estimator=candidate_ipv_estimator,
        repo_root=repo_root,
    )
    anchor_bytes = _jsonl_bytes(anchor_rows)
    anchor_by_slot = _anchor_rows_by_slot(anchor_rows)
    anchor_bytes_by_cell_key: dict[tuple[str, str], bytes] = {}
    anchor_count_by_cell_key: dict[tuple[str, str], int] = {}
    for feature_id in DOMAIN.FEATURE_FAMILIES:
        for horizon_id in DOMAIN.HORIZON_AXIS:
            selected_for_cell = [
                row
                for row in anchor_rows
                if row["feature_id"] == feature_id and row["horizon_id"] == horizon_id
            ]
            key = (feature_id, horizon_id)
            anchor_bytes_by_cell_key[key] = _jsonl_bytes(selected_for_cell)
            anchor_count_by_cell_key[key] = len(selected_for_cell)
    domain_groups: dict[tuple[str, str, str], list[Mapping[str, str]]] = {}
    for row in anchor_domain_rows:
        key = (row["segment_id"], row["feature_id"], row["horizon_id"])
        domain_groups.setdefault(key, []).append(row)
    slot_summaries: dict[tuple[str, str, str, int], _SlotSummary] = {}
    for scene in ordered_scenes:
        segment_id = scene.domain.segment_id
        for feature_id in DOMAIN.FEATURE_FAMILIES:
            sampling_id = feature_id.split("-")[1]
            rate_hz = dict(DOMAIN.SAMPLING_AXIS)[sampling_id]
            for horizon_id in DOMAIN.HORIZON_AXIS:
                group = domain_groups.get((segment_id, feature_id, horizon_id))
                if not group:
                    raise G2ROrchestrationError("anchor-domain group missing for cell/scene")
                for candidate_ordinal, _candidate_id in CANDIDATES:
                    selected = anchor_by_slot.get(
                        (segment_id, feature_id, horizon_id, candidate_ordinal), []
                    )
                    selected_bytes = _jsonl_bytes(selected)
                    slice_sha = hashlib.sha256(selected_bytes).hexdigest()
                    readouts: Mapping[str, float] | None = None
                    if group[0]["membership_status"] != "AVAILABLE":
                        if selected:
                            raise G2ROrchestrationError(
                                "terminal domain group emitted anchor rows"
                            )
                        status = group[0]["membership_status"]
                        reason = group[0]["reason_code"]
                    else:
                        expected_ticks = [int(row["tau_tick_or_NA"]) for row in group]
                        observed_ticks = [int(row["tau_tick"]) for row in selected]
                        if observed_ticks != expected_ticks:
                            raise G2ROrchestrationError("candidate anchor tick-set drift")
                        status, reason = _select_anchor_failure(selected, priority)
                        if status == "AVAILABLE":
                            try:
                                readouts = W3.physical_time_readouts(
                                    [row["tau_tick"] / rate_hz for row in selected],
                                    [row["nex"]["value"] for row in selected],
                                    [row["nmd"]["value"] for row in selected],
                                    [row["amd"]["value"] for row in selected],
                                )
                            except W3.M3ScoringNumericalFailure:
                                status = "M3_SCORING_NUMERICAL_FAILURE"
                                reason = "F_M3_SCORING_NUMERICAL_FAILURE"
                    slot_summaries[
                        (segment_id, feature_id, horizon_id, candidate_ordinal)
                    ] = _SlotSummary(
                        anchor_row_count=len(selected),
                        anchor_slice_sha256=slice_sha,
                        upstream_status=status,
                        reason_code=reason,
                        readouts=readouts,
                    )

    feature_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    predictor_rows: list[dict[str, Any]] = []
    for cell in cells:
        cell_feature_start = len(feature_rows)
        cell_mask_start = len(mask_rows)
        for scene in ordered_scenes:
            segment_id = scene.domain.segment_id
            candidate_masks: list[dict[str, Any]] = []
            for candidate_ordinal, candidate_id in CANDIDATES:
                summary = slot_summaries[
                    (segment_id, cell.feature_id, cell.horizon_id, candidate_ordinal)
                ]
                status = summary.upstream_status
                reason = summary.reason_code
                if summary.readouts is None:
                    predictor = _na_value(reason)
                else:
                    predictor = _finite_value(summary.readouts[cell.readout_id])
                feature_row = {
                    "schema_version": "rq014-g2r-blind-feature-row-v1",
                    "cell_index": cell.cell_index,
                    "cell_id": cell.cell_id,
                    "segment_id": segment_id,
                    "candidate_ordinal": candidate_ordinal,
                    "candidate_id": candidate_id,
                    "feature_id": cell.feature_id,
                    "sampling_id": cell.sampling_id,
                    "temporal_id": cell.temporal_id,
                    "horizon_id": cell.horizon_id,
                    "readout_id": cell.readout_id,
                    "predictor_value": predictor,
                    "upstream_status": status,
                    "reason_code": reason,
                    "anchor_row_count": summary.anchor_row_count,
                    "anchor_slice_sha256": summary.anchor_slice_sha256,
                }
                if set(feature_row) != FEATURE_ROW_KEYS or status_by_reason.get(reason) != status:
                    raise G2ROrchestrationError("feature-bank row contract drift")
                feature_rows.append(feature_row)
                candidate_masks.append(
                    {
                        "candidate_ordinal": candidate_ordinal,
                        "candidate_id": candidate_id,
                        "available": status == "AVAILABLE",
                        "predictor_finite": status == "AVAILABLE",
                        "upstream_status": status,
                        "reason_code": reason,
                    }
                )
            rollup = W3.scene_cell_rollup(
                [
                    {
                        **candidate,
                        "status": candidate["upstream_status"],
                        "predictor_value": feature_rows[-3 + offset]["predictor_value"].get(
                            "value"
                        ),
                    }
                    for offset, candidate in enumerate(candidate_masks)
                ],
                repo_root,
            )
            mask_row = {
                "schema_version": "rq014-g2r-availability-mask-row-v1",
                "cell_index": cell.cell_index,
                "cell_id": cell.cell_id,
                "segment_id": segment_id,
                "candidates": candidate_masks,
                "all_three_available": rollup["all_three_available"],
                "all_three_deviations_finite": rollup["all_three_deviations_finite"],
                "deviation_vector_nonconstant": rollup["deviation_vector_nonconstant"],
                "blind_cell_scene_eligible": rollup["blind_cell_scene_eligible"],
                "scene_cell_status": rollup["scene_cell_status"],
                "reason_code": rollup["reason_code"],
            }
            if set(mask_row) != MASK_ROW_KEYS:
                raise G2ROrchestrationError("availability-mask exact-key drift")
            mask_rows.append(mask_row)

        cell_feature_rows = feature_rows[cell_feature_start:]
        cell_mask_rows = mask_rows[cell_mask_start:]
        candidate_status_rows = [
            {
                "segment_id": row["segment_id"],
                "candidate_ordinal": row["candidate_ordinal"],
                "status": row["upstream_status"],
                "reason_code": row["reason_code"],
            }
            for row in cell_feature_rows
        ]
        cell_status = W3.cell_rollup(candidate_status_rows, repo_root)
        cell_key = (cell.feature_id, cell.horizon_id)
        matching_anchor_bytes = anchor_bytes_by_cell_key[cell_key]
        feature_slice = _jsonl_bytes(cell_feature_rows)
        mask_slice = _jsonl_bytes(cell_mask_rows)
        predictor_row = {
            "schema_version": "rq014-g2r-predictor-manifest-row-v1",
            "cell_index": cell.cell_index,
            "cell_id": cell.cell_id,
            "sampling_id": cell.sampling_id,
            "temporal_id": cell.temporal_id,
            "horizon_id": cell.horizon_id,
            "readout_id": cell.readout_id,
            "registered_scene_count": len(ordered_scenes),
            "registered_candidate_slot_count": len(cell_feature_rows),
            "terminal_candidate_slot_count": sum(
                row["upstream_status"] != "AVAILABLE" for row in cell_feature_rows
            ),
            "available_candidate_slot_count": sum(
                row["upstream_status"] == "AVAILABLE" for row in cell_feature_rows
            ),
            "all_three_available_scene_count": sum(
                row["all_three_available"] for row in cell_mask_rows
            ),
            "finite_nonconstant_scene_count": sum(
                row["blind_cell_scene_eligible"] for row in cell_mask_rows
            ),
            "cell_terminal_status": cell_status["cell_terminal_status"],
            "cell_fatal_upstream_status_or_NA": cell_status[
                "cell_fatal_upstream_status_or_NA"
            ],
            "reason_code": cell_status["reason_code"],
            "anchor_score_row_count": anchor_count_by_cell_key[cell_key],
            "anchor_score_slice_sha256": hashlib.sha256(matching_anchor_bytes).hexdigest(),
            "feature_bank_row_count": len(cell_feature_rows),
            "feature_bank_slice_sha256": hashlib.sha256(feature_slice).hexdigest(),
            "availability_mask_row_count": len(cell_mask_rows),
            "availability_mask_slice_sha256": hashlib.sha256(mask_slice).hexdigest(),
        }
        if set(predictor_row) != PREDICTOR_ROW_KEYS:
            raise G2ROrchestrationError("predictor-manifest exact-key drift")
        predictor_rows.append(predictor_row)

    feature_bytes = _jsonl_bytes(feature_rows)
    mask_bytes = _jsonl_bytes(mask_rows)
    predictor_bytes = _jsonl_bytes(predictor_rows)
    eligible_by_scene = {segment_id: True for segment_id in scene_ids}
    for row in mask_rows:
        eligible_by_scene[row["segment_id"]] &= bool(row["blind_cell_scene_eligible"])
    common_ids = tuple(
        segment_id for segment_id in scene_ids if eligible_by_scene[segment_id]
    )
    common_bytes, common_support_id = _common_support_bytes(common_ids)

    direct_refs = {
        "g2r_anchor_scores": _direct_artifact_ref(
            "g2r_anchor_scores.jsonl",
            "rq014-g2r-anchor-score-row-v1",
            anchor_bytes,
            len(anchor_rows),
        ),
        "g2r_blind_feature_bank": _direct_artifact_ref(
            "g2r_blind_feature_bank.jsonl",
            "rq014-g2r-blind-feature-row-v1",
            feature_bytes,
            len(feature_rows),
        ),
        "g2r_availability_masks": _direct_artifact_ref(
            "g2r_availability_masks.jsonl",
            "rq014-g2r-availability-mask-row-v1",
            mask_bytes,
            len(mask_rows),
        ),
        "common_support_blind_manifest": _direct_artifact_ref(
            "common_support_blind_manifest.csv",
            "rq014-common-support-blind-manifest-v1",
            common_bytes,
            len(common_ids),
        ),
        "g2r_predictor_manifest": _direct_artifact_ref(
            "g2r_predictor_manifest.jsonl",
            "rq014-g2r-predictor-manifest-row-v1",
            predictor_bytes,
            len(predictor_rows),
        ),
    }
    artifact_refs = {
        **{key: value.as_dict() for key, value in prerequisite_artifacts.items()},
        **{key: value.as_dict() for key, value in direct_refs.items()},
    }
    if set(artifact_refs) != {
        "wod_scene_anchor_domain",
        "wod_scene_anchor_domain_manifest",
        "g2r_anchor_scores",
        "g2r_blind_feature_bank",
        "g2r_availability_masks",
        "common_support_blind_manifest",
        "g2r_predictor_manifest",
        "nc_pretstar_history_only_receipt",
    }:
        raise G2ROrchestrationError("output-manifest artifact universe drift")

    counts = {
        "registered_scene_count": len(ordered_scenes),
        "candidate_slots_per_scene": 3,
        "registered_feature_family_count": 16,
        "registered_horizon_count": 2,
        "registered_readout_count": 10,
        "registered_cell_count": len(cells),
        "feature_bank_row_count": len(feature_rows),
        "availability_mask_row_count": len(mask_rows),
        "predictor_manifest_row_count": len(predictor_rows),
        "anchor_domain_group_count": len(domain_groups),
        "common_support_scene_count": len(common_ids),
    }
    if not fixture_mode and counts != {
        "registered_scene_count": EXPECTED_SCENE_COUNT,
        "candidate_slots_per_scene": 3,
        "registered_feature_family_count": 16,
        "registered_horizon_count": 2,
        "registered_readout_count": 10,
        "registered_cell_count": EXPECTED_CELL_COUNT,
        "feature_bank_row_count": EXPECTED_FEATURE_BANK_ROWS,
        "availability_mask_row_count": EXPECTED_MASK_ROWS,
        "predictor_manifest_row_count": EXPECTED_CELL_COUNT,
        "anchor_domain_group_count": EXPECTED_ANCHOR_GROUP_COUNT,
        "common_support_scene_count": len(common_ids),
    }:
        raise G2ROrchestrationError("production output cardinality drift")

    output_manifest = {
        "schema_version": "rq014-g2r-output-manifest-v1",
        "operation": OPERATION,
        "run_id": run_id,
        "git_commit": git_commit,
        "created_at_utc": created_at_utc,
        "source_contracts": _source_contract_refs(contract, repo_root),
        "lineage": {key: dict(lineage[key]) for key in sorted(lineage)},
        "artifacts": artifact_refs,
        "counts": counts,
        "canonical_cell_ids_sha256": contract["grid_contract"][
            "canonical_cell_ids_sha256"
        ],
        "common_support_id": common_support_id,
        "forbidden_output_scan": {
            "rating_field_count": 0,
            "leaderboard_file_count": 0,
            "recovery_ledger_file_count": 0,
        },
        "status": "COMPLETE",
    }
    manifest_bytes = W2.canonical_json_bytes(output_manifest)
    artifacts = {
        "g2r_anchor_scores.jsonl": anchor_bytes,
        "g2r_blind_feature_bank.jsonl": feature_bytes,
        "g2r_availability_masks.jsonl": mask_bytes,
        "common_support_blind_manifest.csv": common_bytes,
        "g2r_predictor_manifest.jsonl": predictor_bytes,
        "g2r_output_manifest.json": manifest_bytes,
    }
    _validate_direct_output_boundary(artifacts)
    row_counts = {
        "g2r_anchor_scores.jsonl": len(anchor_rows),
        "g2r_blind_feature_bank.jsonl": len(feature_rows),
        "g2r_availability_masks.jsonl": len(mask_rows),
        "common_support_blind_manifest.csv": len(common_ids),
        "g2r_predictor_manifest.jsonl": len(predictor_rows),
        "g2r_output_manifest.json": 1,
    }
    return BlindOutputBuild(
        artifacts=artifacts,
        row_counts=row_counts,
        anchor_score_rows=tuple(anchor_rows),
        feature_rows=tuple(feature_rows),
        mask_rows=tuple(mask_rows),
        predictor_rows=tuple(predictor_rows),
        common_support_ids=common_ids,
    )


def scene_passes_blind_predicate(
    mask_rows: Sequence[Mapping[str, Any]],
    segment_id: str,
    *,
    expected_cell_count: int = EXPECTED_CELL_COUNT,
) -> bool:
    """Evaluate the frozen rating-free all-cell scene predicate from mask rows."""

    selected = [row for row in mask_rows if row.get("segment_id") == segment_id]
    if len(selected) != expected_cell_count:
        return False
    cell_indices = [row.get("cell_index") for row in selected]
    if cell_indices != list(range(expected_cell_count)):
        return False
    return all(
        row.get("all_three_available") is True
        and row.get("all_three_deviations_finite") is True
        and row.get("deviation_vector_nonconstant") is True
        and row.get("blind_cell_scene_eligible") is True
        for row in selected
    )


def _validate_direct_output_boundary(artifacts: Mapping[str, bytes]) -> None:
    if tuple(artifacts) != DIRECT_ARTIFACT_NAMES:
        raise G2ROrchestrationError("direct artifact order or names drift")
    forbidden_names = ("leaderboard", "recovery_ledger")
    if any(token in name.lower() for name in artifacts for token in forbidden_names):
        raise G2ROrchestrationError("forbidden G3/D4 artifact emitted")
    for name, data in artifacts.items():
        if not data or not data.endswith(b"\n") or b"\r" in data:
            raise G2ROrchestrationError(f"noncanonical stored bytes: {name}")
    direct_rows = b"".join(
        artifacts[name]
        for name in artifacts
        if name != "g2r_output_manifest.json"
    ).lower()
    if b"preference_score" in direct_rows or b"observed_rating" in direct_rows:
        raise G2ROrchestrationError("rating field leaked into direct G2R outputs")


def write_staged_artifacts(output_root: Path, build: BlindOutputBuild) -> None:
    """Write six standalone files to an existing empty non-symlink directory."""

    root = output_root.resolve()
    if output_root.is_symlink() or not root.is_dir() or any(root.iterdir()):
        raise G2ROrchestrationError("staging output root must be an empty regular directory")
    for name in DIRECT_ARTIFACT_NAMES:
        path = root / name
        if path.exists() or path.is_symlink():
            raise G2ROrchestrationError(f"staging artifact already exists: {name}")
        path.write_bytes(build.artifacts[name])
        if path.stat().st_size != len(build.artifacts[name]):
            raise G2ROrchestrationError(f"staging artifact size mismatch: {name}")


__all__ = [
    "ArtifactReference",
    "BlindOutputBuild",
    "CandidateTerminalError",
    "CellSpec",
    "G2ROrchestrationError",
    "SceneBlindInput",
    "build_anchor_score_rows",
    "build_blind_output_artifacts",
    "canonical_cells",
    "load_verified_scorer_rows",
    "scene_passes_blind_predicate",
    "write_staged_artifacts",
]
