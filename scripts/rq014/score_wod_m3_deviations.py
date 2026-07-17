#!/usr/bin/env python3
"""Rating-blind RQ014 G2R M3 scoring, deviation, and readout kernel.

This W3 module consumes the checksum-frozen W2 32-column M3 input rows.  Its
scalar deviation/readout functions are scorer-independent.  The scorer path is
an integration surface only: it verifies and loads the frozen RQ009 M3 bundle,
calls the reviewed pre-mask quantile/calibration helpers directly, and keeps OOD
results diagnostic.  It does not implement the 320-cell orchestration, expose a
managed operation, access ratings, or authorize ``rq014_r2_blind_feature_build``.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.rq014.build_wod_m3_anchors import (
    M3_INPUT_COLUMNS,
    canonical_json_bytes,
    m3_input_row_bytes_and_sha256,
    sha256_file,
)


G2R_OUTPUT_CONTRACT_PATH = Path("reports/plans/RQ014_g2r_output_contract_v1.json")
G2R_OUTPUT_CONTRACT_SHA256 = (
    "3be8da8e49fddee75ce387b502c0d1d6e16da232d34e208da4c66e2a4d2f36dc"
)
M3_SCORER_SHA256 = "b04999aba29a82fb71a97ac22c728479a7734e24a0b32189d08f95184d74f253"
M3_SCORER_SIZE_BYTES = 88_306_301
READOUT_ORDER = (
    "NEX_MEAN",
    "NEX_MAX",
    "NEX_LAST",
    "NEX_FOUT",
    "NMD_MEAN",
    "NMD_MAX",
    "NMD_LAST",
    "AMD_MEAN",
    "AMD_MAX",
    "AMD_LAST",
)
_TYPED_NA_BY_COLUMN = {
    "closing_ttc_anchor": "F_M3_INPUT_TTC_UNDEFINED",
    "apet_online_proxy": "F_M3_INPUT_APET_UNDEFINED",
}


class G2RScoringError(ValueError):
    """Fail-closed input, authority, or scorer integration failure."""


class M3ScoringNumericalFailure(G2RScoringError):
    """Typed A09/A10 failure for invalid intervals or derived values."""

    status = "M3_SCORING_NUMERICAL_FAILURE"
    reason_code = "F_M3_SCORING_NUMERICAL_FAILURE"


@dataclass(frozen=True)
class PreMaskM3Score:
    """One verified W2 row's pre-OOD-mask M3 interval and diagnostics."""

    m3_input_row_sha256: str
    q_0p5: float
    lo_90: float
    hi_90: float
    support_gate_pass: bool
    ood_abstain: bool


def _repo_root(repo_root: Path | None = None) -> Path:
    return (repo_root or Path(__file__).resolve().parents[2]).resolve()


def _load_output_contract(repo_root: Path | None = None) -> Mapping[str, Any]:
    import json

    path = _repo_root(repo_root) / G2R_OUTPUT_CONTRACT_PATH
    if sha256_file(path) != G2R_OUTPUT_CONTRACT_SHA256:
        raise G2RScoringError("G2R output contract SHA-256 drift")
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise G2RScoringError("G2R output contract is unreadable") from exc
    if contract.get("authority_status") != "W1_OUTPUT_SCHEMA_FROZEN_OPERATION_DENIED":
        raise G2RScoringError("G2R denied-operation authority status drift")
    return contract


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise M3ScoringNumericalFailure(f"{label} is not a binary64 number")
    result = float(value)
    if not math.isfinite(result):
        raise M3ScoringNumericalFailure(f"{label} is nonfinite")
    return 0.0 if result == 0.0 else result


def pointwise_deviations(
    value: float,
    lower: float,
    median: float,
    upper: float,
) -> tuple[float, float, float]:
    """Compute frozen pointwise NEX, NMD, and AMD with A09 checks."""

    value, lower, median, upper = (
        _finite_float(item, label=label)
        for item, label in zip(
            (value, lower, median, upper),
            ("candidate IPV", "M3 lower", "M3 median", "M3 upper"),
        )
    )
    if not lower < median < upper:
        raise M3ScoringNumericalFailure("M3 interval must satisfy strict L<M<U")
    if value < lower:
        nex = (lower - value) / (median - lower)
    elif value > upper:
        nex = (value - upper) / (upper - median)
    else:
        nex = 0.0
    if value < median:
        nmd = (median - value) / (median - lower)
    elif value > median:
        nmd = (value - median) / (upper - median)
    else:
        nmd = 0.0
    result = (nex, nmd, abs(value - median))
    if not all(math.isfinite(item) for item in result):
        raise M3ScoringNumericalFailure("derived deviation is nonfinite")
    return tuple(0.0 if item == 0.0 else item for item in result)


def physical_time_readouts(
    times_s: Iterable[float],
    nex: Iterable[float],
    nmd: Iterable[float],
    amd: Iterable[float],
) -> dict[str, float]:
    """Reduce aligned anchor deviations to the frozen ten readouts."""

    times = tuple(_finite_float(item, label="anchor time") for item in times_s)
    vectors = {
        "NEX": tuple(_finite_float(item, label="NEX") for item in nex),
        "NMD": tuple(_finite_float(item, label="NMD") for item in nmd),
        "AMD": tuple(_finite_float(item, label="AMD") for item in amd),
    }
    if len(times) < 2 or any(len(values) != len(times) for values in vectors.values()):
        raise M3ScoringNumericalFailure("readout needs at least two aligned anchors")
    deltas = tuple(right - left for left, right in zip(times, times[1:]))
    if any(not math.isfinite(delta) or delta <= 0.0 for delta in deltas):
        raise M3ScoringNumericalFailure("anchor times must be strictly increasing")
    duration = times[-1] - times[0]
    if not math.isfinite(duration) or duration <= 0.0:
        raise M3ScoringNumericalFailure("anchor duration must be positive and finite")

    def trapezoid_mean(values: tuple[float, ...]) -> float:
        area = math.fsum(
            delta * (left + right) / 2.0
            for delta, left, right in zip(deltas, values, values[1:])
        )
        return area / duration

    result: dict[str, float] = {}
    for source_id, values in vectors.items():
        result[f"{source_id}_MEAN"] = trapezoid_mean(values)
        result[f"{source_id}_MAX"] = max(values)
        result[f"{source_id}_LAST"] = values[-1]
    outside = tuple(1.0 if value > 0.0 else 0.0 for value in vectors["NEX"])
    result["NEX_FOUT"] = trapezoid_mean(outside)
    if set(result) != set(READOUT_ORDER) or not all(
        math.isfinite(value) for value in result.values()
    ):
        raise M3ScoringNumericalFailure("derived readout is missing or nonfinite")
    return {key: 0.0 if result[key] == 0.0 else result[key] for key in READOUT_ORDER}


def candidate_status_rows(repo_root: Path | None = None) -> list[dict[str, Any]]:
    """Return the exact A10 candidate namespace with availability flags."""

    rows = _load_output_contract(repo_root)["status_contract"]["candidate_upstream_statuses"]
    return [
        {
            **row,
            "available": row["status"] == "AVAILABLE",
            "predictor_finite": row["status"] == "AVAILABLE",
        }
        for row in rows
    ]


def global_fatal_rows(repo_root: Path | None = None) -> list[dict[str, Any]]:
    """Return the exact A10 global-fatal status-to-reason mappings."""

    rows = _load_output_contract(repo_root)["status_contract"][
        "global_fatal_status_reason_mappings"
    ]
    return [
        {
            **row,
            "cell_fatal_upstream_status_or_NA": row["status"],
            "cell_terminal_status": "CELL_FATAL",
            "global_abort": True,
        }
        for row in rows
    ]


def _validated_failure_rows(
    rows: Sequence[Mapping[str, Any]], repo_root: Path | None = None
) -> list[Mapping[str, Any]]:
    definitions = {
        row["reason_code"]: row
        for row in _load_output_contract(repo_root)["status_contract"][
            "candidate_upstream_statuses"
        ]
    }
    validated: list[Mapping[str, Any]] = []
    for row in rows:
        reason = row.get("reason_code")
        expected = definitions.get(reason)
        if expected is None or row.get("status") != expected["status"]:
            raise G2RScoringError(f"unregistered candidate status/reason: {reason!r}")
        if "reason_priority" in row and row["reason_priority"] != expected["reason_priority"]:
            raise G2RScoringError(f"candidate reason-priority drift: {reason}")
        if expected["status"] != "AVAILABLE":
            validated.append(row)
    return validated


def select_scene_candidate_failure(
    rows: Sequence[Mapping[str, Any]], repo_root: Path | None = None
) -> tuple[str, str]:
    """Select a scene-cell candidate failure by the frozen A10 precedence."""

    if not rows:
        raise G2RScoringError("candidate status rows are empty")
    failures = _validated_failure_rows(rows, repo_root)
    if not failures:
        return "AVAILABLE", "F_AVAILABLE_CONTINUE"
    priorities = {
        row["reason_code"]: row["reason_priority"]
        for row in _load_output_contract(repo_root)["status_contract"][
            "candidate_upstream_statuses"
        ]
    }
    try:
        selected = min(
            failures,
            key=lambda row: (
                priorities[row["reason_code"]],
                int(row["candidate_ordinal"]),
                row["reason_code"].encode("utf-8"),
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise G2RScoringError("candidate failure row is malformed") from exc
    return str(selected["status"]), str(selected["reason_code"])


def scene_cell_rollup(
    rows: Sequence[Mapping[str, Any]], repo_root: Path | None = None
) -> dict[str, Any]:
    """Propagate three candidate statuses to one rating-blind scene-cell."""

    try:
        ordinals = {int(row["candidate_ordinal"]) for row in rows}
    except (KeyError, TypeError, ValueError) as exc:
        raise G2RScoringError("candidate ordinal is malformed") from exc
    if len(rows) != 3 or ordinals != {1, 2, 3}:
        raise G2RScoringError("scene-cell requires candidate ordinals 1,2,3 exactly once")
    status, reason = select_scene_candidate_failure(rows, repo_root)
    if status != "AVAILABLE":
        return {
            "all_three_available": False,
            "all_three_deviations_finite": False,
            "blind_cell_scene_eligible": False,
            "deviation_vector_nonconstant": False,
            "reason_code": reason,
            "scene_cell_status": status,
        }
    values: list[float] = []
    for row in sorted(rows, key=lambda item: int(item["candidate_ordinal"])):
        if row.get("available") is not True or row.get("predictor_finite") is not True:
            raise G2RScoringError("AVAILABLE candidate flags are inconsistent")
        values.append(_finite_float(row.get("predictor_value"), label="candidate deviation"))
    nonconstant = len(set(values)) > 1
    return {
        "all_three_available": True,
        "all_three_deviations_finite": True,
        "blind_cell_scene_eligible": nonconstant,
        "deviation_vector_nonconstant": nonconstant,
        "reason_code": "F_AVAILABLE_CONTINUE",
        "scene_cell_status": "AVAILABLE",
    }


def cell_rollup(
    candidate_rows: Sequence[Mapping[str, Any]], repo_root: Path | None = None
) -> dict[str, Any]:
    """Propagate nonfatal candidate/scene failures to one terminal cell."""

    if not candidate_rows:
        raise G2RScoringError("cell status rows are empty")
    failures = _validated_failure_rows(candidate_rows, repo_root)
    if not failures:
        reason = "F_AVAILABLE_CONTINUE"
    else:
        priorities = {
            row["reason_code"]: row["reason_priority"]
            for row in _load_output_contract(repo_root)["status_contract"][
                "candidate_upstream_statuses"
            ]
        }
        try:
            selected = min(
                failures,
                key=lambda row: (
                    priorities[row["reason_code"]],
                    str(row.get("segment_id", "")).encode("utf-8"),
                    int(row["candidate_ordinal"]),
                    row["reason_code"].encode("utf-8"),
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise G2RScoringError("cell failure row is malformed") from exc
        reason = str(selected["reason_code"])
    return {
        "cell_fatal_upstream_status_or_NA": "NA",
        "cell_terminal_status": "TERMINAL",
        "global_abort": False,
        "reason_code": reason,
    }


def m3_rows_to_frame(rows: Sequence[Mapping[str, Any]]):
    """Decode W2 canonical rows into the exact model DataFrame in memory."""

    import numpy as np
    import pandas as pd

    records: list[dict[str, Any]] = []
    for row in rows:
        if set(row) != {"schema_version", "columns", "values"}:
            raise G2RScoringError("M3 input row has an exact-key mismatch")
        if row["schema_version"] != "rq014-g2r-m3-input-row-v1":
            raise G2RScoringError("M3 input row schema drift")
        if tuple(row["columns"]) != M3_INPUT_COLUMNS or len(row["values"]) != len(
            M3_INPUT_COLUMNS
        ):
            raise G2RScoringError("M3 input row column order drift")
        decoded: dict[str, Any] = {}
        for index, (column, value) in enumerate(zip(M3_INPUT_COLUMNS, row["values"])):
            if index < 25:
                if isinstance(value, Mapping):
                    expected_reason = _TYPED_NA_BY_COLUMN.get(column)
                    if dict(value) != {"kind": "NA", "reason_code": expected_reason}:
                        raise G2RScoringError(f"invalid typed NA for {column}")
                    decoded[column] = np.nan
                else:
                    decoded[column] = _finite_float(value, label=column)
            elif not isinstance(value, str) or not value:
                raise G2RScoringError(f"invalid categorical token for {column}")
            else:
                decoded[column] = value
        records.append(decoded)
    return pd.DataFrame.from_records(records, columns=M3_INPUT_COLUMNS)


def _verify_scorer_path(scorer_path: Path) -> Path:
    path = scorer_path.resolve()
    if scorer_path.is_symlink() or not path.is_file():
        raise G2RScoringError("M3 scorer must be a regular non-symlink file")
    if path.stat().st_size != M3_SCORER_SIZE_BYTES or sha256_file(path) != M3_SCORER_SHA256:
        raise G2RScoringError("M3 scorer size or SHA-256 mismatch")
    return path


def score_pre_mask_from_bundle(
    rows: Sequence[Mapping[str, Any]], scorer_bundle: Mapping[str, Any]
) -> list[PreMaskM3Score]:
    """Call the frozen model helpers and retain calibrated bounds before masking."""

    import numpy as np
    from sociality_estimation.verifier import model

    required = {"tier_model", "gate_model", "radii", "feature_contract"}
    if required - set(scorer_bundle):
        raise G2RScoringError("M3 scorer bundle is missing required keys")
    try:
        required_columns = tuple(
            scorer_bundle["feature_contract"]["required_input_columns"]
        )
    except (KeyError, TypeError) as exc:
        raise G2RScoringError("M3 feature contract is malformed") from exc
    if len(required_columns) != len(M3_INPUT_COLUMNS) or set(required_columns) != set(
        M3_INPUT_COLUMNS
    ):
        raise G2RScoringError("M3 feature-contract column universe drift")
    frame = m3_rows_to_frame(rows)
    quantiles, _, _ = model.predict_tier_quantiles(scorer_bundle["tier_model"], frame)
    if quantiles.shape != (len(rows), len(model.QUANTILE_LEVELS)):
        raise G2RScoringError("M3 quantile output shape drift")
    lower_raw = quantiles[:, model.Q_INDEX[0.05]].astype(np.float32, copy=False)
    median = quantiles[:, model.Q_INDEX[0.50]].astype(np.float32, copy=False)
    upper_raw = quantiles[:, model.Q_INDEX[0.95]].astype(np.float32, copy=False)
    try:
        radius = float(scorer_bundle["radii"]["90"]["c_alpha"])
    except (KeyError, TypeError, ValueError) as exc:
        raise G2RScoringError("M3 90-percent calibration radius is malformed") from exc
    calibrated_lower, calibrated_upper = model.calibrated_bounds(
        lower_raw, upper_raw, radius
    )
    gate_ok, _ = model.apply_gate(frame, scorer_bundle["gate_model"])
    gate_ok = np.asarray(gate_ok, dtype=bool)
    if gate_ok.shape != (len(rows),):
        raise G2RScoringError("M3 support-gate output shape drift")

    output: list[PreMaskM3Score] = []
    for index, row in enumerate(rows):
        q_0p5 = float(median[index])
        lo_90 = float(calibrated_lower[index])
        hi_90 = float(calibrated_upper[index])
        pointwise_deviations(q_0p5, lo_90, q_0p5, hi_90)
        _, row_sha256 = m3_input_row_bytes_and_sha256(row)
        output.append(
            PreMaskM3Score(
                m3_input_row_sha256=row_sha256,
                q_0p5=0.0 if q_0p5 == 0.0 else q_0p5,
                lo_90=0.0 if lo_90 == 0.0 else lo_90,
                hi_90=0.0 if hi_90 == 0.0 else hi_90,
                support_gate_pass=bool(gate_ok[index]),
                ood_abstain=not bool(gate_ok[index]),
            )
        )
    return output


def score_pre_mask_m3(
    rows: Sequence[Mapping[str, Any]], scorer_path: Path
) -> list[PreMaskM3Score]:
    """Verify/load the pinned scorer and score W2 rows without OOD masking."""

    from sociality_estimation.verifier.scorer import load_scorer

    path = _verify_scorer_path(scorer_path)
    return score_pre_mask_from_bundle(rows, load_scorer(path, verify_hash=True))


__all__ = [
    "G2RScoringError",
    "M3ScoringNumericalFailure",
    "PreMaskM3Score",
    "candidate_status_rows",
    "cell_rollup",
    "global_fatal_rows",
    "m3_rows_to_frame",
    "physical_time_readouts",
    "pointwise_deviations",
    "scene_cell_rollup",
    "score_pre_mask_from_bundle",
    "score_pre_mask_m3",
    "select_scene_candidate_failure",
]
