from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.rq014 import build_wod_m3_anchors as W2
from scripts.rq014 import score_wod_m3_deviations as W3
from sociality_estimation.verifier import model


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests/fixtures/rq014_g2r_v1"
READOUT_FIXTURE = FIXTURE_ROOT / "deviations_readouts_v1.json"
STATUS_FIXTURE = FIXTURE_ROOT / "availability_statuses_v1.json"
M3_INPUT_FIXTURE = FIXTURE_ROOT / "m3_input_row_expected.json"


def _strict_load(path: Path) -> Any:
    def reject_constant(token: str) -> None:
        raise ValueError(f"nonfinite token: {token}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate key: {key}")
            output[key] = value
        return output

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


def _decode(token: str) -> float:
    special = {
        "NONFINITE_NAN": math.nan,
        "NONFINITE_POSITIVE_INFINITY": math.inf,
        "NONFINITE_NEGATIVE_INFINITY": -math.inf,
    }
    return special[token] if token in special else float.fromhex(token)


def _pointwise_result(values: tuple[float, float, float, float]) -> tuple[Any, str, str]:
    try:
        deviations = W3.pointwise_deviations(*values)
    except W3.M3ScoringNumericalFailure:
        return "NA", "F_M3_SCORING_NUMERICAL_FAILURE", "M3_SCORING_NUMERICAL_FAILURE"
    return (
        {key: value.hex() for key, value in zip(("nex", "nmd", "amd"), deviations)},
        "F_AVAILABLE_CONTINUE",
        "AVAILABLE",
    )


def test_w3_pointwise_and_all_ten_readouts_reproduce_w1_golden_bytes() -> None:
    fixture = _strict_load(READOUT_FIXTURE)
    ordinary = fixture["ordinary_case"]
    anchors: list[dict[str, str]] = []
    pointwise: list[tuple[float, float, float]] = []
    for anchor in ordinary["anchors"]:
        values = tuple(
            float.fromhex(anchor[key])
            for key in ("value_hex", "lower_hex", "median_hex", "upper_hex")
        )
        observed = W3.pointwise_deviations(*values)
        pointwise.append(observed)
        anchors.append(
            {
                "amd_hex": observed[2].hex(),
                "lower_hex": values[1].hex(),
                "median_hex": values[2].hex(),
                "nex_hex": observed[0].hex(),
                "nmd_hex": observed[1].hex(),
                "tau_s_hex": anchor["tau_s_hex"],
                "upper_hex": values[3].hex(),
                "value_hex": values[0].hex(),
            }
        )
    readouts = W3.physical_time_readouts(
        [float.fromhex(anchor["tau_s_hex"]) for anchor in ordinary["anchors"]],
        [row[0] for row in pointwise],
        [row[1] for row in pointwise],
        [row[2] for row in pointwise],
    )

    direct_cases = []
    for case in fixture["direct_pointwise_cases"]:
        encoded = case["inputs_hex_or_token"]
        values = tuple(
            _decode(encoded[key]) for key in ("value", "lower", "median", "upper")
        )
        deviations, reason, status = _pointwise_result(values)
        direct_cases.append(
            {
                "case_id": case["case_id"],
                "expected_deviations_hex_or_NA": deviations,
                "expected_reason_code": reason,
                "expected_status": status,
                "inputs_hex_or_token": encoded,
            }
        )

    invalid_cases = []
    for case in fixture["invalid_interval_cases"]:
        encoded = case["inputs_hex_or_token"]
        values = tuple(
            _decode(encoded[key]) for key in ("value", "lower", "median", "upper")
        )
        with pytest.raises(W3.M3ScoringNumericalFailure):
            W3.pointwise_deviations(*values)
        invalid_cases.append(
            {
                "case_id": case["case_id"],
                "expected_reason_code": "F_M3_SCORING_NUMERICAL_FAILURE",
                "expected_status": "M3_SCORING_NUMERICAL_FAILURE",
                "inputs_hex_or_token": encoded,
            }
        )

    degenerate_cases = []
    for case in fixture["degenerate_readout_cases"]:
        times = [float.fromhex(value) for value in case["times_s_hex"]]
        with pytest.raises(W3.M3ScoringNumericalFailure):
            W3.physical_time_readouts(
                times, [0.0] * len(times), [0.0] * len(times), [0.0] * len(times)
            )
        degenerate_cases.append(
            {
                "case_id": case["case_id"],
                "expected_reason_code": "F_M3_SCORING_NUMERICAL_FAILURE",
                "expected_status": "M3_SCORING_NUMERICAL_FAILURE",
                "times_s_hex": case["times_s_hex"],
            }
        )

    reproduced = {
        "degenerate_readout_cases": degenerate_cases,
        "direct_pointwise_cases": direct_cases,
        "invalid_interval_cases": invalid_cases,
        "ordinary_case": {
            "anchors": anchors,
            "expected_readouts_hex": {
                key: value.hex() for key, value in sorted(readouts.items())
            },
        },
        "readout_order": list(W3.READOUT_ORDER),
        "schema_version": "rq014-g2r-deviations-readouts-golden-v1",
    }
    assert W2.canonical_json_bytes(reproduced) == READOUT_FIXTURE.read_bytes()
    assert hashlib.sha256(READOUT_FIXTURE.read_bytes()).hexdigest() == (
        "075613a890633e99f87f3d09c398141c34735f451f0d164bab45c162ba2c65cc"
    )


@pytest.mark.parametrize(
    "times",
    ([0.0], [0.0, 0.0], [1.0, 0.0], [0.0, math.inf]),
)
def test_w3_readouts_fail_closed_for_degenerate_time_axes(times: list[float]) -> None:
    with pytest.raises(W3.M3ScoringNumericalFailure):
        W3.physical_time_readouts(
            times, [0.0] * len(times), [0.0] * len(times), [0.0] * len(times)
        )


def test_w3_a10_candidate_scene_cell_and_global_rows_reproduce_w1_golden_bytes() -> None:
    fixture = _strict_load(STATUS_FIXTURE)
    precedence_input = copy.deepcopy(fixture["candidate_precedence_case"]["candidate_rows"])
    precedence_status, precedence_reason = W3.select_scene_candidate_failure(precedence_input)
    propagation_input = copy.deepcopy(
        fixture["m3_numerical_failure_propagation"]["candidate_rows"]
    )
    reproduced = {
        "candidate_precedence_case": {
            "candidate_rows": precedence_input,
            "expected_reason_code": precedence_reason,
            "expected_status": precedence_status,
        },
        "candidate_status_rows": W3.candidate_status_rows(),
        "global_fatal_rows": W3.global_fatal_rows(),
        "m3_numerical_failure_propagation": {
            "candidate_rows": propagation_input,
            "expected_cell": W3.cell_rollup(propagation_input),
            "expected_scene_cell": W3.scene_cell_rollup(propagation_input),
        },
        "schema_version": "rq014-g2r-availability-statuses-golden-v1",
    }
    assert W2.canonical_json_bytes(reproduced) == STATUS_FIXTURE.read_bytes()
    assert hashlib.sha256(STATUS_FIXTURE.read_bytes()).hexdigest() == (
        "703eb399029c18275f0efb94db3b42297e67495e780b0ae3a3433ba48cb96461"
    )


def test_w3_a10_available_scene_requires_three_finite_nonconstant_candidates() -> None:
    rows = [
        {
            "available": True,
            "candidate_ordinal": ordinal,
            "predictor_finite": True,
            "predictor_value": value,
            "reason_code": "F_AVAILABLE_CONTINUE",
            "status": "AVAILABLE",
        }
        for ordinal, value in enumerate((0.0, 1.0, 2.0), start=1)
    ]
    observed = W3.scene_cell_rollup(rows)
    assert observed == {
        "all_three_available": True,
        "all_three_deviations_finite": True,
        "blind_cell_scene_eligible": True,
        "deviation_vector_nonconstant": True,
        "reason_code": "F_AVAILABLE_CONTINUE",
        "scene_cell_status": "AVAILABLE",
    }
    bad = copy.deepcopy(rows)
    bad[1]["reason_code"] = "F_NOT_REGISTERED"
    with pytest.raises(W3.G2RScoringError, match="unregistered"):
        W3.scene_cell_rollup(bad)


def test_w3_w2_rows_decode_to_exact_model_frame_without_changing_row_hash() -> None:
    row = _strict_load(M3_INPUT_FIXTURE)
    original = M3_INPUT_FIXTURE.read_bytes()
    frame = W3.m3_rows_to_frame([row])
    assert tuple(frame.columns) == W2.M3_INPUT_COLUMNS
    assert frame.iloc[0].tolist() == row["values"]
    row_bytes, row_hash = W2.m3_input_row_bytes_and_sha256(row)
    assert row_bytes == original
    assert row_hash == "9ecc870d825b39239be06ea142dca1691b9834f882cf3ccd662f87eafde33762"

    typed = copy.deepcopy(row)
    typed["values"][20] = {"kind": "NA", "reason_code": "F_M3_INPUT_TTC_UNDEFINED"}
    typed["values"][21] = {"kind": "NA", "reason_code": "F_M3_INPUT_APET_UNDEFINED"}
    decoded = W3.m3_rows_to_frame([typed])
    assert math.isnan(float(decoded.iloc[0, 20]))
    assert math.isnan(float(decoded.iloc[0, 21]))
    typed["values"][20]["reason_code"] = "F_M3_INPUT_APET_UNDEFINED"
    with pytest.raises(W3.G2RScoringError, match="invalid typed NA"):
        W3.m3_rows_to_frame([typed])


def test_w3_pre_mask_helper_calls_direct_quantile_calibration_and_never_masks_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _strict_load(M3_INPUT_FIXTURE)
    quantiles = np.asarray(
        [[-0.75, -0.5, -0.25, 0.125, 0.5, 0.75, 1.0]], dtype=np.float32
    )
    calls: list[str] = []
    real_calibrated_bounds = model.calibrated_bounds

    def fake_predict(tier_model: object, frame: Any):
        calls.append("predict_tier_quantiles")
        assert tier_model == "tier"
        assert tuple(frame.columns) == W2.M3_INPUT_COLUMNS
        return quantiles.copy(), np.zeros_like(quantiles, dtype=bool), {"rows": 1}

    def recording_bounds(lower: np.ndarray, upper: np.ndarray, radius: float):
        calls.append("calibrated_bounds")
        return real_calibrated_bounds(lower, upper, radius)

    def fake_gate(frame: Any, gate_model: object):
        calls.append("apply_gate_diagnostic_only")
        assert gate_model == "gate"
        return np.asarray([False]), {"gate_pass_rows": 0}

    monkeypatch.setattr(model, "predict_tier_quantiles", fake_predict)
    monkeypatch.setattr(model, "calibrated_bounds", recording_bounds)
    monkeypatch.setattr(model, "apply_gate", fake_gate)
    observed = W3.score_pre_mask_from_bundle(
        [row],
        {
            "feature_contract": {"required_input_columns": sorted(W2.M3_INPUT_COLUMNS)},
            "gate_model": "gate",
            "radii": {"90": {"c_alpha": 0.125}},
            "tier_model": "tier",
        },
    )[0]
    expected_lo, expected_hi = real_calibrated_bounds(
        quantiles[:, model.Q_INDEX[0.05]],
        quantiles[:, model.Q_INDEX[0.95]],
        0.125,
    )
    assert calls == [
        "predict_tier_quantiles",
        "calibrated_bounds",
        "apply_gate_diagnostic_only",
    ]
    assert observed.q_0p5 == float(quantiles[0, model.Q_INDEX[0.5]])
    assert observed.lo_90 == float(expected_lo[0])
    assert observed.hi_90 == float(expected_hi[0])
    assert observed.support_gate_pass is False and observed.ood_abstain is True
    assert math.isfinite(observed.lo_90) and math.isfinite(observed.hi_90)

    with pytest.raises(W3.G2RScoringError, match="feature contract is malformed"):
        W3.score_pre_mask_from_bundle(
            [row],
            {
                "feature_contract": {},
                "gate_model": "gate",
                "radii": {"90": {"c_alpha": 0.125}},
                "tier_model": "tier",
            },
        )


def test_w3_scorer_artifact_gate_rejects_unreviewed_bytes(tmp_path: Path) -> None:
    fake = tmp_path / "m3_scorer.joblib"
    fake.write_bytes(b"not-the-reviewed-scorer")
    with pytest.raises(W3.G2RScoringError, match="size or SHA-256 mismatch"):
        W3.score_pre_mask_m3([_strict_load(M3_INPUT_FIXTURE)], fake)


def test_w3_binds_a08_a15_at_portable_parity_and_keeps_operation_denied() -> None:
    contract = _strict_load(ROOT / W3.G2R_OUTPUT_CONTRACT_PATH)
    binding = contract["fixture_bindings"]["m3_pre_mask_golden"]
    assert binding["binding_status"] == "BOUND_W3_PORTABLE_PARITY_1E-7"
    assert binding["sha256"] == (
        "e7c332a13d18187f2b671042d459af91db8702dcda15917b773692858c44534f"
    )
    assert binding["independent_parity"]["absolute_tolerance"] == 1e-7
    assert "PENDING_W3_SCORER" not in json.dumps(contract, sort_keys=True)
    assert contract["future_operation_binding"]["central_authorization"] == "ALLOWED"
    authorization = _strict_load(ROOT / "configs/research_authorization.json")
    assert "rq014_r2_blind_feature_build" in authorization["authorizations"]["RQ014"][
        "allowed_operations"
    ]
