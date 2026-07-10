from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sociality_estimation.verifier.deviation import raw_envelope_deviation
from sociality_estimation.verifier.anchors import build_m3_anchor_features
from sociality_estimation.verifier.features import relative_state, theil_sen_slope
from sociality_estimation.verifier.model import calibrated_bounds
from sociality_estimation.verifier.scorer import (
    load_scorer,
    quantile_column_name,
    score_anchors,
    score_verifier,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def test_raw_envelope_deviation_contract() -> None:
    signed, absolute, outside = raw_envelope_deviation(
        np.array([0.0, 2.0, -2.0, 1.0, np.nan]),
        np.array([-1.0, -1.0, -1.0, np.nan, -1.0]),
        np.array([1.0, 1.0, 1.0, np.nan, 1.0]),
    )
    np.testing.assert_allclose(signed[:3], [0.0, 1.0, -1.0])
    np.testing.assert_allclose(absolute[:3], [0.0, 1.0, 1.0])
    assert outside.tolist() == [False, True, True, False, False]
    assert np.isnan(signed[3:]).all()


def test_calibrated_bounds_preserve_order_and_epsilon() -> None:
    lower, upper = calibrated_bounds(
        np.array([0.0, 2.0]), np.array([1.0, -1.0]), c_alpha=-0.1
    )
    assert np.all(lower <= upper)
    np.testing.assert_allclose(lower, [0.1 - 1e-10, -1.1 - 1e-10])
    np.testing.assert_allclose(upper, [0.9 + 1e-10, 2.1 + 1e-10])


def test_quantile_column_names() -> None:
    assert quantile_column_name(0.025) == "q_0p025"
    assert quantile_column_name(0.5) == "q_0p5"
    assert quantile_column_name(0.975) == "q_0p975"


def test_sigma01_profile_is_frozen() -> None:
    profile = json.loads((ROOT / "configs/ipv_sigma01_exact.json").read_text())
    assert profile["estimator"] == {
        "solver_mode": "exact",
        "sigma": 0.1,
        "min_observation": 4,
        "reference_clip_margin_m": 60.0,
        "reference_max_points": 40,
        "reference_smooth_points": 40,
    }
    assert profile["sampling"]["nuplan_downsample_factor"] == 2


def test_shared_causal_feature_formulas() -> None:
    state = relative_state(
        np.array([0.0]),
        np.array([0.0]),
        np.array([1.0]),
        np.array([0.0]),
        np.array([10.0]),
        np.array([0.0]),
        np.array([-1.0]),
        np.array([0.0]),
    )
    np.testing.assert_allclose(state["distance"], [10.0])
    np.testing.assert_allclose(state["closing_rate"], [2.0])
    assert theil_sen_slope(np.array([0.0, 1.0, 2.0]), np.array([1.0, 3.0, 5.0])) == 2.0


def test_standardized_history_builds_complete_m3_input() -> None:
    history = pd.DataFrame(
        {
            "timestamp_s": np.arange(10, dtype=float) / 10.0,
            "ego_x": np.arange(10, dtype=float) / 10.0,
            "ego_y": 0.0,
            "ego_vx": 1.0,
            "ego_vy": 0.0,
            "ego_heading": 0.0,
            "counterpart_x": 10.0 - np.arange(10, dtype=float) / 10.0,
            "counterpart_y": 0.0,
            "counterpart_vx": -1.0,
            "counterpart_vy": 0.0,
            "counterpart_heading": np.pi,
            "counterpart_ipv": np.linspace(-0.2, 0.2, 10),
            "counterpart_ipv_error": 0.1,
        }
    )
    categories = {
        "geometry_path_category": "F",
        "geometry_path_relation": "F",
        "turn_pair_label": "S-S",
        "agent_type_pair": "AV;HV",
        "vehicle_type_list": "['AV', 'HV']",
        "av_included": "AV",
        "priority_role": "yield",
    }
    anchor = build_m3_anchor_features(history, categories, case_start_timestamp_s=0.0)
    contract = json.loads(
        (ROOT / "models/rq009_m3/feature_spec_contract.json").read_text()
    )
    assert set(contract["required_input_columns"]).issubset(anchor.columns)
    assert anchor.loc[0, "history_row_count"] == 10
    assert anchor.loc[0, "closing_rate_anchor"] > 0.0


def test_portable_bundle_integrity_and_stable_classes() -> None:
    scorer_path = Path(
        os.environ.get(
            "SOCIALITY_M3_SCORER", ROOT / "models/rq009_m3/m3_scorer.joblib"
        )
    )
    manifest = json.loads((scorer_path.parent / "manifest.json").read_text())
    assert scorer_path.stat().st_size == manifest["artifact"]["size_bytes"]
    assert sha256_file(scorer_path) == manifest["artifact"]["sha256"]
    scorer = load_scorer(scorer_path)
    assert type(scorer["tier_model"]).__module__ == "sociality_estimation.verifier.model"
    assert type(scorer["gate_model"]).__module__ == "sociality_estimation.verifier.model"
    assert scorer["gate_model"].threshold == 1.6072176694869995
    assert scorer["gate_model"].train_reference_rows == 2557510


def test_portable_scorer_matches_frozen_fixture() -> None:
    payload = json.loads(
        (ROOT / "tests/fixtures/m3_verifier_portable_fixture.json").read_text()
    )
    frame = pd.DataFrame(payload["rows"])
    labels = frame.pop("fixture_label").tolist()
    expected = pd.DataFrame(payload["expected"])
    assert expected.pop("fixture_label").tolist() == labels
    actual = score_anchors(frame)
    assert set(actual.columns) == set(expected.columns)
    for column in expected.columns:
        if column in {"support_gate_pass", "ood_abstain"}:
            assert actual[column].tolist() == expected[column].tolist()
        elif pd.api.types.is_numeric_dtype(expected[column]):
            np.testing.assert_allclose(
                actual[column].to_numpy(dtype=float),
                expected[column].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-7,
                equal_nan=True,
            )
        else:
            assert actual[column].tolist() == expected[column].tolist()

    verified = score_verifier(frame, observed_ipv_column="target_ipv_future", level=90)
    assert verified.loc[0, "deviation_abs_exceedance_90"] == 0.0
    assert verified.loc[1, "deviation_signed_exceedance_90"] > 0.0
    assert verified.loc[2, "deviation_signed_exceedance_90"] < 0.0
    assert np.isnan(verified.loc[3:, "deviation_signed_exceedance_90"]).all()
