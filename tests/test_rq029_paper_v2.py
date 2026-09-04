from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from pipelines.simulation.calibrate_rq029_paper_v2 import (
    BASE_OUTPUT,
    BASE_TARGETS,
    AV_REFERENCE,
    DEFAULT_OUTPUT,
    DEFAULT_SOURCE_SESSION,
    DESIGNATED_COUNTERPARTS,
    PAPER_TARGETS,
    _bootstrap_rate_ci,
    _calibrate_run_counts,
    _case_selection_mask,
    _load_designated_counterparts,
    _paper_metrics,
    _weighted_bootstrap_medians,
)
from pipelines.simulation.generate_rq029_human_synthetic import SCENARIOS
from pipelines.simulation.validate_rq029_paper_v2 import validate_dataset


def test_paper_target_contract_preserves_current_printed_values() -> None:
    target = json.loads(PAPER_TARGETS.read_text(encoding="utf-8"))
    assert target["human_runs"] == {
        "drivers": 20,
        "scenarios": 15,
        "runs": 300,
        "missing": 0,
    }
    assert target["human_alpha90"]["run_bootstrap_ci95_rounded"] == [0.034, 0.055]
    assert target["automated_to_human"]["scenario_bootstrap_ci95_rounded"] == [
        1.62,
        2.39,
    ]
    assert target["automated_alpha90"]["scenario_rate_range_paper_display"] == [
        0.027,
        0.174,
    ]
    assert target["automated_alpha90"][
        "scenario_rate_range_from_frozen_counts"
    ] == [0.027434842249657063, 0.17346938775510204]
    assert target["human_signature"]["ego_margin_intervals_available"] is False


def test_synthetic_case_mask_has_186_traceable_runs() -> None:
    selected = _case_selection_mask()
    assert selected.shape == (15, 20)
    assert int(selected.sum()) == 186
    assert set(DESIGNATED_COUNTERPARTS) == set(SCENARIOS)


def test_weighted_bootstrap_median_respects_cluster_multiplicity() -> None:
    values = np.array([1.0, 2.0, 10.0, 20.0])
    clusters = np.array(["a", "a", "b", "b"])
    cluster_index = {"a": 0, "b": 1}
    draw_counts = np.array([[1, 1], [2, 0], [0, 2]], dtype=np.int16)
    medians = _weighted_bootstrap_medians(
        values, clusters, draw_counts, cluster_index
    )
    assert np.allclose(medians, [6.0, 1.5, 15.0])


def test_rate_bootstrap_is_deterministic() -> None:
    n_both = np.array([20, 30, 40, 50])
    n_flagged = np.array([0, 1, 2, 4])
    first = _bootstrap_rate_ci(n_both, n_flagged, seed=19, draws=200)
    second = _bootstrap_rate_ci(n_both, n_flagged, seed=19, draws=200)
    assert first == second


@pytest.mark.skipif(
    not AV_REFERENCE.is_file(),
    reason="local frozen AV reference unavailable",
)
def test_paper_scenario_bootstrap_reproduces_printed_record() -> None:
    human = json.loads(BASE_TARGETS.read_text(encoding="utf-8"))
    automated = json.loads(AV_REFERENCE.read_text(encoding="utf-8"))
    metrics = _paper_metrics(human, automated)
    assert np.allclose(metrics["scenario_bootstrap_ci95"], [1.62100487, 2.39235589])
    assert metrics["draws_above_parity"] == 20_000
    assert metrics["scenarios_av_higher"] == 15


@pytest.mark.skipif(
    not (DEFAULT_SOURCE_SESSION / "vehicle_perception_simulation_trajectory.log").is_file(),
    reason="local Shanghai T11 replay payload unavailable",
)
def test_designated_counterpart_template_covers_all_source_frames() -> None:
    template = _load_designated_counterparts(DEFAULT_SOURCE_SESSION)
    assert len(template) == 4094
    assert set(template.scenario_id) == set(SCENARIOS)
    assert template.designated_counterpart_id.notna().all()
    assert (template.n_frames_total_raw.between(0, 31)).all()
    assert int((template.counterpart_acceleration_mps2 < -3.0).sum()) > 0


@pytest.mark.skipif(
    not (BASE_OUTPUT / "tables/candidate_moments.parquet").is_file(),
    reason="local RQ029 v1 package unavailable",
)
def test_run_calibration_preserves_all_hard_counts() -> None:
    candidates = pd.read_parquet(BASE_OUTPUT / "tables/candidate_moments.parquet")
    targets = json.loads(BASE_TARGETS.read_text(encoding="utf-8"))
    counts, metadata = _calibrate_run_counts(candidates, targets)
    assert len(counts) == 300
    assert int(counts.high_support_run.sum()) == 186
    assert counts.n_candidate.sum() == targets["gates"]["n_candidate_moments"]
    assert counts.n_gate1.sum() == targets["gates"]["n_gate1_pass"]
    assert counts.n_both.sum() == targets["gates"]["n_both_gates"]
    assert counts.n_below_90.sum() == targets["flag_counts"]["90"]["n_below"]
    assert counts.n_above_90.sum() == targets["flag_counts"]["90"]["n_above"]
    observed = counts.assign(flagged=counts.n_below_90 + counts.n_above_90)
    for scenario, group in observed.groupby("scenario_id"):
        target = targets["per_scenario_alpha90"][scenario]
        assert group.n_both.sum() == target["n_both"]
        assert group.flagged.sum() == target["n_flagged"]
    assert np.isfinite(metadata["endpoint_loss"])


@pytest.mark.skipif(
    not (DEFAULT_OUTPUT / "tables/achieved_summary.json").is_file(),
    reason="local RQ029 v2 paper-aligned package unavailable",
)
def test_generated_v2_package_passes_hard_validation() -> None:
    result = validate_dataset(DEFAULT_OUTPUT, write_report=False)
    assert result["validation_status"] == "PASS"
    assert result["checks_hard_failed"] == 0
    assert result["soft_intervals_within_tolerance"] == 4
    assert result["soft_matches"]["alpha90"] is False
    assert result["paper_claim_match_status"] == (
        "POINT_ESTIMATES_EXACT_INTERVALS_PARTIAL"
    )
    assert result["raw_emergent_match_status"] == "NOT_MATCHED_AND_NOT_CLAIMED"
    assert result["full_distribution_match_claimed"] is False


@pytest.mark.skipif(
    not (DEFAULT_OUTPUT / "tables/achieved_summary.json").is_file(),
    reason="tamper test requires the local RQ029 v2 package",
)
def test_v2_validator_rejects_source_and_summary_tampering(tmp_path) -> None:
    tampered = tmp_path / "v2_tampered"
    tampered.mkdir()
    shutil.copy2(DEFAULT_OUTPUT / "manifest.json", tampered / "manifest.json")
    shutil.copy2(DEFAULT_OUTPUT / "README.md", tampered / "README.md")
    shutil.copytree(DEFAULT_OUTPUT / "tables", tampered / "tables")
    (tampered / "validation").mkdir()
    for source in (DEFAULT_OUTPUT / "raw").rglob("*"):
        if not source.is_file():
            continue
        destination = tampered / source.relative_to(DEFAULT_OUTPUT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source)

    template_path = tampered / "tables/designated_counterpart_template.parquet"
    original_template_path = (
        DEFAULT_OUTPUT / "tables/designated_counterpart_template.parquet"
    )
    template = pd.read_parquet(template_path)
    template.loc[0, "designated_counterpart_id"] = "WRONG_COUNTERPART"
    template.loc[0, "counterpart_observed"] = not bool(
        template.loc[0, "counterpart_observed"]
    )
    template.to_parquet(template_path, index=False)
    result = validate_dataset(tampered, write_report=False)
    assert result["validation_status"] == "FAIL"
    assert "source_counterpart_template_recomputed" in result["failed_hard_checks"]

    shutil.copy2(original_template_path, template_path)
    summary_path = tampered / "tables/achieved_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["raw_summary_v1_designated_counterpart"]["ego_ttc"]["lower"][
        "q50"
    ] = 999999.0
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result = validate_dataset(tampered, write_report=False)
    assert result["validation_status"] == "FAIL"
    assert "v1_designated_raw_summary_integrity" in result["failed_hard_checks"]
    assert result["raw_loss_v1"] == pytest.approx(239.51843277905397)

    shutil.copy2(
        DEFAULT_OUTPUT / "tables/achieved_summary.json",
        summary_path,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["paper_claim_match_status"] = "FULL_MATCH"
    summary["paper_claim_metric_layer"] = "all raw and calibrated fields"
    summary["raw_emergent_match_status"] = "MATCHED"
    summary["full_distribution_match_claimed"] = True
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result = validate_dataset(tampered, write_report=False)
    assert result["validation_status"] == "FAIL"
    assert "summary_paper_claim_match_status" in result["failed_hard_checks"]
    assert "summary_raw_emergent_match_not_claimed" in result["failed_hard_checks"]
