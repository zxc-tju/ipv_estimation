from __future__ import annotations

import pytest

from pipelines.simulation.package_rq029_layered_release import (
    DATA_STATUS,
    DEFAULT_OUTPUT,
    EVIDENCE_ROWS,
    LAYER_1_TABLES,
    LAYER_2_TABLES,
    REPO_ROOT,
    SCHEMA_VERSION,
    SOURCE_ROOT,
    _prepare_output,
    validate_release,
)


def test_layer_contract_separates_raw_and_analysis_tables() -> None:
    assert "trajectory_pairs.parquet" in LAYER_1_TABLES
    assert "candidate_moments.parquet" not in LAYER_1_TABLES
    assert "counterpart_windows.parquet" not in LAYER_1_TABLES
    assert "candidate_moments.parquet" in LAYER_2_TABLES
    assert "counterpart_windows.parquet" in LAYER_2_TABLES


def test_synthetic_artifacts_are_never_direct_paper_support() -> None:
    assert all(
        not (row["synthetic"] and row["authority"] == "DIRECT_SUPPORT")
        for row in EVIDENCE_ROWS
    )
    assert any(
        not row["synthetic"] and row["authority"] == "DIRECT_SUPPORT"
        for row in EVIDENCE_ROWS
    )
    target_contract = next(
        row for row in EVIDENCE_ROWS if "target contract" in row["claim_family"]
    )
    assert target_contract["authority"] == "CALIBRATION_CONTRACT_NOT_EVIDENCE"


def test_replace_rejects_noncanonical_release_root(tmp_path) -> None:
    output = tmp_path / "lookalike_release"
    output.mkdir()
    (output / "release_manifest.json").write_text(
        __import__("json").dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "data_status": DATA_STATUS,
                "source_path": str(SOURCE_ROOT.relative_to(REPO_ROOT)),
                "declared_layers": 2,
                "third_layer_status": "NOT_SPECIFIED_BY_USER",
            }
        ),
        encoding="utf-8",
    )
    marker = output / "must_survive.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        _prepare_output(output, replace=True)
    assert marker.read_text(encoding="utf-8") == "preserve"


@pytest.mark.skipif(
    not (DEFAULT_OUTPUT / "release_manifest.json").is_file(),
    reason="local layered RQ029 release unavailable",
)
def test_current_layered_release_passes_light_validation() -> None:
    result = validate_release(DEFAULT_OUTPUT, deep_hash=False, write_report=False)
    assert result["validation_status"] == "PASS"
    assert result["failed_checks"] == []
    assert result["n_raw_files"] == 100
    assert result["n_runs"] == 300
    assert result["n_frames"] == 81_880
