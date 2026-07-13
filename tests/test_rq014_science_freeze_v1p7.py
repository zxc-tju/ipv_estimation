from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from scripts.rq014.spearman_average_midranks import (
    average_midranks,
    spearman_average_midranks,
)
from scripts.rq014.wod_ipv_preprocessing import derive_window_kinematics


ROOT = Path(__file__).resolve().parents[1]
PLANS = ROOT / "reports" / "plans"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_core_tree_digest_covers_exact_import_closure() -> None:
    manifest = _load(PLANS / "RQ014_estimator_core_tree_v1p7.json")
    rows = manifest["files"]
    assert [row["path"] for row in rows] == sorted(row["path"] for row in rows)
    assert len(rows) == 7
    for row in rows:
        assert _sha(ROOT / row["path"]) == row["sha256"]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert hashlib.sha256(payload).hexdigest() == manifest["core_tree_sha256"]
    assert manifest["core_tree_sha256"] == "ffc83befe2f0e45cccd236965166bb14b71a3f258a49897fef49c0468946fb5e"


def test_registry_science_hashes_match_managed_files() -> None:
    valid = _load(PLANS / "RQ014_config_space_v1p6.yaml")
    fixed = valid["fixed_estimator"]
    assert fixed["wod_adapter_sha256"] == _sha(ROOT / "scripts/rq014/wod_ipv_adapter.py")
    assert fixed["preprocessing_contract_sha256"] == _sha(ROOT / "scripts/rq014/wod_ipv_preprocessing.py")
    assert fixed["reference_builder_sha256"] == _sha(ROOT / "scripts/rq014/wod_reference_builder.py")
    version_path = ROOT / "scripts/rq014/spearman_version_manifest_v1.json"
    spearman = valid["statistical_implementation_bindings"]["spearman"]
    assert spearman["implementation_and_version_sha256"] == _sha(version_path)
    version = _load(version_path)
    assert set(version) == {"package", "version", "function", "options"}
    assert version["options"]["implementation_sha256"] == _sha(
        ROOT / version["options"]["implementation_path"]
    )
    mapping = valid["envelope"]["wod_path_type_mapping"]
    for path_key, hash_key in (
        ("source_definition_path", "source_definition_sha256"),
        ("implementation_path", "implementation_sha256"),
        ("version_manifest_path", "version_manifest_sha256"),
        ("golden_fixture_path", "golden_fixture_sha256"),
        ("mapping_table_candidate_path", "mapping_table_sha256"),
        ("mapping_manifest_candidate_path", "mapping_manifest_sha256"),
        ("distribution_summary_path", "distribution_summary_sha256"),
    ):
        assert mapping[hash_key] == _sha(ROOT / mapping[path_key])
    assert (ROOT / mapping["implementation_path"]).stat().st_size == mapping[
        "implementation_size_bytes"
    ]
    table = ROOT / mapping["mapping_table_candidate_path"]
    assert table.stat().st_size == mapping["mapping_table_size_bytes"]
    assert len(table.read_text(encoding="utf-8").splitlines()) - 1 == mapping[
        "mapping_table_row_count"
    ]
    mapping_manifest = _load(ROOT / mapping["mapping_manifest_candidate_path"])
    assert mapping_manifest["mapping"]["sha256"] == mapping["mapping_table_sha256"]
    assert mapping_manifest["row_count"] == mapping["mapping_table_row_count"]
    assert mapping_manifest["contains_rating"] is False
    implementation_version = _load(ROOT / mapping["version_manifest_path"])
    assert implementation_version["options"]["implementation_sha256"] == mapping[
        "implementation_sha256"
    ]
    envelope_contract = _load(PLANS / "RQ014_envelope_builder_contract_v2.json")
    builder = envelope_contract["path_type_contract"]["wod_mapping"]["builder_binding"]
    assert builder["source_definition"]["sha256"] == mapping["source_definition_sha256"]
    assert builder["implementation"]["sha256"] == mapping["implementation_sha256"]
    assert builder["mapping_table"]["sha256"] == mapping["mapping_table_sha256"]


def test_m3_manifest_binds_the_frozen_artifact_set() -> None:
    registry = _load(PLANS / "RQ014_config_space_v1p6.yaml")
    frozen = registry["envelope"]["frozen_m3"]
    manifest = _load(ROOT / frozen["manifest_path"])
    assert _sha(ROOT / frozen["manifest_path"]) == frozen["manifest_sha256"]
    assert _sha(ROOT / frozen["feature_spec_contract_path"]) == frozen["feature_spec_contract_sha256"]
    assert manifest["artifact"]["sha256"] == frozen["scorer_sha256"]
    assert manifest["artifact"]["size_bytes"] == frozen["scorer_size_bytes"]
    assert manifest["feature_contract"]["sha256"] == frozen["feature_spec_contract_sha256"]


def test_exact_window_preprocessing_matches_registered_operator() -> None:
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
    result = derive_window_kinematics(xy, 1.0)
    np.testing.assert_array_equal(result["velocity"], [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    np.testing.assert_array_equal(result["acceleration"], [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_array_equal(result["heading"], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="all-stationary"):
        derive_window_kinematics(np.zeros((3, 2)), 0.25)


def test_managed_spearman_matches_scipy_with_average_ties() -> None:
    x = [1.0, 1.0, 2.0, 4.0, 4.0, 7.0]
    y = [8.0, 6.0, 6.0, 3.0, 2.0, 1.0]
    assert average_midranks(x) == [1.5, 1.5, 3.0, 4.5, 4.5, 6.0]
    assert spearman_average_midranks(x, y) == pytest.approx(stats.spearmanr(x, y).statistic)


@pytest.mark.parametrize(
    ("x", "y", "message"),
    [
        ([1.0, 2.0], [2.0, 1.0], "at least three rows"),
        ([1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], "three distinct pairs"),
        ([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], "nonconstant"),
        ([1.0, 2.0, math.nan], [1.0, 2.0, 3.0], "finite"),
    ],
)
def test_managed_spearman_rejects_unregistered_inputs(x: list[float], y: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        spearman_average_midranks(x, y)


def test_x02_sites_are_legacy_inactive_and_not_active_bindings() -> None:
    valid = _load(PLANS / "RQ014_config_space_v1p6.yaml")
    extension = _load(PLANS / "RQ014_recovery_extension_registry_v1p6.yaml")
    contract = _load(PLANS / "RQ014_execution_contract_v1p5.json")
    x02_valid = valid["statistical_contract_v1p3"]["X02_scale_eligibility"]
    x02_extension = extension["cells"][1]
    assert x02_valid["binding_status"] == "LEGACY_INACTIVE_UNBOUND"
    assert x02_extension["binding_status"] == "LEGACY_INACTIVE_UNBOUND"
    assert all("X02" not in item for item in contract["registry_binding_contract"]["required_binding_ids"])
    assert contract["registry_binding_contract"]["required_binding_count"] == 12


def test_v1p6_registries_forward_bind_authority_and_revert_non_m3_state_change() -> None:
    valid = _load(PLANS / "RQ014_config_space_v1p6.yaml")
    extension = _load(PLANS / "RQ014_recovery_extension_registry_v1p6.yaml")
    historical = _load(PLANS / "RQ014_config_space_v1p5.yaml")
    formal = (
        "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
        "RQ014_formal_G1_v1p6_preflight_20260713.yaml"
    )
    decision = "reports/plans/RQ014_PI_decision_D1_preflight_v1p6_20260713.md"
    for registry in (valid, extension):
        authority = registry["execution_authorization_contract"]
        assert authority["formal_g1_ref"] == formal
        assert authority["scoped_decision_ref"] == decision
    assert valid["sequence_contract"]["state_derivation"] == historical["sequence_contract"][
        "state_derivation"
    ]


def test_round6_amendment_closes_runtime_quantile_and_byte_change_rulings() -> None:
    text = (PLANS / "RQ014_plan_v1p7_amendment_20260713.md").read_text(encoding="utf-8")
    for required in (
        "managed-environment closure v4",
        "joblib",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "preflight never deserializes or scores M3",
        "pre-OOD-mask",
        "scripts/rq014/preflight.py",
        "scripts/rq014/materialize_registry.py",
        "/sequence_contract/state_derivation",
        "/envelope/source",
        "/envelope/form",
        "/envelope/path_types",
        "/envelope/quantiles",
        "/envelope/matched_fields",
        "/envelope/human_episode_weighting",
        "/envelope/builder_contract",
        "/envelope/envelope_gate",
        "/envelope/support_semantics",
        "/envelope/wod_transfer_semantics",
        "/envelope/frozen_m3",
    ):
        assert required in text
