from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.rq014.materialize_registry import (
    ContractError,
    canonical_bytes,
    load_json,
    materialize,
    sha256_file,
)
from scripts.rq014.preflight import (
    ContractError as PreflightContractError,
    canonical_json_bytes,
    validate_anchor_receipt,
    validate_input_manifest_g2,
    validate_materialization_ledger,
    validate_wod_mapping_registry_binding,
    validate_wod_path_type_mapping_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
PLANS = ROOT / "reports" / "plans"
VALID = PLANS / "RQ014_config_space_v1p6.yaml"
FORENSIC = PLANS / "RQ014_forensic_registry_v1p5.yaml"
EXTENSION = PLANS / "RQ014_recovery_extension_registry_v1p6.yaml"
EXECUTION = PLANS / "RQ014_execution_contract_v1p5.json"
RECOVERY = PLANS / "RQ014_recovery_lane_v3.json"
RECOVERY_V2 = PLANS / "RQ014_recovery_lane_v2.json"


def _walk_values(value: object):
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item)


def _walk_keys(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def _pointer(document: object, pointer: str) -> object:
    current = document
    for part in pointer.lstrip("/").split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current


def test_v1p3_history_remains_byte_replayable() -> None:
    manifest = PLANS / "RQ014_plan_v1p3_checksums_20260711.sha256"
    rows = manifest.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 14
    for row in rows:
        expected, relative = row.split("  ", 1)
        assert sha256_file(ROOT / relative) == expected


def test_v1p5_g0_waivers_use_only_legal_terminal_states() -> None:
    forensic = load_json(FORENSIC)
    legal = set(forensic["terminal_states"])
    surfaces = {row["surface_id"]: row for row in forensic["forensic_surfaces"]}
    assert set(surfaces) == {f"F{index:02d}" for index in range(1, 11)}
    assert all(row["status"] in legal for row in surfaces.values())
    assert all(row["status"] != "OPEN" for row in surfaces.values())
    assert "INACCESSIBLE_PI_WAIVED" not in json.dumps(forensic)

    waived = {"F05", "F06", "F07", "F08", "F10"}
    for surface_id in waived:
        row = surfaces[surface_id]
        assert row["status"] == "INACCESSIBLE"
        assert row["complete_scan"] is False
        assert row["closure_basis"] == "PI_WAIVER"
        assert row["negative_finding_claim_allowed"] is False
        assert row["residual_risk_statement_required"] is True
        assert "closure_evidence" not in row
        decision = ROOT / row["waiver"]["decision_ref"]
        assert sha256_file(decision) == row["waiver"]["decision_sha256"]
    assert surfaces["F09"]["closure_basis"] == "LEGACY_STORAGE_UNAVAILABLE"
    assert "waiver" not in surfaces["F09"]
    for surface_id in ("F07", "F08"):
        assert "NO_PRE_CUTOFF_WHOLE_INVENTORY_RECEIPT" in surfaces[surface_id]["reason_codes"]

    closure = forensic["g0_closure"]
    counts = Counter(row["status"] for row in surfaces.values())
    assert closure["status"] == "CLOSED_WITH_INACCESSIBLE_SURFACES"
    assert closure["open_surface_count"] == 0
    assert closure["pi_waived_surface_ids"] == ["F05", "F06", "F07", "F08", "F10"]
    assert closure["terminal_status_counts"] == {
        "FOUND": counts["FOUND"],
        "NOT_FOUND_ON_SCANNED_SURFACES": counts["NOT_FOUND_ON_SCANNED_SURFACES"],
        "INACCESSIBLE": counts["INACCESSIBLE"],
    }
    requirements = forensic["freeze_requirements"]
    assert "post_g1_forensic_compute_preconditions" in requirements
    assert set(requirements["waived_outputs_not_required"]) == {
        "fl05_index_outputs",
        "F05_F08_pass4_surface_artifacts",
    }


def test_v1p5_has_fail_closed_operation_authority() -> None:
    registries = [load_json(path) for path in (VALID, FORENSIC, EXTENSION)]
    for registry in registries:
        assert "execution_authorized" not in set(_walk_keys(registry))
        authority = registry["execution_authorization_contract"]
        assert authority["default_decision"] == "DENY"
        assert authority["authority_path"] == "configs/research_authorization.json"

    execution = load_json(EXECUTION)
    operations = execution["authorization"]["registered_operations"]
    assert operations["rq014_g2_declassification_export"]["status"] == (
        "CONDITIONALLY_AUTHORIZED_AFTER_FORMAL_G1"
    )
    assert operations["rq014_g2_declassification_export"]["rating_access"] == "FORBIDDEN"
    assert operations["rq014_g2_contract_preflight"]["status"] == (
        "CONDITIONALLY_AUTHORIZED_AFTER_FORMAL_G1"
    )
    assert operations["rq014_g2_contract_preflight"]["required_prior_receipts"] == [
        "rq014-g2-declassification-export-receipt-v1",
        "rq014-managed-operation-done-v1",
    ]
    authorization = load_json(ROOT / "configs" / "research_authorization.json")
    assert authorization["authorizations"]["RQ014"]["allowed_operations"] == [
        "rq014_g2_declassification_export",
        "rq014_g2_contract_preflight",
    ]
    assert authorization["authorizations"]["RQ014"]["preflight_decision_path"] == (
        "reports/plans/RQ014_PI_decision_D1_preflight_v1p6_20260713.md"
    )
    assert operations["rq014_g2_resource_pilot"]["status"].startswith("DENY_")
    assert operations["rq014_g2_blind_build"]["status"].startswith("DENY_")
    assert operations["rq014_g2p_power_simulation"]["status"].startswith("DENY_")

    central = load_json(ROOT / "configs" / "research_authorization.json")
    assert central["authorizations"]["RQ014"]["allowed_operations"] == [
        "rq014_g2_declassification_export",
        "rq014_g2_contract_preflight",
    ]


def test_v1p5_score_stripped_and_staged_manifest_contract_is_rating_blind() -> None:
    execution = load_json(EXECUTION)
    blind = execution["score_stripped_input_contract"]
    assert blind["raw_tfrecord_classification"].startswith("contains embedded preference_score")
    assert set(blind["forbidden_g2_formats"]) == {"TFRecord", "protobuf", "pickle"}
    assert "candidate_states.csv" in blind["required_bundle_files"]
    g2 = execution["staged_input_manifests"]["G2"]
    assert g2["rating_access"] == "FORBIDDEN"
    assert "ratings_manifest" in g2["forbidden_roles"]
    assert g2["split_membership"] == "NOT_YET_FROZEN"
    assert execution["staged_input_manifests"]["G3"]["availability"] == (
        "LEGACY_OPTIONAL_UNREGISTERED_AND_UNAUTHORIZED_IN_V1P5"
    )

    schema = load_json(PLANS / "RQ014_score_stripped_schema_v1.json")
    assert schema["tfrecord_inputs_visible_to_g2"] is False
    assert schema["pickle_inputs_visible_to_g2"] is False
    candidate_columns = schema["files"]["candidate_states.csv"]["columns"]
    assert "raw_sample_index" in candidate_columns
    assert "dropped_as_tstar_duplicate" in candidate_columns
    assert "effective_time_s" in candidate_columns
    assert "preference_score" not in candidate_columns

    anchor = load_json(PLANS / "RQ014_blind_anchor_receipt_v1p5.json")
    assert anchor["g2_rho_recomputation"] == "FORBIDDEN"
    assert [(row["anchor_id"], row["public_scene_n"]) for row in anchor["anchors"]] == [
        ("A1", 75),
        ("A2", 98),
        ("A3", 75),
        ("A4", 47),
    ]


def test_recovery_lane_searches_true_history_future_and_combined_windows_without_power_gate() -> None:
    recovery = load_json(RECOVERY)
    assert recovery["schema_version"] == "rq014-historical-recovery-lane-v3"
    assert recovery["known_target"]["remembered_direction"] == "negative"
    assert recovery["known_target"]["null_hypothesis_discovery"] is False
    assert recovery["claim_boundary"]["p_values_gate_recovery"] is False
    assert recovery["claim_boundary"]["prospective_power_gates_recovery"] is False
    feature_bank = recovery["rating_blind_feature_bank"]
    recipes = {row["temporal_id"]: row for row in feature_bank["temporal_axis"]["recipes"]}
    assert recipes["CH-W10"]["interval"] == "[tau-1.0,tau]"
    assert recipes["LF-W10"]["interval"] == "[tau,tau+1.0]"
    assert recipes["HF-W10"]["interval"] == "[tau-1.0,tau+1.0] with tau included once"
    assert {"TP", "TF"} <= set(recipes)
    assert feature_bank["feature_family_enumeration"]["registered_family_count"] == 16
    assert feature_bank["predictor_cell_enumeration"]["registered_predictor_cell_count"] == 320
    assert "envelope_axis" not in feature_bank
    screen = recovery["full_data_recovery_screen"]
    assert screen["split"].startswith("none")
    assert screen["registered_leaderboard_row_count"] == 960
    assert screen["ranking"]["wait_for_all_rows_terminal"] is True
    assert screen["ranking"]["top_recipe_count"] == 1
    assert recovery["optional_prospective_validation"]["gating_for_recovery"] is False

    execution = load_json(EXECUTION)
    registry = load_json(VALID)
    recovery_path = str(RECOVERY.relative_to(ROOT))
    recovery_schema = recovery["schema_version"]
    assert (
        execution["primary_scientific_authority"]["path"]
        == registry["primary_recovery_contract"]["path"]
        == recovery_path
    )
    assert (
        execution["primary_scientific_authority"]["schema_version"]
        == registry["primary_recovery_contract"]["schema_version"]
        == recovery_schema
    )
    operations = execution["authorization"]["registered_operations"]
    assert operations["rq014_r2_blind_feature_build"]["status"].startswith("DENY_")
    assert operations["rq014_r3_full_rating_join_and_rank"]["status"].startswith("DENY_")
    assert operations["rq014_r4_clean_replay"]["status"].startswith("DENY_")
    assert operations["rq014_r2_blind_feature_build"]["scientific_contracts"] == [recovery_path]
    assert operations["rq014_r3_full_rating_join_and_rank"]["scientific_contract"] == recovery_path
    assert operations["rq014_r4_clean_replay"]["scientific_contract"] == recovery_path
    g2 = execution["staged_input_manifests"]["G2"]
    assert g2["required_roles"] == [
        "wod_score_stripped_bundle_manifest",
        "wod_score_stripped_sanitization_receipt",
        "wod_path_type_mapping_manifest",
        "blind_anchor_receipt",
    ]
    staged = execution["staged_input_manifests"]["recovery_lane_v3"]
    assert staged["scientific_contract"] == recovery_path
    assert staged["predictor_cell_count"] == 320
    assert staged["terminal_leaderboard_row_count"] == 960
    assert "InterHub" not in json.dumps({"g2": g2, "staged": staged})


def test_v1p7_registry_binding_targets_cover_every_prefilled_value() -> None:
    execution = load_json(EXECUTION)
    policy = execution["registry_binding_contract"]
    registries = {
        "valid_scientific": load_json(VALID),
        "forensic": load_json(FORENSIC),
        "recovery_extension": load_json(EXTENSION),
    }
    assert policy["required_binding_count"] == 12
    assert len(policy["required_binding_ids"]) == 12
    assert set(policy["binding_targets"]) == set(policy["required_binding_ids"])
    assert policy["source_binding_mode"] == "VERIFY_PREFILLED_EXACT"
    assert sum(
        value == "TO_FREEZE_AT_G2"
        for registry in registries.values()
        for value in _walk_values(registry)
    ) == 0
    bindings = _valid_bindings()
    for binding_id, target in policy["binding_targets"].items():
        assert _pointer(registries[target["registry"]], target["pointer"]) == bindings[binding_id]

    x02 = registries["recovery_extension"]["cells"][1]
    assert x02["extension_id"] == "X02"
    assert x02["binding_status"] == "LEGACY_INACTIVE_UNBOUND"
    gate = x02["source_definition_gate"]
    assert gate["source_definition_sha256"] == "LEGACY_INACTIVE_UNBOUND"
    assert gate["wod_mapping_sha256"] == "LEGACY_INACTIVE_UNBOUND"
    assert gate["artifact_set_sha256"] == "LEGACY_INACTIVE_UNBOUND"
    assert all("X02" not in binding_id for binding_id in policy["required_binding_ids"])


def _valid_bindings() -> dict[str, str]:
    execution = load_json(EXECUTION)
    registries = {
        "valid_scientific": load_json(VALID),
        "forensic": load_json(FORENSIC),
        "recovery_extension": load_json(EXTENSION),
    }
    policy = execution["registry_binding_contract"]
    return {
        binding_id: _pointer(registries[target["registry"]], target["pointer"])
        for binding_id, target in policy["binding_targets"].items()
    }


def test_registry_materialization_is_deterministic_and_never_mutates_source(tmp_path: Path) -> None:
    output = tmp_path / "materialized"
    output.mkdir()
    freeze = output / "registry_bindings.g2.json"
    freeze.write_bytes(
        canonical_bytes(
            {
                "schema_version": "rq014-registry-bindings-g2-v1",
                "stage": "G2",
                "bindings": _valid_bindings(),
            }
        )
    )
    before = {path: sha256_file(path) for path in (VALID, FORENSIC, EXTENSION)}
    first = materialize(
        repo_root=ROOT,
        contract_path=EXECUTION,
        freeze_values_path=freeze,
        output_dir=output,
    )
    second = materialize(
        repo_root=ROOT,
        contract_path=EXECUTION,
        freeze_values_path=freeze,
        output_dir=output,
    )
    assert first["ledger_sha256"] == second["ledger_sha256"]
    assert before == {path: sha256_file(path) for path in before}
    for item in first["outputs"].values():
        payload = (output / item["path"]).read_text(encoding="utf-8")
        assert "TO_FREEZE_AT_G2" not in payload
        assert sha256_file(output / item["path"]) == item["sha256"]
    validated = validate_materialization_ledger(
        ledger_path=output / "materialization_ledger.json",
        repo_root=ROOT,
        contract=load_json(EXECUTION),
    )
    assert set(validated["outputs"]) == {"valid_scientific", "forensic", "recovery_extension"}


def test_registry_materialization_validator_rejects_empty_source_and_output_sets(
    tmp_path: Path,
) -> None:
    output = tmp_path / "materialized"
    output.mkdir()
    bindings = _valid_bindings()
    freeze = output / "registry_bindings.g2.json"
    freeze.write_bytes(
        canonical_bytes(
            {
                "schema_version": "rq014-registry-bindings-g2-v1",
                "stage": "G2",
                "bindings": bindings,
            }
        )
    )
    ledger = {
        "schema_version": "rq014-registry-materialization-ledger-g2-v1",
        "stage": "G2",
        "execution_contract": {
            "path": str(EXECUTION.relative_to(ROOT)),
            "sha256": sha256_file(EXECUTION),
        },
        "freeze_values": {"path": freeze.name, "sha256": sha256_file(freeze)},
        "materializer_sha256": sha256_file(ROOT / "scripts" / "rq014" / "materialize_registry.py"),
        "source_registries": {},
        "bindings": bindings,
        "outputs": {},
    }
    ledger_path = output / "materialization_ledger.json"
    ledger_path.write_bytes(canonical_json_bytes(ledger))
    with pytest.raises(PreflightContractError, match="source registries are incomplete"):
        validate_materialization_ledger(
            ledger_path=ledger_path,
            repo_root=ROOT,
            contract=load_json(EXECUTION),
        )


def test_blind_anchor_rejects_extra_rating_fields(tmp_path: Path) -> None:
    source = load_json(PLANS / "RQ014_blind_anchor_receipt_v1p5.json")
    source["hidden_human_rating_rows"] = [5, 4, 1]
    path = tmp_path / "anchor.json"
    path.write_bytes(canonical_json_bytes(source))
    with pytest.raises(PreflightContractError, match="keys differ"):
        validate_anchor_receipt(path)


def test_g2_input_manifest_rejects_binary_roles_and_path_aliases(tmp_path: Path) -> None:
    root = tmp_path / "inputs"
    root.mkdir()
    roles = [
        "wod_score_stripped_bundle_manifest",
        "wod_score_stripped_sanitization_receipt",
        "wod_path_type_mapping_manifest",
        "blind_anchor_receipt",
    ]
    entries = []
    for index, role in enumerate(roles):
        suffix = ".pkl" if index == 0 else ".json"
        path = root / f"input_{index}{suffix}"
        path.write_bytes(b"{}\n")
        entries.append(
            {
                "input_id": f"input_{index}",
                "role": role,
                "absolute_path": str(path),
                "sha256": sha256_file(path),
                "contains_rating": False,
            }
        )
    manifest = tmp_path / "input_manifest.g2.json"
    manifest.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "rq014-input-manifest-g2-v1",
                "stage": "G2",
                "parent_manifest_sha256": None,
                "entries": entries,
            }
        )
    )
    with pytest.raises(PreflightContractError, match="must resolve to a JSON"):
        validate_input_manifest_g2(
            manifest_path=manifest,
            contract=load_json(EXECUTION),
            allowed_roots=[root],
        )

    json_path = root / "input_1.json"
    entries[0].update(
        absolute_path=str(json_path),
        sha256=sha256_file(json_path),
    )
    manifest.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "rq014-input-manifest-g2-v1",
                "stage": "G2",
                "parent_manifest_sha256": None,
                "entries": entries,
            }
        )
    )
    with pytest.raises(PreflightContractError, match="Aliased G2 input path"):
        validate_input_manifest_g2(
            manifest_path=manifest,
            contract=load_json(EXECUTION),
            allowed_roots=[root],
        )


def test_wod_path_type_mapping_manifest_is_checksum_bound_and_canonical(
    tmp_path: Path,
) -> None:
    root = tmp_path / "inputs" / "RQ014" / "wod_path_type_mapping" / "v1"
    root.mkdir(parents=True)
    mapping = root / "wod_path_type_mapping.csv"
    mapping.write_text(
        "segment_id,tstar_context_step,path_type\nseg-1,4,CP\nseg-1,8,HO\n",
        encoding="utf-8",
    )
    manifest = root / "manifest.json"
    manifest.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "rq014-wod-path-type-mapping-manifest-v1",
                "contains_rating": False,
                "mapping": {
                    "path": str(mapping),
                    "size_bytes": mapping.stat().st_size,
                    "sha256": sha256_file(mapping),
                    "format": "RFC4180_CSV",
                },
                "row_count": 2,
                "key_columns": ["segment_id", "tstar_context_step"],
                "value_column": "path_type",
                "allowed_values": ["CP", "HO", "MP", "F"],
            }
        )
    )
    validated = validate_wod_path_type_mapping_manifest(manifest, mapping_root=root)
    assert validated["mapping_sha256"] == sha256_file(mapping)
    assert validated["row_count"] == 2
    binding_id = "valid.envelope.wod_path_type_mapping.mapping_table_sha256"
    validate_wod_mapping_registry_binding(
        validated,
        {"bindings": {binding_id: validated["mapping_sha256"]}},
    )
    with pytest.raises(PreflightContractError, match="differs from reviewed registry binding"):
        validate_wod_mapping_registry_binding(
            validated,
            {"bindings": {binding_id: "f" * 64}},
        )

    mapping.write_text(
        "segment_id,tstar_context_step,path_type\nseg-1,4,CP\nseg-1,4,HO\n",
        encoding="utf-8",
    )
    payload = load_json(manifest)
    payload["mapping"]["size_bytes"] = mapping.stat().st_size
    payload["mapping"]["sha256"] = sha256_file(mapping)
    manifest.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(PreflightContractError, match="Duplicate WOD path-type mapping"):
        validate_wod_path_type_mapping_manifest(manifest, mapping_root=root)


def test_registry_materialization_rejects_binding_that_differs_from_reviewed_source(
    tmp_path: Path,
) -> None:
    bindings = _valid_bindings()
    bindings["valid.fixed_estimator.core_tree_sha"] = "f" * 64
    output = tmp_path / "out"
    output.mkdir()
    freeze = output / "registry_bindings.g2.json"
    freeze.write_bytes(
        canonical_bytes(
            {
                "schema_version": "rq014-registry-bindings-g2-v1",
                "stage": "G2",
                "bindings": bindings,
            }
        )
    )
    with pytest.raises(ContractError, match="Prefilled source binding mismatch"):
        materialize(
            repo_root=ROOT,
            contract_path=EXECUTION,
            freeze_values_path=freeze,
            output_dir=output,
        )


def test_recovery_lane_v2_bytes_remain_frozen_history() -> None:
    assert sha256_file(RECOVERY_V2) == (
        "c1d3a8c4faeb04871e15d7d1d0f07edfd45b8e6904bdd5ac7e05fa3f1f412d7d"
    )
