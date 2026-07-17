from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.rq014 import build_g2r_blind_outputs as W4
from scripts.rq014 import build_wod_m3_anchors as W2
from scripts.rq014 import build_wod_scene_anchor_domain as DOMAIN


ROOT = Path(__file__).resolve().parents[1]
W1 = ROOT / "tests" / "fixtures" / "rq014_g2r_v1"
W2B = ROOT / "tests" / "fixtures" / "rq014_g2r_w2b"
W4_FIXTURES = ROOT / "tests" / "fixtures" / "rq014_g2r_w4"
SCORER = ROOT / "models" / "rq009_m3" / "m3_scorer.joblib"


def _strict_load(path: Path):
    def reject_constant(token: str) -> None:
        raise ValueError(token)

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(key)
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


def _real_scene() -> W4.SceneBlindInput:
    candidate = {
        tick: (float(tick), 0.02 * float(tick) ** 2) for tick in range(-9, 9)
    }
    counterpart = {
        tick: (0.8 * float(tick) + 1.0, 2.0 - 0.05 * float(tick))
        for tick in range(-9, 9)
    }
    return W4.SceneBlindInput(
        domain=DOMAIN.SceneDomainInput(
            "scene-real",
            "CP",
            {
                "R04N": DOMAIN.SamplingTimelines(
                    candidates={candidate_id: candidate for _, candidate_id in W4.CANDIDATES},
                    counterpart=counterpart,
                )
            },
        ),
        route_intent="straight",
    )


def _structural_scene() -> W4.SceneBlindInput:
    return W4.SceneBlindInput(
        domain=DOMAIN.SceneDomainInput(
            "scene-structural",
            "NA",
            {},
            terminal_status="INELIGIBLE_BLIND",
        ),
        route_intent="straight",
    )


def _fixture_inputs():
    scenes = (_structural_scene(), _real_scene())
    domain_rows = DOMAIN.build_anchor_domain_rows([scene.domain for scene in scenes])
    lineage = {
        key: {
            "path": f"fixture/{key}",
            "size_bytes": len(key),
            "sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
        }
        for key in W4.LINEAGE_KEYS
    }
    prerequisite_artifacts = {
        "wod_scene_anchor_domain": W4.ArtifactReference(
            "wod_scene_anchor_domain.csv",
            "rq014-wod-scene-anchor-domain-v1",
            len(DOMAIN.encode_anchor_domain_csv(domain_rows)),
            hashlib.sha256(DOMAIN.encode_anchor_domain_csv(domain_rows)).hexdigest(),
            len(domain_rows),
        ),
        "wod_scene_anchor_domain_manifest": W4.ArtifactReference(
            "wod_scene_anchor_domain_manifest.json",
            "rq014-wod-scene-anchor-domain-manifest-v1",
            1,
            "1" * 64,
            1,
        ),
        "nc_pretstar_history_only_receipt": W4.ArtifactReference(
            "nc_pretstar_history_only_receipt.json",
            "rq014-nc-pretstar-history-only-receipt-v1",
            1,
            "2" * 64,
            1,
        ),
    }
    return scenes, domain_rows, lineage, prerequisite_artifacts


def _build_fixture_in_process() -> W4.BlindOutputBuild:
    scenes, domain_rows, lineage, refs = _fixture_inputs()
    return W4.build_blind_output_artifacts(
        scenes,
        domain_rows,
        run_id="RQ014_G2R_W4_FIXTURE",
        git_commit="1" * 40,
        created_at_utc="2026-07-17T00:00:00Z",
        lineage=lineage,
        prerequisite_artifacts=refs,
        scorer_path=SCORER,
        fixture_mode=True,
    )


def _build_fixture(tmp_path: Path) -> W4.BlindOutputBuild:
    """Run the real scorer in an isolated process, then return its exact bytes.

    Scikit-learn initializes an OpenMP runtime when the pinned scorer predicts.
    Keeping that runtime in the pytest process can poison the later resource-
    pilot fork pool on macOS.  Process isolation preserves the real W2-to-W3
    integration while ensuring that the scorer runtime exits with its child.
    """
    payload_path = tmp_path / "mini_build.pickle"
    test_path = Path(__file__).resolve()
    command = (
        "import pickle, runpy\n"
        f"namespace = runpy.run_path({str(test_path)!r})\n"
        "build = namespace['_build_fixture_in_process']()\n"
        f"with open({str(payload_path)!r}, 'wb') as handle:\n"
        "    pickle.dump(build, handle, protocol=pickle.HIGHEST_PROTOCOL)\n"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(ROOT), str(ROOT / "src"), environment.get("PYTHONPATH"))
        if value
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return pickle.loads(payload_path.read_bytes())


@pytest.fixture(scope="module")
def mini_build(tmp_path_factory: pytest.TempPathFactory) -> W4.BlindOutputBuild:
    return _build_fixture(tmp_path_factory.mktemp("rq014-g2r-w4-mini-build"))


def test_w4_reproduces_w1_order_predicate_schema_shapes_and_anchor_golden() -> None:
    order = _strict_load(W1 / "canonical_cell_order_golden.json")
    cells = W4.canonical_cells()
    payload = b"".join((cell.cell_id + "\n").encode("utf-8") for cell in cells)
    assert len(cells) == order["cell_count"] == 320
    assert len(payload) == order["canonical_cell_id_payload_size_bytes"] == 9184
    assert hashlib.sha256(payload).hexdigest() == order["canonical_cell_ids_sha256"]
    assert cells[0].cell_id == order["first_cell_id"]
    assert cells[-1].cell_id == order["last_cell_id"]

    predicate = _strict_load(W1 / "blind_scene_predicate_golden.json")
    for example in predicate["examples"]:
        masks = []
        for cell_index in range(predicate["cell_count"]):
            values = example["overrides_by_cell_index"].get(
                str(cell_index), example["default_cell"]
            )
            available = values["available"]
            predictors = values["predictor_values_or_NA"]
            finite = all(isinstance(value, (int, float)) for value in predictors)
            nonconstant = finite and len({0.0 if value == 0.0 else value for value in predictors}) > 1
            eligible = all(available) and finite and nonconstant
            masks.append(
                {
                    "cell_index": cell_index,
                    "segment_id": example["fixture_id"],
                    "all_three_available": all(available),
                    "all_three_deviations_finite": all(available) and finite,
                    "deviation_vector_nonconstant": nonconstant,
                    "blind_cell_scene_eligible": eligible,
                }
            )
        assert W4.scene_passes_blind_predicate(
            masks, example["fixture_id"]
        ) is example["expected"]

    shapes = _strict_load(W1 / "schema_shape_goldens.json")["schemas"]
    assert set(shapes["anchor_score_row"]["exact_keys"]) == W4.ANCHOR_SCORE_KEYS
    assert set(shapes["blind_feature_row"]["exact_keys"]) == W4.FEATURE_ROW_KEYS
    assert set(shapes["availability_mask_row"]["exact_keys"]) == W4.MASK_ROW_KEYS
    assert set(shapes["predictor_manifest_row"]["exact_keys"]) == W4.PREDICTOR_ROW_KEYS
    binding = _strict_load(W2B / "anchor_domain_15328_golden.binding.json")
    anchor_bytes = (W2B / "anchor_domain_15328_golden.csv").read_bytes()
    assert len(anchor_bytes) == binding["artifact_size_bytes"] == 1_272_338
    assert hashlib.sha256(anchor_bytes).hexdigest() == binding["artifact_sha256"]
    assert binding["group_count"] == 15_328


def test_w4_mini_universe_emits_all_six_canonical_artifacts_and_bound_slices(
    mini_build: W4.BlindOutputBuild,
) -> None:
    assert tuple(mini_build.artifacts) == W4.DIRECT_ARTIFACT_NAMES
    assert mini_build.common_support_ids == ()
    assert not W4.scene_passes_blind_predicate(mini_build.mask_rows, "scene-real")
    assert not W4.scene_passes_blind_predicate(
        mini_build.mask_rows, "scene-structural"
    )
    assert len(mini_build.feature_rows) == 320 * 2 * 3
    assert len(mini_build.mask_rows) == 320 * 2
    assert len(mini_build.predictor_rows) == 320
    assert all(
        data.endswith(b"\n") and b"\r" not in data
        for data in mini_build.artifacts.values()
    )

    first = mini_build.predictor_rows[0]
    feature_slice = mini_build.feature_rows[: 2 * 3]
    mask_slice = mini_build.mask_rows[:2]
    assert first["feature_bank_slice_sha256"] == hashlib.sha256(
        b"".join(W2.canonical_json_bytes(dict(row)) for row in feature_slice)
    ).hexdigest()
    assert first["availability_mask_slice_sha256"] == hashlib.sha256(
        b"".join(W2.canonical_json_bytes(dict(row)) for row in mask_slice)
    ).hexdigest()
    matching_anchors = [
        row
        for row in mini_build.anchor_score_rows
        if row["feature_id"]
        == first["cell_id"].replace("RR3", "F", 1).rsplit("-", 2)[0]
        and row["horizon_id"] == first["horizon_id"]
    ]
    assert first["anchor_score_slice_sha256"] == hashlib.sha256(
        b"".join(W2.canonical_json_bytes(dict(row)) for row in matching_anchors)
    ).hexdigest()

    manifest = json.loads(mini_build.artifacts["g2r_output_manifest.json"])
    assert set(manifest) == set(
        _strict_load(W1 / "schema_shape_goldens.json")["schemas"]["output_manifest"][
            "exact_keys"
        ]
    )
    assert set(manifest["artifacts"]) == {
        "wod_scene_anchor_domain",
        "wod_scene_anchor_domain_manifest",
        "g2r_anchor_scores",
        "g2r_blind_feature_bank",
        "g2r_availability_masks",
        "common_support_blind_manifest",
        "g2r_predictor_manifest",
        "nc_pretstar_history_only_receipt",
    }
    direct_artifacts = {
        "g2r_anchor_scores": "g2r_anchor_scores.jsonl",
        "g2r_blind_feature_bank": "g2r_blind_feature_bank.jsonl",
        "g2r_availability_masks": "g2r_availability_masks.jsonl",
        "common_support_blind_manifest": "common_support_blind_manifest.csv",
        "g2r_predictor_manifest": "g2r_predictor_manifest.jsonl",
    }
    for artifact_id, name in direct_artifacts.items():
        reference = manifest["artifacts"][artifact_id]
        assert reference["relative_path"] == name
        assert reference["size_bytes"] == len(mini_build.artifacts[name])
        assert reference["sha256"] == hashlib.sha256(
            mini_build.artifacts[name]
        ).hexdigest()
        assert reference["row_count"] == mini_build.row_counts[name]
    assert manifest["forbidden_output_scan"] == {
        "rating_field_count": 0,
        "leaderboard_file_count": 0,
        "recovery_ledger_file_count": 0,
    }
    joined = b"".join(mini_build.artifacts.values()).lower()
    assert b"observed_rating" not in joined and b"preference_score" not in joined
    direct_rows = b"".join(
        mini_build.artifacts[name] for name in W4.DIRECT_ARTIFACT_NAMES[:-1]
    ).lower()
    assert b"leaderboard" not in direct_rows
    assert b"recovery_ledger" not in direct_rows


def test_w4_mini_universe_is_deterministic_and_matches_bound_fixture(
    mini_build: W4.BlindOutputBuild, tmp_path: Path,
) -> None:
    second = _build_fixture(tmp_path)
    assert second.artifacts == mini_build.artifacts
    expected = _strict_load(W4_FIXTURES / "mini_universe_artifact_manifest.json")
    observed = {
        name: {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "row_count": mini_build.row_counts[name],
        }
        for name, data in mini_build.artifacts.items()
    }
    assert observed == expected["artifacts"]
    assert expected["common_support_ids"] == list(mini_build.common_support_ids)
    assert expected["production_full_scale_execution"] == "DEFERRED_W5_HPC"
    assert expected["pipeline_bindings"] == {
        "candidate_ipv_estimator": (
            "scripts/rq014/build_g2r_blind_outputs.py:"
            "_default_candidate_ipv_estimator"
        ),
        "feature_builder": (
            "scripts/rq014/build_wod_m3_anchors.py:"
            "build_feature_family_m3_input_row"
        ),
        "m3_context_alignment": W2.ALIGNMENT_PRIMARY,
        "scorer_sha256": "b04999aba29a82fb71a97ac22c728479a7734e24a0b32189d08f95184d74f253",
    }


def test_w4_terminal_only_structural_scene_propagates_ineligible_blind(
    mini_build: W4.BlindOutputBuild,
) -> None:
    structural_features = [
        row
        for row in mini_build.feature_rows
        if row["segment_id"] == "scene-structural"
    ]
    assert len(structural_features) == 320 * 3
    assert {
        (row["upstream_status"], row["reason_code"])
        for row in structural_features
    } == {("INELIGIBLE_BLIND", "F_SAFE_PRIMITIVE_OR_FIXTURE_GATE")}
    structural_masks = [
        row
        for row in mini_build.mask_rows
        if row["segment_id"] == "scene-structural"
    ]
    assert len(structural_masks) == 320
    assert {row["scene_cell_status"] for row in structural_masks} == {
        "INELIGIBLE_BLIND"
    }


def test_w4_d4_primary_fingerprint_rejects_sensitivity_mutation(
    mini_build: W4.BlindOutputBuild,
) -> None:
    expected = _strict_load(W4_FIXTURES / "mini_universe_artifact_manifest.json")
    binding = expected["d4_primary_anchor_binding"]
    matching = [
        row
        for row in mini_build.anchor_score_rows
        if all(row[key] == value for key, value in binding["row_identity"].items())
    ]
    assert len(matching) == 1
    assert matching[0]["m3_input_row_sha256_or_NA"] == binding["m3_input_row_sha256"]

    scene = _real_scene()
    sampling = scene.domain.sampling["R04N"]
    focal_reference = W4._scene_focal_reference(scene)
    with pytest.raises(W2.WodM3KernelError):
        W2.build_feature_family_m3_input_row(
            sampling.candidates["C1"],
            sampling.counterpart,
            temporal_id="CH-W10",
            tau_tick=8,
            rate_hz=4,
            h_common_tick=8,
            case_start_tick=-9,
            scene_focal_reference=focal_reference,
            counterpart_is_vehicle=True,
            m3_context_alignment=W2.ALIGNMENT_SENSITIVITY,
        )


def test_w4_production_rejects_injected_producer() -> None:
    scenes, domain_rows, lineage, refs = _fixture_inputs()
    with pytest.raises(W4.G2ROrchestrationError, match="forbids injected"):
        W4.build_blind_output_artifacts(
            scenes,
            domain_rows,
            run_id="RQ014_G2R_W4_FIXTURE",
            git_commit="1" * 40,
            created_at_utc="2026-07-17T00:00:00Z",
            lineage=lineage,
            prerequisite_artifacts=refs,
            score_rows=lambda _rows: (),
        )


def test_w4_scorer_hash_gate_fails_closed(tmp_path: Path) -> None:
    fake_scorer = tmp_path / "m3_scorer.joblib"
    fake_scorer.write_bytes(b"not reviewed")
    with pytest.raises(W4.G2ROrchestrationError, match="size or SHA-256 mismatch"):
        W4.load_verified_scorer_rows(fake_scorer)


def test_w4_staged_write_is_exact_and_noclobber(
    mini_build: W4.BlindOutputBuild, tmp_path: Path
) -> None:
    output_root = tmp_path / "staged"
    output_root.mkdir()
    W4.write_staged_artifacts(output_root, mini_build)
    assert {
        path.name: path.read_bytes() for path in sorted(output_root.iterdir())
    } == dict(mini_build.artifacts)
    with pytest.raises(W4.G2ROrchestrationError, match="empty regular directory"):
        W4.write_staged_artifacts(output_root, mini_build)


def test_w4_keeps_managed_operation_denied_and_frozen_boundaries() -> None:
    authorization = _strict_load(ROOT / "configs" / "research_authorization.json")
    assert W4.OPERATION not in authorization["authorizations"]["RQ014"][
        "allowed_operations"
    ]
    source = (ROOT / "scripts" / "rq014" / "build_g2r_blind_outputs.py").read_text()
    assert "rq014_r2_blind_feature_build_receipt.json" not in source
    assert "DONE.json" not in source
    assert not (ROOT / "configs" / "run_specs" / "RQ014_g2r.template.json").exists()
