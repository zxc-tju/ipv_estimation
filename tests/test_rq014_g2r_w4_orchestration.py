from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.rq014 import build_g2r_blind_outputs as W4
from scripts.rq014 import build_wod_m3_anchors as W2
from scripts.rq014 import build_wod_scene_anchor_domain as DOMAIN
from scripts.rq014 import score_wod_m3_deviations as W3


ROOT = Path(__file__).resolve().parents[1]
W1 = ROOT / "tests" / "fixtures" / "rq014_g2r_v1"
W2B = ROOT / "tests" / "fixtures" / "rq014_g2r_w2b"
W4_FIXTURES = ROOT / "tests" / "fixtures" / "rq014_g2r_w4"


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


def _scene(segment_id: str, offsets: tuple[float, float, float]) -> W4.SceneBlindInput:
    sampling = {}
    for sampling_id, rate_hz in DOMAIN.SAMPLING_AXIS:
        ticks = range(-3 * rate_hz, 5 * rate_hz + 1)
        candidates = {
            candidate_id: {
                tick: (float(tick), 0.0 if tick <= 0 else offsets[ordinal - 1])
                for tick in ticks
            }
            for ordinal, candidate_id in W4.CANDIDATES
        }
        sampling[sampling_id] = DOMAIN.SamplingTimelines(
            candidates=candidates,
            counterpart={tick: (float(tick), 2.0) for tick in ticks},
        )
    return W4.SceneBlindInput(
        domain=DOMAIN.SceneDomainInput(segment_id, "CP", sampling),
        route_intent="straight",
    )


def _fixture_inputs():
    # Deliberately supply reverse lexical order; stored rows must use raw UTF-8 order.
    scenes = (
        _scene("scene-2", (0.0, 0.3, 0.4)),
        _scene("scene-10", (0.0, 0.1, 0.2)),
    )
    domain_rows = DOMAIN.build_anchor_domain_rows([scene.domain for scene in scenes])
    base_m3_row = _strict_load(W1 / "m3_input_row_expected.json")

    def feature_builder(candidate_positions, _counterpart_positions, **kwargs):
        marker = float(candidate_positions[kwargs["tau_tick"]][1])
        if (
            marker == 0.3
            and kwargs["rate_hz"] == 10
            and kwargs["temporal_id"] == "TF"
            and kwargs["tau_tick"] == 10
        ):
            raise W4.CandidateTerminalError(
                "M3_SCORING_NUMERICAL_FAILURE",
                "F_M3_SCORING_NUMERICAL_FAILURE",
                "deterministic fixture candidate failure",
            )
        row = copy.deepcopy(base_m3_row)
        row["values"][0] = marker + kwargs["tau_tick"] / kwargs["rate_hz"]
        return row

    def ipv_estimator(*, candidate_positions, tau_tick, **_kwargs):
        return float(candidate_positions[tau_tick][1])

    def score_rows(rows):
        output = []
        for row in rows:
            _, row_sha256 = W2.m3_input_row_bytes_and_sha256(row)
            output.append(
                W3.PreMaskM3Score(
                    m3_input_row_sha256=row_sha256,
                    q_0p5=0.0,
                    lo_90=-0.05,
                    hi_90=0.05,
                    support_gate_pass=True,
                    ood_abstain=False,
                )
            )
        return output

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
    return scenes, domain_rows, feature_builder, ipv_estimator, score_rows, lineage, prerequisite_artifacts


def _build_fixture() -> W4.BlindOutputBuild:
    scenes, domain_rows, feature_builder, ipv_estimator, score_rows, lineage, refs = (
        _fixture_inputs()
    )
    return W4.build_blind_output_artifacts(
        scenes,
        domain_rows,
        score_rows=score_rows,
        run_id="RQ014_G2R_W4_FIXTURE",
        git_commit="1" * 40,
        created_at_utc="2026-07-17T00:00:00Z",
        lineage=lineage,
        prerequisite_artifacts=refs,
        feature_builder=feature_builder,
        candidate_ipv_estimator=ipv_estimator,
        fixture_mode=True,
    )


@pytest.fixture(scope="module")
def mini_build() -> W4.BlindOutputBuild:
    return _build_fixture()


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
    assert mini_build.common_support_ids == ("scene-10",)
    assert W4.scene_passes_blind_predicate(mini_build.mask_rows, "scene-10")
    assert not W4.scene_passes_blind_predicate(mini_build.mask_rows, "scene-2")
    assert len(mini_build.feature_rows) == 320 * 2 * 3
    assert len(mini_build.mask_rows) == 320 * 2
    assert len(mini_build.predictor_rows) == 320
    terminal_features = [
        row
        for row in mini_build.feature_rows
        if row["upstream_status"] == "M3_SCORING_NUMERICAL_FAILURE"
    ]
    assert len(terminal_features) == 20
    assert {
        (row["segment_id"], row["candidate_id"], row["reason_code"])
        for row in terminal_features
    } == {("scene-2", "C2", "F_M3_SCORING_NUMERICAL_FAILURE")}
    assert sum(
        row["scene_cell_status"] == "M3_SCORING_NUMERICAL_FAILURE"
        for row in mini_build.mask_rows
    ) == 20
    assert sum(row["terminal_candidate_slot_count"] == 1 for row in mini_build.predictor_rows) == 20
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
    mini_build: W4.BlindOutputBuild,
) -> None:
    second = _build_fixture()
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


def test_w4_production_cardinality_and_scorer_hash_gates_fail_closed(tmp_path: Path) -> None:
    scenes, domain_rows, feature_builder, ipv_estimator, score_rows, lineage, refs = (
        _fixture_inputs()
    )
    with pytest.raises(W4.G2ROrchestrationError, match="exactly 479 scenes"):
        W4.build_blind_output_artifacts(
            scenes,
            domain_rows,
            score_rows=score_rows,
            run_id="RQ014_G2R_W4_FIXTURE",
            git_commit="1" * 40,
            created_at_utc="2026-07-17T00:00:00Z",
            lineage=lineage,
            prerequisite_artifacts=refs,
            feature_builder=feature_builder,
            candidate_ipv_estimator=ipv_estimator,
        )
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
