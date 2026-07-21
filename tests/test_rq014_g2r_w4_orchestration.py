from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
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
SCORER = ROOT / "models" / "rq009_m3" / "m3_scorer.joblib"
SCORER_ANCHOR_FIELDS = (
    "m3_q_0p5",
    "m3_lo_90",
    "m3_hi_90",
    "nex",
    "nmd",
    "amd",
)
ANCHOR_NUMERIC_IDENTITY_FIELDS = (
    "segment_id",
    "feature_id",
    "horizon_id",
    "tau_tick",
    "candidate_ordinal",
)
FEATURE_NUMERIC_IDENTITY_FIELDS = (
    "cell_index",
    "segment_id",
    "candidate_ordinal",
)
PORTABLE_ATOL = 1e-7


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
        tick: (float(tick), 0.02 * float(tick) ** 2) for tick in range(-12, 9)
    }
    counterpart = {
        tick: (0.8 * float(tick) + 1.0, 2.0 - 0.05 * float(tick))
        for tick in range(-12, 9)
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


def _structural_scene(segment_id: str = "scene-structural") -> W4.SceneBlindInput:
    return W4.SceneBlindInput(
        domain=DOMAIN.SceneDomainInput(
            segment_id,
            "NA",
            {},
            terminal_status="INELIGIBLE_BLIND",
        ),
        route_intent="straight",
    )


def _worker_failure_scene(segment_id: str) -> W4.SceneBlindInput:
    candidate = {-1: (-1.0, 0.0), 0: (0.0, 0.0)}
    counterpart = {-1: (0.0, 1.0), 0: (1.0, 1.0)}
    return W4.SceneBlindInput(
        domain=DOMAIN.SceneDomainInput(
            segment_id,
            "CP",
            {
                "R04N": DOMAIN.SamplingTimelines(
                    candidates={
                        candidate_id: candidate
                        for _, candidate_id in W4.CANDIDATES
                    },
                    counterpart=counterpart,
                )
            },
        ),
        route_intent="straight",
    )


def _available_sampling(rate_hz: int) -> DOMAIN.SamplingTimelines:
    ticks = range(-12, 2 * rate_hz + 1)
    candidate = {
        tick: (
            float(tick) / rate_hz,
            0.02 * (float(tick) / rate_hz) ** 2,
        )
        for tick in ticks
    }
    counterpart = {
        tick: (
            0.8 * float(tick) / rate_hz + 1.0,
            2.0 - 0.05 * float(tick) / rate_hz,
        )
        for tick in ticks
    }
    return DOMAIN.SamplingTimelines(
        candidates={candidate_id: candidate for _, candidate_id in W4.CANDIDATES},
        counterpart=counterpart,
    )


def _mixed_sampling_scene(available_sampling_id: str) -> W4.SceneBlindInput:
    terminal = DOMAIN.SamplingTimelines(
        candidates={},
        counterpart={},
        terminal_status="INELIGIBLE_TIMELINE_SUPPORT",
    )
    sampling = {
        "R04N": _available_sampling(4) if available_sampling_id == "R04N" else terminal,
        "R10L": _available_sampling(10) if available_sampling_id == "R10L" else terminal,
    }
    return W4.SceneBlindInput(
        domain=DOMAIN.SceneDomainInput("scene-mixed", "CP", sampling),
        route_intent="straight",
    )


def _fixture_inputs(
    scenes: tuple[W4.SceneBlindInput, ...] | None = None,
    domain_rows: list[dict[str, str]] | None = None,
):
    if scenes is None:
        scenes = (_structural_scene(), _real_scene())
    if domain_rows is None:
        domain_rows = DOMAIN.build_anchor_domain_rows(
            [scene.domain for scene in scenes]
        )
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


def _declared_source_refs(contract, _repo_root):
    return {
        source_id: {
            "relative_path": contract["source_bindings"][source_id]["path"],
            "size_bytes": contract["source_bindings"][source_id]["size_bytes"],
            "sha256": contract["source_bindings"][source_id]["sha256"],
        }
        for source_id in W4.SOURCE_CONTRACT_IDS
    }


def _identity_digest(rows, fields) -> str:
    identities = [[row[field] for field in fields] for row in rows]
    return hashlib.sha256(W2.canonical_json_bytes(identities)).hexdigest()


def _portable_structure(build: W4.BlindOutputBuild) -> dict:
    """Remove only scorer-portable numerics and hashes derived from their bytes."""

    anchor_rows = []
    for source in build.anchor_score_rows:
        row = dict(source)
        for field in SCORER_ANCHOR_FIELDS:
            value = row[field]
            if value.get("kind") == "FINITE_FLOAT":
                row[field] = {
                    "kind": "FINITE_FLOAT",
                    "value": "PORTABLE_ATOL_1E-7",
                }
        anchor_rows.append(row)

    feature_rows = []
    for source in build.feature_rows:
        row = dict(source)
        if row["predictor_value"].get("kind") == "FINITE_FLOAT":
            row["predictor_value"] = {
                "kind": "FINITE_FLOAT",
                "value": "PORTABLE_ATOL_1E-7",
            }
        row["anchor_slice_sha256"] = "SCORER_DERIVED_SHA256"
        feature_rows.append(row)

    predictor_rows = []
    for source in build.predictor_rows:
        row = dict(source)
        for field in (
            "anchor_score_slice_sha256",
            "feature_bank_slice_sha256",
            "availability_mask_slice_sha256",
        ):
            row[field] = "SCORER_DERIVED_SHA256"
        predictor_rows.append(row)

    return {
        "anchor_score_rows": anchor_rows,
        "common_support_ids": list(build.common_support_ids),
        "feature_rows": feature_rows,
        "mask_rows": [dict(row) for row in build.mask_rows],
        "predictor_rows": predictor_rows,
        "row_counts": dict(build.row_counts),
    }


def _scorer_numeric_projection(build: W4.BlindOutputBuild) -> dict:
    anchor_rows = [
        row for row in build.anchor_score_rows if row["upstream_status"] == "AVAILABLE"
    ]
    feature_rows = [
        row
        for row in build.feature_rows
        if row["predictor_value"].get("kind") == "FINITE_FLOAT"
    ]
    return {
        "absolute_tolerance": PORTABLE_ATOL,
        "anchor_rows": {
            "identity_fields": list(ANCHOR_NUMERIC_IDENTITY_FIELDS),
            "identity_sha256": _identity_digest(
                anchor_rows, ANCHOR_NUMERIC_IDENTITY_FIELDS
            ),
            "row_count": len(anchor_rows),
            "values_by_field": {
                field: [row[field]["value"] for row in anchor_rows]
                for field in SCORER_ANCHOR_FIELDS
            },
        },
        "feature_rows": {
            "identity_fields": list(FEATURE_NUMERIC_IDENTITY_FIELDS),
            "identity_sha256": _identity_digest(
                feature_rows, FEATURE_NUMERIC_IDENTITY_FIELDS
            ),
            "predictor_values": [
                row["predictor_value"]["value"] for row in feature_rows
            ],
            "row_count": len(feature_rows),
        },
        "portable_fixture_sha256": (
            "ae62b9fddba53308d319ccef5a70d56a9f0ae243fe009aa3f85e36cb20fcee37"
        ),
        "relative_tolerance": 0.0,
        "reviewed_test_reference": "tests/test_verifier_runtime.py:152",
    }


def _d4_input_row(alignment: str) -> dict:
    scene = _real_scene()
    sampling = scene.domain.sampling["R04N"]
    focal_reference = W4._scene_focal_reference(scene, {"R04N"})
    return W2.build_feature_family_m3_input_row(
        sampling.candidates["C1"],
        sampling.counterpart,
        temporal_id="CH-W10",
        tau_tick=8,
        rate_hz=4,
        h_common_tick=8,
        case_start_tick=-12,
        scene_focal_reference=focal_reference,
        counterpart_is_vehicle=True,
        m3_context_alignment=alignment,
    )


def _assert_portable_numerics(
    build: W4.BlindOutputBuild, expected: dict
) -> None:
    observed = _scorer_numeric_projection(build)
    assert {
        key: observed[key]
        for key in (
            "absolute_tolerance",
            "portable_fixture_sha256",
            "relative_tolerance",
            "reviewed_test_reference",
        )
    } == {
        key: expected[key]
        for key in (
            "absolute_tolerance",
            "portable_fixture_sha256",
            "relative_tolerance",
            "reviewed_test_reference",
        )
    }
    assert expected["absolute_tolerance"] == PORTABLE_ATOL
    assert expected["relative_tolerance"] == 0.0
    for section, value_key in (
        ("anchor_rows", "values_by_field"),
        ("feature_rows", "predictor_values"),
    ):
        assert observed[section]["identity_fields"] == expected[section][
            "identity_fields"
        ]
        assert observed[section]["identity_sha256"] == expected[section][
            "identity_sha256"
        ]
        assert observed[section]["row_count"] == expected[section]["row_count"]
        if section == "anchor_rows":
            assert set(observed[section][value_key]) == set(
                expected[section][value_key]
            )
            for field in SCORER_ANCHOR_FIELDS:
                assert len(observed[section][value_key][field]) == len(
                    expected[section][value_key][field]
                ) == expected[section]["row_count"]
                for actual, frozen in zip(
                    observed[section][value_key][field],
                    expected[section][value_key][field],
                ):
                    assert math.isclose(
                        actual, frozen, rel_tol=0.0, abs_tol=PORTABLE_ATOL
                    )
        else:
            assert len(observed[section][value_key]) == len(
                expected[section][value_key]
            ) == expected[section]["row_count"]
            for actual, frozen in zip(
                observed[section][value_key], expected[section][value_key]
            ):
                assert math.isclose(
                    actual, frozen, rel_tol=0.0, abs_tol=PORTABLE_ATOL
                )


def _install_sandbox_sysconf_fallback() -> None:
    """Permit multiprocessing semaphores in the restricted test sandbox."""

    real_sysconf = W4.os.sysconf

    def sandbox_safe_sysconf(name):
        try:
            return real_sysconf(name)
        except PermissionError:
            if name == "SC_SEM_NSEMS_MAX":
                return -1
            raise

    W4.os.sysconf = sandbox_safe_sysconf


def _build_fixture_in_process(worker_count: int | None = None) -> W4.BlindOutputBuild:
    if worker_count is not None:
        W4.G2R_PREPASS_MAX_WORKERS = worker_count
    _install_sandbox_sysconf_fallback()
    # Phase 1 deliberately leaves governance digests for the Phase 2 cascade.
    W4._source_contract_refs = _declared_source_refs
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


def _build_fixture(
    tmp_path: Path, worker_count: int | None = None
) -> W4.BlindOutputBuild:
    """Run the real scorer in an isolated process, then return its exact bytes.

    Scikit-learn initializes an OpenMP runtime when the pinned scorer predicts.
    Keeping that runtime in the pytest process can poison the later resource-
    pilot fork pool on macOS.  Process isolation preserves the real W2-to-W3
    integration while ensuring that the scorer runtime exits with its child.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    payload_path = tmp_path / "mini_build.pickle"
    test_path = Path(__file__).resolve()
    command = (
        "import pickle, runpy\n"
        f"namespace = runpy.run_path({str(test_path)!r})\n"
        f"build = namespace['_build_fixture_in_process']({worker_count!r})\n"
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
    progress = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith("g2r-prepass scene ")
    ]
    assert len(progress) == 2
    assert all(line.endswith("s") for line in progress)
    return pickle.loads(payload_path.read_bytes())


@pytest.fixture(scope="module")
def mini_build(tmp_path_factory: pytest.TempPathFactory) -> W4.BlindOutputBuild:
    return _build_fixture(
        tmp_path_factory.mktemp("rq014-g2r-w4-mini-build"), worker_count=2
    )


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


@pytest.mark.parametrize("available_sampling_id", ["R10L", "R04N"])
def test_w4_focal_reference_selects_frozen_order_available_sampling_branch(
    available_sampling_id: str,
) -> None:
    scene = _mixed_sampling_scene(available_sampling_id)
    domain_rows = DOMAIN.build_anchor_domain_rows([scene.domain])
    available_rows = [
        row for row in domain_rows if row["membership_status"] == "AVAILABLE"
    ]
    assert available_rows
    assert {
        row["feature_id"].split("-", 2)[1] for row in available_rows
    } == {available_sampling_id}

    sampling = scene.domain.sampling[available_sampling_id]
    shared_ticks = set.intersection(
        *(set(sampling.candidates[candidate_id]) for _, candidate_id in W4.CANDIDATES)
    )
    history = [
        sampling.candidates["C1"][tick]
        for tick in sorted(shared_ticks)
        if tick <= 0
    ]
    expected_reference = W2.build_scene_focal_reference(
        history, route_intent=scene.route_intent
    ).tolist()
    observed_references = []
    base_row = _strict_load(W1 / "m3_input_row_expected.json")

    def feature_builder(_candidate, _counterpart, **kwargs):
        observed_references.append(kwargs["scene_focal_reference"].tolist())
        return base_row

    def ipv_estimator(**_kwargs):
        return 0.0

    def score_rows(rows):
        output = []
        for row in rows:
            _, row_sha256 = W2.m3_input_row_bytes_and_sha256(row)
            output.append(
                W3.PreMaskM3Score(
                    m3_input_row_sha256=row_sha256,
                    q_0p5=0.0,
                    lo_90=-1.0,
                    hi_90=1.0,
                    support_gate_pass=True,
                    ood_abstain=False,
                )
            )
        return output

    anchor_rows = W4.build_anchor_score_rows(
        [scene],
        domain_rows,
        score_rows=score_rows,
        feature_builder=feature_builder,
        candidate_ipv_estimator=ipv_estimator,
    )
    assert anchor_rows
    assert {row["sampling_id"] for row in anchor_rows} == {available_sampling_id}
    assert observed_references
    assert all(reference == expected_reference for reference in observed_references)


def test_w4_parallel_prepass_is_byte_identical_to_serial(
    tmp_path: Path, mini_build: W4.BlindOutputBuild,
) -> None:
    serial = _build_fixture(tmp_path / "serial", worker_count=1)
    parallel = mini_build

    assert tuple(serial.artifacts) == tuple(parallel.artifacts) == W4.DIRECT_ARTIFACT_NAMES
    for name in W4.DIRECT_ARTIFACT_NAMES:
        assert parallel.artifacts[name] == serial.artifacts[name]


def test_w4_scene_worker_is_scorer_free_and_returns_complete_failures() -> None:
    test_path = Path(__file__).resolve()
    command = f"""
import math
import os
import runpy
import sys

def scorer_module(name):
    return (name == 'joblib' or name.startswith('joblib.') or
            name == 'sociality_estimation.verifier.scorer')

assert not any(scorer_module(name) for name in sys.modules)
namespace = runpy.run_path({str(test_path)!r})
W4 = namespace['W4']
assert not any(scorer_module(name) for name in sys.modules)
scene = namespace['_worker_failure_scene']('scene-worker')
domain_row = {{
    'segment_id': 'scene-worker',
    'feature_id': 'F-R04N-CH-W10',
    'horizon_id': 'H20',
    'path_type_or_NA': 'CP',
    'h_common_tick_or_NA': 'NA',
    'tau_tick_or_NA': '4',
    'membership_status': 'AVAILABLE',
    'reason_code': 'F_AVAILABLE_CONTINUE',
}}
status_by_reason, _ = W4._status_tables(W4._load_contract())
task = W4._ScenePrepassTask(
    scene=scene,
    indexed_domain_rows=((0, domain_row),),
    status_by_reason=tuple(status_by_reason.items()),
)
W4._initialize_g2r_prepass_worker()
result = W4._run_g2r_scene_prepass(task)
assert not any(scorer_module(name) for name in sys.modules)
assert result.segment_id == 'scene-worker'
assert result.pending == ()
assert [(index, ordinal) for index, ordinal, _ in result.emitted] == [
    (0, 1), (0, 2), (0, 3)
]
assert all(
    set(row) == W4.ANCHOR_SCORE_KEYS and
    row['upstream_status'] == 'INELIGIBLE_TIMELINE_SUPPORT' and
    row['reason_code'] == 'F_TIMELINE_SUPPORT'
    for _, _, row in result.emitted
)
assert math.isfinite(result.elapsed_s) and result.elapsed_s >= 0.0
assert all(os.environ[var] == '1' for var in W4._G2R_BLAS_THREAD_VARIABLES)
"""
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
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


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
    assert expected["same_runtime_determinism"] == "BYTE_EXACT_ALL_SIX_ARTIFACTS_X2"
    assert dict(mini_build.row_counts) == expected["artifact_row_counts"]
    assert expected["common_support_ids"] == list(mini_build.common_support_ids)
    structure_bytes = W2.canonical_json_bytes(_portable_structure(mini_build))
    assert len(structure_bytes) == expected["portable_structure"]["size_bytes"]
    assert hashlib.sha256(structure_bytes).hexdigest() == expected[
        "portable_structure"
    ]["sha256"]
    _assert_portable_numerics(
        mini_build, expected["scorer_numeric_portable_projection"]
    )
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
    binding = expected["d4_input_binding"]
    primary = binding["primary"]
    primary_bytes = bytes.fromhex(primary["canonical_json_utf8_hex"])
    assert len(primary_bytes) == primary["size_bytes"]
    assert hashlib.sha256(primary_bytes).hexdigest() == primary["sha256"]
    assert primary["alignment"] == W2.ALIGNMENT_PRIMARY
    matching = [
        row
        for row in mini_build.anchor_score_rows
        if all(row[key] == value for key, value in binding["row_identity"].items())
    ]
    assert len(matching) == 1
    assert matching[0]["m3_input_row_sha256_or_NA"] == primary["sha256"]

    sensitivity = binding["sensitivity"]
    sensitivity_bytes, sensitivity_sha256 = W2.m3_input_row_bytes_and_sha256(
        _d4_input_row(W2.ALIGNMENT_SENSITIVITY)
    )
    assert sensitivity["alignment"] == W2.ALIGNMENT_SENSITIVITY
    assert sensitivity_bytes.hex() == sensitivity["canonical_json_utf8_hex"]
    assert len(sensitivity_bytes) == sensitivity["size_bytes"]
    assert sensitivity_sha256 == sensitivity["sha256"]
    assert sensitivity_sha256 != primary["sha256"]


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


def test_w4_authorization_keeps_standalone_kernel_rating_blind() -> None:
    authorization = _strict_load(ROOT / "configs" / "research_authorization.json")
    assert W4.OPERATION in authorization["authorizations"]["RQ014"][
        "allowed_operations"
    ]
    source = (ROOT / "scripts" / "rq014" / "build_g2r_blind_outputs.py").read_text()
    assert "rq014_r2_blind_feature_build_receipt.json" not in source
    assert "DONE.json" not in source
    template = _strict_load(ROOT / "configs" / "run_specs" / "RQ014_g2r.template.json")
    assert template["operation"] == W4.OPERATION
    assert template["resource_profile_id"] == "rq014-g2r-cpu-v1"
