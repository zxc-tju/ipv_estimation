#!/usr/bin/env python3
"""Re-export the frozen RQ009 M3 scorer under stable package class paths."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import joblib
import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn

from sociality_estimation.verifier import model


EXPECTED_LEGACY_SHA256 = "bf9a0c7ae41ba9efcb2ad997aaac1b7881d7788cf8dadd01252c17ed7a6b0ba5"
EXPECTED_CALIBRATION_SHA256 = "d8e1048bb4b0109d3c0b428cc879cb3b04610758103d6eb17643a181a9e942f6"
EXPECTED_CONTRACT_SHA256 = "cc69467f8134fcdb297220b34a70ab2b40f2bc25b1074714ac29831f82a15fbb"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_legacy_calibration(path: Path) -> None:
    spec = importlib.util.spec_from_file_location("rq009_calibration", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import legacy calibration module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def portable_tier_model(legacy: object) -> model.TierModel:
    old_spec = legacy.spec
    old_prep = legacy.preprocessor
    spec = model.FeatureSpec(
        name=old_spec.name,
        numeric=list(old_spec.numeric),
        categorical=list(old_spec.categorical),
        zero_numeric=list(old_spec.zero_numeric),
        channel_slot_names=list(old_spec.channel_slot_names),
    )
    preprocessor = model.Preprocessor(
        numeric=list(old_prep.numeric),
        categorical=list(old_prep.categorical),
        zero_numeric=list(old_prep.zero_numeric),
        imputer=old_prep.imputer,
        encoder=old_prep.encoder,
        categorical_mask=list(old_prep.categorical_mask),
    )
    return model.TierModel(spec=spec, preprocessor=preprocessor, models=dict(legacy.models))


def portable_gate_model(legacy: object) -> model.GateModel:
    return model.GateModel(
        numeric=list(legacy.numeric),
        categorical=list(legacy.categorical),
        support_columns=list(legacy.support_columns),
        joint_cell_columns=list(legacy.joint_cell_columns),
        imputer=legacy.imputer,
        scaler=legacy.scaler,
        encoder=legacy.encoder,
        tree=legacy.tree,
        threshold=float(legacy.threshold),
        train_rows=int(legacy.train_rows),
        train_reference_rows=int(legacy.train_reference_rows),
        guard_rows=int(legacy.guard_rows),
        guard_eligible_rows=int(legacy.guard_eligible_rows),
        support_levels=dict(legacy.support_levels),
        unsupported_levels=dict(legacy.unsupported_levels),
        supported_joint_cells=tuple(sorted(legacy.supported_joint_cells)),
        unsupported_joint_cells=list(legacy.unsupported_joint_cells),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-scorer", type=Path, required=True)
    parser.add_argument("--legacy-calibration", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    legacy_sha = sha256_file(args.legacy_scorer)
    if legacy_sha != EXPECTED_LEGACY_SHA256:
        raise RuntimeError(
            f"Unexpected legacy scorer SHA-256: {legacy_sha}; expected {EXPECTED_LEGACY_SHA256}"
        )
    if sha256_file(args.legacy_calibration) != EXPECTED_CALIBRATION_SHA256:
        raise RuntimeError("Unexpected legacy calibration SHA-256")
    if sha256_file(args.feature_contract) != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("Unexpected legacy feature contract SHA-256")
    import_legacy_calibration(args.legacy_calibration)
    legacy = joblib.load(args.legacy_scorer, mmap_mode=None)
    portable = dict(legacy)
    portable["kind"] = "portable_frozen_RQ009_M3_scorer"
    portable["bundle_schema_version"] = 1
    portable["legacy_scorer_sha256"] = legacy_sha
    portable["runtime_module"] = "sociality_estimation.verifier.model"
    portable.pop("calibration_py", None)
    portable.pop("calibration_module_name", None)
    portable["tier_model"] = portable_tier_model(legacy["tier_model"])
    portable["gate_model"] = portable_gate_model(legacy["gate_model"])

    legacy_contract = json.loads(args.feature_contract.read_text(encoding="utf-8"))
    if legacy["feature_contract"] != legacy_contract:
        raise RuntimeError("External and embedded legacy feature contracts differ")
    runtime_contract = dict(legacy_contract)
    runtime_contract["output_columns"] = [
        "pred_human_ipv_point",
        "q_0p025",
        "q_0p05",
        "q_0p1",
        "q_0p5",
        "q_0p9",
        "q_0p95",
        "q_0p975",
        "lo_80",
        "hi_80",
        "width_80",
        "lo_90",
        "hi_90",
        "width_90",
        "lo_95",
        "hi_95",
        "width_95",
        "support_gate_pass",
        "ood_abstain",
    ]
    runtime_contract["runtime_contract_note"] = (
        "Quantile column names match the accepted helper outputs; the legacy "
        "contract used padded q_0p10/q_0p50/q_0p90 spellings."
    )
    portable["feature_contract"] = runtime_contract

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scorer_path = args.output_dir / "m3_scorer.joblib"
    contract_path = args.output_dir / "feature_spec_contract.json"
    legacy_contract_path = args.output_dir / "legacy_feature_spec_contract.json"
    manifest_path = args.output_dir / "manifest.json"
    joblib.dump(portable, scorer_path, compress=("lzma", 3))
    contract_path.write_text(
        json.dumps(runtime_contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    legacy_contract_path.write_text(
        json.dumps(legacy_contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "bundle": "rq009_m3",
        "rq009_run_id": portable["rq009_run_id"],
        "artifact": {
            "path": scorer_path.name,
            "size_bytes": scorer_path.stat().st_size,
            "sha256": sha256_file(scorer_path),
            "legacy_sha256": legacy_sha,
            "compression": "joblib-lzma-3",
        },
        "feature_contract": {
            "path": contract_path.name,
            "sha256": sha256_file(contract_path),
            "legacy_path": legacy_contract_path.name,
            "legacy_sha256": sha256_file(legacy_contract_path),
        },
        "frozen_parameters": {
            "random_state": int(portable["random_state"]),
            "quantile_levels": list(portable["quantile_levels"]),
            "gate_k": model.GATE_K,
            "gate_threshold": float(portable["gate_model"].threshold),
            "gate_train_reference_rows": int(portable["gate_model"].train_reference_rows),
            "radii": portable["radii"],
        },
        "build_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "source_shas": portable.get("source_shas", {}),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
