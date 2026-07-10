"""Portable loader and scorer for the frozen RQ009 M3 IPV verifier."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

import joblib
import numpy as np
import pandas as pd

from sociality_estimation.verifier import model
from sociality_estimation.verifier.deviation import raw_envelope_deviation


LOGGER = logging.getLogger("sociality_estimation.verifier")
MODEL_ENV_VAR = "SOCIALITY_M3_SCORER"
MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "rq009_m3"
DEFAULT_SCORER_PATH = MODEL_DIR / "m3_scorer.joblib"
DEFAULT_MANIFEST_PATH = MODEL_DIR / "manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_for(scorer_path: Path) -> Optional[Path]:
    default = scorer_path.with_name("manifest.json")
    return default if default.exists() else None


@lru_cache(maxsize=4)
def load_scorer(
    path: Optional[str | Path] = None,
    *,
    verify_hash: bool = True,
) -> Mapping[str, Any]:
    scorer_path = Path(path or os.environ.get(MODEL_ENV_VAR, DEFAULT_SCORER_PATH)).resolve()
    if not scorer_path.is_file():
        raise FileNotFoundError(
            f"M3 scorer not found at {scorer_path}. Set {MODEL_ENV_VAR} or sync the model bundle."
        )
    manifest_path = _manifest_for(scorer_path)
    manifest: Optional[Mapping[str, Any]] = None
    if verify_hash:
        if manifest_path is None:
            raise FileNotFoundError(f"Model manifest missing beside {scorer_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("schema_version", -1)) != 1:
            raise RuntimeError("Unsupported M3 bundle schema")
        expected = str(manifest["artifact"]["sha256"])
        observed = sha256_file(scorer_path)
        if observed != expected:
            raise RuntimeError(
                f"M3 scorer checksum mismatch: expected {expected}, observed {observed}"
            )
        if scorer_path.stat().st_size != int(manifest["artifact"]["size_bytes"]):
            raise RuntimeError("M3 scorer size does not match its manifest")
        contract_path = scorer_path.with_name(str(manifest["feature_contract"]["path"]))
        if sha256_file(contract_path) != str(manifest["feature_contract"]["sha256"]):
            raise RuntimeError("M3 feature contract checksum mismatch")
    scorer = joblib.load(scorer_path, mmap_mode=None)
    required = {"tier_model", "gate_model", "radii", "feature_contract"}
    missing = required - set(scorer)
    if missing:
        raise RuntimeError(f"M3 scorer is missing keys: {sorted(missing)}")
    if manifest is not None:
        contract_path = scorer_path.with_name(str(manifest["feature_contract"]["path"]))
        external_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if scorer["feature_contract"] != external_contract:
            raise RuntimeError("Embedded and external M3 feature contracts differ")
        frozen = manifest["frozen_parameters"]
        if list(scorer["quantile_levels"]) != list(frozen["quantile_levels"]):
            raise RuntimeError("M3 quantile levels differ from the manifest")
        if int(frozen["gate_k"]) != model.GATE_K:
            raise RuntimeError("M3 gate k differs from the runtime")
        if float(scorer["gate_model"].threshold) != float(frozen["gate_threshold"]):
            raise RuntimeError("M3 gate threshold differs from the manifest")
        if int(scorer["gate_model"].train_reference_rows) != int(
            frozen["gate_train_reference_rows"]
        ):
            raise RuntimeError("M3 gate reference count differs from the manifest")
    return scorer


def quantile_column_name(quantile: float) -> str:
    text = f"{quantile:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"q_{text}"


def _check_columns(frame: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        preview = ", ".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f" ... (+{len(missing) - 20})"
        raise ValueError(f"Input frame is missing M3 columns: {preview}{suffix}")


def score_anchors(
    frame: pd.DataFrame,
    scorer_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Score already-derived M3 anchor features with CQR and OOD abstention."""
    scorer = load_scorer(scorer_path)
    contract = scorer["feature_contract"]
    _check_columns(frame, list(contract["required_input_columns"]))

    quantiles, _, _ = model.predict_tier_quantiles(scorer["tier_model"], frame)
    gate_ok, _ = model.apply_gate(frame, scorer["gate_model"])
    output = pd.DataFrame(index=frame.index)
    for column in (
        "case_key",
        "scene_unique_id",
        "anchor_frame_index",
        "anchor_timestamp",
        "perspective",
        "source_dataset",
    ):
        if column in frame.columns:
            output[column] = frame[column].to_numpy()

    for quantile in model.QUANTILE_LEVELS:
        output[quantile_column_name(quantile)] = quantiles[:, model.Q_INDEX[quantile]]
    output["pred_human_ipv_point"] = quantiles[:, model.Q_INDEX[0.50]]

    for alpha in model.ALPHAS:
        label = model.ALPHA_LABEL[alpha]
        lower_level, upper_level = model.QUANTILE_BY_ALPHA[alpha]
        lower = quantiles[:, model.Q_INDEX[lower_level]].astype(np.float32, copy=False)
        upper = quantiles[:, model.Q_INDEX[upper_level]].astype(np.float32, copy=False)
        radius = float(scorer["radii"][label]["c_alpha"])
        calibrated_lower, calibrated_upper = model.calibrated_bounds(lower, upper, radius)
        calibrated_lower[~gate_ok] = np.nan
        calibrated_upper[~gate_ok] = np.nan
        output[f"lo_{label}"] = calibrated_lower
        output[f"hi_{label}"] = calibrated_upper
        output[f"width_{label}"] = (calibrated_upper - calibrated_lower).astype(np.float32)

    output["support_gate_pass"] = gate_ok
    output["ood_abstain"] = ~gate_ok
    return output


def score_verifier(
    frame: pd.DataFrame,
    *,
    observed_ipv_column: str,
    level: int = 90,
    scorer_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Score the envelope and append canonical raw deviation columns."""
    if level not in (80, 90, 95):
        raise ValueError("level must be one of 80, 90, or 95")
    if observed_ipv_column not in frame.columns:
        raise ValueError(f"Observed IPV column not found: {observed_ipv_column}")
    output = score_anchors(frame, scorer_path)
    observed = frame[observed_ipv_column].to_numpy(dtype=float, copy=False)
    signed, absolute, outside = raw_envelope_deviation(
        observed,
        output[f"lo_{level}"].to_numpy(dtype=float, copy=False),
        output[f"hi_{level}"].to_numpy(dtype=float, copy=False),
    )
    output[observed_ipv_column] = observed
    usable = np.isfinite(signed)
    output[f"deviation_usable_{level}"] = usable
    output[f"deviation_signed_exceedance_{level}"] = signed
    output[f"deviation_abs_exceedance_{level}"] = absolute
    output[f"deviation_outside_{level}"] = outside
    output[f"deviation_upper_tail_{level}"] = usable & (signed > 0.0)
    output[f"deviation_lower_tail_{level}"] = usable & (signed < 0.0)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, default=None)
    parser.add_argument("--observed-ipv-column")
    parser.add_argument("--level", type=int, default=90)
    args = parser.parse_args()
    frame = pd.read_parquet(args.input) if args.input.suffix == ".parquet" else pd.read_csv(args.input)
    if args.observed_ipv_column:
        output = score_verifier(
            frame,
            observed_ipv_column=args.observed_ipv_column,
            level=args.level,
            scorer_path=args.scorer,
        )
    else:
        output = score_anchors(frame, args.scorer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".parquet":
        output.to_parquet(args.output, index=False)
    else:
        output.to_csv(args.output, index=False)
    LOGGER.info("Wrote %s rows to %s", len(output), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
