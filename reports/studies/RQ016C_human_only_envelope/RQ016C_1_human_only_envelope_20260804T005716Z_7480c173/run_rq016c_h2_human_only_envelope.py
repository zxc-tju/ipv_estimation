#!/usr/bin/env python3
"""RQ016C-H2: fit the fixed human-only envelope and prove OnSite rows can be scored."""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


REPO_ROOT = Path(__file__).resolve().parents[4]
REF_SCRIPT = (
    REPO_ROOT
    / "reports"
    / "studies"
    / "RQ016_human_envelope_rebuild"
    / "RQ016_1_envelope_rebuild_20260803T134808Z_d23fa836"
    / "run_rq016_a1_envelope_rebuild.py"
)
PREVIOUS_KEY_NUMBERS = REF_SCRIPT.with_name("key_numbers.json")

WORK_DIR = REPO_ROOT / ".codex-fleet" / "rq016c-human-only-envelope" / "work" / "H2"
REPORT_PATH = (
    REPO_ROOT
    / ".codex-fleet"
    / "rq016c-human-only-envelope"
    / "board"
    / "reports"
    / "RQ016C_2_human_only_envelope_fixed.md"
)
KEY_NUMBERS_PATH = WORK_DIR / "key_numbers.json"
MODEL_DIR = WORK_DIR / "envelope_model"
MODEL_PICKLE_PATH = MODEL_DIR / "rq016c_h2_envelope.pkl"
H1_KEY_NUMBERS_PATH = REPO_ROOT / ".codex-fleet" / "rq016c-human-only-envelope" / "work" / "H1" / "key_numbers.json"
ONSITE_PATH = (
    REPO_ROOT
    / "data"
    / "derived"
    / "onsite_competition"
    / "RQ012B_event_harm"
    / "stage3plus"
    / "onsite_anchors_multi"
    / "onsite_m3_av_anchors_multi_allvalid.parquet"
)

H2_CATEGORICAL_CONTEXT = [
    "geometry_path_category",
    "geometry_path_relation",
    "turn_pair_label",
    "priority_role",
]
H2_SUPPORT_CATEGORICAL = ["geometry_path_category", "priority_role"]
AUDIT_ONLY_COLUMNS = ["vehicle_type_list"]
FILTER_ONLY_COLUMNS = ["agent_type_pair", "av_included"]
HUMAN_PAIR = "HV;HV"
HUMAN_AV_INCLUDED = "all_HV"
FORBIDDEN_FEATURE_COLUMNS = {"agent_type_pair", "av_included", "source_dataset"}
FORBIDDEN_OLD_IPV_COLUMNS = {
    "counterpart_ipv_current",
    "counterpart_ipv_error_current",
    "counterpart_ipv_slope_pre_anchor",
    "M4_ONLY_ego_self_anchor_ipv_current",
    "M4_ONLY_ego_self_anchor_ipv_error_current",
}

EXPECTED_HUMAN_FOLD_ROWS = {
    "train": 974_984,
    "calibration": 481_088,
    "guard_tune": 499_893,
    "test": 486_660,
}
EXPECTED_STATUS_OK_FOLD_ROWS = {
    "train": 1_290_663,
    "calibration": 629_593,
    "guard_tune": 646_772,
    "test": 635_618,
}
EXPECTED_HUMAN_SPLIT_ROWS = {"development": 1_752_509, "guard": 690_116}
EXPECTED_HUMAN_TOTAL_ROWS = 2_442_625
EXPECTED_CONTEXT_CELLS = 12
EXPECTED_ONSITE_ROWS = 67_861
EXPECTED_ONSITE_CELLS = 9
EXPECTED_ONSITE_MISSING_CELLS = 0
EXPECTED_ONSITE_MIN_SUPPORT = 2_209
EXPECTED_ONSITE_MIN_CELL = "CP|equal"
EXPECTED_ONSITE_MIN_CELL_ROWS = 116


def load_reference_module() -> Any:
    spec = importlib.util.spec_from_file_location("rq016_a1_reference", REF_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import reference script: {REF_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.M2_CATEGORICAL_CONTEXT = list(H2_CATEGORICAL_CONTEXT)
    module.GATE_SUPPORT_CATEGORICAL = list(H2_SUPPORT_CATEGORICAL)
    module.GATE_JOINT_CELL = list(H2_SUPPORT_CATEGORICAL)
    return module


ref = load_reference_module()


def utc_now() -> str:
    return subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def pct_percent(numerator: int | float, denominator: int | float, digits: int = 4) -> str:
    return f"{pct(numerator, denominator) * 100:.{digits}f}%"


def product_row_key_parts(product_keys: pd.Series) -> pd.DataFrame:
    extracted = product_keys.astype("string").str.extract(
        r"^case_key=(?P<case_key>.*?)\|anchor_frame_index=(?P<anchor_frame_index>-?\d+)\|perspective=(?P<perspective>.*?)\|source_dataset=(?P<source_dataset>.*?)$"
    )
    if extracted.isna().any(axis=None):
        bad = product_keys[extracted.isna().any(axis=1)].head(3).tolist()
        raise RuntimeError(f"product_row_key_parse_failed examples={bad}")
    extracted["anchor_frame_index"] = extracted["anchor_frame_index"].astype("int64")
    return extracted


def matrix_columns() -> list[str]:
    columns = set(ref.KEY_COLUMNS + [ref.FOLD_COLUMN, ref.TARGET_COLUMN])
    columns.update(ref.M2_NUMERIC_CONTEXT)
    columns.update(H2_CATEGORICAL_CONTEXT)
    columns.update(AUDIT_ONLY_COLUMNS)
    columns.update(FILTER_ONLY_COLUMNS)
    columns.update(ref.GATE_DISTANCE_NUMERIC)
    columns.update(H2_SUPPORT_CATEGORICAL)
    return sorted(columns)


def load_joined_folds(ledger: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    ledger_parts = product_row_key_parts(ledger["product_row_key"])
    case_keys = sorted(ledger_parts["case_key"].astype(str).unique().tolist())
    dataset = ds.dataset(ref.MATRIX_ROOT, format="parquet", partitioning="hive")
    columns = matrix_columns()
    ledger_lookup = ledger.set_index("product_row_key", drop=False)
    folds: Dict[str, pd.DataFrame] = {}
    fold_counts: Dict[str, Dict[str, Any]] = {}
    matrix_rows_read = 0
    matrix_key_duplicates = 0
    joined_rows = 0

    for fold in ref.FOLDS:
        table = dataset.to_table(
            columns=columns,
            filter=(ds.field("case_key").isin(case_keys)) & (ds.field("fold") == fold),
        )
        matrix = table.to_pandas()
        matrix_rows_read += int(len(matrix))
        matrix["anchor_frame_index"] = matrix["anchor_frame_index"].astype("int64")
        matrix["product_row_key"] = ref.product_row_key(matrix)
        matrix_key_duplicates += int(matrix.duplicated("product_row_key").sum())
        joined = matrix.merge(
            ledger_lookup[["status", "reason_code", "rq007_split", "context_cell_key"]],
            left_on="product_row_key",
            right_index=True,
            how="inner",
            validate="one_to_one",
        )
        joined_rows += int(len(joined))
        for column in (
            ref.KEY_COLUMNS
            + [ref.FOLD_COLUMN, "status", "reason_code", "rq007_split", "context_cell_key"]
            + FILTER_ONLY_COLUMNS
        ):
            joined[column] = joined[column].astype("string")
        status_ok = joined.loc[joined["status"] == "OK"]
        human = status_ok.loc[status_ok["agent_type_pair"] == HUMAN_PAIR]
        folds[fold] = joined.reset_index(drop=True)
        fold_counts[fold] = {
            "matrix_rows_read_after_k2_case_filter": int(len(matrix)),
            "joined_dev_guard_rows": int(len(joined)),
            "status_ok_rows": int(len(status_ok)),
            "human_status_ok_rows": int(len(human)),
            "human_share_of_status_ok": pct(int(len(human)), int(len(status_ok))),
        }
        del table, matrix, joined, status_ok, human
        gc.collect()

    diag = {
        "matrix_source_path": str(ref.MATRIX_ROOT.relative_to(REPO_ROOT)),
        "k2_case_key_filter_unique_cases": int(len(case_keys)),
        "matrix_rows_read_after_k2_case_filter": matrix_rows_read,
        "matrix_product_row_key_duplicates": matrix_key_duplicates,
        "joined_rows": joined_rows,
        "ledger_rows": int(len(ledger)),
        "ledger_left_join_hits": joined_rows,
        "ledger_left_join_misses": int(len(ledger) - joined_rows),
        "one_to_zero_or_one_ok": bool(
            joined_rows == len(ledger)
            and int(ledger.duplicated("product_row_key").sum()) == 0
            and matrix_key_duplicates == 0
        ),
        "fold_counts": fold_counts,
    }
    return folds, diag


def human_frames(joined_folds: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for fold, frame in joined_folds.items():
        selected = frame.loc[(frame["status"] == "OK") & (frame["agent_type_pair"] == HUMAN_PAIR)].copy()
        out[fold] = selected.reset_index(drop=True)
    return out


def assert_feature_contract() -> Dict[str, Any]:
    feature_columns = set(ref.M2_NUMERIC_CONTEXT) | set(H2_CATEGORICAL_CONTEXT)
    gate_columns = set(H2_SUPPORT_CATEGORICAL)
    old_ipv_in_features = sorted(feature_columns & FORBIDDEN_OLD_IPV_COLUMNS)
    old_ipv_in_gate = sorted(set(ref.GATE_DISTANCE_NUMERIC) & FORBIDDEN_OLD_IPV_COLUMNS)
    forbidden_in_features = sorted(feature_columns & FORBIDDEN_FEATURE_COLUMNS)
    forbidden_in_gate = sorted(gate_columns & {"agent_type_pair", "av_included"})
    checks = {
        "agent_type_pair_not_in_m2_features": "agent_type_pair" not in feature_columns,
        "av_included_not_in_m2_features": "av_included" not in feature_columns,
        "agent_type_pair_not_in_gate_keys": "agent_type_pair" not in gate_columns,
        "av_included_not_in_gate_keys": "av_included" not in gate_columns,
        "source_dataset_not_in_predictors": "source_dataset" not in feature_columns,
        "old_ipv_channels_not_in_features": not old_ipv_in_features,
        "old_ipv_channels_not_in_gate_distance": not old_ipv_in_gate,
    }
    if not all(checks.values()):
        raise AssertionError(
            "feature_contract_failed "
            + json.dumps(
                {
                    "checks": checks,
                    "forbidden_in_features": forbidden_in_features,
                    "forbidden_in_gate": forbidden_in_gate,
                    "old_ipv_in_features": old_ipv_in_features,
                    "old_ipv_in_gate": old_ipv_in_gate,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return {
        "checks": checks,
        "numeric_context": list(ref.M2_NUMERIC_CONTEXT),
        "categorical_context": list(H2_CATEGORICAL_CONTEXT),
        "support_gate_categorical": list(H2_SUPPORT_CATEGORICAL),
        "support_gate_distance_numeric": list(ref.GATE_DISTANCE_NUMERIC),
        "filter_only_columns_not_predictors": list(FILTER_ONLY_COLUMNS),
        "audit_only_columns_not_predictors": list(AUDIT_ONLY_COLUMNS),
        "source_dataset_used_as_predictor": False,
    }


def old_contract_negative_control() -> Dict[str, str]:
    mutated_feature_columns = set(ref.M2_NUMERIC_CONTEXT) | set(H2_CATEGORICAL_CONTEXT + ["agent_type_pair"])
    try:
        forbidden = sorted(mutated_feature_columns & FORBIDDEN_FEATURE_COLUMNS)
        if forbidden:
            raise AssertionError(f"feature_contract_failed forbidden_in_features={forbidden}")
    except AssertionError as exc:
        return {"status": "EXPECTED_FAIL", "failure_output": str(exc)}
    return {"status": "UNEXPECTED_PASS", "failure_output": ""}


def validate_counts(frames: Mapping[str, pd.DataFrame], join_diag: Mapping[str, Any]) -> Dict[str, Any]:
    fold_human = {fold: int(len(frames[fold])) for fold in ref.FOLDS}
    fold_status_ok = {
        fold: int(join_diag["fold_counts"][fold]["status_ok_rows"])
        for fold in ref.FOLDS
    }
    all_human = pd.concat([frames[fold][["rq007_split", "agent_type_pair", "av_included"]] for fold in ref.FOLDS])
    split_counts = {str(k): int(v) for k, v in all_human["rq007_split"].value_counts().sort_index().items()}
    invalid_split_rows = int((~all_human["rq007_split"].isin(["development", "guard"])).sum())
    agent_type_counts = {str(k): int(v) for k, v in all_human["agent_type_pair"].value_counts(dropna=False).items()}
    av_included_counts = {str(k): int(v) for k, v in all_human["av_included"].value_counts(dropna=False).items()}
    mismatches = {
        "fold_human": fold_human != EXPECTED_HUMAN_FOLD_ROWS,
        "fold_status_ok": fold_status_ok != EXPECTED_STATUS_OK_FOLD_ROWS,
        "split_counts": split_counts != EXPECTED_HUMAN_SPLIT_ROWS,
        "total": int(len(all_human)) != EXPECTED_HUMAN_TOTAL_ROWS,
        "invalid_split_rows": invalid_split_rows != 0,
        "agent_type_constant": agent_type_counts != {HUMAN_PAIR: EXPECTED_HUMAN_TOTAL_ROWS},
        "av_included_constant": av_included_counts != {HUMAN_AV_INCLUDED: EXPECTED_HUMAN_TOTAL_ROWS},
    }
    if any(mismatches.values()):
        raise AssertionError(
            "reference_pool_count_failed "
            + json.dumps(
                {
                    "mismatches": mismatches,
                    "fold_human": fold_human,
                    "expected_fold_human": EXPECTED_HUMAN_FOLD_ROWS,
                    "fold_status_ok": fold_status_ok,
                    "expected_fold_status_ok": EXPECTED_STATUS_OK_FOLD_ROWS,
                    "split_counts": split_counts,
                    "expected_split_counts": EXPECTED_HUMAN_SPLIT_ROWS,
                    "total": int(len(all_human)),
                    "expected_total": EXPECTED_HUMAN_TOTAL_ROWS,
                    "invalid_split_rows": invalid_split_rows,
                    "agent_type_counts": agent_type_counts,
                    "av_included_counts": av_included_counts,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return {
        "fold_human_rows": fold_human,
        "fold_status_ok_rows": fold_status_ok,
        "split_counts": split_counts,
        "total_rows": int(len(all_human)),
        "invalid_split_rows": invalid_split_rows,
        "agent_type_pair_counts": agent_type_counts,
        "av_included_counts": av_included_counts,
        "human_share_of_status_ok_by_fold": {
            fold: pct(fold_human[fold], fold_status_ok[fold]) for fold in EXPECTED_HUMAN_FOLD_ROWS
        },
    }


def joint_cell_series(frame: pd.DataFrame, columns: Sequence[str] = H2_SUPPORT_CATEGORICAL) -> pd.Series:
    cat = ref.categorical_frame(frame, columns)
    values = cat.iloc[:, 0].astype(str)
    for column in columns[1:]:
        values = values + "|" + cat[column].astype(str)
    return values


def context_support(frames: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
    all_frame = pd.concat([frames[fold] for fold in ref.FOLDS], ignore_index=True)
    train_frame = frames["train"]

    def counts_for(frame: pd.DataFrame) -> pd.DataFrame:
        return (
            pd.DataFrame({"context_cell": joint_cell_series(frame), "case_key": frame["case_key"]})
            .groupby("context_cell")
            .agg(rows=("case_key", "size"), cases=("case_key", "nunique"))
            .reset_index()
            .sort_values("context_cell")
        )

    all_counts = counts_for(all_frame)
    train_counts = counts_for(train_frame)
    if int(len(all_counts)) != EXPECTED_CONTEXT_CELLS:
        raise AssertionError(f"context_cell_count_failed cells={len(all_counts)} expected={EXPECTED_CONTEXT_CELLS}")
    return {
        "context_columns": list(H2_SUPPORT_CATEGORICAL),
        "all_rows": {
            "cells": int(len(all_counts)),
            "min_rows": int(all_counts["rows"].min()),
            "counts": all_counts.to_dict(orient="records"),
        },
        "train_rows_for_gate_fit": {
            "cells": int(len(train_counts)),
            "min_rows": int(train_counts["rows"].min()),
            "counts": train_counts.to_dict(orient="records"),
        },
    }


def compute_cell_radii(
    calibration_frame: pd.DataFrame,
    q_pred: np.ndarray,
    y: np.ndarray,
    gate_ok: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    cells = joint_cell_series(calibration_frame).to_numpy()
    result: Dict[str, Dict[str, Any]] = {}
    for cell in sorted(set(cells.astype(str))):
        cell_mask = (cells.astype(str) == cell) & gate_ok
        result[cell] = {"calibration_gate_passing_rows": int(cell_mask.sum())}
        for alpha in ref.ALPHAS:
            label = ref.ALPHA_LABEL[alpha]
            q_lo_level, q_hi_level = ref.QUANTILE_BY_ALPHA[alpha]
            q_lo = q_pred[:, ref.Q_INDEX[q_lo_level]]
            q_hi = q_pred[:, ref.Q_INDEX[q_hi_level]]
            scores = np.maximum(q_lo[cell_mask] - y[cell_mask], y[cell_mask] - q_hi[cell_mask])
            radius, n, rank = ref.conformal_radius(scores, alpha)
            result[cell][label] = {
                "alpha": float(alpha),
                "nominal": float(ref.NOMINAL_BY_ALPHA[alpha]),
                "c_alpha": float(radius),
                "calibration_n": int(n),
                "rank": int(rank),
                "used_for_primary_scoring": False,
            }
    return result


def build_score_table_from_predictions(
    frame: pd.DataFrame,
    q_pred: np.ndarray,
    radii: Mapping[str, Mapping[str, Any]],
    gate_ok: np.ndarray,
) -> pd.DataFrame:
    if "product_row_key" not in frame.columns:
        frame = frame.copy()
        frame["product_row_key"] = ref.product_row_key(frame)
    out = frame[["product_row_key"] + [c for c in H2_SUPPORT_CATEGORICAL if c in frame.columns]].copy()
    out["context_cell"] = joint_cell_series(frame)
    out["mechanism2_gate_ok"] = gate_ok.astype(bool)
    if ref.TARGET_COLUMN in frame.columns:
        out[ref.TARGET_COLUMN] = frame[ref.TARGET_COLUMN].to_numpy(dtype=np.float64, copy=False)
    for alpha in ref.ALPHAS:
        label = ref.ALPHA_LABEL[alpha]
        q_lo_level, q_hi_level = ref.QUANTILE_BY_ALPHA[alpha]
        q_lo = q_pred[:, ref.Q_INDEX[q_lo_level]].astype(np.float64, copy=False)
        q_hi = q_pred[:, ref.Q_INDEX[q_hi_level]].astype(np.float64, copy=False)
        lo, hi = ref.calibrated_bounds(q_lo, q_hi, float(radii[label]["c_alpha"]))
        out[f"lo_{label}"] = lo
        out[f"hi_{label}"] = hi
        if ref.TARGET_COLUMN in frame.columns:
            y = frame[ref.TARGET_COLUMN].to_numpy(dtype=np.float64, copy=False)
            verdict = np.where(~gate_ok, "abstain", np.where((y >= lo) & (y <= hi), "support", "not_support"))
            out[f"verdict_{label}"] = verdict
    return out


def category_support_mask(frame: pd.DataFrame, artifact: Mapping[str, Any]) -> np.ndarray:
    gate = artifact["gate"]
    support_levels = gate["support_levels"]
    supported_joint_cells = set(gate["supported_joint_cells"])
    mask = np.ones(len(frame), dtype=bool)
    for column in H2_SUPPORT_CATEGORICAL:
        values = frame[column].astype("string").fillna("__MISSING__").to_numpy()
        supported = set(support_levels[column].keys())
        mask &= np.fromiter((str(value) in supported for value in values), dtype=bool, count=len(values))
    cell_values = joint_cell_series(frame).to_numpy()
    mask &= np.fromiter((str(value) in supported_joint_cells for value in cell_values), dtype=bool, count=len(cell_values))
    return mask


def transform_gate_matrix(frame: pd.DataFrame, artifact: Mapping[str, Any]) -> np.ndarray:
    gate = artifact["gate"]
    numeric = gate["imputer"].transform(ref.numeric_frame(frame, ref.GATE_DISTANCE_NUMERIC)).astype(np.float32, copy=False)
    numeric = gate["scaler"].transform(numeric).astype(np.float32, copy=False)
    categorical = gate["encoder"].transform(ref.categorical_frame(frame, H2_SUPPORT_CATEGORICAL)).astype(np.float32, copy=False)
    return np.hstack([numeric, categorical]).astype(np.float32, copy=False)


def score_with_artifact(artifact: Mapping[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    prep = artifact["preprocessor"]
    numeric = prep["imputer"].transform(ref.numeric_frame(frame, prep["numeric"])).astype(np.float32, copy=False)
    categorical = prep["encoder"].transform(ref.categorical_frame(frame, prep["categorical"])).astype(np.float32, copy=False)
    x = np.hstack([numeric, categorical]).astype(np.float32, copy=False)
    q_pred = np.column_stack(
        [artifact["quantile_models"][str(q)].predict(x) for q in artifact["quantile_levels"]]
    ).astype(np.float32)
    q_pred = np.sort(q_pred, axis=1).astype(np.float32)

    category_ok = category_support_mask(frame, artifact)
    distance_ok = np.zeros(len(frame), dtype=bool)
    if category_ok.any():
        gate_x = transform_gate_matrix(frame, artifact)
        distances = ref.mean_knn_distance(artifact["gate"]["tree"], gate_x[category_ok], k=int(artifact["gate"]["k"]))
        distance_ok[category_ok] = distances <= float(artifact["gate"]["threshold"])
    gate_ok = category_ok & distance_ok
    return build_score_table_from_predictions(frame, q_pred, artifact["global_conformal_radii"], gate_ok)


def write_score_helper() -> None:
    helper = '''#!/usr/bin/env python3
"""Score external rows with the persisted RQ016C-H2 envelope artifact."""
from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path

import pandas as pd

RUN_SCRIPT = Path(__file__).with_name("run_rq016c_h2_human_only_envelope.py")
spec = importlib.util.spec_from_file_location("rq016c_h2_runner", RUN_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import runner: {RUN_SCRIPT}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def read_rows(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported input suffix: {suffix}")


def write_rows(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(path, index=False)
        return
    if suffix == ".csv":
        frame.to_csv(path, index=False)
        return
    raise ValueError(f"unsupported output suffix: {suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("envelope_model") / "rq016c_h2_envelope.pkl")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.model.open("rb") as handle:
        artifact = pickle.load(handle)
    rows = read_rows(args.input)
    scored = module.score_with_artifact(artifact, rows)
    write_rows(scored, args.output)
    print(f"wrote {args.output} rows={len(scored)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    path = WORK_DIR / "score_external_rows.py"
    path.write_text(helper, encoding="utf-8")
    path.chmod(0o755)


def write_howto() -> None:
    text = f"""# HOWTO score external rows with the RQ016C-H2 envelope

This directory contains the fitted human-only envelope. Scoring does not refit any model.

## Required input columns

External rows must contain the same context columns used by this envelope:

- Numeric context: `{', '.join(ref.M2_NUMERIC_CONTEXT)}`
- Categorical context: `{', '.join(H2_CATEGORICAL_CONTEXT)}`
- Support-gate columns: `{', '.join(H2_SUPPORT_CATEGORICAL)}`

Do not add `agent_type_pair`, `av_included`, or `source_dataset` as predictors. They are not part of the fitted context. If `target_ipv_future` is present, the scoring script also writes `support`, `not_support`, or `abstain` verdicts for each alpha layer. If `target_ipv_future` is absent, it writes intervals and the mechanism-two gate flag only.

## Command

```bash
~/.rq009_codex_fleet/venv/bin/python .codex-fleet/rq016c-human-only-envelope/work/H2/score_external_rows.py \\
  --model .codex-fleet/rq016c-human-only-envelope/work/H2/envelope_model/rq016c_h2_envelope.pkl \\
  --input path/to/external_rows.parquet \\
  --output path/to/scored_rows.parquet
```

The input may be `.parquet` or `.csv`; the output suffix controls the output format.

## Output columns

- `mechanism2_gate_ok`: `True` when the row passes the support gate.
- `lo_80`, `hi_80`, `lo_90`, `hi_90`, `lo_95`, `hi_95`: interval bounds from the persisted global conformal radii.
- `verdict_80`, `verdict_90`, `verdict_95`: written only when `target_ipv_future` is present. Values are `support`, `not_support`, or `abstain`.

Per-cell calibration radii are stored in `conformal_radii_by_cell.json` for audit. The primary scoring path uses `conformal_radii_global.json`, matching the RQ016 split-conformal calculation.
"""
    (MODEL_DIR / "HOWTO_score_external_rows.md").write_text(text, encoding="utf-8")


def save_model_artifacts(
    model: Any,
    gate: Any,
    global_radii: Mapping[str, Mapping[str, Any]],
    cell_radii: Mapping[str, Mapping[str, Any]],
    support: Mapping[str, Any],
    selected_params: Mapping[str, Any],
    timestamp_utc: str,
) -> Dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "version": "RQ016C-H2-human-only-envelope-v1",
        "created_utc": timestamp_utc,
        "reference_script": str(REF_SCRIPT.relative_to(REPO_ROOT)),
        "feature_contract": {
            "numeric_context": list(ref.M2_NUMERIC_CONTEXT),
            "categorical_context": list(H2_CATEGORICAL_CONTEXT),
            "support_gate_categorical": list(H2_SUPPORT_CATEGORICAL),
            "support_gate_distance_numeric": list(ref.GATE_DISTANCE_NUMERIC),
            "excluded_predictors": sorted(FORBIDDEN_FEATURE_COLUMNS),
            "excluded_old_ipv_channels": sorted(FORBIDDEN_OLD_IPV_COLUMNS),
            "target_column": ref.TARGET_COLUMN,
        },
        "selected_hgb_params": dict(selected_params),
        "quantile_levels": [float(q) for q in ref.QUANTILE_LEVELS],
        "alphas": {ref.ALPHA_LABEL[a]: float(a) for a in ref.ALPHAS},
        "global_conformal_radii": json.loads(json.dumps(global_radii)),
        "cell_conformal_radii": json.loads(json.dumps(cell_radii)),
        "preprocessor": {
            "numeric": list(model.spec.numeric),
            "categorical": list(model.spec.categorical),
            "imputer": model.preprocessor.imputer,
            "encoder": model.preprocessor.encoder,
            "categorical_mask": list(model.preprocessor.categorical_mask or []),
        },
        "quantile_models": {str(float(q)): fitted for q, fitted in model.models.items()},
        "gate": {
            "imputer": gate.imputer,
            "scaler": gate.scaler,
            "encoder": gate.encoder,
            "tree": gate.tree,
            "threshold": float(gate.threshold),
            "k": 25,
            "support_levels": gate.support_levels,
            "unsupported_levels": gate.unsupported_levels,
            "supported_joint_cells": sorted(gate.supported_joint_cells),
            "unsupported_joint_cells": list(gate.unsupported_joint_cells),
        },
    }
    with MODEL_PICKLE_PATH.open("wb") as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(MODEL_DIR / "manifest.json", {
        "version": artifact["version"],
        "created_utc": timestamp_utc,
        "model_pickle": MODEL_PICKLE_PATH.name,
        "reference_script": artifact["reference_script"],
        "primary_scoring_radii": "conformal_radii_global.json",
        "cell_radii_diagnostics": "conformal_radii_by_cell.json",
    })
    write_json(MODEL_DIR / "feature_contract.json", artifact["feature_contract"])
    write_json(MODEL_DIR / "conformal_radii_global.json", artifact["global_conformal_radii"])
    write_json(MODEL_DIR / "conformal_radii_by_cell.json", artifact["cell_conformal_radii"])
    write_json(MODEL_DIR / "support_gate.json", {
        "support_gate_categorical": list(H2_SUPPORT_CATEGORICAL),
        "support_gate_distance_numeric": list(ref.GATE_DISTANCE_NUMERIC),
        "threshold": float(gate.threshold),
        "k": 25,
        "support_levels": gate.support_levels,
        "supported_joint_cells": sorted(gate.supported_joint_cells),
        "unsupported_joint_cells": list(gate.unsupported_joint_cells),
        "support_counts": support,
    })
    counts_path = MODEL_DIR / "support_counts_by_cell.csv"
    with counts_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["context_cell", "rows", "cases"])
        writer.writeheader()
        for row in support["all_rows"]["counts"]:
            writer.writerow(row)
    write_howto()
    write_score_helper()
    return {
        "model_dir": str(MODEL_DIR.relative_to(REPO_ROOT)),
        "model_pickle": str(MODEL_PICKLE_PATH.relative_to(REPO_ROOT)),
        "files": sorted(p.name for p in MODEL_DIR.iterdir() if p.is_file()),
    }


def onsite_landing_preview(human_support: Mapping[str, Any]) -> Dict[str, Any]:
    onsite = pd.read_parquet(ONSITE_PATH, columns=H2_SUPPORT_CATEGORICAL)
    onsite["context_cell"] = joint_cell_series(onsite)
    onsite_counts = onsite.groupby("context_cell").size().reset_index(name="onsite_rows").sort_values("context_cell")
    human_counts = pd.DataFrame(human_support["all_rows"]["counts"]).rename(columns={"rows": "human_rows", "cases": "human_cases"})
    merged = onsite_counts.merge(human_counts, on="context_cell", how="left")
    missing = merged.loc[merged["human_rows"].isna(), "context_cell"].astype(str).tolist()
    min_row = merged.sort_values(["human_rows", "context_cell"]).iloc[0].to_dict()
    checks = {
        "rows": int(len(onsite)) == EXPECTED_ONSITE_ROWS,
        "cells": int(len(onsite_counts)) == EXPECTED_ONSITE_CELLS,
        "missing_cells": int(len(missing)) == EXPECTED_ONSITE_MISSING_CELLS,
        "min_support": int(min_row["human_rows"]) == EXPECTED_ONSITE_MIN_SUPPORT,
        "min_cell": str(min_row["context_cell"]) == EXPECTED_ONSITE_MIN_CELL,
        "min_cell_onsite_rows": int(min_row["onsite_rows"]) == EXPECTED_ONSITE_MIN_CELL_ROWS,
    }
    if not all(checks.values()):
        raise AssertionError(
            "onsite_landing_preview_failed "
            + json.dumps(
                {
                    "checks": checks,
                    "rows": int(len(onsite)),
                    "cells": int(len(onsite_counts)),
                    "missing_cells": missing,
                    "min_row": min_row,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return {
        "source_path": str(ONSITE_PATH.relative_to(REPO_ROOT)),
        "rows": int(len(onsite)),
        "context_columns": list(H2_SUPPORT_CATEGORICAL),
        "cells": int(len(onsite_counts)),
        "missing_cells": missing,
        "min_human_support_among_onsite_cells": int(min_row["human_rows"]),
        "min_support_cell": str(min_row["context_cell"]),
        "min_support_cell_onsite_rows": int(min_row["onsite_rows"]),
        "counts": merged.fillna({"human_rows": 0, "human_cases": 0}).astype({"human_rows": "int64", "human_cases": "int64"}).to_dict(orient="records"),
        "checks": checks,
    }


def unique_list(values: pd.Series) -> list[str]:
    return sorted(values.astype("string").fillna("__MISSING__").astype(str).unique().tolist())


def value_counts_records(values: pd.Series) -> list[Dict[str, Any]]:
    counts = values.astype("string").fillna("__MISSING__").astype(str).value_counts(dropna=False)
    return [{"value": str(value), "rows": int(rows)} for value, rows in counts.items()]


def read_onsite_context_frame(include_target: bool = False) -> pd.DataFrame:
    columns = list(
        dict.fromkeys(
            ref.KEY_COLUMNS
            + list(ref.M2_NUMERIC_CONTEXT)
            + H2_CATEGORICAL_CONTEXT
            + H2_SUPPORT_CATEGORICAL
            + AUDIT_ONLY_COLUMNS
            + FILTER_ONLY_COLUMNS
            + ([ref.TARGET_COLUMN] if include_target else [])
        )
    )
    frame = pd.read_parquet(ONSITE_PATH, columns=columns)
    if len(frame) != EXPECTED_ONSITE_ROWS:
        raise AssertionError(f"onsite_row_count_failed rows={len(frame)} expected={EXPECTED_ONSITE_ROWS}")
    return frame


def all_human_values(frames: Mapping[str, pd.DataFrame], column: str) -> pd.Series:
    return pd.concat([frames[fold][column] for fold in ref.FOLDS], ignore_index=True)


def category_vocabulary_coverage(
    frames: Mapping[str, pd.DataFrame],
    onsite: pd.DataFrame,
    categorical_context: Sequence[str],
    support_gate_columns: Sequence[str],
) -> Dict[str, Any]:
    rows: list[Dict[str, Any]] = []
    seen_by_column: Dict[str, set[str]] = {}
    for column in sorted(set(categorical_context) | set(support_gate_columns)):
        seen_by_column[column] = set(unique_list(all_human_values(frames, column)))

    for scope, columns in [
        ("categorical_context", list(categorical_context)),
        ("support_gate_key", list(support_gate_columns)),
    ]:
        for column in columns:
            onsite_values = onsite[column].astype("string").fillna("__MISSING__").astype(str)
            seen = seen_by_column[column]
            hit_mask = onsite_values.isin(seen)
            unmatched_values = sorted(set(onsite_values.unique().tolist()) - seen)
            rows.append(
                {
                    "scope": scope,
                    "column": column,
                    "matched_rows": int(hit_mask.sum()),
                    "total_rows": int(len(onsite_values)),
                    "coverage": pct(int(hit_mask.sum()), int(len(onsite_values))),
                    "reference_seen_values": sorted(seen),
                    "onsite_values": unique_list(onsite_values),
                    "onsite_value_counts": value_counts_records(onsite_values),
                    "unmatched_values": unmatched_values,
                    "pass": bool(int(hit_mask.sum()) == int(len(onsite_values)) and not unmatched_values),
                }
            )
    failures = [row for row in rows if not row["pass"]]
    result = {
        "source_reference_pool": {
            "path": str(ref.MATRIX_ROOT.relative_to(REPO_ROOT)),
            "filter": "K2 ledger joined rows with status == OK and agent_type_pair == HV;HV across train/calibration/guard_tune/test",
            "columns": list(dict.fromkeys(list(categorical_context) + list(support_gate_columns))),
            "rows": EXPECTED_HUMAN_TOTAL_ROWS,
        },
        "source_onsite": {
            "path": str(ONSITE_PATH.relative_to(REPO_ROOT)),
            "filter": "all rows",
            "columns": list(dict.fromkeys(list(categorical_context) + list(support_gate_columns))),
            "rows": int(len(onsite)),
        },
        "rows": rows,
        "all_pass": bool(not failures),
        "failures": failures,
    }
    if failures:
        first = failures[0]
        raise AssertionError(
            "vocabulary_coverage_failed "
            + json.dumps(
                {
                    "scope": first["scope"],
                    "column": first["column"],
                    "matched_rows": first["matched_rows"],
                    "total_rows": first["total_rows"],
                    "unmatched_values": first["unmatched_values"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return result


def numeric_range_comparison(frames: Mapping[str, pd.DataFrame], onsite: pd.DataFrame) -> Dict[str, Any]:
    quantiles = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    rows: list[Dict[str, Any]] = []
    completely_outside: list[str] = []
    for column in ref.M2_NUMERIC_CONTEXT:
        human = pd.to_numeric(all_human_values(frames, column), errors="coerce").to_numpy(dtype=np.float64)
        on = pd.to_numeric(onsite[column], errors="coerce").to_numpy(dtype=np.float64)
        human = human[np.isfinite(human)]
        on = on[np.isfinite(on)]
        human_q = np.quantile(human, quantiles) if len(human) else np.full(len(quantiles), np.nan)
        onsite_q = np.quantile(on, quantiles) if len(on) else np.full(len(quantiles), np.nan)
        human_min = float(human_q[0])
        human_max = float(human_q[-1])
        onsite_min = float(onsite_q[0])
        onsite_max = float(onsite_q[-1])
        outside = bool(onsite_max < human_min or onsite_min > human_max)
        if outside:
            completely_outside.append(column)
        rows.append(
            {
                "column": column,
                "human_finite_rows": int(len(human)),
                "onsite_finite_rows": int(len(on)),
                "human_min": human_min,
                "human_max": human_max,
                "human_quantiles": {str(q): float(v) for q, v in zip(quantiles, human_q)},
                "onsite_min": onsite_min,
                "onsite_max": onsite_max,
                "onsite_quantiles": {str(q): float(v) for q, v in zip(quantiles, onsite_q)},
                "onsite_completely_outside_human_minmax": outside,
                "onsite_has_values_below_human_min": bool(onsite_min < human_min),
                "onsite_has_values_above_human_max": bool(onsite_max > human_max),
            }
        )
    return {
        "source_reference_pool": {
            "path": str(ref.MATRIX_ROOT.relative_to(REPO_ROOT)),
            "filter": "K2 ledger joined rows with status == OK and agent_type_pair == HV;HV across train/calibration/guard_tune/test",
            "columns": list(ref.M2_NUMERIC_CONTEXT),
            "rows": EXPECTED_HUMAN_TOTAL_ROWS,
        },
        "source_onsite": {
            "path": str(ONSITE_PATH.relative_to(REPO_ROOT)),
            "filter": "all rows",
            "columns": list(ref.M2_NUMERIC_CONTEXT),
            "rows": int(len(onsite)),
        },
        "quantiles": quantiles,
        "rows": rows,
        "onsite_completely_outside_features": completely_outside,
    }


def onsite_landing_for_columns(
    frames: Mapping[str, pd.DataFrame],
    onsite: pd.DataFrame,
    support_columns: Sequence[str],
    enforce_expected: bool,
) -> Dict[str, Any]:
    human = pd.concat(
        [frames[fold][list(dict.fromkeys(list(support_columns) + ["case_key"]))] for fold in ref.FOLDS],
        ignore_index=True,
    )
    human["context_cell"] = joint_cell_series(human, support_columns)
    onsite_cells = onsite[list(support_columns)].copy()
    onsite_cells["context_cell"] = joint_cell_series(onsite_cells, support_columns)
    human_counts = (
        human.groupby("context_cell")
        .agg(human_rows=("case_key", "size"), human_cases=("case_key", "nunique"))
        .reset_index()
        .sort_values("context_cell")
    )
    onsite_counts = onsite_cells.groupby("context_cell").size().reset_index(name="onsite_rows").sort_values("context_cell")
    merged = onsite_counts.merge(human_counts, on="context_cell", how="left")
    missing_rows = merged.loc[merged["human_rows"].isna(), "onsite_rows"].sum()
    missing_cells = merged.loc[merged["human_rows"].isna(), "context_cell"].astype(str).tolist()
    filled = merged.fillna({"human_rows": 0, "human_cases": 0}).astype({"human_rows": "int64", "human_cases": "int64"})
    supported = filled.loc[filled["human_rows"] > 0]
    min_row = supported.sort_values(["human_rows", "context_cell"]).iloc[0].to_dict() if len(supported) else {}
    checks = {
        "rows": int(len(onsite)) == EXPECTED_ONSITE_ROWS,
        "cells": int(len(onsite_counts)) == EXPECTED_ONSITE_CELLS,
        "missing_cells": int(len(missing_cells)) == EXPECTED_ONSITE_MISSING_CELLS,
        "min_support": bool(min_row) and int(min_row["human_rows"]) == EXPECTED_ONSITE_MIN_SUPPORT,
        "min_cell": bool(min_row) and str(min_row["context_cell"]) == EXPECTED_ONSITE_MIN_CELL,
        "min_cell_onsite_rows": bool(min_row) and int(min_row["onsite_rows"]) == EXPECTED_ONSITE_MIN_CELL_ROWS,
    }
    result = {
        "source_reference_pool": {
            "path": str(ref.MATRIX_ROOT.relative_to(REPO_ROOT)),
            "filter": "K2 ledger joined rows with status == OK and agent_type_pair == HV;HV across train/calibration/guard_tune/test",
            "columns": list(support_columns) + ["case_key"],
            "rows": EXPECTED_HUMAN_TOTAL_ROWS,
        },
        "source_onsite": {
            "path": str(ONSITE_PATH.relative_to(REPO_ROOT)),
            "filter": "all rows",
            "columns": list(support_columns),
            "rows": int(len(onsite)),
        },
        "support_columns": list(support_columns),
        "rows": int(len(onsite)),
        "cells": int(len(onsite_counts)),
        "missing_cells": missing_cells,
        "missing_rows": int(missing_rows),
        "min_human_support_among_onsite_cells": int(min_row["human_rows"]) if min_row else 0,
        "min_support_cell": str(min_row["context_cell"]) if min_row else None,
        "min_support_cell_onsite_rows": int(min_row["onsite_rows"]) if min_row else 0,
        "counts": filled.to_dict(orient="records"),
        "checks": checks,
    }
    if enforce_expected and not all(checks.values()):
        raise AssertionError(
            "onsite_landing_check_failed "
            + json.dumps(
                {
                    "support_columns": list(support_columns),
                    "rows": int(len(onsite)),
                    "cells": int(len(onsite_counts)),
                    "missing_cells": missing_cells,
                    "missing_rows": int(missing_rows),
                    "min_row": min_row,
                    "checks": checks,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return result


def negative_controls(frames: Mapping[str, pd.DataFrame], onsite: pd.DataFrame) -> Dict[str, Any]:
    controls: Dict[str, Any] = {}
    try:
        category_vocabulary_coverage(
            frames,
            onsite,
            H2_CATEGORICAL_CONTEXT + ["vehicle_type_list"],
            H2_SUPPORT_CATEGORICAL,
        )
    except AssertionError as exc:
        controls["vehicle_type_list_back_in_context"] = {
            "status": "EXPECTED_FAIL",
            "failure_output": str(exc),
        }
    else:
        controls["vehicle_type_list_back_in_context"] = {"status": "UNEXPECTED_PASS", "failure_output": ""}

    try:
        onsite_landing_for_columns(
            frames,
            onsite,
            H2_SUPPORT_CATEGORICAL + ["agent_type_pair"],
            enforce_expected=True,
        )
    except AssertionError as exc:
        controls["agent_type_pair_back_in_support_gate"] = {
            "status": "EXPECTED_FAIL",
            "failure_output": str(exc),
        }
    else:
        controls["agent_type_pair_back_in_support_gate"] = {"status": "UNEXPECTED_PASS", "failure_output": ""}

    bad = {name: row for name, row in controls.items() if row["status"] != "EXPECTED_FAIL"}
    if bad:
        raise AssertionError(f"negative_control_not_failing {json.dumps(bad, ensure_ascii=False, sort_keys=True)}")
    return controls


def summarize_numeric(values: pd.Series) -> Dict[str, Any]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return {"finite_rows": 0}
    return {
        "finite_rows": int(len(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
    }


def mechanism1_boundary_check() -> Dict[str, Any]:
    dataset = ds.dataset(ref.K2_LEDGER_ROOT, format="parquet", partitioning="hive")
    mse_columns = [f"mse_{idx}" for idx in range(7)]
    columns = ["artifact_id", "status", "reason_code"] + mse_columns
    table = dataset.to_table(columns=columns, filter=ds.field("artifact_id") == "onsite_dense_timeseries")
    frame = table.to_pandas()
    status_nonnull = int(frame["status"].notna().sum())
    reason_nonnull = int(frame["reason_code"].notna().sum())
    mse_nonnull = {column: int(frame[column].notna().sum()) for column in mse_columns}
    checks = {
        "rows": int(len(frame)) == 281_268,
        "status_nonnull_zero": status_nonnull == 0,
        "reason_code_nonnull_zero": reason_nonnull == 0,
        "all_mse_nonnull_zero": all(value == 0 for value in mse_nonnull.values()),
    }
    if not all(checks.values()):
        raise AssertionError(
            "onsite_mechanism1_boundary_failed "
            + json.dumps(
                {
                    "rows": int(len(frame)),
                    "status_nonnull": status_nonnull,
                    "reason_code_nonnull": reason_nonnull,
                    "mse_nonnull": mse_nonnull,
                    "checks": checks,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return {
        "source_path": str(ref.K2_LEDGER_ROOT.relative_to(REPO_ROOT)),
        "filter": "artifact_id == onsite_dense_timeseries",
        "rows": int(len(frame)),
        "status_nonnull": status_nonnull,
        "reason_code_nonnull": reason_nonnull,
        "mse_nonnull": mse_nonnull,
        "source_columns": ["artifact_id", "status", "reason_code"] + mse_columns,
        "checks": checks,
    }


def score_onsite_dryrun() -> Dict[str, Any]:
    with MODEL_PICKLE_PATH.open("rb") as handle:
        artifact = pickle.load(handle)
    rows = read_onsite_context_frame(include_target=False)
    scored = score_with_artifact(artifact, rows)
    for label in ["80", "90", "95"]:
        scored[f"width_{label}"] = scored[f"hi_{label}"] - scored[f"lo_{label}"]
    output_path = WORK_DIR / "onsite_scoring_dryrun.parquet"
    summary_path = WORK_DIR / "onsite_scoring_dryrun_summary.json"
    ordered = [
        "product_row_key",
        "context_cell",
        *H2_SUPPORT_CATEGORICAL,
        "mechanism2_gate_ok",
        "lo_80",
        "hi_80",
        "width_80",
        "lo_90",
        "hi_90",
        "width_90",
        "lo_95",
        "hi_95",
        "width_95",
    ]
    scored = scored.loc[:, ordered]
    scored.to_parquet(output_path, index=False)
    pass_rows = int(scored["mechanism2_gate_ok"].sum())
    per_cell = (
        scored.groupby("context_cell")
        .agg(rows=("mechanism2_gate_ok", "size"), pass_rows=("mechanism2_gate_ok", "sum"))
        .reset_index()
        .sort_values("context_cell")
    )
    per_cell["fail_rows"] = per_cell["rows"] - per_cell["pass_rows"]
    per_cell["pass_rate"] = per_cell["pass_rows"] / per_cell["rows"]
    width_summary = {}
    for label in ["80", "90", "95"]:
        width_summary[label] = {
            "all_rows": summarize_numeric(scored[f"width_{label}"]),
            "gate_passing_rows": summarize_numeric(scored.loc[scored["mechanism2_gate_ok"], f"width_{label}"]),
        }
    summary = {
        "source_path": str(ONSITE_PATH.relative_to(REPO_ROOT)),
        "source_filter": "all rows from onsite_m3_av_anchors_multi_allvalid.parquet; target_ipv_future intentionally not loaded",
        "source_columns": list(
            dict.fromkeys(ref.KEY_COLUMNS + list(ref.M2_NUMERIC_CONTEXT) + H2_CATEGORICAL_CONTEXT + H2_SUPPORT_CATEGORICAL)
        ),
        "model_path": str(MODEL_PICKLE_PATH.relative_to(REPO_ROOT)),
        "output_path": str(output_path.relative_to(REPO_ROOT)),
        "rows": int(len(scored)),
        "loaded_persisted_model_only": True,
        "did_not_refit": True,
        "mechanism2_gate_pass_rows": pass_rows,
        "mechanism2_gate_fail_rows": int(len(scored) - pass_rows),
        "mechanism2_gate_pass_rate": pct(pass_rows, int(len(scored))),
        "per_cell": per_cell.to_dict(orient="records"),
        "width_summary": width_summary,
    }
    write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path.relative_to(REPO_ROOT))
    return summary


def numeric_health(
    frames: Mapping[str, pd.DataFrame],
    metrics: Mapping[str, Mapping[str, Any]],
    prediction_health: Mapping[str, Any],
) -> Dict[str, Any]:
    target_counts: Dict[str, Any] = {}
    for fold, frame in frames.items():
        y = frame[ref.TARGET_COLUMN].to_numpy(dtype=np.float64, copy=False)
        target_counts[fold] = {
            "rows": int(len(frame)),
            "target_nan": int(np.isnan(y).sum()),
            "target_pos_inf": int(np.isposinf(y).sum()),
            "target_neg_inf": int(np.isneginf(y).sum()),
        }
    return {
        "target_numeric_counts": target_counts,
        "m2_numeric_feature_counts": {fold: ref.frame_numeric_health(frame, ref.M2_NUMERIC_CONTEXT) for fold, frame in frames.items()},
        "coverage_in_0_1": {label: bool(0.0 <= float(row["coverage"]) <= 1.0) for label, row in metrics.items()},
        "negative_width_rows": {label: int(row["negative_width_rows"]) for label, row in metrics.items()},
        "prediction_health": prediction_health,
    }


def compare_to_previous_b(metrics: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    previous = json.loads(PREVIOUS_KEY_NUMBERS.read_text(encoding="utf-8"))
    previous_b = previous["arms"]["B_status_ok"]["metrics"]
    out: Dict[str, Any] = {
        "source_path": str(PREVIOUS_KEY_NUMBERS.relative_to(REPO_ROOT)),
        "previous_arm": "B_status_ok",
        "note": "human_only_minus_previous_B; previous B contains status OK rows from all agent_type_pair values",
        "by_alpha": {},
    }
    for label in ["80", "90", "95"]:
        now = metrics[label]
        old = previous_b[label]
        out["by_alpha"][label] = {
            "coverage_human_only": float(now["coverage"]),
            "coverage_previous_B": float(old["coverage"]),
            "coverage_delta_human_minus_previous_B": float(now["coverage"] - old["coverage"]),
            "covered_human_only": int(now["covered_n"]),
            "n_human_only": int(now["n"]),
            "covered_previous_B": int(old["covered_n"]),
            "n_previous_B": int(old["n"]),
            "mean_width_human_only": float(now["mean_width"]),
            "mean_width_previous_B": float(old["mean_width"]),
            "mean_width_delta_human_minus_previous_B": float(now["mean_width"] - old["mean_width"]),
            "mean_width_pct_human_vs_previous_B": float((now["mean_width"] / old["mean_width"] - 1.0) * 100.0),
            "abstention_human_only": float(now["abstention"]),
            "abstention_previous_B": float(old["abstention"]),
            "abstention_delta_human_minus_previous_B": float(now["abstention"] - old["abstention"]),
            "abstained_human_only": int(now["abstained_n"]),
            "total_human_only": int(now["total_n"]),
            "abstained_previous_B": int(old["abstained_n"]),
            "total_previous_B": int(old["total_n"]),
        }
    return out


def compare_to_h1(metrics: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    h1 = json.loads(H1_KEY_NUMBERS_PATH.read_text(encoding="utf-8"))
    h1_metrics = h1["human_only_envelope"]["metrics"]
    out: Dict[str, Any] = {
        "source_path": str(H1_KEY_NUMBERS_PATH.relative_to(REPO_ROOT)),
        "note": "H2_minus_H1; both use the same pure-human row filter, support gate key, distance features, folds, alphas, conformal calculation, and random state; the feature-contract difference is removing vehicle_type_list from categorical_context",
        "h1_categorical_context": h1["feature_contract"]["categorical_context"],
        "h2_categorical_context": list(H2_CATEGORICAL_CONTEXT),
        "removed_from_h1": sorted(set(h1["feature_contract"]["categorical_context"]) - set(H2_CATEGORICAL_CONTEXT)),
        "added_in_h2": sorted(set(H2_CATEGORICAL_CONTEXT) - set(h1["feature_contract"]["categorical_context"])),
        "by_alpha": {},
    }
    for label in ["80", "90", "95"]:
        now = metrics[label]
        old = h1_metrics[label]
        out["by_alpha"][label] = {
            "coverage_h2": float(now["coverage"]),
            "coverage_h1": float(old["coverage"]),
            "coverage_delta_h2_minus_h1": float(now["coverage"] - old["coverage"]),
            "covered_h2": int(now["covered_n"]),
            "n_h2": int(now["n"]),
            "covered_h1": int(old["covered_n"]),
            "n_h1": int(old["n"]),
            "mean_width_h2": float(now["mean_width"]),
            "mean_width_h1": float(old["mean_width"]),
            "mean_width_delta_h2_minus_h1": float(now["mean_width"] - old["mean_width"]),
            "median_width_h2": float(now["median_width"]),
            "median_width_h1": float(old["median_width"]),
            "median_width_delta_h2_minus_h1": float(now["median_width"] - old["median_width"]),
            "abstention_h2": float(now["abstention"]),
            "abstention_h1": float(old["abstention"]),
            "abstention_delta_h2_minus_h1": float(now["abstention"] - old["abstention"]),
            "abstained_h2": int(now["abstained_n"]),
            "total_h2": int(now["total_n"]),
            "abstained_h1": int(old["abstained_n"]),
            "total_h1": int(old["total_n"]),
        }
    return out


def run_human_envelope(
    frames: Mapping[str, pd.DataFrame],
    selected_params: Mapping[str, Any],
    random_state: int,
    timestamp_utc: str,
) -> Dict[str, Any]:
    start = time.time()
    print(f"[{utc_now()}] fit support gate", flush=True)
    gate, gate_payload, _, guard_diag = ref.fit_gate(frames["train"], frames["guard_tune"])
    cal_gate_ok, cal_gate_diag = ref.apply_gate(frames["calibration"], gate)
    test_gate_ok, test_gate_diag = ref.apply_gate(frames["test"], gate)
    print(f"[{utc_now()}] fit M2 quantile model", flush=True)
    model = ref.fit_tier_model(frames["train"], selected_params, random_state)
    print(f"[{utc_now()}] predict calibration/test", flush=True)
    q_cal, cal_pred_health = ref.predict_quantiles(model, frames["calibration"])
    y_cal = frames["calibration"][ref.TARGET_COLUMN].to_numpy(dtype=np.float32, copy=False)
    radii = ref.compute_radii(q_cal, y_cal, cal_gate_ok)
    cell_radii = compute_cell_radii(frames["calibration"], q_cal, y_cal, cal_gate_ok)
    del q_cal, y_cal
    gc.collect()
    q_test, test_pred_health = ref.predict_quantiles(model, frames["test"])
    metrics = ref.score_test_frame(frames["test"], q_test, radii, test_gate_ok)
    support = context_support(frames)
    model_artifacts = save_model_artifacts(model, gate, radii, cell_radii, support, selected_params, timestamp_utc)

    sample_indices = np.arange(min(256, len(frames["test"])))
    sample_frame = frames["test"].iloc[sample_indices].copy()
    main_sample = build_score_table_from_predictions(sample_frame, q_test[sample_indices], radii, test_gate_ok[sample_indices])
    with MODEL_PICKLE_PATH.open("rb") as handle:
        loaded_artifact = pickle.load(handle)
    loaded_sample = score_with_artifact(loaded_artifact, sample_frame)
    max_abs_diff = 0.0
    bitwise_equal = True
    compared_columns = []
    for label in ["80", "90", "95"]:
        for side in ["lo", "hi"]:
            column = f"{side}_{label}"
            compared_columns.append(column)
            a = main_sample[column].to_numpy(dtype=np.float64)
            b = loaded_sample[column].to_numpy(dtype=np.float64)
            if not np.array_equal(a, b):
                bitwise_equal = False
                max_abs_diff = max(max_abs_diff, float(np.max(np.abs(a - b))))
    gate_equal = bool(np.array_equal(main_sample["mechanism2_gate_ok"].to_numpy(), loaded_sample["mechanism2_gate_ok"].to_numpy()))
    verdict_equal = True
    for label in ["80", "90", "95"]:
        column = f"verdict_{label}"
        verdict_equal &= bool(np.array_equal(main_sample[column].astype(str).to_numpy(), loaded_sample[column].astype(str).to_numpy()))
    if not (bitwise_equal and gate_equal and verdict_equal):
        raise AssertionError(
            f"persisted_scoring_selftest_failed bitwise_equal={bitwise_equal} gate_equal={gate_equal} verdict_equal={verdict_equal} max_abs_diff={max_abs_diff}"
        )
    sample_frame.to_parquet(WORK_DIR / "selftest_sample_rows.parquet", index=False)
    main_sample.to_parquet(WORK_DIR / "selftest_main_scores.parquet", index=False)
    loaded_sample.to_parquet(WORK_DIR / "selftest_loaded_scores.parquet", index=False)

    del q_test, model
    gc.collect()
    return {
        "elapsed_s": round(time.time() - start, 3),
        "row_counts": {fold: int(len(frame)) for fold, frame in frames.items()},
        "gate": {
            "fit": gate_payload,
            "guard_tune": guard_diag,
            "calibration": cal_gate_diag,
            "test": test_gate_diag,
        },
        "conformal_radii": radii,
        "conformal_radii_by_cell": cell_radii,
        "metrics": metrics,
        "context_support": support,
        "prediction_health": {"calibration": cal_pred_health, "test": test_pred_health},
        "model_artifacts": model_artifacts,
        "persisted_scoring_selftest": {
            "sample_rows": int(len(sample_frame)),
            "bitwise_equal_interval_bounds": bool(bitwise_equal),
            "gate_equal": bool(gate_equal),
            "verdict_equal": bool(verdict_equal),
            "max_abs_interval_diff": float(max_abs_diff),
            "compared_columns": compared_columns,
            "sample_rows_path": str((WORK_DIR / "selftest_sample_rows.parquet").relative_to(REPO_ROOT)),
            "main_scores_path": str((WORK_DIR / "selftest_main_scores.parquet").relative_to(REPO_ROOT)),
            "loaded_scores_path": str((WORK_DIR / "selftest_loaded_scores.parquet").relative_to(REPO_ROOT)),
        },
    }


def metric_table(metrics: Mapping[str, Mapping[str, Any]], source_k2: str, source_matrix: str) -> str:
    lines = [
        "| alpha | coverage | covered / gate-passing rows | mean width | median width | mechanism-two abstention |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in ["80", "90", "95"]:
        row = metrics[label]
        lines.append(
            f"| {label} | {float(row['coverage']):.6f} | {int(row['covered_n']):,}/{int(row['n']):,} | "
            f"{float(row['mean_width']):.6f} | {float(row['median_width']):.6f} | "
            f"{pct_percent(int(row['abstained_n']), int(row['total_n']))} "
            f"({int(row['abstained_n']):,}/{int(row['total_n']):,}; filter=纯人-人 test fold; "
            f"source={source_k2} + {source_matrix}; columns=status/rq007_split/fold/agent_type_pair) |"
        )
    return "\n".join(lines)


def h1_comparison_table(comparison: Mapping[str, Any]) -> str:
    lines = [
        "| alpha | H2 coverage / H1 coverage | delta | H2 mean width / H1 mean width | H2 median width / H1 median width | H2 abstention / H1 abstention |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in ["80", "90", "95"]:
        row = comparison["by_alpha"][label]
        lines.append(
            f"| {label} | {float(row['coverage_h2']):.6f} / {float(row['coverage_h1']):.6f} | "
            f"{float(row['coverage_delta_h2_minus_h1']):+.6f} | "
            f"{float(row['mean_width_h2']):.6f} / {float(row['mean_width_h1']):.6f} | "
            f"{float(row['median_width_h2']):.6f} / {float(row['median_width_h1']):.6f} | "
            f"{float(row['abstention_h2']):.6f} / {float(row['abstention_h1']):.6f} |"
        )
    return "\n".join(lines)


def support_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = ["| context cell | rows | cases |", "|---|---:|---:|"]
    for row in rows:
        lines.append(f"| `{row['context_cell']}` | {int(row['rows']):,} | {int(row['cases']):,} |")
    return "\n".join(lines)


def onsite_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = ["| OnSite context cell | OnSite rows | human support rows | human support cases |", "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| `{row['context_cell']}` | {int(row['onsite_rows']):,} | {int(row['human_rows']):,} | {int(row['human_cases']):,} |"
        )
    return "\n".join(lines)


def vocabulary_table(rows: Sequence[Mapping[str, Any]], reference_path: str, onsite_path: str) -> str:
    lines = [
        "| scope | column | hit rows | unmatched OnSite values |",
        "|---|---|---:|---|",
    ]
    for row in rows:
        hit = int(row["matched_rows"])
        total = int(row["total_rows"])
        lines.append(
            f"| {row['scope']} | `{row['column']}` | "
            f"{pct_percent(hit, total)} ({hit:,}/{total:,}; filter=all OnSite rows; "
            f"source={onsite_path}; column={row['column']}; reference={reference_path}; "
            f"reference_filter=K2 status OK and agent_type_pair HV;HV) | "
            f"`{row['unmatched_values']}` |"
        )
    return "\n".join(lines)


def numeric_range_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| numeric context | human min | human p50 | human max | OnSite min | OnSite p50 | OnSite max | complete outside? |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['column']}` | {float(row['human_min']):.6g} | {float(row['human_quantiles']['0.5']):.6g} | "
            f"{float(row['human_max']):.6g} | {float(row['onsite_min']):.6g} | "
            f"{float(row['onsite_quantiles']['0.5']):.6g} | {float(row['onsite_max']):.6g} | "
            f"{row['onsite_completely_outside_human_minmax']} |"
        )
    return "\n".join(lines)


def dryrun_cell_table(rows: Sequence[Mapping[str, Any]], output_path: str) -> str:
    lines = [
        "| context cell | pass rows | fail rows | pass rate |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        passed = int(row["pass_rows"])
        total = int(row["rows"])
        failed = int(row["fail_rows"])
        lines.append(
            f"| `{row['context_cell']}` | {passed:,} | {failed:,} | "
            f"{pct_percent(passed, total)} ({passed:,}/{total:,}; filter=context_cell == {row['context_cell']}; "
            f"source={output_path}; column=mechanism2_gate_ok) |"
        )
    return "\n".join(lines)


def width_summary_table(width_summary: Mapping[str, Any]) -> str:
    lines = [
        "| alpha | rows used | min | p05 | p50 | p95 | mean | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ["80", "90", "95"]:
        row = width_summary[label]["all_rows"]
        lines.append(
            f"| {label} | {int(row['finite_rows']):,} | {float(row['min']):.6f} | {float(row['p05']):.6f} | "
            f"{float(row['p50']):.6f} | {float(row['p95']):.6f} | {float(row['mean']):.6f} | {float(row['max']):.6f} |"
        )
    return "\n".join(lines)


def build_report(payload: Mapping[str, Any], timestamp_utc: str) -> str:
    counts = payload["reference_pool_counts"]
    result = payload["human_only_envelope"]
    metrics = result["metrics"]
    comp = payload["comparison_to_h1"]
    support = result["context_support"]
    onsite = payload["onsite_landing_check"]
    vocab = payload["category_vocabulary_coverage"]
    numeric_ranges = payload["numeric_range_comparison"]
    dryrun = payload["onsite_scoring_dryrun"]
    mechanism1 = payload["onsite_mechanism1_boundary"]
    health = payload["numeric_health"]
    selftest = result["persisted_scoring_selftest"]
    neg = payload["negative_controls"]
    source_k2 = payload["k2_ledger"]["source_path"]
    source_matrix = payload["join_health"]["matrix_source_path"]
    h1_source = comp["source_path"]
    m90 = metrics["90"]
    c90 = comp["by_alpha"]["90"]
    outside = numeric_ranges["onsite_completely_outside_features"]
    dry_pass = int(dryrun["mechanism2_gate_pass_rows"])
    dry_total = int(dryrun["rows"])
    report = f"""# RQ016C-H2 fixed human-only envelope

本轮要解决的问题是：在线验证要判断一辆自动驾驶车表现出的社会交互倾向像不像人。IPV（Interaction Preference Value）是表示交互倾向的标量；判定由两道串联弃权机制构成，机制一先判断这一帧的 IPV 数值能不能进入后续比较，机制二再把通过机制一的数值与人类参照分布（envelope）比较。

整体已经走到：RQ015 冻结机制一，RQ016/RQ016C 在准备机制二的人类参照分布。H1 已在纯人-人样本上拟合 envelope，但 H1 把 `vehicle_type_list` 留在类别 context 中；纯人-人参照池没有 `AV` 取值，真实 OnSite 行全部带 `['AV','HV']`，所以 H1 的持久化产物不能用于它唯一的外部打分用途。本次是 H2 修正重跑，只把类别 context 从 H1 的 5 项改为 4 项：`geometry_path_category`、`geometry_path_relation`、`turn_pair_label`、`priority_role`；其余样本口径、fold、支持门、距离特征、alpha 层和 conformal 计算方式沿用 H1。

## 结论

H2 产物已能在真实 OnSite 67,861 行上完成打分路径 dry-run：只加载 `{dryrun['model_path']}`，不重新拟合，输出 `{dryrun['output_path']}` 和 `{dryrun['summary_path']}`。支持门通过率为 {pct_percent(dry_pass, dry_total)} ({dry_pass:,}/{dry_total:,}; filter=all OnSite rows; source={dryrun['output_path']}; column=mechanism2_gate_ok)。

这个 dry-run 只证明机制二打分管线在真实 OnSite 行上可运行，不构成对任何一辆自动驾驶车的判定。机制一边界的只读核验是：K2 台账 `{mechanism1['source_path']}` 中 `artifact_id == onsite_dense_timeseries` 有 {mechanism1['rows']:,} 行，`status` 非空为 {mechanism1['status_nonnull']:,}/{mechanism1['rows']:,}，`reason_code` 非空为 {mechanism1['reason_code_nonnull']:,}/{mechanism1['rows']:,}，七个 `mse_0..mse_6` 非空计数分别为 `{mechanism1['mse_nonnull']}`；筛选条件为 `artifact_id == onsite_dense_timeseries`，来源列为 `{mechanism1['source_columns']}`。机制一未通过之前，不进入机制二作车辆层面的范围判断。

纯人-人参照池合计 {counts['total_rows']:,} 行；来源为 K2 台账 `{source_k2}` 的 `product_row_key/status/rq007_split/measurement_role` 与 RQ009 矩阵 `{source_matrix}` 的 `case_key/anchor_frame_index/perspective/source_dataset/fold/agent_type_pair/av_included` 精确连接后筛选 `status == OK` 且 `agent_type_pair == HV;HV`。split 组成是 development {counts['split_counts']['development']:,} + guard {counts['split_counts']['guard']:,}，参与计算行中 `rq007_split` 不在 `{{development, guard}}` 的实测计数为 {counts['invalid_split_rows']}。

90% 名义层的纯人-人 envelope coverage = {float(m90['coverage']):.6f}，分子分母为 {int(m90['covered_n']):,}/{int(m90['n']):,}；筛选条件为纯人-人 test fold 且机制二支持门通过；来源列为 RQ009 矩阵 `target_ipv_future/fold/agent_type_pair` 与 K2 `status/rq007_split`。mean width = {float(m90['mean_width']):.6f}，分母为同一批机制二支持门通过的 {int(m90['n']):,} 行。机制二弃权率 = {pct_percent(int(m90['abstained_n']), int(m90['total_n']))} ({int(m90['abstained_n']):,}/{int(m90['total_n']):,}; filter=纯人-人 test fold; source={source_k2} + {source_matrix}; columns=status/rq007_split/fold/agent_type_pair)。

## Alpha 层结果

{metric_table(metrics, source_k2, source_matrix)}

表内 coverage 用小数表示，不写成百分数；coverage 分母是对应 alpha 下机制二支持门通过的纯人-人 test 行，来源列同上。表内机制二弃权率分母是纯人-人 test 行，分子是支持门未通过行。

## 与 H1 对照

{h1_comparison_table(comp)}

H1 来源为 `{h1_source}`。H1 与 H2 的逐项对照只解释一次规格修正：H1 类别 context 为 `{comp['h1_categorical_context']}`，H2 类别 context 为 `{comp['h2_categorical_context']}`，移除项为 `{comp['removed_from_h1']}`。两者同用纯人-人行筛选、22 项数值 context、`geometry_path_category + priority_role` 支持门分格键、12 项支持门距离特征、80/90/95 三个 alpha 层、RQ009 fold 结构、同一 conformal 计算方式和同一 random state。

## 样本计数自查

fold 计数逐项相符：

| fold | pure human rows | status OK rows before human filter | pure human share |
|---|---:|---:|---:|
| train | {counts['fold_human_rows']['train']:,} | {counts['fold_status_ok_rows']['train']:,} | {pct_percent(counts['fold_human_rows']['train'], counts['fold_status_ok_rows']['train'])} ({counts['fold_human_rows']['train']:,}/{counts['fold_status_ok_rows']['train']:,}; filter=fold == train and status == OK; source={source_k2} + {source_matrix}; columns=status/fold/agent_type_pair) |
| calibration | {counts['fold_human_rows']['calibration']:,} | {counts['fold_status_ok_rows']['calibration']:,} | {pct_percent(counts['fold_human_rows']['calibration'], counts['fold_status_ok_rows']['calibration'])} ({counts['fold_human_rows']['calibration']:,}/{counts['fold_status_ok_rows']['calibration']:,}; filter=fold == calibration and status == OK; source={source_k2} + {source_matrix}; columns=status/fold/agent_type_pair) |
| guard_tune | {counts['fold_human_rows']['guard_tune']:,} | {counts['fold_status_ok_rows']['guard_tune']:,} | {pct_percent(counts['fold_human_rows']['guard_tune'], counts['fold_status_ok_rows']['guard_tune'])} ({counts['fold_human_rows']['guard_tune']:,}/{counts['fold_status_ok_rows']['guard_tune']:,}; filter=fold == guard_tune and status == OK; source={source_k2} + {source_matrix}; columns=status/fold/agent_type_pair) |
| test | {counts['fold_human_rows']['test']:,} | {counts['fold_status_ok_rows']['test']:,} | {pct_percent(counts['fold_human_rows']['test'], counts['fold_status_ok_rows']['test'])} ({counts['fold_human_rows']['test']:,}/{counts['fold_status_ok_rows']['test']:,}; filter=fold == test and status == OK; source={source_k2} + {source_matrix}; columns=status/fold/agent_type_pair) |

这些比例的筛选条件为 K2 精确连接后 `status == OK` 的各 RQ009 fold，分子再筛 `agent_type_pair == HV;HV`；来源列为 `{source_k2}` 的 `status` 与 `{source_matrix}` 的 `fold/agent_type_pair`。

## 特征集裁定执行

代码断言结果：`agent_type_pair`、`av_included`、`vehicle_type_list` 不在 M2 特征列表；`agent_type_pair`、`av_included` 不在支持门分格键；`source_dataset` 不在预测变量；`counterpart_ipv_current/counterpart_ipv_error_current/counterpart_ipv_slope_pre_anchor` 不在特征或支持门距离特征。理据是：`vehicle_type_list` 编码场景中各车辆类型，它对 OnSite 的判别内容正是“这里有一辆自动驾驶车”，而车辆是否为自动驾驶车是被检验对象，不是它所处的情境；保留它会使外部行落入训练中从未出现的类别。

## 类别词表覆盖

四个类别 context 特征全部通过词表覆盖断言。

{vocabulary_table(vocab['rows'], vocab['source_reference_pool']['path'], vocab['source_onsite']['path'])}

## 数值值域对照

下表对 22 项数值 context 比较纯人-人参照池与 OnSite 全量行的 min/p50/max。参照池筛选条件为 K2 精确连接后 `status == OK` 且 `agent_type_pair == HV;HV`，来源 `{numeric_ranges['source_reference_pool']['path']}`，列为 22 项数值 context；OnSite 来源 `{numeric_ranges['source_onsite']['path']}`，筛选条件为 all rows，列为同名 22 项数值 context。OnSite 完全落在参照池 min/max 之外的特征：`{outside}`。

{numeric_range_table(numeric_ranges['rows'])}

## 逐格支撑量

新分格键为 `geometry_path_category + priority_role`，纯人-人参照池共有 {support['all_rows']['cells']} 格，最小格样本数 {support['all_rows']['min_rows']:,}。

{support_table(support['all_rows']['counts'])}

## OnSite 落格预演

OnSite 源文件 `{onsite['source_onsite']['path']}` 读取 {onsite['rows']:,} 行，列为 `geometry_path_category/priority_role`；落入 {onsite['cells']} 格，缺格 {len(onsite['missing_cells'])} 个。落入 OnSite 的格中，人类支撑最小的是 `{onsite['min_support_cell']}`，人类支撑 {onsite['min_human_support_among_onsite_cells']:,} 行，OnSite 该格 {onsite['min_support_cell_onsite_rows']:,} 行。

{onsite_table(onsite['counts'])}

## 真实 OnSite 全量 dry-run

dry-run 只加载持久化模型，不重新拟合；输入来自 `{dryrun['source_path']}`，筛选条件为 all rows，列为 `{dryrun['source_columns']}`，并且刻意不加载 `target_ipv_future`。输出 `{dryrun['output_path']}` 每行包含 `lo_80/hi_80/width_80`、`lo_90/hi_90/width_90`、`lo_95/hi_95/width_95`、`mechanism2_gate_ok` 和 `context_cell`。

逐格支持门通过率：

{dryrun_cell_table(dryrun['per_cell'], dryrun['output_path'])}

区间宽度分布：

{width_summary_table(dryrun['width_summary'])}

## 负对照

1. 把 `vehicle_type_list` 放回类别 context 后，词表覆盖断言状态 `{neg['vehicle_type_list_back_in_context']['status']}`，失败输出：

```text
{neg['vehicle_type_list_back_in_context']['failure_output']}
```

2. 把 `agent_type_pair` 放回支持门分格键后，OnSite 落格断言状态 `{neg['agent_type_pair_back_in_support_gate']['status']}`，失败输出：

```text
{neg['agent_type_pair_back_in_support_gate']['failure_output']}
```

## 持久化模型

已拟合 envelope 保存在 `{result['model_artifacts']['model_dir']}`。其中 `rq016c_h2_envelope.pkl` 含条件分位数模型、数值 imputer、类别 encoder、支持门 scaler/encoder/kNN tree、全局 conformal 半径和逐格 calibration 半径；`feature_contract.json` 固化列清单；`support_gate.json` 固化支持门规则与逐格支撑量；`HOWTO_score_external_rows.md` 说明如何给外部行打分。打分接口自测从 test fold 取 {selftest['sample_rows']} 行，只加载持久化产物、不重新拟合，区间边界逐位一致为 `{selftest['bitwise_equal_interval_bounds']}`，支持门一致为 `{selftest['gate_equal']}`，判定一致为 `{selftest['verdict_equal']}`，最大边界差 {selftest['max_abs_interval_diff']:.1e}。

## 自查

held_out 断言：参与计算行中 `rq007_split` 不在 `{{development, guard}}` 的计数为 {counts['invalid_split_rows']}；来源列为 `{source_k2}` 的 `rq007_split`。本轮没有打开受保护 confirmation 划分文件。

数值健康：test fold 目标列 NaN/正无穷/负无穷计数为 {health['target_numeric_counts']['test']['target_nan']}/{health['target_numeric_counts']['test']['target_pos_inf']}/{health['target_numeric_counts']['test']['target_neg_inf']}；80/90/95 三层负宽度行数为 {health['negative_width_rows']}; coverage 均落在 [0,1]：`{health['coverage_in_0_1']}`。

## 待监督方拍板

本轮没有新增需要监督方拍板的阈值、授权或样本口径。若监督方不接受“车辆类型不是 context 变量”这一裁定执行方式，后果是 OnSite 行会重新遇到训练时未出现类别或支持门缺格问题，需要另开一轮定义新的外部打分合同。

state: WAITING_ON_COMMANDER
timestamp_utc: {timestamp_utc}
"""
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-state", type=int, default=20260626)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        payload = json.loads(KEY_NUMBERS_PATH.read_text(encoding="utf-8"))
        timestamp_utc = utc_now()
        payload["created_utc"] = timestamp_utc
        write_json(KEY_NUMBERS_PATH, payload)
        REPORT_PATH.write_text(build_report(payload, timestamp_utc), encoding="utf-8")
        print(f"[{timestamp_utc}] rewrote {REPORT_PATH.relative_to(REPO_ROOT)}", flush=True)
        return 0

    started = utc_now()
    print(f"[{started}] assert feature contract and load inputs", flush=True)
    feature_contract = assert_feature_contract()
    selected_params = ref.load_selected_hgb_params()
    ledger, ledger_diag = ref.load_k2_target_ledger()
    print(f"[{utc_now()}] join RQ009 matrix to K2-ledger case domain", flush=True)
    joined_folds, join_diag = load_joined_folds(ledger)
    frames = human_frames(joined_folds)
    counts = validate_counts(frames, join_diag)
    print(f"[{utc_now()}] validate OnSite vocabulary and landing checks", flush=True)
    onsite_context = read_onsite_context_frame(include_target=False)
    vocabulary = category_vocabulary_coverage(frames, onsite_context, H2_CATEGORICAL_CONTEXT, H2_SUPPORT_CATEGORICAL)
    numeric_ranges = numeric_range_comparison(frames, onsite_context)
    onsite_landing = onsite_landing_for_columns(frames, onsite_context, H2_SUPPORT_CATEGORICAL, enforce_expected=True)
    negative = negative_controls(frames, onsite_context)
    mechanism1_boundary = mechanism1_boundary_check()
    timestamp_before_fit = utc_now()
    result = run_human_envelope(frames, selected_params, args.random_state, timestamp_before_fit)
    onsite = onsite_landing_preview(result["context_support"])
    if onsite["counts"] != onsite_landing["counts"]:
        raise AssertionError("onsite_landing_mismatch_between_model_support_and_frame_check")
    print(f"[{utc_now()}] run persisted-model OnSite dry-run", flush=True)
    onsite_dryrun = score_onsite_dryrun()
    comp = compare_to_h1(result["metrics"])
    health = numeric_health(frames, result["metrics"], result["prediction_health"])
    timestamp_utc = utc_now()
    payload: Dict[str, Any] = {
        "created_utc": timestamp_utc,
        "started_utc": started,
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "reference_script": str(REF_SCRIPT.relative_to(REPO_ROOT)),
        "selected_hgb_params": selected_params,
        "feature_contract": feature_contract,
        "k2_ledger": ledger_diag,
        "join_health": join_diag,
        "reference_pool_counts": counts,
        "category_vocabulary_coverage": vocabulary,
        "numeric_range_comparison": numeric_ranges,
        "onsite_landing_check": onsite_landing,
        "negative_controls": negative,
        "onsite_mechanism1_boundary": mechanism1_boundary,
        "human_only_envelope": result,
        "comparison_to_h1": comp,
        "onsite_landing_preview": onsite,
        "onsite_scoring_dryrun": onsite_dryrun,
        "numeric_health": health,
    }
    write_json(KEY_NUMBERS_PATH, payload)
    REPORT_PATH.write_text(build_report(payload, timestamp_utc), encoding="utf-8")
    print(f"[{timestamp_utc}] wrote {KEY_NUMBERS_PATH.relative_to(REPO_ROOT)}", flush=True)
    print(f"[{timestamp_utc}] wrote {REPORT_PATH.relative_to(REPO_ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
