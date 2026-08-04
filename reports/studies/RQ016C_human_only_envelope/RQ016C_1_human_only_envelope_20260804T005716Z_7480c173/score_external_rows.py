#!/usr/bin/env python3
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
