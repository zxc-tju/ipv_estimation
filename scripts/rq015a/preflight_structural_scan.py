#!/usr/bin/env python3
"""RQ015A 预执行结构核验（可复现）。

**边界**：本脚本只读取结构性字段——列名、主键、行数、split/fold 归属、
文件存在性。它**不解析、不聚合、不输出任何 `ipv_*` measurement 数值**，
因此运行它不构成 RQ015A 的执行，也不构成新的 held_out 暴露。

`--allow-measurement-columns` 这样的开关**故意不提供**：想读 measurement，
必须走已授权的 ledger builder，而不是这里。

产出：`reports/knowledge/RQ015A_ipv_estimability_labelling/preflight_contract_verification_*.md`
所引用的全部数字。

用法：
    python3 scripts/rq015a/preflight_structural_scan.py --repo-root . [--section all]
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
import os
import sys

csv.field_size_limit(10 ** 9)

RQ009_RUN = ("data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/"
             "RQ009_1_dynamic_envelope_20260625T121905Z_98c433de")
SPLIT_CSV = ("data/derived/interhub/RQ007_interaction_conditioned_ipv_estimability/"
             "RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_outputs/splits/"
             "case_split_assignment.csv")
SIGMA01 = f"{RQ009_RUN}/03_features/target_hw4/sigma01_hw4_ipv_timeseries.csv"
FEATURE_GLOB = f"{RQ009_RUN}/03_features/matrix/fold=*/source_dataset=*/*.parquet"
ONSITE_CSV = ("data/derived/onsite_competition/RQ012B_event_harm/stage3plus/"
              "onsite_anchors/onsite_ipv_timeseries.csv")

# 任何名称匹配这些前缀的列都禁止被本脚本读取。
FORBIDDEN_COLUMN_PREFIXES = ("ipv_", "target_ipv", "counterpart_ipv", "M4_ONLY_")


def _assert_structural(columns) -> None:
    """守卫：本脚本读到 measurement 列即视为越权，fail closed。"""
    bad = [c for c in columns
           if any(c.startswith(p) for p in FORBIDDEN_COLUMN_PREFIXES)]
    if bad:
        raise RuntimeError(
            f"preflight scan attempted to read measurement columns {bad}; "
            "this script is structural-only by contract")


def load_split(root: str):
    path = os.path.join(root, SPLIT_CSV)
    with open(path, newline="") as f:
        mapping = {r["case_id"]: r["split"] for r in csv.DictReader(f)}
    counts = collections.Counter(mapping.values())
    return mapping, {"path": SPLIT_CSV, "cases": len(mapping),
                     "per_split": dict(counts)}


def scan_sigma01(root: str, split):
    """C1/C2/C11 的数字来源。只读 scene_unique_id / frame_index。"""
    cols = ["scene_unique_id", "frame_index"]
    _assert_structural(cols)
    path = os.path.join(root, SIGMA01)
    n = 0
    per_split = collections.Counter()
    d0 = collections.Counter()
    frame_zero = collections.Counter()
    cases = collections.defaultdict(set)
    unmapped = set()
    with open(path, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        i_sid = header.index("scene_unique_id")
        i_fi = header.index("frame_index")
        for row in rd:
            n += 1
            sid = row[i_sid]
            sp = split.get(sid, "__UNMAPPED__")
            if sp == "__UNMAPPED__":
                unmapped.add(sid)
            per_split[sp] += 1
            fi = int(row[i_fi])
            if fi < 4:
                d0[sp] += 1
            if fi == 0:
                frame_zero[sp] += 1
            cases[sp].add(sid)
    dev_guard_physical = per_split["development"] + per_split["guard"]
    dev_guard_d0 = d0["development"] + d0["guard"]
    return {
        "artifact_id": "interhub_sigma01_hw4_timeseries",
        "total_data_rows": n,
        "rows_per_split": dict(per_split),
        "d0_rows_per_split": dict(d0),
        "frame_index_zero_per_split": dict(frame_zero),
        "cases_per_split": {k: len(v) for k, v in cases.items()},
        "unmapped_scene_ids": len(unmapped),
        # C1：identity_1 的正确基数是"未排除 D0"的物理行数
        "dev_guard_physical_rows": dev_guard_physical,
        "dev_guard_d0_rows": dev_guard_d0,
        "dev_guard_post_d0_rows": dev_guard_physical - dev_guard_d0,
        "expansion_factor": 2,
        "collapse_factor": 1,
        "measurement_rows_expected": dev_guard_physical * 2,
        "not_attempted_measurement_rows_expected": dev_guard_d0 * 2,
        # C11：每 case 恰一行 frame_index==0 ⇒ 逐 case 0 起且连续
        "frame_index_is_case_local_0_based": all(
            frame_zero[s] == len(cases[s]) for s in ("development", "guard", "held_out")),
    }


def scan_feature_matrix(root: str, split):
    """C3–C6 的数字来源。只读 case_key / fold / source_dataset。"""
    import pandas as pd  # 延迟导入：本函数是唯一需要 parquet 的部分

    cols = ["case_key", "fold", "source_dataset"]
    _assert_structural(cols)
    rows = collections.Counter()
    cases = collections.defaultdict(set)
    cross = collections.Counter()
    parts = sorted(glob.glob(os.path.join(root, FEATURE_GLOB)))
    if not parts:
        return {"artifact_id": "rq009_feature_matrix", "error": "no parquet parts found"}
    engine = _parquet_engine()
    for part in parts:
        df = pd.read_parquet(part, engine=engine, columns=cols)
        _assert_structural(list(df.columns))
        for fold, sub in df.groupby("fold", observed=True):
            rows[str(fold)] += len(sub)
            cases[str(fold)].update(sub["case_key"].unique())
        for ck, fd in zip(df["case_key"].values, df["fold"].values):
            cross[(str(fd), split.get(ck, "__UNMAPPED__"))] += 1
    dev = sum(c for (f, s), c in cross.items() if s == "development")
    guard = sum(c for (f, s), c in cross.items() if s == "guard")
    held = sum(c for (f, s), c in cross.items() if s == "held_out")
    folds_with_held_out = sorted({f for (f, s), c in cross.items()
                                  if s == "held_out" and c > 0})
    return {
        "artifact_id": "rq009_feature_matrix",
        "parquet_parts": len(parts),
        "parquet_engine": engine,
        "measurement_rows_per_fold": dict(rows),
        "cases_per_fold": {k: len(v) for k, v in cases.items()},
        "fold_x_rq007split_rowcounts": {f"{a}|{b}": c for (a, b), c in sorted(cross.items())},
        "total_rows": sum(rows.values()),
        "dev_rows": dev, "guard_rows": guard, "held_out_rows": held,
        "dev_guard_rows": dev + guard,
        # C6：fold 与 split 正交 —— 每个 fold 都含 held_out
        "folds_containing_held_out": folds_with_held_out,
        "fold_is_a_valid_split_proxy": len(folds_with_held_out) == 0,
    }


def scan_onsite(root: str):
    """C7/C13 的数字来源。只读表头与主键列。"""
    path = os.path.join(root, ONSITE_CSV)
    with open(path, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        n = 0
        first_frame_index = None
        for row in rd:
            n += 1
            if first_frame_index is None:
                first_frame_index = row[header.index("frame_index")]
    return {
        "artifact_id": "onsite_dense_timeseries",
        "physical_rows": n,
        "has_case_key_column": "case_key" in header,
        "has_case_id_column": "case_id" in header,
        "first_row_frame_index": first_frame_index,
        # C7：首行 frame_index 非 0 ⇒ 全局规则与 min 相减规则均不可用
        "global_frame_rule_is_valid": first_frame_index == "0",
        "ipv_column_names_present": sorted(
            c for c in header if c.startswith("ipv_")),
    }


def scan_candidate_grids(root: str):
    """C8/C12：K 由源码常量确定，不外推。用正则读取，避免 import scipy。"""
    import re
    path = os.path.join(root, "src/sociality_estimation/core/agent.py")
    grids = {}
    pat = re.compile(r"^(\w*agent_IPV_range)\s*=\s*np\.array\(\[([^\]]+)\]\)\s*\*\s*math\.pi\s*/\s*(\d+)")
    with open(path) as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                name, body, denom = m.groups()
                ks = [int(x) for x in body.replace(" ", "").split(",")]
                grids[name] = {
                    "K": len(ks), "multipliers": ks, "denominator": int(denom),
                    "uniform_fallback_error": 1.0 - 1.0 / math.sqrt(len(ks)),
                }
    return {"candidate_grids": grids,
            "K_is_constant_across_codebase": len({g["K"] for g in grids.values()}) == 1}


def _parquet_engine() -> str:
    for name in ("pyarrow", "fastparquet"):
        try:
            __import__(name)
            return name
        except ImportError:
            continue
    raise RuntimeError(
        "no parquet engine available; run spec must declare pyarrow or fastparquet "
        "(the 'stdlib only' claim is false: feature matrix and OnSite anchors are parquet)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--section", default="all",
                    choices=["all", "split", "sigma01", "feature_matrix", "onsite", "grids"])
    args = ap.parse_args()
    root = os.path.abspath(args.repo_root)

    split, split_info = load_split(root)
    out = {"scan_kind": "STRUCTURAL_ONLY",
           "reads_measurement_fields": False,
           "repo_root": root,
           "split_source": split_info}
    want = args.section
    if want in ("all", "sigma01"):
        out["sigma01"] = scan_sigma01(root, split)
    if want in ("all", "feature_matrix"):
        out["feature_matrix"] = scan_feature_matrix(root, split)
    if want in ("all", "onsite"):
        out["onsite"] = scan_onsite(root)
    if want in ("all", "grids"):
        out["candidate_grids"] = scan_candidate_grids(root)
    json.dump(out, sys.stdout, indent=2, ensure_ascii=False, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
