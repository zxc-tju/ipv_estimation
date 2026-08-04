#!/usr/bin/env python3
"""Reproduce RQ016B-F2 ego-identity counts from existing local artifacts.

This script is read-only with respect to RQ009, RQ016, and data/derived inputs.
It writes only the F2 JSON evidence file and the F2 markdown report.
"""

from __future__ import annotations

import ast
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds


REPO_ROOT = Path(__file__).resolve().parents[4]
WORK_DIR = REPO_ROOT / ".codex-fleet" / "rq016b-wod-onsite-feasibility" / "work" / "F2"
REPORT_PATH = REPO_ROOT / ".codex-fleet" / "rq016b-wod-onsite-feasibility" / "board" / "reports" / "RQ016B_2_ego_identity.md"
JSON_PATH = WORK_DIR / "ego_identity.json"

RQ009_RUN = "RQ009_1_dynamic_envelope_20260625T121905Z_98c433de"
FEATURE_MATRIX_ROOT = (
    REPO_ROOT
    / "data"
    / "derived"
    / "interhub"
    / "RQ009_dynamic_counterpart_conditioned_envelope"
    / RQ009_RUN
    / "03_features"
    / "matrix"
)
K2_RQ009_ROOT = REPO_ROOT / "data" / "derived" / "rq015k_logdomain_gate" / "l1_v1" / "artifact_id=rq009_feature_matrix"
PRIMARY_DATA = (
    REPO_ROOT
    / "data"
    / "derived"
    / "interhub"
    / "20260612_sigma_0_1_full_rerun"
    / "00_hpc_outputs"
    / "sigma01_ipv_timeseries.csv"
)
BUILD_FEATURES = (
    REPO_ROOT
    / "reports"
    / "studies"
    / "RQ009_dynamic_counterpart_conditioned_envelope"
    / RQ009_RUN
    / "02_process"
    / "03_features"
    / "build_features.py"
)
FINALIZE_FEATURES = BUILD_FEATURES.with_name("finalize_features.py")
FEATURE_DICTIONARY = BUILD_FEATURES.with_name("feature_dictionary.csv")
DATA_HEALTH = (
    REPO_ROOT
    / "reports"
    / "studies"
    / "RQ009_dynamic_counterpart_conditioned_envelope"
    / RQ009_RUN
    / "02_process"
    / "02_provenance"
    / "data_health.json"
)
AGENT_TYPE_SOURCE = REPO_ROOT / "pipelines" / "interhub" / "tools" / "update_ipv_distribution_report.py"

EXPECTED_TOTALS = {
    "all": 6_397_266,
    "fold=train": 2_558_374,
    "fold=calibration": 1_266_282,
    "fold=guard_tune": 1_302_044,
    "fold=test": 1_270_566,
    "rq016_B": 635_618,
}
EXPECTED_B_AVHV = 148_958
EXPECTED_SOURCE_TYPE_ROWS = {"AV;HV": 951_217, "HV;HV": 2_744_764}
EXPECTED_MATRIX_AVHV = 1_659_568


def utc_now() -> str:
    return subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip()


def pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def classify_counts(frame: pd.DataFrame) -> Counter[str]:
    pair = frame["agent_type_pair"].astype("string")
    perspective = frame["perspective"].astype("string")
    counts: Counter[str] = Counter()
    counts["E1"] = int(((pair == "AV;HV") & (perspective == "key_agent_1")).sum())
    counts["E2"] = int(((pair == "AV;HV") & (perspective == "key_agent_2")).sum())
    counts["E3"] = int((pair == "HV;HV").sum())
    counts["unexpected"] = int(
        (~(
            ((pair == "AV;HV") & perspective.isin(["key_agent_1", "key_agent_2"]))
            | (pair == "HV;HV")
        )).sum()
    )
    counts["total"] = int(len(frame))
    counts["AV;HV"] = int((pair == "AV;HV").sum())
    counts["HV;HV"] = int((pair == "HV;HV").sum())
    return counts


def merge_counter(dst: Counter[str], src: Counter[str]) -> None:
    for key, value in src.items():
        dst[key] += int(value)


def build_group_record(name: str, counts: Counter[str], source: str, columns: list[str], filter_text: str) -> dict[str, Any]:
    denominator = int(counts["total"])
    return {
        "name": name,
        "denominator": denominator,
        "filter": filter_text,
        "source_file": source,
        "source_columns": columns,
        "counts": {
            key: {
                "numerator": int(counts[key]),
                "denominator": denominator,
                "percent": pct(int(counts[key]), denominator),
            }
            for key in ["E1", "E2", "E3"]
        },
        "checks": {
            "E1_plus_E2_plus_E3": int(counts["E1"] + counts["E2"] + counts["E3"]),
            "unexpected": int(counts["unexpected"]),
            "AV_HV": int(counts["AV;HV"]),
            "HV_HV": int(counts["HV;HV"]),
        },
    }


def scan_feature_matrix() -> dict[str, Any]:
    dataset = ds.dataset(FEATURE_MATRIX_ROOT, format="parquet", partitioning="hive")
    scanner = dataset.scanner(
        columns=["fold", "perspective", "agent_type_pair", "av_included"],
        batch_size=200_000,
    )
    all_counts: Counter[str] = Counter()
    fold_counts: dict[str, Counter[str]] = defaultdict(Counter)
    perspective_by_pair: dict[str, Counter[str]] = defaultdict(Counter)
    av_included_counts: Counter[str] = Counter()

    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        counts = classify_counts(frame)
        merge_counter(all_counts, counts)
        for fold, group in frame.groupby("fold", sort=False):
            merge_counter(fold_counts[str(fold)], classify_counts(group))
        for (pair, perspective), group in frame.groupby(["agent_type_pair", "perspective"], sort=False):
            perspective_by_pair[str(pair)][str(perspective)] += int(len(group))
        for value, count in frame["av_included"].value_counts(dropna=False).items():
            av_included_counts[str(value)] += int(count)

    groups = {
        "all": build_group_record(
            "entire RQ009 feature matrix",
            all_counts,
            str(FEATURE_MATRIX_ROOT),
            ["fold", "agent_type_pair", "perspective", "av_included"],
            "all rows in RQ009 feature matrix",
        )
    }
    for fold in ["train", "calibration", "guard_tune", "test"]:
        groups[f"fold={fold}"] = build_group_record(
            f"RQ009 fold={fold}",
            fold_counts[fold],
            str(FEATURE_MATRIX_ROOT / f"fold={fold}"),
            ["fold", "agent_type_pair", "perspective", "av_included"],
            f"fold == {fold}",
        )

    return {
        "groups": groups,
        "perspective_by_agent_type_pair": {
            pair: dict(counter) for pair, counter in sorted(perspective_by_pair.items())
        },
        "av_included_counts": dict(av_included_counts),
    }


def validate_source_type_order() -> dict[str, Any]:
    columns = ["scene_unique_id", "key_agents", "key_agents_type", "vehicle_type", "AV_included", "key_agent_1", "key_agent_2"]
    total_rows = 0
    type_counts: Counter[str] = Counter()
    avhv_first_agent_ego = 0
    avhv_first_agent_not_ego = 0
    avhv_key_agent_1_not_ego = 0
    avhv_examples: list[dict[str, str]] = []
    avhv_bad_examples: list[dict[str, str]] = []

    for chunk in pd.read_csv(PRIMARY_DATA, usecols=columns, dtype=str, chunksize=500_000):
        total_rows += int(len(chunk))
        type_counts.update({str(k): int(v) for k, v in chunk["key_agents_type"].value_counts(dropna=False).items()})
        avhv = chunk["key_agents_type"].eq("AV;HV")
        if avhv.any():
            first_agent = chunk.loc[avhv, "key_agents"].str.split(";", n=1).str[0]
            avhv_first_agent_ego += int(first_agent.eq("ego").sum())
            avhv_first_agent_not_ego += int((~first_agent.eq("ego")).sum())
            avhv_key_agent_1_not_ego += int((~chunk.loc[avhv, "key_agent_1"].eq("ego")).sum())
            if len(avhv_examples) < 5:
                avhv_examples.extend(
                    chunk.loc[avhv, columns].drop_duplicates(["scene_unique_id"]).head(5 - len(avhv_examples)).to_dict("records")
                )
            bad = chunk.loc[avhv & ~chunk["key_agents"].str.startswith("ego;"), columns]
            if not bad.empty and len(avhv_bad_examples) < 5:
                avhv_bad_examples.extend(bad.head(5 - len(avhv_bad_examples)).to_dict("records"))

    return {
        "source_file": str(PRIMARY_DATA),
        "source_columns": columns,
        "total_source_rows": total_rows,
        "key_agents_type_counts": dict(type_counts),
        "AV_HV_rows": int(type_counts.get("AV;HV", 0)),
        "AV_HV_first_key_agent_is_ego_rows": avhv_first_agent_ego,
        "AV_HV_first_key_agent_not_ego_rows": avhv_first_agent_not_ego,
        "AV_HV_key_agent_1_display_not_ego_rows": avhv_key_agent_1_not_ego,
        "examples": avhv_examples,
        "bad_examples": avhv_bad_examples,
        "interpretation": "For source rows with key_agents_type == AV;HV, the first key agent is the AV ego id in all observed rows.",
    }


def get_k2_ok_target_keys() -> set[str]:
    dataset = ds.dataset(K2_RQ009_ROOT, format="parquet", partitioning="hive")
    expression = (
        (pc.field("artifact_id") == "rq009_feature_matrix")
        & (pc.field("measurement_role") == "target_future")
        & (pc.field("status") == "OK")
    )
    scanner = dataset.scanner(columns=["product_row_key"], filter=expression, batch_size=200_000)
    keys: set[str] = set()
    for batch in scanner.to_batches():
        column = batch.column("product_row_key").to_pylist()
        keys.update(str(value) for value in column)
    return keys


def row_keys(frame: pd.DataFrame) -> pd.Series:
    return (
        "case_key="
        + frame["case_key"].astype("string")
        + "|anchor_frame_index="
        + frame["anchor_frame_index"].astype("int64").astype("string")
        + "|perspective="
        + frame["perspective"].astype("string")
        + "|source_dataset="
        + frame["source_dataset"].astype("string")
    )


def parse_vehicle_types(value: Any) -> list[str]:
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def wrong_vehicle_list_position_counts(frame: pd.DataFrame) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in frame[["perspective", "vehicle_type_list"]].to_dict("records"):
        types = parse_vehicle_types(record["vehicle_type_list"])
        if len(types) < 2:
            counts["unclassified"] += 1
            continue
        if record["perspective"] == "key_agent_1":
            ego_type, counterpart_type = types[0], types[1]
        elif record["perspective"] == "key_agent_2":
            ego_type, counterpart_type = types[1], types[0]
        else:
            counts["unclassified"] += 1
            continue
        if ego_type == "AV":
            counts["E1"] += 1
        elif ego_type == "HV" and counterpart_type == "AV":
            counts["E2"] += 1
        elif ego_type == "HV" and counterpart_type == "HV":
            counts["E3"] += 1
        else:
            counts["unclassified"] += 1
    counts["total"] = int(len(frame))
    return counts


def scan_b_domain(ok_target_keys: set[str]) -> dict[str, Any]:
    test_root = FEATURE_MATRIX_ROOT / "fold=test"
    dataset = ds.dataset(test_root, format="parquet", partitioning="hive")
    scanner = dataset.scanner(
        columns=[
            "case_key",
            "anchor_frame_index",
            "perspective",
            "source_dataset",
            "agent_type_pair",
            "vehicle_type_list",
            "av_included",
        ],
        batch_size=200_000,
    )
    b_counts: Counter[str] = Counter()
    vehicle_list_counts: Counter[str] = Counter()
    wrong_counts: Counter[str] = Counter()
    row_count_before_join = 0
    joined_rows = 0

    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        row_count_before_join += int(len(frame))
        matched = frame.loc[row_keys(frame).isin(ok_target_keys)].copy()
        joined_rows += int(len(matched))
        if matched.empty:
            continue
        merge_counter(b_counts, classify_counts(matched))
        vehicle_list_counts.update({str(k): int(v) for k, v in matched["vehicle_type_list"].value_counts(dropna=False).items()})
        merge_counter(wrong_counts, wrong_vehicle_list_position_counts(matched))

    group = build_group_record(
        "RQ016 B arm domain",
        b_counts,
        f"{FEATURE_MATRIX_ROOT / 'fold=test'} joined to {K2_RQ009_ROOT}",
        [
            "case_key",
            "anchor_frame_index",
            "perspective",
            "source_dataset",
            "agent_type_pair",
            "vehicle_type_list",
            "product_row_key",
            "measurement_role",
            "status",
        ],
        'feature fold == test; K2 artifact_id == "rq009_feature_matrix"; measurement_role == "target_future"; status == "OK"; product_row_key exact match',
    )
    return {
        "group": group,
        "feature_test_rows_before_k2_join": row_count_before_join,
        "k2_ok_target_key_count": len(ok_target_keys),
        "joined_rows": joined_rows,
        "vehicle_type_list_counts": dict(vehicle_list_counts),
        "negative_control_wrong_vehicle_type_list_position": {
            "rule": "Incorrectly treat vehicle_type_list[0]/[1] as key_agent_1/key_agent_2 types.",
            "counts": {
                key: int(wrong_counts.get(key, 0))
                for key in ["E1", "E2", "E3", "unclassified", "total"]
            },
            "observed_E1_plus_E2": int(wrong_counts.get("E1", 0) + wrong_counts.get("E2", 0)),
            "expected_E1_plus_E2_from_agent_type_pair_AV_HV": EXPECTED_B_AVHV,
            "status": "FAIL_EXPECTED"
            if int(wrong_counts.get("E1", 0) + wrong_counts.get("E2", 0)) != EXPECTED_B_AVHV
            else "UNEXPECTED_PASS",
        },
    }


def assert_results(result: dict[str, Any]) -> None:
    groups = result["feature_matrix"]["groups"]
    for key, expected in EXPECTED_TOTALS.items():
        group = result["b_domain"]["group"] if key == "rq016_B" else groups[key]
        actual = int(group["denominator"])
        if actual != expected:
            raise AssertionError(f"{key} denominator mismatch: actual={actual} expected={expected}")
        checks = group["checks"]
        if checks["unexpected"] != 0:
            raise AssertionError(f"{key} unexpected classifications: {checks['unexpected']}")
        if checks["E1_plus_E2_plus_E3"] != expected:
            raise AssertionError(
                f"{key} E1+E2+E3 mismatch: actual={checks['E1_plus_E2_plus_E3']} expected={expected}"
            )

    all_avhv = groups["all"]["checks"]["AV_HV"]
    if all_avhv != EXPECTED_MATRIX_AVHV:
        raise AssertionError(f"matrix AV;HV mismatch: actual={all_avhv} expected={EXPECTED_MATRIX_AVHV}")

    b_group = result["b_domain"]["group"]
    b_avhv = int(b_group["checks"]["AV_HV"])
    if b_avhv != EXPECTED_B_AVHV:
        raise AssertionError(f"B-domain AV;HV mismatch: actual={b_avhv} expected={EXPECTED_B_AVHV}")
    b_e1_e2 = int(b_group["counts"]["E1"]["numerator"] + b_group["counts"]["E2"]["numerator"])
    if b_e1_e2 != EXPECTED_B_AVHV:
        raise AssertionError(f"B-domain E1+E2 mismatch: actual={b_e1_e2} expected={EXPECTED_B_AVHV}")

    source_counts = result["source_type_order"]["key_agents_type_counts"]
    for key, expected in EXPECTED_SOURCE_TYPE_ROWS.items():
        actual = int(source_counts.get(key, 0))
        if actual != expected:
            raise AssertionError(f"source key_agents_type {key} mismatch: actual={actual} expected={expected}")
    if result["source_type_order"]["AV_HV_first_key_agent_not_ego_rows"] != 0:
        raise AssertionError("source AV;HV rows include first key agent not equal to ego")
    if result["b_domain"]["negative_control_wrong_vehicle_type_list_position"]["status"] != "FAIL_EXPECTED":
        raise AssertionError("negative control did not fail as expected")


def count_line(group: dict[str, Any], key: str) -> str:
    item = group["counts"][key]
    return f'{item["numerator"]:,}/{item["denominator"]:,} = {item["percent"]:.4f}%'


def make_matrix_table(groups: dict[str, Any]) -> str:
    rows = [
        "| 群体 | 类别 | 行数与百分比 | 筛选条件 | 来源文件与列 |",
        "|---|---:|---:|---|---|",
    ]
    labels = {
        "all": "整个 RQ009 feature matrix",
        "fold=train": "train",
        "fold=calibration": "calibration",
        "fold=guard_tune": "guard_tune",
        "fold=test": "test",
    }
    filters = {
        "E1": '`agent_type_pair == "AV;HV"` 且 `perspective == "key_agent_1"`',
        "E2": '`agent_type_pair == "AV;HV"` 且 `perspective == "key_agent_2"`',
        "E3": '`agent_type_pair == "HV;HV"`',
    }
    source_cols = "`fold`, `agent_type_pair`, `perspective`, `av_included`"
    for group_key in ["all", "fold=train", "fold=calibration", "fold=guard_tune", "fold=test"]:
        group = groups[group_key]
        for category, name in [("E1", "E1: ego 是 AV"), ("E2", "E2: ego 是人、对手是 AV"), ("E3", "E3: 纯人-人")]:
            group_filter = group["filter"]
            rows.append(
                f"| {labels[group_key]} | {name} | {count_line(group, category)} | {group_filter}; {filters[category]} | `{group['source_file']}`; columns {source_cols} |"
            )
    return "\n".join(rows)


def make_b_table(group: dict[str, Any]) -> str:
    rows = [
        "| 类别 | 行数与百分比 | 筛选条件 | 来源文件与列 |",
        "|---|---:|---|---|",
    ]
    filters = {
        "E1": 'B 域条件; `agent_type_pair == "AV;HV"` 且 `perspective == "key_agent_1"`',
        "E2": 'B 域条件; `agent_type_pair == "AV;HV"` 且 `perspective == "key_agent_2"`',
        "E3": 'B 域条件; `agent_type_pair == "HV;HV"`',
    }
    source_cols = (
        "`case_key`, `anchor_frame_index`, `perspective`, `source_dataset`, "
        "`agent_type_pair`, `vehicle_type_list`, K2 `product_row_key`, `measurement_role`, `status`"
    )
    for category, name in [("E1", "E1: ego 是 AV"), ("E2", "E2: ego 是人、对手是 AV"), ("E3", "E3: 纯人-人")]:
        rows.append(
            f"| {name} | {count_line(group, category)} | {filters[category]} | `{group['source_file']}`; columns {source_cols} |"
        )
    return "\n".join(rows)


def render_report(result: dict[str, Any]) -> str:
    groups = result["feature_matrix"]["groups"]
    b_group = result["b_domain"]["group"]
    b_nc = result["b_domain"]["negative_control_wrong_vehicle_type_list_position"]
    timestamp = result["timestamp_utc"]

    b_e1 = b_group["counts"]["E1"]["numerator"]
    b_e2 = b_group["counts"]["E2"]["numerator"]
    b_e3 = b_group["counts"]["E3"]["numerator"]
    b_total = b_group["denominator"]

    negative_control = json.dumps(b_nc, ensure_ascii=False, indent=2)

    return f"""# RQ016B-F2 ego identity audit

## 定位

最终目标是在线验证：把一辆自动驾驶车表现出的社会交互倾向与人类参照分布比较。RQ016 已经重建了人类参照分布，下一步要把它用于 OnSite 自动驾驶车数据。本次 F2 是进入下一步之前的只读事实查证：确认 RQ009/RQ016 envelope 里的目标列到底属于哪一方，以及其中是否混入自动驾驶车自己的目标值。

## 结论先行

有。RQ016 B 臂域共 {b_total:,} 行，来源是 RQ009 `fold == test` feature rows 精确连接 K2 台账中 `artifact_id == "rq009_feature_matrix"`、`measurement_role == "target_future"`、`status == "OK"` 的 `product_row_key`。其中 E1，即 ego 是 AV、目标值是自动驾驶车自己的 IPV，为 {b_e1:,}/{b_total:,} = {pct(b_e1, b_total):.4f}%；E2，即 ego 是人、对手是 AV，为 {b_e2:,}/{b_total:,} = {pct(b_e2, b_total):.4f}%；E3，即纯人-人，为 {b_e3:,}/{b_total:,} = {pct(b_e3, b_total):.4f}%。

关键核对通过：B 臂域的 E1+E2 = {b_e1 + b_e2:,}/{b_total:,} = {pct(b_e1 + b_e2, b_total):.4f}%，正好等于监督方给出的 B 臂域 `AV;HV` 行数 148,958/{b_total:,} = {pct(148_958, b_total):.4f}%。

## Q1: `target_ipv_future` 是谁的 IPV

结论：`target_ipv_future` 是该行 `ego_key_agent` 的未来 IPV；`counterpart_ipv_current` 是该行 `counterpart_key_agent` 的当前 IPV。

代码证据：

```python
# reports/.../build_features.py:665-674
if perspective == "key_agent_1":
    ego_id, cp_id = key_agent_1, key_agent_2
    prefix_ego, prefix_cp = "key_agent_1", "key_agent_2"
    target_ipv = targets["target_ipv_key_agent_1_by_row"][target_final_pos]
elif perspective == "key_agent_2":
    ego_id, cp_id = key_agent_2, key_agent_1
    prefix_ego, prefix_cp = "key_agent_2", "key_agent_1"
    target_ipv = targets["target_ipv_key_agent_2_by_row"][target_final_pos]
```

同一函数随后把 `ego_id` 写入 `ego_key_agent`，把 `cp_id` 写入 `counterpart_key_agent`，并把 `target_ipv` 写入 `target_ipv_future`：`build_features.py:724-726` 与 `build_features.py:779`。`counterpart_ipv_current` 明确来自 `row[f"ipv_{{prefix_cp}}"]`，即另一方当前列：`build_features.py:774`。目标 lookup 来源列是 `TARGET_HW4 ipv_key_agent_1/2 at frame t*+6`：`build_features.py:619-623`、`feature_dictionary.csv:57`。`finalize_features.py` 只把 `target_ipv_future` 读入分布审计，不改变它的身份口径：`finalize_features.py:153`。

字典原文也一致：`perspective` 是 “key_agent_1 or key_agent_2 as ego”，`ego_key_agent` 是从 `key_agents` 解析的 ego id，`counterpart_ipv_current` 是 counterpart alias，`target_ipv_future` 是 ego hw=4 IPV：`feature_dictionary.csv:7-9`、`feature_dictionary.csv:52`、`feature_dictionary.csv:57`。

## Q2: 每一行的 ego 是 AV 还是人

可确定，但不能用 `vehicle_type_list` 的位置直接判定。

确定 ego 与对手身份的代码规则是：`key_agents` 被分号切成两个 agent，第一位是 `key_agent_1`，第二位是 `key_agent_2`；`perspective == "key_agent_1"` 时 ego 是第一位，`perspective == "key_agent_2"` 时 ego 是第二位。对应代码在 `build_features.py:492-496`、`build_features.py:658`、`build_features.py:665-674`、`build_features.py:724-726`。`data_health.json:443-444` 也记录了同一合同：`key_agents` 分号切分，列位置 `key_agent_1/key_agent_2` 对应第一/第二 agent。

车辆类型的可用口径是 `agent_type_pair`，它来自源列 `key_agents_type`：`build_features.py:741` 与 `feature_dictionary.csv:24`。非报告代码中对 key-agent 车辆类型的生成说明是：先把 pkl `metadata.vehicle_type` 与 `metadata.track_ids` 对齐，再按 CSV `key_agents` 顺序抽取 key agent 1/2 的类型；见 `pipelines/interhub/tools/update_ipv_distribution_report.py:291-297`、`pipelines/interhub/tools/update_ipv_distribution_report.py:665-685`。

本次脚本只读源结构列复核了这个顺序：`{PRIMARY_DATA}` 的 `key_agents_type == "AV;HV"` 源行共有 {result["source_type_order"]["AV_HV_rows"]:,}/{result["source_type_order"]["total_source_rows"]:,} = {pct(result["source_type_order"]["AV_HV_rows"], result["source_type_order"]["total_source_rows"]):.4f}%，筛选条件为源列 `key_agents_type == "AV;HV"`，来源列为 `key_agents`, `key_agents_type`, `vehicle_type`, `AV_included`, `key_agent_1`, `key_agent_2`；这些行中 `key_agents` 第一位为 `ego` 的有 {result["source_type_order"]["AV_HV_first_key_agent_is_ego_rows"]:,}/{result["source_type_order"]["AV_HV_rows"]:,} = {pct(result["source_type_order"]["AV_HV_first_key_agent_is_ego_rows"], result["source_type_order"]["AV_HV_rows"]):.4f}%，第一位不是 `ego` 的为 0/{result["source_type_order"]["AV_HV_rows"]:,} = 0.0000%。因此在 RQ009 feature matrix 中使用以下规则：

- `agent_type_pair == "AV;HV"` 且 `perspective == "key_agent_1"`：E1，ego 是 AV。
- `agent_type_pair == "AV;HV"` 且 `perspective == "key_agent_2"`：E2，ego 是人、对手是 AV。
- `agent_type_pair == "HV;HV"`：E3，纯人-人。

`vehicle_type_list` 不能独立映射到 key-agent 编号。构造代码只是把源列 `vehicle_type` 原样写入 `vehicle_type_list`：`build_features.py:742`；字典也只说它是 “Vehicle type metadata list”：`feature_dictionary.csv:25`。本次 B 臂域中 `vehicle_type_list` 的主要取值包括：

```json
{json.dumps(dict(list(result["b_domain"]["vehicle_type_list_counts"].items())[:10]), ensure_ascii=False, indent=2)}
```

这些列表常含 3 个或更多元素，且元素没有 agent id；所以不能从 `vehicle_type_list[0]` 或 `vehicle_type_list[1]` 直接推出 ego 类型。

## Q3: 三类计数

### 整个 RQ009 feature matrix 与各 fold

{make_matrix_table(groups)}

全矩阵核对：`AV;HV` 为 {groups["all"]["checks"]["AV_HV"]:,}/{groups["all"]["denominator"]:,} = {pct(groups["all"]["checks"]["AV_HV"], groups["all"]["denominator"]):.4f}%，筛选条件为 `agent_type_pair == "AV;HV"`，来源文件 `{FEATURE_MATRIX_ROOT}`，来源列 `agent_type_pair`。四个 fold 的 E1+E2+E3 都等于各自分母，`unexpected == 0`。

### RQ016 B 臂域

B 臂域定义：RQ009 `fold == test` 行，构造 `case_key=<...>|anchor_frame_index=<...>|perspective=<...>|source_dataset=<...>`，精确连接 K2 台账 `{K2_RQ009_ROOT}` 中 `artifact_id == "rq009_feature_matrix"`、`measurement_role == "target_future"`、`status == "OK"` 的 `product_row_key`。

{make_b_table(b_group)}

## 自查

1. B 臂域 `E1 + E2`：{b_e1:,} + {b_e2:,} = {b_e1 + b_e2:,}，等于监督方给出的 148,958。
2. 各群体 `E1 + E2 + E3` 都等于分母；脚本对 `all`、四个 fold、B 臂域逐项断言，且 `unexpected == 0`。
3. 负对照：故意错误地把 `vehicle_type_list[0]` / `vehicle_type_list[1]` 当作 `key_agent_1` / `key_agent_2` 类型。该规则在 B 臂域失败，输出如下：

```json
{negative_control}
```

4. Q1 的结论可由 `build_features.py:665-674`、`build_features.py:724-779`、`feature_dictionary.csv:52`、`feature_dictionary.csv:57` 直接验证。
5. 本次没有读取 RQ014 致盲评分字段；没有读取 `target_ipv_future` 数值列来做 Q3 计数；没有按 RQ007 split 筛选 held-out 行。Q3 第 1、2 项只用 RQ009 结构列 `fold`, `agent_type_pair`, `perspective`, `av_included`；B 臂域使用 K2 的 `product_row_key`, `measurement_role`, `status`。

## 复跑

```bash
python3 .codex-fleet/rq016b-wod-onsite-feasibility/work/F2/rq016b_f2_ego_identity.py
```

输出：

- JSON: `.codex-fleet/rq016b-wod-onsite-feasibility/work/F2/ego_identity.json`
- Report: `.codex-fleet/rq016b-wod-onsite-feasibility/board/reports/RQ016B_2_ego_identity.md`

state: WAITING_ON_COMMANDER
timestamp_utc: {timestamp}
"""


def main() -> int:
    timestamp = utc_now()
    result: dict[str, Any] = {
        "task": "RQ016B-F2 ego identity audit",
        "timestamp_utc": timestamp,
        "inputs": {
            "feature_matrix_root": str(FEATURE_MATRIX_ROOT),
            "k2_rq009_root": str(K2_RQ009_ROOT),
            "primary_data": str(PRIMARY_DATA),
            "build_features": str(BUILD_FEATURES),
            "finalize_features": str(FINALIZE_FEATURES),
            "feature_dictionary": str(FEATURE_DICTIONARY),
            "data_health": str(DATA_HEALTH),
            "agent_type_source": str(AGENT_TYPE_SOURCE),
        },
        "method": {
            "classification_rule": {
                "E1": 'agent_type_pair == "AV;HV" and perspective == "key_agent_1"',
                "E2": 'agent_type_pair == "AV;HV" and perspective == "key_agent_2"',
                "E3": 'agent_type_pair == "HV;HV"',
            },
            "b_domain_join_key": "case_key=<case_key>|anchor_frame_index=<anchor_frame_index>|perspective=<perspective>|source_dataset=<source_dataset>",
            "b_domain_k2_filter": {
                "artifact_id": "rq009_feature_matrix",
                "measurement_role": "target_future",
                "status": "OK",
            },
        },
        "code_evidence": {
            "target_identity": [
                "build_features.py:665-674 perspective chooses ego/counterpart prefixes and target ipv_key_agent_1/2",
                "build_features.py:724-726 writes ego_key_agent and counterpart_key_agent",
                "build_features.py:774 writes counterpart_ipv_current from prefix_cp",
                "build_features.py:779 writes target_ipv_future from target_ipv",
                "feature_dictionary.csv:52 counterpart_ipv_current definition",
                "feature_dictionary.csv:57 target_ipv_future definition",
            ],
            "type_identity": [
                "build_features.py:741 copies key_agents_type into agent_type_pair",
                "build_features.py:742 copies vehicle_type into vehicle_type_list",
                "feature_dictionary.csv:24-25 documents those source columns",
                "pipelines/interhub/tools/update_ipv_distribution_report.py:665-685 aligns vehicle_type to track_ids, then key_agents order",
            ],
        },
    }

    result["source_type_order"] = validate_source_type_order()
    result["feature_matrix"] = scan_feature_matrix()
    ok_target_keys = get_k2_ok_target_keys()
    result["b_domain"] = scan_b_domain(ok_target_keys)
    assert_results(result)

    JSON_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(render_report(result), encoding="utf-8")
    print(json.dumps({"status": "PASS", "json": str(JSON_PATH), "report": str(REPORT_PATH)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
