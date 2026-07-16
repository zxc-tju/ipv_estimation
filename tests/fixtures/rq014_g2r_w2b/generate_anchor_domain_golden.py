#!/usr/bin/env python3
"""Independently generate the W2b 479-scene terminal anchor-domain golden."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path


COLUMNS = (
    "segment_id", "feature_id", "horizon_id", "path_type_or_NA",
    "h_common_tick_or_NA", "tau_tick_or_NA", "membership_status", "reason_code",
)
FEATURES = tuple(
    f"F-{sampling}-{temporal}"
    for sampling in ("R04N", "R10L")
    for temporal in ("CH-W10", "CH-W25", "LF-W10", "LF-W25", "HF-W10", "HF-W25", "TP", "TF")
)


def main() -> int:
    root = Path(__file__).resolve().parent
    csv_path = root / "anchor_domain_15328_golden.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        for scene_index in range(479):
            for feature_id in FEATURES:
                for horizon_id in ("H20", "HFEAS"):
                    writer.writerow(
                        (
                            f"scene-{scene_index:03d}", feature_id, horizon_id, "NA", "NA", "NA",
                            "MISSING_WOD_PATH_TYPE", "F_MISSING_WOD_PATH_TYPE",
                        )
                    )
    data = csv_path.read_bytes()
    binding = {
        "artifact_path": "tests/fixtures/rq014_g2r_w2b/anchor_domain_15328_golden.csv",
        "artifact_sha256": hashlib.sha256(data).hexdigest(),
        "artifact_size_bytes": len(data),
        "data_row_count": 15328,
        "group_count": 15328,
        "schema_version": "rq014-g2r-w2b-anchor-domain-golden-binding-v1",
    }
    (root / "anchor_domain_15328_golden.binding.json").write_text(
        json.dumps(binding, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(binding["artifact_sha256"], len(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
