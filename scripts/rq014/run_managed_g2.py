#!/usr/bin/env python3
"""Run a fixed, managed RQ014 G2 operation from the tracked checkout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.rq014.preflight import canonical_json_bytes, run_preflight
from scripts.rq014.run_resource_pilot import run_resource_pilot


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite non-identical receipt: {path}")
        path.chmod(0o444)
        return
    path.write_bytes(payload)
    path.chmod(0o444)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    preflight = subparsers.add_parser("contract-preflight")
    preflight.add_argument("--base", type=Path, required=True)
    preflight.add_argument("--repo-root", type=Path, required=True)
    preflight.add_argument("--execution-contract", type=Path, required=True)
    preflight.add_argument("--expected-exporter-git-commit", required=True)
    preflight.add_argument("--expected-exporter-environment-sha256", required=True)
    pilot = subparsers.add_parser("resource-pilot")
    pilot.add_argument("--run-id", required=True)
    pilot.add_argument("--lane-v3", type=Path, required=True)
    pilot.add_argument("--bundle-root", type=Path, required=True)
    pilot.add_argument("--wod-path-type-mapping-manifest", type=Path, required=True)
    pilot.add_argument("--contract-preflight-receipt", type=Path, required=True)
    pilot.add_argument("--contract-preflight-done", type=Path, required=True)
    pilot.add_argument("--m3-parity-fixture", type=Path, required=True)
    for operation_parser in (preflight, pilot):
        operation_parser.add_argument("--m3-artifact", type=Path, required=True)
        operation_parser.add_argument("--m3-artifact-size-bytes", type=int, required=True)
        operation_parser.add_argument("--m3-artifact-sha256", required=True)
        operation_parser.add_argument("--input-manifest", type=Path, required=True)
        operation_parser.add_argument("--sanitization-receipt", type=Path, required=True)
        operation_parser.add_argument("--materialization-ledger", type=Path, required=True)
        operation_parser.add_argument("--declassification-export-receipt", type=Path, required=True)
        operation_parser.add_argument("--declassification-export-done", type=Path, required=True)
        operation_parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.operation == "contract-preflight":
        receipt = run_preflight(
            base=args.base.resolve(),
            repo_root=args.repo_root.resolve(),
            execution_contract_path=args.execution_contract.resolve(),
            m3_artifact_ref={
                "path": str(args.m3_artifact),
                "size_bytes": args.m3_artifact_size_bytes,
                "sha256": args.m3_artifact_sha256,
            },
            input_manifest_path=args.input_manifest.resolve(),
            sanitization_receipt_path=args.sanitization_receipt.resolve(),
            materialization_ledger_path=args.materialization_ledger.resolve(),
            declassification_export_receipt_path=args.declassification_export_receipt.resolve(),
            declassification_export_done_path=args.declassification_export_done.resolve(),
            expected_exporter_git_commit=args.expected_exporter_git_commit,
            expected_exporter_environment_sha256=args.expected_exporter_environment_sha256,
        )
        receipt_name = "rq014_g2_contract_preflight_receipt.json"
        operation_name = "rq014_g2_contract_preflight"
    else:
        receipt = run_resource_pilot(
            run_id=args.run_id,
            lane_path=args.lane_v3.resolve(),
            bundle_root=args.bundle_root.resolve(),
            input_manifest_path=args.input_manifest.resolve(),
            sanitization_receipt_path=args.sanitization_receipt.resolve(),
            materialization_ledger_path=args.materialization_ledger.resolve(),
            mapping_manifest_path=args.wod_path_type_mapping_manifest.resolve(),
            m3_artifact_path=args.m3_artifact.resolve(),
            m3_artifact_size_bytes=args.m3_artifact_size_bytes,
            m3_artifact_sha256=args.m3_artifact_sha256,
            m3_parity_fixture_path=args.m3_parity_fixture.resolve(),
            export_receipt_path=args.declassification_export_receipt.resolve(),
            export_done_path=args.declassification_export_done.resolve(),
            preflight_receipt_path=args.contract_preflight_receipt.resolve(),
            preflight_done_path=args.contract_preflight_done.resolve(),
        )
        receipt_name = "rq014_g2_resource_pilot_receipt.json"
        operation_name = "rq014_g2_resource_pilot"
    payload = canonical_json_bytes(receipt)
    output = args.output_root.resolve()
    _write_once(output / receipt_name, payload)
    if receipt.get("status") == "PASS":
        _write_once(
            output / "DONE.json",
            canonical_json_bytes(
                {
                    "schema_version": "rq014-managed-operation-done-v1",
                    "operation": operation_name,
                    "receipt_sha256": __import__("hashlib").sha256(payload).hexdigest(),
                    "status": "PASS",
                }
            ),
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
