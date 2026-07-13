#!/usr/bin/env python3
"""Run a fixed, managed RQ014 G2 operation from the tracked checkout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.rq014.preflight import canonical_json_bytes, run_preflight


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
    parser.add_argument("operation", choices=["contract-preflight"])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--m3-artifact", type=Path, required=True)
    parser.add_argument("--m3-artifact-size-bytes", type=int, required=True)
    parser.add_argument("--m3-artifact-sha256", required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--sanitization-receipt", type=Path, required=True)
    parser.add_argument("--materialization-ledger", type=Path, required=True)
    parser.add_argument("--declassification-export-receipt", type=Path, required=True)
    parser.add_argument("--declassification-export-done", type=Path, required=True)
    parser.add_argument("--expected-exporter-git-commit", required=True)
    parser.add_argument("--expected-exporter-environment-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

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
    payload = canonical_json_bytes(receipt)
    output = args.output_root.resolve()
    _write_once(output / "rq014_g2_contract_preflight_receipt.json", payload)
    _write_once(
        output / "DONE.json",
        canonical_json_bytes(
            {
                "schema_version": "rq014-managed-operation-done-v1",
                "operation": "rq014_g2_contract_preflight",
                "receipt_sha256": __import__("hashlib").sha256(payload).hexdigest(),
                "status": "PASS",
            }
        ),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
