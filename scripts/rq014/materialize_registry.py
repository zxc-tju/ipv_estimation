#!/usr/bin/env python3
"""Verify and materialize immutable RQ014 registries from a G2 SHA ledger."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any


HEX64 = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDER = "TO_FREEZE_AT_G2"
DEFAULT_CONTRACT = Path("reports/plans/RQ014_execution_contract_v1p5.json")


class ContractError(ValueError):
    """Raised when a registry materialization contract fails closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ContractError(f"Non-finite JSON value: {token}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot load JSON contract {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"Top-level JSON object required: {path}")
    return value


def canonical_bytes(value: Any, *, trailing_newline: bool = True) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return payload + (b"\n" if trailing_newline else b"")


def _pointer_parts(pointer: str) -> list[str]:
    if not pointer.startswith("/"):
        raise ContractError(f"JSON pointer must start with '/': {pointer}")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def get_pointer(document: Any, pointer: str) -> Any:
    current = document
    for part in _pointer_parts(pointer):
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError) as exc:
                raise ContractError(f"Invalid list pointer {pointer}") from exc
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise ContractError(f"Missing pointer target {pointer}")
    return current


def set_pointer(document: Any, pointer: str, value: str) -> None:
    parts = _pointer_parts(pointer)
    current = document
    for part in parts[:-1]:
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current[part]
    leaf = parts[-1]
    if isinstance(current, list):
        current[int(leaf)] = value
    else:
        current[leaf] = value


def count_placeholder(value: Any) -> int:
    if isinstance(value, dict):
        return sum(count_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return sum(count_placeholder(item) for item in value)
    return int(value == PLACEHOLDER)


def _write_once_or_identical(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise ContractError(f"Refusing materialization symlink: {path}")
    if path.exists():
        if path.read_bytes() != payload:
            raise ContractError(f"Refusing to overwrite non-identical artifact: {path}")
        return
    path.write_bytes(payload)


def _x02_composite(source_definition_sha256: str, wod_mapping_sha256: str) -> str:
    payload = canonical_bytes(
        {
            "source_definition_sha256": source_definition_sha256,
            "wod_mapping_sha256": wod_mapping_sha256,
        },
        trailing_newline=False,
    )
    return hashlib.sha256(payload).hexdigest()


def materialize(
    *,
    repo_root: Path,
    contract_path: Path,
    freeze_values_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if contract_path.is_symlink() or freeze_values_path.is_symlink() or output_dir.is_symlink():
        raise ContractError("Registry materialization inputs/output may not be symlinks")
    repo_root = repo_root.resolve()
    contract_path = contract_path.resolve()
    freeze_values_path = freeze_values_path.resolve()
    contract = load_json(contract_path)
    if contract.get("schema_version") != "rq014-execution-contract-v1p5":
        raise ContractError("Wrong RQ014 execution contract schema")

    policy = contract.get("registry_binding_contract")
    if not isinstance(policy, dict):
        raise ContractError("Missing registry_binding_contract")
    required_ids = policy.get("required_binding_ids")
    targets = policy.get("binding_targets")
    if not isinstance(required_ids, list) or not isinstance(targets, dict):
        raise ContractError("Malformed binding ID/target contract")
    if len(required_ids) != policy.get("required_binding_count"):
        raise ContractError("Binding count does not match required_binding_ids")
    if set(required_ids) != set(targets):
        raise ContractError("Binding IDs and targets differ")

    freeze_values = load_json(freeze_values_path)
    if freeze_values_path.read_bytes() != canonical_bytes(freeze_values):
        raise ContractError("Registry bindings must use canonical JSON bytes")
    if set(freeze_values) != {"schema_version", "stage", "bindings"}:
        raise ContractError("Freeze-values object has missing or unexpected keys")
    if freeze_values["schema_version"] != "rq014-registry-bindings-g2-v1":
        raise ContractError("Wrong registry-bindings schema")
    if freeze_values["stage"] != "G2":
        raise ContractError("Only G2 materialization is supported")
    bindings = freeze_values["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != set(required_ids):
        raise ContractError("Freeze binding keys must exactly equal the G2 allowlist")
    for binding_id, value in bindings.items():
        if not isinstance(value, str) or not HEX64.fullmatch(value):
            raise ContractError(f"Binding is not a lowercase SHA-256: {binding_id}")

    for left, right in policy.get("cross_registry_equalities", []):
        if bindings[left] != bindings[right]:
            raise ContractError(f"Cross-registry binding mismatch: {left} != {right}")
    registry_paths = contract.get("active_registries")
    if not isinstance(registry_paths, dict):
        raise ContractError("Missing active_registries")
    registries: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    source_paths: dict[str, str] = {}
    for name, relative in registry_paths.items():
        source = (repo_root / relative).resolve()
        try:
            source.relative_to(repo_root)
        except ValueError as exc:
            raise ContractError(f"Registry escapes repository: {source}") from exc
        registries[name] = load_json(source)
        source_hashes[name] = sha256_file(source)
        source_paths[name] = str(relative)

    binding_mode = policy.get("source_binding_mode", "MATERIALIZE_PLACEHOLDERS")
    placeholder_count = sum(count_placeholder(registry) for registry in registries.values())
    materialized = copy.deepcopy(registries)
    if binding_mode == "VERIFY_PREFILLED_EXACT":
        if placeholder_count != 0:
            raise ContractError("Prefilled source registries may not retain placeholders")
        for binding_id in required_ids:
            target = targets[binding_id]
            if set(target) != {"registry", "pointer"}:
                raise ContractError(f"Malformed target for {binding_id}")
            registry_name = target["registry"]
            if registry_name not in materialized:
                raise ContractError(f"Unknown target registry for {binding_id}")
            if get_pointer(materialized[registry_name], target["pointer"]) != bindings[binding_id]:
                raise ContractError(f"Prefilled source binding mismatch: {binding_id}")
    elif binding_mode == "MATERIALIZE_PLACEHOLDERS":
        if placeholder_count != policy["required_binding_count"]:
            raise ContractError(
                f"Expected {policy['required_binding_count']} placeholders, found {placeholder_count}"
            )
        for binding_id in required_ids:
            target = targets[binding_id]
            if set(target) != {"registry", "pointer"}:
                raise ContractError(f"Malformed target for {binding_id}")
            registry_name = target["registry"]
            if registry_name not in materialized:
                raise ContractError(f"Unknown target registry for {binding_id}")
            if get_pointer(materialized[registry_name], target["pointer"]) != PLACEHOLDER:
                raise ContractError(f"Target is not the registered placeholder: {binding_id}")
            set_pointer(materialized[registry_name], target["pointer"], bindings[binding_id])
    else:
        raise ContractError(f"Unknown registry source-binding mode: {binding_mode}")

    if any(count_placeholder(registry) for registry in materialized.values()):
        raise ContractError("Unresolved G2 placeholder remains")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if freeze_values_path.parent != output_dir or freeze_values_path.name != "registry_bindings.g2.json":
        raise ContractError(
            "G2 registry bindings must be output_dir/registry_bindings.g2.json"
        )
    outputs: dict[str, dict[str, str]] = {}
    for name, document in materialized.items():
        path = output_dir / f"{name}.materialized.json"
        payload = canonical_bytes(document)
        _write_once_or_identical(path, payload)
        outputs[name] = {"path": path.name, "sha256": hashlib.sha256(payload).hexdigest()}

    for name, relative in source_paths.items():
        if sha256_file(repo_root / relative) != source_hashes[name]:
            raise ContractError(f"Source registry changed during materialization: {name}")

    ledger = {
        "schema_version": "rq014-registry-materialization-ledger-g2-v1",
        "stage": "G2",
        "execution_contract": {
            "path": str(contract_path.relative_to(repo_root)),
            "sha256": sha256_file(contract_path),
        },
        "freeze_values": {
            "path": freeze_values_path.name,
            "sha256": sha256_file(freeze_values_path),
        },
        "materializer_sha256": sha256_file(Path(__file__).resolve()),
        "source_registries": {
            name: {"path": source_paths[name], "sha256": source_hashes[name]}
            for name in sorted(source_paths)
        },
        "bindings": {key: bindings[key] for key in sorted(bindings)},
        "outputs": {key: outputs[key] for key in sorted(outputs)},
    }
    ledger_path = output_dir / "materialization_ledger.json"
    ledger_payload = canonical_bytes(ledger)
    _write_once_or_identical(ledger_path, ledger_payload)
    ledger["ledger_path"] = str(ledger_path)
    ledger["ledger_sha256"] = hashlib.sha256(ledger_payload).hexdigest()
    return ledger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--freeze-values", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    contract = args.contract if args.contract.is_absolute() else repo_root / args.contract
    ledger = materialize(
        repo_root=repo_root,
        contract_path=contract,
        freeze_values_path=args.freeze_values,
        output_dir=args.output_dir,
    )
    print(json.dumps(ledger, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
