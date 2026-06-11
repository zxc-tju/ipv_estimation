"""Regression checks for production IPV likelihood parameters."""

from __future__ import annotations

import ast
from pathlib import Path


def _agent_sigma_literal() -> float:
    source = Path(__file__).resolve().parents[1] / "agent.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "sigma" for target in node.targets):
            return float(ast.literal_eval(node.value))
    raise AssertionError("agent.py does not define module-level sigma")


def test_agent_likelihood_sigma_is_0_1() -> None:
    assert _agent_sigma_literal() == 0.1


if __name__ == "__main__":
    test_agent_likelihood_sigma_is_0_1()
