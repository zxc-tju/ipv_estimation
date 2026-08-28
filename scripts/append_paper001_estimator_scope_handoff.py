#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "reports/knowledge/PAPER001_online_sociality_verification_manuscript/imported_from_paper_repo_20260620/agent_handoff.md"
LOG = ROOT / "main_workflow.log"
MARKER = "## 2026-08-28 — Estimator validation moved to the published-method supplement"

text = HANDOFF.read_text(encoding="utf-8")
if MARKER in text:
    raise SystemExit("Handoff entry already exists")

entry = """

## 2026-08-28 — Estimator validation moved to the published-method supplement

Paper repository merge: `ff9a4f97d4427741628b65bb62833e9db3394419` (PR #10).

Files changed in the paper repository: `main.tex`, `claims_register.md`, `structure.md`, and `CLAUDE.md`.

Summary: Implemented the final RQ027 scope decision in the manuscript. The trajectory-based IPV estimator is now treated as a previously published method component, supported by Zhao, Sun and Wang, T-ITS 2024. The standalone estimator-validation Results subsection was removed. The main text carries a concise citation and keeps the monitoring framework figure; estimator implementation, interaction-specific diagnostics and episode-summary sensitivity now appear in Supplementary Note 1 and Supplementary Figure S1. Estimator and readability equations were moved into Methods. Candidate concentration is described as an operational readability/candidate-discrimination boundary, while the paper remains centred on the context-conditioned human-reference monitor.

Structure: Results now contain four semantic sections and six main figures. `claims_register.md` and `structure.md` advance to v4.3. RQ027's independent-generator pilot remains in the research repository as a bounded internal diagnostic and is not introduced into the manuscript's frontstage narrative.

Verification: paper PR CI passed source lint, LaTeX compilation, bibliography processing, undefined-reference/citation checks and PDF artifact generation.
"""
HANDOFF.write_text(text.rstrip() + entry.rstrip() + "\n", encoding="utf-8")

with LOG.open("a", encoding="utf-8") as handle:
    handle.write(
        "\n[2026-08-28 PAPER001 estimator-scope handoff]\n"
        "Paper PR #10 merged at ff9a4f97d4427741628b65bb62833e9db3394419. "
        "The manuscript now cites the published T-ITS estimator validation, moves detailed diagnostics to Supplementary Note 1, and removes the standalone recovery Results section.\n"
    )

print("PAPER001 handoff appended")
