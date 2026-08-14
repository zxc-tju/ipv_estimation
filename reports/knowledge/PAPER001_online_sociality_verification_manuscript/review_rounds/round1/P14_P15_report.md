# P14/P15 round-1 figure report

## Context and overall status

This task makes two PI-gated, surgical changes to the NMI paper figures: P14 removes quotable ratio labels from unsupported steering rows in Figure 5, and P15 separates abstention from verdict paths in Figure 0. Both scripts were edited and only the two requested PDF/PNG pairs were regenerated in `out/`.

Status: **stopped at acceptance gate A2**. The supplied checker has no declaration path for non-tick numeric removals or plan IDs. In accordance with `P14_P15_TASK.md`, no regenerated figure was copied into the paper repository. The checker and `declared_additions.json` were not modified.

## P14 — `fig5_consequence`

Before: panel c printed `2.35×` above `not supported` for net heading change and `2.04×` above `not supported` for peak yaw rate. The grey point and grey 95% interval whisker were present on each row.

After: each unsupported row prints only the grey wording `not supported`. The grey point and interval whisker remain unchanged, and the source data are unchanged.

Direct PDF numeric-token comparison against the pre-task paper-repository figure:

- Removed: `2.35` once and `2.04` once.
- Added: none.

## P15 — `fig0_concept`

Before: mechanism 1's `no` branch and mechanism 2's verdict output joined a shared baseline that fed all three state boxes, so `no reading` could be read as part of the verdict path.

After: mechanism 1's grey `no` branch terminates directly and only at the grey `no reading` box. Mechanism 2 has separate teal and red arrows to `inside range` and `outside range · atypical · either side`. The observable-situation box remains an input to mechanism 2. Abstention is no longer presented as a verdict.

Direct PDF numeric-token comparison against the pre-task paper-repository figure:

- Removed: none.
- Added: none.

## Acceptance results

Command: `python3 check_acceptance.py`, with `PYTHONPYCACHEPREFIX` and `XDG_CACHE_HOME` set to task-specific `/tmp` directories.

Overall exit code: `1`.

- A1 files/signatures: PASS for all eight existing output pairs.
- A2 numeric-token multiset: **FAIL**. For `fig5_consequence`, the checker reports the intended `2.35` and `2.04` removals as missing. The full-set run also replays earlier addition/tick declarations against paper-repository figures that already contain those changes, producing pre-existing baseline/ledger mismatches for other figures, including Figure 0. A direct before/after comparison confirms that P15 changed no numeric tokens and P14 changed only the two intended tokens.
- A3 banned wording: PASS.
- A4 font family: PASS; both regenerated figures use Arial only.
- A5 canvas: PASS; `fig0_concept` is 180.000 × 150.000 mm and `fig5_consequence` is 180.000 × 125.000 mm.
- A6 minimum text size: PASS; 5.3 pt for `fig0_concept` and 6.5 pt for `fig5_consequence`.
- A7 palette: PASS; no unknown colours.
- A8 figH watermark: PASS; `SYNTHETIC TARGET`, `data_status=SYNTHETIC_TARGET`, and the exact header are intact.
- A9 canvas overflow: PASS; zero offending word boxes for both regenerated figures and all other outputs.
- Structured visual QA: PASS, score 96/100. Both requested changes are visually unambiguous, and no overlap or canvas overflow is visible.

The acceptance evidence is in `audit/A1_files_exist.txt` through `audit/A9_canvas_overflow.txt`. The visual verdict is in `.omx/state/p14_p15_style_harmonisation/ralph-progress.json`.

## Required explicit exemption

The current ledger schema accepts only `numeric_additions`, `tick_tokens_removed`, and `tick_tokens_added`; any `numeric_removals` or `plan_id` key is rejected as unknown. No checker rule was changed.

Proposed ledger declaration:

```json
"numeric_removals": [
  {
    "panel": "c",
    "token": "2.35",
    "count": 1,
    "plan_id": "P14",
    "reason": "unsupported row: total_heading_change_deg has excludes_one=false"
  },
  {
    "panel": "c",
    "token": "2.04",
    "count": 1,
    "plan_id": "P14",
    "reason": "unsupported row: max_abs_yaw_rate_dps has excludes_one=false"
  }
]
```

Proposed exemption wording:

> Plan P14 authorizes exactly one removal of numeric token `2.35` and exactly one removal of numeric token `2.04` from the panel-c text of `fig5_consequence`. These removals apply only to labels on rows whose `excludes_one` value is false. The plotted point, confidence-interval whisker, source data, axis ticks, and all other numeric tokens must remain unchanged.

Until that schema/rule is explicitly approved, A2 cannot pass for the requested P14 output.

## Files and delivery state

Edited scripts:

- `restyle_fig5_consequence.py`
- `restyle_fig0_concept.py`

Regenerated only in this scratch directory:

- `out/fig5_consequence.pdf`
- `out/fig5_consequence.png`
- `out/fig0_concept.pdf`
- `out/fig0_concept.png`

Paper-repository figure copies: **none**, because A2 failed and the task requires stopping at this gate. The existing four paper-repository target files retain their pre-task SHA-256 hashes.

All twelve non-target files in `out/` retained their pre-task SHA-256 hashes, including both figH files. `declared_additions.json` also remained unchanged (`SHA-256 5814b96689601c2c80bfbb9b4d66b0187db8bf6f46a0ea3fd7c971d7f7776eac`).

Report copy:

- `review_rounds/round1/P14_P15_report.md`

No Python bytecode cache modified within the inspected OneDrive source or paper-figure directories; render and checker caches were directed to task-specific `/tmp` paths. No commit or push was performed.
