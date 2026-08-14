# Round-4 U08 + U09 figure execution log

## Context and status

U08 and U09 make two label-only changes in manuscript Figures 2 and 4 through the existing `T5_style_harmonisation` Python restyle pipeline. Frozen inputs, plotted data, marks, colours, panel geometry, the acceptance checker, `main.tex`, and protected Fig. H were not edited by this figure task. No `pdflatex`, Git commit, or push was run.

Final status: **COMPLETE**. Both regenerated figures passed the preserved-Git-baseline acceptance harness with exit code `0`, the requested strings were confirmed in the PDF text layers and original-resolution PNGs, all raster differences were confined to the intended label regions, and the four accepted PDF/PNG binaries were copied to the paper repository only after those checks passed.

## U08 — Fig. 2c flip-rate annotation

- Script: `T5_style_harmonisation/restyle_fig1_measurable.py`.
- Edit: panel-c annotation changed from `22% sign flips` to `7–22% sign flips across rule pairs`, using a typographic en dash. The final annotation uses two lines for the new phrase (`7–22% sign flips` / `across rule pairs`) at the existing 6.5 pt annotation size; the existing two preceding lines and anchor are unchanged.
- Ledger: `fig1_measurable.numeric_additions` gained exactly one string token, `"7"`, for plan item U08. The pre-ledger harness reported `unexpected={'7': 1}` and no missing token; the existing `22` remained one-for-one and therefore required no declaration.
- Mechanical source-scope proof: reconstructing the old source string in memory reproduces the pre-edit script SHA-256 `1a6e7d7cce3a2f04eb5e06837eeadebe7db45de34cc8008c8218f8aab68153cf`. Final script SHA-256: `95d11a956c1a2c30af7467f2b01a9fcc458e67e0dc657b51912542ef54b183f1`.
- Acceptance A2: original `35` numeric tokens; declared output additions `4605`, `4701`, `4743`, `8130`, and U08 `7`, once each; restyled `40`; `missing={}` and `unexpected={}`.
- PDF text-layer check: exact text `7–22% sign flips` present.
- Pixel check against the pre-copy repository PNG: unchanged `4251 × 2480` px canvas; `31,131` changed pixels; changed bounding box `x=3580–3974`, `y=1570–1795`; declared panel-c annotation region `x=3550–4000`, `y=1540–1820`; changed pixels outside the region: `0`.

## U09 — Fig. 4a candidate-span label

- Script: `T5_style_harmonisation/restyle_fig3_monitor.py`.
- Edit: panel-a x-axis label now reads on two lines: `Nominal coverage level` / `(candidate span 3π/4 ≈ 2.36 rad)`. This uses the figure's existing plain-text Arial typography and the unchanged 6.5 pt axis-label size.
- Layout choice and deviation: the first render appended the parenthetical to the vertical y-axis label. Numeric-token detection was correct, but A9 found five words extending `6.034 pt` beyond the left MediaBox. Following the plan's wrap/reposition allowance, the parenthetical was moved to the panel-a x-axis label. The accepted render has zero A9 offenders; no type-size reduction or geometry change was needed.
- Ledger: `fig3_monitor.numeric_additions` gained exactly `"3"`, `"4"`, and `"2.36"` for plan item U09. The pre-ledger harness reported `unexpected={'2.36': 1, '3': 1, '4': 1}`, exactly matching the final declarations.
- Mechanical source-scope proof: reconstructing the old source string in memory reproduces the pre-edit script SHA-256 `5f5cfcbfd84f308f1a66989dbee22b44b93a9827eecd558dfc1434ec5cd55aed`. Final script SHA-256: `c76128851ea4530ad49f881e82dad2ff160be7c640b3fc35a1a9f4c85fa593e3`.
- Acceptance A2: original `42` numeric tokens; declared/fixed additions `0.6` once, `1` once, `461937` twice, and U09 `2.36`, `3`, and `4` once each; restyled `49`; `missing={}` and `unexpected={}`.
- PDF text-layer check: exact text `(candidate span 3π/4 ≈ 2.36 rad)` present.
- Pixel check against the pre-copy repository PNG: unchanged `4251 × 1700` px canvas; `12,204` changed pixels; changed bounding box `x=395–1241`, `y=1510–1564`; declared panel-a axis-label region `x=360–1280`, `y=1480–1590`; changed pixels outside the region: `0`.

## Declarations ledger and checker integrity

- `declared_additions.json` changed only by the four U08/U09 addition strings above. Reconstructing those removals in memory reproduces its pre-edit SHA-256 `ce54f9b586192083eb3a984d92712d045e38469c259013953b1eb56e598aa455`; final SHA-256: `a951fdee373e352dcc1afc9eba43d45eff8cbcb4517df9010f01b4fe3af6902e`.
- `check_acceptance.py` was not edited; final SHA-256 equals the pre-run value: `492fdca2fae902b07f39924fca78c73c27f250d208a68e6901cf63a94c86c6fe`.
- The ledger schema accepts additions only as string tokens under each figure; U08 and U09 provenance is therefore recorded in this log rather than embedded as unsupported `plan_id` fields.

## Final acceptance output

Command:

```text
python3 check_acceptance.py \
  --only fig1_measurable \
  --only fig3_monitor \
  --original-dir /tmp/u08-u09-acceptance.YpLom3/figures
```

The comparison PDFs were extracted from paper commit `3776b44acd60dcb349a2426267e3fed6d1b43ec5`, matching the preserved Git-baseline P14/P15/Q24/S14 protocol.

```text
exit_code=0
PASS A1 — files exist and have valid signatures
PASS fig1_measurable: PDF[bytes=46363 signature=b'%PDF-' eof=True] PNG[bytes=445273 signature=True iend_terminator=True]
PASS fig3_monitor: PDF[bytes=45079 signature=b'%PDF-' eof=True] PNG[bytes=374694 signature=True iend_terminator=True]
PASS A2 — numeric-token multiset unchanged except declared additions/removals
PASS fig1_measurable: original=35 declared={'4605': 1, '4701': 1, '4743': 1, '7': 1, '8130': 1} declared_removed={} tick_removed={} tick_added={} restyled=40 unavailable_removed={} undeclared_removed={} overdeclared_removed={} missing={} unexpected={} tick_proof=not_required
PASS fig3_monitor: original=42 declared={'0.6': 1, '1': 1, '2.36': 1, '3': 1, '4': 1, '461937': 2} declared_removed={} tick_removed={} tick_added={} restyled=49 unavailable_removed={} undeclared_removed={} overdeclared_removed={} missing={} unexpected={} tick_proof=not_required
PASS A3 — banned wording absent
PASS fig1_measurable: banned_hits=none
PASS fig3_monitor: banned_hits=none
PASS A4 — single Arial or Helvetica family
PASS fig1_measurable: faces=['Arial-BoldMT', 'ArialMT'] family=Arial
PASS fig3_monitor: faces=['Arial-BoldMT', 'ArialMT'] family=Arial
PASS A5 — canvas dimensions
PASS fig1_measurable: width=180.000 mm target=180.0±0.5; height=105.000 mm target=105.0±3.0
PASS fig3_monitor: width=180.000 mm target=180.0±0.5; height=72.000 mm target=72.0±3.0
PASS A6 — minimum text size at least 5 pt
PASS fig1_measurable: text_operators=68 min_size_pt=6.5
PASS fig3_monitor: text_operators=64 min_size_pt=6.5
PASS A7 — palette conformance
PASS fig1_measurable: colors={'#000000': 3, '#242424': 34, '#4B5C9B': 41, '#5A5A5A': 2, '#606060': 15, '#8A8F98': 23, '#C3C8CE': 13, '#FFFFFF': 13} unknown={}
PASS fig3_monitor: colors={'#000000': 15, '#242424': 21, '#2C737A': 39, '#4B5C9B': 6, '#5A5A5A': 2, '#606060': 13, '#8A8F98': 38, '#FFFFFF': 33} unknown={}
SKIP A8 — synthetic watermark intact
figH not selected
PASS A9 — no word box extends beyond the MediaBox
PASS fig1_measurable: words=143 offenders=0 MediaBox_checked=1 page(s)
PASS fig3_monitor: words=119 offenders=0 MediaBox_checked=1 page(s)
```

## Visual verdict

- Original-resolution inspection confirmed both requested label changes are present, legible, non-overlapping, and inside the page.
- Structured visual-verdict score: `98/100`, threshold `90`, verdict `pass`.
- The pixel checks above show zero differences outside the two declared text regions, so data marks, colours, and all other panel regions are pixel-identical to the pre-copy repository PNGs.

## Protected Fig. H SHA-256 verification

All three values matched before generation, after final rendering, and after the target-file copy. The Fig. H PDF/PNG values also match between pipeline `out/` and the paper repository.

| Protected file | SHA-256 |
|---|---|
| `restyle_figH_human_arm.py` | `8feaa1ce7aaadda0891bcff5fd176891898f5cd4c99431863c8a632f37ad9037` |
| `figH_human_arm_TARGET_SYNTHETIC.pdf` | `a70513f58d4c6a3962511f6797030d7537d13405d5f4010f5e904d6e1309e7d3` |
| `figH_human_arm_TARGET_SYNTHETIC.png` | `bb1c5817ac7c1adf3230d870a055cf01269226ab4a9465c4e7e1b585f3835f15` |

## Files copied last

Destination: paper repository `figures/`. Each destination passed byte-for-byte `cmp` against `T5_style_harmonisation/out/` after copying.

| File | Timestamp | Bytes | SHA-256 |
|---|---|---:|---|
| `fig1_measurable.pdf` | 2026-08-11 08:10:54 +0800 | 46,363 | `867bc8f3343d90b2ebd81ef4e293b7b7cdc6eedbb7bdcd9b91505f9a2169136a` |
| `fig1_measurable.png` | 2026-08-11 08:10:54 +0800 | 445,273 | `f9b186c128a256f7fd0f63878ec7d6a070eae1fd59fe3c2194b324d27b0f5854` |
| `fig3_monitor.pdf` | 2026-08-11 08:10:54 +0800 | 45,079 | `7a79813c24656aafb1aad75907b37f8b9898ca78071a88ce6610cd5e12c090e7` |
| `fig3_monitor.png` | 2026-08-11 08:10:54 +0800 | 374,694 | `b0f71ecad5e0310550edc2fa189fd52a786413a249c6eda7c11226d6aa46aab9` |

## Execution deviations and repository boundary

1. The shell has no `python` command, so the pipeline and checker were run with `/usr/bin/python3`; the selected plotting backend remained Python throughout.
2. The initial U09 y-axis placement failed A9 only, as documented above; the accepted x-axis placement preserved the minimum text size and label-only scope.
3. A concurrent `main.tex` modification appeared in the paper worktree after the four-file copy. This figure task did not read, write, compile, or stage `main.tex`; the external modification was preserved untouched. The only paper-repository writes made here were the four listed figure binaries.
4. No TeX command, commit, push, staging operation, or Fig. H regeneration/copy was performed.
