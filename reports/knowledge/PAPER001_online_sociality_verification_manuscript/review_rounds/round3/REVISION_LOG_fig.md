# Round-3 S14 figure execution log

## Context and status

S14 makes one wording-only change to the panel-c annotation in manuscript Figure 1. The existing `T5_style_harmonisation` Python restyle pipeline was used without changing frozen data, plotted marks, colours, the acceptance checker, its declarations, `main.tex`, or Fig. H. No `pdflatex`, Git commit, or push was run.

Final status: **COMPLETE**. The regenerated `fig0_concept` passed the targeted acceptance harness with exit code `0`, the PNG passed original-resolution visual and localized pixel checks, Fig. H remained byte-identical, and the PDF/PNG pair was copied to the paper repository only after those checks passed.

## Script edit and layout choice

- Script: `T5_style_harmonisation/restyle_fig0_concept.py`.
- Panel-c annotation changed from `outside the human reference range` to `outside the human reference range (over-yielding side)`.
- The final label is rendered on two lines, with `(over-yielding side)` on the second line. A first one-line render failed only A9 because the last word extended beyond the right MediaBox (`word='sid'`, `xMax=513.151 pt` versus `510.236 pt`). The permitted two-line wrap removed the overflow while keeping the label clear of data marks and reference bands.
- Mechanical scope proof: replacing the new source string with the old string in memory reproduces the pre-edit script SHA-256 exactly (`a15791d8ed4951a66bbfc311ed811fb4f56d6d38f99d6a6a9d3f975464af2587`). Final script SHA-256: `9ea59586b1d4d96a8b94036214b4e49c1fd8b21430e7169e21b9a701ed1f5013`.

## Acceptance output

Command:

```text
python3 check_acceptance.py --only fig0_concept --original-dir /tmp/s14-acceptance.ZaeXi7/figures
```

The comparison PDF was extracted from paper commit `3776b44acd60dcb349a2426267e3fed6d1b43ec5`, matching the preserved Git-baseline P14/P15/Q24 protocol. The checker and declarations were not edited; their SHA-256 values at completion were `492fdca2fae902b07f39924fca78c73c27f250d208a68e6901cf63a94c86c6fe` and `ce54f9b586192083eb3a984d92712d045e38469c259013953b1eb56e598aa455`.

```text
exit_code=0
PASS A1 — files exist and have valid signatures
PASS fig0_concept: PDF[bytes=52834 signature=b'%PDF-' eof=True] PNG[bytes=643002 signature=True iend_terminator=True]
PASS A2 — numeric-token multiset unchanged except declared additions/removals
PASS fig0_concept: original=39 declared={} declared_removed={} tick_removed={'-20': 1, '60': 1} tick_added={'-10': 1, '10': 1, '30': 1} restyled=40 unavailable_removed={} undeclared_removed={} overdeclared_removed={} missing={} unexpected={} tick_proof=PASS path=tick_token_proof_fig0_concept.json removed={'-20': 1, '60': 1} added={'-10': 1, '10': 1, '30': 1}
PASS A3 — banned wording absent
PASS fig0_concept: banned_hits=none
PASS A4 — single Arial or Helvetica family
PASS fig0_concept: faces=['Arial-BoldMT', 'ArialMT'] family=Arial
PASS A5 — canvas dimensions
PASS fig0_concept: width=180.000 mm target=180.0±0.5; height=150.000 mm target=150.0±3.0
PASS A6 — minimum text size at least 5 pt
PASS fig0_concept: text_operators=75 min_size_pt=5.3
PASS A7 — palette conformance
PASS fig0_concept: colors={'#000000': 9, '#242424': 26, '#2C737A': 19, '#4B5C9B': 30, '#606060': 17, '#8A8F98': 17, '#9AA0A6': 5, '#9CC6CA': 2, '#B64342': 13, '#C6DFE1': 4, '#D98884': 5, '#E6F1F1': 4, '#E9EBEC': 12, '#FFFFFF': 14} unknown={}
SKIP A8 — synthetic watermark intact
figH not selected
PASS A9 — no word box extends beyond the MediaBox
PASS fig0_concept: words=142 offenders=0 MediaBox_checked=1 page(s)
```

## Visual and pixel checks

- Original-resolution PNG inspection confirmed both requested text lines are present, legible, and non-overlapping with the time-series marks, annotation leader, or reference bands.
- Pixel comparison used the pre-copy repository PNG on the unchanged `4251 × 3543` px canvas.
- Changed pixels: `9,596`; changed bounding box: `x=2956–3464`, `y=1936–2161`.
- Declared panel-c label region: `x=2950–4000`, `y=1930–2200`; changed pixels outside this region: `0`.
- Therefore all raster changes are confined to the annotation label region; marks, colours, and layout elsewhere are pixel-identical.

## Fig. H protected-scope verification

The following SHA-256 values matched before generation and after final rendering. The PDF and PNG values were also identical between pipeline `out/` and the paper repository:

| Protected file | SHA-256 |
|---|---|
| `restyle_figH_human_arm.py` | `8feaa1ce7aaadda0891bcff5fd176891898f5cd4c99431863c8a632f37ad9037` |
| `figH_human_arm_TARGET_SYNTHETIC.pdf` | `a70513f58d4c6a3962511f6797030d7537d13405d5f4010f5e904d6e1309e7d3` |
| `figH_human_arm_TARGET_SYNTHETIC.png` | `bb1c5817ac7c1adf3230d870a055cf01269226ab4a9465c4e7e1b585f3835f15` |

## Files copied

Destination: `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/2_PaperWriting/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle/figures/`

Both destination files were verified byte-for-byte against pipeline `out/` after copying.

| File | Timestamp | Bytes | SHA-256 |
|---|---|---:|---|
| `fig0_concept.pdf` | `2026-08-11 06:39:21 +0800` | 52,834 | `d6c14383c2207702280ccc2d5cf6379d7056d9b54d33b07b4540c83d4f1f43cb` |
| `fig0_concept.png` | `2026-08-11 06:39:21 +0800` | 643,002 | `773d56cb521a5ac9d6e6a6d64307b10c1eac5118bbba6c7874eb3ef3f751e759` |

## Deviations

1. The `python` command is unavailable on this host; the installed Python 3 runtime at `/Library/Developer/CommandLineTools/usr/bin/python3` was used for all rendering and visual QA. No alternate plotting backend was used.
2. The first literal one-line label overflowed the right MediaBox and failed A9; the allowed two-line wrap was used, after which the complete acceptance run exited `0`.
3. The paper worktree was clean at the initial status check. A concurrent modification to `main.tex` appeared later during this run. This figure task did not read, edit, revert, compile, stage, or commit that file; the external change was preserved untouched.
