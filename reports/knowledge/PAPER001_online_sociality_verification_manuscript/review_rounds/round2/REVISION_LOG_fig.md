# Round-2 Q24 figure execution log

## Context and status

Q24 updates title/label wording in manuscript Figures 2, 4 and 5 and makes one cosmetic marker/label pass in Figure 1. The existing restyle pipeline was used without changing any frozen data, upstream study generator, acceptance-checker logic, `main.tex`, or Fig. 6/`figH`. No `pdflatex`, commit, or push was run.

Final status: **COMPLETE**. The four figures regenerated successfully, the targeted acceptance run exited `0`, all four PNGs passed original-resolution visual inspection and localized pixel comparison, and the eight authorized PDF/PNG binaries were copied to the paper repository only after those checks passed.

All four supplied before-title strings matched the current scripts exactly; there were no source-string discrepancies.

## Change (i) — `fig1_measurable` (manuscript Fig. 2)

- Script edited: `T5_style_harmonisation/restyle_fig1_measurable.py`.
- Panel-a title changed from `The reading becomes readable only during a real interaction` to `The reading is identified most sharply during the real interaction`.
- Acceptance: A1 and A3–A7/A9 passed. A2 passed with 35 baseline numeric tokens plus the four pre-existing declared case-count additions (`4,743`, `4,605`, `4,701`, `8,130`) = 39 output tokens; all removal/missing/unexpected counters were empty.
- Visual check: the new title is fully visible on one line without touching the estimate/CI column. Pixel comparison against the pre-copy repository PNG found 34,037 changed pixels, all confined to title coordinates x=1,078–2,612 and y=135–196 on the 4,251 × 2,480 px canvas; changed pixels outside the declared title region: 0. Layout, marks, values and colours elsewhere are pixel-identical.
- Copied: `fig1_measurable.pdf` and `fig1_measurable.png` at `2026-08-10 23:40:14 +0800`.

## Change (ii) — `fig3_monitor` (manuscript Fig. 4)

- Script edited: `T5_style_harmonisation/restyle_fig3_monitor.py`.
- Panel-b title changed from `Coverage stays at nominal` to `Coverage within 0.6 pp of nominal`.
- Ledger edited: `T5_style_harmonisation/declared_additions.json`; `fig3_monitor.numeric_additions` now contains one `0.6` token for Q24.
- Acceptance: A1 and A3–A7/A9 passed. A2 passed with 42 baseline tokens plus the three pre-existing fixed additions (`461,937` twice and `1` once) plus the Q24 `0.6` token = 46 output tokens; all removal/missing/unexpected counters were empty.
- Visual check: the title is fully visible and separated from panels a and c. Pixel comparison found 15,274 changed pixels, all confined to title coordinates x=2,057–2,765 and y=158–218 on the 4,251 × 1,700 px canvas; changed pixels outside the declared title region: 0. Layout, data marks, values and colours elsewhere are pixel-identical.
- Copied: `fig3_monitor.pdf` and `fig3_monitor.png` at `2026-08-10 23:40:14 +0800`.

## Change (iii) — `fig5_consequence` (manuscript Fig. 5)

- Script edited: `T5_style_harmonisation/restyle_fig5_consequence.py`.
- Panel-b title changed from `Emergency margins stay at or below the within-range rate` to `Emergency margins are rarer at every supported threshold`.
- Panel-d title changed from `Counterpart braking stays at or below the within-range rate` to `Counterpart braking is rarer at every threshold`. The deliberate b/d wording difference is retained exactly.
- Acceptance: A1 and A3–A7/A9 passed. A2 passed with 50 baseline numeric tokens plus the 25 pre-existing declared additions minus the two P14 declared removals = 73 output tokens; all removal/missing/unexpected counters were empty.
- Visual check: both titles are fully visible as two-line headings, with no collision with denominators or adjacent panels. Pixel comparison found 51,840 changed pixels in exactly two title bands (y=29–158 and y=1,506–1,635; x=2,698–3,822) on the 4,251 × 2,952 px canvas; changed pixels outside the two declared title regions: 0. Layout, data marks, intervals, values and colours elsewhere are pixel-identical.
- Copied: `fig5_consequence.pdf` and `fig5_consequence.png` at `2026-08-10 23:40:14 +0800`.

## Change (iv) — `fig0_concept` (manuscript Fig. 1)

- Script edited: `T5_style_harmonisation/restyle_fig0_concept.py`.
- Panel-a conflict-zone marker radius increased from 4.0 m to 5.0 m; the fill alpha increased from 0.15 to 0.24 and a dark 1.1-pt outline was added.
- The four trajectory time labels were reduced from 6.5 pt to 5.8 pt and offset farther from their trajectory marks.
- Acceptance: A1 and A3–A7/A9 passed. A2 passed with 39 baseline tokens, the proved panel-a x-axis removal of `−20` and `60`, and the proved additions of `−10`, `10` and `30`, yielding 40 output tokens. The PDF-bound tick proof passed and all mismatch counters were empty.
- Visual check: the conflict-zone marker is conspicuous; `13.2 s`, `17.5 s`, `19.0 s` and `22.0 s` are all clear of both trajectories. Pixel comparison found 25,522 changed pixels confined to x=807–1,337 and y=485–1,321 on the 4,251 × 3,543 px canvas; changed pixels outside the declared marker/time-label region: 0. The remaining panels, layout, data marks and colours are pixel-identical.
- Copied: `fig0_concept.pdf` and `fig0_concept.png` at `2026-08-10 23:40:14 +0800`.

## Acceptance harness

Command:

```text
python3 check_acceptance.py \
  --only fig0_concept \
  --only fig1_measurable \
  --only fig3_monitor \
  --only fig5_consequence \
  --original-dir /tmp/q24-acceptance.E5b3Ed/figures
```

The comparison PDFs were extracted from paper commit `3776b44acd60dcb349a2426267e3fed6d1b43ec5`, the preserved pre-restyle acceptance baseline already used by the round-1 P14/P15 precedent.

- A1 files/signatures: PASS for all four PDF/PNG pairs.
- A2 numeric-token multiset: PASS for all four figures; Q24 adds exactly one `0.6` token.
- A3 banned wording: PASS, no hits.
- A4 font family: PASS, Arial only.
- A5 canvas: PASS; widths are exactly 180.000 mm and heights are 150/105/72/125 mm as specified.
- A6 minimum type: PASS; minimum is 5.3/6.5/6.5/6.5 pt for fig0/fig1/fig3/fig5.
- A7 palette: PASS; unknown-colour counter is empty for all four.
- A8 synthetic watermark: SKIP because figH was deliberately not selected.
- A9 canvas overflow: PASS; 0 out-of-MediaBox words for all four.
- Overall exit code: `0`.

## Deviations and protected-scope checks

1. `declared_additions.json` does not support addition records with `figure`, `count`, `plan_id` and `reason`; its live checker schema stores additions as string tokens under each figure, while only removals have typed metadata objects. The requested Q24 addition was therefore encoded as `fig3_monitor.numeric_additions: ["0.6"]`, exactly as the unchanged checker accepts. The plan ID and reason cannot be embedded without changing the checker and are recorded here instead: Q24; panel-b title states the frozen maximum coverage deviation (frozen deviations +0.03/+0.28/+0.57 pp).
2. A pre-edit run against the default originals directory failed A2 because that directory is the live paper repository and already contains the round-1 restyled binaries, while the ledger is cumulative from the pre-restyle baseline. The final run used the preserved Git baseline above, matching the existing P14/P15 protocol. No checker rule was weakened, bypassed or edited.
3. The three directly used frozen-input areas report no Git diff after generation. The scripts only read their source files.
4. `restyle_figH_human_arm.py` and both pipeline/repository figH binaries retain their pre-run SHA-256 values: script `8feaa1ce...9037`; PDF `a70513f5...e7d3`; PNG `bb1c5817...5f15`. FigH was neither regenerated nor copied.
5. `main.tex` was not written, and no TeX command was executed.

## Repository files copied

Destination: `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/2_PaperWriting/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle/figures/`

Every destination was verified byte-for-byte against `T5_style_harmonisation/out/` after copying.

| File | Timestamp | Bytes | SHA-256 |
|---|---|---:|---|
| `fig0_concept.pdf` | 2026-08-10 23:40:14 +0800 | 52,732 | `115372446fcdf07c7ffc548df47b50d087be470991861234187c8c7f27246dea` |
| `fig0_concept.png` | 2026-08-10 23:40:14 +0800 | 632,660 | `fdbc25f8639b0014b94f1df8d11813b3f7e8879153ffdb2e655cf36f98c7c587` |
| `fig1_measurable.pdf` | 2026-08-10 23:40:14 +0800 | 46,261 | `5cc9bffb864da483ae1bb4539cb6ecd1d153436b27b869c7d6f2526546ad2626` |
| `fig1_measurable.png` | 2026-08-10 23:40:14 +0800 | 437,082 | `bc9e1ee464d092270bf6e9d1fc2640e6c213a7aa5af937178bc37906f6e30d9f` |
| `fig3_monitor.pdf` | 2026-08-10 23:40:14 +0800 | 44,373 | `218de302e10ab86525163cbd5115b379ac0965e64f4faa65555ae95fb69dde87` |
| `fig3_monitor.png` | 2026-08-10 23:40:14 +0800 | 357,130 | `282d0134372810a4a20408040e262833d2cd1cad8080b6e8ac5764bced9346a3` |
| `fig5_consequence.pdf` | 2026-08-10 23:40:14 +0800 | 69,373 | `4357ce75ca2eb1eb6522724067f79c88e3b5a32011f86f2d42eda4d36a75cd5e` |
| `fig5_consequence.png` | 2026-08-10 23:40:14 +0800 | 690,825 | `123481d8aa7367c38268844b27d3a605f6cbb457e8cb0710894f80ae2953b8fd` |

