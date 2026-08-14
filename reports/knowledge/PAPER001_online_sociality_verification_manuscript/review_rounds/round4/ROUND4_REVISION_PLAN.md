# Round-4 revision plan (U01–U10)

Base: `main.tex` @ ad18609 on `paper/human-arm-target`. Paper repo:
`/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/2_PaperWriting/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle/`
Research repo (evidence): `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/`

Global constraints screened on EVERY item: the 14 `\targetnum{...}` call sites stay byte-identical
(no U-item touches a line containing one — verified per item below); figH watermark and C7 discipline
untouched; no `estimab*`, no standalone "passive", no model codenames or audit vocabulary enters the
main text; no number changes except to match frozen files (cited per item); abstention never rendered
neutral; no self-weakening framing — every correction below is an accuracy repair or a precision
upgrade. E2-gated sentences (abstract "defining population", contribution 6, §2.5 title, "survives
the move", "sits at the native level") and the abstract adjective "narrow" (new E15) are NOT edited
by this plan — they await the PI (ESCALATIONS.md). The E11 companion block in Methods 4.5 is NOT
edited (charter: attacks on the printed p=0.1493 are CLARIFY/response-letter, not text retreat).

W1 note: the Fig-5 float oversize watch item was NOT flagged by any round-4 reviewer; per the charter
conditional, no layout item is issued. The remedy (caption trim to a Methods cross-reference, or a
dedicated float page) stays on the shelf.

Execution: U01–U07 and U10's micro-edit are `main.tex` edits (captions live in main.tex). U08–U09 are
figure-binary changes through the existing restyle pipeline + acceptance harness
(`.codex-fleet/paper-figure-upgrade/work/T5_style_harmonisation/`), label-only, with the numeric-token
delta declared to the harness. Compile after each block; run the standing verification battery at the
end (targetnum diff 14/14 byte-identical, vocabulary greps, single References heading, compile exit 0).

---

## Correctness first

### U01 — Repair the LODO-direction sentence in the Fig-3 caption
- Target: `main.tex` Fig-3 (fig2_context) caption, L315-317.
- Anchor: `predicting the held-out source leaves the variance explained at or barely`
- Current: `Fitting the full state description on every source but one and predicting the held-out source leaves the variance explained at or barely above zero in all four folds.`
- Edit: `Fitting the full state description on every source but one and predicting the held-out source leaves the variance explained never above $+0.03$ in any fold and below zero in two ($+0.026$ Waymo, $+0.017$ Lyft, $-0.195$ Argoverse-2, $-0.276$ nuPlan).`
- Evidence: `reports/studies/RQ004_ipv_state_space/RQ004_1_state_space_law_nature_20260618/02_process/agent_I_figures/F7_lodo_summary_source_data.csv` (case_mean_ipv rows, column `full_state_space_r2`: +0.0262 / +0.0175 / −0.1945 / −0.2760); the four values are already embedded in the panel (pdftotext of `figures/fig2_context.pdf`).
- Constraints: mandatory accuracy repair ("at or barely above zero" is false for two folds); the no-transferable-law point becomes stronger, so this is press-conference-positive; no `\targetnum` in the caption; numbers copied from the frozen figure source at figure precision.
- Resolves: R4-31 (D-m1); consistent with A-M2's reading of the panel.

### U02 — Disambiguate the Fig-6 caption asterisk sentence
- Target: `main.tex` Fig-6 (figH) caption, L527-528.
- Anchor: `whiskers are case-bootstrap $95\%$ intervals where defined, and`
- Current: `...whiskers are case-bootstrap $95\%$ intervals where defined, and an asterisk marks an interval excluding no difference.`
- Edit: `...whiskers are case-bootstrap $95\%$ intervals where defined, and an asterisk marks a ratio whose interval excludes parity (the no-difference value, ratio $1$); entries whose intervals admit parity carry no asterisk.`
- Evidence: the caption's own panel definition ("expressed ... as the flagged-to-within ratio"), so the no-difference value is 1; semantics unchanged, wording disambiguated (the current phrase parses two ways and reviewer D read the inverse of the intended, standard convention).
- Constraints: caption text only — the Fig-6 caption contains no `\targetnum` (verified L521-534); figure binary, SYNTHETIC watermark and C7 comment block untouched; the numeric-interval display demand (C-m7's second half, A-m9) is logged under E10 for the swap-time regeneration, not executed here.
- Resolves: R4-32 (D-m2; C-m7 wording half).

### U03 — Replace the round-3 "certifies" verb (regression repair)
- Target: `main.tex` Methods 4.4, L669.
- Anchor: `so the conformal step certifies, rather than repairs, the situational quantile model.`
- Current: `The fitted radii are near zero ($c_\alpha = 1.4\times10^{-3}$, $1.2\times10^{-6}$ and $0.0$\,rad at the $80\%$, $90\%$ and $95\%$ levels), so the conformal step certifies, rather than repairs, the situational quantile model.`
- Edit: `The fitted radii are near zero ($c_\alpha = 1.4\times10^{-3}$, $1.2\times10^{-6}$ and $0.0$\,rad at the $80\%$, $90\%$ and $95\%$ levels), so the conformal step finds essentially nothing to repair in the situational quantile model.`
- Evidence: radii unchanged (`RQ021 .../key_numbers.json` human_only_envelope conformal_radii: 0.0014435 / 1.1921e-06 / 0.0). The verb "certifies" was inserted by round-3 S12 (verified via `git diff a4f99f8..ad18609`) and is attacked by B-M2/C-M2 as claiming a guarantee the construction disclaims (the marginal/empirical estimand is printed twice in the same subsection).
- Constraints: keeps the positive fact (the quantile model needs no repair); removes only the guarantee-flavoured verb; no numbers change; not a concession — the exchangeability reconstruction demand stays B05.
- Resolves: R4-09 (B-M2 facet, C-M2 facet).

## Consistency

### U05 — De-collide "supported" from the human-support gate (three micro-edits; abstract untouched)
- Target 1: `main.tex` Fig-5 caption, L443-444.
  - Anchor: `A threshold is called supported when its per-endpoint interval excludes no difference; the`
  - Edit: `A threshold is called supported when its per-endpoint interval excludes a zero difference --- a statement about the interval, distinct from the monitor's human-support gate; the panel titles use the word in exactly this sense.`
- Target 2: `main.tex` §2.4, L474.
  - Anchor: `at every threshold whose interval excludes no difference (margins under`
  - Edit: replace `excludes no difference` with `excludes a zero difference` (rest of the sentence unchanged).
- Target 3: `main.tex` §2.6, L542-543.
  - Anchor: `at the supported thresholds they are one-third`
  - Current: `---at the supported thresholds they are one-third to one-half as frequent---`
  - Edit: `---at the thresholds whose intervals exclude a zero difference (Section 2.4) they are one-third to one-half as frequent---`
- Evidence: fig5_data.json `ego_danger_bootstrap` / braking thresholds (three margin bins and all three braking bins exclude zero; <3 s does not — verified round 3); no numbers change.
- Constraints: the abstract's E3-decided phrase "at any supported threshold" is NOT touched; the Fig-5 panel titles (round-2 retitles) are NOT touched; "excludes a zero difference" removes the double-parse ambiguity that also afflicted the Fig-6 caption; the definition sentence now explicitly separates the two senses of "support" (B-m11's collision point).
- Resolves: R4-21 term-collision half (B-m11; D-m3 wording half); B-M7's multiplicity/non-inferiority demand stays B08.

### U07 — Relocate and extend the Methods vocabulary block
- Target: `main.tex` Methods 4.4, L670-673 (current location) → immediately after the first sentence of Methods 4.4 (the split-conformal formula sentence ending `...over accepted calibration moments.`).
- Anchor (block to move): `Vocabulary, used consistently: an anchor row is one agent-moment eligible for`
- Edit: move the existing vocabulary sentence unchanged, and append to it: `A scene is one contiguous recorded segment and the unit of the frozen split; an interaction case is one interacting vehicle pair within a scene (Section 2.2); on the benchmark, a scenario is one of the staged templates and a scenario run is one system's or one driver's traversal of it (Section 2.4).`
- Evidence: definitional only — every fact restates printed protocol statements (whole-scene splits, §2.2 interaction cases, §2.4 scenario templates/runs); no numbers introduced (deliberately avoids re-printing "15", which is plain text for the AV arm but `\targetnum`-wrapped for the human arm).
- Constraints: no `\targetnum` lines touched; resolves the defined-after-use ordering (the block currently sits after "accepted calibration moments" has been used four times); two-layer discipline unaffected.
- Resolves: R4-41 (C-m2, D-m9).

## Completeness (frozen numbers surfaced)

### U04 — State the absolute width beside the relative sharpening in §2.3
- Target: `main.tex` §2.3, L346 (after `...spans essentially the whole admissible scale and can never flag a moment.` and before the S05 marginal-coverage sentence).
- Edit: insert: `In absolute terms the conditioned $90\%$ range still averages $1.87$\,rad (5th--95th percentiles $1.35$--$2.28$\,rad) of the $2.36$\,rad admissible span: the reference is wide, and a flag therefore marks a clear departure from the human range rather than a fine discrimination.`
- Evidence: `RQ021 .../key_numbers.json` (mean 90% width 1.87 rad; percentiles 1.35–2.28; admissible span 2.3562 = 3π/4) — all three numbers already printed in the Fig-4 caption; this surfaces them beside the main-text relative claim.
- Constraints: press-conference-positive framing (a wide reference = a conservative instrument whose flags are meaningful — this is the paper's own abstention-discipline ethos, not a concession); D-M8 explicitly demands this surfacing; the abstract adjective "narrow" is NOT touched here (E15 decision); no `\targetnum` nearby.
- Resolves: R4-49 executable half (D-M8, D-P6 part).

### U06 — Disclose the ED2 tie-break in the caption
- Target: `main.tex` Fig ED2 caption, L856-858.
- Anchor: `(at least five flagged frames in one contiguous\nstretch; counterpart speed decrease of at least $20\%$; automated-vehicle speed change within $10\%$).`
- Edit: append after that sentence: `Of the two runs meeting all criteria, the displayed one ranks first under the selection score fixed together with them --- the counterpart's speed-drop fraction minus the automated vehicle's absolute speed-change fraction ($0.675$ vs $0.552$).`
- Evidence: `.codex-fleet/paper-figure-upgrade/work/T3_fig5_case/case_screening_all120.csv` (rank 1 T17:B3 score 0.6746; rank 2 T18:B3 score 0.5518, both eligible) and `notes.md` ("Selection score: S = R_c − R_a"; "thresholds were fixed before choosing the primary and backup"; ranking key: eligible, score desc, alert count desc, case ID).
- Constraints: numbers copied from the frozen screening ledger; discloses a pre-fixed deterministic rule (integrity-positive); no `\targetnum`; the run identity stays undisclosed (no ranking of systems — consistent with the caption's existing no-ranking sentence).
- Resolves: R4-36 (A-m11).

## Polish

### U08 — FIX-FIGURE: harmonise the Fig-2c flip-rate annotation
- Target: restyle pipeline `T5_style_harmonisation/restyle_fig1_measurable.py` → regenerate `figures/fig1_measurable.pdf/.png`.
- Current in-panel text: `22% sign flips`; caption says flips span `$7$--$22\%$ ... depending on which pair of rules is compared`.
- Edit: panel-c annotation → `7--22% sign flips across rule pairs`. Label-only; no marks, data or colours change; run the acceptance harness with the declared numeric-token delta (token `22` → tokens `7`, `22`); pixel diff confined to the label region; figH untouched (SHA check).
- Evidence: the 7–22% range is caption-printed and frozen (episode-summary rule comparison, Methods 4.2); no per-pair value is added.
- Constraints: FIX-FIGURE eligibility (label-only, numbers already frozen/printed); per-pair flip rates with uncertainty remain BACKLOG B09.
- Resolves: R4-26 (A-m4 display half).

### U09 — FIX-FIGURE: name the width denominator on the Fig-4a axis
- Target: restyle pipeline `T5_style_harmonisation/restyle_fig3_monitor.py` → regenerate `figures/fig3_monitor.pdf/.png`.
- Current: panel-a widths are shown as a percentage of the admissible range; the denominator is named only in the caption and Methods 4.1.
- Edit: append to the panel-a axis (or subtitle) label: `(candidate span $3\pi/4 \approx 2.36$ rad)`. Label-only; declared numeric-token addition (`3`, `4`, `2.36`); pixel diff confined to the label region; figH untouched.
- Evidence: Fig-4 caption already prints "the span of the candidate grid, $3\pi/4\approx2.36$\,rad" (RQ021 key_numbers admissible span 2.3562).
- Constraints: FIX-FIGURE eligibility (label-only, caption-frozen numbers); reviewers missed the printed reconciliation twice (round 3 B-m1/C-m5, round 4 A-m6), so axis surfacing has demonstrated value; grid-boundary sensitivity stays B07.
- Resolves: R4-28 (A-m6 axis half).

### U10 — CLARIFY: E11-attack response package (response letter + one cross-reference token)
- Target (text, minimal): `main.tex` §2.4, L427.
  - Anchor: `the association survives a\nplacebo test that reassigns whole flag sequences across scenario runs`
  - Edit: append `(Methods 4.5)` → `...the association survives a placebo test that reassigns whole flag sequences across scenario runs (Methods 4.5); a single run containing flagged...`
- Target (response letter): draft the round-4 rebuttal paragraph for A-M3/C-M6/D-M4's attacks on the printed p=0.1493:
  1. The two tests answer different questions, and the manuscript says which is which (Methods 4.5: "the exposure placebo above is the test specific to flag timing"). The case-label permutation destroys case composition, not timing; the exposure placebo preserves composition and permutes timing — timing-specificity is the placebo's role.
  2. The magnitude evidence does not rest on either p-value: it rests on the fixed three-second-window case-clustered intervals excluding zero at all three levels with six-combination sign agreement (`RQ018 decision.md` KC-C1), with the contract-window CI printed as the acknowledged boundary.
  3. Elevating the full inferential record into Results would invert the paper's two-layer design; the Results sentence carries the qualitative claim and now points directly at the Methods record.
  4. Include the standing reviewer-error note: the ED1 "with with" title does not exist in the PDF text layer or the figure (third consecutive round; three reviewers this round) — `manuscript_ad18609.txt` grep evidence.
- Evidence: `RQ018_abnormal_ipv_degradation/decision.md` L40 + B5 (Required Qualification wording, verified register-exact in the manuscript); `rq018_rerun/negative_controls.json` (draws=200, statistic, seeds).
- Constraints: charter-mandated disposition — CLARIFY with response-letter framing, NO text retreat and NO edit to the E11 companion block; the added cross-reference is a pointer, not a claim change.
- Resolves: R4-05 (A-M3 facet, A-q7; C-M6 facet, C-q11; D-M4 facet); R4-35 letter note (A-m10, B-m14, C-m10).

---

## Not in this plan (by design)

- E2-gated wording (five named sites) and the new E15 abstract adjective "narrow"; all Fig-6/human-arm
  numeric material (E10/C7); ethics content (E5); declarations + TOPS disclosure (E8/E14); benchmark
  facts incl. counterpart policy and the 267-vs-285 ledger (E6); release vehicle (E7): see ESCALATIONS.md.
- All new-analysis demands: BACKLOG (B01–B26 standing; B27–B29 new this round).
- The E11 companion block itself: stands as printed (PI-ratified register-exact wording).

## Post-execution verification battery (same as round 3, plus)

1. 14 `\targetnum{` occurrences byte-identical to ad18609; abstract diff empty.
2. `estimab* = 0`; standalone `passive = 0`; no M0–M5/CQR/source-guard/claims-register tokens in main text.
3. Compile exit 0; single References heading; ED order intact; 36-page ballpark.
4. NEW greps: `at or barely above zero` gone (U01); `asterisk marks an interval excluding no difference` gone (U02); `certifies, rather than repairs` gone (U03); `excludes no difference` count = 0 in main.tex (U05 covers both remaining sites); `Vocabulary, used consistently` appears before the estimand sentence (U07); `1.87` present in §2.3 body (U04); `0.675` present in the ED2 caption (U06).
5. U08/U09 only: acceptance harness exit 0; pixel diffs confined to declared label regions; declared numeric-token deltas match; figH SHA-256 unchanged.
6. E11 block unchanged: grep `0.1493`, `-2.6100`, `+0.1372`, `0.0199` each exactly once in main.tex.
