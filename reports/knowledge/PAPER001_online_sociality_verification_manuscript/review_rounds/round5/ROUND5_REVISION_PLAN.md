# Round-5 revision plan (V01–V10) — FINAL round

Base: `main.tex` @ 985a757 on `paper/human-arm-target`. Paper repo:
`/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/2_PaperWriting/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle/`
Research repo (evidence): `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/`

Global constraints screened on EVERY item: the 14 `\targetnum{...}` call sites stay byte-identical
(no V-item touches a line containing one — verified per item); figH SYNTHETIC watermark and C7
discipline untouched; the E11 companion block in Methods 4.5 is NOT edited (grep quadruple
0.0199 / 0.1493 / −2.6100 / +0.1372 must stay exactly once each); no `estimab*`, no standalone
"passive", no model codenames or audit vocabulary enters the main text; numbers appear only as
copies from frozen files (cited per item); abstention never rendered neutral; no self-weakening
framing — every item is an accuracy repair, a precision upgrade, or a frozen-evidence surfacing.
E2-gated sites (abstract "defining population", §2.5 title, "survives the move", "sits at the
native level", contribution 6) and the abstract are NOT edited by this plan. All figure binaries
untouched this round (0 FIX-FIGURE items — no figure defect was flagged).

---

## Correctness first

### V01 — Repair the §2.3 "issued ... abstains" self-contradiction
- Target: `main.tex` L351-352.
- Anchor: `Where a reading is issued, it abstains for lack of human support on only a small`
- Current: `Where a reading is issued, it abstains for lack of human support on only a small fraction of moments ($5.08\%$ on the held-out natural-driving test fold; abstention on the real-vehicle benchmark is larger and is reported with its reasons in Methods), ...`
- Edit: `Where a reading is available, the monitor abstains for lack of human support on only a small fraction of readable moments ($5.08\%$ on the held-out natural-driving test fold; abstention on the real-vehicle benchmark is larger and is reported with its reasons in Section 2.4 and Methods), ...`
  (This single edit also executes V07 — same sentence, one pass.)
- Evidence: 5.08% = 24,723/486,660 where 486,660 is the readable test-fold row count (reference-pool rows are readable by construction: `RQ021_2_.../key_numbers_e2.json` pool_counts.fold_human_rows.test = 486660; Methods 4.4 prints "support abstention is 5.08% (24,723 of 486,660)"). §2.4 carries the benchmark two-level reasons (55.3% / 20.8% + 36-run accounting), so the pointer "Section 2.4 and Methods" is accurate.
- Constraints: wording repair only, no number changes; no `\targetnum` on the line; readability vocabulary (not `estimab*`).
- Resolves: R5-45 (D-m1) + R5-21 pointer half (D-M4).

### V02 — Align the U04 sentence to "candidate span" (regression seam)
- Target: `main.tex` L348.
- Anchor: `(5th--95th percentiles $1.35$--$2.28$\,rad) of the $2.36$\,rad admissible span: the reference is wide,`
- Edit: replace `admissible span` with `candidate span` (rest unchanged).
- Evidence: Fig-4 caption L365 defines "The admissible range is the span of the candidate grid, $3\pi/4\approx2.36$ rad"; Methods 4.1 L618 states "the admissible span quoted in Fig. 4 is the candidate span $3\pi/4$"; the round-4 U09 axis label reads "candidate span $3\pi/4\approx2.36$ rad". One name for one quantity at the point of first contact.
- Constraints: terminology only; the caption's defined equivalence stays; no numbers change.
- Resolves: R5-47 (D-m5); narrows A-m6/C-m8's residual confusion surface.

### V03 — "Not recoverable" → "not recovered" in the Fig-4 caption
- Target: `main.tex` L378-379.
- Anchor: `; the remaining $79.1\%$ is not recoverable`
- Current: `...; the remaining $79.1\%$ is not recoverable from the situation description---which is what makes an online reading necessary rather than redundant.`
- Edit: `...; the remaining $79.1\%$ is not recovered by the situation description---which is what makes an online reading necessary rather than redundant.`
- Evidence: R² definition and both n printed in the same caption (1−SSE/SST, 486,660 / 461,937; RQ021-KC-C2 out-of-fold R² 0.209). "Recovered by" states the fitted-model fact; "recoverable" asserted irreducibility beyond the frozen basis.
- Constraints: mandatory-accuracy precision repair; the necessity argument survives verbatim; no numbers change.
- Resolves: R5-43 (C-m10).

### V04 — Scope "no engineering failures" to the replay (register phrase kept)
- Target: `main.tex` L414.
- Anchor: `returns a verdict on 14{,}099 ($20.8\%$), with no engineering failures --- no solver or pipeline failure on any`
- Edit: `returns a verdict on 14{,}099 ($20.8\%$), with no engineering failures in this replay --- no solver or pipeline failure on any candidate moment --- ...` (insert `in this replay` after `failures`; rest unchanged).
- Evidence: RQ017-KC-C1 REQUIRES the zero-engineering-failure statement with denominator 67,861 ("不得省略「工程失败 0」") — the term stays; B-m9's point (timing/sensing/integration failures untested) is honoured by the replay scoping, which matches the printed Methods scope ("end-to-end real-time operation ... not evaluated").
- Constraints: register-mandated fact retained verbatim in substance; two-word scoping insertion; no numbers change.
- Resolves: R5-38 (B-m9).

### V05 — Replace "ill-posed" with the exact claim
- Target: `main.tex` L295-296.
- Anchor: `state; a scalar sociality score is ill-posed. We report this as a state-dependent regularity of the`
- Current: `Whatever the human reference is, it must therefore be conditioned on the interaction state; a scalar sociality score is ill-posed. We report this as ...`
- Edit: `Whatever the human reference is, it must therefore be conditioned on the interaction state; a single scalar sociality score cannot serve as that reference. We report this as ...`
- Evidence: the operative argument is already printed one sentence earlier ("their direction reverses with risk, which no scalar score can represent" — RQ004 sign-gated deltas +0.058/+0.001/−0.034, all verified); "ill-posed" is manuscript-authored (absent from `RQ004 decision.md`) and mathematically over-broad (A-m5 is right that a scalar can have a defined use; it just cannot be the state-conditioned reference).
- Constraints: precision swap, zero information loss, keeps the full force of the reference-construction argument; no numbers change; press-conference-compatible (the claim becomes harder to attack).
- Resolves: R5-28 (A-m5).

## Completeness (frozen evidence surfaced)

### V06 — Name the quantile model and print the situation-vector feature contract (Methods 4.3)
- Target 1: `main.tex` L642.
  - Anchor: `online-computable risk proxies, and support/readability state), estimated by conditional quantile regression`
  - Edit: `...estimated by conditional quantile regression (histogram-based gradient-boosted quantile models, one per quantile level, with frozen hyperparameters: learning rate $0.06$, $72$ boosting iterations, maximum $31$ leaf nodes, minimum $80$ samples per leaf, L2 regularisation $0.01$, $255$ bins, no early stopping)`
- Target 2: `main.tex` L657-660 (immediately BEFORE the exclusion sentence `Excluded from the reference are observed post-encroachment time, ...`).
  - Edit: insert: `Concretely, $z_t$ comprises four categorical descriptors (path-geometry category, path relation, turn-pair label and priority role) and $22$ numeric channels: elapsed interaction time and history length; ego and counterpart velocity components and headings at the anchor; relative position components, distance, velocity components, speed, closing rate and heading difference at the anchor; short-window means and a dispersion of relative distance, relative speed and closing rate; and two online risk proxies --- the closing-rate time-to-collision at the anchor and an anticipated post-encroachment proxy computed only from information available at time $t$.`
- Evidence: `reports/studies/RQ021_contemporaneous_envelope/RQ021_1_contemporaneous_envelope_20260805T160425Z_43b4bff/key_numbers.json` → `selected_hgb_params` (learning_rate 0.06, max_iter 72, max_leaf_nodes 31, min_samples_leaf 80, l2_regularization 0.01, max_bins 255, early_stopping false) and `feature_contract` (categorical_context: geometry_path_category, geometry_path_relation, turn_pair_label, priority_role; numeric_context: 22 named channels incl. `closing_ttc_anchor`, `apet_online_proxy`, `_wx` window statistics).
- Constraints: pure surfacing of frozen manifest facts into Methods (charter-CLARIFY: "adding existing frozen numbers to the text, with source file named"); no codename vocabulary (HGB spelled out as its standard description); two-layer intact (Methods only); no `\targetnum` nearby; resolves the "operative model deferred to code" objection in its fourth round.
- Resolves: R5-16 (C-m6, B-q7, A-q9, B-M10 part); materially blunts C-M5/B-q7's "unnamed risk proxies" and feeds B28's audit-table half (anchor vs window provenance is now visible in print).

### V10 — Surface the frozen support-limit diagnosis (Methods 4.5 boundary)
- Target: `main.tex` L758, after `...we report this as an observed difference without attributing a cause, and the readability gate treats both populations identically.`
- Edit: append: `The binding constraint on benchmark coverage is situational rather than volumetric: in the 36 always-silent runs the readability pass rate ($53.78\%$) is close to the benchmark average ($55.30\%$) while the human-support pass rate is $0.07\%$ (against $32.32\%$ overall), and support does not follow sample volume --- one high-volume situation cell with $1{,}148{,}133$ human anchor rows passes support on $14.58\%$ of its benchmark moments while a $45{,}283$-row cell passes on $47.03\%$. The support gate compares kinematic neighbourhoods, not IPV values.`
- Evidence: `reports/knowledge/RQ017_onsite_mechanism_one/decision.md` RQ017-KC-C3 (ACCEPTED; designated section "Discussion（方法边界）"; required qualification "support gate compares kinematic neighbourhoods (12 distance features), not IPV values" — carried verbatim in the closing sentence). Internal cell codes (MP|yield, F|priority) are NOT printed — reader-facing phrasing keeps codebook tokens out.
- Constraints: knowledge-layer frozen claim, surfaced in its designated section; answers "what limits deployment coverage" without speculation; abstention stays a statement about the reference (consistent with §2.4's framing); no `\targetnum`.
- Resolves: R5-21 substantive half (D-M4's "discuss what coverage of conflict-rich regimes is needed"); strengthens the response to A-M9/C-M3 on operational utility.

## Consistency / response-letter CLARIFYs (no text change beyond the above)

### V07 — §2.3 abstention pointer — EXECUTED INSIDE V01 (same sentence). No separate edit.

### V08 — Absolute-width attack response (B-m8, D §6b) — response letter only
- Framing (charter-mandated, no retreat): (1) the 1.87-of-2.36 disclosure is the paper's own (added deliberately in round 4; Fig-4 caption + §2.3); (2) a wide human range is a property of human behavioural diversity under the situation description, not an instrument defect — the monitor's value claim is the relative sharpening (21%/20%/8%, frozen) plus calibration, and a wide reference makes a flag a conservative, clear-departure event (the sentence B quotes SAYS this); (3) the width distribution (5th–95th pct 1.35–2.28 rad) is printed; per-source width distributions are logged as backlog B09.
- Evidence: RQ021 key_numbers metrics (mean 1.8651; p05 1.3521; p95 2.2792); reductions 20.7/20.5/8.4%.
- Resolves: R5-22.

### V09 — E11 companion attacks, round-5 recurrence — response letter only
- Reuse `../round4/RESPONSE_NOTES_round4.md` ¶1–3 verbatim (roles of the two tests; magnitude rests on fixed-window intervals; two-layer placement) and add: (a) C-q10's "why is the timing placebo the confirmatory one" is answered IN PRINT — Methods 4.5 states "the exposure placebo above is the test specific to flag timing"; the case-label permutation destroys composition, not timing; (b) the exceedance count 4/201 and the case-label 30/201 are in the frozen `rq018_rerun/negative_controls.json` and can be quoted in the letter (NOT added to the PI-ratified block); (c) draw-count precision is logged as B27 with a pre-commitment note.
- Resolves: R5-06 (+R5-07 letter half).

---

## Not in this plan (by design)

- E2-gated wording (five sites + the new sixth swap-time site "the flag means the same thing"), all
  Fig-6/human-arm numeric material, and the abstract: ESCALATIONS (E2/E10/E16).
- Ethics/consent/safety facts (E5), declarations + TOPS disclosure (E8/E14), organiser facts
  (counterpart control, 65-run reasons, maturity classes — E6), release vehicle (E7).
- Per-source deployed-reference coverage (`insample_by_source`): frozen but not knowledge-layer
  registered → B31 (one-line PI freeze, then a one-sentence Methods 4.4 addendum becomes legal).
- All new-analysis demands: BACKLOG (B01–B29 standing; B30–B31 new).

## Post-execution verification battery

1. 14 `\targetnum{` occurrences byte-identical to 985a757; abstract diff empty.
2. E11 block: grep `0.1493`, `0.0199`, `-2.6100`, `+0.1372` each exactly once.
3. `estimab*` = 0; standalone `passive` = 0; no M0–M5/CQR/source-guard/claims-register tokens in main text.
4. NEW greps: `Where a reading is issued` = 0 (V01); `is not recoverable` = 0 (V03); `rad admissible span` = 0 in §2.3 body (V02 — the Fig-4 caption's defined "admissible range" phrasing stays); `no engineering failures in this replay` = 1 (V04); `ill-posed` = 0 (V05); `gradient-boosted` = 1 in Methods (V06); `1{,}148{,}133` = 1 (V10); `Section 2.4 and Methods` = 1 (V01/V07).
5. Compile exit 0; single References heading; ED order intact; ~36-page ballpark; figH SHA-256 unchanged (no figure touched).
