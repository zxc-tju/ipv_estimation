# Round-3 revision plan (S01–S16)

Base: `main.tex` @ a4f99f8 on `paper/human-arm-target`. Paper repo:
`/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/2_PaperWriting/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle/`
Research repo (evidence): `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/`

Global constraints screened on EVERY item: the 14 `\targetnum{...}` call sites stay byte-identical
(none of S01–S16 touches a line containing one, except S06 whose target sentence at L462 contains no
`\targetnum`); figH watermark and C7 discipline untouched; no `estimab*`, no "passive", no model
codenames or audit vocabulary enters the main text; no number changes except to match frozen files
(each cited below); abstention never rendered neutral; no self-weakening framing — every correction
below is an accuracy repair or a precision upgrade, not a concession. E2-gated sentences (abstract
"defining population", contribution 6, §2.5 title, "survives the move", "sits at the native level")
are NOT edited by this plan — they await the PI (ESCALATIONS.md).

Execution note: S01–S13 and S15–S16 are `main.tex` edits (captions included — captions live in
main.tex). S14 is the only figure-binary change and uses the existing restyle pipeline + acceptance
harness. Compile after each block; run the round-2 verification battery at the end (targetnum diff,
vocabulary greps, single References heading, compile exit 0).

---

## Correctness / register-compliance first

### S01 — Un-nest the engineering-failures parenthetical; link the two failure universes
- Target: `main.tex` L407-410 (§2.4 accounting paragraph).
- Anchor: `returns a verdict on 14{,}099 ($20.8\%$; no engineering failures (no solver or pipeline failure on any`
- Current: `...the monitor returns a verdict on 14{,}099 ($20.8\%$; no engineering failures (no solver or pipeline failure on any candidate moment)), and in 231 of the 267 scenario runs...`
- Edit: `...the monitor returns a verdict on 14{,}099 ($20.8\%$), with no engineering failures --- no solver or pipeline failure on any candidate moment --- and in 231 of the 267 scenario runs...`
  Additionally, in Methods 4.4, at the end of the natural-corpus accounting sentence (`...$17{,}416$ exact ties and $1{,}826$ solver failures.`), append: `The $1{,}826$ solver failures belong to this natural-corpus accounting; on the benchmark, no candidate moment failed the solver or pipeline (Section 2.4).`
- Evidence: RQ017-KC-C1 (`reports/knowledge/RQ017_onsite_mechanism_one/decision.md` L30: 67,861 / 55.2971% / 20.7763% / 工程失败 0); RQ021 key_numbers k2_ledger reason counts (1,826).
- Constraints: numbers unchanged; RQ017's required "engineering failures 0" phrasing kept.
- Resolves: R3-23 (D-m5, REGRESSION), R3-24 (A-m10, C-m10).

### S02 — Reposition the "comfortable" sentence onto the measured quantities
- Target: `main.tex` L474-476 (§2.4 close).
- Anchor: `the monitor registers is that comfortable interactions stop being comfortable, not that dangerous ones`
- Current: `What the monitor registers is that comfortable interactions stop being comfortable, not that dangerous ones become more common.`
- Edit: `What the monitor registers is that ordinarily comfortable margins tighten, not that dangerous events become more common.`
- Evidence: RQ018-KC-C1/C2 (`reports/knowledge/RQ018_abnormal_ipv_degradation/decision.md`): the compression sits in the median/upper quartile of the margin distribution; emergency rates do not rise. "Comfort" is not a frozen endpoint anywhere.
- Constraints: keeps the body-vs-tail contrast (press-conference: the rhetorical point survives); removes the unmeasured comfort construct (mandatory accuracy correction).
- Resolves: R3-20 (A-M7 language facet; C-M7 language part).

### S03 — Make the Fig-5 caption title associational (RQ018-B1 compliance)
- Target: `main.tex` L429 (Fig-5 caption opening).
- Anchor: `\caption{\textbf{Atypicality compresses ordinary interaction on both sides without adding emergencies.}`
- Current: `Atypicality compresses ordinary interaction on both sides without adding emergencies.`
- Edit: `\caption{\textbf{Flagged-assertive moments are followed by compressed ordinary interaction on both sides, with no rise in emergency rates at the supported thresholds.}`
- Evidence: RQ018-B1 (`decision.md` Boundaries: 描述性关联，不得使用因果表述 — descriptive association only, no causal formulation); the "no rise ... supported thresholds" phrasing mirrors the E3-decided abstract wording.
- Constraints: verdict-language discipline; matches the decided E3 formula; body text already uses "is followed by".
- Resolves: R3-09 caption facet (C-M7 "including titles and figure headings"; A-M3 language part).

### S04 — Define "supported" in the Fig-5 caption
- Target: `main.tex` Fig-5 caption, panel-b sentence (L441-443).
- Anchor: `All four thresholds were tested and all four are displayed with their intervals; the $<3$\,s interval admits no difference.`
- Edit: append one sentence: `A threshold is called supported when its per-endpoint interval excludes no difference; the panel titles use the word in exactly this sense.`
- Evidence: fig5_data.json `ego_danger_bootstrap` (three margin bins exclude zero; <3 s CI [−8.33, +3.15] crosses zero); RQ018-KC-C2 Required Qualification (only <1/<1.5/<2 citable as evidence).
- Constraints: the round-2 panel title stays (data-accurate and register-scoped); the caption's existing "Associations are descriptive" sentence already covers B-m7's other half.
- Resolves: R3-22 (B-m7, REGRESSION tag).

### S05 — Surface the marginal-coverage boundary in the main text
- Target: `main.tex` §2.3, L341-346.
- Anchor: `at coverage within 0.6 percentage points of nominal---and it stays informative at the`
- Edit: after the sentence ending `...can never flag a moment.` insert: `The coverage statement is empirical and marginal over accepted moments; it is not a per-situation or per-interaction guarantee (Methods 4.4).`
- Evidence: Methods 4.4 already states exactly this (frozen estimand, RQ021 key_numbers conformal_radii); this surfaces it beside the main-text claim.
- Constraints: two-layer discipline (no audit vocabulary; "guarantee" used only in negation, mirroring Methods); phrased as scope, not concession.
- Resolves: R3-06 (D-m12; C-M3 main-text facet).

## Consistency

### S06 — Gloss "judgeable" at first Results use
- Target: `main.tex` L462.
- Anchor: `counting both sides, $9.8\%$ of judgeable moments are flagged`
- Edit: `counting both sides, $9.8\%$ of judgeable moments (candidate moments that pass both gates; Methods 4.4) are flagged`
- Evidence: the glossary definition itself (Methods 4.4, L667-670).
- Constraints: no `\targetnum` on this line (verified); "anchor row" needs no Results gloss (first use is in Methods).
- Resolves: R3-42 (D-m1).

### S07 — Drop the unassessed "competent" qualifier in the Introduction question
- Target: `main.tex` L118-119.
- Anchor: `is this behaviour within\nthe range that competent humans exhibit?`
- Current: `...given the current interaction state, is this behaviour within the range that competent humans exhibit?`
- Edit: `...given the current interaction state, is this behaviour within the range that human drivers exhibit in the same situation?`
- Evidence: reference pool = recorded HV-HV drivers, no competence assessment exists (RQ021 key_numbers source_reference_pool). L166's "competent human driver" stays — it reports the cited external benchmarks' own construct.
- Constraints: strengthens precision; "in the same situation" reinforces the conditioning thesis (press-conference-positive).
- Resolves: R3-49 (A-m12).

### S08 — Algorithm 2: optional signal output + reason-code precedence
- Target: `main.tex` Algorithm 2 (L793-795) and Methods 4.4 reason-code sentence (L699-703).
- Anchor 1: `\RETURN (range, verdict, signal)`
- Edit 1: `\RETURN (range, verdict) \COMMENT{plus signal only where the optional persistence rule is instantiated}`
- Anchor 2: `\textsc{Low-Human-Support} or \textsc{Out-of-Distribution}; otherwise it emits`
- Edit 2: before `otherwise it emits`, insert: `reason codes are assigned by the first failing gate in the evaluation order of Algorithm 2, so exactly one code is returned;`
- Evidence: Algorithm 2's own sequential gate order (frozen construction); persistence layer marked unevaluated (Q07 wording retained verbatim).
- Constraints: no behavioural claim added; interface description only.
- Resolves: R3-37 (B-m2, B-m3).

### S09 — Grid-vs-domain reconciliation cross-reference in Methods 4.1
- Target: `main.tex` Methods 4.1, after the sentence ending `...seven candidate preferences $\theta_k\in\{-3,-2,-1,0,1,2,3\}\times\pi/8$;` (L611-612).
- Edit: insert: `the grid spans $[-3\pi/8,3\pi/8]$ inside the declared domain $[-\pi/2,\pi/2]$ of Eq.~1, and the admissible span quoted in Fig.~4 is the candidate span $3\pi/4$ (Fig.~4 caption).`
- Evidence: RQ021 key_numbers (admissible span 2.3562 = 3π/4); the Fig-4 caption already carries the same reconciliation (Q14).
- Constraints: no new numbers; pure cross-reference. Boundary-mass sensitivity remains B07.
- Resolves: R3-38 (B-m1, C-m5 reconciliation part).

## Completeness (frozen numbers surfaced)

### S10 — Define the ego post-verdict window
- Target: `main.tex` Methods 4.5, L719-721.
- Anchor: `The ego-side outcome is the minimum time-to-collision\nto the counterpart over the post-verdict window`
- Edit: `The ego-side outcome is the minimum time-to-collision to the counterpart over the post-verdict window, which runs from the verdict to the end of that run's evaluated window (the counterpart-side battery instead uses the fixed three-second window below); time-to-collision is computed per frame as distance over closing rate, and frames in which the pair is not closing do not enter the minimum`
- Evidence: `reports/studies/RQ021_contemporaneous_envelope/RQ021_1_.../rq018_rerun/RQ018_1_association.md` L11 (合同窗口 `[anchor_frame_index, target_window_end_frame_index]`; 3-s window by `time_s`; TTC = distance/closing rate; non-closing frames excluded; all-diverging windows missing).
- Constraints: consistent with the already-printed undefined-margin sentence (9.1% vs 8.2%); no number changes.
- Resolves: R3-25 (A-m3).

### S11 — Print the placebo construction (draws + statistic)
- Target: `main.tex` Methods 4.5, L726-728.
- Anchor: `distinguishes the real flag timing from permuted timings (empirical $p=0.0199$)`
- Edit: `distinguishes the real flag timing from permuted timings (200 whole-trajectory reassignment draws; the comparison statistic is the absolute case-clustered $t$; empirical $p=0.0199$)`
- Evidence: `rq018_rerun/negative_controls.json` (`placebo_draws: 200`, method string "Shuffle whole case exposure trajectories against recipient case outcomes...", `comparison_statistic: absolute case-clustered t statistic`, p = 4/201 = 0.01990).
- Constraints: does NOT pre-empt E11 — whether the companion qualification (case-label permutation p=0.1493; contract-window CI) must be printed is the PI's E11 decision; this item is compatible with either outcome.
- Resolves: R3-26 placebo half (A-m4; C-m4 placebo part). Bootstrap interval-type/seed documentation remains B06.

### S12 — Print the conformal radii
- Target: `main.tex` Methods 4.4, sentence ending `...over all accepted test-fold moments ($461{,}937$).` (L666-668).
- Edit: insert before `Vocabulary, used consistently:`: `The fitted radii are near zero ($c_\alpha = 1.4\times10^{-3}$, $1.2\times10^{-6}$ and $0.0$\,rad at the $80\%$, $90\%$ and $95\%$ levels), so the conformal step certifies, rather than repairs, the situational quantile model.`
- Evidence: RQ021 key_numbers `human_only_envelope.conformal_radii` (c_alpha 0.0014435 / 1.1921e-06 / 0.0; ranks 364,580 / 410,152 / 432,938 of n=455,723).
- Constraints: framing is press-conference-positive and factually exact; width distributions already in the Fig-4 caption.
- Resolves: R3-29 (B-m6 radii part).

### S13 — State threshold nesting and panel-d weighting in the Fig-5 caption
- Target: `main.tex` Fig-5 caption, panel-d/denominator block (L450-453).
- Anchor: `panel \textbf{d} aggregates\nthe acceleration records within those windows ($13{,}800$ vs $310{,}246$ records).`
- Edit: append after that sentence: `Margin and braking thresholds are nested (a moment below $1$\,s is also below the wider thresholds). Panel \textbf{d} is record-weighted; its uncertainty, like all others, is resampled over the $175$ scenario runs.`
- Evidence: fig5_data.json `ego_danger_shares` / `counterpart_braking_thresholds` (nested threshold construction; record denominators 13,800/310,246); RQ019-KC-C2 (record universe).
- Constraints: no re-weighting computed (run-weighted sensitivity stays B20).
- Resolves: R3-32 (B-m8 nesting part; C-m11 weighting part).

## Polish

### S14 — FIX-FIGURE: name the side in the Fig-1c panel annotation
- Target: restyle pipeline `T5_style_harmonisation/restyle_fig0_concept.py` → regenerate `figures/fig0_concept.pdf/.png`.
- Current in-panel text: `outside the human reference range`; caption says `above the $90\%$ range (the over-yielding side)`.
- Edit: panel-c annotation → `outside the human reference range (over-yielding side)`. Text-label change only; no data, marks or colours change; run the acceptance harness with the preserved Git baseline (P14/P15/Q24 protocol; A2 numeric-token multiset unchanged — no numeric token added); verify pixel diff confined to the label region; figH untouched (SHA check).
- Evidence: caption already states the side (main.tex L227-228); the displayed readings sit above the band by construction of the case.
- Constraints: FIX-FIGURE eligibility (label-only, data already in frozen files); charter watermark rules untouched.
- Resolves: R3-45 (D-m11).

### S15 — Early scope sentence: the IPV is one dimension
- Target: `main.tex` §2.1, after the sentence ending `...($\theta>0$ cooperative).` (L207-208).
- Edit: insert: `The IPV is one continuous, online-computable dimension of social interaction --- how strongly the agent weights the counterpart's cost against its own progress; signalling conventions, gap acceptance and multi-agent coordination are outside its scope (Discussion).`
- Evidence: Discussion already carries the scope boundary late (L575); this echoes it forward without changing any claim.
- Constraints: does not reopen E4 (title/abstract untouched); positively framed (names what the dimension IS first).
- Resolves: R3-02 facet (D-M7 "state early that IPV is one dimension").

### S16 — Planner-interface precision: safety subordination and the degraded mode
- Target: `main.tex` planner-interface paragraph, L797-800.
- Anchor: `a hesitation/efficiency warning, and a degraded unverified mode on abstention. This mapping is`
- Current: `...a fallback candidate under sustained competitive deviation at high physical risk, a hesitation/efficiency warning, and a degraded unverified mode on abstention. This mapping is an interface specification, not an external-validity result; a planner fallback is a control action and is never labelled \textsc{Within-Norm}.`
- Edit: `...a fallback candidate under sustained competitive deviation at high physical risk, a hesitation/efficiency warning, and --- on abstention --- a degraded mode in which the social verdict is simply unavailable. This mapping is an interface specification, not an external-validity result; the social layer is advisory and subordinate to collision-safety enforcement, no social verdict or abstention relaxes or delays any safety constraint, and a planner fallback is a control action never labelled \textsc{Within-Norm}.`
- Evidence: interface-specification statement (no empirical claim); consistent with §2.6's dissociation framing and the E9-decided scope note.
- Constraints: abstention stays non-neutral ("verdict unavailable", not "compliant"); no runtime-performance claim added (B13 untouched).
- Resolves: R3-16 facet (B-M8 arbitration/"degraded unverified mode"; B-q12).

---

## Not in this plan (by design)

- E2-gated wording (abstract "defining population", contribution 6 transfer sentence, §2.5 title,
  "survives the move", "sits at the native level"), all Fig-6/human-arm material (E10/C7), ethics
  content (E5), declarations (E8), benchmark facts (E6/E14), release vehicle (E7), event-level-null
  and placebo-companion disclosure (E11): see ESCALATIONS.md.
- All new-analysis demands: BACKLOG (B01–B26).

## Post-execution verification battery (same as round 2, plus)

1. 14 `\targetnum{` occurrences byte-identical to a4f99f8; abstract diff empty.
2. `estimab* = 0`; standalone `passive = 0`; no M0–M5/CQR/source-guard/claims-register tokens in main text (the two existing benign "claims register" comment/Methods-replacement checks unchanged from round 2).
3. Compile exit 0; single References heading; ED order intact; 35-page ballpark.
4. NEW: grep confirms `no engineering failures (no` gone (S01); `comfortable interactions stop` gone (S02); `Atypicality compresses` gone from the caption (S03); `(range, verdict, signal)` gone (S08).
5. S14 only: acceptance harness exit 0; pixel diff confined to declared label region; figH SHA-256 unchanged.
