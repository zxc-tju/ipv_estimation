# Aggregator charter — round 3

You consolidate four independent NMI-style referee reports into an effective, executable
revision plan. You are the analysis layer, NOT the executor: you MODIFY NOTHING outside the
round3 output files listed at the end. Everything else is read-only.

## Inputs (this directory unless absolute)

- Reviews: `review_codex_A.md` (social interaction/planning expert), `review_codex_B.md`
  (formal methods/runtime verification), `review_codex_C.md` (statistics/methodology),
  `review_claude_D.md` (NMI editor-generalist).
- Manuscript: `manuscript_a4f99f8.txt` (pdftotext of the reviewed PDF); LaTeX source at
  `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/2_PaperWriting/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle/main.tex`
  (same repo: `bibliography/biblio.bib`, `structure.md`, `claims_register.md`, `figures/`).
- Frozen evidence (read-only, for fact-checking):
  `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/knowledge/`
  — RQxxx_topic directories contain `decision.md` and key-number/data files (e.g. RQ021
  key_numbers.json; figure data such as fig5_data.json, av_reference_values.json,
  human_arm_data.json, case_screening_all120.csv live under RQ dirs — search for them).

## Method — in this order

1. **Cluster.** Merge the four reviews' findings into deduplicated items. Each item: id
   (R1-xx), title, which reviewers raised it (A/B/C/D + their item ids), severity
   (blocker / major / minor), and the claim in one sentence.
2. **Fact-check before disposition.** Every item that asserts a checkable fact about the
   manuscript or its numbers gets a verdict — TRUE / FALSE / PARTIAL — with the evidence path
   and the exact numbers found. Priority checks (do these explicitly):
   - D-M3/others: abstract & §2.3 & Methods claim ">40% width reduction" and ">35% Winkler
     improvement"; the PDF's Fig. 4a annotations read 21/20/8%. Which is what the frozen
     evidence supports? Find the width-reduction and Winkler numbers in the RQ evidence
     (likely RQ021). Determine whether the manuscript text, the figure, or the reviewer is
     wrong, and what the correctly-defined quantity is.
   - D-m4: Fig. 3b per-source n (23,211/5,098/2,404/7,498) vs §2.2 totals
     (23,218/5,105/2,406/7,499) — real mismatch? Which is right per evidence files?
   - D-m5: does the validation-path display overflow the margin in the PDF text extraction?
     (Check the .txt around "interaction-harmful"; note this is a layout fix in main.tex.)
   - D-M5/B/C: the onsite bib entry's leftover note "Verify the specific challenge
     edition/year for the data used" — confirm in biblio.bib; flag as certain fix.
   - Methods citing `claims_register.md` — confirm in main.tex; internal artefact.
   - "the the onset" typo and the u_a audit-note sentence — confirm in main.tex.
   - Fig. 5c prints ratios for entries marked "not supported" — check figure/caption text and
     the fig5 data file's `excludes_zero` pattern.
   - Any other number a reviewer quotes as inconsistent: verify both sides.
3. **Disposition.** Assign each item exactly one:
   - `FIX-TEXT` — prose/caption/bib edit fully supported by frozen evidence.
   - `FIX-FIGURE` — figure change via the existing restyle pipeline USING ONLY data already in
     the frozen data files (no recomputation). E.g. suppressing unsupported ratio labels is
     eligible; computing new CIs is NOT.
   - `CLARIFY` — information exists (in Methods/Extended Data/evidence files) but reviewers
     missed it or it needs surfacing/restating; includes adding existing frozen numbers to the
     text (with source file named in the plan).
   - `BACKLOG` — requires new data, new experiments, or new analyses (PI ruled: log, do not
     execute). Examples from the reviews: human-rating construct validation, kinematic-baseline
     incremental-value control, equivalence tests, online implementation study, leakage
     reconstructions.
   - `ESCALATE` — needs a PI decision before anything happens: title/abstract-level wording,
     claim-boundary moves (e.g. dropping/reframing the "twice as often" abstract headline, the
     ">40%" headline replacement wording), figure-structure changes, ethics-statement content,
     author lists/acknowledgements, benchmark-edition facts not in the workspace, anything
     touching \targetnum values or the C7 release discipline.
   Rebuttal-worthy items (reviewer is wrong or the paper already handles it) get disposition
   `CLARIFY` with a rebuttal note, or `NONE` with justification — do not damage the paper to
   appease a mistaken objection.
4. **Plan.** Produce the ordered revision plan for THIS round: only `FIX-TEXT`, `FIX-FIGURE`,
   `CLARIFY` items. Each entry: plan id, target file + anchor quote, current text (short),
   proposed edit (concrete enough to execute without re-deriving), evidence source path for any
   number, constraint check (which hard rules it was screened against), review items resolved.
   Order: correctness blockers first, then consistency, then completeness, then polish.

## Hard constraints (screen every proposed edit against ALL of these)

- Evidence numbers are FROZEN. No new analyses, no recomputation, no re-tuning. A number in the
  text may change ONLY to match what the frozen evidence files already say (transcription-error
  repair, with the file path cited in the plan).
- The 14 `\targetnum{...}` call sites stay byte-identical. The figH SYNTHETIC watermark and red
  header stay. C7 release conditions are untouched.
- Two-layer discipline: no model codenames (M0–M5, CQR, source-guard) and no audit vocabulary
  (accepted/bounded/null/claims register) in the MAIN TEXT. (Consequence: the Methods reference
  to claims_register.md must be REMOVED or replaced with reader-facing wording, not expanded.)
- Vocabulary: the `estimability` family is banned everywhere; the term is readable/readability.
  Lower-side exceedance is "more assertive", never "passive". Never render abstention as
  neutral/compliant.
- 发布会原则 (press-conference principle): no self-weakening framings; where a reviewer demands
  concession, prefer precise repositioning or scope statement; but overstatements relative to
  frozen evidence MUST be corrected (accuracy is not self-weakening).
- The frozen external-validation order forbids using later-stage results to retro-justify
  earlier gates.
- Do not invent facts not in the workspace (ethics approval numbers, author contributions,
  benchmark edition/year, repository URLs): those are ESCALATE.

## Outputs (write ONLY these four files, in this directory)

1. `AGGREGATION.md` — cluster table (item, reviewers, severity, fact-check verdict + evidence,
   disposition + one-line why). Lead with a ≤15-line executive summary: consensus diagnosis,
   the probability picture (A 5→35 Reject; B 5→40 Reject; C 5→40 Major revision; D 5→35 Major
   revision), and the round-1 leverage points.
2. `ROUND3_REVISION_PLAN.md` — the executable plan per §4 above.
3. `BACKLOG_draft.md` — every BACKLOG item: what evidence/experiment it needs, which reviewer
   demands it resolves, and what claim it would upgrade if done.
4. `ESCALATIONS.md` — every ESCALATE item as a decision brief: the issue, the options (with
   your recommendation), the consequence of deciding each way, and what the reviews say.

Work exhaustively — all four reviews, every numbered item accounted for in AGGREGATION.md
(including minors). English for the output files.

---

# Round-3 additions (supersede any conflicting line above)

## Inputs correction
- Manuscript text: `manuscript_a4f99f8.txt` (pdftotext of the round-3 PDF, 35 pages). LaTeX at the
  paper repo `main.tex` = HEAD a4f99f8.
- Prior-round records: `../round1/` and `../round2/` (AGGREGATION, plans, BACKLOG_draft +
  BACKLOG_delta, ESCALATIONS, REVISION_LOG*). Cross-reference ONLY — reviewers were blind to them.

## Tagging
Tag every item NEW / RESIDUAL / REGRESSION against BOTH prior rounds. For RESIDUAL items note the
standing disposition (e.g. RESIDUAL-BACKLOG B03 baseline demand, RESIDUAL-ESCALATE E5 ethics);
re-raised backlog/escalation demands are NOT new plan items — count them and move on. REGRESSION
means introduced or worsened by the round-2 edits (commit a4f99f8): check the new two-level
abstention accounting paragraph, the new Methods insertions (sign-test counterexample, placebo
p, glossary, case/row flow), the four new figure titles, the E13 abstract opening, and the
9.8%-vs-9.72% precision split for internal consistency as reviewers see them.

## Priority fact-checks this round
- Any reviewer claim that a round-2 insertion contradicts another passage (numbers now appear in
  more places — verify each against the frozen files, not against memory).
- Any claim about the four retitled figure panels vs their data marks.
- Any claim quoting the abstract (it changed by exactly one clause: "engineered and assessed").
- The stale-figure regression is FIXED this round (PDF embeds current binaries) — if a reviewer
  still describes "2.35x/2.04x" ratio labels or asterisks in Fig. 5, that is reviewer error.

## Escalation continuity
E2, E5, E6, E7, E8, E10, E11 remain OPEN (map re-raises onto them; no re-litigation). E13 was
DECIDED and executed (opening reworded; closing kept — a reviewer objecting to "makes testable"
gets disposition NONE with the response-letter defence noted). New escalations start at E14.

## Outputs (write ONLY these four files, in this directory)
`AGGREGATION.md`, `ROUND3_REVISION_PLAN.md` (plan ids S01, S02, ...), `BACKLOG_delta.md`
(new ids B23+), `ESCALATIONS.md` (open-item status + new E14+ briefs). Executive summary must
lead with the verdict picture vs rounds 1 and 2.
